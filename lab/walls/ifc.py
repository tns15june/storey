"""Minimal STEP reader for the clinic IFC: doors, spaces, space boundaries, placements. Data only."""
import re
import numpy as np


def parse(path):
    recs = {}
    buf = ''
    for line in open(path, errors='ignore'):
        buf += line.strip()
        if not buf.endswith(';'):
            continue
        m = re.match(r"#(\d+)=\s*([A-Z0-9]+)\((.*)\);$", buf)
        buf = ''
        if m:
            recs[int(m[1])] = (m[2], m[3])
    return recs


def args(s):
    out, stack, cur, i = [], [], [], 0
    cur = out
    tok = ''
    instr = False
    while i < len(s):
        c = s[i]
        if instr:
            if c == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    tok += "'"; i += 1
                else:
                    instr = False
            else:
                tok += c
        elif c == "'":
            instr = True; tok = 'S:'
        elif c == '(':
            n = []; cur.append(n); stack.append(cur); cur = n
        elif c == ')' or c == ',':
            if tok:
                cur.append(tok); tok = ''
            if c == ')':
                cur = stack.pop()
        else:
            tok += c
        i += 1
    if tok:
        cur.append(tok)
    return out


def ref(x):
    return int(x[1:]) if isinstance(x, str) and x.startswith('#') else None


class Model:
    def __init__(self, path):
        self.r = parse(path)
        self.cache = {}

    def a(self, i):
        if i not in self.cache:
            self.cache[i] = args(self.r[i][1])
        return self.cache[i]

    def of(self, t):
        return [i for i, v in self.r.items() if v[0] == t]

    def point(self, i):
        return np.array([float(x) for x in self.a(i)[0]])

    def axis2(self, i):
        """4x4 of IFCAXIS2PLACEMENT3D or 2D."""
        t = self.r[i][0]
        a = self.a(i)
        p = self.point(ref(a[0]))
        p = np.r_[p, 0] if len(p) == 2 else p
        M = np.eye(4); M[:3, 3] = p
        if t == 'IFCAXIS2PLACEMENT3D':
            z = np.array([float(x) for x in self.a(ref(a[1]))[0]]) if ref(a[1]) else np.array([0, 0, 1.])
            x = np.array([float(v) for v in self.a(ref(a[2]))[0]]) if ref(a[2]) else np.array([1., 0, 0])
        else:
            z = np.array([0, 0, 1.])
            x = np.array([float(v) for v in self.a(ref(a[1]))[0]] + [0]) if len(a) > 1 and ref(a[1]) else np.array([1., 0, 0])
        z = z / np.linalg.norm(z); x = x - x.dot(z) * z; x = x / np.linalg.norm(x); y = np.cross(z, x)
        M[:3, 0], M[:3, 1], M[:3, 2] = x, y, z
        return M

    def place(self, i):
        """World matrix of an IFCLOCALPLACEMENT."""
        a = self.a(i)
        M = self.axis2(ref(a[1]))
        return self.place(ref(a[0])) @ M if ref(a[0]) else M

    def space_poly(self, sid):
        """Outline of a space in world XY metres, from its extruded profile."""
        a = self.a(sid)
        W = self.place(ref(a[5]))
        rep = ref(a[6])
        for sr in self.a(rep)[2]:
            for item in self.a(ref(sr))[3]:
                it = ref(item)
                while self.r[it][0] in ('IFCBOOLEANCLIPPINGRESULT', 'IFCBOOLEANRESULT'):
                    it = ref(self.a(it)[1])  # roof clips do not change the plan outline
                if self.r[it][0] != 'IFCEXTRUDEDAREASOLID':
                    continue
                ea = self.a(it)
                prof = ref(ea[0])
                M = W @ self.axis2(ref(ea[1]))
                pa = self.a(prof)
                if self.r[prof][0] == 'IFCRECTANGLEPROFILEDEF':
                    x, y = float(pa[3]) / 2, float(pa[4]) / 2
                    Q = self.axis2(ref(pa[2]))
                    h = np.array([[-x, -y, 0, 1], [x, -y, 0, 1], [x, y, 0, 1], [-x, y, 0, 1]])
                    return (M @ Q @ h.T).T[:, :2]
                if self.r[prof][0] in ('IFCARBITRARYCLOSEDPROFILEDEF', 'IFCARBITRARYPROFILEDEFWITHVOIDS'):
                    c = ref(pa[2])
                    if self.r[c][0] == 'IFCPOLYLINE':
                        P = np.array([self.point(ref(q))[:2] for q in self.a(c)[0]])
                    else:
                        P = []
                        for seg in self.a(c)[0]:
                            s = self.a(ref(seg))
                            cu = ref(s[2])
                            if self.r[cu][0] == 'IFCPOLYLINE':
                                P += [self.point(ref(q))[:2] for q in self.a(cu)[0]]
                            elif self.r[cu][0] == 'IFCTRIMMEDCURVE':
                                # shortcut: a curved edge is reduced to its two trim points, fine for room outlines
                                for t in (self.a(cu)[1], self.a(cu)[2]):
                                    q = [x for x in t if ref(x)]
                                    if q:
                                        P.append(self.point(ref(q[0]))[:2])
                        P = np.array(P)
                    h = np.c_[P, np.zeros(len(P)), np.ones(len(P))]
                    return (M @ h.T).T[:, :2]
        return None


if __name__ == '__main__':
    import collections
    m = Model('/Users/tsharma/Github/Meta/data/plans/nibs_clinic/Clinic_A_20110906_optimized.ifc')
    sp = m.of('IFCSPACE'); doors = m.of('IFCDOOR'); bd = m.of('IFCRELSPACEBOUNDARY')
    print(len(sp), len(doors), len(bd))
    print(m.a(sp[0])[:9])
    ok = 0; bad = collections.Counter()
    for s in sp:
        try:
            p = m.space_poly(s)
            ok += p is not None
            if p is None: bad['none'] += 1
        except Exception as e:
            bad[repr(e)[:50]] += 1
    print('polys', ok, bad.most_common(5))
    d2s = collections.defaultdict(set); kinds = collections.Counter()
    for b in bd:
        a = m.a(b)
        s, e = ref(a[4]), ref(a[5])
        t = m.r[e][0] if e else 'None'
        kinds[t] += 1
        if t == 'IFCDOOR':
            d2s[e].add(s)
    print(kinds.most_common(8), 'doors with boundaries', len(d2s), collections.Counter(len(v) for v in d2s.values()))
    print('door placement', m.place(ref(m.a(doors[0])[5]))[:3, 3])
    print(collections.Counter(m.a(s)[2][:3] if isinstance(m.a(s)[2], str) else '' for s in sp).most_common(6))
