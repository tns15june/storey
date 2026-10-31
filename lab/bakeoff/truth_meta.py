"""Title-block truth, set by reading each sheet. Lists are the accepted answers. '' means nothing is printed."""
import re

G = ['g']
# key: (classes, sheet numbers, levels, scale denominators; 0 means no single scale is printed)
T = {
 'n161': (['floor_plan'], ['A18-1'], ['1'], [150]),
 'n79': (['schedule'], ['A5-1'], ['none'], [0]),
 'u1': (['code_summary'], ['G002'], ['none', '3'], [0]),
 'c3': (['floor_plan'], ['A2.2'], ['main'], [48]),
 'hc1': (['floor_plan'], ['2', 'DC-349-W'], ['1'], [96]),
 'bf0': (['floor_plan'], ['', '55'], G, [0]),
 'n1': (['cover'], ['T1-1'], ['none'], [0]),
 'n3': (['cover', 'other'], ['T1-3'], ['none'], [0]),
 'n7': (['site_plan'], ['C-4'], ['none'], [300]),
 'n52': (['floor_plan'], ['A2-1'], ['1'], [150]),
 'n53': (['floor_plan'], ['A2-2'], ['1'], [50]),
 'n59': (['floor_plan'], ['A2-8'], ['2'], [150]),
 'n76': (['schedule'], ['A4-2'], ['none'], [0]),
 'n91': (['elevation'], ['A7-1'], ['none'], [100]),
 'n93': (['section'], ['A7-3'], ['none'], [100]),
 'n122': (['detail', 'floor_plan', 'section'], ['A11-1'], ['none', '1', '2'], [40, 0]),
 'n162': (['floor_plan'], ['A18-2'], ['2'], [150]),
 'n166': (['floor_plan', 'other'], ['S2-1'], ['1', 'none'], [100]),
 'n171': (['schedule'], ['S3-2'], ['none'], [0]),
 'u0': (['cover'], ['G001'], ['none', '3'], [0]),
 'u2': (['floor_plan'], ['G003'], ['3'], [48]),
 'u3': (['floor_plan'], ['AD101'], ['3'], [48]),
 'u12': (['floor_plan'], ['MD100'], ['3'], [48, 0]),
 'u15': (['schedule', 'other'], ['E001'], ['none', '3'], [0]),
 'c0': (['cover'], ['A1.0'], ['none'], [0]),
 'c2': (['floor_plan'], ['A2.1'], G, [48]),   # title block says MAIN, the drawing and schedules say GROUND
 'c4': (['floor_plan'], ['A2.3'], ['upper'], [48]),
 'c5': (['floor_plan', 'other'], ['A2.4'], ['attic', 'roof', 'none'], [48]),
 'c7': (['elevation'], ['A4.0'], ['none'], [48]),
 'a0': (['cover', 'code_summary'], ['G-001'], ['none'], [0]),
 'a5': (['site_plan'], ['AS-101'], ['none'], [192]),
 'a7': (['floor_plan'], ['A1-101'], ['none', '1', 'g'], [48]),
 'a11': (['elevation', 'section'], ['A1-201'], ['none'], [48]),
 'a20': (['floor_plan', 'other'], ['S1-201'], ['none', 'foundation'], [48]),
 'ab5': (['detail', 'schedule'], ['A6.0'], ['none'], [0, 2, 12, 16, 24]),
 'bx4': (['floor_plan', 'other'], ['A2.3'], ['attic', 'roof', 'none'], [48]),
 'af0': (['cover'], ['C-1'], ['none'], [0]),
 'af1': (['floor_plan'], ['A-1'], ['none', '1', 'g'], [256]),
 'af2': (['floor_plan'], ['A-2'], ['none', '1', 'g'], [192]),
 'br0': (['floor_plan'], ['', '1'], ['1'], [96]),
 'hc0': (['floor_plan'], ['1', 'DC-349-W'], ['basement'], [96]),
 'ha1': (['floor_plan'], ['1', 'DC-349-T'], ['basement'], [48]),
 'hv1': (['floor_plan'], ['1-2'], ['1'], [96]),
 'bk1': (['floor_plan'], [''], ['lg'], [0]),
 'sc1': (['floor_plan'], [''], ['1'], [0]),
}
LEVEL_WORDS = [('lg', r'alsó|lower ground|souterrain'), ('g', r'ground|földszint|foldszint|^0$|level 0'), ('basement', r'basement|pince|cellar|^-1$'),
               ('1', r'first|1st|^1$|level 1|^l1$'), ('2', r'second|2nd|^2$|level 2'), ('3', r'third|3rd|^3$|level 3'),
               ('main', r'main'), ('upper', r'upper'), ('attic', r'attic'), ('roof', r'roof'), ('foundation', r'foundation')]


def norm_level(t):
    t = (t or '').strip().lower()
    if t in ('', 'none', 'n/a', 'na', 'null', 'not applicable'):
        return 'none'
    for k, rx in LEVEL_WORDS:
        if re.search(rx, t):
            return k
    return t


def norm_sheet(t):
    t = re.sub(r'\s+', '', (t or '').upper())
    t = re.sub(r'^SHEET(NO\.?|NUMBER)?:?', '', t)
    m = re.match(r'^(\d+)OF\d+(SHEETS)?$', t)
    return m.group(1) if m else ('' if t in ('NONE', 'N/A', 'NA') else t)


def scale_denom(t):
    """Scale text to a ratio denominator: 1/4" = 1'-0" is 48, 1:150 is 150, 0 when none or AS SHOWN."""
    t = (t or '').upper().replace('’', "'").replace('”', '"').replace('″', '"').replace('′', "'")
    m = re.search(r'1\s*:\s*(\d+)', t)
    if m:
        return int(m.group(1))
    m = re.search(r'(?:(\d+)[- ])?(\d+)\s*/\s*(\d+)\s*"?\s*[=-]\s*1\'', t) or None
    if m:
        w = int(m.group(1) or 0)
        return round(12 / (w + int(m.group(2)) / int(m.group(3))))
    m = re.search(r'(\d+)\s*"\s*=\s*1\'', t)
    if m:
        return round(12 / int(m.group(1)))
    return 0


def score(key, out):
    c, n, l, d = T[key]
    return dict(cls=out['sheet_class'] in c, num=norm_sheet(out['sheet_number']) in [norm_sheet(x) for x in n],
                level=norm_level(out['level']) in l, scale=scale_denom(out['scale']) in d)


if __name__ == '__main__':
    assert scale_denom('1/4" = 1\'-0"') == 48 and scale_denom('SCALE: 1:150') == 150 and scale_denom('AS SHOWN') == 0
    assert scale_denom('1/8"=1\'') == 96 and scale_denom('3/64"=1\'') == 256 and scale_denom('1-1/2" = 1\'-0"') == 8
    assert scale_denom('1/8" - 1\'-0"') == 96
    assert norm_level('FIRST FLOOR') == '1' and norm_level('Földszint') == 'g' and norm_level('Alsó földszint') == 'lg'
    assert norm_sheet('Sheet 2 of 10') == '2' and norm_sheet(' a2.1 ') == 'A2.1' and norm_sheet('N/A') == ''
    print('ok', len(T))
