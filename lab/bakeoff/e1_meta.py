"""E1: title-block reading sweep. Usage: e1_meta.py sweep | wide | repeat"""
import sys
sys.path.insert(0, 'lab/bakeoff')
from common import *  # noqa
import truth_meta as tm

SCHEMA = obj(sheet_class={"type": "string", "enum": ["floor_plan", "site_plan", "elevation", "section", "schedule", "code_summary", "cover", "detail", "other"]},
             sheet_number=STR, sheet_title=STR, level=STR, scale=STR, confidence=INT)
PROMPT = """You are reading one sheet of a building drawing set. Fill every field.
sheet_class: the main content of the sheet. Use code_summary for a code analysis or code data sheet. Use schedule for a sheet that is mainly tables. A plan of one floor is floor_plan, even when it also carries small schedules or is a life safety, demolition or framing plan.
sheet_number: the drawing number printed in the title block, such as A2.1, G002 or A18-1. Prefer it over a count like SHEET 161 OF 316. If the only identifier is a count, give the first number of the count. If none is printed, give an empty string.
sheet_title: the title as printed.
level: the floor this sheet shows, in the words printed in the main drawing title, such as FIRST FLOOR. Trust the main drawing title over the title block if they disagree. Empty string if the sheet does not show one floor.
scale: the scale printed next to the main drawing, copied exactly, such as 1/4" = 1'-0" or 1:150. Empty string if there is only AS SHOWN, NTS or no scale.
confidence: 0 to 100, how sure you are that sheet_number, level and scale are all correct.
"""
PROMPT_TB = PROMPT + "The first image is the whole sheet. The next images are zoomed crops of its right edge and bottom edge, where title blocks usually are. Text there may be rotated.\n"


def whole(key, px):
    return lambda: [s.full(key, px)]


def tb(key):
    return lambda: [s.full(key, 2000)] + s.title_crops(key)


def run_cfg(keys, cfg, runs=(0,)):
    model, effort, mode = CFG[cfg]
    jobs = []
    for k in keys:
        for r in runs:
            imgs = tb(k) if mode == 'tb' else whole(k, mode)
            p = PROMPT_TB if mode == 'tb' else PROMPT
            jobs.append(lambda k=k, r=r, imgs=imgs, p=p: call('meta', k, cfg, r, p, imgs, SCHEMA, model, effort))
    return jobs


S13, S12, W = 'muse-spark-1.3', 'muse-spark-1.2', 'muse-wonder-1.0'
CFG = {'s13-min-2k': (S13, 'minimal', 2000), 's13-low-2k': (S13, 'low', 2000), 's13-med-2k': (S13, 'medium', 2000), 's13-high-2k': (S13, 'high', 2000),
       's13-low-3k': (S13, 'low', 3000), 's13-low-4k': (S13, 'low', 4000), 's13-low-tb': (S13, 'low', 'tb'), 's13-med-tb': (S13, 'medium', 'tb'),
       's12-low-2k': (S12, 'low', 2000), 's12-low-tb': (S12, 'low', 'tb'), 'w-low-2k': (W, 'low', 2000), 'w-low-tb': (W, 'low', 'tb')}
DEV = list(s.CORE) + ['n7', 'n93', 'a7', 'c2', 'hv1', 'u2']


def table():
    rs = rows('meta')
    print(f"{'cfg':14}{'n':>4}{'class':>7}{'num':>6}{'level':>7}{'scale':>7}{'all3':>6}{'usd/sh':>8}{'s/sh':>7}{'in':>7}{'out':>7}")
    for cfg in CFG:
        for scope, keys in (('dev', DEV), ('all', list(tm.T))):
            r = [x for x in rs if x['cfg'] == cfg and x['key'] in keys]
            if not r or (scope == 'all' and len({x['key'] for x in r}) <= len(DEV)):
                continue
            sc = [tm.score(x['key'], x['out']) for x in r]
            n = len(r)
            a = lambda f: sum(x[f] for x in sc) / n
            all3 = sum(x['num'] and x['level'] and x['scale'] for x in sc) / n
            print(f"{cfg[:14]:14}{n:4}{a('cls'):7.2f}{a('num'):6.2f}{a('level'):7.2f}{a('scale'):7.2f}{all3:6.2f}{sum(x['usd'] for x in r)/n:8.4f}{sum(x['seconds'] for x in r)/n:7.1f}{sum(x['input'] for x in r)/n:7.0f}{sum(x['output'] for x in r)/n:7.0f}  {scope}")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'table'
    if what == 'sweep':
        pool([j for c in CFG for j in run_cfg(DEV, c)])
    if what == 'wide':
        pool([j for c in sys.argv[2:] for j in run_cfg(list(tm.T), c)])
    if what == 'repeat':
        pool([j for c in sys.argv[2:] for j in run_cfg(DEV, c, (1, 2))])
    table()
