import sys, tempfile
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/defects')
import suite
from suite import *
suite.RES = Path(tempfile.mkdtemp())
def boom(*a, **k): raise RuntimeError('no API here')
m.sam_video = boom
for i in ['erase_habs_va_bldg1_elev2_3', 'erase_habs_stelizabeths_center_xshaft_c_1', 'erase_schenley_sw_1', 'move_schenley_n_0', 'erase_habs_va_bldg1_west_2']:
    c = json.load(open(OUT / 'runs' / f'{i}.json'))['case']; c['grid'] = True
    suite.do(c)
    r = json.load(open(suite.RES / f'{i}.json'))
    fr, names = STACK[c['b']]
    print(i)
