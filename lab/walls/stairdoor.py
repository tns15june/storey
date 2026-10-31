import sys; sys.path.insert(0,'/Users/tsharma/Github/Meta/lab/walls')
sys.argv=['x','0']
import collections, numpy as np, ifc
m=ifc.Model('/Users/tsharma/Github/Meta/data/plans/nibs_clinic/Clinic_A_20110906_optimized.ifc')
sp={s:(m.a(s)[2][2:],m.a(s)[7][2:] if isinstance(m.a(s)[7],str) else '') for s in m.of('IFCSPACE')}
for b in m.of('IFCRELSPACEBOUNDARY'):
    a=m.a(b); e=ifc.ref(a[5]); s=ifc.ref(a[4])
    if e and m.r[e][0]=='IFCDOOR' and 'STAIR' in sp[s][1].upper():
        print(sp[s], m.a(e)[2][2:40], a[7] , a[8])
