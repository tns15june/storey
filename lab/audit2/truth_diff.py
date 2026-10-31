"""Do the three truth files agree on shared stacks? No API. Compares per-stair floor-presence sets, and box centres after the presence floor order is checked."""
import json
A=json.load(open('lab/audit/truth.json'))
TS={k:v for k,v in json.load(open('data/lab/tightseed/truth.json')).items() if not k.startswith('_')}
P=json.load(open('lab/presence/truth.json'))['buildings']
print('audit keys', list(A)[:8])
for b in ('bme_k','habs_va_bldg1','wfu_huffman'):
    print('\n==',b)
    a=A.get(b)
    if a is None:
        print('no audit entry'); 
    else:
        print('audit structure', type(a), (list(a)[:6] if isinstance(a,dict) else a[:1]))
    print('tightseed:', {k:(sorted(int(f) for f in s['boxes']), s['cls']) for k,s in TS[b]['stairs'].items()})
    print('presence :', {k:(sorted(int(f) for f in s['boxes']), s['cls'], s['kind'], s.get('unsure')) for k,s in P[b]['objects'].items() if s['kind'] in('stair','stairs')})
