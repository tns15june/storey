import json, collections
r=json.load(open('/Users/tsharma/Github/Meta/data/lab/audit2/gate_jitter.json'))
print(len(r),'fits')
ok=[x for x in r if 'error' not in x]
print('errors',[x for x in r if 'error' in x][:3])
for lvl in (.02,.05):
    q=[x for x in ok if x['level']==lvl]
    print(lvl,'n',len(q),'passed',sum(x['passed'] for x in q),'failed',sum(not x['passed'] for x in q))
    print('  passed with dev>40', [(x['b'],x['f'],x['inliers'],x['dev_px']) for x in q if x['passed'] and (x['dev_px'] or 0)>40])
    print('  failed on floors frozen good', [(x['b'],x['f'],x['inliers'],x['frozen_inliers']) for x in q if not x['passed'] and x['frozen_aligned']])
print('min passing inliers', sorted((x['inliers'],x['b'],x['f'],x['dev_px']) for x in ok if x['passed'])[:6])
print('huffman f3', [(x['level'],x['inliers'],x['passed'],x['dev_px']) for x in ok if x['b']=='wfu_huffman' and x['f']=='3'])
