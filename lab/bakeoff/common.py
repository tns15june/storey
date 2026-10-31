"""Call runner with a result cache, so a rerun of any step spends nothing for work already done."""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '/Users/tsharma/Github/Meta/lab')
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/bakeoff')
import metalab as m  # noqa: E402
import sheets as s  # noqa: E402

RUNS = s.OUT / 'runs.jsonl'
_lock = threading.Lock()
_done = {}
if RUNS.exists():
    for line in RUNS.read_text().splitlines():
        r = json.loads(line)
        _done[(r['exp'], r['key'], r['cfg'], r['run'])] = r


def cached_tokens(info, model):
    pi, pc, po = m.PRICES[model]
    return round((info['input'] * pi + info['output'] * po - info['usd'] * 1e6) / (pi - pc))


def save(row):
    with _lock:
        _done[(row['exp'], row['key'], row['cfg'], row['run'])] = row
        with open(RUNS, 'a') as f:
            f.write(json.dumps(row) + '\n')


def call(exp, key, cfg, run, prompt, images, schema, model='muse-spark-1.3', effort=None):
    """images is a function, so nothing renders when the result is already cached."""
    k = (exp, key, cfg, run)
    if k in _done:
        return _done[k]
    try:
        out, info = m.spark(prompt, images=images(), model=model, schema=schema, effort=effort, max_tokens=24000)
    except m.BudgetExceeded:
        raise
    except Exception as e:
        print('ERR', k, str(e)[:200], flush=True)
        return None
    row = dict(exp=exp, key=key, cfg=cfg, run=run, model=model, effort=effort, out=out, usd=info['usd'], seconds=info['seconds'],
               input=info['input'], output=info['output'], cached=cached_tokens(info, model))
    save(row)
    return row


def pool(jobs, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        res = [f.result() for f in [ex.submit(j) for j in jobs]]
    print(f"{len(jobs)} jobs done, bakeoff spend {m.spent('bakeoff'):.3f}", flush=True)
    return res


def rows(exp):
    return [r for k, r in _done.items() if k[0] == exp]


def obj(**props):
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


STR, NUM = {"type": "string"}, {"type": "number"}
INT = {"type": "integer"}


def arr(item):
    return {"type": "array", "items": item}
