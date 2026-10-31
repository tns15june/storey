"""Input tokens versus sheet resolution, from the input_tokens endpoint (no generation, no charge documented)."""
import json, sys
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab'); sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/bakeoff')
import metalab as m, sheets as s
rows = []
for key in ('n161', 'u1'):
    for px in (1000, 1500, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 7000, 8000):
        im = s.full(key, px)
        body = {"model": "muse-spark-1.3", "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}, {"type": "input_image", "image_url": m.data_url(im)}]}]}
        try:
            r = m._retry(lambda: m.client.post("/responses/input_tokens", body=body, cast_to=object))
        except Exception as e:  # the decode limit is a finding, so record it
            r = str(e)[:160]
        rows.append({"key": key, "px": px, "shape": im.shape[:2], "bytes_b64k": len(body["input"][0]["content"][1]["image_url"]) // 1000, "r": r})
        print(rows[-1], flush=True)
json.dump(rows, open(s.OUT / 'tokens.json', 'w'), indent=1, default=str)
