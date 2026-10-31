"""Start Storey: python -m storey [--host 127.0.0.1] [--port 8000]"""
import argparse

import uvicorn

from .server import RUN_KEY, access_key

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()
if access_key() == RUN_KEY:
    print(f"Open http://127.0.0.1:{a.port}/#key={RUN_KEY}")
uvicorn.run("storey.server:app", host=a.host, port=a.port, log_level="warning")
