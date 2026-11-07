"""Bounded PixMo-Count image integrity check against the card's own snippet.

The card (allenai/pixmo-count) says: image_bytes = requests.get(example["image_url"]).content;
sha256(image_bytes).hexdigest() == example["image_sha256"]. This fetches N evenly spaced train rows
exactly that way (default requests headers), sequentially with a delay, and records what came back.

usage: python tools/step05/pixmo_hash_check.py OUT_JSON [N]
"""

import hashlib
import io
import json
import sys
import time

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download
from PIL import Image

REPO, REV = "allenai/pixmo-count", "ebf51cf70e45d12374e64d475862e4f8d21d31d0"
out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
path = hf_hub_download(REPO, "data/train-00000-of-00001.parquet", revision=REV, repo_type="dataset")
t = pq.read_table(path, columns=["image_url", "image_sha256", "count", "label"]).to_pylist()
idx = [round(i * (len(t) - 1) / (n - 1)) for i in range(n)]
rows = []
for i in idx:
    r = t[i]
    rec = {"row": i, "url": r["image_url"], "expected": r["image_sha256"]}
    try:
        resp = requests.get(r["image_url"], timeout=30)
        body = resp.content
        rec.update(status=resp.status_code, final_url=resp.url, redirects=[h.status_code for h in resp.history], content_type=resp.headers.get("Content-Type"),
                   bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), last_modified=resp.headers.get("Last-Modified"), etag=resp.headers.get("ETag"))
        rec["match"] = rec["sha256"] == r["image_sha256"]
        if resp.status_code == 200:
            try:
                im = Image.open(io.BytesIO(body))
                rec.update(format=im.format, size=list(im.size), exif=bool(im.getexif()))
            except Exception as e:
                rec["decode_error"] = type(e).__name__
    except requests.RequestException as e:
        rec.update(status=None, error=type(e).__name__)
    rows.append(rec)
    time.sleep(1.0)
hosts = {}
for rec in rows:
    h = rec["url"].split("/")[2]
    s = hosts.setdefault(h, {"rows": 0, "status_200": 0, "match": 0})
    s["rows"] += 1
    s["status_200"] += rec.get("status") == 200
    s["match"] += bool(rec.get("match"))
summary = {"repo": REPO, "revision": REV, "train_rows": len(t), "sampled": len(rows), "method": "requests.get(url).content with default headers, as in the card",
           "status": {str(k): sum(1 for r in rows if r.get("status") == k) for k in sorted({r.get("status") for r in rows}, key=str)},
           "matches": sum(bool(r.get("match")) for r in rows), "redirected": sum(bool(r.get("redirects")) for r in rows), "hosts": hosts}
json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
