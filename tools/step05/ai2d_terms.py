"""Collect the evidence on AI2D's two published terms: the licence bundled in AI2's archive and the
CC BY-SA label on AI2's dataset page and the AWS Open Data listing.

Fetches (no credentials needed; GH_TOKEN is used if set to avoid GitHub rate limits):
- the AWS registry listing at HEAD, its commit history and the PR that added it;
- the licence field of every listing added in that PR, with the same files at the merge commit;
- the S3 object metadata and version list of ai2d-all.zip, its central directory and its text members;
- Wayback captures of allenai.org/data/diagrams, prior.allenai.org/projects/diagram-understanding and,
  up to 2021-06, allenai.org/data/tqa (licence field and licence mentions per capture);
- the TQA archive README, for comparison with the label the same batch gave TQA;
- the AI2D paper (arXiv 1603.07396v1), for the image collection method.

Raw files go to OUT_DIR (machine-local); the summary goes to OUT_JSON.

usage: python tools/step05/ai2d_terms.py OUT_DIR OUT_JSON
"""

import base64
import gzip
import hashlib
import json
import os
import re
import struct
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import requests

REGISTRY = "awslabs/open-data-registry"
LISTING = "datasets/allenai-diagrams.yaml"
ZIP = "https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip"
TQA_ZIP = "https://ai2-public-datasets.s3.amazonaws.com/tqa/tqa_train_val_test.zip"
PAPER = "https://arxiv.org/pdf/1603.07396v1"
# (url, last capture month, marker of a real page). TQA is only needed around the date the labels appeared.
PAGES = {"diagrams": ("allenai.org/data/diagrams", None, b"__NEXT_DATA__"), "tqa": ("allenai.org/data/tqa", "202106", b"__NEXT_DATA__"),
         "prior": ("prior.allenai.org/projects/diagram-understanding", None, b"AI2D Dataset")}

out_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
http = requests.Session()
gh_headers = {"Accept": "application/vnd.github+json"}
if os.environ.get("GH_TOKEN"):
    gh_headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"


def save(name, body):
    (out_dir / name).write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def gh(path):
    r = http.get(f"https://api.github.com/{path}", headers=gh_headers, timeout=60)
    r.raise_for_status()
    return r.json()


def gh_file(path, ref):
    return base64.b64decode(gh(f"repos/{REGISTRY}/contents/{path}?ref={ref}")["content"])


def field(yaml_bytes, key):
    m = re.search(rf"^{key}:\s*(.+)$", yaml_bytes.decode(), re.M)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------- AWS listing
head = gh(f"repos/{REGISTRY}/commits/HEAD")["sha"]
listing = gh_file(LISTING, head)
commits = gh(f"repos/{REGISTRY}/commits?path={LISTING}&per_page=100")
pr = gh(f"repos/{REGISTRY}/commits/{commits[-1]['sha']}/pulls")[0]
pr_files = [f["filename"] for f in gh(f"repos/{REGISTRY}/pulls/{pr['number']}/files?per_page=100")]
pr_commits = [c["sha"] for c in gh(f"repos/{REGISTRY}/pulls/{pr['number']}/commits?per_page=100")]
comments = gh(f"repos/{REGISTRY}/issues/{pr['number']}/comments?per_page=100")
aws = {
    "registry_head": head,
    "listing": {"path": LISTING, "sha256": save("aws-listing.yaml", listing), "License": field(listing, "License"),
                "ManagedBy": field(listing, "ManagedBy"), "Documentation": field(listing, "Documentation")},
    "listing_commits": [{"sha": c["sha"], "date": c["commit"]["author"]["date"], "author": c["commit"]["author"]["name"],
                         "message": c["commit"]["message"].splitlines()[0]} for c in commits],
    "pull_request": {"number": pr["number"], "url": pr["html_url"], "user": pr["user"]["login"], "author_association": pr["author_association"],
                     "created_at": pr["created_at"], "merged_at": pr["merged_at"], "body": pr["body"],
                     "author_comments": [c["body"] for c in comments if c["user"]["login"] == pr["user"]["login"]]},
    "licence_of_ai2d_listing_in_each_pr_commit": sorted({field(gh_file(LISTING, s), "License") for s in pr_commits}),
    "licences_in_pr": {f.split("/")[-1]: field(gh_file(f, pr["merge_commit_sha"]), "License") for f in pr_files},
}


# --------------------------------------------------------------- S3 archive
def ranged(url, a, b):
    r = http.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
    r.raise_for_status()
    return r.content


def zip_members(url, prefix):
    size = int(http.head(url, timeout=60).headers["Content-Length"])
    tail = ranged(url, size - 65558, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    cd = ranged(url, cd_off, cd_off + cd_size - 1)
    members, p = [], 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, mtime, mdate, crc, csize, _, nlen, xlen, clen = struct.unpack("<HHHIIIHHH", cd[p + 10:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode()
        date = f"{1980 + (mdate >> 9):04d}-{(mdate >> 5) & 15:02d}-{mdate & 31:02d}T{mtime >> 11:02d}:{(mtime >> 5) & 63:02d}:{(mtime & 31) * 2:02d}"
        members.append({"name": name, "method": method, "crc": crc, "csize": csize, "offset": off, "dos_time": date})
        p += 46 + nlen + xlen + clen
    texts = {}
    for m in members:
        if not m["name"].startswith("__MACOSX") and re.search(r"(licen[cs]e|readme)[^/]*$", m["name"], re.I):
            local = ranged(url, m["offset"], m["offset"] + 29)
            start = m["offset"] + 30 + sum(struct.unpack("<HH", local[26:30]))
            body = ranged(url, start, start + m["csize"] - 1)
            body = zlib.decompress(body, -15) if m["method"] == 8 else body
            assert zlib.crc32(body) & 0xFFFFFFFF == m["crc"], m["name"]
            texts[m["name"]] = {"dos_time": m["dos_time"], "sha256": save(prefix + m["name"].replace("/", "__"), body), "text": body.decode("utf-8", "replace")}
    dates = sorted(m["dos_time"] for m in members)
    return {"url": url, "bytes": size, "members": len(members), "member_dos_times": [dates[0], dates[-1]], "texts": texts}


h = http.head(ZIP, timeout=60).headers
versions = http.get("https://ai2-public-datasets.s3.amazonaws.com/?versions&prefix=diagrams/", timeout=60).content
archive = {
    "head": {k: h.get(k) for k in ("Last-Modified", "ETag", "Content-Length", "x-amz-version-id")},
    "object_versions": re.findall(r"<VersionId>([^<]+)</VersionId><IsLatest>([^<]+)</IsLatest><LastModified>([^<]+)</LastModified>", versions.decode()),
    "documentation_url_now": None,
    **zip_members(ZIP, "ai2d-"),
}
r = http.get("https://allenai.org/data/diagrams", allow_redirects=False, timeout=60)
archive["documentation_url_now"] = {"status": r.status_code, "location": r.headers.get("location")}
tqa = zip_members(TQA_ZIP, "tqa-")
tqa_terms = [line.strip() for t in tqa["texts"].values() for line in t["text"].splitlines() if re.search(r"licen[cs]e", line, re.I)]


# --------------------------------------------------------------- AI2 dataset pages (Wayback)
def capture(url, ts, marker):
    # Wayback answers bursts with a short "request is being blocked" page; such a capture is
    # recorded as fetch_blocked, never as a page without a licence statement.
    path = out_dir / f"wayback-{url.split('/')[-1]}-{ts}.html"
    if path.exists() and marker in path.read_bytes():
        return path.read_bytes()
    for attempt in range(4):
        time.sleep(5 * 2 ** attempt)
        try:
            body = http.get(f"https://web.archive.org/web/{ts}id_/https://{url}", timeout=120).content
        except requests.RequestException:
            continue
        if body[:2] == b"\x1f\x8b":  # some captures are stored gzipped without a Content-Encoding header
            body = gzip.decompress(body)
        if marker in body:
            path.write_bytes(body)
            return body
    return None


def captures(url, until, marker):
    rows = http.get("https://web.archive.org/cdx/search/cdx", params={"url": url, "output": "json", "fl": "timestamp,statuscode", "collapse": "digest", **({"to": until} if until else {})}, timeout=120).json()[1:]
    result = []
    for ts, status in rows:
        rec = {"timestamp": ts, "status": status}
        if status == "200":
            body = capture(url, ts, marker)
            if body is None:
                rec["fetch_blocked"] = True
            else:
                rec["sha256"] = hashlib.sha256(body).hexdigest()
                rec["licenseName"] = sorted({x.decode() for x in re.findall(rb'"licenseName":"([^"]*)"', body)})
                rec["licenseUrl"] = sorted({x.decode() for x in re.findall(rb'"licenseU[Rr][Ll]":"([^"]*)"', body)})
                text = re.sub(rb"<script.*?</script>|<[^>]+>", b" ", body, flags=re.S)
                rec["licence_mentions"] = sorted({m.decode("utf-8", "replace") for m in re.findall(rb"[^.]{0,80}(?:licen[cs]e|creative commons|cc by)[^.]{0,80}", text, re.I)})
        result.append(rec)
    return result


pages = {k: captures(*v) for k, v in PAGES.items()}

paper = http.get(PAPER, timeout=120).content
summary = {
    "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "aws_listing": aws,
    "archive": archive,
    "allenai_pages": pages,
    "tqa_comparison": {"aws_and_page_label": aws["licences_in_pr"].get("allenai-tqa.yaml"), "archive": {k: v for k, v in tqa.items() if k != "texts"},
                       "archive_licence_lines": tqa_terms},
    "paper": {"url": PAPER, "sha256": save("ai2d-paper-1603.07396v1.pdf", paper),
              "collection_quote_page_9_checked_by_hand": "The images are collected by scraping Google Image Search with seed terms derived from the chapter titles in Grade 1 - 6 science textbooks. Each image is annotated using Amazon Mechanical Turk (AMT)."},
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({"listing": aws["listing"]["License"], "commits": len(aws["listing_commits"]), "zip_texts": list(archive["texts"]),
                  "page_labels": {k: sorted({(c["timestamp"][:6], tuple(c.get("licenseName", []))) for c in v if c["status"] == "200"}) for k, v in pages.items()}}, indent=1))
