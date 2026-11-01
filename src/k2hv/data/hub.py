"""Pinned Hugging Face file listings, verified downloads, range reads and URL media fetches."""

import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from .runtime import file_sha256, temp_path


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(path)
    tmp.write_text(text)
    os.replace(tmp, path)


def listing(root, repo, revision):
    """File list with sizes and LFS sha256 at a pinned revision, cached under the data root."""
    path = Path(root) / "inventory" / "listings" / f"{repo.replace('/', '__')}@{revision}.json"
    if path.exists():
        return json.loads(path.read_text())
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"{repo} resolved to {info.sha}, expected {revision}")
    files = {f.rfilename: {"size": f.size, "sha256": f.lfs.sha256 if f.lfs else None} for f in info.siblings}
    data = {"repo": repo, "revision": revision, "files": files}
    atomic_write_text(path, json.dumps(data, indent=1, sort_keys=True))
    return data


def download(root, repo, revision, filename):
    """Download one file at a pinned revision and verify its LFS sha256 once."""
    meta = listing(root, repo, revision)["files"][filename]
    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(Path(root) / "hf")))
    marker = Path(root) / "inventory" / "verified" / f"{meta['sha256'] or hashlib.sha256(filename.encode()).hexdigest()}.json"
    if meta["sha256"] and not marker.exists():
        actual = file_sha256(path)
        if actual != meta["sha256"]:
            raise RuntimeError(f"{repo}@{revision}/{filename}: sha256 {actual} != {meta['sha256']}")
        atomic_write_text(marker, json.dumps({"repo": repo, "revision": revision, "file": filename, "sha256": actual, "size": path.stat().st_size}))
    return path


def remote(repo, revision, filename, block_size=8 << 20):
    """Range-read handle for pilots; whole-file hashes are not verified on this path."""
    return HfFileSystem().open(f"datasets/{repo}@{revision}/{filename}", "rb", block_size=block_size)


class UrlError(RuntimeError):
    pass


PERMANENT = ("url_http_400", "url_http_401", "url_http_403", "url_http_404", "url_http_410", "url_http_451", "url_hash_mismatch", "url_too_large")


def fetch_url(root, url, expected_sha256=None, timeout=20, max_bytes=50 << 20, attempts=4):
    """Fetch an image URL once into the data root and verify the supplied hash.

    Permanent failures are cached so reruns reproduce them; rate limits, server errors and
    timeouts are retried with backoff and not cached. Raises UrlError(reason).
    """
    import requests

    name = expected_sha256 or hashlib.sha256(url.encode()).hexdigest()
    cache = Path(root) / "media" / "url" / name[:2] / name
    failure = cache.with_suffix(".failed")
    if cache.exists():
        return cache.read_bytes()
    if failure.exists():
        raise UrlError(failure.read_text().strip())
    reason = "url_error"
    for attempt in range(attempts):
        if attempt:
            time.sleep(min(30, 2 ** attempt))
        try:
            with requests.get(url, timeout=timeout, stream=True, headers={"User-Agent": "k2hv-data/1.0"}) as r:
                if r.status_code != 200:
                    reason = f"url_http_{r.status_code}"
                    if reason in PERMANENT:
                        break
                    continue
                data = b""
                for chunk in r.iter_content(1 << 16):
                    data += chunk
                    if len(data) > max_bytes:
                        raise UrlError("url_too_large")
        except UrlError as e:
            reason = str(e)
            break
        except requests.exceptions.Timeout:
            reason = "url_timeout"
            continue
        except requests.exceptions.RequestException:
            reason = "url_connection"
            continue
        if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
            reason = "url_hash_mismatch"
            break
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path(cache)
        tmp.write_bytes(data)
        os.replace(tmp, cache)
        return data
    if reason in PERMANENT:
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(reason)
    raise UrlError(reason)


def pinned_file(root, spec):
    """Download a pinned URL once into the data root, verify its sha256, and optionally extract one zip member."""
    import zipfile

    import requests

    root = Path(root) / "media" / "pinned"
    archive = root / f"{spec['sha256']}-{Path(spec['url']).name}"
    if not archive.exists() or file_sha256(archive) != spec["sha256"]:
        archive.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path(archive)
        with requests.get(spec["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if file_sha256(tmp) != spec["sha256"]:
            tmp.unlink()
            raise RuntimeError(f"{spec['url']} does not match the pinned sha256")
        os.replace(tmp, archive)
    if not spec.get("member"):
        return archive
    out = root / spec["sha256"] / spec["member"]
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path(out)
        with zipfile.ZipFile(archive) as z, open(tmp, "wb") as f:
            f.write(z.read(spec["member"]))
        os.replace(tmp, out)
    return out


class HttpRangeFile:
    """Seekable read-only file over HTTP Range requests, with a small block cache (for zip members)."""

    def __init__(self, url, block=1 << 20):
        import requests

        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        if r.headers.get("Accept-Ranges") != "bytes":
            raise UrlError("url_no_range_support")
        self.size = int(r.headers["Content-Length"])
        self.etag = r.headers.get("ETag")
        self.last_modified = r.headers.get("Last-Modified")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            start = i * self.block
            end = min(self.size, start + self.block) - 1
            for attempt in range(4):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                    if r.status_code == 206:
                        self.cache[i] = r.content
                        break
                except Exception:
                    pass
                time.sleep(1 + attempt)
            else:
                raise UrlError("url_range_failed")
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, self.block)
            chunk = self._block(i)[off : off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def close(self):
        self.session.close()
