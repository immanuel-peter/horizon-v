"""Evaluation exclusion, near-duplicate image grouping and question/answer deduplication."""

import re
from collections import defaultdict

import numpy as np

from . import images, text

SUFFIX = re.compile(r"(?i)^(give|write|answer|provide|keep|make|offer|quick|short|concise|your (answer|response)|ensure|be |respond|use |reply|indicate|select|choose|solve the problem|choose the correct option|please|short answer|a very).{0,120}$")


def core_question(question):
    """Question text without trailing brevity/format instruction lines and media markers."""
    lines = [line for line in question.replace("<image>", "").split("\n") if line.strip()]
    while len(lines) > 1 and SUFFIX.match(lines[-1].strip()):
        lines.pop()
    return text.key(" ".join(lines))


def _u64(values):
    return np.asarray(values, dtype=np.int64).view(np.uint64)


def hamming_pairs(left, right, threshold, chunk=4096):
    """Indices (i, j, distance) with Hamming(left[i], right[j]) <= threshold."""
    a, b = _u64(left), _u64(right)
    out = []
    for start in range(0, len(a), chunk):
        x = a[start : start + chunk, None] ^ b[None, :]
        d = np.bitwise_count(x)
        i, j = np.nonzero(d <= threshold)
        out.extend(zip((i + start).tolist(), j.tolist(), d[i, j].tolist()))
    return out


def aligned_metrics(a, b, size=256):
    """Compare two decoded images at one aligned size.

    Returns aspect-ratio difference, gradient (edge) correlation of the grayscale renders, and the
    largest mean absolute RGB difference over a 16x16-pixel block grid. Re-encoded or resized copies
    keep both high correlation and small local differences; same-template charts, maps and forms
    with different content fail one of them.
    """
    a, b = images.to_rgb(a), images.to_rgb(b)
    ratio_a, ratio_b = a.width / a.height, b.width / b.height
    w, h = size, max(16, round(size / ratio_a))
    x = np.asarray(a.resize((w, h), images.Image.Resampling.LANCZOS), np.float64)
    y = np.asarray(b.resize((w, h), images.Image.Resampling.LANCZOS), np.float64)
    gx, gy = x.mean(2), y.mean(2)

    def grad(z):
        return np.hypot(np.diff(z, axis=0)[:, :-1], np.diff(z, axis=1)[:-1, :])

    p, q = grad(gx), grad(gy)
    p, q = p - p.mean(), q - q.mean()
    denom = np.sqrt((p * p).sum() * (q * q).sum())
    edge = float((p * q).sum() / denom) if denom else float(np.abs(gx - gy).mean() < 2)
    diff = np.abs(x - y).mean(2) / 255
    bh, bw = h // 16, w // 16
    block = float(diff[: bh * 16, : bw * 16].reshape(bh, 16, bw, 16).mean((1, 3)).max())
    return {"aspect_diff": round(abs(ratio_a / ratio_b - 1), 4), "edge": round(edge, 4), "max_block": round(block, 4)}


def classify_pair(m, cfg):
    if m["aspect_diff"] > cfg["max_aspect_diff"]:
        return None
    if m["edge"] >= cfg["confirm_edge"] or (m["edge"] >= cfg["confirm_edge_local"] and m["max_block"] <= cfg["confirm_max_block"]):
        return "confirmed"
    if m["edge"] >= cfg["uncertain_edge"]:
        return "uncertain"
    return None


def eval_candidates(cand, evals, cfg):
    """Stage 1: pHash distance and coarse thumbnail correlation. Returns {cand_index: [(eval_index, distance, thumb_sim)]}."""
    found = defaultdict(list)
    for i, j, d in hamming_pairs(cand["phash"], evals["phash"], cfg["eval_phash_distance"]):
        sim = images.thumb_similarity(cand["thumb"][i], evals["thumb"][j])
        if sim >= cfg["prefilter_similarity"]:
            found[i].append((j, d, round(sim, 4)))
    return found


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(phash, thumb, cfg, max_bucket=2000):
    """Stage-1 candidate pairs: pHash distance <= group distance (pigeonhole over four 16-bit bands) and similar thumbnails."""
    h = _u64(phash)
    threshold = cfg["group_phash_distance"]
    seen = set()
    pairs = []
    oversized = 0
    for band in range(4):
        keys = (h >> np.uint64(16 * band)) & np.uint64(0xFFFF)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        bounds = np.flatnonzero(np.diff(sorted_keys)) + 1
        for group in np.split(order, bounds):
            if len(group) < 2:
                continue
            if len(group) > max_bucket:
                oversized += 1
                group = group[:max_bucket]
            sub = h[group]
            d = np.bitwise_count(sub[:, None] ^ sub[None, :])
            ii, jj = np.nonzero(np.triu(d <= threshold, 1))
            for a, b in zip(group[ii].tolist(), group[jj].tolist()):
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                if images.thumb_similarity(thumb[a], thumb[b]) >= cfg["prefilter_similarity"]:
                    pairs.append((a, b))
    return pairs, oversized
