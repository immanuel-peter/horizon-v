"""Render README.md and docs/**/*.md into a single-file VS Code-style viewer."""

import argparse
import datetime
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
VENDOR = {
    "marked.min.js": "https://cdn.jsdelivr.net/npm/marked@15.0.12/marked.min.js",
    "highlight.min.js": "https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/highlight.min.js",
    "django.min.js": "https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/languages/django.min.js",
}

STANDALONE_CSS = """html, body { margin: 0; height: 100%; overflow: hidden; background: #1f1f1f; }
.window { height: 100vh; height: 100dvh; }"""

# T3 Code embeds reject viewport units and size the frame to the page.
EMBED_CSS = """html, body { margin: 0; }
.window { height: 720px; border: 1px solid #2b2b2b; border-radius: var(--radius, 10px); }
@media (max-width: 480px) { .window { height: 640px; } }"""


def collect_docs() -> dict:
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    return {
        "files": [{"path": p.relative_to(ROOT).as_posix(), "content": p.read_text()} for p in paths if p.is_file()],
        "defaultPath": "docs/README.md",
        "built": datetime.date.today().isoformat(),
    }


def script_safe(js: str) -> str:
    # Keeps the HTML parser from treating <!-- or <script inside a script as markup.
    return re.sub(r"<(/?script|!--)", r"\\x3C\1", js, flags=re.IGNORECASE)


def between(text: str, start: str, end: str) -> str:
    i = text.index(start) + len(start)
    return text[i : text.index(end, i)]


def build_page(docs: dict) -> str:
    vendor = "\n".join(script_safe((TOOLS / "vendor" / name).read_text()) for name in VENDOR)
    payload = json.dumps(docs, ensure_ascii=False).replace("<", "\\u003c")
    return (
        (TOOLS / "docs_viewer.html").read_text()
        .replace("/*__FRAME_CSS__*/", STANDALONE_CSS)
        .replace("/*__VENDOR_JS__*/", vendor)
        .replace("/*__BOOT__*/", f"boot({payload});")
    )


def build_bundle(docs: dict) -> dict:
    """Viewer split into parts, for hosts that load it with a small inline loader."""
    html = (TOOLS / "docs_viewer.html").read_text()
    return {
        "css": between(html, "<style>", "</style>").replace("/*__FRAME_CSS__*/", EMBED_CSS),
        "body": between(html, "<body>", "<script>/*__VENDOR_JS__*/</script>").strip(),
        "vendor": list(VENDOR.values()),
        "script": between(html, "<script>/*__VENDOR_JS__*/</script>\n<script>", "/*__BOOT__*/"),
        "docs": docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "index.html")
    parser.add_argument("--bundle", type=pathlib.Path, help="also write the embed bundle as JSON")
    args = parser.parse_args()

    docs = collect_docs()
    html = build_page(docs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{args.out}: {len(html):,} chars, {len(docs['files'])} docs")
    if args.bundle:
        args.bundle.write_text(json.dumps(build_bundle(docs), ensure_ascii=False))
        print(f"{args.bundle}: {args.bundle.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
