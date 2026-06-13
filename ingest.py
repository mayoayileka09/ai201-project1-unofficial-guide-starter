"""Stage 1 of the RAG pipeline: load documents and clean them.

Run:  python ingest.py

For each source in sources.py this script:
  1. Fetches the page (or the Discourse .json API for forum threads),
  2. Saves the raw response to documents/raw/  (so we never re-hit the network
     while iterating on cleaning, and have an audit trail),
  3. Extracts only the substantive text (reviews, advice, opinions) and strips
     boilerplate (nav, ads, cookie banners, share buttons, footers, HTML
     entities, quoted reply blocks),
  4. Writes one cleaned .txt per source to documents/clean/.

Local files: drop any manually-collected .txt/.md files into documents/local/
and they are picked up and cleaned too (useful for JS-heavy sources you copy
by hand, e.g. Rate My Professors style pages).
"""

import html
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

from sources import SOURCES

RAW_DIR = "documents/raw"
CLEAN_DIR = "documents/clean"
LOCAL_DIR = "documents/local"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}

# Lines/phrases that show up as site furniture, not content. Matched
# case-insensitively against whole stripped lines, or as substrings to drop.
BOILERPLATE_PATTERNS = [
    r"^advertisement$",
    r"^for premium support please call",
    r"^read more$",
    r"^share this",
    r"^sign up",
    r"^subscribe",
    r"^leave a (comment|reply)",
    r"^\d+ comments?$",
    r"^posted on ",
    r"^tags?:",
    r"^categories?:",
    r"^skip to (main )?content$",
    r"^menu$",
    r"^search$",
    r"^cookie",
    r"accept all cookies",
    r"^\[close\]$",                       # leftover modal close button
    r"student loan relief program",       # AOL injected off-topic promo
    r"please leave a (review|comment)",   # CollegeDormReviews site call-to-action
]
BOILERPLATE_RE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]

# Once a line matches one of these, it and everything after it is footer /
# related-article furniture — truncate the document there.
END_MARKER_PATTERNS = [
    r"appeared first on",
    r"^more from ",
    r"^you might also (want to read|like)",
    r"^related posts?$",
    r"^recommended for you",
    r"^more stories",
]
END_MARKER_RE = [re.compile(p, re.IGNORECASE) for p in END_MARKER_PATTERNS]


def truncate_at_end_markers(text):
    """Cut the document at the first footer/related-links marker."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and any(rx.search(s) for rx in END_MARKER_RE):
            return "\n".join(lines[:i]).strip()
    return text


def normalize_text(text):
    """Unescape HTML entities and normalize whitespace, preserving paragraphs."""
    text = html.unescape(text)                 # &amp; &#39; &nbsp; -> & ' space
    text = text.replace("\xa0", " ")           # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)        # collapse intra-line whitespace
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # collapse blank-line runs
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def drop_boilerplate_lines(text):
    """Remove whole lines that match known site-furniture patterns."""
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if any(rx.search(s) for rx in BOILERPLATE_RE):
            continue
        kept.append(ln)
    # collapse blank runs again after dropping
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", "\n".join(kept))
    return out.strip()


def fetch(url, as_json=False):
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json() if as_json else r.text


# ---------- per-kind extractors ----------

def extract_article(html_text):
    """Standard article page: prefer a main content container, else all <p>."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                     "aside", "form", "button", "figure", "iframe"]):
        tag.decompose()

    # Try the usual main-content containers first.
    container = None
    for sel in ["article", "main", ".entry-content", ".post-content",
                ".article-content", ".td-post-content", ".post-body"]:
        node = soup.select_one(sel)
        if node and len(node.get_text(strip=True)) > 400:
            container = node
            break
    scope = container or soup

    paras = [p.get_text(" ", strip=True) for p in scope.find_all(["p", "li"])]
    paras = [p for p in paras if len(p) > 1]
    return "\n\n".join(paras)


def extract_discourse(url):
    """Discourse forum: hit the .json topic API and concatenate every post."""
    data = fetch(url.rstrip("/") + ".json", as_json=True)
    title = data.get("title", "")
    posts = data.get("post_stream", {}).get("posts", [])
    blocks = []
    if title:
        blocks.append(f"THREAD: {title}")
    for p in posts:
        # Some threads return `cooked` as double-escaped HTML (&lt;p&gt;...).
        # Unescaping first means BeautifulSoup sees real tags either way and
        # get_text() strips them instead of leaving literal <p> in the output.
        cooked = html.unescape(p.get("cooked", ""))
        soup = BeautifulSoup(cooked, "html.parser")
        # Strip quoted-reply blocks so we don't duplicate text across posts.
        for q in soup.select("aside.quote, blockquote"):
            q.decompose()
        body = soup.get_text(" ", strip=True)
        if body:
            blocks.append(body)
    return "\n\n".join(blocks), data  # also return raw json for archiving


def extract_xenforo(html_text):
    """XenForo forum: post bodies live in .bbWrapper; drop quoted blocks."""
    soup = BeautifulSoup(html_text, "html.parser")
    wrappers = soup.select(".message-body .bbWrapper") or soup.select(".bbWrapper")
    blocks = []
    for w in wrappers:
        for q in w.select("blockquote, .bbCodeBlock"):
            q.decompose()
        body = w.get_text(" ", strip=True)
        if body:
            blocks.append(body)
    return "\n\n".join(blocks)


def extract_homepage(html_text):
    """Landing page: only generic <p> copy is reachable without a browser."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n\n".join(p for p in paras if len(p) > 1)


# ---------- driver ----------

def process_source(src):
    name, kind, url = src["name"], src["kind"], src["url"]
    raw_text = None

    if kind == "discourse":
        cleaned_body, raw_json = extract_discourse(url)
        with open(f"{RAW_DIR}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(raw_json, f, ensure_ascii=False)
        extracted = cleaned_body
    else:
        raw_text = fetch(url)
        with open(f"{RAW_DIR}/{name}.html", "w", encoding="utf-8") as f:
            f.write(raw_text)
        if kind == "article":
            extracted = extract_article(raw_text)
        elif kind == "xenforo":
            extracted = extract_xenforo(raw_text)
        elif kind == "homepage":
            extracted = extract_homepage(raw_text)
        else:
            raise ValueError(f"unknown kind: {kind}")

    cleaned = drop_boilerplate_lines(truncate_at_end_markers(normalize_text(extracted)))
    header = f"# {src['label']}\n# Source: {url}\n\n"
    with open(f"{CLEAN_DIR}/{name}.txt", "w", encoding="utf-8") as f:
        f.write(header + cleaned)
    return len(cleaned)


def process_local_files():
    """Clean any manually-collected files dropped into documents/local/."""
    results = []
    if not os.path.isdir(LOCAL_DIR):
        return results
    for fn in sorted(os.listdir(LOCAL_DIR)):
        if fn.startswith(".") or not fn.lower().endswith((".txt", ".md")):
            continue
        with open(f"{LOCAL_DIR}/{fn}", encoding="utf-8") as f:
            text = f.read()
        cleaned = drop_boilerplate_lines(truncate_at_end_markers(normalize_text(text)))
        out_name = "local_" + os.path.splitext(fn)[0]
        header = f"# Local file: {fn}\n\n"
        with open(f"{CLEAN_DIR}/{out_name}.txt", "w", encoding="utf-8") as f:
            f.write(header + cleaned)
        results.append((out_name, len(cleaned)))
    return results


def main():
    for d in (RAW_DIR, CLEAN_DIR, LOCAL_DIR):
        os.makedirs(d, exist_ok=True)

    print(f"Ingesting {len(SOURCES)} web sources...\n")
    total = 0
    for src in SOURCES:
        try:
            n = process_source(src)
            total += 1
            flag = "  (thin — JS-rendered)" if (src["kind"] == "homepage" and n < 2500) else ""
            print(f"  [{src['id']:>2}] {src['name']:<32} {n:>6} chars{flag}")
        except Exception as e:
            print(f"  [{src['id']:>2}] {src['name']:<32} FAILED: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(0.5)  # be polite to servers

    local = process_local_files()
    if local:
        print("\nLocal files:")
        for name, n in local:
            print(f"       {name:<32} {n:>6} chars")

    print(f"\nDone. Cleaned text written to {CLEAN_DIR}/ ({total} web sources"
          f"{', ' + str(len(local)) + ' local' if local else ''}).")


if __name__ == "__main__":
    main()
