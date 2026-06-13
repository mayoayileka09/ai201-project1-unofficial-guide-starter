"""Stage 2 of the RAG pipeline: chunk the cleaned documents.

Run:  python chunk.py

Implements the Chunking Strategy from planning.md:
  - RecursiveCharacterTextSplitter (LangChain)
  - chunk_size = 400 tokens, chunk_overlap = 50 tokens
  - "tokens" are counted with the all-MiniLM-L6-v2 tokenizer (the same model
    used downstream for embeddings), so the 400/50 numbers are measured against
    the model that will actually consume these chunks.

The recursive splitter prefers to break on paragraph, then sentence, then word
boundaries before resorting to a hard cut — this honors planning.md's mitigation
of "prefer sentence-boundary-aware splitting over hard token counts."

Output:
  - chunks.jsonl  : one JSON object per chunk (text + source metadata)
  - prints per-source counts, the total, a token-length distribution, and
    5 representative chunks for manual inspection.
"""

import glob
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# ---- Strategy from planning.md -------------------------------------------
CHUNK_SIZE = 256      # tokens (matches all-MiniLM-L6-v2's max input; see planning.md)
CHUNK_OVERLAP = 50    # tokens
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# --------------------------------------------------------------------------

CLEAN_DIR = "documents/clean"
OUT_PATH = "chunks.jsonl"


def parse_doc(path):
    """Split a cleaned .txt into (metadata, body).

    The first lines beginning with '#' are the ingest header (label + source
    URL). We keep those as metadata instead of embedding them in chunk text.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    label, url = os.path.basename(path), ""
    body_lines = []
    in_header = True
    for line in text.splitlines():
        if in_header and line.startswith("#"):
            if line.startswith("# Source:"):
                url = line.replace("# Source:", "").strip()
            elif line.startswith("# "):
                label = line[2:].strip()
            continue
        in_header = False
        body_lines.append(line)
    return label, url, "\n".join(body_lines).strip()


def main():
    print(f"Loading tokenizer for {EMBED_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    def n_tokens(s):
        return len(tokenizer.encode(s, add_special_tokens=False))

    files = sorted(glob.glob(f"{CLEAN_DIR}/*.txt"))
    all_chunks = []
    per_source = []

    for path in files:
        label, url, body = parse_doc(path)
        name = os.path.splitext(os.path.basename(path))[0]
        pieces = splitter.split_text(body)
        pieces = [p for p in pieces if p.strip()]   # guard: drop empty/whitespace chunks
        per_source.append((name, len(pieces)))
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "id": f"{name}__{i}",
                "source_name": name,
                "source_label": label,
                "source_url": url,
                "chunk_index": i,
                "n_tokens": n_tokens(piece),
                "text": piece,
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- report -----------------------------------------------------------
    print("\n=== Chunks per source ===")
    for name, n in per_source:
        print(f"  {name:<34} {n:>4}")

    total = len(all_chunks)
    toks = [c["n_tokens"] for c in all_chunks]
    print(f"\n=== TOTAL CHUNKS: {total} ===")
    print(f"Token length per chunk -> min {min(toks)}, "
          f"max {max(toks)}, avg {sum(toks) / len(toks):.0f}")
    tiny = sum(1 for t in toks if t < 50)
    print(f"Chunks under 50 tokens (possible fragments): {tiny}")

    if total < 50:
        print("NOTE: < 50 chunks total — chunks may be too large for the corpus size.")
    elif total > 2000:
        print("NOTE: > 2000 chunks total — chunks may be too small.")
    else:
        print("Chunk count is within the healthy 50-2000 range.")

    # ---- 5 representative chunks, spread across the corpus ----------------
    print("\n=== 5 REPRESENTATIVE CHUNKS (inspect for standalone meaning) ===")
    step = max(1, total // 5)
    for k in range(5):
        idx = min(k * step, total - 1)
        c = all_chunks[idx]
        print(f"\n--- chunk #{idx}  [{c['source_name']}]  "
              f"({c['n_tokens']} tokens) ---")
        print(c["text"])

    print(f"\nWrote {total} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
