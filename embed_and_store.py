"""Stage 3 of the RAG pipeline: embed chunks and store them in ChromaDB.

Implements planning.md's Retrieval Approach:
  - Embedding model: all-MiniLM-L6-v2 (sentence-transformers, local, no API key)
  - Vector store:    ChromaDB, local persistent collection
  - Similarity:      cosine
  - Top-k:           5

Usage:
  python embed_and_store.py            # (re)build the collection from chunks.jsonl
  python embed_and_store.py --test     # build, then run eval queries and print results
  python embed_and_store.py --query "your question here" [--k 5]

The retrieve() function is imported by Milestone 5 (generation).
"""

import argparse
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.jsonl"
DB_PATH = "chroma_db"
COLLECTION = "dorm_guide"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 5

# 3+ of the 5 evaluation-plan questions from planning.md.
EVAL_QUERIES = [
    "What do students say about sharing a bathroom with a full dorm floor?",
    "What items do students most commonly regret not bringing to their dorm?",
    "What are the biggest roommate conflict triggers according to students?",
    "Is it a good idea to room with your best friend from high school?",
]

# A SentenceTransformer is expensive to construct, so load it once and reuse.
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection():
    """Open (do not recreate) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION)


def build():
    """Embed every chunk and (re)load it into a fresh ChromaDB collection."""
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    model = get_model()
    texts = [c["text"] for c in chunks]
    # normalize_embeddings=True pairs with cosine space below.
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=DB_PATH)
    # Start clean so re-runs don't duplicate or mix stale vectors.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_name": c["source_name"],
                "source_label": c["source_label"],
                "source_url": c["source_url"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    print(f"Stored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION}' at ./{DB_PATH}/")
    return collection


def retrieve(query, k=DEFAULT_K, collection=None):
    """Return the top-k most relevant chunks for `query`.

    Each result: {rank, distance, text, source_label, source_name,
    source_url, chunk_index}. distance is cosine distance (0 = identical,
    higher = less related); roughly, < 0.6 is a solid match.
    """
    collection = collection or get_collection()
    q_emb = get_model().encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for rank, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
    ):
        out.append({
            "rank": rank,
            "distance": dist,
            "text": doc,
            "source_label": meta["source_label"],
            "source_name": meta["source_name"],
            "source_url": meta["source_url"],
            "chunk_index": meta["chunk_index"],
        })
    return out


def print_results(query, results, preview=320):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    for r in results:
        flag = "  <-- weak match (dist > 0.6)" if r["distance"] > 0.6 else ""
        print(f"\n[{r['rank']}] distance={r['distance']:.3f}  "
              f"source={r['source_name']} #{r['chunk_index']}{flag}")
        text = r["text"].strip().replace("\n", " ")
        print(f"    {text[:preview]}{'...' if len(text) > preview else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="run the evaluation queries after building")
    ap.add_argument("--query", type=str, help="run a single ad-hoc query")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--no-build", action="store_true",
                    help="skip rebuilding; query the existing collection")
    args = ap.parse_args()

    collection = get_collection() if args.no_build else build()

    if args.query:
        print_results(args.query, retrieve(args.query, k=args.k, collection=collection))
    if args.test:
        print(f"\n\n########## RETRIEVAL TEST (k={args.k}) ##########")
        for q in EVAL_QUERIES:
            print_results(q, retrieve(q, k=args.k, collection=collection))


if __name__ == "__main__":
    main()
