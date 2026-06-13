"""Stage 4 of the RAG pipeline: grounded answer generation.

Implements the Generation step from planning.md's architecture diagram:
  retrieve top-k chunks  ->  build a grounded prompt  ->  Groq LLM  ->  answer

Grounding is enforced three ways, not just suggested:
  1. A strict system prompt that forbids using any knowledge outside the
     provided context and mandates a fixed refusal string when the context is
     insufficient.
  2. A relevance gate: if even the best retrieved chunk is a weak match
     (cosine distance above DISTANCE_GATE), we refuse *without* calling the LLM,
     so an off-topic question can't coax a confident hallucination.
  3. Source attribution is built programmatically from the retrieved chunks'
     metadata and returned in result["sources"] — it does NOT depend on the
     model remembering to cite. The model is also asked to cite inline [n]
     markers, but the authoritative source list is code-generated.

Usage:
  python query.py "What should I bring to my dorm?"
  from query import ask
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from embed_and_store import retrieve

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
# Cosine distance above which we treat retrieval as "no real match" and refuse.
# Our on-topic eval queries score 0.31-0.53; clearly off-topic queries land
# well above this, so 0.70 cleanly separates covered from uncovered questions.
DISTANCE_GATE = 0.70
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are The Unofficial Guide, a candid assistant that answers questions "
    "about college dorm and housing life using ONLY the student-written excerpts "
    "provided in the CONTEXT.\n\n"
    "Rules:\n"
    "1. Use ONLY information found in the CONTEXT. Do not use any outside or "
    "prior knowledge, and do not guess or generalize beyond what the excerpts "
    "say.\n"
    f"2. If the CONTEXT does not contain enough information to answer, reply "
    f"with EXACTLY this sentence and nothing else: \"{REFUSAL}\"\n"
    "3. Cite the excerpts you use with their bracketed numbers, e.g. [1], [3].\n"
    "4. Answer in a conversational, honest tone — like an upperclassman giving "
    "real advice — but every claim must trace back to the CONTEXT."
)


def _client():
    key = os.getenv("GROQ_API_KEY")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "GROQ_API_KEY missing or placeholder. Put a real key in .env "
            "(get one free at https://console.groq.com)."
        )
    return Groq(api_key=key)


def build_context(chunks):
    """Render retrieved chunks as numbered, source-labeled context blocks."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] (source: {c['source_label']})\n{c['text'].strip()}"
        )
    return "\n\n".join(blocks)


def format_sources(chunks):
    """Programmatic, de-duplicated source list from retrieved metadata."""
    seen, sources = set(), []
    for c in chunks:
        key = c["source_name"]
        if key in seen:
            continue
        seen.add(key)
        label = c["source_label"]
        url = c.get("source_url", "")
        sources.append(f"{label} ({url})" if url else label)
    return sources


def ask(question, k=TOP_K):
    """End-to-end: retrieve, ground, generate.

    Returns {"answer": str, "sources": list[str], "chunks": list[dict],
    "grounded": bool}. `grounded` is False when we refused for lack of context.
    """
    chunks = retrieve(question, k=k)

    # Relevance gate — refuse before spending an LLM call on an off-topic query.
    if not chunks or chunks[0]["distance"] > DISTANCE_GATE:
        return {"answer": REFUSAL, "sources": [], "chunks": chunks,
                "grounded": False}

    context = build_context(chunks)
    user_msg = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above, following the rules."
    )

    resp = _client().chat.completions.create(
        model=MODEL,
        temperature=0.2,  # low: keep answers tight to the context
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    refused = answer.strip().rstrip(".").lower() == REFUSAL.rstrip(".").lower()
    return {
        "answer": answer,
        # If the model refused, don't attach sources — nothing was used.
        "sources": [] if refused else format_sources(chunks),
        "chunks": chunks,
        "grounded": not refused,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What should I bring to my dorm?"
    result = ask(q)
    print(f"\nQ: {q}\n")
    print("ANSWER:\n" + result["answer"] + "\n")
    if result["sources"]:
        print("SOURCES:")
        for s in result["sources"]:
            print("  • " + s)
    print(f"\n[grounded={result['grounded']}, "
          f"top_distance={result['chunks'][0]['distance']:.3f}]"
          if result["chunks"] else "[no chunks retrieved]")
