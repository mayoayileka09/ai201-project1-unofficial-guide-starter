# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) question-answering system about **college dorm and
housing life**, grounded in real student reviews, forum threads, and blog posts.

**Pipeline:** ingest & clean (`ingest.py`) → chunk (`chunker.py`) → embed & store
in ChromaDB (`embed_and_store.py`) → grounded generation (`query.py`) → Gradio UI
(`app.py`).

**Run it:**
```bash
pip install -r requirements.txt
python ingest.py            # fetch + clean the 10 sources -> documents/clean/
python chunker.py           # chunk -> chunks.jsonl  (68 chunks)
python embed_and_store.py   # embed + load into ./chroma_db/
python app.py               # Gradio UI at http://localhost:7860
```
Requires a free `GROQ_API_KEY` in `.env` (see `.env.example`).

---

## Domain

This system covers **what it's actually like to live in a college dorm** —
roommates, what to pack, communal bathrooms, and adjusting to freshman year.

Official channels (university housing pages, brochures) describe dorms in
sanitized, recycled marketing language: square footage, meal-plan tiers, and
amenity lists. They don't tell you that your roommate might store a birthday
cake in the fridge for four months, that you should never walk barefoot in a
communal bathroom, or that rooming with your high-school best friend can quietly
wreck the friendship. That lived-experience knowledge is scattered across
forums, student blogs, and review sites — exactly the kind of crowdsourced,
informal, honest content this system curates and makes queryable.

---

## Document Sources

10 sources spanning review sites, two forum platforms, student/university blogs,
and news articles — chosen for variety of perspective (current students, alumni
reflecting a year later, parents, and viral social-media anecdotes).

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | College Dorm Reviews | Review site (landing page) | https://collegedormreviews.com |
| 2 | The Daily Pennsylvanian | News article | https://www.thedp.com/article/2016/06/new-student-issue-tips-dorm-living |
| 3 | College Confidential — random vs. chosen roommate | Forum thread (Discourse) | https://talk.collegeconfidential.com/t/random-roommate-vs-choosing-roommate/1811434 |
| 4 | College Confidential — single vs. roommate | Forum thread (Discourse) | https://talk.collegeconfidential.com/t/single-vs-roommate-freshman-year/125580 |
| 5 | AnandTech — what to bring to a dorm | Forum thread (XenForo) | https://forums.anandtech.com/threads/things-that-must-be-brought-into-a-freshman-dorm.196580/ |
| 6 | AnandTech — roommates in college | Forum thread (XenForo) | https://forums.anandtech.com/threads/roommates-in-college.833171/ |
| 7 | Amherst Student Blog | Blog post | https://admissionstudentblogs.wordpress.amherst.edu/?p=2911 |
| 8 | Purdue Ambassador Blog | Blog post | https://ag.purdue.edu/agry/ambassadorblog/dorm-life-advice |
| 9 | Grown and Flown | Blog article | https://grownandflown.com/student-wishes-she-had-known-before-freshman-year-college/ |
| 10 | In The Know / AOL | News article | https://www.aol.com/lifestyle/college-students-compare-freshman-dorm-183712927.html |

**Scraping notes (this was not uniform):** A generic `requests` + BeautifulSoup
scrape only worked for the article/blog sources. Two sources needed
platform-specific handling: **College Confidential** (#3, #4) is a JavaScript-
rendered Discourse forum that returns an empty `<body>` to a plain GET, so I
fetch its `.json` topic API instead; **AnandTech** (#5, #6) is XenForo, where
post text lives in `.bbWrapper` divs rather than `<p>` tags. **College Dorm
Reviews** (#1) renders its actual per-dorm reviews with JavaScript, so only its
generic descriptive copy is reachable without a headless browser — this source
is intentionally "thin" and its limits show up in the evaluation below.

---

## Chunking Strategy

**Chunk size:** 256 tokens (measured with the `all-MiniLM-L6-v2` tokenizer)

**Overlap:** 50 tokens

**Splitter:** LangChain `RecursiveCharacterTextSplitter`, configured
`from_huggingface_tokenizer(...)` with the embedding model's own tokenizer and
separators `["\n\n", "\n", ". ", "! ", "? ", " ", ""]` so it prefers paragraph
and sentence boundaries before resorting to a hard cut.

**Preprocessing before chunking:** HTML stripped; site furniture removed (nav,
ads, cookie banners, share buttons, "read more," related-article tails, a
`[CLOSE]` modal artifact, an injected student-loan promo, and a "please leave a
review" call-to-action); HTML entities (`&amp;`, `&#39;`, `&nbsp;`) unescaped;
Discourse quote-blocks dropped to avoid duplicating quoted text across posts.
Source label and URL are stored as **metadata**, not embedded into chunk text.

**Why these choices fit my documents:** The corpus is short, conversational
forum posts and blog snippets, not long-form articles. 256 tokens is large
enough to hold a complete anecdote (a roommate war story, a packing tip with its
reasoning) but small enough that one chunk is about one thought.

**Final chunk count:** **68 chunks** across 10 documents — avg 202 tokens, max
256, min 51, zero empty chunks. Comfortably inside the healthy 50–2,000 range.

**Why 400 → 256 (a deliberate change from the original spec):** I first specced
400 tokens. Measuring it revealed two problems: (1) it produced only **46
chunks**, below the 50-chunk floor that signals chunks are too large; and (2)
`all-MiniLM-L6-v2` **truncates input at 256 tokens**, so the back third of every
full 400-token chunk would never reach the embedding vector while still being
returned as context — a silent retrieval bug. Dropping to 256 makes every chunk
fit the model exactly and raised the count to 68. This change is recorded in
`planning.md`.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` (via `sentence-transformers`), producing
384-dimensional embeddings. Stored in a **ChromaDB** persistent collection with
**cosine** distance (`hnsw:space: "cosine"`); embeddings are L2-normalized at
encode time. Retrieval returns **top-k = 5**.

**Why:** It runs locally with no API key, no rate limits, and no cost, and it's
well-suited to the short, informal text in this corpus. It also made the
256-token chunk decision concrete, since 256 is exactly its max input length.

**Production tradeoff reflection:** If I were deploying this for real users with
no cost constraint, I'd weigh:

- **Accuracy vs. latency/cost.** A larger model like OpenAI's
  `text-embedding-3-large` or `instructor-xl` would produce stronger embeddings
  for nuanced or differently-phrased queries (this matters directly — see the
  Failure Case below, where MiniLM failed to connect a negatively-phrased query
  to positively-phrased advice). The cost is API fees and higher latency.
- **Context length.** MiniLM's 256-token limit forced my chunk size.
  `text-embedding-3-large` handles up to 8,191 tokens, which would allow larger,
  more context-rich chunks without truncation, reducing the chance that a
  complete thought gets split.
- **Domain specificity.** Student slang and informal phrasing are under-
  represented in most embedding training data. A model fine-tuned on student-
  generated text would improve precision, though that's impractical at this
  scale.

---

## Grounded Generation

**LLM:** Groq `llama-3.3-70b-versatile` (free tier, OpenAI-compatible), called at
`temperature=0.2`.

Grounding is enforced **three ways**, so it does not rely on the model choosing
to behave:

**1. System-prompt grounding instruction** (the actual instruction used):

> Use ONLY information found in the CONTEXT. Do not use any outside or prior
> knowledge, and do not guess or generalize beyond what the excerpts say. If the
> CONTEXT does not contain enough information to answer, reply with EXACTLY this
> sentence and nothing else: *"I don't have enough information on that."* Cite
> the excerpts you use with their bracketed numbers, e.g. [1], [3].

Retrieved chunks are passed as numbered, source-labeled `CONTEXT` blocks.

**2. A relevance gate (structural, not prompt-based).** Before any LLM call, if
the best retrieved chunk's cosine distance exceeds **0.70**, the system returns
the refusal string immediately and never calls the model. On-topic eval queries
score 0.25–0.42; an off-topic query ("best graphics card for gaming") scored
0.878 and was refused with no LLM call — so the model is never even given the
chance to hallucinate on uncovered topics.

**3. Programmatic source attribution.** `result["sources"]` is built in code from
the retrieved chunks' metadata (de-duplicated source label + URL). It does **not**
depend on the model remembering to cite. The model is *also* asked to add inline
`[n]` markers, but the authoritative source list is code-generated, so attribution
is guaranteed whenever an answer is produced.

**How source attribution is surfaced in the response:** The Gradio UI shows the
answer in one box and a bulleted "Retrieved from" list (source name + URL) in a
second box. If the system refuses for lack of context, the source list is empty
(nothing was used to answer).

---

## Evaluation Report

Run on all 5 planning.md questions (top-k = 5, `llama-3.3-70b-versatile`).
Distances are the top retrieved chunk's cosine distance.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about sharing a bathroom with a full dorm floor? | Communal bathrooms can be dirty/crowded; bring shower shoes/flip flops; manage hygiene in shared space | Cited Purdue's advice to bring a shower caddy, flip flops, and a robe; noted dorm bathrooms vary in size and stalls fill up at rush hour. Honestly flagged that the corpus doesn't say much beyond this. (top dist 0.371) | Partially relevant (topic is thin in corpus) | **Partially accurate** |
| 2 | What items do students most commonly regret not bringing? | Shower caddy, mattress topper, command hooks, power strip, basic medicine | Listed fan, rug, air purifier, command hooks, power strip, mini-fridge; correctly noted the strongest theme is *over*-packing (decorative pillows, excess school supplies). Missed "mattress topper." (top dist 0.315) | Relevant | **Partially accurate** |
| 3 | Biggest roommate conflict triggers? | Sleep-schedule mismatches, guests/SO staying over, cleanliness, study-time noise | Cited cleanliness, lifestyle/partying differences, disrespecting personal space (smoking in-room, closing a laptop), and clashing sleep/study schedules ("studying at 2am"). (top dist 0.364) | Relevant | **Accurate** |
| 4 | Room with your high-school best friend? | Mixed — many warn it damages the friendship; others say it works with clear communication | "Not necessarily a good idea… if it doesn't work out it can damage the friendship; okay only if you know them well enough and have similar lifestyle habits." Matches the mixed verdict closely. (top dist 0.421) | Relevant | **Accurate** |
| 5 | Staying in your dorm room too much freshman year? | Leave the room — library, dining hall, clubs; isolation hurts social life and motivation | Talked about dorms being bad for studying and roommates making you more outgoing; **explicitly admitted "there's no direct mention"** of the consequences of staying in too much. Did not surface the corpus's actual "get out / don't be a stranger" advice. (top dist 0.378) | **Off-target** | **Inaccurate** |

**Overall:** 2 accurate, 2 partially accurate, 1 inaccurate. The grounding held
throughout — no answer invented facts; where the corpus was thin, the model said
so rather than filling the gap from training knowledge.

---

## Failure Case Analysis

**Question that failed:** *"What do students say about staying in your dorm room
too much freshman year?"* (Q5)

**What the system returned:** An answer about dorms being poor study environments
and roommates encouraging sociability, with the explicit admission that "there's
no direct mention of the consequences of staying in your dorm room too much."
The expected advice — *get out of your room, you'll isolate yourself, go to the
library/dining hall/clubs* — never appeared.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval**
failure with two compounding causes, and crucially **the relevant content IS in
the corpus** — it just never reached the top-5.

1. **Embedding semantic gap.** The query is framed as a *negative behavior*
   ("staying in your dorm room too much"). The corpus's relevant advice is framed
   *positively and indirectly*: the Daily Pennsylvanian says *"Don't be a
   stranger: make friends with your hall… feel comfortable hanging outside of
   your room,"* and Purdue says *"Study other places on campus rather than just
   your dorm room."* Neither contains the words "staying too much" or
   "isolation." `all-MiniLM-L6-v2`, a small bi-encoder, embedded the query
   closest to chunks that share surface vocabulary ("dorm room," "study,"
   "freshman year") rather than the chunks that share the *intended meaning*.

2. **A too-broad chunk diluted the signal.** The DP advice that actually answers
   the question lives in a single chunk that packs ~8 unrelated tips (hide your
   food, do laundry, buy a mattress pad, "don't be a stranger," watch your dorm
   measurements…). Averaging all those topics into one 255-token embedding
   weakens the "socializing" signal, so even the right chunk matches a
   socializing query only weakly.

**Evidence:** Ranking all 68 chunks by distance for this query, the DP chunk
containing the real answer sits at **rank 28 (distance 0.518)** — far outside the
top-5 (which ranged 0.378–0.410). Because the correct context was never
retrieved, generation had nothing to ground the expected answer in and honestly
fell back to the loosely-related chunks it was given.

**What I would change to fix it:**
- **Topic-segment the multi-tip sources.** The DP and Grown-and-Flown posts are
  lists of distinct tips; splitting on each tip (rather than by token count)
  would give "don't be a stranger" its own focused embedding instead of burying
  it among seven unrelated ones.
- **Query expansion / HyDE.** Rephrase or expand the query before retrieval
  (e.g., generate a hypothetical answer and embed *that*), which would bridge the
  negative-behavior → positive-advice gap.
- **A stronger embedding model** (e.g., `text-embedding-3-large`) that better
  captures intent over surface vocabulary — directly the tradeoff discussed in
  the Embedding Model section.

---

## Spec Reflection

**One way the spec helped me during implementation:** Defining the Retrieval
Approach and the architecture diagram *before* coding meant each implementation
step had concrete targets — embedding model, top-k, vector store, and distance
metric were all decided, so `embed_and_store.py` was a direct translation rather
than a series of mid-build decisions. Writing the five evaluation questions up
front was especially valuable: they gave me a fixed yardstick to test retrieval
against at Milestone 4, which is how I caught that Q5 was weak well before the
final write-up instead of discovering it during grading.

**One way my implementation diverged from the spec, and why:** Two divergences.
(1) The architecture diagram originally named the Claude API
(`claude-sonnet-4-6`) for generation, but the project's `requirements.txt` and
free-tier guidance pointed to Groq, so I implemented generation with Groq
`llama-3.3-70b-versatile` — same role in the pipeline, different provider, chosen
for zero cost and no API-key friction. (2) The spec assumed a single
BeautifulSoup/`requests` ingestion path for all sources, but two of my forum
sources were JavaScript-rendered (Discourse) or stored post text outside `<p>`
tags (XenForo), forcing source-specific extraction (a `.json` API call and a
`.bbWrapper` selector). I'd written "noisy forum content" as an anticipated
challenge in planning.md, but the harder reality was simply *reaching* that
content at all.

---

## AI Usage

I used Claude (in Claude Code) as the AI tool throughout. Two specific instances:

**Instance 1 — Ingestion & chunking**

- *What I gave the AI:* My planning.md Documents table, Chunking Strategy section,
  and architecture diagram, and asked it to implement a script that loads each
  source, cleans boilerplate, and chunks with `RecursiveCharacterTextSplitter` at
  my specified size/overlap.
- *What it produced:* An `ingest.py` + `chunker.py` pipeline using
  `requests`/BeautifulSoup and the LangChain splitter.
- *What I changed or overrode:* Two things. First, when the generic scrape
  returned empty text for College Confidential and AnandTech, I directed it to
  add platform-specific extraction — the Discourse `.json` API and the XenForo
  `.bbWrapper` selector — rather than accept empty documents. Second, I overrode
  the chunk size from 400 to 256 after measuring that 400 produced only 46 chunks
  *and* exceeded MiniLM's 256-token limit (silent truncation); the AI implemented
  400 as specced, and I made the call to change it and update planning.md.

**Instance 2 — Grounded generation**

- *What I gave the AI:* My grounding requirement ("answer from retrieved context
  only, with source attribution"), the desired output format (answer + source
  list), and the Gradio skeleton, and asked it to wire retrieval to the Groq LLM.
- *What it produced:* A first version where grounding lived entirely in the
  system prompt and citations were left to the model to write.
- *What I changed or overrode:* I judged prompt-only grounding too weak to trust,
  so I directed two structural additions: (1) a cosine-distance **relevance gate**
  that refuses off-topic queries *before* the LLM is called, and (2)
  **programmatic source attribution** built from chunk metadata in code, so
  citations are guaranteed rather than dependent on the model. I verified both by
  running an uncovered question ("best graphics card") and confirming it refused
  with no sources.

**Instance 3 — Debugging a runtime crash**

- *What I gave the AI:* The `ModuleNotFoundError` / `RuntimeError: Numpy is not
  available` traceback when launching the app in my virtual environment.
- *What it produced / I directed:* It diagnosed that dependencies had been
  installed into the wrong interpreter and that `torch 2.2.x` is incompatible
  with NumPy 2.x, then pinned `numpy<2` in `requirements.txt` and reinstalled into
  `.venv`. I verified the fix by re-running encode, the store build, a query, and
  the app end-to-end.
