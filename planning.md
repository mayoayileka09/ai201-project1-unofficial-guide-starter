# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | College Dorm Reviews | Crowdsourced student reviews of 6,500+ dorms across 1,400 schools, rated on size, bathrooms, noise, and party scene | https://collegedormreviews.com |
| 2 | The Daily Pennsylvanian | Penn upperclassmen give blunt do's and don'ts to incoming freshmen (hide your food, do laundry early, befriend your hall) | https://www.thedp.com/article/2016/06/new-student-issue-tips-dorm-living |
| 3 | College Confidential | Long thread weighing random vs. self-chosen roommates, with real outcomes from students and parents | https://talk.collegeconfidential.com/t/random-roommate-vs-choosing-roommate/1811434 |
| 4 | College Confidential | Single vs. roommate debate — students candidly weigh privacy against the social/"college experience" tradeoff | https://talk.collegeconfidential.com/t/single-vs-roommate-freshman-year/125580 |
| 5 | AnandTech Forums | Crowd-built list of what to actually bring to a freshman dorm, including overlooked items and roommate coordination tips | https://forums.anandtech.com/threads/things-that-must-be-brought-into-a-freshman-dorm.196580/ |
| 6 | AnandTech Forums | Unfiltered roommate war stories and success stories across multiple years of dorm living | https://forums.anandtech.com/threads/roommates-in-college.833171/ |
| 7 | Amherst Student Blog | Honest one-year-later reflection on what gear got used (noise-canceling headphones, shower slides, command hooks) vs. what stayed in the closet | https://admissionstudentblogs.wordpress.amherst.edu/?p=2911 |
| 8 | Purdue Ambassador Blog | First-person survival advice on roommate communication, shower caddies, and finding study spots outside your room | https://ag.purdue.edu/agry/ambassadorblog/dorm-life-advice |
| 9 | Grown and Flown | Ithaca student's seven things she wishes she'd known — mattress toppers, burnout, prioritizing your own wellbeing | https://grownandflown.com/student-wishes-she-had-known-before-freshman-year-college/ |
| 10 | In The Know / AOL | Roundup of viral TikTok freshman roommate experiences — real red flags and what made some pairings work | https://www.aol.com/lifestyle/college-students-compare-freshman-dorm-183712927.html |

---


## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
