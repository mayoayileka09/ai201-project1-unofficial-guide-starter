"""Stage 5 of the RAG pipeline: a Gradio web interface for The Unofficial Guide.

Run:  python app.py
Then open http://localhost:7860

Wraps query.ask() — retrieval + grounded generation — in a minimal UI that
shows the answer and the programmatically-attributed sources side by side.
"""

import gradio as gr

from query import ask

EXAMPLES = [
    "What items do students most commonly regret not bringing to their dorm?",
    "Is it a good idea to room with your best friend from high school?",
    "What are the biggest roommate conflict triggers?",
    "What do students say about staying in your dorm room too much?",
]


def handle_query(question):
    if not question or not question.strip():
        return "Please enter a question.", ""
    result = ask(question.strip())
    answer = result["answer"]
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources — the guide didn't have enough information)"
    return answer, sources


with gr.Blocks(title="The Unofficial Guide") as demo:
    gr.Markdown(
        "# 🏠 The Unofficial Guide to Dorm Life\n"
        "Ask about dorm living, roommates, and what to pack. Answers come "
        "**only** from real student reviews, forum threads, and blog posts — "
        "if the documents don't cover it, the guide will say so."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What should I bring to my dorm?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
