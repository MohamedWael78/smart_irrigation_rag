"""
Offline evaluation harness for the AquaMind retrieval + generation pipeline.

Run with:  python eval.py

Scores each question on:
  - faithfulness        : is the answer grounded in the retrieved context?
  - answer_relevancy     : does the answer actually address the question?
  - context_precision     : is the retrieved context relevant / well-ranked?
  - context_recall        : did retrieval surface what was needed?

Extend TEST_SET below with real Q&A pairs from your own documents
(the "ground_truth" field) as you gather them -- this file is meant to be a
living regression suite you re-run after changing chunking, retrieval
weights, or the rerank cutoff, not a one-off script.

Requires: pip install ragas datasets
Optionally set LANGCHAIN_API_KEY (see app.py) to also get full traces in
LangSmith for any run that goes through the agent.
"""
import os

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from retriever import setup_advanced_retriever
from tools import calculate_drip_irrigation  # noqa: F401 (kept for parity with app tools)

# --- Edit this with real questions + expected answers from your own docs ---
TEST_SET = [
    {
        "question": "What is the Kc value for tomatoes at mid-season?",
        "ground_truth": "The mid-season crop coefficient (Kc mid) for tomatoes is approximately 1.15 per FAO-56.",
    },
    {
        "question": "What emitter spacing is recommended for sandy soil?",
        "ground_truth": "Sandy soils have low lateral water movement, so closer emitter spacing (e.g. 20-30 cm) is generally recommended compared to clay soils.",
    },
    {
        "question": "Why would soil moisture read 100% but plants still wilt?",
        "ground_truth": "This can indicate a faulty/miscalibrated sensor, waterlogging causing root anoxia, or a disease/root problem preventing water uptake despite adequate soil moisture.",
    },
]


def run_retrieval(retriever, question: str):
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    return contexts


def build_dataset():
    retriever = setup_advanced_retriever()
    rows = {"question": [], "contexts": [], "ground_truth": [], "answer": []}

    for item in TEST_SET:
        q = item["question"]
        contexts = run_retrieval(retriever, q)
        # NOTE: for a full end-to-end eval (including generation quality),
        # swap this line to call your agent_executor and capture its
        # final "output" instead of joining raw context.
        answer = " ".join(contexts[:1]) if contexts else ""

        rows["question"].append(q)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["ground_truth"])
        rows["answer"].append(answer)

    return Dataset.from_dict(rows)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY in your environment before running eval.py")
        return

    dataset = build_dataset()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print("\n=== AquaMind RAG Evaluation ===")
    print(result)
    df = result.to_pandas()
    df.to_csv("eval_results.csv", index=False)
    print("\nSaved per-question breakdown to eval_results.csv")


if __name__ == "__main__":
    main()
