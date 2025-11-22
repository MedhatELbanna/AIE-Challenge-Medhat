# src/main.py

import os
from pathlib import Path

from dotenv import load_dotenv

from .pdf_loader import load_pdf_text
from .rag_pipeline import build_index, answer_compliance_question

load_dotenv()


def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Create a .env file with {name}=... in the project root."
        )
    return value


def main() -> None:
    print("=== AIE Challenge – Mini HVAC Compliance Checker (Custom RAG) ===\n")

    # Ensure API key is set
    _ = get_env_var("OPENAI_API_KEY")

    # Ask for PDF paths
    spec_path = input("Enter path to SPEC PDF (e.g., data/spec.pdf): ").strip()
    submittal_path = input("Enter path to SUBMITTAL PDF (e.g., data/submittal.pdf): ").strip()

    spec_path = Path(spec_path)
    submittal_path = Path(submittal_path)

    if not spec_path.exists():
        print(f"[ERROR] Spec file not found: {spec_path}")
        return

    if not submittal_path.exists():
        print(f"[ERROR] Submittal file not found: {submittal_path}")
        return

    print("\nLoading PDFs...")
    spec_text = load_pdf_text(str(spec_path))
    submittal_text = load_pdf_text(str(submittal_path))

    print(f"- Spec length: {len(spec_text)} characters")
    print(f"- Submittal length: {len(submittal_text)} characters")

    print("\nBuilding index (creating embeddings)...")
    corpus, embeddings = build_index(spec_text, submittal_text)
    print(f"Index ready with {len(corpus)} chunks.\n")

    default_question = (
        "Does the contractor HVAC submittal comply with the project specification? "
        "Summarize key alignments and mismatches."
    )
    print(f"Default question:\n{default_question}\n")
    custom = input("Press ENTER to use default, or type a custom question: ").strip()
    question = custom if custom else default_question

    print("\nQuerying model...")
    answer, chunks = answer_compliance_question(corpus, embeddings, question)

    print("\n=== AI Compliance Assessment ===\n")
    print(answer)

    print("\n=== Source Chunks Used ===")
    for i, ch in enumerate(chunks, start=1):
        snippet = ch["text"][:200].replace("\n", " ")
        print(f"\n[{i}] Source: {ch['source']}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
