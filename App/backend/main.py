# backend/app/main.py

import os
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from rag_pipeline import build_index, answer_compliance_question


app = FastAPI(title="Technical Compliance Checker API")
@app.get("/")
def root():
    return {"status": "ok", "message": "Technical Compliance Checker API is running"}
# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Set {name} in your deployment environment."
        )
    return value


def pdf_to_text(file: UploadFile) -> str:
    content = file.file.read()
    reader = PdfReader(BytesIO(content))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    spec: UploadFile = File(...),
    submittal: UploadFile = File(...),
    question: str = Form(
        "Does the contractor HVAC submittal comply with the project specification? "
        "Summarize key alignments and mismatches."
    ),
):
    # Ensure API key exists (will raise early if not)
    _ = get_env_var("OPENAI_API_KEY")

    spec_text = pdf_to_text(spec)
    submittal_text = pdf_to_text(submittal)

    corpus, embeddings = build_index(spec_text, submittal_text)
    answer, chunks = answer_compliance_question(corpus, embeddings, question)

    return {
        "answer": answer,
        "chunks": [
            {"source": ch["source"], "text": ch["text"][:400]}
            for ch in chunks
        ],
    }
