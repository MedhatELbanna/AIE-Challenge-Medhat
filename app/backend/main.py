# backend/app/main.py

import os
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
# Serve static frontend
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("backend/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


from backend.rag_pipeline import build_index, answer_compliance_question

# Load environment variables from .env (useful locally; on Railway it uses real env vars)
load_dotenv()

# Create FastAPI app FIRST
app = FastAPI(title="Technical Compliance Checker API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: in production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Technical Compliance Checker API is running",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug_env")
def debug_env():
    return {
        "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
    }


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
