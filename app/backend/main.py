# backend/app/main.py

import os
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

from backend.rag_pipeline import build_index, answer_compliance_question

# Load environment variables
load_dotenv()

# 1) Create FastAPI app
app = FastAPI(title="Technical Compliance Checker API")

# 2) Serve static frontend
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# 3) Root serves the HTML UI
@app.get("/", response_class=HTMLResponse)
def index():
    with open("backend/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# 4) API health
@app.get("/health")
def health():
    return {"status": "ok"}

# 5) Debug env
@app.get("/debug_env")
def debug_env():
    return {"OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY"))}

# 6) CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7) Utility functions
def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set."
        )
    return value

def pdf_to_text(file: UploadFile) -> str:
    content = file.file.read()
    reader = PdfReader(BytesIO(content))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)

# 8) Main analyze endpoint
@app.post("/analyze")
async def analyze(
    spec: UploadFile = File(...),
    submittal: UploadFile = File(...),
    question: str = Form(
        "Does the contractor HVAC submittal comply with the project specification? "
        "Summarize key alignments and mismatches."
    ),
):
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
