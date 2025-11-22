# Technical Compliance Checker

An AI-powered **Technical Compliance Checker** that compares an engineering **Project Specification** against a **Contractor Submittal** and provides a structured compliance assessment.

This project is built as part of the **AI Makerspace – AI Engineering Bootcamp (AIE) Challenge**.

---

## 🔍 What it does

Given two PDFs:

- `spec.pdf` – the official project specification  
- `submittal.pdf` – the contractor’s technical submittal  

The app:

1. Extracts text from both documents  
2. Splits them into semantic chunks labeled as `spec` or `submittal`  
3. Creates embeddings for all chunks using OpenAI  
4. Retrieves the most relevant chunks for a compliance question  
5. Asks an LLM to produce a **clear, structured compliance assessment**, including:
   - Overall verdict (Compliant / Partially Compliant / Non-compliant)  
   - Key matches between spec and submittal  
   - Key mismatches or missing points  
   - Assumptions / limitations  

Domain example used here: **HVAC system specifications vs HVAC equipment submittal.**

---

## 🏗 Tech Stack

- **Language:** Python  
- **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)  
- **Vector math:** NumPy  
- **PDF parsing:** `pypdf`  
- **Config:** `.env` + `python-dotenv`

---

## 📁 Project structure

```text
AIE-Challenge-Medhat/
│
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI entry point
│   ├── pdf_loader.py         # PDF → text
│   └── rag_pipeline.py       # Simple custom RAG (embeddings + retrieval + LLM)
│
├── data/
│   ├── spec.pdf              # Project specification (example: HVAC spec)
│   └── submittal.pdf         # Contractor submittal (example: HVAC package)
│
├── requirements.txt
├── .env                      # Contains OPENAI_API_KEY (not committed)
├── .gitignore
└── README.md
