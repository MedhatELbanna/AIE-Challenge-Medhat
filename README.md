# 🏗️ Technical Compliance Checker  
### AI-Powered Engineering Submittal vs Specification Analyzer

This project is a full-stack AI application that analyzes **contractor submittals vs project specifications** using a custom **RAG (Retrieval-Augmented Generation)** pipeline.

It was built to satisfy the **AI Makerspace AIE Challenge** requirement of deploying a real LLM application accessible through a public URL.

---

## 🚀 Features

### ✔ Upload PDF Specifications  
### ✔ Upload PDF Contractor Submittals  
### ✔ Ask Optional Custom Questions  
### ✔ RAG Vector Search Using PDF Chunks  
### ✔ AI Compliance Reasoning  
### ✔ Traceability (Source Chunks Shown to User)  
### ✔ Modern Full-Stack App (Next.js + FastAPI)  
### ✔ Deployable to Railway + Vercel  

---

## 🧠 Tech Stack

### **Frontend**
- Next.js 14 (App Router)
- React
- Tailwind CSS
- File uploads using FormData
- Environment-based API routing

### **Backend**
- FastAPI
- OpenAI embeddings + reasoning
- PDF text extraction with PyPDF
- Recursive chunking
- FAISS vectorstore
- CORS enabled

### **Infrastructure**
- Vercel (Frontend)
- Railway (Backend)
- GitHub (Version Control)

---

## 📁 Folder Structure

AIE-Challenge-Medhat/
│
├── app/
│ ├── backend/
│ │ ├── main.py
│ │ ├── rag_pipeline.py
│ │ ├── requirements.txt
│ │ ├── Procfile
│ │ └── .env (local only)
│ │
│ └── frontend/
│ ├── app/page.tsx
│ ├── package.json
│ ├── .env.local (local only)
│ ├── styles/globals.css
│ └── ...
│
└── README.md