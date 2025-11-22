# backend/app/rag_pipeline.py

from typing import List, Tuple, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()   
client = OpenAI()


def _split_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - overlap, 1)

    return chunks


def build_corpus(spec_text: str, submittal_text: str) -> List[Dict]:
    spec_chunks = _split_text(spec_text)
    submittal_chunks = _split_text(submittal_text)

    corpus: List[Dict] = []
    for ch in spec_chunks:
        corpus.append({"text": ch, "source": "spec"})
    for ch in submittal_chunks:
        corpus.append({"text": ch, "source": "submittal"})
    return corpus


def _embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    resp = client.embeddings.create(model=model, input=texts)
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.stack(vectors, axis=0)


def build_index(spec_text: str, submittal_text: str) -> Tuple[List[Dict], np.ndarray]:
    corpus = build_corpus(spec_text, submittal_text)
    texts = [c["text"] for c in corpus]
    embeddings = _embed_texts(texts)
    return corpus, embeddings


def _cosine_sim_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    if doc_matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    d_norm = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8)
    sims = d_norm @ q_norm
    return sims


def answer_compliance_question(
    corpus: List[Dict],
    embeddings: np.ndarray,
    question: str,
    top_k: int = 8,
    model: str = "gpt-4o-mini",
) -> Tuple[str, List[Dict]]:
    q_vec = _embed_texts([question])[0]
    sims = _cosine_sim_matrix(q_vec, embeddings)
    top_idx = np.argsort(sims)[::-1][:top_k]
    selected_chunks = [corpus[i] for i in top_idx]

    context_parts: List[str] = []
    for i, ch in enumerate(selected_chunks, start=1):
        context_parts.append(f"[{i}] source={ch['source']}\n{ch['text']}\n")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are an engineering document compliance assistant. "
        "You are given chunks from 'spec' and 'submittal' documents. "
        "Using ONLY the provided context, assess compliance."
    )

    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Your answer must be structured as:\n"
        "1. Overall verdict: (Compliant / Partially Compliant / Non-compliant)\n"
        "2. Key matches (bullet points)\n"
        "3. Key mismatches or missing points\n"
        "4. Assumptions / limitations.\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    answer = resp.choices[0].message.content
    return answer, selected_chunks
