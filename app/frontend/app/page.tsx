"use client";

import React, { useState } from "react";

type Chunk = {
  source: string;
  text: string;
};

export default function Home() {
  const [specFile, setSpecFile] = useState<File | null>(null);
  const [submittalFile, setSubmittalFile] = useState<File | null>(null);
  const [question, setQuestion] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [error, setError] = useState<string | null>(null);

  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setAnswer(null);
    setChunks([]);

    if (!specFile || !submittalFile) {
      setError("Please upload both Spec and Submittal PDFs.");
      return;
    }

    const formData = new FormData();
    formData.append("spec", specFile);
    formData.append("submittal", submittalFile);

    if (question.trim()) {
      formData.append("question", question.trim());
    }

    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Request failed");
      }

      const data = await res.json();
      setAnswer(data.answer);
      setChunks(data.chunks || []);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-900 text-slate-50 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl bg-slate-800 rounded-2xl shadow-lg p-6 space-y-6">
        <h1 className="text-2xl font-semibold text-center">
          Technical Compliance Checker
        </h1>
        <p className="text-sm text-slate-300 text-center">
          Upload a project specification and contractor submittal (PDFs), and
          the app will analyze technical compliance using an AI RAG pipeline.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Spec PDF</label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setSpecFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-slate-200"
            />
          </div>

          <div>
            <label className="block text-sm font-font-medium mb-1">
              Submittal PDF
            </label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setSubmittalFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-slate-200"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Question (optional)
            </label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Default: Does the contractor HVAC submittal comply with the project specification?"
              className="w-full rounded-md bg-slate-900 border border-slate-700 p-2 text-sm"
              rows={3}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 rounded-md bg-indigo-500 hover:bg-indigo-600 disabled:bg-indigo-900 text-sm font-semibold"
          >
            {loading ? "Analyzing..." : "Analyze Compliance"}
          </button>
        </form>

        {error && (
          <div className="text-red-400 text-sm bg-red-950/40 p-3 rounded-md">
            {error}
          </div>
        )}

        {answer && (
          <div className="space-y-3">
            <h2 className="text-lg font-semibold">AI Compliance Assessment</h2>
            <div className="bg-slate-900 rounded-md p-3 text-sm whitespace-pre-wrap">
              {answer}
            </div>

            <h3 className="text-sm font-semibold mt-2">Source Chunks Used</h3>
            <div className="space-y-2 max-h-64 overflow-auto">
              {chunks.map((ch, idx) => (
                <div
                  key={idx}
                  className="bg-slate-900/70 rounded-md p-2 text-xs"
                >
                  <div className="text-[0.7rem] text-slate-400 mb-1">
                    [{idx + 1}] Source: {ch.source}
                  </div>
                  <div>{ch.text}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
