AI Resume Screening Agent

A smart AI-powered system that automatically analyzes resumes, evaluates candidate-job fit, and generates detailed insights using Sentence Transformers + Gemini AI.
This project provides an HR-ready automated screening tool with a clean Streamlit frontend and a FastAPI backend.

Overview

The AI Resume Screening Agent takes a Job Description and multiple resumes (PDF/DOCX/TXT) and performs:
Resume text extraction
Embedding-based similarity scoring
Gemini-powered reasoning & scoring
Strengths & weaknesses generation
Final ranking of candidates
Clean UI with detailed results and CSV export
This makes resume screening faster, more consistent, and more reliable for HR teams.


Features :
1. Multi-format Resume Parser:

Supports:PDF/DOCX/TXT

2. Embedding-Based Matching:
   
Uses Sentence Transformers (MiniLM L6 v2) for semantic similarity.

3. Gemini AI Evaluation:
   
Gemini provides: Final Candidate Score (0–100)
Strengths (3 points)
Weaknesses (2 points)
HR-style reasoning
JSON output

4. Interactive Web UI (Streamlit):
   
Upload resumes
Paste job description
See candidate cards
Expand strengths/weaknesses
Resume preview section
Summary table
Download CSV report

5. Secure Architecture:

Backend API on FastAPI
Frontend on Streamlit
Environment variable–based API key

Limitations :

Depends on text extraction quality (badly formatted PDFs may give less text).
Gemini scoring may vary slightly depending on resume structure.
No database (files processed live).
Not a full ATS system (no tracking or multi-round evaluation).


Architecture Diagram :

                   ┌───────────────────────────────┐
                   │         Streamlit UI           │
                   │  - Upload resumes              │
                   │  - Paste job description       │
                   │  - Display results             │
                   └───────────────┬───────────────┘
                                   │  (HTTP POST /rank)
                                   ▼
                   ┌───────────────────────────────┐
                   │            FastAPI            │
                   │         /rank endpoint        │
                   └───────────────┬───────────────┘
                                   │
     ┌─────────────────────────────┴─────────────────────────────┐
     │                 Resume Processing Pipeline                  │
     │-------------------------------------------------------------│
     │ 1. Text Extraction (PDF / DOCX / TXT)                       │
     │ 2. Sentence Embeddings (MiniLM)                             │
     │ 3. Cosine Similarity                                        │
     │ 4. Gemini LLM Scoring (Score + Strengths + Weaknesses)      │
     │ 5. Final Score Aggregation (Similarity + LLM score)         │
     └─────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │         JSON Response          │
                   │  - similarity score            │
                   │  - final score                 │
                   │  - strengths / weaknesses      │
                   │  - ranking                     │
                   └───────────────────────────────┘
              
  Tech Stack :
  
Frontend:
       Streamlit,
       Pandas,
       Requests

Backend:
       FastAPI,
       Uvicorn,
       CORS Middleware

AI & NLP:
       Sentence Transformers (MiniLM),
       Google Gemini 1.5 Flash,
       Document Parsing,
       pdfplumber,
       python-docx,
       txt decoding

Other:
      NumPy,
      Virtualenv,
      HTTP multipart handling

Setup & Run Instructions:

Backend Setup:
  Navigate to backend folder: cd backend
  
  Create virtual environment: python -m venv venv
  
  Activate environment: venv\Scripts\activate
  
  Install dependencies: pip install -r requirements.txt
  
  Set Gemini API key: set GEMINI_API_KEY=your_key_here
  
  Run backend server: uvicorn app.main:app --reload --port 8000

Frontend Setup:
  Open another terminal
  
  Go to frontend folder: cd frontend
  
  Install requirements: pip install -r requirements.txt
  
  Run Streamlit UI: streamlit run streamlit_app.py
  
  Open browser: http://localhost:8501
