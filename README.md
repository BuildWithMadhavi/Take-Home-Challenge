# Clinical Reasoning Platform

## 1. Project Overview
This system provides NG12 Cancer Risk Assessor and Conversational NG12 Chat modes using Vertex AI, a local persistent vector database, and strict citation enforcement. The system is PHI-aware and does not provide diagnoses.

## 2. Architecture Diagram
Ingestion flow:
- NG12 PDF -> chunking -> Vertex AI embeddings -> ChromaDB (persistent)

RAG flow:
- User input -> embed -> retrieve chunks -> Gemini reasoning -> JSON output with citations

Decision flow:
- If no NG12 evidence -> Insufficient Evidence
- Otherwise -> assessment grounded in citations

## 3. Functional Requirements
- Assessment logic uses only retrieved NG12 PDF text
- Chat mode supports multi-turn queries
- Every clinical statement must include citations

## 4. Non-Functional Requirements
- Security: env-based config, input validation, no PHI in logs, read-only vector DB
- Compliance: deterministic outputs, citation traceability, explicit disclaimer
- Reliability: graceful failure on empty retrieval, vector DB failure, Gemini timeout
- Availability: stateless API for horizontal scaling
- Observability: structured logs with correlation IDs and counts

## 5. Setup Instructions
- Create a Python environment
- Set Vertex AI credentials via environment variables
- Place NG12 PDF at `data/ng12.pdf`
- Run ingestion (see below)

## 6. Running the System
- Run ingestion:
  - `python ingestion/ingest_ng12.py --project YOUR_PROJECT_ID`
- Start API locally:
  - `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Run via Docker:
  - `docker build -t clinical-platform .`
  - `docker run -p 8000:8000 -e CDS_PROJECT_ID=YOUR_PROJECT_ID clinical-platform`
- Access frontend:
  - open `frontend/index.html`

## 7. Testing Guide
- Run unit tests: `pytest tests/unit`
- Run integration tests: `pytest tests/integration`
- Run black-box tests: `pytest tests/blackbox`
- Interpret results by checking status codes and citation presence

## 8. Failure Modes & Safeguards
- Hallucinations prevented by NG12-only prompt and citation enforcement
- Insufficient evidence returns explicit refusal
- Known limitations: depends on NG12 PDF content quality

## 9. Deliverables Mapping
- Part 1: Ingestion + RAG + FastAPI endpoints
- Part 2: Tests and compliance guardrails
- Evaluation: security, reliability, observability, and traceability covered in code and tests

## Disclaimer
This system is for educational and demonstration purposes only.
It does not provide medical diagnoses or treatment recommendations and should not be used in clinical practice.
