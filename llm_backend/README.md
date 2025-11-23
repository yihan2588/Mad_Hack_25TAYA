# Violin Backend (FastAPI)

This backend powers the violin comparison + AI feedback system.

## Running Locally

```bash
cd llm_backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
