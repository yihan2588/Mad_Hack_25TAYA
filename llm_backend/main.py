from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

from utils_midi import run_midi_comparison
from ai_feedback import generate_ai_feedback

app = FastAPI()

# Enable CORS so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LLMRequest(BaseModel):
    diffJson: str


@app.get("/health")
def health():
    return {"status": "ok"}


# === MIDI Comparison Endpoint ===
@app.post("/api/compare-midi")
async def compare_midi(referenceFile: UploadFile = File(...),
                       studentFile: UploadFile = File(...)):

    # Save uploaded files to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, "reference.mid")
        stu_path = os.path.join(tmpdir, "student.mid")

        with open(ref_path, "wb") as f:
            f.write(await referenceFile.read())

        with open(stu_path, "wb") as f:
            f.write(await studentFile.read())

        # Compute comparison
        result = run_midi_comparison(ref_path, stu_path)

        return result


# === LLM Feedback API ===
@app.post("/api/llm-feedback")
async def llm_feedback(req: LLMRequest):
    feedback = generate_ai_feedback(req.diffJson)
    return {"feedback": feedback}

