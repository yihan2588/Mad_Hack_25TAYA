from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from pathlib import Path

from utils_midi import (
    load_transcription_model,
    transcribe_audio_files,
    compare_performances,
)

app = FastAPI(title="Violin Backend")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# HEALTH CHECK
# ------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------
# 1) API: Compare two MIDI files
#    route: POST /api/compare-midi
# ---------------------------------------------------
@app.post("/api/compare-midi")
async def compare_midi(referenceFile: UploadFile = File(...), studentFile: UploadFile = File(...)):

    try:
        # Save temp files
        ref_tmp = Path(tempfile.mktemp(suffix=".mid"))
        stu_tmp = Path(tempfile.mktemp(suffix=".mid"))

        with ref_tmp.open("wb") as f:
            shutil.copyfileobj(referenceFile.file, f)

        with stu_tmp.open("wb") as f:
            shutil.copyfileobj(studentFile.file, f)

        # Run comparison
        result = compare_performances(ref_tmp, stu_tmp)

        return result.to_json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# 2) API: Transcribe audio (m4a → wav → midi)
#    route: POST /api/transcribe
# ---------------------------------------------------
@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Save incoming audio
        audio_tmp = Path(tempfile.mktemp(suffix=".m4a"))
        with audio_tmp.open("wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Load MUSC model (slow the first time)
        model = load_transcription_model("violin")

        # Transcribe into MIDI
        midi_paths = transcribe_audio_files([audio_tmp], model)

        with open(midi_paths[0], "rb") as f:
            midi_data = f.read()

        return {
            "midi_base64": midi_data.hex()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# 3) API: AI Feedback
#    route: POST /api/llm-feedback
# ---------------------------------------------------
from ai_feedback import generate_feedback  # your Gemini/OpenAI wrapper

@app.post("/api/llm-feedback")
async def llm_feedback(payload: dict):
    try:
        diff_json = payload.get("diffJson")
        result = generate_feedback(diff_json)
        return {"feedback": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
