"""FastAPI server exposing the MIDI comparison backend to the frontend."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from yata_team import (
    ComparisonConfig,
    TranscriptionConfig,
    align_midi_first_note,
    compare_performances,
    load_transcription_model,
    transcribe_audio_files,
)

app = FastAPI(title="Violon Comparison Backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


TRANSCRIPTION_MODEL = None
TRANSCRIPTION_CONFIG = TranscriptionConfig()


def _ensure_transcription_model():
    global TRANSCRIPTION_MODEL
    if TRANSCRIPTION_MODEL is None:
        TRANSCRIPTION_MODEL = load_transcription_model()
    return TRANSCRIPTION_MODEL


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    """Persist an ``UploadFile`` to disk."""

    with destination.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return destination


def _is_midi_file(path: Path) -> bool:
    return path.suffix.lower() in {".mid", ".midi"}


def _audio_or_midi_to_aligned_midi(upload: UploadFile, tmpdir: Path) -> Path:
    saved_path = _save_upload(upload, tmpdir / upload.filename)
    if _is_midi_file(saved_path):
        align_midi_first_note(saved_path)
        return saved_path

    model = _ensure_transcription_model()
    midi_paths = transcribe_audio_files(
        [saved_path],
        model,
        config=TRANSCRIPTION_CONFIG,
        output_dir=tmpdir,
    )
    midi_path = midi_paths[0]
    align_midi_first_note(midi_path)
    return midi_path


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/compare")
async def compare_endpoint(
    reference_audio: UploadFile = File(...),
    student_audio: UploadFile = File(...),
) -> dict:
    """Compare two uploaded performances (audio or MIDI) and return analysis."""

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            reference_path = _audio_or_midi_to_aligned_midi(reference_audio, tmpdir_path)
            student_path = _audio_or_midi_to_aligned_midi(student_audio, tmpdir_path)

            result = compare_performances(reference_path, student_path, config=ComparisonConfig())
            return json.loads(result.to_json())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface unexpected backend issues
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
