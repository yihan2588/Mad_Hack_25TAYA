"""FastAPI server exposing the MIDI comparison backend to the frontend."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from yata_team import ComparisonConfig, align_midi_first_note, compare_performances

app = FastAPI(title="Violon Comparison Backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    """Persist an ``UploadFile`` to disk."""

    with destination.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return destination


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/compare")
async def compare_endpoint(
    reference_midi: UploadFile = File(...),
    student_midi: UploadFile = File(...),
) -> dict:
    """Compare two uploaded MIDI files and return the analysis JSON."""

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            reference_path = _save_upload(reference_midi, tmpdir_path / "reference.mid")
            student_path = _save_upload(student_midi, tmpdir_path / "student.mid")

            align_midi_first_note(reference_path)
            align_midi_first_note(student_path)

            result = compare_performances(reference_path, student_path, config=ComparisonConfig())
            return json.loads(result.to_json())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface unexpected backend issues
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
