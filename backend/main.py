"""FastAPI server exposing the performance analysis pipeline."""
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pretty_midi
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Ensure the MUSC repository (if present) is importable before loading the model.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUSC_REPO = PROJECT_ROOT / "MUSC_violin"
if MUSC_REPO.exists() and str(MUSC_REPO) not in sys.path:
    sys.path.append(str(MUSC_REPO))

from yata_team import (  # noqa: E402
    ComparisonConfig,
    MatchEvaluation,
    NoteError,
    TranscriptionConfig,
    align_midi_first_note,
    compare_performances,
    convert_to_wav,
    load_transcription_model,
    safe_transcribe_local,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="YATA Performance Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transcription_model: Any | None = None
transcription_config = TranscriptionConfig()
comparison_config = ComparisonConfig()
analysis_lock = asyncio.Lock()


@app.on_event("startup")
def _load_transcriber() -> None:
    """Load the MUSC model once so audio uploads can be transcribed."""

    global transcription_model
    try:
        transcription_model = load_transcription_model()
        logger.info("Loaded MUSC model for instrument '%s'", "violin")
    except Exception as exc:  # pragma: no cover - depends on GPU setup
        transcription_model = None
        logger.warning("MUSC model unavailable: %s", exc)
        logger.warning("Audio uploads will be rejected until the model loads correctly.")


def transcribe_audio(audio_path: Path) -> pretty_midi.PrettyMIDI:
    """Transcribe *audio_path* into a PrettyMIDI object using the MUSC model."""

    if transcription_model is None:
        raise RuntimeError(
            "The transcription model is not loaded. Ensure MUSC weights are available or "
            "upload MIDI files instead."
        )

    wav_path = convert_to_wav(
        audio_path,
        sample_rate=transcription_config.target_sample_rate,
        mono=transcription_config.mono,
    )
    midi_obj = safe_transcribe_local(
        transcription_model,
        str(wav_path),
        batch_size=transcription_config.batch_size,
        postprocessing=transcription_config.postprocessing,
    )
    return midi_obj


def align_midi_start(midi_path: Path) -> Path:
    """Align the first significant note of ``midi_path`` to zero."""

    return align_midi_first_note(midi_path, min_note_duration=comparison_config.min_note_duration)


async def _save_upload(upload: UploadFile, directory: Path, prefix: str) -> Path:
    """Persist *upload* inside *directory* and return the stored path."""

    original = Path(upload.filename or prefix)
    suffix = original.suffix or ""
    dest = directory / f"{prefix}{suffix}"
    data = await upload.read()
    dest.write_bytes(data)
    return dest


def _ensure_midi_file(src_path: Path, role: str) -> Path:
    """Convert audio uploads to MIDI and align the first onset."""

    suffix = src_path.suffix.lower()
    if suffix in {".mid", ".midi"}:
        midi_path = src_path
    else:
        midi_obj = transcribe_audio(src_path)
        midi_path = src_path.with_suffix(".mid")
        midi_obj.write(str(midi_path))
        logger.info("Transcribed %s audio to MIDI at %s", role, midi_path)

    align_midi_start(midi_path)
    return midi_path


def _accuracy_from_evaluation(evaluation: MatchEvaluation) -> str:
    if evaluation.student_error_type == "ok":
        return "good"
    return "poor" if evaluation.is_significant else "moderate"


def _chart_data_from_result(result: ComparisonResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for evaluation in result.match_evaluations:
        ref_note = result.reference_notes[evaluation.reference_index]
        stud_note = result.student_notes_warped[evaluation.student_index]
        rows.append(
            {
                "time": ref_note.start,
                "master": ref_note.pitch,
                "user": stud_note.pitch,
                "accuracy": _accuracy_from_evaluation(evaluation),
                "status": evaluation.student_error_type,
                "pitchDiff": evaluation.pitch_diff_semitones,
                "onsetDiff": evaluation.onset_diff_sec,
                "offsetDiff": evaluation.offset_diff_sec,
            }
        )

    for idx in result.missing_reference_indices:
        note = result.reference_notes[idx]
        rows.append(
            {
                "time": note.start,
                "master": note.pitch,
                "user": None,
                "accuracy": "poor",
                "status": "missed_note",
            }
        )

    for idx in result.extra_student_indices:
        note = result.student_notes_warped[idx]
        rows.append(
            {
                "time": note.start,
                "master": None,
                "user": note.pitch,
                "accuracy": "poor",
                "status": "extra_note",
            }
        )

    rows.sort(key=lambda row: row["time"])
    return rows


def _stats_from_chart(chart_data: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(chart_data)
    counts = {"good": 0, "moderate": 0, "poor": 0}
    for row in chart_data:
        counts[row["accuracy"]] = counts.get(row["accuracy"], 0) + 1

    def pct(count: int) -> float:
        return (count / total * 100.0) if total else 0.0

    overall = int(round(pct(counts["good"])))
    return {
        "overallScore": overall,
        "goodPercent": round(pct(counts["good"]), 1),
        "moderatePercent": round(pct(counts["moderate"]), 1),
        "poorPercent": round(pct(counts["poor"]), 1),
        "totalNotes": total,
    }


def _serialize_error_report(report: Sequence[NoteError]) -> List[Dict[str, Any]]:
    return [error.to_dict() for error in report]


@app.post("/analyze")
async def analyze_performance(
    student_file: UploadFile = File(...),
    master_file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Analyze two uploaded performances and return chart-friendly data."""

    async with analysis_lock:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            student_path = await _save_upload(student_file, tmpdir, "student")
            master_path = await _save_upload(master_file, tmpdir, "master")

            try:
                student_midi = _ensure_midi_file(student_path, "student")
                master_midi = _ensure_midi_file(master_path, "master")
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            try:
                comparison = compare_performances(master_midi, student_midi, config=comparison_config)
            except Exception as exc:
                logger.exception("Comparison failed")
                raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc

            chart_data = _chart_data_from_result(comparison)
            stats = _stats_from_chart(chart_data)
            error_report = _serialize_error_report(comparison.error_report)

            return {
                "chartData": chart_data,
                "stats": stats,
                "errorReport": error_report,
            }


@app.get("/")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "modelLoaded": transcription_model is not None,
    }
