import sys
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import pretty_midi
import numpy as np

# --- 1. Setup & Model Loading ---
# Add the cloned repo to path so imports work
sys.path.append("MUSC_violin")
try:
    from musc.model import PretrainedModel
except ImportError:
    PretrainedModel = None
    print("Warning: MUSC_violin repo not found in path. Ensure it is cloned.")

app = FastAPI()

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production limit this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Variable
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def load_model():
    global model
    if PretrainedModel is None:
        print("PretrainedModel unavailable; ensure repo is cloned before serving.")
        return
    print(f"Loading model on {device}...")
    try:
        model = PretrainedModel(instrument="violin").to(device)
        print("Model loaded successfully.")
    except Exception as e:  # pragma: no cover - depends on GPU availability
        print(f"Error loading model: {e}")


# --- 2. Helper Functions ---

def convert_to_wav(in_path: Path) -> Path:
    """Convert input audio to 44.1kHz mono wav using ffmpeg."""
    wav_path = in_path.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-ac",
        "1",
        "-ar",
        "44100",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path


def transcribe_audio(audio_path: Path) -> pretty_midi.PrettyMIDI:
    """Transcribe audio file to PrettyMIDI object."""
    if model is None:
        raise RuntimeError("Transcription model not loaded")

    wav_path = convert_to_wav(audio_path)

    if hasattr(model, "transcribe"):
        out = model.transcribe(str(wav_path), postprocessing="spotify")
    elif hasattr(model, "transcribe_file"):
        out = model.transcribe_file(str(wav_path), postprocessing="spotify")
    else:
        raise RuntimeError("Could not find transcribe method on model")

    if isinstance(out, pretty_midi.PrettyMIDI):
        return out
    if isinstance(out, dict) and "midi" in out:
        return out["midi"]
    return out[0]


def align_midi_start(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Shift MIDI so the first note starts at 0."""
    starts = []
    for inst in pm.instruments:
        for n in inst.notes:
            if (n.end - n.start) >= 0.03:
                starts.append(n.start)

    if not starts:
        return pm

    t0 = min(starts)

    for inst in pm.instruments:
        for n in inst.notes:
            n.start = max(0.0, n.start - t0)
            n.end = max(0.0, n.end - t0)
    return pm


# --- 3. DTW & Comparison Logic ---

def extract_notes(pm, min_dur=0.05):
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            if (n.end - n.start) >= min_dur:
                notes.append(
                    {
                        "start": float(n.start),
                        "end": float(n.end),
                        "pitch": int(n.pitch),
                        "velocity": int(n.velocity),
                    }
                )
    notes.sort(key=lambda x: x["start"])
    return notes


def merge_same_pitch_notes(notes, gap_tol=0.3):
    if not notes:
        return notes
    merged = [notes[0].copy()]
    for n in notes[1:]:
        last = merged[-1]
        gap = n["start"] - last["end"]
        if n["pitch"] == last["pitch"] and gap <= gap_tol:
            last["end"] = max(last["end"], n["end"])
        else:
            merged.append(n.copy())
    return merged


def dtw_path_simple(seqA, seqB):
    """Simple DTW on pitch sequences."""
    n, m = len(seqA), len(seqB)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seqA[i - 1] - seqB[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return path[::-1]


def compare_performances(pm_ref, pm_student):
    notes_ref = merge_same_pitch_notes(extract_notes(pm_ref))
    notes_stud = merge_same_pitch_notes(extract_notes(pm_student))

    if not notes_ref or not notes_stud:
        return {"error": "Not enough notes to compare"}

    pitch_ref = [n["pitch"] for n in notes_ref]
    pitch_stud = [n["pitch"] for n in notes_stud]

    path = dtw_path_simple(pitch_ref, pitch_stud)

    comparison_data = []

    ref_map = {i: [] for i in range(len(notes_ref))}
    for i, j in path:
        ref_map[i].append(j)

    for i in range(len(notes_ref)):
        ref_note = notes_ref[i]
        student_indices = ref_map[i]

        if not student_indices:
            comparison_data.append(
                {
                    "time": ref_note["start"],
                    "master": ref_note["pitch"],
                    "user": None,
                    "accuracy": "poor",
                    "status": "missed_note",
                }
            )
        else:
            j_best = int(np.median(student_indices))
            stud_note = notes_stud[j_best]

            diff = stud_note["pitch"] - ref_note["pitch"]
            abs_diff = abs(diff)

            status = "good"
            if abs_diff == 0:
                status = "good"
            elif abs_diff <= 2:
                status = "moderate"
            else:
                status = "poor"

            comparison_data.append(
                {
                    "time": ref_note["start"],
                    "master": ref_note["pitch"],
                    "user": stud_note["pitch"],
                    "accuracy": status,
                    "status": "match" if status == "good" else "wrong_pitch",
                }
            )

    return comparison_data


# --- 4. API Endpoints ---


@app.post("/analyze")
async def analyze_performance(
    student_file: UploadFile = File(...),
    master_file: UploadFile = File(...),
):
    with tempfile.TemporaryDirectory() as temp_dir:
        t_dir = Path(temp_dir)

        stud_path = t_dir / student_file.filename
        with stud_path.open("wb") as f:
            f.write(await student_file.read())

        if stud_path.suffix.lower() in [".mid", ".midi"]:
            pm_student = pretty_midi.PrettyMIDI(str(stud_path))
        else:
            pm_student = transcribe_audio(stud_path)

        pm_student = align_midi_start(pm_student)

        mast_path = t_dir / master_file.filename
        with mast_path.open("wb") as f:
            f.write(await master_file.read())

        if mast_path.suffix.lower() in [".mid", ".midi"]:
            pm_master = pretty_midi.PrettyMIDI(str(mast_path))
        else:
            pm_master = transcribe_audio(mast_path)

        pm_master = align_midi_start(pm_master)

        chart_data = compare_performances(pm_master, pm_student)

        if isinstance(chart_data, dict) and chart_data.get("error"):
            raise HTTPException(status_code=400, detail=chart_data["error"])

        total = len(chart_data)
        good = len([x for x in chart_data if x["accuracy"] == "good"])
        moderate = len([x for x in chart_data if x["accuracy"] == "moderate"])
        poor = len([x for x in chart_data if x["accuracy"] == "poor"])

        score = int((good / total) * 100) if total > 0 else 0

        stats = {
            "overallScore": score,
            "goodPercent": f"{(good / total) * 100:.1f}" if total else "0",
            "moderatePercent": f"{(moderate / total) * 100:.1f}" if total else "0",
            "poorPercent": f"{(poor / total) * 100:.1f}" if total else "0",
        }

        return {"stats": stats, "chartData": chart_data}


@app.get("/")
def health_check():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}
