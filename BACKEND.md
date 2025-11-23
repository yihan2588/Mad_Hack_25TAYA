# Backend MIDI Comparison Module

The `yata_team.py` module now exposes importable functions so the frontend (or a
future service layer) can drive audio→MIDI transcription and comparison without
depending on the Colab notebook.

## Quick start

```python
from pathlib import Path
from yata_team import ComparisonConfig, compare_performances

result = compare_performances(
    good_midi_path=Path("/path/to/reference.mid"),
    bad_midi_path=Path("/path/to/student.mid"),
    config=ComparisonConfig(),  # optional overrides
)

# Send the JSON to the frontend
payload = result.to_json()
print(payload)
```

`ComparisonResult` contains:
- the normalized reference notes
- the warped student notes
- match statistics (significant errors vs. ok matches)
- lists of missing and extra notes
- a student-centric error report you can display directly in the UI

## Optional audio transcription

If the frontend provides raw audio files, load the MUSC violin model once and
re-use it to convert each file into MIDI before calling `compare_performances`.

```python
from yata_team import load_transcription_model, transcribe_audio_files

model = load_transcription_model()
audio_files = ["/tmp/good.m4a", "/tmp/bad.m4a"]
midi_files = transcribe_audio_files(audio_files, model)
```

Additional helpers are available for aligning MIDI start times
(`align_midi_first_note`) and for converting arbitrary audio into WAV
(`convert_to_wav`).

## Configuration knobs

`ComparisonConfig` exposes the same tolerances that were tuned in the notebook:
- note merge gap and minimum duration to reduce transcription noise
- greedy match tolerances for what counts as the same note
- thresholds that define when a timing or pitch difference is “significant”

These values can be tweaked per-piece (for example, to flag smaller errors on
fast passages or to loosen duration matching).

## Running the backend API

The React frontend expects a REST endpoint at `/api/compare` that accepts two
uploads (either audio `.m4a/.wav` or `.mid` files). The backend converts audio
to WAV, runs the MUSC transcription model, and then calls
`compare_performances`.

```bash
# one-time setup (installs venv, ffmpeg check, MUSC repo, deps)
./scripts/setup_backend.sh

# start the API (uses the venv automatically)
./scripts/run_backend.sh
```

During development the React app proxies requests to
`http://localhost:8000`, so the frontend will automatically talk to the local
FastAPI process once it is running.

### Transcription requirements

- `scripts/setup_backend.sh` checks for `ffmpeg` on PATH; install it (e.g., with
  Homebrew on macOS) before running the script.
- The script clones the MUSC violin transcription repo
  (https://github.com/MTG/violin-transcription), installs its Python requirements,
  and writes a `.pth` entry so `import musc` resolves to that checkout.
- PyTorch + (optionally) CUDA drivers are required for the MUSC model. On CPU
  the transcription step will run but may take significantly longer.
