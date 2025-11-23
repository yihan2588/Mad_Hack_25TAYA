"""Backend utilities for comparing MIDI performances.

This module refactors the original Colab notebook into reusable functions that:
- optionally transcribe audio files into MIDI using the MUSC violin model;
- align and normalize MIDI timelines; and
- compare a reference ("good") MIDI against a student ("bad") MIDI to highlight
  pitch and timing deviations.

The functions defined here are importable so that a separate frontend can call
into the MIDI comparison pipeline without depending on Colab or notebook UI.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pretty_midi

try:  # torch is optional if transcription is never used.
    import torch
except ImportError:  # pragma: no cover - torch might be missing in lightweight envs
    torch = None  # type: ignore


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------
@dataclass
class TranscriptionConfig:
    """Configuration used when converting raw audio into MIDI."""

    batch_size: int = 32
    postprocessing: str = "spotify"
    target_sample_rate: int = 44_100
    mono: bool = True


def load_transcription_model(
    instrument: str = "violin",
    device: Optional[str] = None,
):
    """Load the MUSC pretrained model.

    Parameters
    ----------
    instrument:
        Instrument passed to ``musc.model.PretrainedModel``.
    device:
        Desired PyTorch device. When ``None`` the function prefers CUDA if
        available and falls back to CPU.
    """

    if device is None:
        if torch is not None and torch.cuda.is_available():  # pragma: no branch
            device = "cuda"
        else:
            device = "cpu"

    try:
        from musc.model import PretrainedModel
    except ImportError as exc:  # pragma: no cover - depends on env setup
        raise RuntimeError(
            "The MUSC violin repository is not available. Ensure it is installed "
            "and importable before requesting transcription."
        ) from exc

    model = PretrainedModel(instrument=instrument).to(device)
    model.eval()
    return model


def convert_to_wav(
    in_path: PathLike,
    *,
    output_path: Optional[PathLike] = None,
    sample_rate: int = 44_100,
    mono: bool = True,
) -> Path:
    """Convert arbitrary audio into a WAV file that the transcription model expects."""

    in_path = Path(in_path)
    if output_path is None:
        output_path = in_path.with_suffix(".wav")
    out_path = Path(output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-ar",
        str(sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(str(out_path))

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def _normalize_to_pretty_midi(out: Any) -> pretty_midi.PrettyMIDI:
    """Normalize several output formats used by the MUSC codebase."""

    if isinstance(out, pretty_midi.PrettyMIDI):
        return out
    if isinstance(out, dict) and "midi" in out:
        return out["midi"]
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], pretty_midi.PrettyMIDI):
        return out[0]
    raise TypeError(f"Unexpected transcription return type: {type(out)}")


def safe_transcribe_local(
    model: Any,
    audio_path: PathLike,
    *,
    batch_size: int = 32,
    postprocessing: str = "spotify",
) -> pretty_midi.PrettyMIDI:
    """Call whichever transcription method the loaded MUSC model exposes."""

    candidates: List[Callable[..., Any]] = []

    if hasattr(model, "transcribe"):
        candidates.append(model.transcribe)
    if hasattr(model, "transcribe_file"):
        candidates.append(model.transcribe_file)
    if hasattr(model, "transcriber") and hasattr(model.transcriber, "transcribe"):
        candidates.append(model.transcriber.transcribe)  # type: ignore[attr-defined]

    tried: List[Tuple[str, str]] = []

    for fn in candidates:
        # Always pass ``audio_path`` positionally because some signatures are strict.
        patterns = [
            lambda: fn(audio_path, batch_size=batch_size, postprocessing=postprocessing),
            lambda: fn(audio_path, batch_size=batch_size),
            lambda: fn(audio_path, postprocessing=postprocessing),
            lambda: fn(audio_path),
        ]

        for call in patterns:
            try:
                return _normalize_to_pretty_midi(call())
            except TypeError as err:
                tried.append((fn.__qualname__, str(err)))
            except Exception:
                raise

    info = "\n".join(f"{name}: {err}" for name, err in tried)
    raise RuntimeError(f"Could not transcribe local file. Tried:\n{info}")


def transcribe_audio_files(
    audio_paths: Sequence[PathLike],
    model: Any,
    *,
    config: Optional[TranscriptionConfig] = None,
    output_dir: Optional[PathLike] = None,
) -> List[Path]:
    """Transcribe a sequence of audio files into MIDI files."""

    config = config or TranscriptionConfig()
    output_dir = Path(output_dir) if output_dir else None

    midi_paths: List[Path] = []
    for path in audio_paths:
        path = Path(path)
        wav_path = convert_to_wav(
            path,
            output_path=path.with_suffix(".wav"),
            sample_rate=config.target_sample_rate,
            mono=config.mono,
        )
        midi_obj = safe_transcribe_local(
            model,
            str(wav_path),
            batch_size=config.batch_size,
            postprocessing=config.postprocessing,
        )
        midi_dest = (output_dir / path.with_suffix(".mid").name) if output_dir else path.with_suffix(".mid")
        midi_obj.write(str(midi_dest))
        midi_paths.append(midi_dest)
    return midi_paths


# ---------------------------------------------------------------------------
# MIDI alignment helpers
# ---------------------------------------------------------------------------
@dataclass
class MidiNote:
    start: float
    end: float
    pitch: int
    velocity: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def centroid(self) -> float:
        return 0.5 * (self.start + self.end)

    def to_dict(self) -> Dict[str, float]:
        return {
            "start": self.start,
            "end": self.end,
            "pitch": self.pitch,
            "velocity": self.velocity,
            "duration": self.duration,
            "centroid": self.centroid,
        }


@dataclass
class ComparisonConfig:
    merge_gap: float = 0.3
    min_note_duration: float = 0.3
    match_time_tolerance: float = 0.3
    match_pitch_tolerance: int = 1
    significant_onset_tolerance: float = 0.5
    significant_pitch_tolerance: int = 5
    significant_offset_tolerance: float = 0.5


@dataclass
class MatchEvaluation:
    reference_index: int
    student_index: int
    onset_diff_sec: float
    pitch_diff_semitones: float
    offset_diff_sec: float
    student_error_type: str
    is_significant: bool

    def to_dict(self) -> Dict[str, Union[int, float, str, bool]]:
        return dataclasses.asdict(self)


@dataclass
class NoteError:
    student_error_type: str
    time_sec: float
    gt_pitch: Optional[int]
    student_pitch: Optional[int]
    pitch_diff_semitones: Optional[float] = None
    onset_diff_sec: Optional[float] = None
    offset_diff_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[Union[str, float, int]]]:
        return dataclasses.asdict(self)


@dataclass
class ComparisonResult:
    reference_notes: List[MidiNote]
    student_notes_warped: List[MidiNote]
    match_evaluations: List[MatchEvaluation]
    missing_reference_indices: List[int]
    extra_student_indices: List[int]
    significant_matches: List[MatchEvaluation] = field(default_factory=list)
    ok_matches: List[MatchEvaluation] = field(default_factory=list)
    error_report: List[NoteError] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize the result so the frontend can consume the data easily."""

        payload = {
            "reference_notes": [n.to_dict() for n in self.reference_notes],
            "student_notes_warped": [n.to_dict() for n in self.student_notes_warped],
            "match_evaluations": [m.to_dict() for m in self.match_evaluations],
            "missing_reference_indices": self.missing_reference_indices,
            "extra_student_indices": self.extra_student_indices,
            "significant_matches": [m.to_dict() for m in self.significant_matches],
            "ok_matches": [m.to_dict() for m in self.ok_matches],
            "error_report": [e.to_dict() for e in self.error_report],
        }
        return json.dumps(payload, indent=2)


def first_note_time(pm: pretty_midi.PrettyMIDI, min_dur: float = 0.03) -> float:
    """Return the onset of the first note that is longer than ``min_dur`` seconds."""

    starts: List[float] = []
    for inst in pm.instruments:
        for note in inst.notes:
            if (note.end - note.start) >= min_dur:
                starts.append(note.start)
    if not starts:
        for inst in pm.instruments:
            for note in inst.notes:
                starts.append(note.start)
    if not starts:
        raise ValueError("The MIDI file contains no notes")
    return float(min(starts))


def shift_pretty_midi(pm: pretty_midi.PrettyMIDI, shift: float) -> pretty_midi.PrettyMIDI:
    """Shift every time-based event in *pm* backwards by ``shift`` seconds."""

    for inst in pm.instruments:
        for note in inst.notes:
            note.start = max(0.0, note.start - shift)
            note.end = max(0.0, note.end - shift)
        for pb in inst.pitch_bends:
            pb.time = max(0.0, pb.time - shift)
        for cc in inst.control_changes:
            cc.time = max(0.0, cc.time - shift)
    return pm


def align_midi_first_note(
    midi_path: PathLike,
    *,
    min_note_duration: float = 0.03,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Align the first note of *midi_path* to time zero and return the saved path."""

    midi_path = Path(midi_path)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    offset = first_note_time(pm, min_note_duration=min_note_duration)
    shifted = shift_pretty_midi(pm, offset)

    dest = Path(output_path) if output_path else midi_path
    shifted.write(str(dest))
    return dest


# ---------------------------------------------------------------------------
# MIDI comparison core
# ---------------------------------------------------------------------------

def extract_notes(
    pm: pretty_midi.PrettyMIDI,
    *,
    min_note_duration: float,
) -> List[MidiNote]:
    """Extract notes that are longer than ``min_note_duration`` seconds."""

    notes: List[MidiNote] = []
    for inst in pm.instruments:
        for note in inst.notes:
            if (note.end - note.start) >= min_note_duration:
                notes.append(MidiNote(note.start, note.end, note.pitch, note.velocity))
    notes.sort(key=lambda n: n.start)
    return notes


def merge_same_pitch_notes(notes: Sequence[MidiNote], *, gap_tol: float) -> List[MidiNote]:
    """Merge consecutive notes with the same pitch if the gap between them is tiny."""

    if not notes:
        return []

    merged = [dataclasses.replace(notes[0])]
    for note in notes[1:]:
        last = merged[-1]
        gap = note.start - last.end
        if note.pitch == last.pitch and gap <= gap_tol:
            last.end = max(last.end, note.end)
            last.velocity = max(last.velocity, note.velocity)
        else:
            merged.append(dataclasses.replace(note))
    return merged


def dtw_path(seq_a: Sequence[int], seq_b: Sequence[int]) -> List[Tuple[int, int]]:
    """Simple dynamic time warping based on absolute pitch difference."""

    n, m = len(seq_a), len(seq_b)
    dist = np.full((n + 1, m + 1), np.inf)
    dist[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq_a[i - 1] - seq_b[j - 1])
            dist[i, j] = cost + min(dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1])

    i, j = n, m
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        options = [dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1]]
        step = int(np.argmin(options))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return path[::-1]


def build_time_warp(notes_a: Sequence[MidiNote], notes_b: Sequence[MidiNote]) -> Callable[[float], float]:
    """Create a monotonic warp that maps student times into the reference space."""

    pitch_a = [note.pitch for note in notes_a]
    pitch_b = [note.pitch for note in notes_b]
    path = dtw_path(pitch_a, pitch_b)

    mapping: Dict[int, List[int]] = {}
    for i, j in path:
        mapping.setdefault(i, []).append(j)

    anchors: List[Tuple[float, float]] = []
    last_tb = -np.inf
    for i in sorted(mapping.keys(), key=lambda idx: notes_a[idx].start):
        js = mapping[i]
        j_med = int(np.median(js))
        ta = notes_a[i].start
        tb = notes_b[j_med].start
        if tb > last_tb:
            anchors.append((ta, tb))
            last_tb = tb

    if len(anchors) < 2:
        raise RuntimeError("DTW warp failed: need at least two anchor points")

    t_a = np.array([a for a, _ in anchors])
    t_b = np.array([b for _, b in anchors])

    def warp_time(tb: float) -> float:
        return float(np.interp(tb, t_b, t_a))

    return warp_time


def warp_notes(notes: Sequence[MidiNote], warp_fn: Callable[[float], float]) -> List[MidiNote]:
    """Apply ``warp_fn`` to every note's start/end time."""

    warped = [
        MidiNote(start=warp_fn(note.start), end=warp_fn(note.end), pitch=note.pitch, velocity=note.velocity)
        for note in notes
    ]
    warped.sort(key=lambda note: note.start)
    return warped


def greedy_match(
    notes_ref: Sequence[MidiNote],
    notes_student: Sequence[MidiNote],
    *,
    time_tol: float,
    pitch_tol: int,
) -> Tuple[List[Tuple[int, int, float, float]], List[int], List[int]]:
    """Greedy matching that respects chronological order (works for monophonic parts)."""

    used = np.zeros(len(notes_student), dtype=bool)
    matches: List[Tuple[int, int, float, float]] = []
    missing: List[int] = []

    for i, note_ref in enumerate(notes_ref):
        candidates: List[Tuple[float, int, float, float]] = []
        for j, note_student in enumerate(notes_student):
            if used[j]:
                continue
            dt = abs(note_student.start - note_ref.start)
            dp = abs(note_student.pitch - note_ref.pitch)
            if dt <= time_tol and dp <= pitch_tol:
                candidates.append((dt + 0.05 * dp, j, dt, dp))
        if not candidates:
            missing.append(i)
            continue
        candidates.sort(key=lambda item: item[0])
        _, j_best, dt_best, dp_best = candidates[0]
        used[j_best] = True
        matches.append((i, j_best, dt_best, dp_best))

    extra = [idx for idx, flag in enumerate(used) if not flag]
    return matches, missing, extra


def evaluate_matches(
    matches: Sequence[Tuple[int, int, float, float]],
    notes_ref: Sequence[MidiNote],
    notes_student: Sequence[MidiNote],
    config: ComparisonConfig,
) -> List[MatchEvaluation]:
    """Label matches as significant or OK based on pitch/timing thresholds."""

    evaluations: List[MatchEvaluation] = []
    for i, j, dt_onset, dp in matches:
        ref_note = notes_ref[i]
        stud_note = notes_student[j]
        dt_offset = abs(stud_note.end - ref_note.end)

        is_sig = (
            dp >= config.significant_pitch_tolerance
            or dt_onset >= config.significant_onset_tolerance
            or dt_offset >= config.significant_offset_tolerance
        )
        err_type = (
            "wrong_pitch"
            if dp >= config.significant_pitch_tolerance
            else "wrong_timing"
            if dt_onset >= config.significant_onset_tolerance
            else "wrong_duration"
            if dt_offset >= config.significant_offset_tolerance
            else "ok"
        )
        evaluations.append(
            MatchEvaluation(
                reference_index=i,
                student_index=j,
                onset_diff_sec=dt_onset,
                pitch_diff_semitones=dp,
                offset_diff_sec=dt_offset,
                student_error_type=err_type,
                is_significant=is_sig,
            )
        )
    return evaluations


def build_error_report(
    evaluations: Sequence[MatchEvaluation],
    missing: Sequence[int],
    extra: Sequence[int],
    notes_ref: Sequence[MidiNote],
    notes_student: Sequence[MidiNote],
) -> List[NoteError]:
    """Create a student-centric error list combining all error categories."""

    rows: List[NoteError] = []
    for idx in missing:
        note = notes_ref[idx]
        rows.append(NoteError("missed_note", time_sec=note.start, gt_pitch=note.pitch, student_pitch=None))
    for idx in extra:
        note = notes_student[idx]
        rows.append(NoteError("extra_note", time_sec=note.start, gt_pitch=None, student_pitch=note.pitch))
    for evaluation in evaluations:
        if evaluation.student_error_type == "ok":
            continue
        ref_note = notes_ref[evaluation.reference_index]
        stud_note = notes_student[evaluation.student_index]
        rows.append(
            NoteError(
                student_error_type=evaluation.student_error_type,
                time_sec=ref_note.start,
                gt_pitch=ref_note.pitch,
                student_pitch=stud_note.pitch,
                pitch_diff_semitones=evaluation.pitch_diff_semitones,
                onset_diff_sec=evaluation.onset_diff_sec,
                offset_diff_sec=evaluation.offset_diff_sec,
            )
        )
    rows.sort(key=lambda row: row.time_sec)
    return rows


def compare_performances(
    good_midi_path: PathLike,
    bad_midi_path: PathLike,
    *,
    config: Optional[ComparisonConfig] = None,
) -> ComparisonResult:
    """Compare two MIDI files and return structured deviation data."""

    config = config or ComparisonConfig()

    pm_ref = pretty_midi.PrettyMIDI(str(good_midi_path))
    pm_student = pretty_midi.PrettyMIDI(str(bad_midi_path))

    ref_notes = merge_same_pitch_notes(
        extract_notes(pm_ref, min_note_duration=config.min_note_duration),
        gap_tol=config.merge_gap,
    )
    student_notes = merge_same_pitch_notes(
        extract_notes(pm_student, min_note_duration=config.min_note_duration),
        gap_tol=config.merge_gap,
    )

    warp_fn = build_time_warp(ref_notes, student_notes)
    student_warped = merge_same_pitch_notes(
        warp_notes(student_notes, warp_fn),
        gap_tol=config.merge_gap,
    )

    matches, missing, extra = greedy_match(
        ref_notes,
        student_warped,
        time_tol=config.match_time_tolerance,
        pitch_tol=config.match_pitch_tolerance,
    )
    evaluations = evaluate_matches(matches, ref_notes, student_warped, config)
    significant = [e for e in evaluations if e.is_significant]
    ok = [e for e in evaluations if not e.is_significant]
    report = build_error_report(evaluations, missing, extra, ref_notes, student_warped)

    return ComparisonResult(
        reference_notes=ref_notes,
        student_notes_warped=student_warped,
        match_evaluations=evaluations,
        missing_reference_indices=list(missing),
        extra_student_indices=list(extra),
        significant_matches=significant,
        ok_matches=ok,
        error_report=report,
    )


__all__ = [
    "TranscriptionConfig",
    "ComparisonConfig",
    "MidiNote",
    "MatchEvaluation",
    "NoteError",
    "ComparisonResult",
    "load_transcription_model",
    "convert_to_wav",
    "safe_transcribe_local",
    "transcribe_audio_files",
    "align_midi_first_note",
    "compare_performances",
]
