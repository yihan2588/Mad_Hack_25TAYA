"""
Backend utilities for comparing violin performances.

Pipeline (matches your Colab):
audio (.m4a/.wav/...) -> WAV (44.1kHz mono) -> MUSC transcription -> aligned MIDI
-> DTW time warp -> greedy match -> significant error report + visualization.

Exports:
- compare_pipeline: main end-to-end entry point for audio or MIDI inputs
- compare_performances: MIDI-only comparison
- make_side_by_side_plot / plot_side_by_side: visualization helper
- ComparisonResult.to_severe_json(): severe-only JSON (error_report + summary counts)
- ComparisonResult.to_json(): full JSON (kept for backward compatibility)
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pretty_midi

# matplotlib for plot output (Agg backend safe for HF)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (after backend set)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


PathLike = Union[str, Path]

AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".ogg"}
MIDI_EXTS = {".mid", ".midi"}


# ---------------------------------------------------------------------------
# JSON sanitization (fixes numpy/torch scalars not serializable)
# ---------------------------------------------------------------------------
def _to_jsonable(x: Any) -> Any:
    """
    Recursively convert numpy / torch scalar types (and other oddballs)
    into plain Python types so json.dumps won't crash.
    """
    # numpy scalars (np.float64, np.bool_, etc.)
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
    except Exception:
        pass

    # torch scalars / tensors
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            if x.ndim == 0:
                return x.item()
            return x.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    return x


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------
@dataclass
class TranscriptionConfig:
    batch_size: int = 32
    postprocessing: str = "spotify"
    target_sample_rate: int = 44_100
    mono: bool = True
    instrument: str = "violin"


def load_transcription_model(instrument: str = "violin", device: Optional[str] = None):
    """
    Load MUSC PretrainedModel. Assumes MUSC repo is already on sys.path
    (your app.py clones MUSC_violin and inserts path).
    """
    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    try:
        from musc.model import PretrainedModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import musc.PretrainedModel. "
            "Make sure violin-transcription repo is cloned and on sys.path."
        ) from exc

    model = PretrainedModel(instrument=instrument).to(device)
    model.eval()
    return model, device


def convert_to_wav(
    in_path: PathLike,
    *,
    output_path: Optional[PathLike] = None,
    sample_rate: int = 44_100,
    mono: bool = True,
) -> Path:
    """
    Convert input audio to 44.1kHz mono wav (MUSC expects this).
    Uses ffmpeg (installed via packages.txt).
    """
    in_path = Path(in_path)
    if output_path is None:
        output_path = in_path.with_suffix(".wav")
    out_path = Path(output_path)

    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-ar", str(sample_rate)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(str(out_path))

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def _normalize_to_pretty_midi(out: Any) -> pretty_midi.PrettyMIDI:
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
    """
    Try multiple MUSC entry points (matching your Colab safe_transcribe_local).
    Always pass audio_path positionally.
    """
    candidates: List[Callable[..., Any]] = []

    if hasattr(model, "transcribe"):
        candidates.append(model.transcribe)
    if hasattr(model, "transcribe_file"):
        candidates.append(model.transcribe_file)
    if hasattr(model, "transcriber") and hasattr(model.transcriber, "transcribe"):
        candidates.append(model.transcriber.transcribe)  # type: ignore[attr-defined]

    tried: List[Tuple[str, str]] = []

    for fn in candidates:
        patterns = [
            lambda: fn(str(audio_path), batch_size=batch_size, postprocessing=postprocessing),
            lambda: fn(str(audio_path), batch_size=batch_size),
            lambda: fn(str(audio_path), postprocessing=postprocessing),
            lambda: fn(str(audio_path)),
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
            "start": float(self.start),
            "end": float(self.end),
            "pitch": int(self.pitch),
            "velocity": int(self.velocity),
            "duration": float(self.duration),
            "centroid": float(self.centroid),
        }


@dataclass
class ComparisonConfig:
    merge_gap: float = 0.3
    min_note_duration: float = 0.3
    match_time_tolerance: float = 0.3
    match_pitch_tolerance: int = 1
    significant_onset_tolerance: float = 0.3
    significant_pitch_tolerance: int = 1
    significant_offset_tolerance: float = 0.3


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

    def summary_counts(self) -> Dict[str, int]:
        return {
            "n_reference_notes": len(self.reference_notes),
            "n_student_notes_warped": len(self.student_notes_warped),
            "n_matches": len(self.match_evaluations),
            "n_missing_reference": len(self.missing_reference_indices),
            "n_extra_student": len(self.extra_student_indices),
            "n_significant": len(self.significant_matches),
        }

    def to_severe_json(self) -> str:
        """
        Severe-only payload:
        - summary counts
        - error_report (student-centric severe events)
        """
        payload = {
            "summary": self.summary_counts(),
            "error_report": [e.to_dict() for e in self.error_report],
        }
        payload = _to_jsonable(payload)
        return json.dumps(payload, indent=2)

    def to_json(self) -> str:
        """
        Full payload (kept for backward compatibility).
        """
        payload = {
            "reference_notes": [n.to_dict() for n in self.reference_notes],
            "student_notes_warped": [n.to_dict() for n in self.student_notes_warped],
            "match_evaluations": [m.to_dict() for m in self.match_evaluations],
            "missing_reference_indices": self.missing_reference_indices,
            "extra_student_indices": self.extra_student_indices,
            "significant_matches": [m.to_dict() for m in self.significant_matches],
            "ok_matches": [m.to_dict() for m in self.ok_matches],
            "error_report": [e.to_dict() for e in self.error_report],
            "summary": self.summary_counts(),
        }
        payload = _to_jsonable(payload)
        return json.dumps(payload, indent=2)


def first_note_time(pm: pretty_midi.PrettyMIDI, min_dur: float = 0.03) -> float:
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
    midi_path = Path(midi_path)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    offset = first_note_time(pm, min_note_duration)
    shifted = shift_pretty_midi(pm, offset)
    dest = Path(output_path) if output_path else midi_path
    shifted.write(str(dest))
    return dest


# ---------------------------------------------------------------------------
# MIDI comparison core (matches Colab logic)
# ---------------------------------------------------------------------------
def extract_notes(pm: pretty_midi.PrettyMIDI, *, min_note_duration: float) -> List[MidiNote]:
    notes: List[MidiNote] = []
    for inst in pm.instruments:
        for note in inst.notes:
            if (note.end - note.start) >= min_note_duration:
                notes.append(MidiNote(float(note.start), float(note.end), int(note.pitch), int(note.velocity)))
    notes.sort(key=lambda n: n.start)
    return notes


def merge_same_pitch_notes(notes: Sequence[MidiNote], *, gap_tol: float) -> List[MidiNote]:
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

    t_a = np.array([a for a, _ in anchors], dtype=float)
    t_b = np.array([b for _, b in anchors], dtype=float)

    def warp_time(tb: float) -> float:
        return float(np.interp(tb, t_b, t_a))

    return warp_time


def warp_notes(notes: Sequence[MidiNote], warp_fn: Callable[[float], float]) -> List[MidiNote]:
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
        matches.append((i, j_best, float(dt_best), float(dp_best)))

    extra = [idx for idx, flag in enumerate(used) if not flag]
    return matches, missing, extra


def evaluate_matches(
    matches: Sequence[Tuple[int, int, float, float]],
    notes_ref: Sequence[MidiNote],
    notes_student: Sequence[MidiNote],
    config: ComparisonConfig,
) -> List[MatchEvaluation]:
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
                reference_index=int(i),
                student_index=int(j),
                onset_diff_sec=float(dt_onset),
                pitch_diff_semitones=float(dp),
                offset_diff_sec=float(dt_offset),
                student_error_type=str(err_type),
                is_significant=bool(is_sig),
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
    rows: List[NoteError] = []
    for idx in missing:
        note = notes_ref[idx]
        rows.append(NoteError("missed_note", time_sec=note.start, gt_pitch=note.pitch, student_pitch=None))
    for idx in extra:
        note = notes_student[idx]
        rows.append(NoteError("extra_note", time_sec=note.start, gt_pitch=None, student_pitch=note.pitch))
    for ev in evaluations:
        if ev.student_error_type == "ok":
            continue
        ref_note = notes_ref[ev.reference_index]
        stud_note = notes_student[ev.student_index]
        rows.append(
            NoteError(
                student_error_type=ev.student_error_type,
                time_sec=ref_note.start,
                gt_pitch=ref_note.pitch,
                student_pitch=stud_note.pitch,
                pitch_diff_semitones=ev.pitch_diff_semitones,
                onset_diff_sec=ev.onset_diff_sec,
                offset_diff_sec=ev.offset_diff_sec,
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


# ---------------------------------------------------------------------------
# Visualization (same as Colab, but returns fig instead of plt.show())
# ---------------------------------------------------------------------------
def plot_side_by_side(
    notesA: Sequence[MidiNote],
    notesB: Sequence[MidiNote],
    sig_matches: Sequence[MatchEvaluation],
    missingA: Sequence[int],
    extraB: Sequence[int],
    titleA: str,
    titleB: str,
) -> plt.Figure:
    """
    A is correct → never colored red.
    B is student → highlight only B errors.
    Returns a matplotlib Figure (HF/Gradio friendly).
    """
    sigB_idx = {m.student_index for m in sig_matches} | set(extraB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    def draw_gt(ax, notes, title):
        for n in notes:
            ax.plot([n.start, n.end], [n.pitch, n.pitch], linewidth=3, alpha=0.9, color="gray")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Pitch")
        ax.grid(True, alpha=0.2)

    def draw_student(ax, notes, sig_idx, title):
        for k, n in enumerate(notes):
            color = "red" if k in sig_idx else "gray"
            ax.plot([n.start, n.end], [n.pitch, n.pitch], linewidth=3, alpha=0.9, color=color)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Pitch")
        ax.grid(True, alpha=0.2)

    draw_gt(axes[0], notesA, f"{titleA} (ground truth)")
    draw_student(axes[1], notesB, sigB_idx, f"{titleB} (student, warped-to-gt)")

    import matplotlib.lines as mlines
    ok_line = mlines.Line2D([], [], color="gray", linewidth=3, label="Correct / matched")
    sig_line = mlines.Line2D([], [], color="red", linewidth=3, label="Student error")
    fig.legend(handles=[ok_line, sig_line], loc="upper center", ncol=2)

    plt.tight_layout()
    return fig


def make_side_by_side_plot(
    result: ComparisonResult,
    titleA: str = "Reference",
    titleB: str = "Student",
) -> plt.Figure:
    """Convenience wrapper using a ComparisonResult."""
    return plot_side_by_side(
        result.reference_notes,
        result.student_notes_warped,
        result.significant_matches,
        result.missing_reference_indices,
        result.extra_student_indices,
        titleA,
        titleB,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline for audio OR MIDI inputs
# ---------------------------------------------------------------------------
def _is_audio(path: PathLike) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def _is_midi(path: PathLike) -> bool:
    return Path(path).suffix.lower() in MIDI_EXTS


def _ensure_midi_from_input(
    input_path: PathLike,
    *,
    model: Any,
    tconfig: TranscriptionConfig,
    device: str,
    work_dir: Path,
) -> Path:
    """
    If input is MIDI: copy+align it into work_dir.
    If input is audio: wav->transcribe->save MIDI->align into work_dir.
    Returns aligned MIDI path.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if _is_midi(input_path):
        midi_out = work_dir / input_path.with_suffix(".mid").name
        midi_out.write_bytes(input_path.read_bytes())
        align_midi_first_note(midi_out, min_note_duration=0.03)
        return midi_out

    if _is_audio(input_path):
        wav_path = convert_to_wav(
            input_path,
            output_path=work_dir / input_path.with_suffix(".wav").name,
            sample_rate=tconfig.target_sample_rate,
            mono=tconfig.mono,
        )
        midi_obj = safe_transcribe_local(
            model,
            wav_path,
            batch_size=tconfig.batch_size,
            postprocessing=tconfig.postprocessing,
        )
        midi_out = work_dir / input_path.with_suffix(".mid").name
        midi_obj.write(str(midi_out))
        align_midi_first_note(midi_out, min_note_duration=0.03)
        return midi_out

    raise ValueError(f"Unsupported file type: {suffix}. Provide audio or MIDI.")


def compare_pipeline(
    good_input: PathLike,
    bad_input: PathLike,
    *,
    config: Optional[ComparisonConfig] = None,
    tconfig: Optional[TranscriptionConfig] = None,
    model: Optional[Any] = None,
    device: Optional[str] = None,
    work_dir: Optional[PathLike] = None,
) -> ComparisonResult:
    """
    Full pipeline mirroring Colab:
    input (audio OR midi) -> aligned MIDI -> compare -> result.
    """
    config = config or ComparisonConfig()
    tconfig = tconfig or TranscriptionConfig()

    needs_transcription = _is_audio(good_input) or _is_audio(bad_input)

    if needs_transcription and model is None:
        model, device = load_transcription_model(instrument=tconfig.instrument, device=device)

    if device is None:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="yata_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    good_midi = _ensure_midi_from_input(
        good_input, model=model, tconfig=tconfig, device=device, work_dir=work_dir
    )
    bad_midi = _ensure_midi_from_input(
        bad_input, model=model, tconfig=tconfig, device=device, work_dir=work_dir
    )

    return compare_performances(good_midi, bad_midi, config=config)


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
    "align_midi_first_note",
    "compare_performances",
    "compare_pipeline",
    "plot_side_by_side",
    "make_side_by_side_plot",
]