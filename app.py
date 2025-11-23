import json
import sys
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gradio as gr

# Import AI analyzer
try:
    from ai_analyzer import analyze_performance_with_ai
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False
    print("[WARNING] AI analyzer not available. Install google-generativeai to enable AI features.")

# ---- ZeroGPU-compatible import ----
try:
    import spaces
except ImportError:
    class _DummySpaces:
        @staticmethod
        def GPU(fn=None, **outer_kwargs):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper
            if fn is None:
                return decorator
            return decorator(fn)
    spaces = _DummySpaces()
# -----------------------------------

from yata_team import (
    AUDIO_EXTS,
    ComparisonConfig,
    TranscriptionConfig,
    compare_pipeline,
    load_transcription_model,
    make_side_by_side_plot,
)

_MODEL: Optional[Any] = None
_DEVICE: Optional[str] = None


APP_CSS = """
/* ===== Basic Color System (neutral, high contrast) ===== */
/* ===== Theme-aware Color System ===== */
:root, [data-theme="light"] {
 --bg-page: #f3f4f6;
 --surface: #ffffff;
 --glass: rgba(255, 255, 255, 0.85);
 --glass-strong: rgba(255, 255, 255, 0.92);
 --border: #e5e7eb;
 --text-main: #111827;
 --text: #111827;
 --text-muted: #4b5563;
 --muted: #4b5563;
 --accent: #2563eb;
 --accent-1: #6d8bff;
 --accent-3: #a78bfa;
 --good: rgba(16, 185, 129, 0.15);
 --good-bg: #ecfdf3;
 --good-border: #bbf7d0;
 --good-text: #166534;
 --moderate: rgba(234, 179, 8, 0.15);
 --mod-bg: #fffbeb;
 --mod-border: #facc15;
 --mod-text: #92400e;
 --poor: rgba(244, 63, 94, 0.15);
 --poor-bg: #fef2f2;
 --poor-border: #fecaca;
 --poor-text: #b91c1c;
 --tip-bg: #0f172a;
 --tip-fg: #f8fafc;
}


/* Dark mode colors */
[data-theme="dark"], .dark {
 --bg-page: #0f172a;
 --surface: #1e293b;
 --glass: rgba(30, 41, 59, 0.85);
 --glass-strong: rgba(30, 41, 59, 0.92);
 --border: #334155;
 --text-main: #f1f5f9;
 --text: #f1f5f9;
 --text-muted: #cbd5e1;
 --muted: #cbd5e1;
 --accent: #3b82f6;
 --accent-1: #818cf8;
 --accent-3: #c084fc;
 --good: rgba(34, 197, 94, 0.2);
 --good-bg: rgba(34, 197, 94, 0.1);
 --good-border: rgba(34, 197, 94, 0.3);
 --good-text: #86efac;
 --moderate: rgba(234, 179, 8, 0.2);
 --mod-bg: rgba(234, 179, 8, 0.1);
 --mod-border: rgba(234, 179, 8, 0.3);
 --mod-text: #fde047;
 --poor: rgba(239, 68, 68, 0.2);
 --poor-bg: rgba(239, 68, 68, 0.1);
 --poor-border: rgba(239, 68, 68, 0.3);
 --poor-text: #fca5a5;
 --tip-bg: #f8fafc;
 --tip-fg: #0f172a;
}

body {
  background: var(--bg-page);
  color: var(--text-main);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Gradio base */
.gradio-container {
  background: transparent !important;
}

/* ========= HERO ========= */
.yata-hero{
  text-align:center;
  margin: 0.5rem 0 2rem;
  color: var(--text);
}
.yata-hero h1{
  font-size: clamp(2.1rem, 4vw, 3.2rem);
  margin: 0.35rem 0 0.5rem;
  font-weight: 800;
  letter-spacing: 0.02em;
}
.yata-hero p{
  font-size: clamp(1rem, 1.6vw, 1.15rem);
  color: var(--muted);
}
.yata-pill{
  display:inline-flex;
  gap:.5rem;
  align-items:center;
  padding: .45rem 1rem;
  border-radius: 999px;
  background: var(--glass);
  border: 1px solid var(--border);
  font-size: .9rem;
  color: var(--text);
  backdrop-filter: blur(6px);
}

/* ========= UPLOAD CARDS ========= */
.yata-upload-card{
  background: var(--glass);
  border:1px solid var(--border);
  border-radius: 1.2rem;
  padding: 1rem 1.2rem;
  box-shadow: 0 6px 18px rgba(15,23,42,0.18);
  backdrop-filter: blur(8px);
  min-height: 240px;
  display:flex;
  flex-direction: column;
  gap: .65rem;
}
.yata-upload-card h2{
  font-size: 1.4rem;
  margin: 0;
  font-weight: 700;
  color: var(--text);
}
.yata-upload-card .subtitle{
  color: var(--muted);
  font-size: .98rem;
  margin-top: -0.25rem;
}

/* ========= FILE INPUT ========= */
.yata-file-input label, .yata-file-input button{
  width:100%;
  border-radius: 1rem !important;
  border: 1px dashed rgba(148,163,184,0.8) !important;
  background: rgba(255,255,255,0.04) !important;
  color: var(--text) !important;
  min-height: 150px;
  display:flex;
  flex-direction:column;
  align-items:flex-start;
  justify-content:center;
  gap:0.35rem;
  padding: 1rem 1.25rem !important;
  font-size:0.97rem;
  transition: all .18s ease;
}
.yata-file-input label:hover, .yata-file-input button:hover{
  transform: translateY(-1px);
  border-color: rgba(99,102,241,0.8) !important;
  background: rgba(99,102,241,0.08) !important;
  box-shadow: inset 0 0 0 1px rgba(99,102,241,0.25);
}
.yata-file-input input { color: var(--text) !important; }
.yata-file-input .file-upload{
  width:100%;
}
.yata-file-input .label-text{
  font-weight:600;
}

.yata-audio-preview{
  margin-top: .3rem;
  border: 1px solid rgba(148,163,184,0.5);
  border-radius: 0.8rem;
  padding: 0.35rem 0.65rem;
  background: rgba(15,23,42,0.05);
}
.yata-audio-preview audio{
  height: 32px;
}

/* ========= CTA ========= */
.yata-action{
  display:flex;
  justify-content:center;
  margin: 1.7rem 0 1.2rem;
}
.yata-action button{
  width: min(720px, 100%);
  background-image: linear-gradient(90deg, var(--accent-1), var(--accent-3));
  border:none;
  color:#0b1020;
  font-weight: 800;
  padding: 0.95rem 2.6rem;
  border-radius:999px;
  font-size:1.12rem;
  box-shadow: 0 16px 40px rgba(109,139,255,0.35);
  transition: all .18s ease;
}
.yata-action button:hover{
  transform: translateY(-2px) scale(1.01);
  filter: brightness(1.06);
}
.yata-action button:disabled{ opacity:.6; transform:none; }

/* ========= STATS ========= */
.yata-stats{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
  gap:1rem;
  margin: 0.6rem 0 1.6rem;
}
.yata-stat-card{
  border-radius: 1.2rem;
  padding: 1.15rem 1.25rem;
  color: var(--text);
  border: 1px solid var(--border);
  background: var(--glass-strong);
  box-shadow: 0 12px 26px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}
.yata-stat-label{
  font-size: .8rem;
  opacity: .9;
  margin-bottom: .35rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
}
.yata-stat-value{
  font-size: 2.15rem;
  font-weight: 800;
  line-height: 1.05;
}
#stats_output:empty { display:none; }

/* allow tooltips to escape rounded cards */
.yata-top-row,
.yata-card,
.yata-metric-card,
.yata-summary-card {
  overflow: visible !important;
}

/* info icon + tooltip */
.yata-info{
  --tip-bg: #0f172a;
  --tip-fg: #f8fafc;

  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:18px;
  height:18px;
  border-radius:999px;
  border:1px solid rgba(148,163,184,0.9);
  font-size:.7rem;
  font-weight:700;
  margin-left:0.35rem;
  color:var(--text);
  cursor:help;
  position:relative;
  background:rgba(255,255,255,0.6);
  z-index:50;
}

.yata-info::after{
  content: attr(data-tip);
  position:absolute;
  left:50%;
  top:calc(100% + 8px);
  transform:translateX(-50%) translateY(-2px);
  background:var(--tip-bg);
  color:var(--tip-fg);
  padding:0.35rem 0.55rem;
  border-radius:0.5rem;
  font-size:0.75rem;
  line-height:1.2;
  white-space:normal;
  width:max-content;
  max-width:220px;
  box-shadow:0 8px 24px rgba(15,23,42,0.25);
  opacity:0;
  visibility:hidden;
  pointer-events:none;
  transition:opacity .12s ease, transform .12s ease, visibility 0s linear .12s;
}

.yata-info::before{
  content:"";
  position:absolute;
  left:50%;
  top:calc(100% + 2px);
  transform:translateX(-50%);
  border:6px solid transparent;
  border-top:none;
  border-bottom-color:var(--tip-bg);
  opacity:0;
  visibility:hidden;
  transition:opacity .12s ease, visibility 0s linear .12s;
}

.yata-info:hover::after,
.yata-info:hover::before{
  opacity:1;
  visibility:visible;
  transform:translateX(-50%) translateY(0);
  transition-delay:0s;
}

/* ========= PANELS ========= */
.yata-panel{
  background: #f8fafc;
  border-radius: 1.15rem;
  padding: 1.4rem 1.5rem;
  box-shadow: 0 16px 40px rgba(0,0,0,0.22);
  margin-bottom: 1.1rem;
  border: 1px solid rgba(15,23,42,0.08);
}
.yata-panel-light{ color:#0f172a; }
.yata-panel-light .gr-markdown,
.yata-markdown, .yata-markdown p, .yata-markdown li{
  color:#0f172a !important;
  font-size: 1.02rem;
}
.yata-panel pre{
  background: rgba(15,23,42,0.06);
  border-radius: .7rem;
  padding: 1rem;
  color:#0f172a;
}

/* ========= ACCORDION ========= */
details{
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 1rem;
  padding: .25rem .75rem;
  margin-bottom: .8rem;
  color: var(--text);
  backdrop-filter: blur(10px);
}
summary{
  cursor:pointer;
  font-weight:700;
  color: var(--text);
  padding: .55rem .35rem;
}

/* ========= TIP ========= */
.yata-tip, .yata-tip p{
  color: var(--muted) !important;
  text-align:center;
}

/* Responsive */
@media (max-width: 680px){
  .yata-upload-card{ min-height: 210px; }
  .yata-file-input label, .yata-file-input button{ min-height: 130px; }
}
"""


def _ensure_musc_repo():
    root = Path(__file__).resolve().parent
    musc_dir = root / "MUSC_violin"

    if not musc_dir.exists():
        print("[bootstrap] Cloning MUSC violin-transcription repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/MTG/violin-transcription.git",
             str(musc_dir)],
            check=True,
        )

    if str(musc_dir) not in sys.path:
        sys.path.insert(0, str(musc_dir))
        print("[bootstrap] Added MUSC_violin to sys.path.")


def _get_model() -> Tuple[Any, str]:
    global _MODEL, _DEVICE
    if _MODEL is None:
        _ensure_musc_repo()
        _MODEL, _DEVICE = load_transcription_model(instrument="violin")
    return _MODEL, _DEVICE  # type: ignore


def _coerce_path(file_val: Union[str, Any]) -> str:
    if file_val is None:
        return ""
    if isinstance(file_val, str):
        return file_val
    if hasattr(file_val, "name"):
        return str(file_val.name)
    raise ValueError("Unsupported file input type.")


def _audio_preview_value(file_val: Union[str, Any]):
    path = _coerce_path(file_val)
    if not path:
        return gr.update(value=None, visible=False)
    if Path(path).suffix.lower() not in AUDIO_EXTS:
        return gr.update(value=None, visible=False)
    return gr.update(value=path, visible=True)


def make_summary(result_dict: Dict[str, Any]) -> str:
    s = result_dict.get("summary", {})
    info = ""
    return f"""
### Summary {info}
- Reference notes: **{s.get("n_reference_notes", 0)}**
- Student notes: **{s.get("n_student_notes_warped", 0)}**
- Matched pairs: **{s.get("n_matches", 0)}**
- Missing note: **{s.get("n_missing_reference", 0)}**
- Extra note: **{s.get("n_extra_student", 0)}**
- Wrong notes: **{s.get("n_significant", 0)}**
"""


def compute_frontend_stats(result_dict: Dict[str, Any]) -> Dict[str, int]:
    summary = result_dict.get("summary", {})
    total_ref = max(int(summary.get("n_reference_notes", 0)), 1)
    errors = result_dict.get("error_report", [])
    counts = Counter(entry.get("student_error_type", "") for entry in errors)

    moderate_types = {"wrong_timing", "wrong_duration"}
    poor_types = {"wrong_pitch", "missed_note", "extra_note"}

    moderate_count = sum(counts[t] for t in moderate_types)
    poor_count = sum(counts[t] for t in poor_types)
    good_count = max(total_ref - moderate_count - poor_count, 0)

    def pct(part: int) -> int:
        return max(0, min(100, int(round(100 * part / total_ref))))

    overall_penalty = (moderate_count * 0.5 + poor_count) / total_ref
    overall = max(0, min(100, int(round(100 * (1 - overall_penalty)))))

    return {
        "overallScore": overall,
        "goodPercent": pct(good_count),
        "moderatePercent": pct(moderate_count),
        "poorPercent": pct(poor_count),
    }


def render_stats_html(stats: Dict[str, int]) -> str:
    if not stats:
        return ""
    return f"""
    <div class='yata-stats'>
        <div class='yata-stat-card' style='background: linear-gradient(135deg, rgba(109,139,255,0.18), rgba(167,139,250,0.20));'>
            <div class='yata-stat-label'>Overall Score</div>
            <div class='yata-stat-value'>{stats.get('overallScore', 0)}%</div>
        </div>
        <div class='yata-stat-card' style='background: linear-gradient(135deg, var(--good), rgba(16,185,129,0.10));'>
            <div class='yata-stat-label'>Excellent <span class="yata-info" data-tip="Notes played perfectly.">i</span></div>
            <div class='yata-stat-value'>{stats.get('goodPercent', 0)}%</div>
        </div>
        <div class='yata-stat-card' style='background: linear-gradient(135deg, var(--moderate), rgba(234,179,8,0.10));'>
            <div class='yata-stat-label'>Moderate <span class="yata-info" data-tip="Notes with noticeable timing or duration drift.">i</span></div>
            <div class='yata-stat-value'>{stats.get('moderatePercent', 0)}%</div>
        </div>
        <div class='yata-stat-card' style='background: linear-gradient(135deg, var(--poor), rgba(244,63,94,0.10));'>
            <div class='yata-stat-label'>Needs Work <span class="yata-info" data-tip="Missed, extra, or badly pitched notes.">i</span></div>
            <div class='yata-stat-value'>{stats.get('poorPercent', 0)}%</div>
        </div>
    </div>
    """


def run_compare(
    reference_file: Union[str, Any],
    student_file: Union[str, Any],
    merge_gap: float,
    min_note_duration: float,
    match_time_tol: float,
    match_pitch_tol: int,
    sig_onset_tol: float,
    sig_pitch_tol: int,
    sig_offset_tol: float,
    batch_size: int,
    postprocessing: str,
):
    ref_path = _coerce_path(reference_file)
    stu_path = _coerce_path(student_file)

    if not ref_path or not stu_path:
        raise gr.Error("Please upload both reference and student files.")

    cconfig = ComparisonConfig(
        merge_gap=merge_gap,
        min_note_duration=min_note_duration,
        match_time_tolerance=match_time_tol,
        match_pitch_tolerance=match_pitch_tol,
        significant_onset_tolerance=sig_onset_tol,
        significant_pitch_tolerance=sig_pitch_tol,
        significant_offset_tolerance=sig_offset_tol,
    )

    tconfig = TranscriptionConfig(
        batch_size=batch_size,
        postprocessing=postprocessing,
        instrument="violin",
    )

    model, device = _get_model()

    result = compare_pipeline(
        ref_path,
        stu_path,
        config=cconfig,
        tconfig=tconfig,
        model=model,
        device=device,
        work_dir=None,
    )

    severe_dict = json.loads(result.to_severe_json())
    fig = make_side_by_side_plot(
        result,
        titleA=Path(ref_path).name,
        titleB=Path(stu_path).name,
    )

    return severe_dict, fig


@spaces.GPU
def run_compare_gpu(*args, **kwargs):
    return run_compare(*args, **kwargs)


theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size="lg",
)

with gr.Blocks(title="YATA Violin Comparator", css=APP_CSS, theme=theme) as demo:
    gr.HTML(
        """
        <div class='yata-hero'>
            <div class='yata-pill'>🎻 YATA vioLin</div>
            <h1>Want to know how well you've played?</h1>
            <p>Drop in your polished reference and your take—we'll spotlight the big things to fix.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(elem_classes=["yata-upload-card"]):
            gr.HTML(
                "<h2>Master Reference</h2>"
                "<p class='subtitle'>Upload the correct audio as reference.</p>"
            )
            reference_in = gr.File(
                label="",
                file_types=[".mid", ".midi", ".m4a", ".wav",
                            ".mp3", ".flac", ".aac", ".ogg"],
                type="filepath",
                elem_classes=["yata-file-input"],
            )
            reference_preview = gr.Audio(
                value=None,
                interactive=False,
                show_label=False,
                elem_classes=["yata-audio-preview"],
                visible=False,
                type="filepath",
            )

        with gr.Column(elem_classes=["yata-upload-card"]):
            gr.HTML(
                "<h2>Your Performance</h2>"
                "<p class='subtitle'>Upload the student performance (to be evaluated).</p>"
            )
            student_in = gr.File(
                label="",
                file_types=[".mid", ".midi", ".m4a", ".wav",
                            ".mp3", ".flac", ".aac", ".ogg"],
                type="filepath",
                elem_classes=["yata-file-input"],
            )
            student_preview = gr.Audio(
                value=None,
                interactive=False,
                show_label=False,
                elem_classes=["yata-audio-preview"],
                visible=False,
                type="filepath",
            )

    with gr.Row(elem_classes=["yata-action"]):
        run_btn = gr.Button("Analyze Performance", variant="primary")

    stats_out = gr.HTML(elem_id="stats_output")
    summary_out = gr.Markdown(
        elem_classes=["yata-panel", "yata-panel-light", "yata-markdown"])

    with gr.Accordion("Comparison settings", open=False):
        merge_gap = gr.Slider(0.0, 1.0, value=0.3,
                              step=0.01, label="Merge gap (sec)")
        min_note_duration = gr.Slider(
            0.0, 1.0, value=0.3, step=0.01, label="Min note duration (sec)")
        match_time_tol = gr.Slider(
            0.0, 1.0, value=0.3, step=0.01, label="Match time tolerance (sec)")
        match_pitch_tol = gr.Slider(
            0, 12, value=1, step=1, label="Match pitch tolerance (semitones)")
        sig_onset_tol = gr.Slider(
            0.0, 2.0, value=0.3, step=0.01, label="Significant onset tolerance (sec)")
        sig_pitch_tol = gr.Slider(
            0, 24, value=1, step=1, label="Significant pitch tolerance (semitones)")
        sig_offset_tol = gr.Slider(
            0.0, 2.0, value=0.3, step=0.01, label="Significant offset tolerance (sec)")

    with gr.Accordion("Transcription settings (audio only)", open=False):
        batch_size = gr.Slider(1, 64, value=32, step=1, label="Batch size")
        postprocessing = gr.Dropdown(
            choices=["spotify"], value="spotify", label="Postprocessing")

    plot_out = gr.Plot(label="Reference vs Student (warped)",
                       elem_classes=["yata-panel"])
    report_out = gr.JSON(
        label="Severe misalignment events (error_report)",
        elem_classes=["yata-panel", "yata-panel-light"],
    )
    
    # AI Analysis output (conditional on whether API key is available)
    ai_analysis_out = gr.Markdown(
        label="🤖 AI Coach Analysis",
        elem_classes=["yata-panel", "yata-panel-light", "yata-markdown"],
        visible=AI_ENABLED
    ) if AI_ENABLED else None

    def _on_click(*args):
        print("[LOG] _on_click: started")
        print("[LOG] _on_click: starting compare_pipeline...")
        severe_dict, fig = run_compare_gpu(*args)
        print("[LOG] _on_click: compare_pipeline finished")
        summary_md = make_summary(severe_dict)
        stats_html = render_stats_html(compute_frontend_stats(severe_dict))
        error_report = severe_dict.get("error_report", [])
        
        # Generate AI analysis if available
        ai_feedback = ""
        if AI_ENABLED:
            try:
                print("[LOG] _on_click: starting AI analysis...")
                ai_feedback = analyze_performance_with_ai({
                    "summary": severe_dict.get("summary", {}),
                    "error_report": error_report,
                })
                print("[LOG] _on_click: AI analysis finished")
            except Exception as e:
                print(f"[ERROR] _on_click: AI analysis failed: {e}")
                ai_feedback = f"⚠️ AI analysis unavailable: {str(e)}"
        
        if AI_ENABLED:
            return stats_html, summary_md, error_report, fig, ai_feedback
        else:
            return stats_html, summary_md, error_report, fig

    outputs = [stats_out, summary_out, report_out, plot_out]
    if AI_ENABLED:
        outputs.append(ai_analysis_out)

    # Quick UI feedback: show a loading placeholder immediately when Analyze is clicked.
    def _show_loading():
        loading_stats = "<div style='padding:1rem;'>analyzing... It might take a while</div>"
        loading_summary = "**analyzing...**"
        loading_report = {}
        loading_plot = None
        if AI_ENABLED:
            loading_ai = "🤖 AI working hard, pls be patient..."
            return loading_stats, loading_summary, loading_report, loading_plot, loading_ai
        return loading_stats, loading_summary, loading_report, loading_plot
    # Register a fast placeholder update first, then the heavy worker. Gradio will run callbacks in order.
    run_btn.click(_show_loading, inputs=[], outputs=outputs)
    run_btn.click(
        _on_click,
        inputs=[
            reference_in,
            student_in,
            merge_gap,
            min_note_duration,
            match_time_tol,
            match_pitch_tol,
            sig_onset_tol,
            sig_pitch_tol,
            sig_offset_tol,
            batch_size,
            postprocessing,
        ],
        outputs=outputs,
    )

    reference_in.change(_audio_preview_value,
                        inputs=reference_in, outputs=reference_preview)
    student_in.change(_audio_preview_value, inputs=student_in,
                      outputs=student_preview)

    gr.Markdown(
        "**Tip:** We flag locally severe errors, not tiny accumulated drift.",
        elem_classes=["yata-tip"],
    )

if __name__ == "__main__":
    demo.queue().launch()
