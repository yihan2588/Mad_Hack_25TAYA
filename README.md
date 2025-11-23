---
title: YATA Violin Comparator
emoji: 🎻
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.10"
pinned: false
---

# YATA Violin Comparator

This Space compares a **reference/master** violin performance to a **student** performance.

You can upload either:
- **Audio** (.m4a, .wav, .mp3, …)  
- **MIDI** (.mid / .midi)

## What happens
1. If inputs are audio:  
   **audio → 44.1kHz mono WAV → MUSC violin transcription → MIDI**
2. Align first note of each MIDI to **t = 0**
3. DTW warp student MIDI to reference timeline
4. Greedy matching + detection of **severe** pitch/time misalignments
5. Output a student-centric `error_report`

## Outputs
- Summary counts
- `error_report`: severe errors (missed notes, extra notes, wrong pitch/timing/duration)
- Full JSON for downstream UI

## Hardware
- Works on CPU.
- ZeroGPU supported.
- You can switch to paid GPU hardware in Space Settings for faster transcription.