import pretty_midi
import numpy as np

# Parameters
merge_gap      = 0.3
min_note_dur   = 0.3
match_time_tol = 0.3
match_pitch_tol = 1
sig_onset_tol  = 0.5
sig_pitch_tol  = 5
sig_offset_tol = 0.5


# -------------------------------
#       NOTE EXTRACTION
# -------------------------------

def extract_notes(pm: pretty_midi.PrettyMIDI, min_dur=min_note_dur):
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            if (n.end - n.start) >= min_dur:
                notes.append({
                    "start": float(n.start),
                    "end": float(n.end),
                    "pitch": int(n.pitch),
                    "velocity": int(n.velocity),
                })
    notes.sort(key=lambda x: x["start"])
    return notes


def merge_same_pitch_notes(notes, gap_tol=merge_gap):
    if not notes:
        return notes
    merged = [notes[0].copy()]
    for n in notes[1:]:
        last = merged[-1]
        gap = n["start"] - last["end"]
        if n["pitch"] == last["pitch"] and gap <= gap_tol:
            last["end"] = max(last["end"], n["end"])
            last["velocity"] = max(last["velocity"], n["velocity"])
        else:
            merged.append(n.copy())
    return merged


# -------------------------------
#       DTW + TIME WARPING
# -------------------------------

def dtw_path(seqA, seqB):
    n, m = len(seqA), len(seqB)
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(seqA[i-1] - seqB[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        step = np.argmin([D[i-1, j], D[i, j-1], D[i-1, j-1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    return path[::-1]


def build_time_warp(notesA, notesB):
    pitchA = [n["pitch"] for n in notesA]
    pitchB = [n["pitch"] for n in notesB]

    path = dtw_path(pitchA, pitchB)

    # Map reference indices → student indices
    mapping = {}
    for i, j in path:
        mapping.setdefault(i, []).append(j)

    # Median matching
    pairs = []
    for i, js in mapping.items():
        j_med = int(np.median(js))
        pairs.append((i, j_med))

    pairs.sort(key=lambda ij: notesA[ij[0]]["start"])

    # Build aligned time arrays
    tA, tB = [], []
    last_tb = -1
    for i, j in pairs:
        ta = notesA[i]["start"]
        tb = notesB[j]["start"]
        if tb > last_tb:
            tA.append(ta)
            tB.append(tb)
            last_tb = tb

    tA = np.array(tA)
    tB = np.array(tB)

    def warp_time(tb):
        return float(np.interp(tb, tB, tA))

    return warp_time


def warp_notes(notesB, warp_fn):
    warped = []
    for n in notesB:
        warped.append({
            **n,
            "start": warp_fn(n["start"]),
            "end": warp_fn(n["end"]),
        })
    warped.sort(key=lambda x: x["start"])
    return warped


# -------------------------------
#       NOTE MATCHING
# -------------------------------

def greedy_match(notesA, notesB):
    usedB = np.zeros(len(notesB), dtype=bool)
    matches, missingA = [], []

    for i, a in enumerate(notesA):
        candidates = []
        for j, b in enumerate(notesB):
            if usedB[j]:
                continue

            dt = abs(b["start"] - a["start"])
            dp = abs(b["pitch"] - a["pitch"])

            if dt <= match_time_tol and dp <= match_pitch_tol:
                candidates.append((dt + 0.05 * dp, j, dt, dp))

        if not candidates:
            missingA.append(i)
            continue

        # best match
        candidates.sort(key=lambda x: x[0])
        _, jbest, dtbest, dpbest = candidates[0]

        usedB[jbest] = True
        matches.append((i, jbest, dtbest, dpbest))

    extraB = [j for j in range(len(notesB)) if not usedB[j]]
    return matches, missingA, extraB


def classify_matches(matches, notesA, notesB):
    recs = []
    for i, j, dt_onset, dp in matches:
        a = notesA[i]
        b = notesB[j]

        dt_offset = abs(b["end"] - a["end"])

        err_type = (
            "wrong_pitch" if dp >= sig_pitch_tol else
            "wrong_timing" if dt_onset >= sig_onset_tol else
            "wrong_duration" if dt_offset >= sig_offset_tol else
            "ok"
        )

        recs.append({
            "reference_index": i,
            "student_index": j,
            "time_sec": a["start"],
            "gt_pitch": a["pitch"],
            "student_pitch": b["pitch"],
            "onset_diff_sec": dt_onset,
            "offset_diff_sec": dt_offset,
            "student_error_type": err_type,
        })

    return recs


# -------------------------------
#       MAIN ENTRY FUNCTION
# -------------------------------

def run_midi_comparison(ref_path, stu_path):
    pmA = pretty_midi.PrettyMIDI(ref_path)
    pmB = pretty_midi.PrettyMIDI(stu_path)

    notesA = merge_same_pitch_notes(extract_notes(pmA))
    notesB = merge_same_pitch_notes(extract_notes(pmB))

    warp_fn = build_time_warp(notesA, notesB)
    notesB_warped = merge_same_pitch_notes(warp_notes(notesB, warp_fn))

    matches, missingA, extraB = greedy_match(notesA, notesB_warped)
    evaluations = classify_matches(matches, notesA, notesB_warped)

    return {
        "reference_notes": notesA,
        "student_notes_warped": notesB_warped,
        "match_evaluations": evaluations,
        "extra_student_indices": extraB,
        "error_report": evaluations,
    }
