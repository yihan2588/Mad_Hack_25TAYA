// backend.js
export const API_BASE_URL =
  process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

// Compare MIDI files
export async function compareMidiFiles({ referenceFile, studentFile }) {
  const formData = new FormData();
  formData.append("referenceFile", referenceFile);
  formData.append("studentFile", studentFile);

  const res = await fetch(`${API_BASE_URL}/api/compare-midi`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("MIDI comparison failed");
  }

  return await res.json();
}

// Get AI feedback
export async function analyzePerformanceWithLLM({ diffJson }) {
  const res = await fetch(`${API_BASE_URL}/api/llm-feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ diffJson }),
  });

  if (!res.ok) {
    throw new Error("AI feedback failed");
  }

  const data = await res.json();
  return data.feedback;
}
