const DEFAULT_BASE_URL = process.env.REACT_APP_BACKEND_URL || "";

const buildUrl = (path) => {
  if (DEFAULT_BASE_URL) {
    return `${DEFAULT_BASE_URL.replace(/\/$/, "")}${path}`;
  }
  return path;
};

export const compareMidiFiles = async ({ referenceFile, studentFile }) => {
  if (!referenceFile || !studentFile) {
    throw new Error("Both reference and student files are required.");
  }

  const formData = new FormData();
  formData.append("reference_midi", referenceFile);
  formData.append("student_midi", studentFile);

  const response = await fetch(buildUrl("/api/compare"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Failed to analyze MIDI files.");
  }

  return response.json();
};
