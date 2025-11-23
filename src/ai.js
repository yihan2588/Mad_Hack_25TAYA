import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const MODEL_NAME = "gemini-2.0-flash";

const VIOLON_SYSTEM_PROMPT = `
You are Violon, an AI violin performance coach.
You specialize in violin pedagogy, intonation analysis, musical timing, and practice planning.
You will receive a diff JSON comparing Ground Truth MIDI and Predicted MIDI.
Your job:
- Understand the mistakes
- Identify patterns
- Recommend concrete skill-based exercises
- Be supportive and helpful
- Sound like a friendly music teacher, not a chatbot
Your output must be clear, actionable, musically accurate, and encouraging.
`.trim();

const VIOLON_USER_PROMPT = `
Analyze the violin performance based on the following comparison data:
{diff_json}

Please provide:
1. A short performance summary (3–4 sentences)
2. The 3 most common mistake patterns (pitch trends, rhythm patterns, etc.)
3. 2–3 specific examples with note names or timestamps
4. A personalized practice plan (3 actionable steps)
5. One motivational closing sentence

Keep it concise (120–180 words).
`.trim();

const parseResponseContent = (response) => {
  if (!response) return "";

  if (Array.isArray(response.content)) {
    return response.content
      .map((part) => (typeof part?.text === "string" ? part.text : ""))
      .join("\n")
      .trim();
  }

  if (typeof response.content === "string") {
    return response.content.trim();
  }

  return "";
};

const resolveApiKey = () => {
    return process.env.REACT_APP_GOOGLE_API_KEY || "";
  };
  console.log("DEBUG KEY:", process.env.REACT_APP_GOOGLE_API_KEY);


export const invokeGemini = async ({
  systemPrompt,
  humanTemplate,
  inputValues,
  temperature = 0.2,
}) => {
  const apiKey = resolveApiKey();

  if (!apiKey) {
    throw new Error(
      "Missing Google API key. Set VITE_GOOGLE_API_KEY or REACT_APP_GOOGLE_API_KEY."
    );
  }

  const trimmedInputs = Object.fromEntries(
    Object.entries(inputValues || {}).map(([key, value]) => [
      key,
      typeof value === "string" ? value.trim() : value,
    ])
  );

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    ["human", humanTemplate],
  ]);

  const promptValue = await promptTemplate.invoke(trimmedInputs);

  const model = new ChatGoogleGenerativeAI({
    model: MODEL_NAME,
    temperature,
    apiKey,
  });

  const response = await model.invoke(promptValue);
  const content = parseResponseContent(response);

  if (!content) {
    throw new Error("No response from AI.");
  }

  return content;
};

export const analyzePerformanceWithLLM = async ({ diffJson }) => {
  const trimmedDiff = (diffJson ?? "").trim();
  if (!trimmedDiff) {
    throw new Error("No diff JSON provided for analysis.");
  }

  return invokeGemini({
    systemPrompt: VIOLON_SYSTEM_PROMPT,
    humanTemplate: VIOLON_USER_PROMPT,
    inputValues: {
      diff_json: trimmedDiff,
    },
    temperature: 0.35,
  });
};

export const analyzeNoteWithRubric = async ({ noteContent, rubric }) => {
  const trimmedNote = (noteContent ?? "").trim();

  if (!trimmedNote) {
    throw new Error("Write something first!");
  }

  const rubricText = (rubric ?? "").trim() || "No rubric provided.";

  return invokeGemini({
    systemPrompt:
      "Analyze the following questions of student notes. identify the corresponding level of the Revised Bloom's Taxonomy (Remembering, Understanding, Applying, Analyzing, Evaluating, or Creating). Assign a corresponding XP value and Badge to the student based on the level achieved. Explain the rationale for your classification, focusing on the cognitive action verb or demand in the student's statement. Suggest specific Feedback that will guide the student to revise the note and achieve the next higher level of Bloom's Taxonomy. Present the results in a clear table format.",
    humanTemplate: "Rubric:\\n{rubric}\\n\\nStudent Note:\\n{note}",
    inputValues: {
      rubric: rubricText,
      note: trimmedNote,
    },
    temperature: 0,
  });
};

