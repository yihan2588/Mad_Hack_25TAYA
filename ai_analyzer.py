"""
AI-powered violin performance analysis using Google Gemini + LangChain.
Ported from JavaScript version to Python with LangChain integration.
"""

import json
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    print("[DEBUG] python-dotenv not installed; .env will not be auto-loaded. Install with 'pip install python-dotenv'")
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


MODEL_NAME = "gemini-2.0-flash"


VIOLON_SYSTEM_PROMPT = """
You are Violon, an AI violin performance coach.
You specialize in violin pedagogy, intonation analysis, musical timing, and practice planning.
You will receive an error report from a violin performance comparison system.
Your job:
- Understand the mistakes
- Identify patterns
- Recommend concrete skill-based exercises
- Be supportive and helpful
- Sound like a friendly music teacher, not a chatbot
Your output must be clear, actionable, musically accurate, and encouraging.
""".strip()

VIOLON_USER_PROMPT = """
Analyze the violin performance based on the following comparison data:
{error_report_json}

Please provide:
1. A short performance summary (3–4 sentences)
2. The 3 most common mistake patterns (pitch trends, rhythm patterns, etc.)
3. 2–3 specific examples with note names or timestamps
4. A personalized practice plan (3 actionable steps)
5. One motivational closing sentence

Keep it concise (120–180 words).
""".strip()


def resolve_api_key() -> str:
    """
    Resolve Google API key from environment variables.
    Tries multiple common variable names (matches JavaScript version).
    """
    # Try various common env var names (same as JS version)
    for var_name in [
        "GOOGLE_API_KEY",
        "VITE_GOOGLE_API_KEY",
        "REACT_APP_GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]:
        key = os.getenv(var_name)
        if key:
            print(f"[DEBUG] Found API key in: {var_name}")
            return key
    return ""


def invoke_gemini(system_prompt: str,
                 human_template: str,
                 input_values: Dict[str, Any],
                 temperature: float = 0.2) -> str:
    """
    Invoke Gemini model with LangChain (Python equivalent of JS version).
   
    Args:
        system_prompt: System instruction for the model
        human_template: User message template with placeholders
        input_values: Values to fill in the template
        temperature: Model temperature (0.0-1.0)
       
    Returns:
        Model response as string
       
    Raises:
        ValueError: If API key is missing
        Exception: If API call fails
    """
    api_key = resolve_api_key()
   
    if not api_key:
        raise ValueError(
            "Missing Google API key. Set VITE_GOOGLE_API_KEY or REACT_APP_GOOGLE_API_KEY."
        )
   
    # Trim input values (matching JS behavior)
    trimmed_inputs = {}
    for key, value in (input_values or {}).items():
        if isinstance(value, str):
            trimmed_inputs[key] = value.strip()
        else:
            trimmed_inputs[key] = value
   
    # Create prompt template (matching JS version)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])
   
    # Create the model
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=api_key,
    )
   
    # Create chain
    chain = prompt_template | model | StrOutputParser()
   
    # Invoke
    response = chain.invoke(trimmed_inputs)
   
    if not response:
        raise ValueError("No response from AI.")
   
    return response


def analyze_performance_with_llm(error_report_json: str,
                                temperature: float = 0.35) -> str:
    """
    Analyze violin performance using LLM (Python equivalent of JS version).
   
    Args:
        error_report_json: JSON string or dict of error report
        temperature: Model temperature
       
    Returns:
        AI-generated analysis
       
    Raises:
        ValueError: If error_report is empty
    """
    # Handle both string and dict inputs
    if isinstance(error_report_json, dict):
        error_report_json = json.dumps(error_report_json, indent=2)
   
    trimmed_diff = (error_report_json or "").strip()
   
    if not trimmed_diff:
        raise ValueError("No diff JSON provided for analysis.")
   
    return invoke_gemini(
        system_prompt=VIOLON_SYSTEM_PROMPT,
        human_template=VIOLON_USER_PROMPT,
        input_values={"error_report_json": trimmed_diff},
        temperature=temperature,
    )


# Backward compatibility: keep old function name
def analyze_performance_with_ai(error_report: Dict[str, Any],
                               temperature: float = 0.35) -> str:
    """
    Analyze violin performance error report using Google Gemini.
   
    Args:
        error_report: The error report JSON/dict from the comparison pipeline
        temperature: Model temperature (0.0-1.0), lower = more deterministic
       
    Returns:
        AI-generated analysis as a string
    """
    return analyze_performance_with_llm(error_report, temperature)


def analyze_performance_batch(error_reports: list,
                             temperature: float = 0.35) -> list:
    """
    Analyze multiple error reports.
   
    Args:
        error_reports: List of error report dicts
        temperature: Model temperature
       
    Returns:
        List of AI-generated analyses
    """
    analyses = []
    for report in error_reports:
        try:
            analysis = analyze_performance_with_ai(report, temperature)
            analyses.append(analysis)
        except Exception as e:
            analyses.append(f"Error analyzing report: {str(e)}")
   
    return analyses


if __name__ == "__main__":
    # Test the API
    test_report = {
        "summary": {
            "n_reference_notes": 100,
            "n_student_notes_warped": 98,
            "n_matches": 95,
            "n_missing_reference": 3,
            "n_extra_student": 1,
            "n_significant": 2,
        },
        "error_report": [
            {
                "ground_truth_onset": 1.5,
                "student_error_type": "wrong_pitch",
                "midi_note": 72,
                "student_midi_note": 70,
                "description": "Played C5 instead of D5",
            },
            {
                "ground_truth_onset": 3.2,
                "student_error_type": "wrong_timing",
                "description": "Note started 0.1 seconds late",
            },
        ],
    }
   
    try:
        result = analyze_performance_with_ai(test_report)
        print("AI Analysis Result:")
        print("=" * 60)
        print(result)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo use this feature, set your Google API key:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
