import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def generate_feedback(diff_json: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
You are a violin instructor. Analyze the following MIDI comparison data and provide
detailed practice suggestions. Here is the data:

{diff_json}
"""
    response = model.generate_content(prompt)
    return response.text
