import json

def generate_ai_feedback(diffJson):
    """
    Placeholder AI feedback — replace with Gemini/OpenAI call later.
    """
    try:
        data = json.loads(diffJson)
    except:
        return "Unable to parse diffJson."

    stats = data.get("stats", {})

    response = "🎻 **Violon AI Feedback**\n\n"
    response += f"• Overall Score: {stats.get('overallScore', 'N/A')}%\n"
    response += f"• Excellent: {stats.get('goodPercent', 'N/A')}%\n"
    response += f"• Moderate: {stats.get('moderatePercent', 'N/A')}%\n"
    response += f"• Needs Work: {stats.get('poorPercent', 'N/A')}%\n\n"
    response += "Focus practice on red-highlighted regions where timing or pitch deviates most."

    return response
