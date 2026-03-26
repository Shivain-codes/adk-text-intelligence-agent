"""
ADK Text Summarization Agent
Performs intelligent text summarization using Google Gemini via ADK framework.
"""

import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# ── Tool Definition ────────────────────────────────────────────────────────────

def summarize_text(text: str, style: str = "concise") -> dict:
    """
    Summarizes the provided text using Gemini.

    Args:
        text:  The input text to summarize (required).
        style: Summary style – 'concise' (2-3 sentences),
               'detailed' (paragraph), or 'bullets' (bullet points).

    Returns:
        A dict with 'summary', 'word_count_original', 'word_count_summary',
        and 'compression_ratio'.
    """
    if not text or not text.strip():
        return {"error": "No text provided to summarize."}

    valid_styles = {"concise", "detailed", "bullets"}
    style = style.lower()
    if style not in valid_styles:
        style = "concise"

    original_words = len(text.split())

    # Return the structured request to Gemini; ADK handles actual inference.
    return {
        "input_text": text,
        "requested_style": style,
        "original_word_count": original_words,
        "instruction": (
            f"Please summarize the following text in a '{style}' style. "
            f"For 'concise' use 2-3 sentences. "
            f"For 'detailed' write a full paragraph. "
            f"For 'bullets' use bullet points. "
            f"Return ONLY the summary, nothing else.\n\nText:\n{text}"
        ),
    }


def classify_text(text: str) -> dict:
    """
    Classifies the topic/domain of the provided text.

    Args:
        text: The input text to classify.

    Returns:
        A dict with 'category', 'confidence', and 'reasoning'.
    """
    if not text or not text.strip():
        return {"error": "No text provided to classify."}

    return {
        "input_text": text,
        "instruction": (
            "Classify the following text into one of these categories: "
            "Technology, Science, Business, Politics, Sports, Entertainment, "
            "Health, Education, Other. "
            "Return a JSON object with keys: 'category', 'confidence' (high/medium/low), "
            "and 'reasoning' (one sentence). Return ONLY the JSON.\n\nText:\n" + text
        ),
    }


# ── Agent Definition ───────────────────────────────────────────────────────────

root_agent = Agent(
    name="text_intelligence_agent",
    model=LiteLlm(model="gemini/gemini-2.0-flash"),
    description=(
        "An AI agent that summarizes text and classifies its topic. "
        "Provide any text and choose a summary style: concise, detailed, or bullets."
    ),
    instruction="""
You are an expert Text Intelligence Agent powered by Google Gemini.

Your two capabilities:
1. **Summarize** – Condense text while preserving key information.
   Styles: concise (2-3 sentences), detailed (full paragraph), bullets (bullet points).
2. **Classify** – Identify the topic/domain of a given text.

When the user sends text to summarize:
- Call the `summarize_text` tool with the text and desired style.
- Use the 'instruction' field from the tool result to generate the actual summary.
- Report the compression ratio (original vs summary word count).

When the user wants text classified:
- Call the `classify_text` tool.
- Return the category, confidence, and reasoning clearly.

Always be concise, accurate, and helpful.
If no style is specified for summarization, default to 'concise'.
""",
    tools=[summarize_text, classify_text],
)
