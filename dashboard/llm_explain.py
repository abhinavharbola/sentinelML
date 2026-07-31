import os
import sys
from pathlib import Path

from groq import Groq

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def build_prompt(event_type, details):
    return (
        "You explain automated MLOps pipeline decisions to a non-technical reader, in one paragraph. "
        "Use only the data given below, do not speculate beyond it.\n\n"
        f"Event type: {event_type}\n"
        f"Event data: {details}\n\n"
        "Explain plainly why the pipeline acted this way."
    )


def explain_event(event_type, details):
    client = get_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": build_prompt(event_type, details)}],
        temperature=0.3,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()