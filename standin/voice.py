"""Natural speech for the cast through OpenRouter (Meta Model API has no text-to-speech).

Each sentence is requested on its own, so the stage can start the first one while
the rest of the line is still being written. Audio is cached on disk by request.
"""

import hashlib
import os
import time
from pathlib import Path

from openai import OpenAI

from .llm import OPENROUTER_BASE_URL, Spend
from .spec import Character, Scenario

MODEL = "elevenlabs/eleven-v4-turbo"
USD_PER_CHAR = 40 / 1_000_000
VOICES = {"m": ["george", "brian", "daniel", "adam"], "f": ["sarah", "alice", "bella"]}


def available() -> bool:
    return bool(os.getenv("CSR_VLM_API_KEY"))


def voice_for(s: Scenario, c: Character) -> str:
    """Different voice per person; people of the same gender take turns through the pool."""
    gender = c.gender if c.gender in VOICES else "m"
    same = [o.name for o in s.cast if (o.gender if o.gender in VOICES else "m") == gender]
    return VOICES[gender][same.index(c.name) % len(VOICES[gender])]


def synthesize(text: str, voice: str, spend: Spend, cache_dir: Path) -> bytes:
    model = os.getenv("STANDIN_TTS_MODEL", MODEL)
    key = hashlib.sha256(f"{model}|{voice}|{text}".encode()).hexdigest()[:24]
    path = cache_dir / f"{key}.mp3"
    if path.exists():
        spend.cached += 1
        return path.read_bytes()
    spend.check()
    t0 = time.time()
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["CSR_VLM_API_KEY"], max_retries=2, timeout=30)
    data = client.audio.speech.create(model=model, voice=voice, input=text, response_format="mp3").content
    spend.add(None, "voice", time.time() - t0)
    spend.usd += len(text) * USD_PER_CHAR
    path.write_bytes(data)
    return data
