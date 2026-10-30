"""LLM client (stand-in for Muse Spark until the Meta Model API is available).

Any OpenAI-compatible endpoint works (OpenRouter by default); swap base URL /
models via env. Paid-call safety:
- every response is cached on disk by request hash, so a rerun replays
  finished calls for free;
- replies are validated against a pydantic schema and re-asked on failure;
- spend is tracked and the run aborts once it passes the budget;
- wall time of each paid call is recorded, since latency is a product metric.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ValidationError

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text


class Spend:
    """Running total shared by every client in a run."""

    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self.usd = 0.0
        self.calls = 0
        self.cached = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds: dict[str, list[float]] = {}  # role -> wall time of each paid call

    def add(self, usage, role: str, seconds: float) -> None:
        self.calls += 1
        self.seconds.setdefault(role, []).append(seconds)
        if usage is None:
            return
        self.tokens_in += usage.prompt_tokens or 0
        self.tokens_out += usage.completion_tokens or 0
        # OpenRouter reports the billed amount on usage.cost
        self.usd += float(getattr(usage, "cost", None) or 0.0)

    def check(self) -> None:
        if self.usd > self.budget:
            raise BudgetExceeded(f"spent ${self.usd:.4f} > budget ${self.budget:.2f}; rerun reuses the cache")

    def latency(self) -> dict[str, str]:
        return {r: f"avg {sum(s) / len(s):.1f}s, max {max(s):.1f}s over {len(s)}" for r, s in self.seconds.items()}

    def summary(self) -> str:
        return (f"${self.usd:.4f} over {self.calls} paid calls ({self.cached} cached), "
                f"{self.tokens_in} in / {self.tokens_out} out tokens")


class LLM:
    def __init__(self, model: str, spend: Spend, cache_dir: Path, temperature: float, role: str):
        self.client = OpenAI(
            base_url=os.getenv("CSR_VLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ["CSR_VLM_API_KEY"],
            max_retries=4,  # backoff on 429 / 5xx / timeouts
            timeout=120,
        )
        self.model = model
        self.spend = spend
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.role = role
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _complete(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.spend.cached += 1
            return json.loads(path.read_text())["content"]
        self.spend.check()
        t0 = time.time()
        resp = self.client.chat.completions.create(**body, extra_body={"usage": {"include": True}})
        self.spend.add(resp.usage, self.role, time.time() - t0)
        content = resp.choices[0].message.content or ""
        path.write_text(json.dumps({"model": self.model, "content": content}, ensure_ascii=False))
        return content

    def ask[T: BaseModel](self, schema: type[T], system: str, text: str, retries: int = 2) -> T:
        """Ask for JSON matching `schema`; re-ask with the validation error on failure."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        err = ""
        for _ in range(retries + 1):
            raw = self._complete(messages)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except ValidationError as e:
                err = str(e)[:1500]
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Your JSON did not match the required shape:\n{err}\n"
                                                "Reply with only the corrected JSON object."},
                ]
        raise RuntimeError(f"{self.model} did not return valid {schema.__name__}: {err}")


class Ping(BaseModel):
    answer: int


def preflight(llm: LLM) -> None:
    """Cheap paid call proving the key and model work before a long run."""
    got = llm.ask(Ping, "Reply with a single JSON object.", 'What is 17 + 25? Return {"answer": <number>}.')
    if got.answer != 42:
        raise RuntimeError(f"{llm.model} failed the preflight (answered {got.answer})")


def generate_image(llm: LLM, prompt: str, aspect_ratio: str = "1:1") -> bytes:
    """Text-to-image via an image-output chat model (stand-in for Muse Image). Cached like text calls."""
    import base64

    body = {"model": llm.model, "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"], "image_config": {"aspect_ratio": aspect_ratio}}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
    path = llm.cache_dir / f"{key}.img"
    if path.exists():
        llm.spend.cached += 1
        return path.read_bytes()
    llm.spend.check()
    t0 = time.time()
    resp = llm.client.chat.completions.create(
        model=llm.model, messages=body["messages"],
        extra_body={"modalities": body["modalities"], "image_config": body["image_config"],
                    "usage": {"include": True}},
    )
    llm.spend.add(resp.usage, llm.role, time.time() - t0)
    images = (resp.choices[0].message.model_extra or {}).get("images") or []
    if not images:
        raise RuntimeError(f"{llm.model} returned no image: {resp.choices[0].message.content!r:.300}")
    data = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    path.write_bytes(data)
    return data
