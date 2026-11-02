"""LLM client: Meta Model API when MODEL_API_KEY is set, otherwise OpenRouter.

Any OpenAI-compatible endpoint works; swap base URL / models via env. Paid-call safety:
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
from collections.abc import Iterator
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ValidationError

META_BASE_URL = "https://api.meta.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Meta reports tokens, not cost. USD per 1M tokens: (input, cached input, output); images per image.
META_PRICES = {"muse-spark": (1.25, 0.15, 4.25), "muse-wonder": (4.00, 0.20, 20.00)}
META_IMAGE_USD = 0.01


class BudgetExceeded(RuntimeError):
    pass


def on_meta() -> bool:
    return bool(os.getenv("MODEL_API_KEY"))


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

    def add(self, usage, role: str, seconds: float, model: str = "") -> None:
        self.calls += 1
        self.seconds.setdefault(role, []).append(seconds)
        if usage is None:
            return
        self.tokens_in += usage.prompt_tokens or 0
        self.tokens_out += usage.completion_tokens or 0
        price = next((v for k, v in META_PRICES.items() if model.startswith(k)), None)
        if price:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            fresh = (usage.prompt_tokens or 0) - cached
            self.usd += (fresh * price[0] + cached * price[1] + (usage.completion_tokens or 0) * price[2]) / 1e6
        else:  # OpenRouter reports the billed amount on usage.cost
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
    def __init__(self, model: str, spend: Spend, cache_dir: Path, temperature: float, role: str,
                 effort: str | None = None):
        self.client = OpenAI(
            base_url=META_BASE_URL if on_meta() else os.getenv("CSR_VLM_BASE_URL", OPENROUTER_BASE_URL),
            api_key=os.environ["MODEL_API_KEY" if on_meta() else "CSR_VLM_API_KEY"],
            max_retries=4,  # backoff on 429 / 5xx / timeouts
            timeout=120,
        )
        self.model = model
        self.effort = effort  # Muse Spark reasoning_effort; low keeps per-turn calls fast
        self.spend = spend
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.role = role
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _complete(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.effort and on_meta():
            body["reasoning_effort"] = self.effort
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.spend.cached += 1
            return json.loads(path.read_text())["content"]
        self.spend.check()
        t0 = time.time()
        extra = {} if on_meta() else {"extra_body": {"usage": {"include": True}}}
        resp = self.client.chat.completions.create(**body, **extra)
        self.spend.add(resp.usage, self.role, time.time() - t0, self.model)
        content = resp.choices[0].message.content or ""
        path.write_text(json.dumps({"model": self.model, "content": content}, ensure_ascii=False))
        return content

    def stream(self, system: str, text: str) -> Iterator[str]:
        """Yield the reply as it is written. Stopping early skips the cache and the cost of that call."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.effort and on_meta():
            body["reasoning_effort"] = self.effort
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.spend.cached += 1
            yield json.loads(path.read_text())["content"]
            return
        self.spend.check()
        t0 = time.time()
        parts, usage = [], None
        # someone is waiting on this one: fail fast and let them send it again
        client = self.client.with_options(timeout=20, max_retries=1)
        for chunk in client.chat.completions.create(**body, stream=True, stream_options={"include_usage": True}):
            usage = chunk.usage or usage
            piece = chunk.choices[0].delta.content if chunk.choices else None
            if piece:
                parts.append(piece)
                yield piece
        self.spend.add(usage, self.role, time.time() - t0, self.model)
        path.write_text(json.dumps({"model": self.model, "content": "".join(parts)}, ensure_ascii=False))

    def stream_full(self, system: str, text: str) -> str:
        return "".join(self.stream(system, text))

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
    """Text-to-image, cached like text calls: Muse Image on Meta, an image-output chat model elsewhere."""
    import base64

    body = {"model": llm.model, "prompt": prompt, "aspect_ratio": aspect_ratio}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
    path = llm.cache_dir / f"{key}.img"
    if path.exists():
        llm.spend.cached += 1
        return path.read_bytes()
    llm.spend.check()
    t0 = time.time()
    if on_meta():
        resp = llm.client.images.generate(
            model=llm.model, prompt=prompt, n=1,
            size={"16:9": "1792x1024", "1:1": "1024x1024"}[aspect_ratio],
            extra_body={"output_format": "jpeg"},
        )
        llm.spend.add(None, llm.role, time.time() - t0)
        llm.spend.usd += META_IMAGE_USD
        data = base64.b64decode(resp.data[0].b64_json)
    else:
        resp = llm.client.chat.completions.create(
            model=llm.model, messages=[{"role": "user", "content": prompt}],
            extra_body={"modalities": ["image", "text"], "image_config": {"aspect_ratio": aspect_ratio},
                        "usage": {"include": True}},
        )
        llm.spend.add(resp.usage, llm.role, time.time() - t0)
        images = (resp.choices[0].message.model_extra or {}).get("images") or []
        if not images:
            raise RuntimeError(f"{llm.model} returned no image: {resp.choices[0].message.content!r:.300}")
        data = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    path.write_bytes(data)
    return data


def edit_image(llm: LLM, image: bytes, prompt: str) -> bytes:
    """Change an existing image with Muse Image, cached like text calls."""
    import base64

    key = hashlib.sha256(image + f"{llm.model}|{prompt}".encode()).hexdigest()[:24]
    path = llm.cache_dir / f"{key}.img"
    if path.exists():
        llm.spend.cached += 1
        return path.read_bytes()
    llm.spend.check()
    t0 = time.time()
    resp = llm.client.images.edit(model=llm.model, image=("image.jpg", image, "image/jpeg"), prompt=prompt, n=1)
    llm.spend.add(None, llm.role, time.time() - t0)
    llm.spend.usd += META_IMAGE_USD
    data = base64.b64decode(resp.data[0].b64_json)
    path.write_bytes(data)
    return data
