"""
Minimal Groq chat client with an on-disk response cache -- Phase B, Day 7.

Groq's API is OpenAI-compatible, so this is one `requests.post` rather than a new SDK
dependency: fewer moving parts, and a fake `session` is trivial to inject in tests.

CACHING (Section 5's stated mitigation for free-tier rate limits): every successful
response is stored on disk keyed by a hash of (model, messages, temperature, max_tokens).
Re-running the eval, changing only the NLI threshold or the scoring script, or resuming
after a rate-limit stall costs zero API calls. Cache hits do not need an API key.
Failures are never cached.

RATE LIMITS: HTTP 429 waits for the server's `Retry-After` header when present (else
exponential backoff), 5xx and network errors back off exponentially, other 4xx fail
immediately with the response body (a bad key or a bad model name should be loud, not
retried six times).

Caveat: temperature=0 makes Groq output near-deterministic, not guaranteed identical across
calls. The cache is what makes a *reported* run exactly reproducible -- commit-worthy
results should be re-scored from the cached answers, not regenerated.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# The plan names Llama 3.3 70B, but a model list pulled with this project's own key on Oct 4 did not include
# it (Groq's public docs list models an individual account may not have). openai/gpt-oss-120b is the largest
# model that key could use -- a documented deviation, see CHANGES.md. Pass --model to the scripts to override.
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_TOKENS = 1500  # gpt-oss is a reasoning model; the cap must leave room beyond its hidden reasoning


def build_payload(messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Dict:
    """Request body, also the cache key. gpt-oss models get low reasoning effort and no returned reasoning
    text: this is retrieval-grounded extraction, not a puzzle, and reasoning tokens eat the token cap."""
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"
        payload["include_reasoning"] = False
    return payload


class GroqError(RuntimeError):
    """status_code is set for HTTP errors that are NOT worth retrying (bad key, unknown model,
    bad request) so callers can stop a whole run instead of repeating the same failure."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class ResponseCache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(payload: Dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict]:
        path = self.dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt cache file is a miss, not a crash

    def put(self, key: str, value: Dict) -> None:
        path = self.dir / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: an interrupted run never leaves a half-written entry


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, cache: Optional[ResponseCache] = None,
                 session=None, max_retries: int = 6, sleep_fn: Callable[[float], None] = time.sleep,
                 timeout: float = 90.0):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.cache = cache
        self.session = session or requests
        self.max_retries = max_retries
        self.sleep_fn = sleep_fn
        self.timeout = timeout

    def chat(self, messages: List[Dict], model: str = DEFAULT_MODEL,
             temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict:
        payload = build_payload(messages, model, temperature, max_tokens)
        key = ResponseCache.make_key(payload)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return {**hit, "cached": True}

        if not self.api_key:
            raise GroqError("GROQ_API_KEY is not set (and this request is not in the cache). "
                            "Put GROQ_API_KEY=... in a .env file at the repo root, or set the env var.")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_problem = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(GROQ_URL, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_problem = f"network error: {exc}"
                if attempt < self.max_retries:          # no pointless sleep after the final attempt
                    self.sleep_fn(min(2 ** attempt, 60))
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    choice = data["choices"][0]
                    text = choice["message"].get("content") or ""
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise GroqError(f"Groq returned a malformed 200 response ({type(exc).__name__}): "
                                    f"{str(getattr(resp, 'text', ''))[:300]}") from exc
                if not text.strip():
                    # A reasoning model that spent its whole token cap thinking returns empty content. That must
                    # never be cached or mistaken for a refusal -- fail loudly instead.
                    raise GroqError(
                        f"Groq returned an empty answer (finish_reason="
                        f"{choice.get('finish_reason')}); the model likely exhausted max_tokens="
                        f"{max_tokens} on reasoning. Re-run with a larger --max-tokens.", status_code=400)
                result = {"text": text, "usage": data.get("usage", {}), "model": data.get("model", model)}
                if self.cache is not None:
                    self.cache.put(key, result)
                return {**result, "cached": False}

            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                try:
                    wait = min(float(retry_after), 120.0)
                except (TypeError, ValueError):
                    wait = min(2 ** attempt, 60)
                last_problem = f"429 rate limited (waiting {wait:.1f}s)"
                if attempt < self.max_retries:
                    self.sleep_fn(wait)
                continue

            if resp.status_code >= 500:
                last_problem = f"{resp.status_code} server error"
                if attempt < self.max_retries:
                    self.sleep_fn(min(2 ** attempt, 60))
                continue

            raise GroqError(f"Groq API error {resp.status_code}: {resp.text[:500]}", status_code=resp.status_code)

        raise GroqError(f"Groq request failed after {self.max_retries + 1} attempts ({last_problem})")
