"""Finnished OpenRouter import done
discovery/llm_client.py — Unified LLM Client for Luka

OpenRouter (primary) → NVIDIA NIM (2 keys) → Cerebras (2 keys) fallback.
Drop-in replacement for all LLM calls.

Usage:
    from discovery.llm_client import call, call_json, is_available

    text = call("Explain phosphine detection on Venus")
    data = call_json("Generate hypotheses about X", max_tokens=8192)
"""

import json
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"

# ── Defaults ──
OPENROUTER_MODEL = "nemotron-3-ultra-550b-a55b:free"
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
CEREBRAS_MODEL = "gpt-oss-120b"

# ── Rate limiting ──
_last_openrouter_call = 0.0
_last_nvidia_call = 0.0
_last_cerebras_call = 0.0

OPENROUTER_MIN_INTERVAL = 2.0
NVIDIA_MIN_INTERVAL = 3.0
CEREBRAS_MIN_INTERVAL = 2.0


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ═══════════════════════════════════════════════════════════════
# OpenRouter — Primary
# ═══════════════════════════════════════════════════════════════

def _get_openrouter_config() -> dict:
    cfg = _load_config()
    key = cfg.get("openrouter_api_key", "")
    model = cfg.get("openrouter_model", OPENROUTER_MODEL)
    if not key:
        return {}
    return {"key": key, "model": model}


def _rate_limit_openrouter():
    global _last_openrouter_call
    now = time.time()
    elapsed = now - _last_openrouter_call
    if elapsed < OPENROUTER_MIN_INTERVAL:
        time.sleep(OPENROUTER_MIN_INTERVAL - elapsed)
    _last_openrouter_call = time.time()


def _call_openrouter(prompt: str, json_mode: bool = False,
                     max_tokens: int = 4096, temperature: float = 0.3) -> str | None:
    """Call OpenRouter API. OpenAI-compatible endpoint."""
    try:
        import requests
    except ImportError:
        return None
    config = _get_openrouter_config()
    if not config:
        return None

    key = config["key"]
    model = config["model"]

    effective_prompt = prompt
    if json_mode and "json" not in prompt.lower():
        effective_prompt = prompt + "\n\nRespond in JSON format."

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": effective_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/subhansh-dev/Luka",
        "X-Title": "Luka Discovery Engine",
    }

    for attempt in range(3):
        _rate_limit_openrouter()
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content and content.strip():
                    return content.strip()
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            elif resp.status_code == 400 and json_mode:
                # json_object not supported — strip and retry
                payload.pop("response_format", None)
                resp2 = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=payload, timeout=120,
                )
                if resp2.status_code == 200:
                    data = resp2.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content and content.strip():
                        return content.strip()
                return None
            elif resp.status_code in (401, 403):
                return None  # bad key
            else:
                print(f"    [LLM] OpenRouter {resp.status_code}: {resp.text[:100]}", flush=True)
                return None
        except requests.exceptions.Timeout:
            continue
        except Exception:
            return None

    return None


# ═══════════════════════════════════════════════════════════════
# NVIDIA NIM — Fallback (2 keys, key rotation)
# ═══════════════════════════════════════════════════════════════

_nvidia_key_idx = 0


def _get_nvidia_keys() -> list[str]:
    cfg = _load_config()
    keys = []
    for k in ["nvidia_api_key", "nvidia_api_key2"]:
        v = cfg.get(k, "")
        if v:
            keys.append(v)
    return keys


def _get_nvidia_config() -> dict:
    cfg = _load_config()
    model = cfg.get("nvidia_model", NVIDIA_MODEL)
    return {"model": model}


def _rate_limit_nvidia():
    global _last_nvidia_call
    now = time.time()
    elapsed = now - _last_nvidia_call
    if elapsed < NVIDIA_MIN_INTERVAL:
        time.sleep(NVIDIA_MIN_INTERVAL - elapsed)
    _last_nvidia_call = time.time()


def _call_nvidia(prompt: str, json_mode: bool = False,
                 max_tokens: int = 4096, temperature: float = 0.3) -> str | None:
    """Call NVIDIA NIM API. Rotates keys on 429/401."""
    try:
        import requests
    except ImportError:
        return None
    global _nvidia_key_idx
    keys = _get_nvidia_keys()
    if not keys:
        return None
    model = _get_nvidia_config().get("model", NVIDIA_MODEL)

    effective_prompt = prompt
    if json_mode and "json" not in prompt.lower():
        effective_prompt = prompt + "\n\nRespond in JSON format."

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": effective_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for key_attempt in range(len(keys)):
        key = keys[(_nvidia_key_idx + key_attempt) % len(keys)]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        for attempt in range(2):
            _rate_limit_nvidia()
            try:
                resp = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers, json=payload, timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content and content.strip():
                        _nvidia_key_idx = (_nvidia_key_idx + key_attempt) % len(keys)
                        return content.strip()
                elif resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    break  # rotate key
                elif resp.status_code in (401, 403):
                    break  # dead key, rotate
                elif resp.status_code == 400 and json_mode:
                    payload.pop("response_format", None)
                    resp2 = requests.post(
                        "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers, json=payload, timeout=120,
                    )
                    if resp2.status_code == 200:
                        data = resp2.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if content and content.strip():
                            _nvidia_key_idx = (_nvidia_key_idx + key_attempt) % len(keys)
                            return content.strip()
                    return None
                else:
                    print(f"    [LLM] NVIDIA {resp.status_code}: {resp.text[:100]}", flush=True)
                    return None
            except requests.exceptions.Timeout:
                continue
            except Exception:
                return None

    _nvidia_key_idx = (_nvidia_key_idx + 1) % max(1, len(keys))
    return None


# ═══════════════════════════════════════════════════════════════
# Cerebras — Last resort (2 keys, key rotation)
# ═══════════════════════════════════════════════════════════════

_cerebras_key_idx = 0


def _get_cerebras_keys() -> list[str]:
    cfg = _load_config()
    keys = []
    for k in ["cerebras_api_key", "cerebras_api_key2"]:
        v = cfg.get(k, "")
        if v:
            keys.append(v)
    return keys


def _call_cerebras(prompt: str, json_mode: bool = False,
                   max_tokens: int = 4096, temperature: float = 0.3,
                   model: str = CEREBRAS_MODEL) -> str | None:
    """Call Cerebras API. No rate limiting, key rotation on 429."""
    try:
        import requests
    except ImportError:
        return None
    global _cerebras_key_idx
    keys = _get_cerebras_keys()
    if not keys:
        return None

    effective_prompt = prompt
    if json_mode and "json" not in prompt.lower():
        effective_prompt = prompt + "\n\nRespond in JSON format."

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": effective_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for key_attempt in range(len(keys)):
        key = keys[(_cerebras_key_idx + key_attempt) % len(keys)]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers=headers, json=payload, timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                choice = data.get("choices", [{}])[0].get("message", {})
                content = choice.get("content", "")
                reasoning = choice.get("reasoning", "")
                if content and content.strip():
                    _cerebras_key_idx = (_cerebras_key_idx + key_attempt) % len(keys)
                    return content.strip()
                if reasoning and len(reasoning) > 2:
                    _cerebras_key_idx = (_cerebras_key_idx + key_attempt) % len(keys)
                    return reasoning.strip()
            elif resp.status_code == 429:
                time.sleep(5)
                continue
            elif resp.status_code in (401, 403):
                continue
            elif resp.status_code == 400 and json_mode:
                payload.pop("response_format", None)
                resp2 = requests.post(
                    "https://api.cerebras.ai/v1/chat/completions",
                headers=headers, json=payload, timeout=60,
                )
                if resp2.status_code == 200:
                    data = resp2.json()
                    choice = data.get("choices", [{}])[0].get("message", {})
                    content = choice.get("content", "")
                    reasoning = choice.get("reasoning", "")
                    if content and content.strip():
                        _cerebras_key_idx = (_cerebras_key_idx + key_attempt) % len(keys)
                        return content.strip()
                    if reasoning and len(reasoning) > 2:
                        _cerebras_key_idx = (_cerebras_key_idx + key_attempt) % len(keys)
                        return reasoning.strip()
            else:
                print(f"    [LLM] Cerebras {resp.status_code}: {resp.text[:100]}", flush=True)
                return None
        except Exception:
            continue

    _cerebras_key_idx = (_cerebras_key_idx + 1) % max(1, len(keys))
    return None


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

_openrouter_cooldown_until = 0.0
_nvidia_cooldown_until = 0.0
_cerebras_cooldown_until = 0.0
COOLDOWN_SECONDS = 15


def call(prompt: str, json_mode: bool = False, max_tokens: int = 4096,
         temperature: float = 0.3, provider: str = "auto") -> str | None:
    """Call with auto-routing: OpenRouter → NVIDIA (2 keys) → Cerebras (2 keys)."""
    global _openrouter_cooldown_until, _nvidia_cooldown_until, _cerebras_cooldown_until

    if provider in ("auto", "openrouter"):
        for chain_attempt in range(3):
            now = time.time()
            # OpenRouter (primary — fastest, most reliable)
            if now >= _openrouter_cooldown_until:
                result = _call_openrouter(prompt, json_mode, max_tokens, temperature)
                if result:
                    return result
                _openrouter_cooldown_until = time.time() + COOLDOWN_SECONDS
            # NVIDIA (fallback, 2 keys with rotation)
            if now >= _nvidia_cooldown_until:
                result = _call_nvidia(prompt, json_mode, max_tokens, temperature)
                if result:
                    return result
                _nvidia_cooldown_until = time.time() + COOLDOWN_SECONDS
            # Cerebras (last resort, 2 keys with rotation)
            if now >= _cerebras_cooldown_until:
                result = _call_cerebras(prompt, json_mode, max_tokens, temperature)
                if result:
                    return result
                _cerebras_cooldown_until = time.time() + COOLDOWN_SECONDS

            if chain_attempt < 2:
                soonest = min(_openrouter_cooldown_until, _nvidia_cooldown_until, _cerebras_cooldown_until)
                wait = max(0, soonest - time.time())
                if 0 < wait < 60:
                    print(f"    [LLM] All providers failed, retrying in {wait:.0f}s "
                          f"(attempt {chain_attempt + 1}/3)...", flush=True)
                    time.sleep(wait + 1)
        return None

    if provider == "nvidia":
        result = _call_nvidia(prompt, json_mode, max_tokens, temperature)
        if result:
            return result
        _nvidia_cooldown_until = time.time() + COOLDOWN_SECONDS
        return _call_cerebras(prompt, json_mode, max_tokens, temperature)

    if provider == "cerebras":
        return _call_cerebras(prompt, json_mode, max_tokens, temperature)

    return None


def call_json(prompt: str, max_tokens: int = 8192,
              temperature: float = 0.3, provider: str = "auto") -> str | None:
    """Call with JSON response mode."""
    return call(prompt, json_mode=True, max_tokens=max_tokens,
                temperature=temperature, provider=provider)


def call_math(prompt: str, max_tokens: int = 8192,
              temperature: float = 0.3, json_mode: bool = False) -> str | None:
    """Math-optimized call — routes through OpenRouter (best model for reasoning)."""
    return call(prompt, json_mode=json_mode, max_tokens=max_tokens,
                temperature=temperature, provider="openrouter")


def call_thinking(prompt: str, max_tokens: int = 32768,
                  temperature: float = 0.7, provider: str = "auto") -> str | None:
    """High-token call for complex reasoning tasks."""
    return call(prompt, False, max_tokens, temperature,
                provider="openrouter" if provider == "auto" else provider)


def is_available(provider: str = "auto") -> bool:
    """Check if any LLM provider is configured."""
    if provider == "openrouter":
        return bool(_get_openrouter_config())
    if provider == "nvidia":
        return bool(_get_nvidia_keys())
    if provider == "cerebras":
        return bool(_get_cerebras_keys())
    return (bool(_get_openrouter_config()) or
            bool(_get_nvidia_keys()) or
            bool(_get_cerebras_keys()))


def get_status() -> dict:
    """Return provider availability status."""
    openrouter_config = _get_openrouter_config()
    nvidia_keys = _get_nvidia_keys()
    cerebras_keys = _get_cerebras_keys()
    nvidia_cfg = _get_nvidia_config()
    primary = ("openrouter" if openrouter_config else
               "nvidia" if nvidia_keys else
               "cerebras" if cerebras_keys else "none")
    return {
        "openrouter": (
            {"available": True, "model": openrouter_config.get("model", "N/A")}
            if openrouter_config else {"available": False}
        ),
        "nvidia": (
            {"available": bool(nvidia_keys), "keys": len(nvidia_keys),
             "model": nvidia_cfg.get("model", "N/A")}
            if nvidia_keys else {"available": False}
        ),
        "cerebras": (
            {"available": bool(cerebras_keys), "keys": len(cerebras_keys),
             "model": CEREBRAS_MODEL}
            if cerebras_keys else {"available": False}
        ),
        "primary": primary,
    }
