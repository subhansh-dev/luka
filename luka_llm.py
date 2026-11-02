"""Finnished OpenRouter import done
luka_llm.py — Drop-in LLM helper for all Luka modules.

OpenRouter → NVIDIA → Cerebras fallback implementation.

Usage (replaces per-file _generate):
    from luka_llm import generate, generate_json, get_client

    text = generate("Explain X")
    text = generate("Explain X", system="You are a scientist")
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "api_keys.json"

DEFAULT_MODEL = "nemotron-3-ultra-550b-a55b:free"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get_openrouter_key() -> str:
    cfg = _load_config()
    return cfg.get("openrouter_api_key", "")


def _get_nvidia_keys() -> list[str]:
    cfg = _load_config()
    return [cfg[k] for k in ["nvidia_api_key", "nvidia_api_key2"] if cfg.get(k)]


def _get_cerebras_keys() -> list[str]:
    cfg = _load_config()
    return [cfg[k] for k in ["cerebras_api_key", "cerebras_api_key2"] if cfg.get(k)]


# ── Singleton clients ──
_openrouter_client = None
_nvidia_key_idx = 0
_cerebras_key_idx = 0


def _get_openrouter_client():
    global _openrouter_client
    key = _get_openrouter_key()
    if not key:
        return None
    if _openrouter_client is None:
        _openrouter_client = {"key": key}
    return _openrouter_client


def _call_openrouter(prompt: str, model: str = DEFAULT_MODEL,
                     temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Call OpenRouter API."""
    try:
        import requests
    except ImportError:
        return ""
    cfg = _get_openrouter_client()
    if not cfg:
        return ""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {cfg['key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/subhansh-dev/Luka",
        "X-Title": "Luka",
    }
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
        return ""
    except Exception:
        return ""


def _call_nvidia(prompt: str, model: str = DEFAULT_MODEL,
                 temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Call NVIDIA NIM API with key rotation."""
    global _nvidia_key_idx
    try:
        import requests
    except ImportError:
        return ""
    keys = _get_nvidia_keys()
    if not keys:
        return ""

    key = keys[_nvidia_key_idx % len(keys)]
    _nvidia_key_idx += 1

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers, json=payload, timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content and content.strip():
                return content.strip()
        elif resp.status_code in (429, 401, 403):
            _nvidia_key_idx += 1  # rotate
        return ""
    except Exception:
        return ""


def _call_cerebras(prompt: str, model: str = "gpt-oss-120b",
                   temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Call Cerebras API with key rotation."""
    global _cerebras_key_idx
    try:
        import requests
    except ImportError:
        return ""
    keys = _get_cerebras_keys()
    if not keys:
        return ""

    key = keys[_cerebras_key_idx % len(keys)]
    _cerebras_key_idx += 1

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
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
                return content.strip()
            if reasoning and len(reasoning) > 2:
                return reasoning.strip()
        elif resp.status_code in (429, 401, 403):
            _cerebras_key_idx += 1  # rotate
        return ""
    except Exception:
        return ""


def generate(model_name: str = DEFAULT_MODEL, prompt: str = "",
             system: str = "", temperature: float = 0.3,
             max_tokens: int = 4096) -> str:
    """Drop-in LLM call: OpenRouter → NVIDIA → Cerebras."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    # OpenRouter (primary)
    result = _call_openrouter(full_prompt, model_name, temperature, max_tokens)
    if result:
        return result

    # NVIDIA (fallback)
    result = _call_nvidia(full_prompt, model_name, temperature, max_tokens)
    if result:
        return result

    # Cerebras (last resort)
    result = _call_cerebras(full_prompt, "gpt-oss-120b", temperature, max_tokens)
    if result:
        return result

    return ""


def generate_json(model_name: str = DEFAULT_MODEL, prompt: str = "",
                  system: str = "", temperature: float = 0.3,
                  max_tokens: int = 4096) -> str:
    """Generate with JSON mode hint."""
    json_prompt = prompt
    if "json" not in prompt.lower():
        json_prompt = prompt + "\n\nRespond in JSON format."
    return generate(model_name, json_prompt, system, temperature, max_tokens)


def get_client():
    """Return the config dict if available."""
    return _get_openrouter_client()


def get_api_key() -> str:
    """Return first available API key."""
    return _get_openrouter_key() or (_get_nvidia_keys() or [""])[0]


def is_available() -> bool:
    return bool(_get_openrouter_key()) or bool(_get_nvidia_keys()) or bool(_get_cerebras_keys())
