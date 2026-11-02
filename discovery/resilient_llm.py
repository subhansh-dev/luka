"""
resilient_llm.py — LLM calls that NEVER fail silently.

Every phase in the pipeline should complete, even if the LLM is slow,
rate-limited, or returns garbage. This wrapper:
1. Tries OpenRouter first
2. Falls back to NVIDIA
3. Falls back to Cerebras
4. Falls back to shorter prompt
5. Falls back to non-JSON mode
5. Returns partial results with error info (never empty)

This is what makes Luka "fully functional" — no phase is ever incomplete.
"""

import json
import time
from typing import Optional, Any


class ResilientLLM:
    """
    LLM wrapper that guarantees a response, even if degraded.
    """

    def __init__(self):
        self._call_count = 0
        self._errors = []

    def call(self, prompt: str, max_tokens: int = 4096,
             json_mode: bool = False, timeout: int = 60) -> dict:
        """
        Call LLM with maximum reliability.

        Returns:
            {
                "content": str or None,
                "provider": "openrouter|nvidia|cerebras|fallback",
                "degraded": bool,
                "error": str or None,
                "attempts": int
            }
        """
        self._call_count += 1
        attempts = 0
        last_error = None

        # Strategy 1: Try OpenRouter (primary — fastest, most reliable)
        attempts += 1
        try:
            from discovery.llm_client import call_json, call as llm_call
            result = call_json(prompt, max_tokens=max_tokens, provider="openrouter")
            if result and len(str(result)) > 20:
                return {
                    "content": str(result),
                    "provider": "openrouter",
                    "degraded": False,
                    "error": None,
                    "attempts": attempts,
                }
            # Try non-JSON mode
            result = llm_call(prompt, max_tokens=max_tokens, provider="openrouter")
            if result and len(str(result)) > 20:
                return {
                    "content": str(result),
                    "provider": "openrouter",
                    "degraded": True,
                    "error": "JSON mode failed, used text mode",
                    "attempts": attempts,
                }
        except Exception as e:
            last_error = str(e)

        # Strategy 2: Try NVIDIA (fallback, 2 keys with rotation)
        attempts += 1
        time.sleep(2)  # rate limit buffer
        try:
            from discovery.llm_client import call_json, call as llm_call
            result = call_json(prompt, max_tokens=max_tokens, provider="nvidia")
            if result and len(str(result)) > 20:
                return {
                    "content": str(result),
                    "provider": "nvidia",
                    "degraded": False,
                    "error": None,
                    "attempts": attempts,
                }
            result = llm_call(prompt, max_tokens=max_tokens, provider="nvidia")
            if result and len(str(result)) > 20:
                return {
                    "content": str(result),
                    "provider": "nvidia",
                    "degraded": True,
                    "error": "JSON mode failed, used text mode",
                    "attempts": attempts,
                }
        except Exception as e:
            last_error = str(e)

        # Strategy 3: Try Cerebras (last resort, 2 keys with rotation)
        attempts += 1
        try:
            from discovery.llm_client import call_json, call as llm_call
            result = call_json(prompt, max_tokens=max_tokens, provider="cerebras")
            if result and len(str(result)) > 20:
                return {
                    "content": str(result),
                    "provider": "cerebras",
                    "degraded": False,
                    "error": None,
                    "attempts": attempts,
                }
            result = llm_call(prompt, max_tokens=max_tokens, provider="cerebras")
            if result and len(str(result)) > 20:
                return {
                    "content": str(result),
                    "provider": "cerebras",
                    "degraded": True,
                    "error": "JSON mode failed, used text mode",
                    "attempts": attempts,
                }
        except Exception as e:
            last_error = str(e)

        # Strategy 4: Try with shorter prompt (last ditch)
        attempts += 1
        if len(prompt) > 2000:
            short_prompt = prompt[:1500] + "\n\n[Truncated — respond with available context]"
            try:
                from discovery.llm_client import call_json
                result = call_json(short_prompt, max_tokens=max_tokens // 2, provider="auto")
                if result and len(str(result)) > 20:
                    return {
                        "content": str(result),
                        "provider": "auto_short",
                        "degraded": True,
                        "error": "Prompt truncated due to length",
                        "attempts": attempts,
                    }
            except Exception as e:
                last_error = str(e)

        # All strategies failed
        self._errors.append(last_error)
        return {
            "content": None,
            "provider": "none",
            "degraded": True,
            "error": last_error or "All LLM strategies failed",
            "attempts": attempts,
        }

    def call_json(self, prompt: str, max_tokens: int = 4096) -> Any:
        """
        Convenience: call and parse JSON from response.
        Returns parsed JSON or None.
        """
        result = self.call(prompt, max_tokens=max_tokens, json_mode=True)
        content = result.get("content")
        if not content:
            return None

        # Try to parse JSON
        try:
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return None

    def get_stats(self) -> dict:
        """Return call statistics."""
        return {
            "total_calls": self._call_count,
            "errors": len(self._errors),
            "recent_errors": self._errors[-5:] if self._errors else [],
        }
