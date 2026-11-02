"""discovery/json_extract.py — Robust JSON extraction from LLM responses.

Handles: markdown code blocks, LaTeX braces, prose wrapping, trailing commas.
"""

import json
import re


def extract_json(raw: str, expected_key: str = None) -> dict | None:
    """Extract JSON from an LLM response that may contain prose, markdown, or LaTeX.

    Args:
        raw: The LLM response string
        expected_key: If set, search for {"expected_key": ...} first (e.g., "theories")

    Returns:
        Parsed dict or None if extraction fails
    """
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()

    # Strip markdown code blocks (handle ```json ... ``` and ``` ... ```)
    if "```" in raw:
        # Find first ``` and last ```
        first = raw.find("```")
        last = raw.rfind("```")
        if first != last:
            # Extract content between first and last ```
            inner = raw[first + 3:last].strip()
            # Remove language tag (json, python, etc.)
            if inner.startswith("json"):
                inner = inner[4:].strip()
            elif inner.startswith("python"):
                inner = inner[6:].strip()
            raw = inner

    # Find JSON start
    json_start = None

    # Method 1: Search for expected key (e.g., {"theories": ...})
    if expected_key:
        match = re.search(r'\{\s*"' + re.escape(expected_key) + '"', raw)
        if match:
            json_start = match.start()

    # Method 2: Find first { followed by a quote (skip LaTeX/math braces)
    if json_start is None:
        for i, ch in enumerate(raw):
            if ch == '{':
                rest = raw[i + 1:].lstrip()
                if rest and rest[0] == '"':
                    json_start = i
                    break

    if json_start is None:
        return None

    # Find matching closing brace (depth tracking)
    def _find_json_end(text, start):
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
        return None

    json_end = _find_json_end(raw, json_start)

    if json_end is None:
        return None

    json_str = raw[json_start:json_end]

    # Try parsing with progressive fixes
    for attempt in range(4):
        try:
            result = json.loads(json_str)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError as e:
            if attempt == 0:
                # Fix trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
            elif attempt == 1:
                # Fix unescaped newlines in strings
                json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            elif attempt == 2:
                # Try extracting just the expected key's value
                if expected_key:
                    arr_match = re.search(
                        r'"' + re.escape(expected_key) + r'"\s*:\s*(\[[\s\S]*\])',
                        json_str
                    )
                    if arr_match:
                        try:
                            arr = json.loads(arr_match.group(1))
                            return {expected_key: arr}
                        except json.JSONDecodeError:
                            pass
                # Try finding ANY valid JSON object in the string
                for m in re.finditer(r'\{[^{}]*\}', json_str):
                    try:
                        result = json.loads(m.group())
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except json.JSONDecodeError:
                        continue

    # Final fallback: extract JSON array directly for expected_key
    if expected_key:
        arr_pattern = r'"' + re.escape(expected_key) + r'"\s*:\s*\['
        arr_match = re.search(arr_pattern, raw)
        if arr_match:
            arr_start = arr_match.end() - 1
            depth = 0
            in_str = False
            esc = False
            for i in range(arr_start, len(raw)):
                ch = raw[i]
                if esc:
                    esc = False
                    continue
                if ch == '\\':
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        arr_str = raw[arr_start:i + 1]
                        for fix_pass in range(3):
                            try:
                                arr = json.loads(arr_str)
                                return {expected_key: arr}
                            except json.JSONDecodeError:
                                arr_str = re.sub(r',\s*}', '}', arr_str)
                                arr_str = re.sub(r',\s*]', ']', arr_str)
                        break

    return None
