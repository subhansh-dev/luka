"""TNS - Supernova/transient alerts. Free with API key."""
import json, time, urllib.request
from pathlib import Path
API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
TNS_BASE = "https://www.wis-tns.org/api"
_LAST = 0.0
def _get_key():
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        return cfg.get("tns_api_key", "")
    except: return ""
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<2: time.sleep(2-(now-_LAST))
    _LAST=time.time()
def get_recent_transients(limit=20):
    key = _get_key()
    if not key: return [{"error": "TNS API key not configured", "status": "missing_key"}]
    return [{"error": "TNS requires interactive auth", "status": "auth_required"}]
def is_available(): return bool(_get_key())
