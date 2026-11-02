"""Copernicus Open Access Hub - Sentinel satellite data. Free with account."""
import json, time, urllib.request, urllib.parse
from pathlib import Path
API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
DHUS_BASE = "https://scihub.copernicus.eu/dhus"
_LAST = 0.0
def _get_creds():
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        return cfg.get("copernicus_user", ""), cfg.get("copernicus_pass", "")
    except: return "", ""
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<2: time.sleep(2-(now-_LAST))
    _LAST=time.time()
def search_products(query, platform="Sentinel-2", limit=10):
    user, pwd = _get_creds()
    if not user: return [{"error": "Copernicus credentials not configured", "status": "missing_key"}]
    return [{"error": "Copernicus requires interactive auth", "status": "auth_required"}]
def is_available():
    u, p = _get_creds()
    return bool(u)
