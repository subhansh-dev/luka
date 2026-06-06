#!/usr/bin/env python3
"""
verify_install.py — Luka Space Discovery Engine Installation Verifier
Checks for necessary files, Python version, and reachability of Space APIs.
"""

import sys
import os
import json
import asyncio
import http.client
import urllib.parse
from pathlib import Path

def print_status(component, status, message=""):
    color = "\033[92m" if status == "OK" else "\033[91m"
    reset = "\033[0m"
    print(f"[{color}{status}{reset}] {component:30} {message}")

async def check_api_reachability(host, path="/", name="API"):
    try:
        conn = http.client.HTTPSConnection(host, timeout=5)
        conn.request("GET", path)
        res = conn.getresponse()
        if res.status < 500:
            print_status(name, "OK", f"Status: {res.status}")
            return True
        else:
            print_status(name, "FAIL", f"Status: {res.status}")
            return False
    except Exception as e:
        print_status(name, "FAIL", f"Error: {str(e)}")
        return False

async def main():
    print("=== Luka Space Discovery Engine: Verification ===\n")

    # 1. Python Version
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 10:
        print_status("Python Version", "OK", f"{sys.version.split()[0]}")
    else:
        print_status("Python Version", "FAIL", "Python 3.10+ required")

    # 2. Core Files
    core_files = [
        "main.py",
        "run_discovery.py",
        "discovery/discovery_pipeline_v2.py",
        "discovery/isro_api.py",
        "discovery/nasa_api.py",
        "discovery/simbad_api.py",
        "discovery/vizier_api.py",
        "discovery/jpl_horizons_api.py",
        "discovery/noaa_api.py",
        "discovery/esa_sky_api.py",
        "discovery/usgs_planetary_api.py",
        "discovery/domains.py"
    ]
    for f in core_files:
        if os.path.exists(f):
            print_status(f"File: {f}", "OK")
        else:
            print_status(f"File: {f}", "FAIL", "Missing!")

    # 3. API Reachability
    print("\n--- Checking Space API Reachability ---")
    await check_api_reachability("api.nasa.gov", "/", "NASA API")
    await check_api_reachability("mosdac.gov.in", "/", "ISRO MOSDAC")
    await check_api_reachability("bhuvan-app1.nrsc.gov.in", "/", "ISRO Bhuvan")
    await check_api_reachability("simbad.u-strasbg.fr", "/simbad/", "SIMBAD")
    await check_api_reachability("vizier.u-strasbg.fr", "/", "VizieR")
    await check_api_reachability("ssd.jpl.nasa.gov", "/", "JPL Horizons")
    await check_api_reachability("sky.esa.int", "/", "ESA Sky")
    await check_api_reachability("swpc.noaa.gov", "/", "NOAA Space Weather")
    await check_api_reachability("arxiv.org", "/list/astro-ph/new", "arXiv astro-ph")

    print("\nVerification complete.")

if __name__ == "__main__":
    asyncio.run(main())
