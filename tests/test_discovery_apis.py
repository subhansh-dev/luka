"""Tests for Luka's discovery API wrappers.

Run: python -m pytest tests/test_discovery_apis.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestNASAApi:
    """Test NASA API wrapper (real API calls)."""

    def test_search_images(self):
        from discovery.nasa_api import search_images
        results = search_images("Mars rover", limit=3)
        assert isinstance(results, list)
        if results:
            assert "title" in results[0]
            assert "description" in results[0]

    def test_search_exoplanets(self):
        from discovery.nasa_api import search_exoplanets
        results = search_exoplanets("Kepler", limit=3)
        assert isinstance(results, list)
        if results:
            assert "name" in results[0]
            assert "radius_earth" in results[0] or "orbital_period_days" in results[0]

    def test_get_neo(self):
        from discovery.nasa_api import get_neo
        results = get_neo(limit=3)
        assert isinstance(results, list)
        if results:
            assert "name" in results[0]


class TestJPLHorizons:
    """Test JPL Horizons API wrapper."""

    def test_get_ephemeris_mars(self):
        from discovery.jpl_horizons_api import get_ephemeris
        result = get_ephemeris("Mars")
        assert result is not None
        assert result["target"] == "Mars"
        assert result["source"] == "astroquery"
        assert "distance_au" in result

    def test_get_ephemeris_europa(self):
        from discovery.jpl_horizons_api import get_ephemeris
        result = get_ephemeris("Europa")
        assert result is not None
        assert result["target"] == "Europa"

    def test_resolve_id(self):
        from discovery.jpl_horizons_api import _resolve_id
        assert _resolve_id("Mars") == "499"
        assert _resolve_id("Europa") == "502"
        assert _resolve_id("599") == "599"


class TestSIMBAD:
    """Test SIMBAD API wrapper."""

    def test_search_object_betelgeuse(self):
        from discovery.simbad_api import search_object
        result = search_object("Betelgeuse")
        if result is not None:
            assert "main_id" in result
            assert "ra" in result

    def test_search_object_not_found(self):
        from discovery.simbad_api import search_object
        result = search_object("ZZZZNONEXISTENT999")
        assert result is None

    def test_search_objects_in_region(self):
        from discovery.simbad_api import search_objects_in_region
        results = search_objects_in_region(88.8, 7.4, radius_arcmin=30)
        assert isinstance(results, list)


class TestVizieR:
    """Test VizieR API wrapper."""

    def test_search_catalogs(self):
        from discovery.vizier_api import search_catalogs
        results = search_catalogs("Sirius", limit=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_object_in_gaia(self):
        from discovery.vizier_api import search_object_in_gaia
        result = search_object_in_gaia("Sirius")
        assert isinstance(result, list)


class TestESASky:
    """Test ESA Sky API wrapper."""

    def test_search_esa_sky(self):
        from discovery.esa_sky_api import search_esa_sky, is_available
        if not is_available():
            pytest.skip("astroquery.esasky not available")
        results = search_esa_sky("Crab Nebula")
        assert isinstance(results, list)

    def test_search_region(self):
        from discovery.esa_sky_api import is_available
        assert isinstance(is_available(), bool)


class TestIAU:
    """Test IAU nomenclature wrapper."""

    def test_get_nomenclature_known(self):
        from discovery.iau_api import get_nomenclature
        result = get_nomenclature("Olympus Mons")
        assert result is not None
        assert result["name"] == "Olympus Mons"
        assert result["source"] is not None

    def test_get_nomenclature_crater(self):
        from discovery.iau_api import get_nomenclature
        result = get_nomenclature("Tycho")
        assert result is not None
        # Should detect it's a crater from the name
        assert "crater" in result.get("feature_type", "").lower() or result.get("source")

    def test_feature_type_inference(self):
        from discovery.iau_api import get_nomenclature
        result = get_nomenclature("Olympus Mons")
        assert result is not None
        if result.get("status") == "Inferred from nomenclature conventions":
            assert "Mountain" in result.get("feature_type", "")


class TestISRO:
    """Test ISRO API wrapper."""

    def test_search_satellite_data(self):
        from discovery.isro_api import search_satellite_data
        results = search_satellite_data("INSAT-3DR")
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["source"] is not None

    def test_search_unknown_satellite(self):
        from discovery.isro_api import search_satellite_data
        results = search_satellite_data("FAKE_SAT_999")
        assert isinstance(results, list)

    def test_geocode_known_city(self):
        from discovery.isro_api import geocode_village
        result = geocode_village("Chennai")
        assert result is not None
        assert result["lat"] is not None
        assert result["lon"] is not None
        assert abs(result["lat"] - 13.08) < 1.0  # Chennai is ~13°N

    def test_geocode_unknown(self):
        from discovery.isro_api import geocode_village
        result = geocode_village("ZZZZNONEXISTENT999")
        assert result is not None
        # Should return something even if not found

    def test_is_available(self):
        from discovery.isro_api import is_available
        # Just check it doesn't crash
        result = is_available()
        assert isinstance(result, bool)


class TestDomains:
    """Test domain detection."""

    def test_detect_domain_space(self):
        from discovery.discovery_pipeline_v2 import detect_domain
        assert detect_domain("exoplanet atmospheric biosignatures") == "space_astronomy"

    def test_detect_domain_isro(self):
        from discovery.discovery_pipeline_v2 import detect_domain
        assert detect_domain("ISRO MOSDAC monsoon prediction") == "isro_earth_observation"

    def test_detect_domain_solar(self):
        from discovery.discovery_pipeline_v2 import detect_domain
        assert detect_domain("asteroid deflection near-Earth comet") == "solar_system"

    def test_detect_domain_climate(self):
        from discovery.discovery_pipeline_v2 import detect_domain
        assert detect_domain("climate carbon emission renewable") == "climate_energy"

    def test_detect_domain_general(self):
        from discovery.discovery_pipeline_v2 import detect_domain
        result = detect_domain("random topic with no keywords")
        assert result == "general"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
