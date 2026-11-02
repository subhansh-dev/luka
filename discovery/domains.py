"""Domain definitions for Luka Space Discovery Engine."""

DOMAINS = {
    "space_astronomy": {
        "label": "Space & Astronomy",
        "description": "Celestial bodies, missions, exoplanets, galaxies, stars — with NASA image + exoplanet archive + arXiv enrichment",
        "aliases": ["space", "astronomy", "astro", "nasa", "cosmos", "astrophysics"],
        "entity_types": {
            "celestial_body":    {"color": "#e056fd", "description": "Planet, moon, asteroid, or other body"},
            "mission":           {"color": "#00b894", "description": "Space mission or program"},
            "telescope":         {"color": "#00cec9", "description": "Telescope or observatory"},
            "exoplanet":         {"color": "#fdcb6e", "description": "Planet outside our solar system"},
            "spectral_feature":  {"color": "#ff6b6b", "description": "Spectral line, band, or signature at specific wavelength"},
            "molecule_gas":      {"color": "#a29bfe", "description": "Specific atmospheric molecule or gas (CH4, CO2, PH3, H2O, O2, NH3)"},
            "stellar_property":  {"color": "#ffeaa7", "description": "Stellar radiation, UV flux, activity metric"},
            "atmospheric_param": {"color": "#74b9ff", "description": "Atmospheric mixing ratio, pressure, temperature, composition metric"},
            "phenomenon":        {"color": "#fd79a8", "description": "Astrophysical phenomenon"},
            "theory":            {"color": "#74b9ff", "description": "Astrophysical theory or model"},
            "star":              {"color": "#f1c40f", "description": "Stellar object"},
            "galaxy":            {"color": "#9b59b6", "description": "Galactic structure"},
            "nebula":            {"color": "#3498db", "description": "Interstellar cloud"},
            "asteroid":          {"color": "#95a5a6", "description": "Small rocky body"},
            "comet":             {"color": "#bdc3c7", "description": "Icy small solar system body"},
            "spacecraft":        {"color": "#2ecc71", "description": "Man-made space vehicle"},
            "instrument":        {"color": "#16a085", "description": "Scientific instrument"},
            "orbit":             {"color": "#e67e22", "description": "Orbital path or parameters"},
        },
        "enrichment": ["nasa_api", "arxiv", "simbad_api", "vizier_api", "iau_api", "esa_sky_api", "mast_api", "ligo_gwosc_api", "gcn_api", "nasa_donki_api"],
        "data_sources": ["nasa_exoplanets"],
        "generation": "hypothesis",
        "extraction_guide": "Extract SPECIFIC scientific variables: exact molecule names (CH4, CO2, PH3, H2O, O2, NH3, N2O), spectral line wavelengths (1.6um, 4.5um, 15um), atmospheric mixing ratios (1ppm, 0.1ppb), stellar UV flux values, planetary equilibrium temperatures (300K, 500K), transit depths (200ppm), orbital periods, stellar types (M-dwarf, G2V), instrument sensitivities (R=100,000), photochemical rates, biosignature metrics (DES, CDD). AVOID broad categories like 'exoplanet' or 'star' without specific type or property.",
    },

    "isro_earth_observation": {
        "label": "ISRO Earth Observation",
        "description": "Indian satellite data, meteorology, oceanography, geospatial — MOSDAC + Bhuvan",
        "aliases": ["isro", "mosdac", "bhuvan", "india_space", "insat", "oceansat", "indian_satellite"],
        "entity_types": {
            "satellite":       {"color": "#e056fd", "description": "ISRO satellite mission (INSAT, OCEANSAT, etc.)"},
            "instrument":      {"color": "#00cec9", "description": "Satellite instrument or sensor (SAPHIR, SCAT, etc.)"},
            "region":          {"color": "#00b894", "description": "Indian subcontinent geographic region"},
            "parameter":       {"color": "#fdcb6e", "description": "Measured parameter (rainfall, SST, soil moisture, etc.)"},
            "phenomenon":      {"color": "#ff6b6b", "description": "Earth phenomenon (cyclone, monsoon, flood, drought)"},
            "ground_station":  {"color": "#74b9ff", "description": "ISRO ground station or facility"},
            "launch_vehicle":  {"color": "#a29bfe", "description": "ISRO launch vehicle (PSLV, GSLV, SSLV)"},
        },
        "enrichment": ["isro_api", "nasa_api", "arxiv"],
        "generation": "hypothesis",
        "extraction_guide": "Extract SPECIFIC ISRO entities: exact satellite names (INSAT-3DR, OCEANSAT-3, SCATSAT-1), instrument names (SAPHIR, SCAT, OCM-3), geographic coordinates (lat/lon), measured parameters with units (SST in degC, rainfall in mm/day, wind in m/s), Indian locations (Sriharikota, Bangalore, Ahmedabad), mission names (Chandrayaan, Mangalyaan, Gaganyaan), launch vehicles (PSLV-C53, GSLV-Mk3). AVOID generic terms like 'satellite' or 'spacecraft' without specific names.",
    },

    "solar_system": {
        "label": "Solar System",
        "description": "Planets, moons, asteroids, and comets within our solar system",
        "aliases": ["solar", "planets", "moon", "asteroid", "comet"],
        "entity_types": {
            "planet":          {"color": "#e67e22", "description": "Major planet in the solar system"},
            "moon":            {"color": "#ecf0f1", "description": "Natural satellite"},
            "asteroid":        {"color": "#95a5a6", "description": "Small rocky body"},
            "comet":           {"color": "#bdc3c7", "description": "Icy body"},
            "spacecraft":      {"color": "#2ecc71", "description": "Space exploration vehicle"},
            "crater":          {"color": "#7f8c8d", "description": "Impact feature"},
            "surface_feature": {"color": "#d35400", "description": "Geological feature on a body"},
            "atmosphere":      {"color": "#3498db", "description": "Gaseous envelope"},
            "magnetosphere":   {"color": "#9b59b6", "description": "Magnetic environment"},
        },
        "enrichment": ["jpl_horizons_api", "nasa_api", "usgs_planetary_api", "jpl_smallbody_api", "mpc_api", "celestrak_api"],
        "data_sources": ["nasa_neo"],
        "generation": "hypothesis",
        "extraction_guide": "Extract SPECIFIC solar system entities: planet names, moon names (Europa, Enceladus), asteroid names/numbers (Bennu, 16 Psyche), spacecraft (Voyager 2, Cassini), named craters (Shackleton), specific surface features, atmospheric components, and magnetic field strengths.",
    },

    "climate_energy": {
        "label": "Space-Derived Climate",
        "description": "Satellite-derived climate data and solar energy metrics — with NASA POWER enrichment",
        "aliases": ["climate", "energy", "solar_power", "nasa_power"],
        "entity_types": {
            "parameter":       {"color": "#fdcb6e", "description": "Measured parameter (insolation, temperature, etc.)"},
            "region":          {"color": "#55efc4", "description": "Geographic region"},
            "satellite":       {"color": "#e056fd", "description": "Observing satellite"},
            "phenomenon":      {"color": "#ff6b6b", "description": "Climate phenomenon observed from space"},
        },
        "enrichment": ["nasa_power", "noaa_api", "noaa_swpc_api", "nasa_donki_api", "semantic_scholar"],
        "generation": "hypothesis",
        "extraction_guide": "Extract SPECIFIC climate variables derived from satellite observations: solar insolation values, surface temperatures, precipitation rates, and atmospheric phenomena like cyclones or heatwaves. Focus on space-derived datasets.",
    },

    "general": {
        "label": "General Space Research",
        "description": "Any space-related scientific topic — entities auto-detected from paper content",
        "aliases": ["general", "space_science", "any_space"],
        "entity_types": {
            "concept":       {"color": "#dfe6e9", "description": "General space science concept"},
            "method":        {"color": "#74b9ff", "description": "Scientific method"},
            "finding":       {"color": "#55efc4", "description": "Key scientific finding"},
            "mission":       {"color": "#00b894", "description": "Space mission"},
        },
        "enrichment": ["semantic_scholar", "arxiv"],
        "generation": "hypothesis",
        "extraction_guide": "Extract SPECIFIC space-related entities and findings. AVOID non-space topics.",
    },
}

DOMAIN_ALIAS_MAP = {}
for key, config in DOMAINS.items():
    for alias in config["aliases"]:
        DOMAIN_ALIAS_MAP[alias] = key


def get_domain(domain_key: str) -> dict | None:
    """Get domain config by key, falling back to alias lookup."""
    if domain_key in DOMAINS:
        return DOMAINS[domain_key]
    return DOMAINS.get(DOMAIN_ALIAS_MAP.get(domain_key))


def list_domains() -> list[dict]:
    """Return list of domain summaries."""
    return [
        {"key": key, "label": cfg["label"], "description": cfg["description"]}
        for key, cfg in DOMAINS.items()
    ]


def entity_types_list(domain_key: str) -> str:
    """Return a formatted string of entity types for a domain."""
    cfg = get_domain(domain_key)
    if not cfg:
        cfg = DOMAINS["general"]
    return ", ".join(sorted(cfg["entity_types"].keys()))


def entity_colors(domain_key: str) -> dict[str, str]:
    """Return entity_type -> color map for a domain."""
    cfg = get_domain(domain_key)
    if not cfg:
        cfg = DOMAINS["general"]
    return {et: info["color"] for et, info in cfg["entity_types"].items()}


DETECT_DOMAIN_PROMPT = """You are a scientific domain classifier. Given a research topic, classify it into ONE of these domains:

%s

Respond with EXACTLY the domain key and nothing else.

Topic: "%s" """


def build_detect_prompt() -> str:
    """Build the prompt prefix for domain detection (without topic)."""
    parts = []
    for key, cfg in DOMAINS.items():
        aliases_str = ", ".join(cfg["aliases"])
        parts.append(f"- {key}: {cfg['description']} (aliases: {aliases_str})")
    return "You are a scientific domain classifier. Given a research topic, classify it into ONE of these domains:\n\n" + "\n".join(parts) + '\n\nRespond with EXACTLY the domain key and nothing else.\n\nTopic: "'
