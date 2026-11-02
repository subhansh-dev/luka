"""Real orbital mechanics using poliastro and skyfield."""
import math
_HAS_POLIASTRO = False
_HAS_SKYFIELD = False
try:
    from poliastro.bodies import Earth, Mars, Jupiter, Sun
    from poliastro.maneuvers import hohmann
    from poliastro.twobody import Orbit
    import astropy.units as u
    _HAS_POLIASTRO = True
except ImportError:
    pass
try:
    from skyfield.api import load, Topos
    _HAS_SKYFIELD = True
except ImportError:
    pass

def hohmann_transfer_delta_v(r1_km, r2_km):
    """Compute delta-v for Hohmann transfer between two circular orbits."""
    if _HAS_POLIASTRO:
        try:
            orb1 = Orbit.circular(Earth, alt=(r1_km - 6371) * u.km)
            orb2 = Orbit.circular(Earth, alt=(r2_km - 6371) * u.km)
            man_dv1, man_dv2 = hohmann(orb1, orb2.r_a)
            dv1 = man_dv1.norm().to(u.m / u.s).value
            dv2 = man_dv2.norm().to(u.m / u.s).value
            return {"delta_v1_m_s": round(dv1, 1), "delta_v2_m_s": round(dv2, 1),
                    "total_delta_v_m_s": round(dv1 + dv2, 1), "method": "poliastro"}
        except Exception as e:
            return {"error": str(e)[:100]}
    # Fallback: analytical Hohmann
    mu = 3.986e14  # Earth gravitational parameter m3/s2
    r1 = r1_km * 1000
    r2 = r2_km * 1000
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    a_transfer = (r1 + r2) / 2
    v_transfer1 = math.sqrt(mu * (2/r1 - 1/a_transfer))
    v_transfer2 = math.sqrt(mu * (2/r2 - 1/a_transfer))
    dv1 = abs(v_transfer1 - v1)
    dv2 = abs(v2 - v_transfer2)
    return {"delta_v1_m_s": round(dv1, 1), "delta_v2_m_s": round(dv2, 1),
            "total_delta_v_m_s": round(dv1 + dv2, 1), "method": "analytical"}

def orbital_period(semi_major_axis_km, mu=3.986e14):
    """Compute orbital period from semi-major axis."""
    a = semi_major_axis_km * 1000
    T = 2 * math.pi * math.sqrt(a**3 / mu)
    return {"period_seconds": round(T, 1), "period_hours": round(T/3600, 2),
            "period_days": round(T/86400, 3)}

def escape_velocity(radius_km, mass_kg=5.972e24):
    """Compute escape velocity from a body."""
    G = 6.674e-11
    v = math.sqrt(2 * G * mass_kg / (radius_km * 1000))
    return {"escape_velocity_m_s": round(v, 1), "escape_velocity_km_s": round(v/1000, 2)}

def is_available():
    return _HAS_POLIASTRO or _HAS_SKYFIELD
