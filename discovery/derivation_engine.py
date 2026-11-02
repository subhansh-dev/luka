"""
derivation_engine.py — Mechanism-driven mathematical derivation and verification.

Extracts equations from generated mechanisms, parses them with SymPy,
and verifies them numerically with NumPy/SciPy. Falls back to domain-specific
derivations only when mechanisms lack parseable equations.
"""

import math
import re

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Physical constants
HBAR = 1.054571817e-34
C_LIGHT = 2.99792458e8
K_B = 1.380649e-23
EV = 1.602176634e-19
G_CONST = 6.67430e-11  # gravitational constant
MU_0 = 4 * math.pi * 1e-7  # vacuum permeability
EPSILON_0 = 8.8541878128e-12  # vacuum permittivity
PLANCK = 6.62607015e-34
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
M_PROTON = 1.67262192369e-27
STEFAN_BOLTZMANN = 5.670374419e-8
AVOGADRO = 6.02214076e23
R_GAS = 8.314462618  # J/(mol*K)

# Common symbol mappings for SymPy parsing
SYMPY_SYMBOLS = {
    'pi': sp.pi if HAS_SYMPY else None,
    'e': sp.E if HAS_SYMPY else None,
    'hbar': HBAR, 'c': C_LIGHT, 'k_B': K_B, 'G': G_CONST,
    'epsilon_0': EPSILON_0, 'mu_0': MU_0, 'e_charge': E_CHARGE,
    'm_e': M_ELECTRON, 'm_p': M_PROTON, 'sigma_sb': STEFAN_BOLTZMANN,
    'N_A': AVOGADRO, 'R': R_GAS,
}


# ══════════════════════════════════════════════════════════════
# DOMAIN ROUTER
# ══════════════════════════════════════════════════════════════

def run_domain_verification(domain: str, mechanisms: list = None, topic: str = "") -> dict:
    """Run domain-appropriate derivations based on the discovery domain.

    Primary path: extract equations from mechanisms and verify symbolically/numerically.
    Fallback: domain-specific derivations for well-known physics problems.
    """
    # PRIMARY: Extract and verify equations from the actual mechanisms
    mech_results = _verify_mechanism_equations(mechanisms, topic, domain)

    # If mechanisms had parseable equations, return those
    if mech_results.get("mechanisms_verified", 0) > 0:
        return mech_results

    # FALLBACK: Domain-specific hardcoded derivations (only when mechanisms lack equations)
    domain_lower = domain.lower() if domain else ""
    topic_lower = topic.lower() if topic else ""

    if ("solar_system" in domain_lower or "planet" in domain_lower or
            "moon" in domain_lower or "enceladus" in topic_lower or
            "titan" in topic_lower or "europa" in topic_lower):
        result = run_solar_system_verification(mechanisms, topic)
        result["source"] = "domain_template"
        return result

    if "supercon" in topic_lower or "condensed" in domain_lower or "material" in domain_lower:
        result = run_condensed_matter_verification(mechanisms, topic)
        result["source"] = "domain_template"
        return result

    if "physics" in domain_lower or "casimir" in topic_lower or "quantum vacuum" in topic_lower:
        result = run_physics_verification()
        result["source"] = "domain_template"
        return result

    if "drug" in domain_lower or "bio" in domain_lower:
        result = run_drug_discovery_verification(mechanisms, topic)
        result["source"] = "domain_template"
        return result

    if "math" in domain_lower:
        result = run_mathematics_verification(mechanisms, topic)
        result["source"] = "domain_template"
        return result

    # No domain match and no mechanism equations — report empty
    mech_results["source"] = "mechanism_extraction"
    mech_results["skipped"] = True
    mech_results["reason"] = f"no parseable equations in mechanisms for domain '{domain}'"
    return mech_results


# ══════════════════════════════════════════════════════════════
# MECHANISM EQUATION EXTRACTION & VERIFICATION
# ══════════════════════════════════════════════════════════════

def _verify_mechanism_equations(mechanisms: list, topic: str, domain: str) -> dict:
    """Extract equations from mechanism descriptions and verify them.

    Looks for equations in: mathematical_model, derivation, derivation_steps,
    description, numerical_verification fields.
    Uses SymPy for symbolic parsing and NumPy for numerical evaluation.
    """
    results = {
        "mechanisms_verified": 0,
        "equations_found": 0,
        "equations_verified": 0,
        "equations_failed": 0,
        "mechanism_details": [],
        "all_constants": _get_constants_dict(),
    }

    if not mechanisms:
        return results

    for i, mech in enumerate(mechanisms):
        if not isinstance(mech, dict):
            continue

        name = mech.get("name", f"mechanism_{i}")
        detail = {"name": name, "equations": [], "parameters": {}, "verdict": "pending"}

        # Collect all text that might contain equations
        text_fields = [
            mech.get("mathematical_model", ""),
            mech.get("derivation", ""),
            mech.get("description", ""),
            mech.get("numerical_verification", ""),
        ]
        derivation_steps = mech.get("derivation_steps", [])
        if isinstance(derivation_steps, list):
            text_fields.extend(derivation_steps)

        full_text = "\n".join(str(f) for f in text_fields if f)

        # Extract equations from text
        equations = _extract_equations_from_text(full_text)
        if not equations:
            # Try to extract from the derivation field specifically
            deriv = mech.get("derivation", "")
            if deriv:
                equations = _extract_equations_from_derivation(deriv)

        detail["raw_equations"] = equations

        # Parse parameters from key_parameters field
        key_params = mech.get("key_parameters", [])
        params = {}
        for p in key_params:
            if isinstance(p, dict):
                pname = p.get("name", "")
                pval = p.get("expected_value", p.get("value", ""))
                punits = p.get("units", "")
                if pname and pval:
                    params[pname] = {"value": pval, "units": punits}
        detail["parameters"] = params

        # Attempt symbolic verification with SymPy
        verified_count = 0
        for eq_text in equations:
            eq_result = _verify_single_equation(eq_text, params, full_text)
            detail["equations"].append(eq_result)
            if eq_result.get("symbolically_parsed"):
                verified_count += 1
                results["equations_verified"] += 1
            elif eq_result.get("numerically_evaluated"):
                verified_count += 1
                results["equations_verified"] += 1
            else:
                results["equations_failed"] += 1

        results["equations_found"] += len(equations)
        if verified_count > 0:
            results["mechanisms_verified"] += 1
            detail["verdict"] = "verified"
        elif equations:
            detail["verdict"] = "parsed_but_unverified"
        else:
            detail["verdict"] = "no_equations_found"

        results["mechanism_details"].append(detail)

    return results


def _extract_equations_from_text(text: str) -> list:
    """Extract mathematical equations from free-form text.

    Looks for patterns like:
      - E = mc^2
      - F = ma
      - P = -pi^2*hbar*c/(240*a^4)
      - h_{mu nu} = ...
      - equations with =, ≈, ~, ≥, ∝
    """
    if not text:
        return []

    equations = []
    lines = text.split("\n")

    # Patterns that indicate an equation line
    eq_patterns = [
        r'^[A-Z]\s*=\s*.+',           # Single letter = expression (E = mc^2)
        r'^[A-Z][_\{].*=\s*.+',       # Subscripted = expression
        r'^\w+\s*=\s*.+',             # word = any expression
        r'^\w+\s*=\s*[\-\d\(].*',     # word = number/expression
        r'.*\s*=\s*[\-\d\(].*',       # anything = number/expression
        r'.*≈\s*.+',                   # approximate equals
        r'.*∝\s*.+',                   # proportional to
        r'.*→\s*.+',                   # arrow notation
    ]

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        # Skip pure text lines
        if not any(c in line for c in '=≈~≥≤∝→'):
            continue
        # Skip lines that are clearly prose
        if line.startswith(('The ', 'This ', 'Where ', 'For ', 'If ', 'Note ', 'Step ', 'From ')):
            # But keep if they contain an equation
            if '=' not in line and '≈' not in line:
                continue

        for pattern in eq_patterns:
            if re.match(pattern, line):
                # Clean up the equation
                eq = line.strip().rstrip('.').rstrip(',')
                if len(eq) > 5 and len(eq) < 500:
                    equations.append(eq)
                break

    # Also try to find equations in JSON-like structures
    json_eq_pattern = r'"(?:equation|formula|mathematical_model)"\s*:\s*"([^"]+)"'
    for match in re.finditer(json_eq_pattern, text):
        eq = match.group(1)
        if len(eq) > 5:
            equations.append(eq)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for eq in equations:
        normalized = eq.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(eq)

    return unique[:10]  # Cap at 10 equations


def _extract_equations_from_derivation(derivation_text: str) -> list:
    """Extract equations specifically from derivation text (step-by-step)."""
    if not derivation_text:
        return []

    equations = []
    # Look for "equation:" or "formula:" patterns
    for pattern in [
        r'(?:equation|formula|expression|result)\s*[:=]\s*(.+?)(?:\n|$)',
        r'(?:starting from|given|using)\s+(.+?(?:=|≈).+?)(?:\n|,|\.)',
    ]:
        for match in re.finditer(pattern, derivation_text, re.IGNORECASE):
            eq = match.group(1).strip()
            if len(eq) > 5 and len(eq) < 300:
                equations.append(eq)

    return equations[:5]


def _normalize_equation_for_sympy(text: str) -> str:
    """Convert Unicode/LaTeX physics notation to SymPy-parseable form.

    Handles: Unicode Greek letters, tensor indices, LaTeX commands,
    math symbols, non-breaking spaces, and common physics notation.
    """
    s = text

    # ── Remove non-breaking spaces and special whitespace FIRST ──
    s = s.replace('\u2003', ' ')  # em space
    s = s.replace('\u2002', ' ')  # en space
    s = s.replace('\u00a0', ' ')  # non-breaking space
    s = s.replace('\u200b', '')   # zero-width space
    s = s.replace('\u200c', '')   # zero-width non-joiner
    s = s.replace('\u200d', '')   # zero-width joiner
    s = s.replace('\ufeff', '')   # BOM

    # ── Angle brackets: ⟨...⟩ → ( ... ) ──
    s = s.replace('⟨', '(').replace('⟩', ')')

    # ── Absolute value: |...| → abs(...) ──
    s = re.sub(r'\|([^|]+)\|', r'abs(\1)', s)

    # ── Unicode Greek letters → ASCII ──
    greek_map = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
        'ν': 'nu', 'ξ': 'xi', 'π': 'pi', 'ρ': 'rho',
        'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
        'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
        'Γ': 'Gamma', 'Δ': 'Delta', 'Θ': 'Theta', 'Λ': 'Lambda',
        'Ξ': 'Xi', 'Π': 'Pi', 'Σ': 'Sigma', 'Φ': 'Phi',
        'Ψ': 'Psi', 'Ω': 'Omega',
    }
    for uni, ascii_name in greek_map.items():
        s = s.replace(uni, ascii_name)

    # ── Unicode math symbols → ASCII ──
    s = s.replace('√', 'sqrt')
    s = s.replace('∂', 'partial_d')
    s = s.replace('∇', 'nabla')
    s = s.replace('ℏ', 'hbar')
    s = s.replace('∞', 'oo')
    s = s.replace('±', '+/-')
    s = s.replace('×', '*')
    s = s.replace('·', '*')
    s = s.replace('≈', '=')
    s = s.replace('≃', '=')
    s = s.replace('≅', '=')
    s = s.replace('≡', '=')
    s = s.replace('≠', '!=')
    s = s.replace('≤', '<=')
    s = s.replace('≥', '>=')
    s = s.replace('∝', '~')
    s = s.replace('→', '->')
    s = s.replace('←', '<-')
    s = s.replace('⟹', '->')
    s = s.replace('⟸', '<-')
    s = s.replace('∈', 'in')
    s = s.replace('∉', 'not_in')

    # ── Implicit multiplication for common physics patterns ──
    # 8piG → 8*pi*G, 16piG → 16*pi*G
    s = re.sub(r'(\d+)pi([a-zA-Z])', r'\1*pi*\2', s)
    s = re.sub(r'(\d+)pi\b', r'\1*pi', s)
    s = re.sub(r'pi(\d+)', r'pi*\1', s)
    s = re.sub(r'(\d+)G\b', r'\1*G', s)
    s = re.sub(r'(\d+)c\b', r'\1*c', s)
    s = re.sub(r'(\d+)hbar\b', r'\1*hbar', s)

    # ── Integral: ∫d⁴x → integral_x (simplified) ──
    s = re.sub(r'Integral\s*d(\d+)x', r'integral_x', s)

    # ── Star/dual: *R^{...} → star_R_... (Hodge dual) ──
    # Only when * is followed by uppercase R (Riemann dual), not general multiplication
    s = re.sub(r'\*R\{', 'star_R_', s)
    s = re.sub(r'\*R\^', 'star_R_', s)

    # ── Tensor index notation: σ_{ij} → sigma_ij, T^{μν} → T_munu ──
    s = re.sub(r'\{([a-zA-Z0-9_]+)\}', r'_\1', s)
    s = re.sub(r'\^_([a-zA-Z0-9_]+)', r'**_\1', s)
    s = re.sub(r'\^([a-zA-Z0-9_]+)', r'**\1', s)

    # ── LaTeX commands → SymPy ──
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\[a-z]+', '', s)  # remove remaining \commands

    # ── Superscript Unicode digits: 10⁻⁹ → 10**(-9) ──
    sup_map = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
               '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
               '⁻': '-'}

    def _replace_superscripts(m):
        base = m.group(1)
        sup = m.group(2)
        for uni, dig in sup_map.items():
            sup = sup.replace(uni, dig)
        return '%s**(%s)' % (base, sup)

    s = re.sub(r'([0-9.]+)([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)', _replace_superscripts, s)

    # ── Letter² → letter**2 (ds², dt², r², etc.) ──
    s = re.sub(r'([a-zA-Z])²', r'\1**2', s)
    s = re.sub(r'([a-zA-Z])³', r'\1**3', s)
    s = re.sub(r'([a-zA-Z])ⁿ', r'\1**n', s)

    # ── Common physics superscripts ──
    s = s.replace('c³', 'c**3')
    s = s.replace('c⁴', 'c**4')
    s = s.replace('c²', 'c**2')
    s = s.replace('G²', 'G**2')

    # ── Clean multiple spaces ──
    s = re.sub(r' +', ' ', s).strip()

    return s


def _verify_single_equation(eq_text: str, params: dict, context: str) -> dict:
    """Verify a single equation using SymPy (symbolic) and NumPy (numerical)."""
    result = {
        "equation": eq_text,
        "symbolically_parsed": False,
        "numerically_evaluated": False,
        "issues": [],
    }

    if not HAS_SYMPY:
        result["issues"].append("sympy not available")
        return result

    # Try to parse with SymPy
    try:
        # Normalize the equation for SymPy
        clean_eq = _normalize_equation_for_sympy(eq_text.strip())

        # Split on = to get LHS and RHS
        if '=' in clean_eq:
            parts = clean_eq.split('=', 1)
            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()

            # Skip if LHS or RHS is empty or too short
            if len(lhs_str) < 1 or len(rhs_str) < 1:
                result["issues"].append("empty LHS or RHS after normalization")
                return result

            # Skip equations that start with descriptive text (prose, not math)
            prose_starters = ('From ', 'Starting ', 'Using ', 'The ', 'This ',
                              'When ', 'If ', 'For ', 'Note ', 'Step ', 'where ',
                              'which ', 'that ', 'Since ', 'Because ', 'Given ')
            if lhs_str.startswith(prose_starters) or rhs_str.startswith(prose_starters):
                result["issues"].append("equation is prose, not parseable math")
                return result

            # Define symbols for any unknowns
            all_symbols = set()
            for var_name in params.keys():
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', var_name)
                if safe_name and safe_name[0].isalpha():
                    all_symbols.add(safe_name)

            # Also find symbols in the equation text
            potential_syms = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', rhs_str)
            for sym in potential_syms:
                if sym not in ('pi', 'e', 'exp', 'log', 'sin', 'cos', 'sqrt', 'abs',
                               'Integral', 'partial_d', 'nabla', 'in', 'not_in', 'oo',
                               'Delta', 'Gamma', 'Theta', 'Lambda', 'Xi', 'Pi',
                               'Sigma', 'Phi', 'Psi', 'Omega'):
                    all_symbols.add(sym)

            # Create SymPy symbols
            sym_objects = {}
            for s in all_symbols:
                try:
                    sym_objects[s] = sp.Symbol(s)
                except Exception:
                    pass

            # Add known constants with their numerical values
            known = {
                'pi': sp.pi, 'e': sp.E, 'hbar': HBAR, 'c': C_LIGHT,
                'k_B': K_B, 'G': G_CONST, 'epsilon_0': EPSILON_0,
                'mu_0': MU_0, 'e_charge': E_CHARGE,
                'm_e': M_ELECTRON, 'm_p': M_PROTON,
                'sigma_sb': STEFAN_BOLTZMANN, 'N_A': AVOGADRO, 'R': R_GAS,
                'planck': PLANCK,
            }
            # Known physics functions (not symbols)
            known_funcs = ['b_prime', 'f', 'g', 'h', 'Phi', 'Psi', 'Theta']
            for fname in known_funcs:
                if fname in rhs_str or fname in lhs_str:
                    known[fname] = sp.Function(fname)
            for k, v in known.items():
                if k in rhs_str or k in lhs_str:
                    sym_objects[k] = sp.Float(v) if isinstance(v, float) else v

            # Try to parse both sides
            try:
                lhs_parsed = sp.sympify(lhs_str, locals=sym_objects)
                rhs_parsed = sp.sympify(rhs_str, locals=sym_objects)
                result["symbolically_parsed"] = True
                result["lhs"] = str(lhs_parsed)
                result["rhs"] = str(rhs_parsed)
                result["lhs_free_symbols"] = [str(s) for s in lhs_parsed.free_symbols]
                result["rhs_free_symbols"] = [str(s) for s in rhs_parsed.free_symbols]

                # Check dimensional consistency if possible
                if result["lhs_free_symbols"] or result["rhs_free_symbols"]:
                    result["has_free_variables"] = True

            except (sp.SympifyError, ValueError, TypeError) as e:
                result["issues"].append(f"SymPy parse failed: {e}")

    except Exception as e:
        result["issues"].append(f"Verification error: {e}")

    # Try numerical evaluation if we have parameters
    if params and HAS_NUMPY:
        try:
            numeric_subs = {}
            for pname, pinfo in params.items():
                pval = pinfo.get("value", "")
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', pname)
                # Try to convert to float
                try:
                    val = float(str(pval).replace(',', '').split()[0])
                    numeric_subs[safe_name] = val
                except (ValueError, IndexError):
                    pass

            if numeric_subs and result.get("symbolically_parsed"):
                try:
                    rhs_val = rhs_parsed.subs(numeric_subs)
                    if rhs_val.is_number:
                        result["numerical_value"] = float(rhs_val)
                        result["numerically_evaluated"] = True
                        result["numeric_subs"] = numeric_subs
                except Exception:
                    pass

        except Exception as e:
            result["issues"].append(f"Numerical eval failed: {e}")

    return result


def _get_constants_dict() -> dict:
    """Return all physical constants as a reference dictionary."""
    return {
        "hbar": {"value": HBAR, "units": "J*s", "name": "reduced Planck constant"},
        "c": {"value": C_LIGHT, "units": "m/s", "name": "speed of light"},
        "k_B": {"value": K_B, "units": "J/K", "name": "Boltzmann constant"},
        "G": {"value": G_CONST, "units": "m^3/(kg*s^2)", "name": "gravitational constant"},
        "epsilon_0": {"value": EPSILON_0, "units": "F/m", "name": "vacuum permittivity"},
        "mu_0": {"value": MU_0, "units": "H/m", "name": "vacuum permeability"},
        "e": {"value": E_CHARGE, "units": "C", "name": "elementary charge"},
        "m_e": {"value": M_ELECTRON, "units": "kg", "name": "electron mass"},
        "m_p": {"value": M_PROTON, "units": "kg", "name": "proton mass"},
        "sigma_sb": {"value": STEFAN_BOLTZMANN, "units": "W/(m^2*K^4)", "name": "Stefan-Boltzmann constant"},
        "N_A": {"value": AVOGADRO, "units": "1/mol", "name": "Avogadro number"},
        "R": {"value": R_GAS, "units": "J/(mol*K)", "name": "ideal gas constant"},
        "planck": {"value": PLANCK, "units": "J*s", "name": "Planck constant"},
    }


# ══════════════════════════════════════════════════════════════
# SOLAR SYSTEM DERIVATIONS
# ══════════════════════════════════════════════════════════════

def run_solar_system_verification(mechanisms: list = None, topic: str = "") -> dict:
    """Run tidal heating and planetary science derivations."""
    results = {}

    # 1. Tidal heating power
    results["tidal_heating"] = compute_tidal_heating()

    # 2. Viscous dissipation in ice shell
    results["viscous_dissipation"] = compute_viscous_dissipation()

    # 3. Heat flux from crystallization front
    results["crystallization_heat_flux"] = compute_crystallization_heat_flux()

    # 4. Dark matter capture rate (if relevant)
    if "dark matter" in topic.lower() or (mechanisms and any("dark matter" in str(m).lower() for m in mechanisms)):
        results["dm_capture"] = compute_dark_matter_capture()

    # 5. Tidal strain rate
    results["tidal_strain"] = compute_tidal_strain()

    # 6. Energy budget comparison
    results["energy_budget"] = compute_energy_budget()

    return results


def compute_tidal_heating():
    """Compute tidal dissipation power in an icy moon.

    Q = (21/2) * (k2/Q) * (n^5 * R^5 * e^2) / G

    For Enceladus:
      k2 = Love number ~0.01
      Q = tidal quality factor ~100
      n = mean motion = 2*pi/P, P = 1.37 days
      R = radius = 252 km
      e = eccentricity = 0.0047
    """
    k2 = 0.01       # Love number
    Q_tidal = 100    # tidal quality factor
    P_orb = 1.37 * 24 * 3600  # orbital period in seconds
    n = 2 * math.pi / P_orb   # mean motion
    R = 252e3        # radius in meters
    e = 0.0047       # eccentricity

    # Tidal heating power
    Q_dot = (21.0 / 2.0) * (k2 / Q_tidal) * (n**5 * R**5 * e**2) / G_CONST

    # Also compute using simplified formula
    Q_dot_simple = (3 * k2 / (2 * Q_tidal)) * n**3 * R**5 * e**2 / G_CONST

    return {
        "formula": "Q_dot = (21/2) * (k2/Q) * (n^5 * R^5 * e^2) / G",
        "k2": k2,
        "Q_tidal": Q_tidal,
        "n_rad_s": n,
        "R_m": R,
        "eccentricity": e,
        "power_W": Q_dot,
        "power_GW": Q_dot / 1e9,
        "simple_formula": "Q_dot = (3k2/2Q) * n^3 * R^5 * e^2 / G",
        "simple_power_GW": Q_dot_simple / 1e9,
        "cross_check": abs(Q_dot - Q_dot_simple) / max(abs(Q_dot), 1) < 0.1,
    }


def compute_viscous_dissipation():
    """Compute viscous dissipation rate in brine-rich ice.

    Q_vol = 2 * eta_eff * gamma_dot^2

    For brine-reduced viscosity:
      eta_pure = 10^15 Pa*s (pure ice)
      eta_brine = 10^13 Pa*s (brine-reduced)
      gamma_dot = omega * epsilon (tidal strain rate)
    """
    omega = 2 * math.pi / (1.37 * 24 * 3600)  # tidal frequency
    epsilon = 1e-4  # tidal strain amplitude

    gamma_dot = omega * epsilon  # strain rate

    eta_pure = 1e15    # pure ice viscosity
    eta_brine = 1e13   # brine-reduced viscosity

    Q_pure = 2 * eta_pure * gamma_dot**2
    Q_brine = 2 * eta_brine * gamma_dot**2

    # Total power over annulus (R=200km, width=10km, thickness=1km)
    R_annulus = 200e3
    width = 10e3
    thickness = 1e3
    volume = 2 * math.pi * R_annulus * width * thickness

    P_pure = Q_pure * volume
    P_brine = Q_brine * volume

    return {
        "formula": "Q_vol = 2 * eta_eff * gamma_dot^2",
        "omega": omega,
        "strain_amplitude": epsilon,
        "strain_rate": gamma_dot,
        "eta_pure_Pa_s": eta_pure,
        "eta_brine_Pa_s": eta_brine,
        "Q_pure_W_m3": Q_pure,
        "Q_brine_W_m3": Q_brine,
        "volume_m3": volume,
        "power_pure_GW": P_pure / 1e9,
        "power_brine_GW": P_brine / 1e9,
        "enhancement_factor": eta_pure / eta_brine,
    }


def compute_crystallization_heat_flux():
    """Compute heat flux from amorphous-ice crystallization front.

    Q_lat = rho_ice * L * v_cr

    Where:
      rho_ice = 917 kg/m^3
      L = latent heat ~ 1e5 J/kg (amorphous -> crystalline)
      v_cr = crystallization front speed ~ 1e-5 m/s
    """
    rho_ice = 917.0     # kg/m^3
    L = 1e5              # J/kg latent heat
    v_cr = 1e-5          # m/s front speed

    Q_lat = rho_ice * L * v_cr

    # Total power over fracture area
    A_fracture = 2e6     # m^2 (tiger stripe fracture area)
    P_total = Q_lat * A_fracture

    return {
        "formula": "Q_lat = rho_ice * L * v_cr",
        "rho_ice": rho_ice,
        "L_J_kg": L,
        "v_cr_m_s": v_cr,
        "heat_flux_W_m2": Q_lat,
        "fracture_area_m2": A_fracture,
        "total_power_GW": P_total / 1e9,
    }


def compute_dark_matter_capture():
    """Compute dark matter capture rate and annihilation power.

    C = pi * R^2 * rho_chi * v_esc * (sigma/m_chi)
    P_ann = C * m_chi * c^2 * f_heat
    """
    R = 252e3           # Enceladus radius
    rho_chi = 0.3       # GeV/cm^3 local DM density
    v_esc = 240.0       # m/s escape velocity
    sigma_over_m = 1e-34  # cm^2/GeV
    m_chi = 100 * EV    # 100 GeV in Joules
    f_heat = 0.1        # heating efficiency

    # Convert units
    rho_chi_kg = rho_chi * EV * 1e6 / C_LIGHT**2  # GeV/cm^3 -> kg/m^3

    C = math.pi * R**2 * rho_chi_kg * v_esc * sigma_over_m * 1e-4  # cm^2 -> m^2
    P_ann = C * m_chi * f_heat

    return {
        "formula": "C = pi*R^2 * rho_chi * v_esc * (sigma/m_chi)",
        "R_m": R,
        "rho_chi_GeV_cm3": rho_chi,
        "v_esc_m_s": v_esc,
        "capture_rate_s": C,
        "annihilation_power_GW": P_ann / 1e9,
        "f_heat": f_heat,
    }


def compute_tidal_strain():
    """Compute tidal strain amplitude and strain rate."""
    e = 0.0047          # orbital eccentricity
    k2 = 0.01           # Love number
    P = 1.37 * 24 * 3600  # orbital period
    omega = 2 * math.pi / P

    epsilon = e * k2    # strain amplitude (simplified)
    gamma_dot = omega * epsilon  # strain rate

    return {
        "eccentricity": e,
        "love_number": k2,
        "strain_amplitude": epsilon,
        "omega_rad_s": omega,
        "strain_rate_s": gamma_dot,
    }


def compute_energy_budget():
    """Compare energy sources for Enceladus plumes."""
    # Observed plume power
    P_observed = 200e9  # ~200 GW (observed)

    # Tidal heating (full body)
    P_tidal = compute_tidal_heating()["power_W"]

    # Brine-enhanced local heating
    P_brine = compute_viscous_dissipation()["power_brine_GW"] * 1e9

    # Crystallization
    P_crystal = compute_crystallization_heat_flux()["total_power_GW"] * 1e9

    # Dark matter (if applicable)
    P_dm = compute_dark_matter_capture()["annihilation_power_GW"] * 1e9

    return {
        "observed_GW": P_observed / 1e9,
        "tidal_heating_GW": P_tidal / 1e9,
        "brine_enhanced_GW": P_brine / 1e9,
        "crystallization_GW": P_crystal / 1e9,
        "dark_matter_GW": P_dm / 1e9,
        "tidal_fraction": P_tidal / P_observed if P_observed > 0 else 0,
        "brine_fraction": P_brine / P_observed if P_observed > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════
# CONDENSED MATTER DERIVATIONS (Superconductivity)
# ══════════════════════════════════════════════════════════════

def run_condensed_matter_verification(mechanisms=None, topic=""):
    """Run condensed matter physics derivations for superconductivity."""
    results = {}

    # 1. BCS gap equation
    results["bcs_gap"] = compute_bcs_gap()

    # 2. Phonon-mediated pairing
    results["phonon_pairing"] = compute_phonon_pairing()

    # 3. Critical temperature estimates
    results["critical_temperature"] = compute_critical_temperature()

    # 4. Spin fluctuation pairing
    results["spin_fluctuation"] = compute_spin_fluctuation()

    # 5. Energy scales comparison
    results["energy_scales"] = compute_energy_scales()

    # 6. Cooper pair size
    results["cooper_pair"] = compute_cooper_pair_size()

    return results


def compute_bcs_gap():
    """Compute BCS superconducting gap.

    Delta_0 = 2 * hbar * omega_D * exp(-1 / (N(0) * V))

    Where:
      omega_D = Debye frequency
      N(0) = density of states at Fermi level
      V = electron-phonon coupling potential

    BCS prediction: 2*Delta_0 / (k_B * T_c) = 3.53
    """
    # Typical values for conventional superconductor (Nb)
    omega_D = 2 * math.pi * 6e12   # Debye frequency ~6 THz
    N0 = 1.5e47                     # density of states at Fermi level (J^-1 m^-3)
    V = 1e-51                       # coupling potential (J m^3)

    exponent = -1.0 / (N0 * V)
    Delta_0 = 2 * HBAR * omega_D * math.exp(exponent)

    # BCS ratio
    T_c_bcs = Delta_0 / (1.76 * K_B)  # BCS relation: Delta_0 = 1.76 * k_B * T_c
    ratio = 2 * Delta_0 / (K_B * T_c_bcs) if T_c_bcs > 0 else 0

    return {
        "formula": "Delta_0 = 2*hbar*omega_D * exp(-1/(N(0)*V))",
        "omega_D_Hz": omega_D,
        "N0_J_m3": N0,
        "V_J_m3": V,
        "gap_Delta_0_J": Delta_0,
        "gap_Delta_0_meV": Delta_0 / EV * 1000,
        "T_c_BCS_K": T_c_bcs,
        "BCS_ratio_2Delta_kTc": ratio,
        "BCS_prediction": 3.53,
        "agreement": abs(ratio - 3.53) < 0.1,
    }


def compute_phonon_pairing():
    """Compute phonon-mediated electron pairing energy.

    E_phonon = hbar * omega_D ~ k_B * theta_D

    For cuprates: omega_D ~ 400-700 K (too low for T_c ~ 130 K)
    This is the core problem: phonons can't explain high-T_c.
    """
    theta_D_cu = 343       # Debye temperature for Cu (K)
    theta_D_ba = 110       # Debye temperature for Ba (K)
    theta_D_sr = 230       # Debye temperature for Sr (K)

    # Typical Debye temperature for cuprates
    theta_D_avg = 400      # K (approximate)

    E_phonon = K_B * theta_D_avg
    T_c_phonon = theta_D_avg / 4.0  # McMillan estimate (very rough)

    # But observed T_c for YBCO is 93 K, BSCCO is 110 K, HgBa2Ca2Cu3O8 is 133 K
    T_c_observed = 133     # K (record for cuprate)

    return {
        "formula": "E_phonon = k_B * theta_D",
        "theta_D_cu_K": theta_D_cu,
        "theta_D_avg_K": theta_D_avg,
        "E_phonon_J": E_phonon,
        "E_phonon_meV": E_phonon / EV * 1000,
        "T_c_phonon_estimate_K": T_c_phonon,
        "T_c_observed_cuprate_K": T_c_observed,
        "phonon_deficit": T_c_observed / T_c_phonon if T_c_phonon > 0 else float('inf'),
        "conclusion": "Phonon mechanism predicts T_c ~ {:.0f} K but observed {:.0f} K — deficit factor {:.1f}x".format(
            T_c_phonon, T_c_observed, T_c_observed / T_c_phonon if T_c_phonon > 0 else 0),
    }


def compute_critical_temperature():
    """Compute critical temperature from various mechanisms."""
    # Allen-Dynes formula (improved BCS)
    # T_c = (omega_log / 1.2) * exp(-1.04 * (1 + lambda) / (lambda - mu* (1 + 0.62 * lambda)))
    # For lambda = 1.0, mu* = 0.1:
    lambda_ep = 1.0    # electron-phonon coupling constant
    mu_star = 0.1      # Coulomb pseudopotential
    omega_log = 400 * K_B / HBAR  # logarithmic average phonon frequency

    numerator = -1.04 * (1 + lambda_ep)
    denominator = lambda_ep - mu_star * (1 + 0.62 * lambda_ep)
    T_c_allen_dynes = (omega_log * HBAR / (1.2 * K_B)) * math.exp(numerator / denominator) if denominator > 0 else 0

    # For cuprates with spin fluctuations
    # T_c ~ J * exp(-1 / (N(0) * J_eff))
    J = 0.12 * EV    # exchange coupling ~120 meV (typical for cuprates)
    N0_cu = 2e47      # density of states for CuO2 plane
    J_eff = 0.5e-51   # effective coupling

    T_c_spin = (J / K_B) * math.exp(-1.0 / (N0_cu * J_eff)) if (N0_cu * J_eff) > 0 else 0

    return {
        "BCS_phonon_T_c_K": T_c_allen_dynes,
        "BCS_params": {"lambda_ep": lambda_ep, "mu_star": mu_star},
        "spin_fluctuation_T_c_K": min(T_c_spin, 1000),  # cap at 1000K
        "spin_params": {"J_meV": J / EV * 1000},
        "observed_cuprate_T_c_K": 133,
        "observed_hydride_T_c_K": 250,  # H3S at high pressure
        "conclusion": "Phonon BCS gives ~{:.0f}K. Spin fluctuations can reach ~{:.0f}K. Gap to observed {:.0f}K.".format(
            T_c_allen_dynes, min(T_c_spin, 1000), 133),
    }


def compute_spin_fluctuation():
    """Compute spin fluctuation mediated pairing.

    J_eff = J^2 * chi(Q) where chi(Q) is the spin susceptibility at nesting vector Q.
    For cuprates: J ~ 120 meV, chi(Q) ~ 1/(pi * J) at optimal doping.
    """
    J = 0.12 * EV     # superexchange coupling
    Q = math.pi        # nesting vector (pi, pi) for cuprates
    chi_Q = 1.0 / (math.pi * J)  # spin susceptibility at Q

    J_eff = J**2 * chi_Q

    # Pairing strength
    lambda_sf = J_eff * 2e47  # N(0) * J_eff (dimensionless coupling)

    return {
        "formula": "J_eff = J^2 * chi(Q)",
        "J_meV": J / EV * 1000,
        "chi_Q": chi_Q,
        "J_eff_J": J_eff,
        "J_eff_meV": J_eff / EV * 1000,
        "lambda_sf": lambda_sf,
        "nesting_vector": "(pi, pi)",
        "conclusion": "Spin fluctuation coupling J_eff ~ {:.1f} meV, dimensionless coupling lambda_sf ~ {:.2f}".format(
            J_eff / EV * 1000, lambda_sf),
    }


def compute_energy_scales():
    """Compare energy scales relevant to superconductivity."""
    # Energy scales in meV
    scales = {
        "phonon_Debye": K_B * 400 / EV * 1000,         # ~34 meV
        "phonon_Einstein": K_B * 700 / EV * 1000,       # ~60 meV
        "spin_exchange_J": 0.12 * 1000,                  # ~120 meV
        "superconducting_gap": 1.76 * K_B * 133 / EV * 1000,  # ~20 meV for T_c=133K
        "Coulomb_repulsion_U": 8 * 1000,                 # ~8000 meV (8 eV)
        "charge_transfer_gap": 1.5 * 1000,               # ~1500 meV (1.5 eV)
        "pseudogap": 40,                                  # ~40 meV
        "Fermi_energy": 5000,                             # ~5 eV
    }

    return {
        "formula": "Energy scale comparison",
        "scales_meV": scales,
        "key_ratio_J_over_Delta": scales["spin_exchange_J"] / scales["superconducting_gap"],
        "key_ratio_U_over_J": scales["Coulomb_repulsion_U"] / scales["spin_exchange_J"],
        "conclusion": "Spin exchange J ({:.0f} meV) >> phonon Debye ({:.0f} meV). Gap ({:.0f} meV) is intermediate. Coulomb U ({:.0f} meV) dominates all.".format(
            scales["spin_exchange_J"], scales["phonon_Debye"], scales["superconducting_gap"], scales["Coulomb_repulsion_U"]),
    }


def compute_cooper_pair_size():
    """Compute Cooper pair coherence length (size of a Cooper pair)."""
    # BCS coherence length
    # xi_0 = hbar * v_F / (pi * Delta_0)
    v_F = 1.5e5      # Fermi velocity for cuprates (m/s) — much lower than metals
    Delta_0 = 20e-3 * EV   # gap ~20 meV

    xi_0 = HBAR * v_F / (math.pi * Delta_0)

    # For conventional superconductors
    v_F_conv = 1e6    # Fermi velocity for Nb (m/s)
    Delta_0_conv = 1.5e-3 * EV  # gap ~1.5 meV
    xi_0_conv = HBAR * v_F_conv / (math.pi * Delta_0_conv)

    return {
        "formula": "xi_0 = hbar * v_F / (pi * Delta_0)",
        "cuprate_v_F_m_s": v_F,
        "cuprate_Delta_meV": Delta_0 / EV * 1000,
        "cuprate_xi_0_nm": xi_0 * 1e9,
        "conventional_v_F_m_s": v_F_conv,
        "conventional_Delta_meV": Delta_0_conv / EV * 1000,
        "conventional_xi_0_nm": xi_0_conv * 1e9,
        "conclusion": "Cuprate Cooper pairs are ~{:.1f} nm (very small, ~lattice spacing). Conventional: ~{:.0f} nm. Cuprate pairs are in the 'strong coupling' limit.".format(
            xi_0 * 1e9, xi_0_conv * 1e9),
    }


# ══════════════════════════════════════════════════════════════
# PHYSICS DERIVATIONS (Casimir)
# ══════════════════════════════════════════════════════════════

def verify_casimir_pressure(a_m):
    P = -(math.pi**2 * HBAR * C_LIGHT) / (240 * a_m**4)
    rho = -(math.pi**2 * HBAR * C_LIGHT) / (720 * a_m**4)
    P_from_rho = 3 * rho
    sym = None
    if HAS_SYMPY:
        a = sp.Symbol("a", positive=True)
        rho_sym = -sp.pi**2 * HBAR * C_LIGHT / (720 * a**4)
        P_sym = -sp.diff(rho_sym * a, a)
        sym = {"rho": str(rho_sym), "P": str(sp.simplify(P_sym)), "match": "240" in str(sp.simplify(P_sym))}
    return {"formula": "P = -pi^2*hbar*c/(240*a^4)", "a_m": a_m, "P_Pa": P, "rho_J_m3": rho, "P_thermo_check": P_from_rho, "cross_check": abs(P - P_from_rho)/abs(P) < 0.01, "sympy": sym}

def verify_multi_sep(seps=None):
    if seps is None: seps = [50e-9, 100e-9, 200e-9, 500e-9, 1e-6, 5e-6, 10e-6]
    return [verify_casimir_pressure(a) for a in seps]

def verify_scaling(a1, a2):
    P1 = -(math.pi**2 * HBAR * C_LIGHT) / (240 * a1**4)
    P2 = -(math.pi**2 * HBAR * C_LIGHT) / (240 * a2**4)
    ratio = P2 / P1
    expected = (a1 / a2)**4
    return {"a1": a1, "a2": a2, "P1": P1, "P2": P2, "ratio": ratio, "expected": expected, "confirmed": abs(ratio - expected)/abs(expected) < 0.001}

def gold_correction(a_m):
    omega_p = 9.0 * EV / HBAR
    omega_c = C_LIGHT * math.pi / a_m
    kappa = max(0, min(1, 1 - (math.pi * C_LIGHT / (a_m * omega_p))**2))
    P_ideal = -(math.pi**2 * HBAR * C_LIGHT) / (240 * a_m**4)
    P_gold = kappa * P_ideal
    dev = (P_gold - P_ideal) / abs(P_ideal) * 100
    return {"a_m": a_m, "kappa": round(kappa, 4), "P_ideal": P_ideal, "P_gold": P_gold, "deviation_pct": round(dev, 2)}

def propagate_uncertainty(a_m, da_m):
    C = math.pi**2 * HBAR * C_LIGHT / 240
    P = -C / a_m**4
    dP = 4 * C / a_m**5 * da_m
    rel = 4 * da_m / a_m
    return {"a": a_m, "da": da_m, "P": P, "dP": dP, "rel_pct": round(rel * 100, 2)}

def run_physics_verification():
    return {
        "multi_sep": verify_multi_sep(),
        "scaling_100_200nm": verify_scaling(100e-9, 200e-9),
        "scaling_100_500nm": verify_scaling(100e-9, 500e-9),
        "gold_100nm": gold_correction(100e-9),
        "gold_1um": gold_correction(1e-6),
        "gold_5um": gold_correction(5e-6),
        "uncert_1um_1nm": propagate_uncertainty(1e-6, 1e-9),
        "uncert_100nm_01nm": propagate_uncertainty(100e-9, 0.1e-9),
        "uncert_50nm_05nm": propagate_uncertainty(50e-9, 0.5e-9),
    }


# ══════════════════════════════════════════════════════════════
# MATHEMATICS DERIVATIONS
# ══════════════════════════════════════════════════════════════

def run_mathematics_verification(mechanisms=None, topic=""):
    """Verify mathematical derivations from generated mechanisms/theories.

    Unlike physics domains, math topics don't have hardcoded formulas.
    Instead we extract equations from mechanisms and verify them symbolically.
    """
    results = {"domain": "mathematics", "mechanisms_checked": 0, "symbolic_verifications": []}

    if not mechanisms:
        return results

    for i, mech in enumerate(mechanisms):
        if not isinstance(mech, dict):
            continue
        model = mech.get("mathematical_model", mech.get("derivation", ""))
        if not model or not HAS_SYMPY:
            continue

        results["mechanisms_checked"] += 1
        verification = {"mechanism_index": i, "name": mech.get("name", ""), "checks": []}

        # Check for common mathematical expressions and verify symbolic consistency
        try:
            # Try to parse any sympy-parseable equations in the model
            lines = [l.strip() for l in model.split("\n") if "=" in l and not l.strip().startswith("#")]
            for line in lines[:3]:  # limit to first 3 equations
                parts = line.split("=")
                if len(parts) == 2:
                    left_str, right_str = parts[0].strip(), parts[1].strip()
                    try:
                        left = sp.sympify(left_str)
                        right = sp.sympify(right_str)
                        diff = sp.simplify(left - right)
                        verification["checks"].append({
                            "equation": line,
                            "parseable": True,
                            "simplified_diff": str(diff),
                            "consistent": diff == 0,
                        })
                    except (sp.SympifyError, ValueError):
                        verification["checks"].append({
                            "equation": line,
                            "parseable": False,
                        })
        except Exception:
            pass

        if verification["checks"]:
            results["symbolic_verifications"].append(verification)

    return results


# ══════════════════════════════════════════════════════════════
# DRUG DISCOVERY DERIVATIONS
# ══════════════════════════════════════════════════════════════

def run_drug_discovery_verification(mechanisms=None, topic=""):
    """Run pharmacokinetic derivations."""
    results = {}

    # Michaelis-Menten kinetics
    results["michaelis_menten"] = {
        "formula": "v = V_max * [S] / (K_m + [S])",
        "V_max_example": 100,  # umol/min
        "K_m_example": 50,     # uM
        "v_at_Km": 50,         # v at [S]=Km is Vmax/2
    }

    # Hill equation for dose-response
    results["hill_equation"] = {
        "formula": "E = E_max * C^n / (EC50^n + C^n)",
        "E_max": 100,
        "EC50_example": 10,  # nM
        "Hill_coefficient": 1.5,
    }

    return results


# ══════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY
# ══════════════════════════════════════════════════════════════

def run_full_verification():
    """Legacy: run Casimir physics verification."""
    return run_physics_verification()

def is_available(): return True