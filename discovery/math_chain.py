"""
math_chain.py - Explicit math derivation chain for mechanisms.

Pipeline: Mechanism -> Equation -> Parameter -> Derivation -> Numerical Example
"""


class MathChain:
    def __init__(self, llm_call=None):
        self.llm_call = llm_call

    def process_mechanisms(self, mechanisms, topic, domain):
        if not self.llm_call or not mechanisms:
            return mechanisms
        for m in mechanisms:
            if not isinstance(m, dict):
                continue
            name = m.get("name", "?")
            desc = m.get("description", m.get("mechanism", ""))
            existing_model = m.get("mathematical_model", "")
            if existing_model and "equation to be derived" not in existing_model.lower():
                if len(existing_model) > 20:
                    continue
            eq_result = self._derive_equation(name, desc, topic, domain)
            if eq_result:
                m["mathematical_model"] = eq_result.get("equation", existing_model)
                m["equation_description"] = eq_result.get("description", "")
                m["equation_variables"] = eq_result.get("variables", [])
            eq = m.get("mathematical_model", "")
            if eq and "equation to be derived" not in eq.lower():
                param_result = self._estimate_parameters(name, eq, desc, topic, domain)
                if param_result:
                    m["key_parameters"] = param_result.get("parameters", [])
                    m["derivation_steps"] = param_result.get("derivation_steps", [])
            params = m.get("key_parameters", [])
            if eq and params and "equation to be derived" not in eq.lower():
                num_result = self._numerical_validation(name, eq, params, desc, topic, domain)
                if num_result:
                    m["numerical_example"] = num_result.get("example", "")
                    m["numerical_result"] = num_result.get("result", "")
                    m["physical_plausibility"] = num_result.get("plausibility", "")
        return mechanisms

    def _derive_equation(self, name, description, topic, domain):
        if not description or len(description) < 20:
            return None
        prompt = (
            "You are a theoretical physicist. Given this mechanism, derive the governing equation.\n\n"
            "MECHANISM: " + name + "\n"
            "DESCRIPTION: " + description[:1200] + "\n"
            "TOPIC: " + topic + "\n"
            "DOMAIN: " + domain + "\n\n"
            "Your task:\n"
            "1. Identify what physical quantity this mechanism affects\n"
            "2. Write the governing equation using standard physics notation\n"
            "3. Define ALL variables with their physical meaning and units\n"
            "4. If derivable from known physics, show the derivation steps\n"
            "5. If no exact equation exists, write the best approximate equation\n\n"
            "BAD: The rate depends on temperature (no equation)\n"
            "GOOD: R(omega) = 1 - exp(-omega/omega_c) where omega_c is the cutoff frequency\n"
            "BETTER: From Fermi's golden rate: R = 2*pi/ℎ |<f|H'|i>|^2 * rho(E), simplified to R(omega) = 1 - exp(-omega/omega_c)\n\n"
            "Output JSON:\n"
            '{"equation": "the equation in LaTeX or standard notation", '
            '"derivation_steps": ["Step 1: start from [known law]", "Step 2: apply [approximation]", "Step 3: arrive at equation"], '
            '"description": "what it means physically", '
            '"variables": [{"symbol": "R(omega)", "meaning": "reflectivity", "units": "dimensionless"}], '
             '"derivation_basis": "from quantum field theory"}'
        )
        try:
            from discovery.llm_client import call_json
            raw = call_json(prompt, max_tokens=4096, provider="auto")
            if not raw:
                raw = self.llm_call(prompt, max_tokens=4096)
            if raw:
                from discovery.json_extract import extract_json
                result = extract_json(raw)
                if result and isinstance(result, dict) and result.get("equation"):
                    return result
        except Exception:
            pass
        return None

    def _estimate_parameters(self, name, equation, description, topic, domain):
        prompt = (
            "You are a physicist estimating parameters for a mechanism.\n\n"
            "MECHANISM: " + name + "\n"
            "EQUATION: " + equation + "\n"
            "DESCRIPTION: " + description[:800] + "\n"
            "TOPIC: " + topic + "\n"
            "DOMAIN: " + domain + "\n\n"
            "For each variable in the equation:\n"
            "1. If known constant (G, c, hbar, k_B, etc.), cite exact value and source\n"
            "2. If derivable from other parameters, SHOW the derivation step by step\n"
            "3. If estimated, give order-of-magnitude with physical reasoning\n\n"
            "Also provide a complete step-by-step derivation from first principles to the equation.\n"
            "Start from known physics laws and show every algebraic step.\n\n"
            "Output JSON:\n"
            '{"parameters": [{"name": "param_name", "value": "numeric value", "units": "units", '
            '"source": "derived|cited|estimated", "source_detail": "paper citation or derivation steps"}],\n'
            ' "derivation_steps": ["Step 1: Start from [law/equation]", "Step 2: Apply [approximation]", '
            '"Step 3: Substitute values", "Step 4: Arrive at final expression"],\n'
             ' "first_principles_derivation": "Complete derivation from fundamental physics"}'
        )
        try:
            from discovery.llm_client import call_json
            raw = call_json(prompt, max_tokens=4096, provider="auto")
            if not raw:
                raw = self.llm_call(prompt, max_tokens=4096)
            if raw:
                from discovery.json_extract import extract_json
                result = extract_json(raw)
                if result and isinstance(result, dict):
                    return result
        except Exception:
            pass
        return None

    def _numerical_validation(self, name, equation, parameters, description, topic, domain):
        param_text = ""
        for p in (parameters or [])[:10]:
            if isinstance(p, dict):
                param_text += "  {} = {} {} (source: {})\n".format(
                    p.get("name", "?"), p.get("value", "?"), p.get("units", ""), p.get("source", "?"))
        prompt = (
            "You are a physicist doing a rigorous numerical verification.\n\n"
            "MECHANISM: " + name + "\n"
            "EQUATION: " + equation + "\n"
            "PARAMETERS:\n" + param_text +
            "DESCRIPTION: " + description[:600] + "\n\n"
            "Your task:\n"
            "1. Plug in the parameter values into the equation\n"
            "2. Show the computation step by step (intermediate results)\n"
            "3. Compute the final numerical result with units\n"
            "4. Check if the result is physically reasonable (order of magnitude, signs, bounds)\n"
            "5. Compare with any known experimental or observational values\n\n"
            "BAD: The effect is significant (no numbers)\n"
            "GOOD: For M = 10 M_sun, omega_c = 3e4 rad/s:\n"
            "  R(10^15 Hz) = 1 - exp(-10^15 / 3e4) = 1 - exp(-3.3e10) ≈ 1.0\n"
            "  This means essentially all radiation is absorbed — consistent with blackbody behavior.\n\n"
            "Output JSON:\n"
            '{"computation_steps": ["Step 1: substitute values", "Step 2: compute intermediate", "Step 3: final result"], '
            '"result": "R = 0.73 meaning the mechanism absorbs 73% of incident radiation", '
            '"numerical_value": 0.73, "units": "dimensionless", '
            '"plausibility": "consistent with Planck blackbody spectrum at T=5778K", '
             '"comparison_with_experiment": "Matches solar irradiance measurements within 5%"}'
        )
        try:
            from discovery.llm_client import call_json
            raw = call_json(prompt, max_tokens=4096, provider="auto")
            if not raw:
                raw = self.llm_call(prompt, max_tokens=4096)
            if raw:
                from discovery.json_extract import extract_json
                result = extract_json(raw)
                if result and isinstance(result, dict):
                    return result
        except Exception:
            pass
        return None
