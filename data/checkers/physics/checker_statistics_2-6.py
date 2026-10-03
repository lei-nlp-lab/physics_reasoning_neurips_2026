import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results with None defaults (no numeric fallbacks)
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.entropy_def = self.results.get('entropy_def', None)
        self.delta_S_formula = self.results.get('Delta_S_formula', None)
        # Keep reference to solution_instance for explanation steps
        self.solution_instance = solution_instance

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_entropy_def_key(self):
        if self.entropy_def is None:
            return False, "Key 'entropy_def' missing or None."
        if not isinstance(self.entropy_def, str):
            return False, f"'entropy_def' should be a string, got {type(self.entropy_def)}."
        return True, "'entropy_def' key present and is a string."

    def test_interface_delta_S_formula_key(self):
        if self.delta_S_formula is None:
            return False, "Key 'Delta_S_formula' missing or None."
        if not isinstance(self.delta_S_formula, str):
            return False, f"'Delta_S_formula' should be a string, got {type(self.delta_S_formula)}."
        return True, "'Delta_S_formula' key present and is a string."

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, may use isclose)
    # ------------------------------------------------------------------
    def test_invariant_entropy_definition(self):
        """Check that the definition matches S = k ln Omega."""
        if self.entropy_def is None:
            return False, "SKIP: entropy_def not available."
        defn = self.entropy_def.lower().replace(" ", "")
        # Look for 'k' and 'ln' and some form of omega
        has_k = 'k' in defn
        has_ln = 'ln' in defn
        has_omega = 'omega' in defn or 'ω' in defn
        if has_k and has_ln and has_omega:
            return True, "Definition includes k ln Omega."
        # Fallback: try to parse as symbolic expression
        try:
            S, k, Omega = sp.symbols('S k Omega')
            expr = self.entropy_def.replace('ln', 'log').replace('\\ln', 'log')
            expr = expr.replace('\\Omega', 'Omega').replace('Ω', 'Omega')
            parsed = sp.sympify(expr, locals={'k': k, 'Omega': Omega, 'S': S, 'log': sp.log})
            if sp.simplify(parsed - (k * sp.log(Omega))) == 0:
                return True, "Symbolic check: S = k ln Omega."
        except Exception:
            pass
        return False, f"Definition '{self.entropy_def}' does not match S = k ln Omega."

    def test_invariant_delta_S_nonnegative(self):
        """Check that the formula implies ΔS ≥ 0 for spontaneous heat flow."""
        if self.delta_S_formula is None:
            return False, "SKIP: Delta_S_formula not available."
        try:
            TB, TC, dE = sp.symbols('TB TC dE', positive=True)
            formula = self.delta_S_formula.replace('T_B', 'TB').replace('T_C', 'TC')
            formula = formula.replace('Delta_E', 'dE').replace('ΔE', 'dE').replace('delta_E', 'dE')
            parsed = sp.sympify(formula, locals={'TB': TB, 'TC': TC, 'dE': dE})
            # Evaluate at a typical hot-to-cold scenario: TB=5, TC=3, dE=2
            subs = {TB: 5, TC: 3, dE: 2}
            val = float(parsed.subs(subs).evalf())
            if val > 0:
                return True, f"ΔS > 0 for hot-to-cold flow (value = {val})."
            return False, f"ΔS should be >0 for hot-to-cold, got {val}."
        except Exception as e:
            return False, f"Cannot parse or evaluate Delta_S formula: {e}"

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_delta_S_derivation(self):
        """Check that explanation steps mention Taylor expansion or derivatives."""
        steps = getattr(self.solution_instance, 'explanation_steps', None) or \
                getattr(self.solution_instance, 'explanation', None) or ""
        if isinstance(steps, list):
            steps = " ".join(steps)
        if not isinstance(steps, str):
            return False, "SKIP: No explanation steps available."
        if any(word in steps.lower() for word in ['taylor', 'expand', 'ds/de', 'first order']):
            return True, "Steps mention Taylor expansion or derivative."
        return False, "Steps do not mention Taylor expansion; derivation may be incomplete."

    def test_modeling_entropy_additivity(self):
        """Check that explanation steps mention S_A = S_B + S_C."""
        steps = getattr(self.solution_instance, 'explanation_steps', None) or \
                getattr(self.solution_instance, 'explanation', None) or ""
        if isinstance(steps, list):
            steps = " ".join(steps)
        if not isinstance(steps, str):
            return False, "SKIP: No explanation steps available."
        if any(phrase in steps for phrase in ['S_A', 'S_B', 'additive', 'sum']):
            return True, "Steps mention entropy additivity."
        return False, "Entropy additivity not mentioned in steps."

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb input, compare outputs)
    # ------------------------------------------------------------------
    def test_monotonicity_delta_S_increases_with_T_B(self):
        """Perturb T_B upward and verify ΔS increases (when T_B > T_C)."""
        if self.delta_S_formula is None:
            return False, "SKIP: Delta_S_formula not available."
        try:
            TB, TC, dE = sp.symbols('TB TC dE', positive=True)
            formula = self.delta_S_formula.replace('T_B', 'TB').replace('T_C', 'TC')
            formula = formula.replace('Delta_E', 'dE').replace('ΔE', 'dE').replace('delta_E', 'dE')
            parsed = sp.sympify(formula, locals={'TB': TB, 'TC': TC, 'dE': dE})
            # Baseline case: T_B0=5, T_C=3, dE=2
            base_val = float(parsed.subs({TB: 5, TC: 3, dE: 2}).evalf())
            # Perturbed: increase T_B to 6
            perturbed_val = float(parsed.subs({TB: 6, TC: 3, dE: 2}).evalf())
            if perturbed_val > base_val:
                return True, f"ΔS increased from {base_val:.4f} to {perturbed_val:.4f} when T_B increased."
            else:
                return False, f"ΔS did not increase: base {base_val}, perturbed {perturbed_val}."
        except Exception as e:
            return False, f"Cannot compute monotonicity: {e}"

    def test_monotonicity_delta_S_increases_with_energy_transfer(self):
        """Perturb Delta_E upward and verify ΔS increases (when T_B > T_C)."""
        if self.delta_S_formula is None:
            return False, "SKIP: Delta_S_formula not available."
        try:
            TB, TC, dE = sp.symbols('TB TC dE', positive=True)
            formula = self.delta_S_formula.replace('T_B', 'TB').replace('T_C', 'TC')
            formula = formula.replace('Delta_E', 'dE').replace('ΔE', 'dE').replace('delta_E', 'dE')
            parsed = sp.sympify(formula, locals={'TB': TB, 'TC': TC, 'dE': dE})
            # Baseline: T_B=5, T_C=3, dE=2
            base_val = float(parsed.subs({TB: 5, TC: 3, dE: 2}).evalf())
            # Perturbed: increase dE to 3
            perturbed_val = float(parsed.subs({TB: 5, TC: 3, dE: 3}).evalf())
            if perturbed_val > base_val:
                return True, f"ΔS increased from {base_val:.4f} to {perturbed_val:.4f} when Delta_E increased."
            else:
                return False, f"ΔS did not increase: base {base_val}, perturbed {perturbed_val}."
        except Exception as e:
            return False, f"Cannot compute monotonicity: {e}"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no re-derivation, no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_entropy_def_format(self):
        if self.entropy_def is None:
            return False, "No entropy definition provided."
        if not isinstance(self.entropy_def, str):
            return False, "Entropy definition must be a string."
        if len(self.entropy_def.strip()) == 0:
            return False, "Entropy definition is empty."
        if '...' in self.entropy_def:
            return False, "Entropy definition appears incomplete."
        return True, "Entropy definition provided and non-empty."

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)