import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract given parameters with safe defaults (None)
        self.G = getattr(solution_instance, 'G', None)
        self.M = getattr(solution_instance, 'M', None)
        self.m = getattr(solution_instance, 'm', None)
        self.C = getattr(solution_instance, 'C', None)
        self.r0 = getattr(solution_instance, 'r0', None)
        # Extract result quantities (from solution_instance.results if available)
        self.T_circ = self._safe_get(solution_instance, 'T_circ')
        self.tau0 = self._safe_get(solution_instance, 'tau0')
        self.T_radial = self._safe_get(solution_instance, 'T_radial')

    def _safe_get(self, obj, key):
        """Try to get key from obj.results, else from obj directly, finally None."""
        if hasattr(obj, 'results') and isinstance(obj.results, dict):
            return obj.results.get(key, None)
        return getattr(obj, key, None)

    def _is_numeric(self, val):
        """Check if val is a numeric (int/float)."""
        return isinstance(val, (int, float)) and not isinstance(val, bool)

    def _is_symbolic(self, val):
        """Check if val is a sympy expression."""
        return isinstance(val, sp.Basic)

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
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface: check that required result keys exist
    # ------------------------------------------------------------------
    def test_interface_T_circ(self):
        if self.T_circ is None:
            return False, "T_circ missing from results"
        return True, "T_circ present"

    def test_interface_tau0(self):
        if self.tau0 is None:
            return False, "tau0 missing from results"
        return True, "tau0 present"

    def test_interface_T_radial(self):
        if self.T_radial is None:
            return False, "T_radial missing from results"
        return True, "T_radial present"

    # ------------------------------------------------------------------
    # Invariant checks: theorem-based physical sanity
    # ------------------------------------------------------------------
    def test_invariant_circular_period_relation(self):
        """T_circ should equal tau0 * (1 - C * tau0^2 / (8*pi^2))"""
        if self.T_circ is None or self.tau0 is None or self.C is None:
            return False, "SKIP: missing T_circ, tau0, or C"
        tau0 = self.tau0
        C = self.C
        if self._is_numeric(tau0) and self._is_numeric(C):
            expected = tau0 * (1 - C * tau0**2 / (8 * math.pi**2))
            if math.isclose(self.T_circ, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"T_circ ≈ {expected} matches within 1%"
            else:
                return False, f"T_circ = {self.T_circ}, expected ≈ {expected}"
        elif self._is_symbolic(tau0) and self._is_symbolic(C):
            expr = tau0 * (1 - C * tau0**2 / (8 * sp.pi**2))
            diff = sp.simplify(self.T_circ - expr)
            if diff == 0:
                return True, "Symbolic relation matches exactly"
            else:
                return False, f"Symbolic difference: {diff}"
        else:
            return False, "Cannot evaluate: mixed types"

    def test_invariant_radial_period_relation(self):
        """T_radial should equal tau0 * (1 - C * tau0^2 / (2*pi^2))"""
        if self.T_radial is None or self.tau0 is None or self.C is None:
            return False, "SKIP: missing T_radial, tau0, or C"
        tau0 = self.tau0
        C = self.C
        if self._is_numeric(tau0) and self._is_numeric(C):
            expected = tau0 * (1 - C * tau0**2 / (2 * math.pi**2))
            if math.isclose(self.T_radial, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"T_radial ≈ {expected} matches within 1%"
            else:
                return False, f"T_radial = {self.T_radial}, expected ≈ {expected}"
        elif self._is_symbolic(tau0) and self._is_symbolic(C):
            expr = tau0 * (1 - C * tau0**2 / (2 * sp.pi**2))
            diff = sp.simplify(self.T_radial - expr)
            if diff == 0:
                return True, "Symbolic relation matches exactly"
            else:
                return False, f"Symbolic difference: {diff}"
        else:
            return False, "Cannot evaluate: mixed types"

    def test_invariant_tau0_keplerian_form(self):
        """If G, M, r0 are given, tau0 should be 2*pi*sqrt(r0^3/(G*M))"""
        if self.tau0 is None or self.G is None or self.M is None or self.r0 is None:
            return False, "SKIP: missing tau0 or G, M, r0"
        G, M, r0 = self.G, self.M, self.r0
        tau0 = self.tau0
        if self._is_numeric(G) and self._is_numeric(M) and self._is_numeric(r0) and self._is_numeric(tau0):
            expected = 2 * math.pi * math.sqrt(r0**3 / (G * M))
            if math.isclose(tau0, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, "tau0 matches Keplerian period"
            else:
                return False, f"tau0 = {tau0}, expected ≈ {expected}"
        elif self._is_symbolic(tau0) and self._is_symbolic(G) and self._is_symbolic(M) and self._is_symbolic(r0):
            expected = 2 * sp.pi * sp.sqrt(r0**3 / (G * M))
            diff = sp.simplify(tau0 - expected)
            if diff == 0:
                return True, "Symbolic tau0 matches Keplerian period"
            else:
                return False, f"Symbolic difference: {diff}"
        else:
            return False, "Cannot evaluate: mixed types"

    # ------------------------------------------------------------------
    # Modeling quality checks based on explanation steps
    # ------------------------------------------------------------------
    def test_modeling_radial_less_than_circular(self):
        """From expansions: T_radial < T_circ (since C>0)"""
        if self.T_circ is None or self.T_radial is None:
            return False, "SKIP: missing T_circ or T_radial"
        if self._is_numeric(self.T_circ) and self._is_numeric(self.T_radial):
            if self.T_radial < self.T_circ:
                return True, "T_radial < T_circ holds"
            else:
                return False, f"T_radial ({self.T_radial}) >= T_circ ({self.T_circ})"
        elif self._is_symbolic(self.T_circ) and self._is_symbolic(self.T_radial):
            diff = self.T_circ - self.T_radial
            if self.C is not None and self._is_symbolic(self.C):
                # T_circ - T_radial = C*tau0^2*(1/(2π^2) - 1/(8π^2)) = (3/8π^2)*C*tau0^2 > 0
                factor = sp.Rational(3, 8) * self.C * (self.tau0**2) / (sp.pi**2)
                if sp.simplify(diff - factor) == 0:
                    return True, "Symbolic inequality holds (C>0)"
                else:
                    return False, f"Symbolic difference not matching: diff = {diff}"
            else:
                return False, "Cannot determine sign without symbolic C"
        else:
            return False, "Cannot evaluate: mixed types"

    def test_modeling_tau0_positive(self):
        """tau0 must be positive (Keplerian period)"""
        if self.tau0 is None:
            return False, "SKIP: no tau0"
        if self._is_numeric(self.tau0):
            if self.tau0 > 0:
                return True, "tau0 positive"
            else:
                return False, f"tau0 = {self.tau0} <= 0"
        else:
            return False, "Cannot numerically assert positivity"

    # ------------------------------------------------------------------
    # Monotonicity: perturb an input (C) and observe effect on period
    # ------------------------------------------------------------------
    def test_monotonicity_T_circ_with_C(self):
        """Doubling C should reduce T_circ (since C appears with negative sign)."""
        if self.T_circ is None or self.tau0 is None or self.C is None:
            return False, "SKIP: missing T_circ, tau0, or C"
        # Ensure numeric values are present for perturbation
        if not (self._is_numeric(self.tau0) and self._is_numeric(self.C)):
            return False, "SKIP: need numeric tau0 and C"
        # Baseline
        baseline_T = self.T_circ
        # Perturb: double C
        C_new = 2 * self.C
        # Compute new T_circ using the theorem-based formula (re-derived in checker)
        expected_new = self.tau0 * (1 - C_new * self.tau0**2 / (8 * math.pi**2))
        # Check that new period is smaller than baseline
        if expected_new < baseline_T:
            return True, f"Doubling C reduces T_circ from {baseline_T} to {expected_new}"
        else:
            return False, f"Monotonicity violated: doubling C gave {expected_new} >= {baseline_T}"

    def test_monotonicity_T_radial_with_C(self):
        """Doubling C should reduce T_radial (since C appears with negative sign)."""
        if self.T_radial is None or self.tau0 is None or self.C is None:
            return False, "SKIP: missing T_radial, tau0, or C"
        if not (self._is_numeric(self.tau0) and self._is_numeric(self.C)):
            return False, "SKIP: need numeric tau0 and C"
        baseline_T = self.T_radial
        C_new = 2 * self.C
        expected_new = self.tau0 * (1 - C_new * self.tau0**2 / (2 * math.pi**2))
        if expected_new < baseline_T:
            return True, f"Doubling C reduces T_radial from {baseline_T} to {expected_new}"
        else:
            return False, f"Monotonicity violated: doubling C gave {expected_new} >= {baseline_T}"

    # ------------------------------------------------------------------
    # Answer sanity (weak, no close/hardcoded values)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        """All periods should be positive and finite."""
        values = [self.T_circ, self.tau0, self.T_radial]
        missing = any(v is None for v in values)
        if missing:
            return False, "SKIP: some results missing"
        for name, val in zip(["T_circ", "tau0", "T_radial"], values):
            if self._is_numeric(val):
                if not (val > 0 and math.isfinite(val)):
                    return False, f"{name} = {val} not positive finite"
        return True, "All periods positive and finite"