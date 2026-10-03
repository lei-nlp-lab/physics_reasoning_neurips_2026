import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (try multiple sources)
        self.hbar = None
        self.m = None
        self.g = None
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.hbar = solution_instance.givens.get('hbar', None)
            self.m = solution_instance.givens.get('m', None)
            self.g = solution_instance.givens.get('g', None)
        else:
            for attr in ['hbar', 'm', 'g']:
                val = getattr(solution_instance, attr, None)
                if val is not None:
                    setattr(self, attr, val)
        # Extract results
        results = getattr(solution_instance, 'results', {}) or {}
        self.alpha0 = results.get('alpha0', None)
        self.E_min = results.get('E_min', None)

        # Also try to extract from solution_instance directly if not in results
        if self.alpha0 is None:
            self.alpha0 = getattr(solution_instance, 'alpha0', None)
        if self.E_min is None:
            self.E_min = getattr(solution_instance, 'E_min', None)

        # Store constants
        self.pi = math.pi

    # --------------------------------------------------------------
    # Interface tests: ensure required keys exist and are valid numbers
    # --------------------------------------------------------------
    def test_interface_alpha0_exists(self):
        if self.alpha0 is None:
            return (False, "alpha0 not provided (None)")
        return (True, "alpha0 is present")

    def test_interface_E_min_exists(self):
        if self.E_min is None:
            return (False, "E_min not provided (None)")
        return (True, "E_min is present")

    def test_interface_alpha0_positive(self):
        if self.alpha0 is None:
            return (False, "SKIP: alpha0 not available")
        if not isinstance(self.alpha0, (int, float)):
            return (False, "alpha0 is not numeric")
        if self.alpha0 <= 0:
            return (False, f"alpha0 = {self.alpha0} is not positive")
        return (True, "alpha0 is positive")

    def test_interface_E_min_finite(self):
        if self.E_min is None:
            return (False, "SKIP: E_min not available")
        if not isinstance(self.E_min, (int, float)):
            return (False, "E_min is not numeric")
        if not math.isfinite(self.E_min):
            return (False, f"E_min is not finite: {self.E_min}")
        return (True, "E_min is finite")

    # --------------------------------------------------------------
    # Invariant tests: theorem‑based physical sanity (use isclose)
    # --------------------------------------------------------------
    def test_invariant_energy_negative(self):
        if self.E_min is None:
            return (False, "SKIP: E_min not available")
        if not isinstance(self.E_min, (int, float)):
            return (False, "E_min is not numeric")
        if self.E_min >= 0:
            return (False, f"E_min = {self.E_min} is not negative (attractive potential)")
        return (True, "E_min is negative")

    def test_invariant_alpha0_positive(self):
        if self.alpha0 is None:
            return (False, "SKIP: alpha0 not available")
        if self.alpha0 <= 0:
            return (False, f"alpha0 = {self.alpha0} is not positive")
        return (True, "alpha0 is positive")

    # --------------------------------------------------------------
    # Modeling tests: check internal consistency of derived quantities
    # --------------------------------------------------------------
    def test_modeling_E_min_from_alpha0(self):
        """Check that given E_min is consistent with given alpha0 and the
        variational formula E(α) = ħ²α²/(2m) - g²√(π/2) α^(3/2)."""
        if any(x is None for x in [self.hbar, self.m, self.g, self.alpha0, self.E_min]):
            return (False, "SKIP: missing one or more required givens (hbar, m, g, alpha0, E_min)")
        if not all(isinstance(x, (int, float)) for x in [self.hbar, self.m, self.g, self.alpha0, self.E_min]):
            return (False, "Non‑numeric value encountered")
        expected_E = (self.hbar**2 * self.alpha0**2) / (2 * self.m) - (self.g**2 * math.sqrt(self.pi / 2) * self.alpha0**1.5)
        if not math.isclose(expected_E, self.E_min, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E(alpha0) formula gives {expected_E}, but E_min = {self.E_min}")
        return (True, "E_min matches variational formula using given alpha0")

    def test_modeling_alpha0_from_minimization(self):
        """Check that alpha0 satisfies dE/dα = 0 condition:
        ħ²α/m - (3/2)g²√(π/2) α^(1/2) = 0."""
        if any(x is None for x in [self.hbar, self.m, self.g, self.alpha0]):
            return (False, "SKIP: missing one or more required givens (hbar, m, g, alpha0)")
        if not all(isinstance(x, (int, float)) for x in [self.hbar, self.m, self.g, self.alpha0]):
            return (False, "Non‑numeric value encountered")
        derivative = (self.hbar**2 * self.alpha0) / self.m - 1.5 * (self.g**2 * math.sqrt(self.pi / 2) * math.sqrt(self.alpha0))
        if not math.isclose(derivative, 0, abs_tol=1e-2):
            return (False, f"dE/dα at alpha0 = {derivative} (should be ~0)")
        return (True, "dE/dα = 0 condition satisfied by alpha0")

    # --------------------------------------------------------------
    # Monotonicity tests: explicitly perturb one input, compare the
    # *re‑optimized* variational bound against the original bound.
    # --------------------------------------------------------------
    def test_monotonicity_E_min_decreases_with_increasing_g(self):
        """Increase g by 5% and recompute the optimal α0 and E_min.
        Because α0 ∝ g^4 and E_min ∝ -g^8, the bound must become
        more negative (decrease)."""
        if any(x is None for x in [self.hbar, self.m, self.g, self.E_min]):
            return (False, "SKIP: missing required givens (hbar, m, g, E_min)")
        if not all(isinstance(x, (int, float)) for x in [self.hbar, self.m, self.g, self.E_min]):
            return (False, "Non‑numeric value")

        g_new = self.g * 1.05
        # Re‑optimize: α0_new = (9 π g_new^4 m^2) / (8 ħ^4)
        alpha0_new = (9 * self.pi * (g_new**4) * (self.m**2)) / (8 * (self.hbar**4))
        # E_min_new = ħ^2 α0_new^2/(2m) - g_new^2 sqrt(π/2) α0_new^(3/2)
        term1 = (self.hbar**2 * alpha0_new**2) / (2 * self.m)
        term2 = (g_new**2 * math.sqrt(self.pi / 2) * alpha0_new**1.5)
        E_new = term1 - term2

        # Verify that E_new is more negative (lower) than the original E_min
        if E_new >= self.E_min:
            return (False, f"Increasing g by 5% gives E_new = {E_new}, which is not less than original E_min = {self.E_min}")
        return (True, "E_min decreases (more negative) when g increases")

    def test_monotonicity_E_min_less_negative_with_increasing_hbar(self):
        """Increase hbar by 5% and recompute the optimal α0 and E_min.
        Because α0 ∝ 1/ħ^4 and E_min ∝ -1/ħ^6, the bound must become
        less negative (increase)."""
        if any(x is None for x in [self.hbar, self.m, self.g, self.E_min]):
            return (False, "SKIP: missing required givens (hbar, m, g, E_min)")
        if not all(isinstance(x, (int, float)) for x in [self.hbar, self.m, self.g, self.E_min]):
            return (False, "Non‑numeric value")

        hbar_new = self.hbar * 1.05
        # Re‑optimize: α0_new = (9 π g^4 m^2) / (8 ħ_new^4)
        alpha0_new = (9 * self.pi * (self.g**4) * (self.m**2)) / (8 * (hbar_new**4))
        # E_min_new = ħ_new^2 α0_new^2/(2m) - g^2 sqrt(π/2) α0_new^(3/2)
        term1 = (hbar_new**2 * alpha0_new**2) / (2 * self.m)
        term2 = (self.g**2 * math.sqrt(self.pi / 2) * alpha0_new**1.5)
        E_new = term1 - term2

        # Verify that E_new is higher (less negative) than the original E_min
        if E_new <= self.E_min:
            return (False, f"Increasing hbar by 5% gives E_new = {E_new}, which is not greater than original E_min = {self.E_min}")
        return (True, "E_min becomes less negative (increases) when hbar increases")

    # --------------------------------------------------------------
    # Weak answer sanity test (only sign, finiteness, magnitude)
    # --------------------------------------------------------------
    def test_answer_sanity_sign_and_magnitude(self):
        if self.E_min is None:
            return (False, "SKIP: E_min not available")
        if not isinstance(self.E_min, (int, float)):
            return (False, "E_min not numeric")
        if not math.isfinite(self.E_min):
            return (False, "E_min not finite")
        if self.E_min >= 0:
            return (False, "E_min should be negative for bound state")
        # Check that magnitude is not ridiculously large (e.g., > 1e100)
        if abs(self.E_min) > 1e100:
            return (False, f"Magnitude of E_min too large: {self.E_min}")
        return (True, "E_min negative and finite")

    # --------------------------------------------------------------
    # Test runner: discover and execute all test_ methods
    # --------------------------------------------------------------
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