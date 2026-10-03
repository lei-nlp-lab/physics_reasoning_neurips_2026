import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Fresnel reflection/transmission and radiation pressure problem.
    All givens and results are extracted with None defaults and checked in tests.
    Unit-test style: interface, invariant, modeling, monotonicity, weak sanity.
    """

    def __init__(self, solution_instance):
        # Excise input givens (problem‑specific; if not provided, tests skip)
        self.S_i = getattr(solution_instance, 'S_i', None)
        self.n1 = getattr(solution_instance, 'n1', None)
        self.n2 = getattr(solution_instance, 'n2', None)
        self.c = getattr(solution_instance, 'c', None)

        # Extract result quantities (prefer results dict, fall back to attributes)
        results = getattr(solution_instance, 'results', {}) or {}
        self.R = results.get('R', None)
        self.T = results.get('T', None)
        self.p = results.get('p', None)

        # Fallback to direct attributes if not in results dict
        if self.R is None:
            self.R = getattr(solution_instance, 'R', None)
        if self.T is None:
            self.T = getattr(solution_instance, 'T', None)
        if self.p is None:
            self.p = getattr(solution_instance, 'p', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All three result keys (R, T, p) exist and are numeric."""
        if any(v is None for v in [self.R, self.T, self.p]):
            return (False, f"Missing key(s): R={self.R}, T={self.T}, p={self.p}")
        if not all(isinstance(v, (int, float)) for v in [self.R, self.T, self.p]):
            return (False, f"Non‑numeric value: R={self.R}, T={self.T}, p={self.p}")
        return (True, "All required keys present and numeric")

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based, use math.isclose where needed)
    # ------------------------------------------------------------------
    def test_invariant_R_plus_T(self):
        """Energy conservation: R + T = 1."""
        if self.R is None or self.T is None:
            return (False, "SKIP: R or T missing")
        if not math.isclose(self.R + self.T, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"R+T = {self.R + self.T:.6f}, expected ≈ 1")
        return (True, f"R+T = {self.R + self.T:.6f} ≈ 1")

    def test_invariant_R_T_bounded(self):
        """
        Both intensity coefficients must be between 0 and 1.
        This is a fundamental property of energy fractions (no negative intensities,
        no more than total energy).
        """
        if self.R is None or self.T is None:
            return (False, "SKIP: R or T missing")
        if self.R < 0 or self.R > 1:
            return (False, f"R = {self.R:.6f} out of [0,1]")
        if self.T < 0 or self.T > 1:
            return (False, f"T = {self.T:.6f} out of [0,1]")
        return (True, f"R = {self.R:.6f}, T = {self.T:.6f} both in [0,1]")

    # ------------------------------------------------------------------
    # Modeling / trace checks (re‑derive from givens; use math.isclose)
    # ------------------------------------------------------------------
    def test_modeling_fresnel_R(self):
        """Reflection coefficient matches R = ((n1-n2)/(n1+n2))^2."""
        if self.R is None:
            return (False, "SKIP: R missing")
        if self.n1 is None or self.n2 is None:
            return (False, "SKIP: n1 or n2 missing")
        expected_R = ((self.n1 - self.n2) / (self.n1 + self.n2)) ** 2
        if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"R = {self.R:.6f}, expected ≈ {expected_R:.6f}")
        return (True, f"R = {self.R:.6f} ≈ {expected_R:.6f}")

    def test_modeling_fresnel_T(self):
        """Transmission coefficient matches T = 4 n1 n2 / (n1+n2)^2."""
        if self.T is None:
            return (False, "SKIP: T missing")
        if self.n1 is None or self.n2 is None:
            return (False, "SKIP: n1 or n2 missing")
        expected_T = (4 * self.n1 * self.n2) / (self.n1 + self.n2) ** 2
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T = {self.T:.6f}, expected ≈ {expected_T:.6f}")
        return (True, f"T = {self.T:.6f} ≈ {expected_T:.6f}")

    def test_trace_pressure_formula(self):
        """Pressure formula p = (S_i/c)*(1 + R - n2*T)."""
        if self.p is None or self.R is None or self.T is None:
            return (False, "SKIP: p, R, or T missing")
        if self.S_i is None or self.c is None or self.n2 is None:
            return (False, "SKIP: S_i, c, or n2 missing")
        expected_p = (self.S_i / self.c) * (1 + self.R - self.n2 * self.T)
        if not math.isclose(self.p, expected_p, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"p = {self.p:.6e}, expected ≈ {expected_p:.6e}")
        return (True, f"p = {self.p:.6e} ≈ {expected_p:.6e}")

    # --------------------------------------------------------------
    # Monotonicity test (explicit input perturbation)
    # --------------------------------------------------------------
    def test_monotonicity_R_increases_with_n2(self):
        """
        When n2 increases (e.g., +10%), the reflection coefficient R should increase.
        We perturb n2 and compute the expected R from the same formula,
        then compare to the solution's R (which corresponds to the unperturbed n2).
        """
        if self.R is None:
            return (False, "SKIP: R missing")
        if self.n1 is None or self.n2 is None:
            return (False, "SKIP: n1 or n2 missing")
        # Baseline
        n2_base = self.n2
        # Perturb by +10%
        n2_perturbed = n2_base * 1.1
        R_perturbed = ((self.n1 - n2_perturbed) / (self.n1 + n2_perturbed)) ** 2
        # Check that R increases: for n1=1, n2>1, R increases with n2
        if not (R_perturbed > self.R - 1e-15):
            return (False, f"Monotonicity violated: n2 increased from {n2_base} to {n2_perturbed}, "
                           f"expected R > {self.R:.6f}, got R_perturbed = {R_perturbed:.6f}")
        return (True, f"R increases when n2 increases by 10% (R ≈ {self.R:.6f} → {R_perturbed:.6f})")

    # --------------------------------------------------------------
    # Weak answer sanity (broad magnitude, no re‑derivation)
    # --------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """Final pressure magnitude is plausible (|p| < 1e-4 N/m²)."""
        if self.p is None:
            return (False, "SKIP: p missing")
        if abs(self.p) > 1e-4:
            return (False, f"p = {self.p:.6e} is unreasonably large (> 1e-4)")
        return (True, f"p = {self.p:.6e} is within plausible range")

    # --------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)