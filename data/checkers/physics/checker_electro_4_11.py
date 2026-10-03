import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional, Callable

class PhysicsChecker:
    """
    Audits a solution for a plane electromagnetic wave normally incident on a glass plate.
    Checks Fresnel reflection, total reflected intensity, and radiation pressure.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dict safely
        self.results: Dict[str, Optional[float]] = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Required givens – try direct attributes first, then results dict, then None
        self.I: Optional[float] = getattr(solution_instance, 'I', None)
        if self.I is None:
            self.I = self.results.get('I', None)

        self.n: Optional[float] = getattr(solution_instance, 'n', None)
        if self.n is None:
            self.n = self.results.get('n', None)

        # Speed of light – do NOT default to 3e8; leave as None if not provided
        self.c: Optional[float] = getattr(solution_instance, 'c', None)
        if self.c is None:
            self.c = self.results.get('c', None)

        # Emitted results (as per pseudocode contract)
        self.R0: Optional[float] = self.results.get('R0', None)
        self.R_total: Optional[float] = self.results.get('R_total', None)
        self.P: Optional[float] = self.results.get('P', None)

    # ──────────────────────────────────────────────
    # Interface tests: required keys and types
    # ──────────────────────────────────────────────
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that R0, R_total, P are present and numeric."""
        missing = []
        for key in ['R0', 'R_total', 'P']:
            val = self.results.get(key)
            if val is None:
                missing.append(key)
            elif not isinstance(val, (int, float)):
                missing.append(f"{key} not numeric")
        if missing:
            return False, f"Missing/invalid: {', '.join(missing)}"
        return True, "All required keys present and numeric"

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that I and n are available."""
        issues = []
        if self.I is None:
            issues.append("I (incident intensity) not found")
        if self.n is None:
            issues.append("n (refractive index) not found")
        if issues:
            return False, "; ".join(issues)
        return True, "I and n available"

    # ──────────────────────────────────────────────
    # Invariant tests: theorem-based physical sanity
    # ──────────────────────────────────────────────
    def test_invariant_r0_formula(self) -> Tuple[bool, str]:
        """Check that R0 matches the Fresnel formula R0 = ((n-1)/(n+1))^2."""
        if self.n is None or self.R0 is None:
            return (False, "SKIP: n or R0 missing")
        expected = ((self.n - 1) / (self.n + 1)) ** 2
        if math.isclose(self.R0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"R0 ≈ {expected:.4f}"
        return False, f"R0 = {self.R0:.4f} ∉ {expected:.4f} ± 1e-2"

    def test_invariant_r0_range(self) -> Tuple[bool, str]:
        """R0 must be in [0, 1) for normal incidence."""
        if self.R0 is None:
            return (False, "SKIP: R0 not available")
        if 0.0 <= self.R0 < 1.0:
            return True, f"R0 = {self.R0:.4f} in [0,1)"
        return False, f"R0 = {self.R0:.4f} outside [0,1)"

    def test_invariant_r_total_range(self) -> Tuple[bool, str]:
        """R_total must be in (0, 1]."""
        if self.R_total is None:
            return (False, "SKIP: R_total not available")
        if 0.0 < self.R_total <= 1.0:
            return True, f"R_total = {self.R_total:.4f} in (0,1]"
        return False, f"R_total = {self.R_total:.4f} outside (0,1]"

    def test_invariant_pressure_formula(self) -> Tuple[bool, str]:
        """Check that P = 2*I*R_total / c (if c is available)."""
        if any(v is None for v in [self.I, self.R_total, self.P, self.c]):
            return (False, "SKIP: I, R_total, P, or c missing")
        expected_P = 2.0 * self.I * self.R_total / self.c
        if math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"P ≈ {expected_P:.4e}"
        return False, f"P = {self.P:.4e} expected {expected_P:.4e}"

    # ──────────────────────────────────────────────
    # Modeling/trace tests: consistency of derived relations
    # ──────────────────────────────────────────────
    def test_modeling_total_reflectance_formula(self) -> Tuple[bool, str]:
        """Check that R_total = 2*R0/(1+R0) from series."""
        if self.R0 is None or self.R_total is None:
            return (False, "SKIP: R0 or R_total missing")
        expected_R_total = 2.0 * self.R0 / (1.0 + self.R0)
        if math.isclose(self.R_total, expected_R_total, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"R_total ≈ {expected_R_total:.4f}"
        return False, f"R_total = {self.R_total:.4f} expected {expected_R_total:.4f}"

    def test_modeling_r_total_inequality(self) -> Tuple[bool, str]:
        """For a non-absorbing plate, R_total > R0 and R_total < 2*R0 (for 0<R0<1)."""
        if self.R0 is None or self.R_total is None:
            return (False, "SKIP: R0 or R_total missing")
        if not (0 < self.R0 < 1):
            return (False, "R0 not in (0,1), inequality may not hold")
        ok = (self.R_total > self.R0) and (self.R_total < 2.0 * self.R0)
        if ok:
            return True, f"R0={self.R0:.4f} < R_total={self.R_total:.4f} < 2*R0={2*self.R0:.4f}"
        return False, f"Inequality violated: R0={self.R0:.4f}, R_total={self.R_total:.4f}"

    # ──────────────────────────────────────────────
    # Monotonicity tests: trend using explicit perturbation
    # ──────────────────────────────────────────────
    def test_monotonicity_r_total_increases_with_n(self) -> Tuple[bool, str]:
        """
        Check that the expected R_total (computed from Fresnel formula) increases as n increases.
        Use a small perturbation (+0.1) and verify derivative sign.
        """
        if self.n is None:
            return (False, "SKIP: n missing")
        n_base = self.n
        delta = 0.1
        n_up = n_base + delta
        # Compute R0 and R_total for base and perturbed n (synthetic but derived from theorem)
        R0_base = ((n_base - 1) / (n_base + 1)) ** 2
        R_total_base = 2.0 * R0_base / (1.0 + R0_base)

        R0_up = ((n_up - 1) / (n_up + 1)) ** 2
        R_total_up = 2.0 * R0_up / (1.0 + R0_up)

        if R_total_up > R_total_base:
            return True, f"n increases → R_total increases: {R_total_base:.4f} → {R_total_up:.4f}"
        return False, f"Monotonicity violated: R_total decreased from {R_total_base:.4f} to {R_total_up:.4f}"

    # ──────────────────────────────────────────────
    # Answer sanity (weak, no isclose, no re-derivation)
    # ──────────────────────────────────────────────
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """Check that all results are finite and non‑negative (except possibly zero)."""
        issues = []
        for key, val in [("R0", self.R0), ("R_total", self.R_total), ("P", self.P)]:
            if val is None:
                continue
            if not math.isfinite(val):
                issues.append(f"{key} is not finite: {val}")
            if val < 0:
                issues.append(f"{key} is negative: {val}")
        if issues:
            return False, "; ".join(issues)
        return True, "All present results are finite and non-negative"

    # ──────────────────────────────────────────────
    # Test runner
    # ──────────────────────────────────────────────
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        # Collect all bound methods whose name starts with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)