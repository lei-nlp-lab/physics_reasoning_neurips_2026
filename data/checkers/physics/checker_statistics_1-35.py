import math
import sympy as sp
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """Auditor for thermodynamic Carnot engine problem with two identical bodies.
    
    Expects a solution instance (object) with the following attributes (or None):
        - .results['T_f'] : final common temperature
        - .results['W']   : work delivered
        - .N, .C, .T1, .T2 : given constants (may be in .givens or directly)
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens (prefer .givens dict, else direct attributes)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.N = givens.get('N', None)
        self.C = givens.get('C', None)
        self.T1 = givens.get('T1', None)
        self.T2 = givens.get('T2', None)

        # Also try direct attributes if givens missing
        if self.N is None:
            self.N = getattr(solution_instance, 'N', None)
        if self.C is None:
            self.C = getattr(solution_instance, 'C', None)
        if self.T1 is None:
            self.T1 = getattr(solution_instance, 'T1', None)
        if self.T2 is None:
            self.T2 = getattr(solution_instance, 'T2', None)

        # Results dictionary – expected keys 'T_f' and 'W'
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.T_f = results.get('T_f', None)
        self.W = results.get('W', None)

        # Derive heat capacity for convenience (only if N and C exist)
        self.C_V = None
        if self.N is not None and self.C is not None:
            self.C_V = self.N * self.C

        # Derived expected values for tests that re‑derive them
        self.expected_T_f = None
        self.expected_W = None
        if self.T1 is not None and self.T2 is not None and self.T1 > 0 and self.T2 > 0:
            self.expected_T_f = math.sqrt(self.T1 * self.T2)
            if self.N is not None and self.C is not None:
                self.expected_W = self.N * self.C * (self.T1 + self.T2 - 2 * self.expected_T_f)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Ensure results contain 'T_f' and 'W'."""
        if self.T_f is None:
            return (False, "Missing 'T_f' in results")
        if self.W is None:
            return (False, "Missing 'W' in results")
        return (True, "Both 'T_f' and 'W' are present")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that essential given parameters are present."""
        missing = []
        if self.N is None:
            missing.append('N')
        if self.C is None:
            missing.append('C')
        if self.T1 is None:
            missing.append('T1')
        if self.T2 is None:
            missing.append('T2')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens (N, C, T1, T2) are available")

    def test_interface_values_finite(self) -> Tuple[bool, str]:
        """Check that computed outputs are finite numbers."""
        if self.T_f is None or not math.isfinite(self.T_f):
            return (False, f"T_f is not a finite number: {self.T_f}")
        if self.W is None or not math.isfinite(self.W):
            return (False, f"W is not a finite number: {self.W}")
        return (True, "T_f and W are finite")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_entropy_conservation(self) -> Tuple[bool, str]:
        """Total entropy change of the two bodies should be zero (reversible)."""
        if None in (self.C_V, self.T1, self.T2, self.T_f):
            return (False, "SKIP: missing C_V, T1, T2, or T_f for entropy check")
        if self.T_f <= 0 or self.T1 <= 0 or self.T2 <= 0:
            return (False, "Temperatures must be positive for logarithms")
        try:
            delta_S = self.C_V * (math.log(self.T_f / self.T1) + math.log(self.T_f / self.T2))
        except Exception as e:
            return (False, f"Entropy calculation error: {e}")
        if math.isclose(delta_S, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Total entropy change ~ {delta_S:.4e} ≈ 0")
        else:
            return (False, f"Total entropy change = {delta_S:.4e}, not near zero")

    def test_invariant_Tf_formula(self) -> Tuple[bool, str]:
        """Check T_f ≈ sqrt(T1*T2) derived from entropy conservation."""
        if None in (self.T1, self.T2, self.T_f):
            return (False, "SKIP: missing T1, T2, or T_f")
        if self.T1 <= 0 or self.T2 <= 0:
            return (False, "Temperatures must be positive")
        expected = math.sqrt(self.T1 * self.T2)
        if math.isclose(self.T_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T_f = {self.T_f} agrees with sqrt(T1*T2) = {expected}")
        else:
            return (False, f"T_f = {self.T_f} != sqrt(T1*T2) = {expected}")

    def test_invariant_work_energy(self) -> Tuple[bool, str]:
        """Work should equal NC(T1+T2-2T_f) by energy conservation."""
        if None in (self.N, self.C, self.T1, self.T2, self.T_f, self.W):
            return (False, "SKIP: missing required values for work check")
        expected = self.N * self.C * (self.T1 + self.T2 - 2 * self.T_f)
        if math.isclose(self.W, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"W = {self.W} agrees with NC(T1+T2-2T_f) = {expected}")
        else:
            return (False, f"W = {self.W} != expected {expected}")

    # ----------------------------------------------------------------------
    # Modeling and trace tests (quality checks, use isclose)
    # ----------------------------------------------------------------------
    def test_modeling_work_nonnegative(self) -> Tuple[bool, str]:
        """Work output must be non‑negative (≥ 0) for any distinct temperatures."""
        if self.W is None:
            return (False, "SKIP: W not available")
        if self.W < 0 and not math.isclose(self.W, 0, abs_tol=1e-9):
            return (False, f"Work is negative: {self.W}")
        return (True, f"Work = {self.W} is ≥ 0")

    def test_trace_zero_work_equal_temperatures(self) -> Tuple[bool, str]:
        """If T1 == T2, the work should be zero (within tolerance)."""
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 not available")
        if not math.isclose(self.T1, self.T2, rel_tol=1e-3):
            return (True, "SKIP: T1 != T2, not testing equal-temperature case")
        if self.W is None:
            return (False, "SKIP: W not available")
        if math.isclose(self.W, 0.0, abs_tol=1e-2):
            return (True, f"W = {self.W} ≈ 0 when T1 = T2 = {self.T1}")
        else:
            return (False, f"W = {self.W} not near zero when T1 = T2 = {self.T1}")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_work_increases_with_T1(self) -> Tuple[bool, str]:
        """Check that a small increase in T1 (keeping T2 fixed) increases work."""
        if None in (self.N, self.C, self.T1, self.T2, self.W):
            return (False, "SKIP: insufficient data for monotonicity test")
        if self.T1 <= 0 or self.T2 <= 0:
            return (False, "Temperatures must be positive")
        # Baseline work using the formula
        baseline_Tf = math.sqrt(self.T1 * self.T2)
        baseline_W = self.N * self.C * (self.T1 + self.T2 - 2 * baseline_Tf)
        # Perturb T1 by +5% (or small delta if T1 very small)
        delta = 0.05 * self.T1 if self.T1 > 1e-6 else 0.1
        T1_new = self.T1 + delta
        Tf_new = math.sqrt(T1_new * self.T2)
        W_new = self.N * self.C * (T1_new + self.T2 - 2 * Tf_new)
        if W_new > baseline_W:
            return (True, f"Increasing T1 by {delta:.4f} increases work (baseline {baseline_W:.6f} -> {W_new:.6f})")
        else:
            return (False, f"Work did not increase when T1 increased: baseline {baseline_W:.6f}, new {W_new:.6f}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one such test; no math.isclose, no re‑derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity checks: T_f positive, between T1 and T2; W finite and non-negative.
        No re-derivation, no isclose, just basic sanity.
        """
        # T_f checks
        if self.T_f is None:
            return (False, "SKIP: T_f not available")
        if not math.isfinite(self.T_f):
            return (False, f"T_f is not finite: {self.T_f}")
        if self.T_f <= 0:
            return (False, f"T_f = {self.T_f} is not positive")
        if self.T1 is not None and self.T2 is not None:
            lo = min(self.T1, self.T2)
            hi = max(self.T1, self.T2)
            if self.T_f < lo or self.T_f > hi:
                return (False, f"T_f = {self.T_f} is not between {lo} and {hi}")
        # W checks (only if both given)
        if self.W is not None:
            if not math.isfinite(self.W):
                return (False, f"W is not finite: {self.W}")
            if self.W < 0:
                return (False, f"Work is negative: {self.W}")
        return (True, "T_f positive and bounded, W finite and non-negative")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, returning (all_pass, logs)."""
        logs: List[str] = []
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