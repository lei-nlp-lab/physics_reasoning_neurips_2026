import math
import sympy as sp
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    PhysicsChecker for the D0 -> K- pi+ decay problem.
    Checks conservation laws, modeling consistency, monotonicity, and provides
    a weak sanity check on the final answers.
    """
    def __init__(self, solution_instance: Any):
        # Extract solution_instance.results safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Required givens (from problem statement or KB)
        self.mD = 1.8       # GeV/c^2
        self.mK = 0.5       # GeV/c^2
        self.mPi = 0.15     # GeV/c^2

        # Results from solution (must be extracted via getattr)
        self.p_K = results.get('p_K', None)
        self.EK = results.get('EK', None)
        self.EPi = results.get('EPi', None)
        self.Q2_answer = results.get('Q2_answer', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys_exist(self) -> Tuple[bool, str]:
        """Check that all required result keys are present and not None."""
        missing = []
        for key, val in [('p_K', self.p_K), ('EK', self.EK),
                         ('EPi', self.EPi), ('Q2_answer', self.Q2_answer)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_values_are_finite_and_positive(self) -> Tuple[bool, str]:
        """Check that numeric results are finite and non-negative (where applicable)."""
        for name, val in [('p_K', self.p_K), ('EK', self.EK), ('EPi', self.EPi)]:
            if val is None:
                return (False, f"SKIP: {name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not a number")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
            if val < 0:
                return (False, f"{name} is negative")
        # Q2 answer should be a string
        if self.Q2_answer is None:
            return (False, "SKIP: Q2_answer is None")
        if not isinstance(self.Q2_answer, str):
            return (False, "Q2_answer is not a string")
        return (True, "All values are finite, positive (where applicable), and of correct type")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_energy_conservation(self) -> Tuple[bool, str]:
        """Check that EK + EPi = mD (within tolerance)."""
        if any(v is None for v in (self.EK, self.EPi)):
            return (False, "SKIP: EK or EPi not available")
        total = self.EK + self.EPi
        if math.isclose(total, self.mD, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"EK+EPi = {total:.4f} ≈ mD = {self.mD}")
        return (False, f"EK+EPi = {total:.4f} ≠ mD = {self.mD}")

    def test_invariant_momentum_magnitude_equal(self) -> Tuple[bool, str]:
        """Check that |p_K| = |p_pi| (from Epi)."""
        if any(v is None for v in (self.p_K, self.EPi)):
            return (False, "SKIP: p_K or EPi not available")
        # compute p_pi from EPi and mPi
        p_pi = math.sqrt(max(0.0, self.EPi**2 - self.mPi**2))
        if math.isclose(self.p_K, p_pi, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_K = {self.p_K:.4f} ≈ p_pi = {p_pi:.4f}")
        return (False, f"p_K = {self.p_K:.4f} ≠ p_pi = {p_pi:.4f}")

    def test_invariant_ek_ge_mk(self) -> Tuple[bool, str]:
        """Check that EK >= mK (relativistic)."""
        if self.EK is None:
            return (False, "SKIP: EK not available")
        if self.EK >= self.mK - 1e-9:
            return (True, f"EK = {self.EK:.4f} >= mK = {self.mK}")
        return (False, f"EK = {self.EK:.4f} < mK = {self.mK}")

    def test_invariant_epi_ge_mpi(self) -> Tuple[bool, str]:
        """Check that EPi >= mPi (relativistic)."""
        if self.EPi is None:
            return (False, "SKIP: EPi not available")
        if self.EPi >= self.mPi - 1e-9:
            return (True, f"EPi = {self.EPi:.4f} >= mPi = {self.mPi}")
        return (False, f"EPi = {self.EPi:.4f} < mPi = {self.mPi}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (based on derivation steps)
    # ----------------------------------------------------------------------
    def test_modeling_pk_derived_formula(self) -> Tuple[bool, str]:
        """
        Re-derive p_K from the standard two-body decay formula using
        given masses and compare to solution's p_K.
        """
        if self.p_K is None:
            return (False, "SKIP: p_K not available")
        # Use formula: p = ( (mD^2 + mPi^2 - mK^2)^2 / (4*mD^2) - mPi^2 )^(1/2)
        numerator = self.mD**2 + self.mPi**2 - self.mK**2
        term = (numerator**2) / (4 * self.mD**2) - self.mPi**2
        if term < 0:
            return (False, f"Term under sqrt is negative: {term:.4f}")
        expected_p = math.sqrt(term)
        # compare with solution
        if math.isclose(self.p_K, expected_p, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_K = {self.p_K:.4f} matches formula value = {expected_p:.4f}")
        # slightly more tolerance for rounding noise
        if math.isclose(self.p_K, expected_p, rel_tol=2e-2, abs_tol=2e-2):
            return (True, f"p_K = {self.p_K:.4f} matches formula within 2% (expected {expected_p:.4f})")
        return (False, f"p_K = {self.p_K:.4f} deviates from derived value {expected_p:.4f}")

    def test_modeling_ek_from_pk(self) -> Tuple[bool, str]:
        """Check that EK = sqrt(p_K^2 + mK^2)."""
        if any(v is None for v in (self.p_K, self.EK)):
            return (False, "SKIP: p_K or EK not available")
        expected = math.sqrt(self.p_K**2 + self.mK**2)
        if math.isclose(self.EK, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"EK = {self.EK:.4f} ≈ sqrt(p_K^2 + mK^2) = {expected:.4f}")
        return (False, f"EK = {self.EK:.4f} ≠ sqrt(p_K^2 + mK^2) = {expected:.4f}")

    def test_modeling_epi_from_pk(self) -> Tuple[bool, str]:
        """Check that EPi = sqrt(p_K^2 + mPi^2)."""
        if any(v is None for v in (self.p_K, self.EPi)):
            return (False, "SKIP: p_K or EPi not available")
        expected = math.sqrt(self.p_K**2 + self.mPi**2)
        if math.isclose(self.EPi, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"EPi = {self.EPi:.4f} ≈ sqrt(p_K^2 + mPi^2) = {expected:.4f}")
        return (False, f"EPi = {self.EPi:.4f} ≠ sqrt(p_K^2 + mPi^2) = {expected:.4f}")

    def test_modeling_q2_answer_plausible_format(self) -> Tuple[bool, str]:
        """Check that Q2_answer is a plausible boolean-like string (no re-derivation)."""
        if self.Q2_answer is None:
            return (False, "SKIP: Q2_answer not available")
        lower = self.Q2_answer.lower().strip()
        if lower in ('true', 'false', 't', 'f', 'yes', 'no'):
            return (True, f"Q2_answer = {self.Q2_answer} is plausibly boolean")
        return (False, f"Q2_answer = {self.Q2_answer} is not a standard boolean string")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb one input and compare direction)
    # ----------------------------------------------------------------------
    def test_monotonicity_pk_decreases_with_mK(self) -> Tuple[bool, str]:
        """
        Increasing mK should decrease the daughter momentum p_K.
        Compute p_K for mK and mK+epsilon using the derived formula.
        """
        if self.p_K is None:
            return (False, "SKIP: p_K not available")
        epsilon = 0.05  # +5% of mK (approx 0.025, but use 0.05 for clarity)
        mK_pert = self.mK + epsilon
        # compute p for perturbed mK
        numerator = self.mD**2 + self.mPi**2 - mK_pert**2
        term = (numerator**2) / (4 * self.mD**2) - self.mPi**2
        if term < 0:
            return (False, f"Perturbed term negative: {term:.4f}")
        p_pert = math.sqrt(term)
        # compare direction (p should be smaller for larger mK)
        if p_pert < self.p_K - 1e-9:
            return (True, f"p_K decreases when mK increased: baseline {self.p_K:.4f} -> perturbed {p_pert:.4f}")
        # If they are very close, it's not a clear monotonic behavior
        if math.isclose(p_pert, self.p_K, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"p_K unchanged (within tolerance) when mK increased: {self.p_K:.4f} vs {p_pert:.4f}")
        return (False, f"p_K did not decrease: baseline {self.p_K:.4f}, perturbed {p_pert:.4f}")

    def test_monotonicity_pk_increases_with_mD(self) -> Tuple[bool, str]:
        """Increasing mD should increase p_K (all else fixed)."""
        if self.p_K is None:
            return (False, "SKIP: p_K not available")
        epsilon = 0.1  # +0.1 GeV
        mD_pert = self.mD + epsilon
        # compute p for perturbed mD
        numerator = mD_pert**2 + self.mPi**2 - self.mK**2
        term = (numerator**2) / (4 * mD_pert**2) - self.mPi**2
        if term < 0:
            return (False, f"Perturbed term negative: {term:.4f}")
        p_pert = math.sqrt(term)
        # compare direction
        if p_pert > self.p_K + 1e-9:
            return (True, f"p_K increases when mD increased: baseline {self.p_K:.4f} -> perturbed {p_pert:.4f}")
        if math.isclose(p_pert, self.p_K, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"p_K unchanged (within tolerance) when mD increased: {self.p_K:.4f} vs {p_pert:.4f}")
        return (False, f"p_K did not increase: baseline {self.p_K:.4f}, perturbed {p_pert:.4f}")

    # ----------------------------------------------------------------------
    # Answer sanity (exactly one, weak, no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak_range_and_type(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that p_K is positive and less than mD,
        and that Q2_answer is a non-empty string. No comparison to target values.
        """
        issues = []
        if self.p_K is not None:
            if not (0 < self.p_K < self.mD):
                issues.append(f"p_K = {self.p_K} not in (0, {self.mD})")
        if self.Q2_answer is not None:
            if not isinstance(self.Q2_answer, str) or len(self.Q2_answer.strip()) == 0:
                issues.append("Q2_answer is not a non-empty string")
        if issues:
            return (False, " | ".join(issues))
        # If both are None, still okay? Better to flag missing data but not crash:
        if self.p_K is None and self.Q2_answer is None:
            return (False, "Both p_K and Q2_answer are None; cannot perform sanity")
        return (True, "p_K in plausible range, Q2_answer is non-empty string")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them,
        collect results, and return (overall_pass, logs).
        """
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)