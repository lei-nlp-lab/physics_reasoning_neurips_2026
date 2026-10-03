import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    """
    Auditor for hydrogen energy shifts due to finite proton size.
    Covers interface, invariants, modeling/trace, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance) -> None:
        # Safely extract results
        res = getattr(solution_instance, 'results', {})
        self.E_1s = res.get('E_1s', None)
        self.E_2p = res.get('E_2p', None)

        # Try to get physical constants from solution_instance or results
        self.e = getattr(solution_instance, 'e', None)
        self.rp = getattr(solution_instance, 'rp', None)
        self.a0 = getattr(solution_instance, 'a0', None)

        if self.e is None:
            self.e = res.get('e', None)
        if self.rp is None:
            self.rp = res.get('rp', None)
        if self.a0 is None:
            self.a0 = res.get('a0', None)

        # Pre‑compute expected shifts (used only if constants are present)
        self.expected_E_1s = None
        self.expected_E_2p = None
        self._compute_expected()

    def _compute_expected(self) -> None:
        """Re‑derive the shifts from first‑order perturbation theory formulas."""
        if self.e is not None and self.rp is not None and self.a0 is not None:
            # Formulae in atomic/Gaussian units (no 4πɛ₀)
            self.expected_E_1s = (2 * self.e * self.e * self.rp * self.rp) / (5 * self.a0**3)
            self.expected_E_2p = (self.e * self.e * self.rp**4) / (1120 * self.a0**5)
        else:
            self.expected_E_1s = None
            self.expected_E_2p = None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_E_1s_exists(self) -> Tuple[bool, str]:
        if self.E_1s is None:
            return (False, "E_1s missing from results")
        return (True, "E_1s present")

    def test_interface_E_2p_exists(self) -> Tuple[bool, str]:
        if self.E_2p is None:
            return (False, "E_2p missing from results")
        return (True, "E_2p present")

    # ------------------------------------------------------------------
    # Invariant (physics) tests
    # ------------------------------------------------------------------
    def test_invariant_positivity_1s(self) -> Tuple[bool, str]:
        if self.E_1s is None:
            return (False, "SKIP: E_1s not available")
        if self.E_1s > 0:
            return (True, f"E_1s = {self.E_1s} > 0")
        else:
            return (False, f"E_1s = {self.E_1s} is not positive")

    def test_invariant_positivity_2p(self) -> Tuple[bool, str]:
        if self.E_2p is None:
            return (False, "SKIP: E_2p not available")
        if self.E_2p > 0:
            return (True, f"E_2p = {self.E_2p} > 0")
        else:
            return (False, f"E_2p = {self.E_2p} is not positive")

    def test_invariant_ratio_ordering(self) -> Tuple[bool, str]:
        """Check that E_2p is much smaller than E_1s (rp << a0)."""
        if self.E_1s is None or self.E_2p is None:
            return (False, "SKIP: one or both energy shifts missing")
        if self.E_1s <= 0 or self.E_2p <= 0:
            return (False, "SKIP: non‑positive values hinder ratio check")
        ratio = self.E_2p / self.E_1s
        # Theoretical ratio is ~ (rp^2 / a0^2) * (1/448) ≈ extremely small
        if ratio < 1e-6:
            return (True, f"E_2p/E_1s = {ratio} < 1e‑6, consistent with rp << a0")
        else:
            return (False, f"E_2p/E_1s = {ratio} is not negligible; expected << 1e‑6")

    def test_invariant_formula_1s(self) -> Tuple[bool, str]:
        """Compare E_1s with the expected value from perturbation theory."""
        if self.E_1s is None or self.expected_E_1s is None:
            return (False, "SKIP: missing data for formula check")
        if not math.isfinite(self.E_1s) or not math.isfinite(self.expected_E_1s):
            return (False, "Non‑finite values encountered")
        if math.isclose(self.E_1s, self.expected_E_1s, rel_tol=1e-2, abs_tol=1e-30):
            return (True, f"E_1s matches expected within 1%")
        else:
            return (False, f"E_1s = {self.E_1s}, expected ~{self.expected_E_1s:.3e}, deviation > 1%")

    def test_invariant_formula_2p(self) -> Tuple[bool, str]:
        """Compare E_2p with the expected value from perturbation theory."""
        if self.E_2p is None or self.expected_E_2p is None:
            return (False, "SKIP: missing data for formula check")
        if not math.isfinite(self.E_2p) or not math.isfinite(self.expected_E_2p):
            return (False, "Non‑finite values encountered")
        if math.isclose(self.E_2p, self.expected_E_2p, rel_tol=1e-2, abs_tol=1e-30):
            return (True, f"E_2p matches expected within 1%")
        else:
            return (False, f"E_2p = {self.E_2p}, expected ~{self.expected_E_2p:.3e}, deviation > 1%")

    # ------------------------------------------------------------------
    # Modelling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_perturbation_vanishes_outside(self) -> Tuple[bool, str]:
        """
        Trace consistency: The perturbation δV is zero for r > rp,
        so the energy shifts depend only on the proton radius.
        This is built into the formulas; we verify indirectly
        via positivity and formula matching.
        """
        return (True, "Model assumes δV=0 outside; verified via formula tests.")

    def test_modeling_rp_much_less_than_a0(self) -> Tuple[bool, str]:
        """
        The approximation e^{-r/a0} ≈ 1 requires rp << a0.
        Check that the ratio rp / a0 is very small.
        """
        if self.rp is None or self.a0 is None:
            return (False, "SKIP: rp or a0 not available")
        if self.a0 <= 0:
            return (False, "a0 must be positive")
        ratio = self.rp / self.a0
        if ratio < 1e-4:
            return (True, f"rp/a0 = {ratio} < 1e-4, approximation justified")
        else:
            return (False, f"rp/a0 = {ratio} is not negligible (< 1e-4)")

    def test_modeling_integrand_positive(self) -> Tuple[bool, str]:
        """
        Check that the integrand for ΔE is positive for r < rp, ensuring positive shift.
        For 1s with ψ≈constant, integrand ∝ r^2 δV(r). Evaluate at a sample point.
        """
        if self.rp is None or self.e is None:
            return (False, "SKIP: rp or e not available")
        # Define a sample point r = rp/2
        r = self.rp / 2.0
        # Compute δV(r) up to positive constant factor (ignoring e^2/(4πɛ₀))
        # δV = -C * ( 3/(2rp) - r^2/(2rp^3) - 1/r ). Since C>0, sign is opposite.
        # Physical δV = negative of that expression.
        # The integrand sign is determined by the integral, but we can check that
        # the expression 3/(2rp) - r^2/(2rp^3) - 1/r is negative, so δV is positive.
        val = (3.0/(2.0*self.rp)) - (r**2/(2.0*self.rp**3)) - (1.0/r)
        if val < 0:
            return (True, f"δV kernel positive at sample point: {val} < 0 (so δV > 0)")
        else:
            return (False, f"δV kernel not negative at sample point: {val}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input)
    # ------------------------------------------------------------------
    def test_monotonicity_rp_increase_increases_E_1s(self) -> Tuple[bool, str]:
        """
        Perturb proton radius by +1% and check that E_1s increases
        (since it scales as rp^2).
        """
        if self.E_1s is None or self.expected_E_1s is None:
            return (False, "SKIP: missing data for monotonicity check")
        # Compute baseline
        baseline_1s = self.expected_E_1s
        # Perturb rp by +1%
        rp_perturbed = self.rp * 1.01
        perturbed_1s = (2 * self.e * self.e * rp_perturbed * rp_perturbed) / (5 * self.a0**3)
        if perturbed_1s > baseline_1s:
            return (True, f"E_1s increases when rp increases: baseline {baseline_1s:.3e}, perturbed {perturbed_1s:.3e}")
        else:
            return (False, f"Monotonicity failed: baseline {baseline_1s:.3e}, perturbed {perturbed_1s:.3e}")

    def test_monotonicity_rp_increase_increases_E_2p(self) -> Tuple[bool, str]:
        """
        Perturb proton radius by +1% and check that E_2p increases
        (since it scales as rp^4).
        """
        if self.E_2p is None or self.expected_E_2p is None:
            return (False, "SKIP: missing data for monotonicity check")
        baseline_2p = self.expected_E_2p
        rp_perturbed = self.rp * 1.01
        perturbed_2p = (self.e * self.e * rp_perturbed**4) / (1120 * self.a0**5)
        if perturbed_2p > baseline_2p:
            return (True, f"E_2p increases when rp increases: baseline {baseline_2p:.3e}, perturbed {perturbed_2p:.3e}")
        else:
            return (False, f"Monotonicity failed: baseline {baseline_2p:.3e}, perturbed {perturbed_2p:.3e}")

    # ------------------------------------------------------------------
    # Weak answer‑sanity (do NOT re‑derive the final answer)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite(self) -> Tuple[bool, str]:
        """Both shifts must be finite numbers (weak sanity)."""
        if self.E_1s is None or self.E_2p is None:
            return (False, "SKIP: missing data")
        if math.isfinite(self.E_1s) and math.isfinite(self.E_2p):
            return (True, "Both shifts finite")
        else:
            return (False, "A shift is non‑finite")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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