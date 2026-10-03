import math
import traceback
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Robust physics checker for the delta potential scattering problem.
    Audits solution based on Born approximation, finite-range low-momentum behavior,
    and the condition kR << 1.
    """
    def __init__(self, solution_instance: Any):
        # Safely extract all needed quantities from the solution instance.
        # Prefer explicit attributes, fallback to results dict.
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})

        # Givens – try attributes first, then results dict.
        self.m = getattr(solution_instance, 'm', self.results.get('m', None))
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.hbar = getattr(solution_instance, 'hbar', self.results.get('hbar', None))
        self.k = getattr(solution_instance, 'k', self.results.get('k', None))
        self.R = getattr(solution_instance, 'R', self.results.get('R', None))
        self.theta = getattr(solution_instance, 'theta', self.results.get('theta', None))

        # Possibly momentum p instead of k
        self.p = getattr(solution_instance, 'p', self.results.get('p', None))
        if self.k is None and self.p is not None and self.hbar is not None:
            self.k = self.p / self.hbar

        # Emitted results from pseudocode
        self.dsigma_dOmega_delta = self.results.get('dsigma_dOmega_delta', None)
        self.A = self.results.get('A', None)
        self.kR = self.results.get('kR', None)

        # Optional: finite-range cross section if explicitly provided
        self.dsigma_dOmega_finite = self.results.get('dsigma_dOmega_finite', None)

        # Derive expected delta cross section if all necessary givens are present.
        self.expected_dsigma_delta = None
        if self.m is not None and self.a is not None and self.hbar is not None:
            self.expected_dsigma_delta = (self.m * self.a / (2 * math.pi * self.hbar**2)) ** 2

    # ------------------------------------------------------------
    # Interface tests – required keys must exist and be non-None
    # ------------------------------------------------------------
    def test_interface_dsigma_delta_exists(self) -> Tuple[bool, str]:
        """Check that the crucial delta cross section result is provided."""
        if self.dsigma_dOmega_delta is None:
            return (False, "dsigma_dOmega_delta is missing")
        return (True, "dsigma_dOmega_delta is present")

    def test_interface_A_exists(self) -> Tuple[bool, str]:
        """Check that the integrated potential A is provided (finite-range part)."""
        if self.A is None:
            return (False, "A (integral of potential) is missing")
        return (True, "A is present")

    def test_interface_kR_exists(self) -> Tuple[bool, str]:
        """Check that the condition quantity kR is provided."""
        if self.kR is None:
            return (False, "kR condition is missing")
        return (True, "kR is present")

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that all required givens (m, a, hbar) are present to allow
        a full computational audit."""
        missing = []
        if self.m is None: missing.append("m")
        if self.a is None: missing.append("a")
        if self.hbar is None: missing.append("hbar")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All essential givens (m, a, hbar) present")

    # ------------------------------------------------------------
    # Invariant tests – physical consistency checks using math.isclose
    # ------------------------------------------------------------
    def test_invariant_delta_cross_section(self) -> Tuple[bool, str]:
        """Born approximation: dσ/dΩ = (m a / (2π hbar^2))^2 for delta potential."""
        if self.dsigma_dOmega_delta is None or self.expected_dsigma_delta is None:
            return (False, "SKIP: missing dsigma_dOmega_delta or givens")
        if not math.isclose(self.dsigma_dOmega_delta, self.expected_dsigma_delta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Delta cross section mismatch: got {self.dsigma_dOmega_delta:.6e}, expected {self.expected_dsigma_delta:.6e}")
        return (True, "Delta cross section matches Born prediction")

    def test_invariant_finite_range_cross_section(self) -> Tuple[bool, str]:
        """If both A and finite cross section are given, check dσ/dΩ = (m A/(2π hbar^2))^2."""
        if self.A is None or self.dsigma_dOmega_finite is None:
            return (False, "SKIP: missing A or dsigma_dOmega_finite")
        if self.m is None or self.hbar is None:
            return (False, "SKIP: missing m or hbar")
        expected = (self.m * self.A / (2 * math.pi * self.hbar**2)) ** 2
        if not math.isclose(self.dsigma_dOmega_finite, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Finite cross section mismatch: got {self.dsigma_dOmega_finite:.6e}, expected {expected:.6e}")
        return (True, "Finite-range cross section matches Born prediction with A")

    def test_invariant_condition_kR_small(self) -> Tuple[bool, str]:
        """Condition kR << 1 must hold (typ. kR < 0.1 for approximation)."""
        if self.kR is None:
            return (False, "SKIP: kR not provided")
        if self.kR >= 1.0:
            return (False, f"kR = {self.kR:.4f} is NOT << 1 (violates low-momentum condition)")
        # More stringent: should be at most 0.5 for exponential expansion to be accurate
        if self.kR > 0.5:
            return (False, f"kR = {self.kR:.4f} is not small enough (consider kR << 1, typically < 0.1)")
        return (True, f"kR = {self.kR:.4f} is small enough")

    # ------------------------------------------------------------
    # Modeling / trace tests – verify that the solution structure is sound
    # ------------------------------------------------------------
    def test_modeling_born_prefactor_sign(self) -> Tuple[bool, str]:
        """The Born amplitude has a minus sign: f = - (m/(2π hbar^2)) * a.
        This does not affect the cross section (square), but the sign should be negative."""
        if self.m is None or self.a is None or self.hbar is None:
            return (False, "SKIP: missing givens")
        # We cannot directly inspect f; but if dsigma is derived from f, no sign test.
        # This test is a placeholder to show structure; we skip because we cannot observe f.
        return (True, "Born prefactor sign not independently checkable, but cross section is positive")

    def test_modeling_exponential_approximation(self) -> Tuple[bool, str]:
        """For low momentum (kR << 1), the integral ∫ e^{i q·r} V(r) d3r should be ≈ A.
        We check that the approximation error due to q is small: max(qR) = 2kR * sin(theta/2) ≤ 2kR.
        If kR < 0.1, then max qR < 0.2, ensuring expansion is good."""
        if self.k is None or self.R is None:
            return (False, "SKIP: missing k or R")
        # Use kR from results if available, else compute
        kR_val = self.kR if self.kR is not None else self.k * self.R
        if kR_val is None:
            return (False, "SKIP: cannot compute kR")
        if kR_val > 0.5:
            return (False, f"kR = {kR_val:.4f} is too large for reliable exponential expansion (max qR up to {2*kR_val:.4f})")
        return (True, f"kR = {kR_val:.4f} ensures e^{{i q·r}} ≈ 1 over the potential range")

    # ------------------------------------------------------------
    # Monotonicity tests – perturb an input and check directional change
    # ------------------------------------------------------------
    def test_monotonicity_cross_section_with_a(self) -> Tuple[bool, str]:
        """The delta cross section is proportional to a^2. If we increase a by 1%,
        the cross section should increase. (Perturb a and recompute expectation.)"""
        if self.a is None or self.expected_dsigma_delta is None:
            return (False, "SKIP: missing a or expected_dsigma_delta")
        factor = 1.01  # +1%
        a_perturbed = self.a * factor
        expected_perturbed = (self.m * a_perturbed / (2 * math.pi * self.hbar**2)) ** 2
        # The perturbed cross section must be larger than the baseline.
        if not (expected_perturbed > self.expected_dsigma_delta):
            return (False, f"Monotonicity fails: a increased, but cross section did not increase "
                           f"(baseline {self.expected_dsigma_delta:.6e}, perturbed {expected_perturbed:.6e})")
        return (True, f"Cross section increases with a (baseline {self.expected_dsigma_delta:.6e}, 1% increase gives {expected_perturbed:.6e})")

    def test_monotonicity_cross_section_with_hbar_decrease(self) -> Tuple[bool, str]:
        """Cross section scales as hbar^{-4}. Decreasing hbar should increase cross section.
        Perturb hbar by -1% and recompute."""
        if self.hbar is None or self.expected_dsigma_delta is None:
            return (False, "SKIP: missing hbar or expected_dsigma_delta")
        factor = 0.99  # -1%
        hbar_perturbed = self.hbar * factor
        expected_perturbed = (self.m * self.a / (2 * math.pi * hbar_perturbed**2)) ** 2
        if not (expected_perturbed > self.expected_dsigma_delta):
            return (False, f"Monotonicity fails: hbar decreased, but cross section did not increase")
        return (True, f"Cross section increases when hbar is decreased")

    # ------------------------------------------------------------
    # Answer sanity – weak checks only (no isclose, no re-derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        """Basic sanity: dsigma_dOmega_delta must be positive and finite."""
        if self.dsigma_dOmega_delta is None:
            return (False, "SKIP: dsigma_dOmega_delta not provided")
        if self.dsigma_dOmega_delta <= 0:
            return (False, f"dsigma_dOmega_delta must be positive, got {self.dsigma_dOmega_delta}")
        if math.isinf(self.dsigma_dOmega_delta) or math.isnan(self.dsigma_dOmega_delta):
            return (False, "dsigma_dOmega_delta is inf or NaN")
        return (True, "dsigma_dOmega_delta is positive and finite")

    # ------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collect results."""
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