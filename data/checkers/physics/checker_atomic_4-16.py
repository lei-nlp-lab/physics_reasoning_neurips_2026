import math
from typing import Any, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for the problem: minimum energy to create antiprotons in proton-proton collisions.
    Checks threshold energy via invariant mass conservation.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract required variables safely. Use None for missing data.
        """
        # Attempt to get the proton rest mass from various possible sources
        self.m: Optional[float] = None
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.m = solution_instance.givens.get('m', None)
            if self.m is None:
                self.m = solution_instance.givens.get('m_proton', None)
                if self.m is None:
                    self.m = solution_instance.givens.get('mass', None)

        # Also try direct attribute (some solution formats store it)
        if self.m is None and hasattr(solution_instance, 'm'):
            self.m = solution_instance.m
        if self.m is None and hasattr(solution_instance, 'proton_mass'):
            self.m = solution_instance.proton_mass

        # Extract E_min from results
        self.E_min: Optional[float] = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.E_min = solution_instance.results.get('E_min', None)

        # Store full instance for potential future use
        self.solution_instance = solution_instance

    # ---------- Interface tests ----------
    def test_interface_has_E_min(self) -> Tuple[bool, str]:
        """Check that the result key 'E_min' is present and contains a number."""
        if self.E_min is None:
            return False, "Missing result 'E_min'"
        if not isinstance(self.E_min, (int, float)):
            return False, f"E_min is not a number (got {type(self.E_min).__name__})"
        return True, "E_min exists and is numeric"

    # ---------- Invariant / physics sanity tests ----------
    def test_invariant_threshold_relation(self) -> Tuple[bool, str]:
        """
        Verify the invariant mass conservation at threshold:
        (E_beam + m)^2 - p_beam^2 = (4m)^2.
        Using E_beam^2 - p_beam^2 = m^2, the condition reduces to 2m^2 + 2m E_beam = 16m^2
        => E_beam = 7m.
        We check numerically with a 1% tolerance.
        """
        if self.m is None:
            return False, "SKIP: proton mass m not available"
        if self.E_min is None:
            return False, "SKIP: E_min not available"

        # Compute invariant mass squared from initial state
        lhs = 2 * self.m**2 + 2 * self.m * self.E_min
        rhs = (4 * self.m)**2

        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Invariant mass matches: (E|m) ~ {self.E_min/self.m:.3f} ≈ 7"
        else:
            return False, f"Invariant mass mismatch: lhs={lhs:.3e}, rhs={rhs:.3e}, ratio={lhs/rhs:.3f}"

    def test_invariant_ratio_energy_mass(self) -> Tuple[bool, str]:
        """Check that E_min/m is approximately 7 (threshold condition)."""
        if self.m is None or self.E_min is None:
            return False, "SKIP: m or E_min missing"
        ratio = self.E_min / self.m
        if math.isclose(ratio, 7.0, rel_tol=1e-2):
            return True, f"E_min/m = {ratio:.3f} ≈ 7"
        else:
            return False, f"E_min/m = {ratio:.3f}, expected ~7"

    # ---------- Weak answer sanity test (secondary) ----------
    def test_answer_sanity_energy_positive_finite(self) -> Tuple[bool, str]:
        """
        Very weak check: E_min must be positive and finite,
        and of order a few GeV (realistic for antiproton production).
        """
        if self.E_min is None:
            return False, "SKIP: E_min not available"
        if not math.isfinite(self.E_min):
            return False, "E_min is not finite"
        if self.E_min <= 0:
            return False, "E_min is not positive"
        # Typical threshold: 6.6 GeV. Check broad range 0.1-100 GeV.
        if not (0.1 <= self.E_min <= 100.0):
            return False, f"E_min = {self.E_min} GeV, outside physically plausible range for this problem"
        return True, "E_min passes sanity: positive, finite, plausible magnitude"

    # ---------- Modeling/trace quality tests (at least two) ----------
    def test_modeling_pseudocode_relationship(self) -> Tuple[bool, str]:
        """
        Check that if m is known, the reported E_min is consistent with
        the formula E_min = 7*m (within tolerance). This is a modeling
        quality check based on the pseudocode plan.
        """
        if self.m is None:
            return False, "SKIP: m not available"
        if self.E_min is None:
            return False, "SKIP: E_min not available"
        expected = 7.0 * self.m
        if math.isclose(self.E_min, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_min = {self.E_min:.3f} matches 7*m = {expected:.3f}"
        else:
            return False, f"E_min = {self.E_min:.3f}, but 7*m = {expected:.3f}"

    def test_modeling_kinetic_energy_ratio(self) -> Tuple[bool, str]:
        """
        Verify that the kinetic energy (K = E_min - m) is close to 6*m,
        which is the derived threshold kinetic energy from the invariant mass condition.
        This is a separate derived quantity that should hold if the modeling is correct.
        """
        if self.m is None:
            return False, "SKIP: proton mass m not available"
        if self.E_min is None:
            return False, "SKIP: E_min not available"

        K_min = self.E_min - self.m
        expected_K = 6.0 * self.m
        if math.isclose(K_min, expected_K, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Kinetic energy K = {K_min:.3f} matches 6*m = {expected_K:.3f}"
        else:
            return False, f"Kinetic energy K = {K_min:.3f}, but 6*m = {expected_K:.3f}"

    # ---------- Monotonicity test ----------
    def test_monotonicity_threshold_energy_increasing_with_mass(self) -> Tuple[bool, str]:
        """
        Verify that the threshold energy increases monotonically with proton mass.
        Perturb the mass by +5% and check that the theoretical threshold energy
        (E_min = 7*m) increases by the same factor.
        If actual E_min is available, also compare it to the perturbed prediction.
        """
        if self.m is None:
            return False, "SKIP: proton mass m not available"

        # Baseline theoretical threshold
        E_baseline = 7.0 * self.m

        # Perturb mass by 5%
        m_perturbed = self.m * 1.05
        E_perturbed = 7.0 * m_perturbed  # derived from the same theorem

        # The perturbed energy must be larger than the baseline
        if not (E_perturbed > E_baseline):
            return False, f"Unexpected: perturbed energy {E_perturbed:.5f} not > baseline {E_baseline:.5f}"

        # If actual E_min is present, verify it's consistent with baseline
        if self.E_min is not None:
            # Check that actual E_min is not far from baseline (should be within 10% due to noise)
            if self.E_min < 0.9 * E_baseline or self.E_min > 1.1 * E_baseline:
                return False, f"Actual E_min {self.E_min:.3f} deviates too much from baseline {E_baseline:.3f}"
            # Also check that actual is less than perturbed (since perturbed > baseline)
            if self.E_min > E_perturbed * 1.05:  # allow small noise
                return False, f"Actual E_min {self.E_min:.3f} exceeds theoretical perturbed {E_perturbed:.3f} too much"
            return True, f"Monotonicity holds: increasing m -> higher E; actual E_min {self.E_min:.3f} consistent"

        # Without actual E_min, still pass because theoretical monotonicity is satisfied
        return True, f"Monotonicity holds: increasing m by 5% increases threshold energy from {E_baseline:.3f} to {E_perturbed:.3f}"

    # ---------- Run all tests ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute all test methods, collect results."""
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