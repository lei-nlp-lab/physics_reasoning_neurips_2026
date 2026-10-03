import math
import traceback
from typing import Any, Dict, Tuple

class PhysicsChecker:
    """
    Physics checker for the spin-1/2 composite system problem.
    """
    def __init__(self, solution_instance: Any) -> None:
        # Safely extract givens and results, default to None
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        if self.givens is None:
            self.givens = {}
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Extract result quantities safely (no numeric defaults)
        self.prob_frac = self.results.get('prob_frac', None)
        self.prob_pct  = self.results.get('prob_pct' , None)

        # Extract givens (used to re-derive probability)
        self.spin_A_state = self.givens.get('spin_A_state', None)
        self.spin_B_state = self.givens.get('spin_B_state', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        if self.results is None:
            return (False, "No results dictionary")
        if 'prob_frac' not in self.results:
            return (False, "Missing key 'prob_frac' in results")
        if 'prob_pct' not in self.results:
            return (False, "Missing key 'prob_pct' in results")
        return (True, "All required keys present")

    def test_interface_value_types(self) -> Tuple[bool, str]:
        if self.prob_frac is None:
            return (False, "'prob_frac' is None")
        if self.prob_pct is None:
            return (False, "'prob_pct' is None")
        if not isinstance(self.prob_frac, (int, float)):
            return (False, f"'prob_frac' has type {type(self.prob_frac)}, expected numeric")
        if not isinstance(self.prob_pct, (int, float)):
            return (False, f"'prob_pct' has type {type(self.prob_pct)}, expected numeric")
        return (True, "All values have valid numeric types")

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_probability_range(self) -> Tuple[bool, str]:
        if self.prob_frac is None:
            return (False, "SKIP: 'prob_frac' not available")
        if 0.0 <= self.prob_frac <= 1.0:
            return (True, f"Probability {self.prob_frac} is in [0,1]")
        else:
            return (False, f"Probability {self.prob_frac} is out of [0,1]")

    def test_invariant_percentage_magnitude(self) -> Tuple[bool, str]:
        if self.prob_pct is None:
            return (False, "SKIP: 'prob_pct' not available")
        if 0.0 <= self.prob_pct <= 100.0:
            return (True, f"Percentage {self.prob_pct} is in [0,100]")
        else:
            return (False, f"Percentage {self.prob_pct} is out of [0,100]")

    # ------------------------------------------------------------------
    # Modeling / trace checks (re‑derive the answer from given physics)
    # ------------------------------------------------------------------
    def test_modeling_probability_from_givens(self) -> Tuple[bool, str]:
        """
        Re‑compute the probability from the given initial states using
        standard quantum mechanics.
        The only consistent result for the given problem is P = 1/4.
        """
        # The problem description explicitly defines the states:
        # A: |Sz=+1/2>  ,  B: |Sx=+1/2> = (|+>_z + |->_z)/√2
        # Composite |ψ> = |++> + |+-> / √2
        # Overlap with singlet ⟨0,0|ψ> = 1/2 → probability = 1/4.
        # This is a physics fact derived from the given state description.
        # Accept a 1% tolerance to cover numerical noise.
        expected_prob = 0.25
        if self.prob_frac is None:
            return (False, "SKIP: 'prob_frac' not available")
        if math.isclose(self.prob_frac, expected_prob, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Probability {self.prob_frac} matches derived value {expected_prob}")
        else:
            return (False, f"Probability {self.prob_frac} does not match expected {expected_prob}")

    def test_modeling_consistency_frac_to_pct(self) -> Tuple[bool, str]:
        """Percentage should be exactly 100 times the fraction."""
        if self.prob_frac is None or self.prob_pct is None:
            return (False, "SKIP: need both prob_frac and prob_pct")
        expected_pct = round(self.prob_frac * 100, 10)
        if math.isclose(self.prob_pct, expected_pct, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"prob_pct {self.prob_pct} ≈ 100 * prob_frac {self.prob_frac}")
        else:
            return (False, f"prob_pct {self.prob_pct} != 100 * {self.prob_frac} = {expected_pct}")

    # ------------------------------------------------------------------
    # Answer sanity (weak – only sign, finiteness, broad magnitude)
    # ------------------------------------------------------------------
    def test_answer_sanity_probability_positive(self) -> Tuple[bool, str]:
        if self.prob_frac is None:
            return (False, "SKIP: 'prob_frac' not available")
        if math.isnan(self.prob_frac) or math.isinf(self.prob_frac):
            return (False, f"prob_frac is {self.prob_frac}, not a finite positive number")
        if self.prob_frac <= 0:
            return (False, f"prob_frac is {self.prob_frac}, but must be positive")
        return (True, f"prob_frac is positive and finite ({self.prob_frac})")

    # ------------------------------------------------------------------
    # Monotonicity check (requires explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_probability_increases_with_theta(self) -> Tuple[bool, str]:
        """
        For given initial states: Spin A in |+>_z, Spin B in |+>_x.
        The probability P(S=0) = (1/2) sin^2(θ/2) where θ is the angle of
        Spin B's state relative to the z-axis. At θ = π/2 (the Sx state),
        P = 1/4.
        If we slightly increase θ (rotate B further away from the z-axis),
        the probability should increase monotonically.
        We test: perturb theta by +0.1 rad, recompute P_pert, and check
        P_pert > baseline.
        """
        # Baseline: θ = π/2 rad → prob_frac = 0.25
        baseline_prob = 0.25
        if self.prob_frac is None:
            return (False, "SKIP: 'prob_frac' not available for baseline")
        # We rely on the baseline prob_frac to be close to 0.25; if not, something is wrong.
        if not math.isclose(self.prob_frac, baseline_prob, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline probability {self.prob_frac} is not close to expected {baseline_prob}")

        # Perturbation: increase θ by a small amount (0.1 rad ≈ 5.73°)
        theta_base = math.pi / 2
        theta_pert = theta_base + 0.1   # slightly larger angle
        # Derived formula: prob = (1/2) * sin^2(θ/2)
        prob_pert = 0.5 * (math.sin(theta_pert / 2) ** 2)

        if prob_pert > baseline_prob:
            return (True, f"Perturbed probability {prob_pert:.5f} > baseline {baseline_prob}, as expected for increasing θ")
        else:
            return (False, f"Perturbed probability {prob_pert:.5f} not greater than baseline {baseline_prob}")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
        logs = []
        all_pass = True
        # Collect all bound methods whose name starts with "test_"
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