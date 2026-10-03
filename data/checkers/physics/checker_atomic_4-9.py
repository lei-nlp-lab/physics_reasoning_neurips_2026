import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for a relativistic kaon decay problem.
    """
    def __init__(self, solution_instance):
        self.sol_inst = solution_instance

        # Extract givens from object attributes then from 'givens' dict
        self.E = getattr(self.sol_inst, 'E', None)
        if self.E is None and hasattr(self.sol_inst, 'givens'):
            self.E = self.sol_inst.givens.get('E')

        self.tau = getattr(self.sol_inst, 'tau', None)
        if self.tau is None and hasattr(self.sol_inst, 'givens'):
            self.tau = self.sol_inst.givens.get('tau')

        self.mKc2 = getattr(self.sol_inst, 'mKc2', None)
        if self.mKc2 is None and hasattr(self.sol_inst, 'givens'):
            self.mKc2 = self.sol_inst.givens.get('mKc2')

        # Speed of light: None if not provided (no numeric default)
        self.c = getattr(self.sol_inst, 'c', None)
        if self.c is None and hasattr(self.sol_inst, 'givens'):
            self.c = self.sol_inst.givens.get('c')

        # Extract result L (distance)
        self.L = None
        if hasattr(self.sol_inst, 'results'):
            self.L = self.sol_inst.results.get('L', None)
        if self.L is None:
            self.L = getattr(self.sol_inst, 'L', None)

    # ------------------------------------------------------------------
    # run_all_tests (discovery + execution)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Interface tests: required keys and types
    # ------------------------------------------------------------------
    def test_interface_result_L_exists(self):
        if self.L is None:
            return (False, "L not found in results")
        return (True, "L present in results")

    # ------------------------------------------------------------------
    # Invariant tests: theorem‑based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_positivity(self):
        if self.L is None:
            return (False, "SKIP: L not available")
        if self.L > 0:
            return (True, f"L = {self.L} > 0")
        return (False, f"L = {self.L} is not positive")

    def test_invariant_gamma_gt_one(self):
        """Relativistic gamma must be greater than 1."""
        if self.E is None or self.mKc2 is None:
            return (True, "SKIP: givens E or mKc2 missing")
        if self.mKc2 == 0:
            return (False, "mKc2 is zero")
        gamma = self.E / self.mKc2
        if gamma > 1.0:
            return (True, f"gamma = {gamma} > 1")
        return (False, f"gamma = {gamma} is not > 1")

    # ------------------------------------------------------------------
    # Modeling / trace tests: consistency with physics relations
    # ------------------------------------------------------------------
    def test_modeling_distance_relation(self):
        """
        Check that L is consistent with L = c * gamma * tau * ln(10)
        using the relativistic time‑dilation formula.
        """
        if self.L is None:
            return (False, "L not available")
        if self.E is None or self.tau is None or self.mKc2 is None or self.c is None:
            return (True, "SKIP: missing givens for E, tau, mKc2, or c")
        if self.mKc2 == 0:
            return (False, "mKc2 is zero")
        gamma = self.E / self.mKc2
        expected_L = self.c * gamma * self.tau * math.log(10)
        if math.isclose(expected_L, self.L, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L = {self.L} matches expected {expected_L:.6g}")
        return (False, f"L = {self.L} does not match expected {expected_L:.6g}")

    def test_modeling_beta_from_gamma(self):
        """
        Check that the derived speed parameter beta = sqrt(1 - 1/gamma^2)
        is physically plausible (0 < beta < 1). This verifies that the
        relativistic relation between gamma and velocity is respected.
        """
        if self.E is None or self.mKc2 is None:
            return (True, "SKIP: missing E or mKc2 to compute gamma")
        if self.mKc2 == 0:
            return (False, "mKc2 is zero")
        gamma = self.E / self.mKc2
        if gamma <= 1.0:
            return (False, f"gamma = {gamma} is not > 1")
        beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
        if 0.0 < beta < 1.0:
            return (True, f"beta = {beta:.4f} from gamma = {gamma} is physical")
        return (False, f"beta = {beta:.4f} is out of physical range")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb one input and check trend
    # ------------------------------------------------------------------
    def test_monotonicity_L_increases_with_E(self):
        """Increase beam energy by 10% → larger L."""
        if self.L is None or self.E is None or self.tau is None or self.mKc2 is None or self.c is None:
            return (True, "SKIP: missing required values")
        baseline = self.L
        perturbed_E = self.E * 1.10
        gamma = perturbed_E / self.mKc2
        expected = self.c * gamma * self.tau * math.log(10)
        if expected > baseline:
            return (True, f"E increased → L increases (baseline {baseline:.6g}, perturbed {expected:.6g})")
        return (False, f"L did not increase with E: baseline {baseline:.6g}, perturbed {expected:.6g}")

    def test_monotonicity_L_increases_with_tau(self):
        """Increase proper lifetime by 10% → larger L."""
        if self.L is None or self.E is None or self.tau is None or self.mKc2 is None or self.c is None:
            return (True, "SKIP: missing required values")
        baseline = self.L
        perturbed_tau = self.tau * 1.10
        gamma = self.E / self.mKc2
        expected = self.c * gamma * perturbed_tau * math.log(10)
        if expected > baseline:
            return (True, f"tau increased → L increases (baseline {baseline:.6g}, perturbed {expected:.6g})")
        return (False, f"L did not increase with tau: baseline {baseline:.6g}, perturbed {expected:.6g}")

    # ------------------------------------------------------------------
    # Weak answer sanity (must NOT use math.isclose or re‑derive)
    # ------------------------------------------------------------------
    def test_answer_sanity_range(self):
        """L should be positive, finite, and within a broad plausible range."""
        if self.L is None:
            return (False, "L not available")
        if not math.isfinite(self.L):
            return (False, f"L = {self.L} is not finite")
        if self.L <= 0:
            return (False, f"L = {self.L} is not positive")
        # Broad range: between 1 mm and 100 km
        if 1e-3 < self.L < 1e5:
            return (True, f"L = {self.L} is within a plausible order of magnitude")
        return (False, f"L = {self.L} is far from any expected value")