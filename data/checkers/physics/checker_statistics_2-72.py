import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the problem:
    Adiabatic expansion of a cavity filled with Planck-distributed radiation.
    Checks the final peak frequency using the adiabatic condition and Wien's law.
    """
    def __init__(self, solution_instance):
        # Extract givens and results with safe defaults (None for missing)
        self.V_i = getattr(solution_instance, 'V_i', None)
        self.V_f = getattr(solution_instance, 'V_f', None)
        self.omega_i = getattr(solution_instance, 'omega_i', None)
        # Results from pseudocode [emit] contract
        self.results = getattr(solution_instance, 'results', {})
        self.omega_f = self.results.get('omega_f', None)
        # Optional quantities: may be used in some tests
        self.T_i = getattr(solution_instance, 'T_i', None)
        self.T_f = getattr(solution_instance, 'T_f', None)

    # ------------------------------------------------------------------
    # Interface tests: check that required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_omega_f_present(self):
        """omega_f must be present in results and be numeric."""
        if self.omega_f is None:
            return False, "Results missing 'omega_f'"
        if not isinstance(self.omega_f, (int, float)):
            return False, f"omega_f is not numeric, got {type(self.omega_f)}"
        return True, "omega_f is present and numeric"

    def test_interface_omega_i_present(self):
        """omega_i must be provided as a given."""
        if self.omega_i is None:
            return False, "Missing given: omega_i"
        return True, "omega_i is present"

    def test_interface_volume_givens(self):
        """V_i and V_f must be present and V_f > V_i for expansion."""
        if self.V_i is None or self.V_f is None:
            return False, "Missing volume givens (V_i or V_f)"
        if self.V_f <= self.V_i:
            return False, "V_f must be larger than V_i (expansion to 2V)"
        return True, f"Volumes given: V_i={self.V_i}, V_f={self.V_f}"

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_adiabatic_t3v(self):
        """
        Adiabatic condition for photon gas: T^3 V = constant.
        If both T_i and T_f are available, check T_f^3 V_f ≈ T_i^3 V_i.
        """
        if self.T_i is None or self.T_f is None or self.V_i is None or self.V_f is None:
            return False, "SKIP: T_i, T_f, V_i, V_f required for this test"
        left = self.T_f ** 3 * self.V_f
        right = self.T_i ** 3 * self.V_i
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Adiabatic condition holds: T_f^3 V_f = {left}, T_i^3 V_i = {right}"
        else:
            return False, f"Adiabatic condition failed: T_f^3 V_f = {left}, T_i^3 V_i = {right} (ratio {left/right})"

    def test_invariant_wien_displacement_combined(self):
        """
        Combined invariant from adiabatic expansion and Wien's law:
        ω_f / ω_i = (V_i / V_f)^{1/3}
        This is the core physical prediction.
        """
        if self.omega_i is None or self.omega_f is None or self.V_i is None or self.V_f is None:
            return False, "SKIP: omega_i, omega_f, V_i, V_f required"
        if self.omega_i == 0:
            return False, "omega_i is zero, cannot compute ratio"
        expected_ratio = (self.V_i / self.V_f) ** (1.0 / 3.0)
        actual_ratio = self.omega_f / self.omega_i
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ω_f/ω_i = {actual_ratio} ≈ (V_i/V_f)^(1/3) = {expected_ratio}"
        else:
            return False, f"ω_f/ω_i = {actual_ratio} but expected {expected_ratio} (deviation {abs(actual_ratio-expected_ratio)})"

    # ------------------------------------------------------------------
    # Modeling / trace tests: check quality of the derived formula
    # ------------------------------------------------------------------
    def test_modeling_omega_f_formula(self):
        """
        Direct numerical check: ω_f = ω_i * (V_i/V_f)^{1/3}.
        This verifies that the result matches the physical relationship.
        """
        if self.omega_i is None or self.omega_f is None or self.V_i is None or self.V_f is None:
            return False, "SKIP: required variables missing"
        expected = self.omega_i * (self.V_i / self.V_f) ** (1.0 / 3.0)
        if math.isclose(self.omega_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"omega_f = {self.omega_f} matches formula {expected}"
        else:
            return False, f"omega_f = {self.omega_f} but formula gives {expected} (difference {self.omega_f - expected})"

    def test_modeling_volume_ratio_specific(self):
        """
        For the specific expansion V_f = 2V_i, the ratio should be 1/2^{1/3}.
        This is a particular case of the invariant test.
        """
        if self.omega_i is None or self.omega_f is None or self.V_i is None or self.V_f is None:
            return False, "SKIP: required variables missing"
        if self.V_f != 2 * self.V_i:
            return False, f"SKIP: V_f is not exactly 2*V_i (V_f={self.V_f}, V_i={self.V_i})"
        expected_ratio = 1.0 / (2.0 ** (1.0/3.0))
        ratio = self.omega_f / self.omega_i
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"For V_f=2V_i, ω_f/ω_i = {ratio} ≈ 1/2^{1/3} = {expected_ratio}"
        else:
            return False, f"For V_f=2V_i, ω_f/ω_i = {ratio} but expected {expected_ratio}"

    # ------------------------------------------------------------------
    # Monotonicity tests: check trends via perturbation
    # ------------------------------------------------------------------
    def test_monotonicity_volume_expansion(self):
        """
        If V_f increases (keeping ω_i, V_i fixed), ω_f should decrease.
        We compute ω_f for the given V_f and for a perturbed V_f' = 1.1*V_f.
        The perturbed ω_f should be smaller.
        """
        if self.omega_i is None or self.V_i is None or self.V_f is None:
            return False, "SKIP: omega_i, V_i, V_f required"
        if self.V_i <= 0 or self.V_f <= 0:
            return False, "Volumes must be positive"
        # Baseline: use the formula to compute ω_f for given volumes
        baseline = self.omega_i * (self.V_i / self.V_f) ** (1.0/3.0)
        # Perturb V_f upward by 10%
        perturbed_V_f = 1.1 * self.V_f
        perturbed_omega_f = self.omega_i * (self.V_i / perturbed_V_f) ** (1.0/3.0)
        if perturbed_omega_f < baseline:
            return True, f"Monotonicity holds: increasing V_f from {self.V_f} to {perturbed_V_f} reduces ω_f from {baseline} to {perturbed_omega_f}"
        else:
            return False, f"Monotonicity violation: expected ω_f to decrease but got {perturbed_omega_f} >= {baseline}"

    def test_monotonicity_omega_i_increase(self):
        """
        If ω_i increases, ω_f should increase proportionally (for fixed volumes).
        Perturb ω_i upward by 10% and check that ω_f increases.
        """
        if self.omega_i is None or self.omega_f is None or self.V_i is None or self.V_f is None:
            return False, "SKIP: omega_i, omega_f, V_i, V_f required"
        if self.omega_i <= 0:
            return False, "omega_i must be positive"
        # Baseline: use the provided omega_f (should be consistent with model)
        baseline_omega_f = self.omega_f
        # Perturb omega_i
        perturbed_omega_i = 1.1 * self.omega_i
        # New omega_f using linear relationship
        expected_perturbed_omega_f = baseline_omega_f * (perturbed_omega_i / self.omega_i)
        if expected_perturbed_omega_f > baseline_omega_f:
            return True, f"Monotonicity holds: ω_i increase by 10% yields ω_f increase from {baseline_omega_f} to {expected_perturbed_omega_f}"
        else:
            return False, f"Monotonicity violation: expected ω_f to increase but got {expected_perturbed_omega_f} <= {baseline_omega_f}"

    # ------------------------------------------------------------------
    # Answer sanity tests (weak, only sign/finite/ordering; at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_and_ordered(self):
        """
        Weak sanity: omega_f must be positive, finite, and less than omega_i (since volume expands).
        No comparison to a derived value; just a basic consistency check.
        """
        if self.omega_f is None or self.omega_i is None:
            return False, "SKIP: omega_f or omega_i missing"
        if not math.isfinite(self.omega_f):
            return False, "omega_f is not finite (inf or NaN)"
        if self.omega_f <= 0:
            return False, f"omega_f = {self.omega_f} is not positive"
        if self.omega_f >= self.omega_i:
            return False, f"omega_f ({self.omega_f}) is not less than omega_i ({self.omega_i}) – expansion should reduce frequency"
        return True, f"omega_f = {self.omega_f} is positive, finite, and less than omega_i"

    # ------------------------------------------------------------------
    # Orchestrator
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs