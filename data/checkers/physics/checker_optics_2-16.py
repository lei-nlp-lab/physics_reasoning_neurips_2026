import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics checker for thin-film quarter-wave coating problem.
    Audits a solution_instance for correctness of thickness and reflectivity.
    """

    def __init__(self, solution_instance):
        # Results from the solution
        self.results = getattr(solution_instance, 'results', {})
        self.d = self.results.get('d', None)
        self.R = self.results.get('R', None)

        # Givens (if provided by the solution)
        self.givens = getattr(solution_instance, 'givens', {})
        self.n0 = self.givens.get('n0', None)          # air index
        self.n_f = self.givens.get('n_f', None)        # film index (TiO2)
        self.n_s = self.givens.get('n_s', None)        # substrate index (glass)
        self.lambda0 = self.givens.get('lambda0', None)  # chosen vacuum wavelength

        # Auxiliary data (optional, not used directly in checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface checks – ensure required result keys exist and have correct type
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.d is None:
            return (False, "Missing required result key 'd'")
        if self.R is None:
            return (False, "Missing required result key 'R'")
        return (True, "Both required keys 'd' and 'R' are present")

    def test_interface_numeric_d(self):
        if self.d is None:
            return (False, "SKIP: d not available")
        if not isinstance(self.d, (int, float)):
            return (False, f"d should be numeric, got {type(self.d).__name__}")
        return (True, "d is a number")

    def test_interface_numeric_R(self):
        if self.R is None:
            return (False, "SKIP: R not available")
        if not isinstance(self.R, (int, float)):
            return (False, f"R should be numeric, got {type(self.R).__name__}")
        return (True, "R is a number")

    # ------------------------------------------------------------------
    # Physical invariant checks – use math.isclose with lenient tolerance
    # ------------------------------------------------------------------
    def test_invariant_quarter_wave_thickness(self):
        """
        Verify that the optical thickness n_f * d equals lambda0 / 4.
        Handles units: d may be in meters or microns; lambda0 assumed SI (m).
        """
        if self.d is None or self.lambda0 is None or self.n_f is None:
            return (False, "SKIP: missing d, lambda0, or n_f")

        # Expected d in meters (SI)
        expected_d_m = self.lambda0 / (4.0 * self.n_f)

        # Try to infer units of d
        if abs(self.d) < 1e-6:          # very small -> likely meters
            expected = expected_d_m
        else:                           # larger -> likely microns
            expected = expected_d_m * 1e6

        if not math.isclose(self.d, expected, rel_tol=1e-2, abs_tol=1e-2*expected):
            return (False, f"d = {self.d} does not match quarter-wave condition "
                           f"(expected {expected})")
        return (True, f"d matches quarter-wave condition (expected {expected})")

    def test_invariant_reflectivity_formula(self):
        """
        Verify that R = ((n_f^2 - n0*n_s) / (n_f^2 + n0*n_s))^2.
        """
        if self.R is None or self.n0 is None or self.n_f is None or self.n_s is None:
            return (False, "SKIP: missing R, n0, n_f, or n_s")

        numerator = self.n_f**2 - self.n0 * self.n_s
        denominator = self.n_f**2 + self.n0 * self.n_s
        expected_R = (numerator / denominator) ** 2

        if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-4):
            return (False, f"R = {self.R} does not match formula (expected {expected_R})")
        return (True, f"R matches reflectivity formula (expected {expected_R})")

    # ------------------------------------------------------------------
    # Modeling quality checks – positivity, range, plausibility
    # ------------------------------------------------------------------
    def test_modeling_positive_thickness(self):
        if self.d is None:
            return (False, "SKIP: d not available")
        if self.d <= 0:
            return (False, f"d should be positive, got {self.d}")
        return (True, f"d = {self.d} > 0")

    def test_modeling_reflectivity_range(self):
        if self.R is None:
            return (False, "SKIP: R not available")
        if not (0.0 <= self.R <= 1.0):
            return (False, f"R should be in [0,1], got {self.R}")
        return (True, f"R = {self.R} is in [0,1]")

    # ------------------------------------------------------------------
    # Monotonicity check – perturb n_f and verify trend in R
    # ------------------------------------------------------------------
    def test_monotonicity_reflectivity_with_nf(self):
        """
        For n_f > sqrt(n0*n_s), increasing n_f increases R.
        If we have the required indices and R, we perturb n_f upward by 10%
        and compare the computed R with the given R.
        """
        if self.R is None or self.n_f is None or self.n0 is None or self.n_s is None:
            return (False, "SKIP: missing n_f, n0, n_s, or R")

        threshold = math.sqrt(self.n0 * self.n_s)
        if self.n_f <= threshold:
            return (False, "SKIP: n_f not in region where R increases with n_f")

        # Baseline R from results (or we could compute from formula; we use the result)
        baseline_R = self.R

        # Perturb n_f +10%
        nf_perturbed = self.n_f * 1.1
        numerator = nf_perturbed**2 - self.n0 * self.n_s
        denominator = nf_perturbed**2 + self.n0 * self.n_s
        perturbed_R = (numerator / denominator) ** 2

        if baseline_R >= perturbed_R:
            return (False, f"Monotonicity violation: R should increase when n_f increases "
                           f"from {self.n_f} to {nf_perturbed}, but given R = {baseline_R} "
                           f"and perturbed R = {perturbed_R}")
        return (True, f"R correctly increases with n_f: given R = {baseline_R} "
                      f"< perturbed R = {perturbed_R}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one allowed) – only broad range checks
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self):
        """
        Very weak sanity: d positive and less than 10 μm; R between 0 and 1.
        No formula re-derivation or expected value hardcoding.
        """
        messages = []
        if self.d is not None:
            if self.d <= 0 or self.d > 10:
                messages.append(f"d = {self.d} outside typical range (0-10 μm)")
        if self.R is not None:
            if self.R < 0 or self.R > 1:
                messages.append(f"R = {self.R} out of [0,1]")
        if messages:
            return (False, "; ".join(messages))
        return (True, "Broad range sanity passed")

    # ------------------------------------------------------------------
    # Main test runner – discovers all test_ methods
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True

        # Sort to ensure deterministic order
        for method_name in sorted(dir(self)):
            if not method_name.startswith("test_"):
                continue
            method = getattr(self, method_name)
            if not callable(method):
                continue

            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{method_name} CRASHED: {e}"

            logs.append(f"{'PASS' if passed else 'FAIL'}: {method_name}: {msg}")
            if not passed:
                all_pass = False

        return (all_pass, logs)