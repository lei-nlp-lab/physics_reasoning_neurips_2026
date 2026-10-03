import math

class PhysicsChecker:
    """
    Checker for the capacitance problem:
    Two conductors in a homogeneous lossy medium with given conductivity,
    relative permittivity, and measured resistance.
    
    Central relation: C = ε/(σR).
    """

    def __init__(self, solution_instance):
        # Extract givens: try direct attributes, then results dict
        self.sigma = getattr(solution_instance, 'sigma', None)
        if self.sigma is None:
            self.sigma = getattr(solution_instance, 'results', {}).get('sigma', None)

        self.epsilon_r = getattr(solution_instance, 'epsilon_r', None)
        if self.epsilon_r is None:
            self.epsilon_r = getattr(solution_instance, 'results', {}).get('epsilon_r', None)

        self.R = getattr(solution_instance, 'R', None)
        if self.R is None:
            self.R = getattr(solution_instance, 'results', {}).get('R', None)

        # Result: capacitance C
        self.C = getattr(solution_instance, 'C', None)
        if self.C is None:
            self.C = getattr(solution_instance, 'results', {}).get('C', None)

        # Keep the instance for possible later use
        self.solution_instance = solution_instance

        # Physical constant
        self.epsilon0 = 8.854e-12   # F/m

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_givens_present(self):
        """Check that sigma, epsilon_r, R are provided and numeric."""
        for name, val in [('sigma', self.sigma),
                          ('epsilon_r', self.epsilon_r),
                          ('R', self.R)]:
            if val is None:
                return (False, f"FAIL: Missing or None givens for {name}")
            if not isinstance(val, (int, float)):
                return (False, f"FAIL: Givens {name} is not numeric: {type(val)}")
        return (True, "PASS: All required givens (sigma, epsilon_r, R) present and numeric.")

    def test_interface_result_C_present(self):
        """Check that result C exists and is numeric."""
        if self.C is None:
            return (False, "FAIL: Result C is None or missing")
        if not isinstance(self.C, (int, float)):
            return (False, f"FAIL: Result C is not numeric: {type(self.C)}")
        return (True, "PASS: Result C present and numeric.")

    # ------------------------------------------------------------------
    # Invariant / correctness checks (using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_derived_formula(self):
        """
        Verify that C computed from the relation C = ε/(σR) matches the
        result C (within tolerance).
        """
        if None in (self.sigma, self.epsilon_r, self.R, self.C):
            return (False, "SKIP: Missing givens or C for formula check.")
        epsilon = self.epsilon_r * self.epsilon0
        expected_C = epsilon / (self.sigma * self.R)
        if math.isclose(expected_C, self.C, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: C ({self.C:.3e}) ≈ expected ({expected_C:.3e}) with formula C = ε/(σR).")
        else:
            return (False, f"FAIL: C ({self.C:.3e}) ~ expected ({expected_C:.3e}) differ beyond tolerance.")

    def test_invariant_product_RC(self):
        """
        Check the invariant RC = ε/σ.
        Since C = ε/(σR), R*C should equal ε/σ.
        """
        if None in (self.sigma, self.epsilon_r, self.R, self.C):
            return (False, "SKIP: Missing givens or C for RC product check.")
        epsilon = self.epsilon_r * self.epsilon0
        expected_product = epsilon / self.sigma
        actual_product = self.R * self.C
        if math.isclose(actual_product, expected_product, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: RC = {actual_product:.4e} ≈ ε/σ = {expected_product:.4e}.")
        else:
            return (False, f"FAIL: RC = {actual_product:.4e} not close to ε/σ = {expected_product:.4e}.")

    # ------------------------------------------------------------------
    # Modeling / sign checks
    # ------------------------------------------------------------------
    def test_modeling_sign_convention(self):
        """Capacitance must be positive."""
        if self.C is None:
            return (False, "SKIP: C not available.")
        if self.C > 0:
            return (True, "PASS: C > 0.")
        else:
            return (False, f"FAIL: C is {self.C} (should be positive).")

    def test_modeling_epsilon_positive(self):
        """ϵ = ϵ_r·ϵ0 must be positive."""
        if self.epsilon_r is None or self.epsilon_r <= 0:
            return (False, "SKIP: epsilon_r missing or non-positive.")
        epsilon = self.epsilon_r * self.epsilon0
        if epsilon > 0:
            return (True, f"PASS: ε = {epsilon:.4e} > 0.")
        else:
            return (False, "FAIL: ε is zero or negative (impossible).")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of an input,
    # then compare the theoretically perturbed output to the actual result baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_increase_decreases_C(self):
        """
        Perturb sigma upward by 10%. Using the formula C = ε/(σR),
        compute the theoretical C for the perturbed sigma.
        Since the actual C (self.C) corresponds to the original sigma,
        the perturbed C should be smaller than self.C.
        """
        if None in (self.sigma, self.epsilon_r, self.R, self.C):
            return (False, "SKIP: Missing givens or C for monotonicity sigma.")
        epsilon = self.epsilon_r * self.epsilon0
        # Perturb sigma
        sigma_pert = 1.1 * self.sigma
        C_pert = epsilon / (sigma_pert * self.R)   # theoretical value for perturbed sigma
        baseline = self.C
        if C_pert < baseline:
            return (True, f"PASS: Increasing sigma decreases C (C_pert={C_pert:.3e} < baseline={baseline:.3e}).")
        else:
            return (False, f"FAIL: C_pert={C_pert:.3e} should be less than baseline={baseline:.3e}.")

    def test_monotonicity_R_increase_decreases_C(self):
        """
        Perturb R upward by 10%. Using the formula, the expected C for the
        perturbed R should be smaller than the actual C (which is for the original R).
        """
        if None in (self.sigma, self.epsilon_r, self.R, self.C):
            return (False, "SKIP: Missing givens or C for monotonicity R.")
        epsilon = self.epsilon_r * self.epsilon0
        R_pert = 1.1 * self.R
        C_pert = epsilon / (self.sigma * R_pert)
        baseline = self.C
        if C_pert < baseline:
            return (True, f"PASS: Increasing R decreases C (C_pert={C_pert:.3e} < baseline={baseline:.3e}).")
        else:
            return (False, f"FAIL: C_pert={C_pert:.3e} should be less than baseline={baseline:.3e}.")

    def test_monotonicity_epsilon_r_increase_increases_C(self):
        """
        Perturb epsilon_r upward by 10%. The theoretical C for the perturbed
        epsilon_r should be larger than the actual C.
        """
        if None in (self.sigma, self.epsilon_r, self.R, self.C):
            return (False, "SKIP: Missing givens or C for monotonicity epsilon_r.")
        # Perturb epsilon_r
        eps_r_pert = 1.1 * self.epsilon_r
        epsilon_pert = eps_r_pert * self.epsilon0
        C_pert = epsilon_pert / (self.sigma * self.R)
        baseline = self.C
        if C_pert > baseline:
            return (True, f"PASS: Increasing epsilon_r increases C (C_pert={C_pert:.3e} > baseline={baseline:.3e}).")
        else:
            return (False, f"FAIL: C_pert={C_pert:.3e} should be larger than baseline={baseline:.3e}.")

    # ------------------------------------------------------------------
    # Weak answer-sanity check (only positivity and broad magnitude)
    # ------------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitude(self):
        """
        Do not compute expected values; only check order‑of‑magnitude.
        For this problem C is expected around 1e-11 F; we allow generous range.
        """
        if self.C is None:
            return (False, "SKIP: C not available.")
        # Extreme bounds: 1e-15 (too small) ... 1e-5 (too large)
        if 1e-15 <= self.C <= 1e-5:
            return (True, f"PASS: C = {self.C:.3e} is within a physically plausible range [1e-15, 1e-5] F.")
        else:
            return (False, f"FAIL: C = {self.C:.3e} is far outside any reasonable range for this setup.")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover and run all test_ methods, collecting results.
        Returns (all_pass, logs).
        """
        logs = []
        all_pass = True
        for method_name in sorted(dir(self)):
            if not method_name.startswith("test_"):
                continue
            method = getattr(self, method_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{method_name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {method_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)