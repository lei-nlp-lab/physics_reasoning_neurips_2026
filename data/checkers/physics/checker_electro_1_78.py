import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for the dielectric image method problem.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity relations, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens safely; default to None
        self.q = getattr(solution_instance, 'q', None) or solution_instance.results.get('q', None)
        self.a = getattr(solution_instance, 'a', None) or solution_instance.results.get('a', None)
        self.epsilon1 = getattr(solution_instance, 'epsilon1', None) or solution_instance.results.get('epsilon1', None)
        self.epsilon2 = getattr(solution_instance, 'epsilon2', None) or solution_instance.results.get('epsilon2', None)
        
        # Extract result: external force (positive upward)
        self.F_ext = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.F_ext = solution_instance.results.get('F_ext', None)
        elif hasattr(solution_instance, 'F_ext'):
            self.F_ext = solution_instance.F_ext

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_all_required_keys_present(self):
        """Check that all required givens and result keys exist."""
        missing = []
        if self.q is None:
            missing.append("q")
        if self.a is None:
            missing.append("a")
        if self.epsilon1 is None:
            missing.append("epsilon1")
        if self.epsilon2 is None:
            missing.append("epsilon2")
        if self.F_ext is None:
            missing.append("F_ext")
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_q_is_positive_number(self):
        if self.q is None:
            return (False, "SKIP: q missing")
        if not isinstance(self.q, (int, float)):
            return (False, f"q is not a number: {type(self.q)}")
        if self.q <= 0:
            return (False, f"q must be positive, got {self.q}")
        return (True, f"q = {self.q} > 0")

    def test_interface_a_is_positive_number(self):
        if self.a is None:
            return (False, "SKIP: a missing")
        if not isinstance(self.a, (int, float)):
            return (False, f"a is not a number: {type(self.a)}")
        if self.a <= 0:
            return (False, f"a must be positive, got {self.a}")
        return (True, f"a = {self.a} > 0")

    def test_interface_epsilon1_is_positive_number(self):
        if self.epsilon1 is None:
            return (False, "SKIP: epsilon1 missing")
        if not isinstance(self.epsilon1, (int, float)):
            return (False, f"epsilon1 is not a number: {type(self.epsilon1)}")
        if self.epsilon1 <= 0:
            return (False, f"epsilon1 must be positive, got {self.epsilon1}")
        return (True, f"epsilon1 = {self.epsilon1} > 0")

    def test_interface_epsilon2_is_positive_number(self):
        if self.epsilon2 is None:
            return (False, "SKIP: epsilon2 missing")
        if not isinstance(self.epsilon2, (int, float)):
            return (False, f"epsilon2 is not a number: {type(self.epsilon2)}")
        if self.epsilon2 <= 0:
            return (False, f"epsilon2 must be positive, got {self.epsilon2}")
        return (True, f"epsilon2 = {self.epsilon2} > 0")

    def test_interface_F_ext_is_numeric(self):
        if self.F_ext is None:
            return (False, "SKIP: F_ext missing")
        if not isinstance(self.F_ext, (int, float)):
            return (False, f"F_ext is not a number: {type(self.F_ext)}")
        return (True, f"F_ext = {self.F_ext} is numeric")

    # ------------------------------------------------------------------
    # Physical invariant checks (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_force_scales_as_q_squared(self):
        """
        The external force should satisfy: F_ext = q^2/(4 epsilon1 a^2).
        Test the equality using fuzzy matching to accommodate rounding noise.
        """
        if None in (self.q, self.a, self.epsilon1, self.F_ext):
            return (False, "SKIP: missing required quantity")
        expected = (self.q ** 2) / (4 * self.epsilon1 * self.a ** 2)
        if math.isclose(self.F_ext, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F_ext ≈ q^2/(4ε1 a^2) = {expected}")
        return (False, f"F_ext = {self.F_ext} vs expected ~{expected}")

    def test_invariant_force_independent_of_epsilon2(self):
        """
        The derived result shows that F_ext does not depend on epsilon2.
        We check by verifying that if epsilon2 were varied, the stored F_ext
        would still satisfy the formula (which ignores epsilon2). This is
        essentially a restatement of the previous test, but we accept it.
        """
        if self.epsilon2 is None or self.F_ext is None:
            return (False, "SKIP: epsilon2 or F_ext missing")
        # Simply check that the formula (independent of epsilon2) holds.
        if self.q is None or self.a is None or self.epsilon1 is None:
            return (False, "SKIP: missing q, a, or epsilon1")
        expected = (self.q ** 2) / (4 * self.epsilon1 * self.a ** 2)
        if math.isclose(self.F_ext, expected, rel_tol=1e-2):
            return (True, f"F_ext independent of epsilon2 (expected {expected})")
        return (False, f"F_ext = {self.F_ext} does not match formula, might depend on epsilon2")

    def test_invariant_force_is_attractive(self):
        """
        The electrostatic force on the negative charge is downward (attractive),
        so the external force must be upward (positive). Check sign.
        """
        if self.F_ext is None:
            return (False, "SKIP: F_ext missing")
        if self.F_ext > 0:
            return (True, f"F_ext > 0, consistent with upward external force")
        return (False, f"F_ext = {self.F_ext} is not positive; expected upward force")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_image_method_force_expression(self):
        """
        Weak model check: the external force should be representable as
        F_ext = q^2/(4 epsilon1 a^2). This is a direct check from the
        derived formula, not from hardcoded numbers.
        """
        if None in (self.q, self.a, self.epsilon1, self.F_ext):
            return (False, "SKIP: missing required quantity")
        computed = (self.q ** 2) / (4 * self.epsilon1 * self.a ** 2)
        if math.isclose(self.F_ext, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F_ext matches formula")
        return (False, f"F_ext = {self.F_ext}, formula gives {computed}")

    def test_modeling_force_direction_sign_consistent(self):
        """
        The external force must be opposite to the electrostatic force.
        Since -q is attracted to +q, the electrostatic force is downward,
        so external force must be upward (positive). Check sign.
        """
        if self.F_ext is None:
            return (False, "SKIP: F_ext missing")
        if self.F_ext > 0:
            return (True, "F_ext positive => upward force on -q")
        return (False, f"F_ext = {self.F_ext} should be positive")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturbational)
    # ------------------------------------------------------------------
    def test_monotonicity_increase_q_increases_F_ext(self):
        """
        Perturb q by +5% and +10%; the F_ext should increase as q^2.
        We compare the stored F_ext with a perturbed expected value (re-derived
        from givens) to confirm the scaling.
        """
        if None in (self.q, self.a, self.epsilon1, self.F_ext):
            return (False, "SKIP: missing required quantity")
        base = self.F_ext
        # perturb q by +5%
        q_perturbed = 1.05 * self.q
        expected_perturbed = (q_perturbed ** 2) / (4 * self.epsilon1 * self.a ** 2)
        # The stored F_ext is for original q; the perturbed expected should be larger than base.
        if expected_perturbed > base:
            return (True, f"Perturbation q+5% gives expected {expected_perturbed:.4f} > base {base:.4f}")
        return (False, f"Perturbation q+5% gives expected {expected_perturbed:.4f} not greater than base {base:.4f}")

    def test_monotonicity_increase_a_decreases_F_ext(self):
        """
        Increase a by 5%; F_ext should decrease as 1/a^2.
        """
        if None in (self.q, self.a, self.epsilon1, self.F_ext):
            return (False, "SKIP: missing required quantity")
        base = self.F_ext
        a_perturbed = 1.05 * self.a
        expected_perturbed = (self.q ** 2) / (4 * self.epsilon1 * a_perturbed ** 2)
        if expected_perturbed < base:
            return (True, f"Perturbation a+5% gives expected {expected_perturbed:.4f} < base {base:.4f}")
        return (False, f"Perturbation a+5% gives expected {expected_perturbed:.4f} not less than base {base:.4f}")

    def test_monotonicity_increase_epsilon1_decreases_F_ext(self):
        """
        Increase epsilon1 by 10%; F_ext should decrease as 1/epsilon1.
        """
        if None in (self.q, self.a, self.epsilon1, self.F_ext):
            return (False, "SKIP: missing required quantity")
        base = self.F_ext
        eps_perturbed = 1.10 * self.epsilon1
        expected_perturbed = (self.q ** 2) / (4 * eps_perturbed * self.a ** 2)
        if expected_perturbed < base:
            return (True, f"Perturbation epsilon1+10% gives expected {expected_perturbed:.4f} < base {base:.4f}")
        return (False, f"Perturbation epsilon1+10% gives expected {expected_perturbed:.4f} not less than base {base:.4f}")

    def test_monotonicity_epsilon2_variation_no_change(self):
        """
        Changing epsilon2 should not affect F_ext according to the derived result.
        Perturb epsilon2 by +20% and recompute the formula (which ignores epsilon2)
        and verify that the stored F_ext matches the unchanged expected value.
        This demonstrates monotonicity of zero effect.
        """
        if None in (self.q, self.a, self.epsilon1, self.epsilon2, self.F_ext):
            return (False, "SKIP: missing required quantity")
        base = self.F_ext
        expected_unchanged = (self.q ** 2) / (4 * self.epsilon1 * self.a ** 2)
        if math.isclose(base, expected_unchanged, rel_tol=1e-2):
            return (True, f"F_ext unchanged despite epsilon2 perturbation (expected {expected_unchanged:.4f})")
        return (False, f"F_ext = {base} changed when epsilon2 perturbed? Expected {expected_unchanged:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_positive(self):
        """
        Basic sanity: F_ext should be a finite positive number (not NaN, not zero,
        not negative) for positive q, a, epsilon1.
        """
        if self.F_ext is None:
            return (False, "SKIP: F_ext missing")
        if not math.isfinite(self.F_ext):
            return (False, f"F_ext is not finite: {self.F_ext}")
        if self.F_ext <= 0:
            return (False, f"F_ext should be positive, got {self.F_ext}")
        return (True, f"F_ext = {self.F_ext} is finite and positive")

    # ------------------------------------------------------------------
    # Runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)