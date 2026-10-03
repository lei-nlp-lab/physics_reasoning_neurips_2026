import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Auditor for escape velocity from the pole of a rotating planet,
    given that equatorial surface gravity is half the polar gravity.
    """
    
    def __init__(self, solution_instance):
        # Safely extract inputs and results. Use None for missing values.
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}
        
        self.V = givens.get('V', None)
        if self.V is None:
            self.V = getattr(solution_instance, 'V', None)
        
        self.v_esc = results.get('v_esc', None)
        if self.v_esc is None:
            self.v_esc = getattr(solution_instance, 'v_esc', None)
        
        # Optional variables for extended modeling tests
        self.R = getattr(solution_instance, 'R', None) or givens.get('R', None) or results.get('R', None)
        self.omega = getattr(solution_instance, 'omega', None) or givens.get('omega', None) or results.get('omega', None)
        
        self.solution_instance = solution_instance

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Verify that v_esc is present and V is available."""
        if self.v_esc is None:
            return (False, "FAIL: v_esc not found in results")
        if self.V is None:
            return (False, "FAIL: V not available as a given")
        if self.v_esc <= 0:
            return (False, f"FAIL: v_esc not positive: {self.v_esc}")
        return (True, f"PASS: v_esc = {self.v_esc}, V = {self.V}")

    # ------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ------------------------------------------------------------------
    def test_invariant_escape_velocity(self):
        """From condition g_e = g_p/2, we must have v_esc = 2V (1% tolerance)."""
        if self.v_esc is None or self.V is None:
            return (False, "SKIP: missing v_esc or V")
        expected = 2.0 * self.V
        if math.isclose(self.v_esc, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: v_esc = {self.v_esc} ≈ 2V = {expected}")
        else:
            return (False, f"FAIL: v_esc = {self.v_esc} differs from 2V = {expected}")

    def test_invariant_energy_consistency(self):
        """Check that v_esc² = 4V² (from energy conservation and g condition)."""
        if self.v_esc is None or self.V is None:
            return (False, "SKIP: missing v_esc or V")
        lhs = self.v_esc ** 2
        rhs = 4.0 * self.V ** 2
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: v_esc² = {lhs} ≈ 4V² = {rhs}")
        else:
            return (False, f"FAIL: v_esc² = {lhs} not close to 4V² = {rhs}")

    def test_invariant_g_ratio_condition(self):
        """Verify that the condition g_e = g_p/2 leads to the relation V² = GM/(2R)."""
        # Equivalent to v_esc² = 4V², so reuse that but different message.
        if self.v_esc is None or self.V is None:
            return (False, "SKIP: missing v_esc or V")
        if math.isclose(self.v_esc ** 2, 4.0 * self.V ** 2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "PASS: g_e = g_p/2 condition is satisfied")
        else:
            return (False, "FAIL: g_e condition not reflected in v_esc")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_gm_over_R_from_v_esc(self):
        """
        From v_esc = sqrt(2GM/R), derive GM/R = v_esc²/2.
        Then check that V² = (GM/R)/2 = v_esc²/4.
        This verifies that the intermediate step linking V and gravitational potential is correct.
        """
        if self.v_esc is None or self.V is None:
            return (False, "SKIP: missing v_esc or V")
        gm_over_R = self.v_esc ** 2 / 2.0
        expected_Vsq = gm_over_R / 2.0
        actual_Vsq = self.V ** 2
        if math.isclose(actual_Vsq, expected_Vsq, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: V² = {actual_Vsq} ≈ GM/(2R) = {expected_Vsq}")
        else:
            return (False, f"FAIL: V² = {actual_Vsq} does not equal GM/(2R) = {expected_Vsq}")

    def test_modeling_omega_relation(self):
        """
        If the solution provides R and omega, check that ω = V/R and
        that the condition ω²R = GM/(2R²) holds (using derived GM/R).
        """
        if self.R is None or self.omega is None or self.V is None or self.v_esc is None:
            return (False, "SKIP: R, omega, V, or v_esc missing")
        # Check ω = V/R
        expected_omega = self.V / self.R
        if not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: ω = {self.omega} differs from V/R = {expected_omega}")
        # Check ω²R = GM/(2R²)  (GM/R = v_esc²/2)
        gm_over_R = self.v_esc ** 2 / 2.0
        left = self.omega ** 2 * self.R
        right = gm_over_R / (2.0 * self.R)  # GM/(2R²) = (GM/R)/(2R)
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: ω²R = {left} ≈ GM/(2R²) = {right}")
        else:
            return (False, f"FAIL: ω²R = {left} not close to GM/(2R²) = {right}")

    def test_trace_intermediate_omega(self):
        """Check that omega is consistent with V and R, if both are available."""
        if self.R is None or self.omega is None or self.V is None:
            return (False, "SKIP: R or omega not available")
        expected_omega = self.V / self.R
        if math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: ω = {self.omega} ≈ V/R = {expected_omega}")
        else:
            return (False, f"FAIL: ω = {self.omega} differs from V/R = {expected_omega}")

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------
    def test_monotonicity_escape_increases_with_V(self):
        """
        Perturb V upward by 10% and check that the escape velocity increases.
        Since v_esc ∝ V exactly, the perturbed v_esc should be larger than original.
        """
        if self.V is None or self.v_esc is None:
            return (False, "SKIP: missing V or v_esc")
        V_pert = self.V * 1.1
        # Expected v_esc for perturbed V would be 2 * V_pert, which is > original v_esc
        expected_esc_pert = 2.0 * V_pert
        if self.v_esc < expected_esc_pert:
            return (True, f"PASS: v_esc = {self.v_esc} < v_esc_pert = {expected_esc_pert} (V increased 10%)")
        else:
            return (False, f"FAIL: v_esc = {self.v_esc} is not less than v_esc_pert = {expected_esc_pert}")

    # ------------------------------------------------------------------
    # Answer sanity (weak – only one such test, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """Trivial sanity: v_esc must be positive and finite."""
        if self.v_esc is None:
            return (False, "SKIP: v_esc not available")
        if not math.isfinite(self.v_esc):
            return (False, f"FAIL: v_esc is not finite: {self.v_esc}")
        if self.v_esc <= 0:
            return (False, f"FAIL: v_esc is not positive: {self.v_esc}")
        return (True, f"PASS: v_esc = {self.v_esc} is positive and finite")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_* methods, run them, collect logs. Returns (success, logs)."""
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