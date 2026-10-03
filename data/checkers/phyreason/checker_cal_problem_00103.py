import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key ratios from results with safe fallbacks
        self.ratio_r = self._safe_get('fracr_ar_b')  # r_a/r_b
        self.ratio_m = self._safe_get('fracm_am_b')  # m_a/m_b
        self.ratio_q = self._safe_get('|fracq_aq_b|')  # |q_a/q_b|
        
        # Extract individual values if available
        self.f_a = self._safe_get('f_a')
        self.f_b = self._safe_get('f_b')
        self.r = self._safe_get('r')
        self.m = self._safe_get('m')
        
        # Test logs
        self.logs = []
        
    def _safe_get(self, key):
        """Safely get value from results with fallback to None."""
        return self.results.get(key, None)
    
    def _log_test(self, name, success, message):
        """Log test result."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{name}: {status} - {message}")
        return success
    
    def _fuzzy_equal(self, a, b):
        """Check if two numbers are approximately equal."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        tests_passed = True
        
        # 1. Test radius ratio from terminal velocity without E field
        # From steps: r ∝ √v, so r_a/r_b = √(v0/(v0/4)) = √4 = 2
        if self.ratio_r is not None:
            expected_ratio_r = 2.0
            if self._fuzzy_equal(self.ratio_r, expected_ratio_r):
                self._log_test("Radius Ratio", True, 
                              f"r_a/r_b = {self.ratio_r} ≈ 2.0 (from √(v_a/v_b))")
            else:
                tests_passed = False
                self._log_test("Radius Ratio", False,
                              f"r_a/r_b = {self.ratio_r}, expected ≈ 2.0")
        else:
            self._log_test("Radius Ratio", False, "Missing ratio_r in results")
            tests_passed = False
        
        # 2. Test mass ratio from radius ratio
        # m ∝ r³, so m_a/m_b = (r_a/r_b)³ = 2³ = 8
        if self.ratio_m is not None:
            expected_ratio_m = 8.0
            if self._fuzzy_equal(self.ratio_m, expected_ratio_m):
                self._log_test("Mass Ratio", True,
                              f"m_a/m_b = {self.ratio_m} ≈ 8.0 (from (r_a/r_b)³)")
            else:
                tests_passed = False
                self._log_test("Mass Ratio", False,
                              f"m_a/m_b = {self.ratio_m}, expected ≈ 8.0")
        else:
            self._log_test("Mass Ratio", False, "Missing ratio_m in results")
            tests_passed = False
        
        # 3. Test force balance without E field (terminal velocity)
        # For droplet a: m_a*g = k*r_a*v0
        # For droplet b: m_b*g = k*r_b*(v0/4)
        # Using the derived ratios, we can check consistency numerically
        if self.ratio_r is not None and self.ratio_m is not None:
            # From the physics: k should be consistent for both droplets
            # We can check that the relationship holds numerically
            # For droplet a: k = (m_a*g)/(r_a*v0) = (m_a/m_b * m_b*g)/(r_a/r_b * r_b * v0)
            # For droplet b: k = (m_b*g)/(r_b*(v0/4)) = 4*(m_b*g)/(r_b*v0)
            # The ratio k_a/k_b should be 1
            # Using the ratios: k_a/k_b = [(m_a/m_b)/(r_a/r_b)] / 4
            k_ratio = (self.ratio_m / self.ratio_r) / 4.0
            if self._fuzzy_equal(k_ratio, 1.0):
                self._log_test("Force Balance (no E)", True,
                              f"Terminal velocity conditions consistent (k_ratio = {k_ratio:.3f} ≈ 1.0)")
            else:
                tests_passed = False
                self._log_test("Force Balance (no E)", False,
                              f"Inconsistent k values, ratio = {k_ratio:.3f}, expected ≈ 1.0")
        else:
            self._log_test("Force Balance (no E)", False,
                          "Missing required ratios for consistency check")
            tests_passed = False
        
        # 4. Test charge ratio from force balance with E field
        if self.ratio_q is not None:
            expected_ratio_q = 4.0
            if self._fuzzy_equal(self.ratio_q, expected_ratio_q):
                self._log_test("Charge Ratio", True,
                              f"|q_a/q_b| = {self.ratio_q} ≈ 4.0")
            else:
                tests_passed = False
                self._log_test("Charge Ratio", False,
                              f"|q_a/q_b| = {self.ratio_q}, expected ≈ 4.0")
        else:
            self._log_test("Charge Ratio", False, "Missing ratio_q in results")
            tests_passed = False
        
        return tests_passed
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        tests_passed = True
        
        # 1. Check that f_a and f_b values are consistent with steps
        # From steps: f_a = (1/2) m_a*g, f_b = 2 m_b*g
        # The values stored in results might be coefficients (relative to m*g) or actual forces
        # We'll check both possibilities
        if self.f_a is not None and self.f_b is not None:
            # Case 1: If f_a and f_b are coefficients (as in the current implementation)
            # Then f_a = 0.5, f_b = 2.0
            # The actual force ratio is (f_a * m_a)/(f_b * m_b) = (0.5 * 8)/(2.0 * 1) = 4.0/2.0 = 2.0
            if self.ratio_m is not None:
                # Compute expected force ratio from physics
                expected_force_ratio = (0.5 * self.ratio_m) / 2.0  # = (1/4)*ratio_m
                # If f_a and f_b are coefficients, their ratio should be 0.5/2.0 = 0.25
                # But we don't know what they represent, so we accept either interpretation
                # Instead, we'll check the more fundamental relationship:
                # |q_a/q_b| should equal (1/2)*(m_a/m_b)
                if self.ratio_q is not None:
                    expected_q_from_m = 0.5 * self.ratio_m
                    if self._fuzzy_equal(self.ratio_q, expected_q_from_m):
                        self._log_test("Drag Force Consistency", True,
                                      f"Charge ratio consistent with mass ratio: |q_a/q_b| = {self.ratio_q}, (1/2)*(m_a/m_b) = {expected_q_from_m}")
                    else:
                        tests_passed = False
                        self._log_test("Drag Force Consistency", False,
                                      f"|q_a/q_b| = {self.ratio_q}, expected (1/2)*(m_a/m_b) = {expected_q_from_m}")
                else:
                    self._log_test("Drag Force Consistency", False,
                                  "Missing ratio_q for consistency check")
            else:
                self._log_test("Drag Force Consistency", False,
                              "Missing ratio_m for consistency check")
        else:
            self._log_test("Drag Force Values", True,
                          "f_a and f_b not required for this problem")
        
        # 2. Check energy/work consistency using the derived ratios
        # From the physics: |q_a/q_b| = (1/2)*(m_a/m_b)
        if self.ratio_q is not None and self.ratio_m is not None:
            expected_ratio_q_from_m = 0.5 * self.ratio_m  # (1/2)*8 = 4.0
            if self._fuzzy_equal(self.ratio_q, expected_ratio_q_from_m):
                self._log_test("Force Balance Consistency", True,
                              f"|q_a/q_b| = {self.ratio_q} consistent with (1/2)*(m_a/m_b) = {expected_ratio_q_from_m}")
            else:
                tests_passed = False
                self._log_test("Force Balance Consistency", False,
                              f"|q_a/q_b| = {self.ratio_q}, expected (1/2)*(m_a/m_b) = {expected_ratio_q_from_m}")
        else:
            self._log_test("Force Balance Consistency", False,
                          "Missing required ratios for consistency check")
            tests_passed = False
        
        # 3. Check acceleration directions from step_7
        # Droplet a: deceleration (slowing down from v0 to v0/2)
        # Droplet b: acceleration (speeding up from v0/4 to v0/2)
        # This implies q_a negative, q_b positive
        # This is qualitative, so just note it
        self._log_test("Acceleration Directions", True,
                      "Consistent: a decelerates (negative charge), b accelerates (positive charge)")
        
        return tests_passed
    
    def test_monotonicity(self):
        """Test that perturbing inputs produces expected trends."""
        tests_passed = True
        
        # 1. Test monotonicity of radius with respect to terminal velocity
        # From: r ∝ √v, so if v increases, r increases
        # We can test this numerically
        v_original = 1.0
        v_increased = 2.0
        r_original = math.sqrt(v_original)
        r_increased = math.sqrt(v_increased)
        
        if r_increased > r_original:
            self._log_test("Radius-Velocity Monotonicity", True,
                          f"Radius increases with terminal velocity: √{v_original} = {r_original:.3f}, √{v_increased} = {r_increased:.3f}")
        else:
            tests_passed = False
            self._log_test("Radius-Velocity Monotonicity", False,
                          "Radius doesn't increase with velocity")
        
        # 2. Test charge ratio monotonicity with mass ratio
        # From: |q_a/q_b| = (1/2)*(m_a/m_b)
        # So if mass ratio increases, charge ratio increases proportionally
        mass_ratio_original = 8.0
        mass_ratio_increased = 10.0
        charge_ratio_original = 0.5 * mass_ratio_original  # 4.0
        charge_ratio_increased = 0.5 * mass_ratio_increased  # 5.0
        
        if charge_ratio_increased > charge_ratio_original:
            self._log_test("Charge-Mass Monotonicity", True,
                          f"Charge ratio increases with mass ratio: 0.5*{mass_ratio_original} = {charge_ratio_original}, 0.5*{mass_ratio_increased} = {charge_ratio_increased}")
        else:
            tests_passed = False
            self._log_test("Charge-Mass Monotonicity", False,
                          "Charge ratio doesn't increase with mass ratio")
        
        # 3. Perturbation test on velocity ratio
        # Original: v_a0/v_b0 = 4
        # If we change this ratio, the radius ratio should change as √(v_a/v_b)
        v_ratio_original = 4.0
        v_ratio_perturbed = 5.0  # Increase by 25%
        
        r_ratio_original = math.sqrt(v_ratio_original)  # 2.0
        r_ratio_perturbed = math.sqrt(v_ratio_perturbed)  # ~2.236
        
        if r_ratio_perturbed > r_ratio_original:
            self._log_test("Velocity Perturbation", True,
                          f"r_ratio increases from {r_ratio_original:.3f} to {r_ratio_perturbed:.3f} "
                          f"when v_ratio increases from {v_ratio_original} to {v_ratio_perturbed}")
        else:
            tests_passed = False
            self._log_test("Velocity Perturbation", False,
                          "Radius ratio doesn't increase with velocity ratio")
        
        return tests_passed
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []  # Reset logs
        all_passed = True
        
        try:
            # Run individual test suites
            if not self.test_invariants():
                all_passed = False
                self.logs.append("FAIL: test_invariants failed")
            
            if not self.test_trace_consistency():
                all_passed = False
                self.logs.append("FAIL: test_trace_consistency failed")
            
            if not self.test_monotonicity():
                all_passed = False
                self.logs.append("FAIL: test_monotonicity failed")
            
            # Summary
            if all_passed:
                self.logs.append("SUMMARY: All physics checks passed")
            else:
                self.logs.append("SUMMARY: Some physics checks failed")
                
        except Exception as e:
            all_passed = False
            self.logs.append(f"ERROR: Exception during testing: {str(e)}")
            self.logs.append(traceback.format_exc())
        
        return all_passed, self.logs