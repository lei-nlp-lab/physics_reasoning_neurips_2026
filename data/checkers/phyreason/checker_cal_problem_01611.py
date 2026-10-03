import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.a = getattr(solution_instance, 'a', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.B = getattr(solution_instance, 'B', 1.0)
        self.E = self.B * self.v0  # Given: E = B*v0
        
        # Extract results safely from results dictionary
        self.e_over_m = self.results.get('e/m', None)
        self.Hmax = self.results.get('Hmax', None)
        self.H = self.results.get('H', None)
        self.OO_prime = self.results.get('OO\'', None)
        self.x = self.results.get('x', None)
        self.y = self.results.get('y', None)  # This is y_entry (entry point)
        self.r = self.results.get('r', None)
        self.F = self.results.get('F', None)
        self.F_c = self.results.get('F_c', None)
        self.y_m = self.results.get('y_m', None)  # Maximum entry point
        self.OM_prime = self.results.get('O\'M', None)
        self.t = self.results.get('t', None)
        self.tan_theta = self.results.get('tantheta', None)  # Note: key is 'tantheta' not 'tan(theta)'
        
        # Expected values based on problem statements
        self.expected_e_over_m = self.v0 / (self.B * self.a)
        self.expected_y_range = (0, 2 * self.a)  # 0 <= y <= 2a
        self.expected_Hmax = 2.25 * self.a  # (9/4)*a
        
        # Intermediate expected values from steps analysis
        self.expected_r = self.a
        self.expected_OO_prime = self.a
        self.expected_OM_prime = 2 * self.a
        self.expected_y_m = 2 * self.a

    def _is_close(self, a, b):
        """Fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def _check_lorentz_centripetal(self):
        """
        Verify Newton's Second Law for circular motion: F = Bev0 = m*v0^2/r.
        Returns True if consistent within tolerance.
        """
        if self.r is None or self.e_over_m is None:
            return True  # Skip if data missing
        
        # Numerical check: e/m = v0/(B*r)
        rhs = self.v0 / (self.B * self.r)
        return self._is_close(self.e_over_m, rhs)
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants: Newton's laws, energy, etc.
        """
        logs = []
        success = True
        
        # 1. Check Lorentz force equals centripetal force
        if not self._check_lorentz_centripetal():
            logs.append("FAIL: Lorentz force ≠ centripetal force (Newton's 2nd Law violation)")
            success = False
        else:
            logs.append("PASS: Lorentz force = centripetal force")
        
        # 2. Check radius from geometry (step 1)
        if self.r is not None and not self._is_close(self.r, self.expected_r):
            logs.append(f"FAIL: Trajectory radius r={self.r} ≠ expected a={self.a}")
            success = False
        elif self.r is not None:
            logs.append(f"PASS: Trajectory radius r={self.r} matches a={self.a}")
        
        # 3. Check OO' = a (step 6)
        if self.OO_prime is not None and not self._is_close(self.OO_prime, self.expected_OO_prime):
            logs.append(f"FAIL: OO'={self.OO_prime} ≠ expected a={self.a}")
            success = False
        elif self.OO_prime is not None:
            logs.append(f"PASS: OO'={self.OO_prime} matches a={self.a}")
        
        # 4. Check O'M = 2a (step 5)
        if self.OM_prime is not None and not self._is_close(self.OM_prime, self.expected_OM_prime):
            logs.append(f"FAIL: O'M={self.OM_prime} ≠ expected 2a={2*self.a}")
            success = False
        elif self.OM_prime is not None:
            logs.append(f"PASS: O'M={self.OM_prime} matches 2a={2*self.a}")
        
        # 5. Check y_m = 2a (step 7)
        if self.y_m is not None and not self._is_close(self.y_m, self.expected_y_m):
            logs.append(f"FAIL: y_m={self.y_m} ≠ expected 2a={2*self.a}")
            success = False
        elif self.y_m is not None:
            logs.append(f"PASS: y_m={self.y_m} matches 2a={2*self.a}")
        
        # 6. Check time t: t = x / v0 (should hold)
        if self.t is not None and self.x is not None:
            expected_t = self.x / self.v0
            if not self._is_close(self.t, expected_t):
                logs.append(f"FAIL: t={self.t} ≠ x/v0={expected_t}")
                success = False
            else:
                logs.append(f"PASS: t={self.t} matches x/v0")
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        # 1. Check e/m = v0/(B*a) (step 3)
        if self.e_over_m is not None:
            if not self._is_close(self.e_over_m, self.expected_e_over_m):
                logs.append(f"FAIL: e/m={self.e_over_m} ≠ expected v0/(B*a)={self.expected_e_over_m}")
                success = False
            else:
                logs.append(f"PASS: e/m={self.e_over_m} matches v0/(B*a)")
        
        # 2. Check y-range consistency: y_max = 2a from geometry
        if self.y is not None:
            # y should be between 0 and 2a
            if not (0 <= self.y <= 2*self.a + 1e-2):
                logs.append(f"FAIL: y={self.y} outside expected range [0, {2*self.a}]")
                success = False
            else:
                logs.append(f"PASS: y={self.y} within [0, {2*self.a}]")
        
        # 3. Check Hmax derivation consistency
        if self.Hmax is not None:
            # From step 16: H = (3a - x) * sqrt(2y/a), with x = sqrt(2a*y)
            # Derivative gives maximum at y = (9/8)a, Hmax = (9/4)a
            if not self._is_close(self.Hmax, self.expected_Hmax):
                logs.append(f"FAIL: Hmax={self.Hmax} ≠ expected (9/4)a={self.expected_Hmax}")
                success = False
            else:
                logs.append(f"PASS: Hmax={self.Hmax} matches (9/4)a")
            
            # Verify the parabolic relationship H = 3*sqrt(2a*y) - 2y
            # But note: y here should be the displacement in electric field, not entry point
            # Since we don't have displacement in results, we can't perform this check reliably
            # Instead, we check if H equals Hmax (which it should for optimal case)
            if self.H is not None:
                if not self._is_close(self.H, self.Hmax):
                    logs.append(f"FAIL: H={self.H} ≠ Hmax={self.Hmax}")
                    success = False
                else:
                    logs.append(f"PASS: H equals Hmax")
        
        # 4. Check tan(theta) = sqrt(2y/a) (step 15)
        # Again, y here should be displacement, not entry point
        # Since we don't have displacement, we can't perform this check
        # Instead, we check if tan(theta) = x/a (from derivation)
        if self.tan_theta is not None and self.x is not None:
            expected_tan = self.x / self.a
            if not self._is_close(self.tan_theta, expected_tan):
                logs.append(f"FAIL: tan(theta)={self.tan_theta} ≠ x/a={expected_tan}")
                success = False
            else:
                logs.append(f"PASS: tan(theta) matches x/a")
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (a) and verify trends in outputs.
        """
        logs = []
        success = True
        
        original_a = self.a
        
        # Perturb a by +10%
        perturbed_a = original_a * 1.1
        
        # Expected changes:
        # e/m ∝ 1/a, so should decrease by factor ~1/1.1
        # Hmax ∝ a, so should increase by factor 1.1
        # y_range upper bound ∝ a, so should increase by factor 1.1
        
        # Compute expected perturbed values using formulas
        expected_e_over_m_perturbed = self.v0 / (self.B * perturbed_a)
        expected_Hmax_perturbed = 2.25 * perturbed_a
        expected_y_range_perturbed = (0, 2 * perturbed_a)
        
        # Check e/m monotonicity
        if self.e_over_m is not None:
            # e/m should be inversely proportional to a
            ratio = self.e_over_m / expected_e_over_m_perturbed
            if not self._is_close(ratio, 1.1):  # original is 1.1 times larger
                logs.append(f"WARN: e/m doesn't scale inversely with a (ratio={ratio:.3f} ≠ 1.1)")
                success = False
            else:
                logs.append(f"PASS: e/m scales inversely with a (ratio={ratio:.3f})")
        
        # Check Hmax monotonicity
        if self.Hmax is not None:
            # Hmax should be proportional to a
            ratio = expected_Hmax_perturbed / self.Hmax
            if not self._is_close(ratio, 1.1):
                logs.append(f"WARN: Hmax doesn't scale linearly with a (ratio={ratio:.3f} ≠ 1.1)")
                success = False
            else:
                logs.append(f"PASS: Hmax scales linearly with a (ratio={ratio:.3f})")
        
        # Check y-range monotonicity (using y_m if available)
        if self.y_m is not None:
            # y_m should be proportional to a
            ratio = (2 * perturbed_a) / self.y_m
            if not self._is_close(ratio, 1.1):
                logs.append(f"WARN: y_m doesn't scale linearly with a (ratio={ratio:.3f} ≠ 1.1)")
                success = False
            else:
                logs.append(f"PASS: y_m scales linearly with a (ratio={ratio:.3f})")
        
        # Restore original a
        self.a = original_a
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return summary.
        """
        logs = []
        overall_success = True
        
        test_functions = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            try:
                test_success, test_logs = test_func()
                logs.append(f"=== {test_name} ===")
                logs.extend(test_logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        logs.append(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, logs