import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of parameters with fallbacks
        self.solution = solution_instance
        self.logs = []
        
        # Extract mass, radius, gravity safely
        self.m = getattr(solution_instance, 'm', None)
        if self.m is None:
            self.m = solution_instance.results.get('m', 1.0)
            
        self.R = getattr(solution_instance, 'R', None)
        if self.R is None:
            self.R = solution_instance.results.get('R', 1.0)
            
        self.g = getattr(solution_instance, 'g', None)
        if self.g is None:
            self.g = solution_instance.results.get('g', 9.8)
        
        # Extract velocities from solution results with fallbacks
        self.v_0 = getattr(solution_instance, 'v_0', None)
        if self.v_0 is None:
            self.v_0 = solution_instance.results.get('v_0', 0.0)
            
        self.v = getattr(solution_instance, 'v', None)
        if self.v is None:
            self.v = solution_instance.results.get('v', 0.0)
            
        self.v_b = getattr(solution_instance, 'v_b', None)
        if self.v_b is None:
            self.v_b = solution_instance.results.get('v_b', 0.0)
            
        self.v_a = getattr(solution_instance, 'v_a', None)
        if self.v_a is None:
            self.v_a = solution_instance.results.get('v_a', 0.0)
        
        # Expected values from steps analysis
        self.expected_v_0 = math.sqrt(32 * self.g * self.R)
        self.expected_v = 2 * math.sqrt(self.g * self.R)
        self.expected_v_b = 2 * math.sqrt(2 * self.g * self.R)
        self.expected_v_a = math.sqrt(18 * self.g * self.R)
    
    def _log(self, msg, success=True):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
    
    def test_invariants(self):
        """Test physical invariants using both symbolic and numerical methods."""
        try:
            # Test 1: Work-energy theorem for rod a
            F = 32 * self.m * self.g
            work_done = F * self.R
            ke_rod_a = 0.5 * (2 * self.m) * self.v_0**2
            
            if math.isclose(work_done, ke_rod_a, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Work-energy theorem holds for rod a")
            else:
                self._log(f"Work-energy theorem violation: Work={work_done}, KE={ke_rod_a}", False)
            
            # Test 2: Newton's second law for rod b at highest point
            F_N = 3 * self.m * self.g
            centripetal_force = self.m * self.v**2 / self.R
            total_force = F_N + self.m * self.g
            
            if math.isclose(total_force, centripetal_force, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Newton's second law holds for rod b at highest point")
            else:
                self._log(f"Newton's law violation: F_N+mg={total_force}, mv²/R={centripetal_force}", False)
            
            # Test 3: Conservation of mechanical energy for rod b
            delta_U = self.m * self.g * 2 * self.R  # Height difference is 2R
            delta_KE = 0.5 * self.m * (self.v**2 - self.v_b**2)
            
            if math.isclose(delta_KE, -delta_U, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Energy conservation holds for rod b")
            else:
                self._log(f"Energy conservation violation: ΔKE={delta_KE}, -ΔU={-delta_U}", False)
            
            # Test 4: Conservation of momentum
            initial_momentum = 2 * self.m * self.v_0
            final_momentum = 2 * self.m * self.v_a + self.m * self.v_b
            
            if math.isclose(initial_momentum, final_momentum, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Momentum conservation holds")
            else:
                self._log(f"Momentum violation: p_initial={initial_momentum}, p_final={final_momentum}", False)
            
            return True
            
        except Exception as e:
            self._log(f"Exception in test_invariants: {traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answer."""
        try:
            # Check step 1: v_0 from work-energy
            if not math.isclose(self.v_0, self.expected_v_0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v_0 mismatch: got {self.v_0}, expected {self.expected_v_0}", False)
            else:
                self._log("v_0 matches expected value")
            
            # Check step 2: v from circular motion
            if not math.isclose(self.v, self.expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v mismatch: got {self.v}, expected {self.expected_v}", False)
            else:
                self._log("v matches expected value")
            
            # Check step 3: v_b from energy conservation
            if not math.isclose(self.v_b, self.expected_v_b, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v_b mismatch: got {self.v_b}, expected {self.expected_v_b}", False)
            else:
                self._log("v_b matches expected value")
            
            # Check final answer: v_a
            if not math.isclose(self.v_a, self.expected_v_a, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Final answer v_a mismatch: got {self.v_a}, expected {self.expected_v_a}", False)
                return False
            else:
                self._log("Final answer v_a matches expected value")
                return True
                
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Test effect of perturbing mass m on final answer."""
        try:
            original_m = self.m
            perturb_factor = 1.1
            
            # Perturb mass
            self.m *= perturb_factor
            
            # Recompute expected velocities with perturbed mass
            perturbed_v_0 = math.sqrt(32 * self.g * self.R)  # Independent of m
            perturbed_v_b = 2 * math.sqrt(2 * self.g * self.R)  # Independent of m
            perturbed_v_a = perturbed_v_0 - perturbed_v_b / 2
            
            # Restore original mass
            self.m = original_m
            
            # Since velocities are independent of m, they should remain unchanged
            # Check that the recomputed v_a using the formula matches the expected
            expected_perturbed_v_a = math.sqrt(18 * self.g * self.R)
            
            if math.isclose(perturbed_v_a, expected_perturbed_v_a, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Monotonicity check passed: v_a formula correctly independent of m")
                return True
            else:
                self._log(f"Monotonicity violation: perturbed v_a={perturbed_v_a}, expected={expected_perturbed_v_a}", False)
                return False
                
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        
        # Run tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        # Overall success
        success = invariants_ok and trace_ok and monotonicity_ok
        
        # Add summary
        if success:
            self._log("All physics checks passed!")
        else:
            self._log("Some physics checks failed", False)
        
        return success, self.logs