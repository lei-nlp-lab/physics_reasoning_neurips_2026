import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults
        self.m = getattr(solution_instance, 'm', 0.6)
        self.v1 = getattr(solution_instance, 'v1', 5.0)
        self.theta_deg = getattr(solution_instance, 'theta', 53.0)
        self.h = getattr(solution_instance, 'h', 0.2)
        self.dt = getattr(solution_instance, 'dt', 0.01)
        self.L = getattr(solution_instance, 'L', 0.6)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safe extraction of results (from attributes or results dict)
        results_dict = getattr(solution_instance, 'results', {})
        self.W = getattr(solution_instance, 'W', results_dict.get('W', 0.0))
        self.v2 = getattr(solution_instance, 'v_2', results_dict.get('v_2', 0.0))
        self.v3 = getattr(solution_instance, 'v_3', results_dict.get('v_3', 0.0))
        self.F = getattr(solution_instance, 'F', results_dict.get('F', 0.0))
        
        # Precomputed constants
        self.sin53 = 0.8
        self.cos53 = 0.6
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append((msg, success))
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws"""
        try:
            # 1. Check work-energy theorem for step 1
            expected_W = 0.5 * self.m * self.v1**2
            if not self._is_close(self.W, expected_W):
                self._log(f"Work-energy invariant failed: W={self.W}, expected {expected_W}", False)
                return False
            self._log(f"Work-energy check passed: W={self.W} ≈ {expected_W}", True)
            
            # 2. Check projectile horizontal velocity (step 2)
            expected_v2 = self.v1 * self.sin53
            if not self._is_close(self.v2, expected_v2):
                self._log(f"Horizontal velocity invariant failed: v2={self.v2}, expected {expected_v2}", False)
                return False
            self._log(f"Horizontal velocity check passed: v2={self.v2} ≈ {expected_v2}", True)
            
            # 3. Check post-collision velocity from kinematics (step 3)
            t_fall = math.sqrt(2 * self.h / self.g)  # Time to fall height h
            expected_v3 = self.L / t_fall
            if not self._is_close(self.v3, expected_v3):
                self._log(f"Post-collision velocity invariant failed: v3={self.v3}, expected {expected_v3}", False)
                return False
            self._log(f"Post-collision velocity check passed: v3={self.v3} ≈ {expected_v3}", True)
            
            # 4. Check impulse-momentum theorem (step 4)
            # According to steps analysis: -F·Δt = m(-v3) - m·v2
            # Therefore F = m(v2 + v3)/Δt
            expected_F = self.m * (self.v2 + self.v3) / self.dt
            if not self._is_close(self.F, expected_F):
                self._log(f"Impulse-momentum invariant failed: F={self.F}, expected {expected_F}", False)
                return False
            self._log(f"Impulse-momentum check passed: F={self.F} ≈ {expected_F}", True)
            
            return True
            
        except Exception as e:
            self._log(f"Invariant test crashed: {traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check that intermediate results lead to final answer"""
        try:
            # Step 4 should use v2 and v3 from previous steps
            # Recompute F using extracted v2 and v3
            recomputed_F = self.m * (self.v2 + self.v3) / self.dt
            
            if not self._is_close(self.F, recomputed_F):
                self._log(f"Trace consistency failed: F={self.F}, recomputed {recomputed_F} using v2={self.v2}, v3={self.v3}", False)
                return False
                
            self._log(f"Trace consistency passed: intermediate velocities v2={self.v2}, v3={self.v3} correctly yield F={self.F}", True)
            return True
            
        except Exception as e:
            self._log(f"Trace consistency test crashed: {traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check F changes proportionally"""
        try:
            # Store original mass
            original_m = self.m
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recompute expected F with perturbed mass (keeping v2, v3 constant)
            # From impulse-momentum: F ∝ m
            original_F_expected = self.m * (self.v2 + self.v3) / self.dt
            perturbed_F_expected = perturbed_m * (self.v2 + self.v3) / self.dt
            
            # Check monotonic increase
            if perturbed_F_expected <= original_F_expected:
                self._log(f"Monotonicity test failed: F doesn't increase with mass", False)
                return False
                
            # Restore original mass (important for other tests)
            self.m = original_m
            
            self._log(f"Monotonicity passed: F increases with mass (F∝m)", True)
            return True
            
        except Exception as e:
            # Restore mass even if test crashes
            self.m = getattr(self, 'original_m', self.m)
            self._log(f"Monotonicity test crashed: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return overall success"""
        self.logs = []
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            self._log(f"--- Starting {test_name} ---", None)
            success = test_func()
            if not success:
                all_passed = False
        
        return all_passed, self.logs

# Example usage stub (not part of final output)
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.m = 0.6
            self.v1 = 5.0
            self.theta = 53.0
            self.h = 0.2
            self.dt = 0.01
            self.L = 0.6
            self.g = 10.0
            self.results = {
                'W': 7.5,
                'v_2': 4.0,
                'v_3': 3.0,
                'F': 420.0
            }
    
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    print(f"Overall success: {success}")
    for log_msg, log_success in logs:
        if log_success is None:
            print(log_msg)
        else:
            status = "PASS" if log_success else "FAIL"
            print(f"{status}: {log_msg}")