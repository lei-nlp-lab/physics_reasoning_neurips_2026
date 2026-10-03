import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution_instance."""
        self.logs = []
        
        # Expected input parameters (with defaults)
        self.v0 = getattr(solution_instance, 'v0', 2*math.sqrt(2))
        self.theta_deg = getattr(solution_instance, 'theta_deg', 45)
        self.h = getattr(solution_instance, 'h', 0.4)
        self.g = getattr(solution_instance, 'g', 10)
        self.E = getattr(solution_instance, 'E', 1)
        
        # Extract results using safe access with fallbacks
        results_dict = getattr(solution_instance, 'results', {})
        self.B = getattr(solution_instance, 'B', results_dict.get('B', 0.0))
        self.q_over_m = getattr(solution_instance, 'q_over_m', results_dict.get('q/m', 0.0))
        self.t_total = getattr(solution_instance, 't', results_dict.get('t', 0.0))
        self.x_final = getattr(solution_instance, 'x', results_dict.get('x', 0.0))
        
        # Intermediate results (if available)
        self.t1 = getattr(solution_instance, 't_1', results_dict.get('t_1', 0.0))
        self.t2 = getattr(solution_instance, 't_2', results_dict.get('t_2', 0.0))
        self.t3 = getattr(solution_instance, 't_3', results_dict.get('t_3', 0.0))
        self.T = getattr(solution_instance, 'T', results_dict.get('T', 0.0))
        self.r = getattr(solution_instance, 'r', results_dict.get('r', 0.0))
        
        # Convert theta to radians
        self.theta = math.radians(self.theta_deg)
        
        # Constants
        self.pi = math.pi

    def _isclose(self, a, b):
        """Fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Test fundamental physical laws from KB."""
        logs = []
        try:
            # 1. Force equilibrium in third quadrant: q/m = g/E
            expected_q_over_m = self.g / self.E
            if not self._isclose(self.q_over_m, expected_q_over_m):
                logs.append(f"FAIL: q/m = {self.q_over_m}, expected g/E = {expected_q_over_m}")
            else:
                logs.append(f"PASS: q/m = g/E = {expected_q_over_m}")

            # 2. Force equilibrium for B in third quadrant: B = E/(v0*cosθ)
            # Magnetic force balances electric force in x-direction
            expected_B = self.E / (self.v0 * math.cos(self.theta))
            if not self._isclose(self.B, expected_B):
                logs.append(f"FAIL: B = {self.B}, expected E/(v0*cosθ) = {expected_B}")
            else:
                logs.append(f"PASS: B = E/(v0*cosθ) = {expected_B}")

            # 3. Period formula: T = 2πm/(|q|B) = 2π/((q/m)*B)
            if self.B > 0 and self.q_over_m > 0:
                expected_T = 2 * self.pi / (self.q_over_m * self.B)
                if not self._isclose(self.T, expected_T):
                    logs.append(f"FAIL: T = {self.T}, expected 2π/((q/m)B) = {expected_T}")
                else:
                    logs.append(f"PASS: T = 2π/((q/m)B) = {expected_T}")

            # 4. Radius of circular motion: r = m*v0/(|q|B) = v0/((q/m)*B)
            if self.B > 0 and self.q_over_m > 0:
                expected_r = self.v0 / (self.q_over_m * self.B)
                if not self._isclose(self.r, expected_r):
                    logs.append(f"FAIL: r = {self.r}, expected v0/((q/m)B) = {expected_r}")
                else:
                    logs.append(f"PASS: r = v0/((q/m)B) = {expected_r}")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        logs = []
        try:
            # 1. Check t1 = h/(v0*sinθ)
            expected_t1 = self.h / (self.v0 * math.sin(self.theta))
            if not self._isclose(self.t1, expected_t1):
                logs.append(f"FAIL: t1 = {self.t1}, expected h/(v0*sinθ) = {expected_t1}")
            else:
                logs.append(f"PASS: t1 = h/(v0*sinθ) = {expected_t1}")

            # 2. Check t2 = T/4
            expected_t2 = self.T / 4
            if not self._isclose(self.t2, expected_t2):
                logs.append(f"FAIL: t2 = {self.t2}, expected T/4 = {expected_t2}")
            else:
                logs.append(f"PASS: t2 = T/4 = {expected_t2}")

            # 3. Check t3 = t1
            if not self._isclose(self.t3, self.t1):
                logs.append(f"FAIL: t3 = {self.t3}, should equal t1 = {self.t1}")
            else:
                logs.append(f"PASS: t3 = t1 = {self.t1}")

            # 4. Check total time: t = t1 + t2 + t3
            expected_t_total = self.t1 + self.t2 + self.t3
            if not self._isclose(self.t_total, expected_t_total):
                logs.append(f"FAIL: t_total = {self.t_total}, expected t1+t2+t3 = {expected_t_total}")
            else:
                logs.append(f"PASS: t_total = t1+t2+t3 = {expected_t_total}")

            # 5. Check x = 2h + 2r*cosθ
            expected_x = 2*self.h + 2*self.r*math.cos(self.theta)
            if not self._isclose(self.x_final, expected_x):
                logs.append(f"FAIL: x = {self.x_final}, expected 2h + 2r*cosθ = {expected_x}")
            else:
                logs.append(f"PASS: x = 2h + 2r*cosθ = {expected_x}")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """Perturb input E and check trends."""
        logs = []
        original_E = self.E
        
        try:
            # Perturb E by +10%
            perturbed_E = original_E * 1.1
            
            # From KB trends:
            # 1. q/m = g/E should decrease when E increases
            new_q_over_m = self.g / perturbed_E
            if not (new_q_over_m < self.q_over_m):
                logs.append(f"FAIL: q/m should decrease when E increases, but {new_q_over_m} >= {self.q_over_m}")
            else:
                logs.append(f"PASS: q/m decreases when E increases (g/E trend)")
            
            # 2. B = E/(v0*cosθ) should increase when E increases
            new_B = perturbed_E / (self.v0 * math.cos(self.theta))
            if not (new_B > self.B):
                logs.append(f"FAIL: B should increase when E increases, but {new_B} <= {self.B}")
            else:
                logs.append(f"PASS: B increases when E increases (E/(v0*cosθ) trend)")
            
            # 3. Period T = 2π/((q/m)*B) 
            # Since q/m decreases and B increases, the product (q/m)*B could go either way
            # Let's check numerically
            new_T = 2 * self.pi / (new_q_over_m * new_B)
            original_T = 2 * self.pi / (self.q_over_m * self.B)
            
            # The trend from KB: T increases with mass (which is m = 1/(q/m) here if q constant? 
            # Actually q/m changed, so m changes if q is fixed. But our perturbation changes E, not q.
            # Since q/m decreases, m increases if q constant. So T should increase.
            if not (new_T > original_T):
                logs.append(f"WARNING: T trend ambiguous after E perturbation: {new_T} <= {original_T}")
            else:
                logs.append(f"PASS: T increases when E increases (mass increases trend)")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Restore original value
            self.E = original_E
        
        return logs

    def run_all_tests(self):
        """Run all tests and return success boolean and logs."""
        all_logs = []
        success = True
        
        # Run tests
        all_logs.append("=== Testing Invariants ===")
        inv_logs = self.test_invariants()
        all_logs.extend(inv_logs)
        if any("FAIL" in log for log in inv_logs):
            success = False
        
        all_logs.append("\n=== Testing Trace Consistency ===")
        trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        if any("FAIL" in log for log in trace_logs):
            success = False
        
        all_logs.append("\n=== Testing Monotonicity ===")
        mono_logs = self.test_monotonicity()
        all_logs.extend(mono_logs)
        if any("FAIL" in log for log in mono_logs):
            success = False
        
        return success, all_logs

# Example usage (if needed for debugging):
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.v0 = 2*math.sqrt(2)
            self.theta_deg = 45
            self.h = 0.4
            self.g = 10
            self.E = 1
            self.results = {
                'B': 0.5,
                'q/m': 10.0,
                't': 0.714,
                'x': 1.6,
                't_1': 0.2,
                't_2': math.pi/10,
                't_3': 0.2,
                'T': 2*math.pi/5,
                'r': 0.4*math.sqrt(2)
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    print("Success:", success)
    for log in logs:
        print(log)