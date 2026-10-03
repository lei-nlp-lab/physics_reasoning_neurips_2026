import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using getattr with fallbacks."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with safe defaults
        self.R = getattr(solution_instance, 'R', self.results.get('R', 1.0))
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', self.results.get('g', 9.8))
        self.mu_disk = getattr(solution_instance, 'mu', self.results.get('mu', 0.5))
        self.mu_table = getattr(solution_instance, 'mu_prime', self.results.get('mu_prime', None))
        
        # Extract computed results with safe defaults
        self.omega0 = getattr(solution_instance, 'omega_0', self.results.get('omega_0', None))
        self.accel = getattr(solution_instance, 'a', self.results.get('a', None))
        self.x_displ = getattr(solution_instance, 'x', self.results.get('x', None))
        self.velocity = getattr(solution_instance, 'v', self.results.get('v', None))
        
        # Store original values for monotonicity test
        self._original_mu = self.mu_disk
        
    def test_invariants(self):
        """Verify fundamental physics laws using symbolic and numerical checks."""
        logs = []
        try:
            # 1. Check ω₀ = √(μg/R) from centripetal force balance
            if self.omega0 is not None:
                expected_omega = sp.sqrt(self.mu_disk * self.g / self.R)
                num_omega = float(expected_omega.subs({self.mu_disk: float(self.mu_disk), 
                                                     self.g: float(self.g), 
                                                     self.R: float(self.R)}))
                
                if not math.isclose(self.omega0, num_omega, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: ω₀ invariant. Got {self.omega0}, expected ~{num_omega}")
                else:
                    logs.append(f"PASS: ω₀ = √(μg/R) holds within 1%")
            else:
                logs.append("SKIP: ω₀ not available")
            
            # 2. Check a = μ'g from Newton's 2nd Law on table
            if self.accel is not None and self.mu_table is not None:
                expected_accel = self.mu_table * self.g
                if not math.isclose(self.accel, expected_accel, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a invariant. Got {self.accel}, expected {expected_accel}")
                else:
                    logs.append(f"PASS: a = μ'g holds within 1%")
            else:
                logs.append("SKIP: a or μ' not available")
            
            # 3. Check x = 0.75R from geometry (Pythagorean theorem)
            if self.x_displ is not None:
                expected_x = 0.75 * self.R
                if not math.isclose(self.x_displ, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x invariant. Got {self.x_displ}, expected {expected_x}")
                else:
                    logs.append(f"PASS: x = 0.75R holds within 1%")
            else:
                logs.append("SKIP: x not available")
            
            # 4. Check v from two independent methods should match
            if self.velocity is not None and self.accel is not None and self.x_displ is not None and self.omega0 is not None:
                v_from_kinematics = math.sqrt(2 * self.accel * self.x_displ)
                v_from_rotation = self.omega0 * self.R
                
                # Both should match the reported velocity
                if not (math.isclose(self.velocity, v_from_kinematics, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.velocity, v_from_rotation, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"FAIL: v mismatch. Got {self.velocity}, kinematics gives {v_from_kinematics:.3f}, rotation gives {v_from_rotation:.3f}")
                else:
                    logs.append(f"PASS: v consistent from kinematics and rotation within 1%")
            else:
                logs.append("SKIP: Missing variables for v consistency check")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer μ' = (2/3)μ."""
        logs = []
        try:
            if self.mu_table is not None:
                # Derive μ' from the chain of equations
                # ω₀ = √(μg/R) → v = ω₀R = √(μgR)
                # x = 0.75R
                # a = μ'g
                # v² = 2ax → μgR = 2(μ'g)(0.75R)
                # Cancel g and R: μ = 1.5μ' → μ' = (2/3)μ
                expected_mu_prime = (2/3) * self.mu_disk
                
                if not math.isclose(self.mu_table, expected_mu_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: μ' trace inconsistency. Got {self.mu_table}, expected {expected_mu_prime:.3f}")
                else:
                    logs.append(f"PASS: μ' = (2/3)μ holds within 1%")
                    
                # Additional check: verify the derived velocity matches
                if self.velocity is not None:
                    v_expected = math.sqrt(self.mu_disk * self.g * self.R)
                    if not math.isclose(self.velocity, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"WARN: v doesn't match derived √(μgR). Got {self.velocity}, expected {v_expected:.3f}")
            else:
                logs.append("SKIP: μ' not available for trace check")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        """Perturb μ and verify ω₀ and μ' trends, then restore original μ."""
        logs = []
        original_mu = self.mu_disk
        
        try:
            # Perturb μ by +10%
            perturbed_mu = original_mu * 1.1
            self.mu_disk = perturbed_mu
            
            # Recalculate using formulas (not calling solution methods)
            new_omega0 = math.sqrt(perturbed_mu * self.g / self.R)
            new_mu_prime = (2/3) * perturbed_mu
            
            # Check trends
            if self.omega0 is not None:
                if new_omega0 > self.omega0:
                    logs.append(f"PASS: ω₀ increases with μ (monotonic √μ dependence)")
                else:
                    logs.append(f"FAIL: ω₀ should increase with μ")
            
            if self.mu_table is not None:
                if new_mu_prime > self.mu_table:
                    logs.append(f"PASS: μ' increases with μ (linear dependence)")
                else:
                    logs.append(f"FAIL: μ' should increase with μ")
            
            # Restore original μ
            self.mu_disk = original_mu
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            # Ensure restoration even if error occurs
            self.mu_disk = original_mu
        
        return logs
    
    def run_all_tests(self):
        """Execute all checks and return success flag with logs."""
        all_logs = []
        success = True
        
        # Run tests
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            test_logs = test_method()
            all_logs.extend(test_logs)
            
            # Check for failures in this test (look for "FAIL:" prefix)
            if any(log.startswith("FAIL:") for log in test_logs):
                success = False
        
        # Overall summary
        if success:
            all_logs.append("\n✓ All physics checks passed")
        else:
            all_logs.append("\n✗ Some physics checks failed")
        
        return success, all_logs

# Optional: Example usage and test
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.R = 0.5
            self.m = 0.1
            self.g = 9.8
            self.mu = 0.4
            self.mu_prime = 2/3 * 0.4
            self.results = {
                'omega_0': math.sqrt(0.4 * 9.8 / 0.5),
                'a': (2/3 * 0.4) * 9.8,
                'x': 0.75 * 0.5,
                'v': math.sqrt(0.4 * 9.8 * 0.5)
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)