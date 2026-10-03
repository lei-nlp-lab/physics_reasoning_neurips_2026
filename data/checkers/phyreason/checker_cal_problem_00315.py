import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Constants from problem statement
        self.R = 0.5  # m
        self.L = 1.2  # m
        self.h0 = 0.8  # m
        self.m = 0.1  # kg
        self.g = 10.0  # m/s^2
        
        # Safely extract results using getattr with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract specific results with safe defaults
        self.F_N = getattr(solution_instance, 'F_N', self.results.get('F_N', None))
        self.x = getattr(solution_instance, 'x', self.results.get('x', None))
        self.v_2 = getattr(solution_instance, 'v_2', self.results.get('v_2', None))
        self.t_2 = getattr(solution_instance, 't_2', self.results.get('t_2', None))
        self.F_N_prime = getattr(solution_instance, 'F_N_prime', 
                                self.results.get('F_{N}\'', self.results.get('F_N_prime', None)))
        self.x_1 = getattr(solution_instance, 'x_1', self.results.get('x_1', None))
        self.x_2 = getattr(solution_instance, 'x_2', self.results.get('x_2', None))
        self.t_1 = getattr(solution_instance, 't_1', self.results.get('t_1', None))
        self.v_1 = getattr(solution_instance, 'v_1', self.results.get('v_1', None))
        
        # For energy conservation, we may need v_0 and v_C
        self.v_0 = getattr(solution_instance, 'v_0', self.results.get('v_0', None))
        self.v_C = getattr(solution_instance, 'v_C', self.results.get('v_C', None))
        
        # Logging
        self.logs = []
        
    def _log(self, msg: str):
        """Add message to logs."""
        self.logs.append(msg)
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2) -> bool:
        """Fuzzy numerical comparison with physics tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self) -> bool:
        """Test fundamental physics laws based on KB."""
        success = True
        
        # 1. Energy conservation from O to C (for both scenarios)
        # Work-energy theorem: -mg*2R = (1/2)m*v_C^2 - (1/2)m*v_0^2
        if self.v_0 is not None and self.v_C is not None:
            delta_KE = 0.5 * self.m * (self.v_C**2 - self.v_0**2)
            work_gravity = -self.m * self.g * 2 * self.R
            if not self._is_close(delta_KE, work_gravity):
                self._log(f"Energy conservation O->C violated: ΔKE={delta_KE:.3f}, W_gravity={work_gravity:.3f}")
                success = False
            else:
                self._log("✓ Energy conservation O->C holds")
                
        # 2. Newton's 2nd law at point C
        # F_N + mg = m*v_C^2/R (upward positive)
        if self.F_N_prime is not None and self.v_C is not None:
            centripetal_force = self.m * self.v_C**2 / self.R
            net_force_up = self.F_N_prime + self.m * self.g
            if not self._is_close(net_force_up, centripetal_force):
                self._log(f"Newton's 2nd law at C violated: F_N+mg={net_force_up:.3f}, m*v²/R={centripetal_force:.3f}")
                success = False
            else:
                self._log("✓ Newton's 2nd law at C holds")
                
        # 3. Newton's 2nd law at point O (for scenario 2)
        # F_N - mg = m*v_0^2/R (downward positive)
        if self.F_N is not None and self.v_0 is not None:
            centripetal_force_O = self.m * self.v_0**2 / self.R
            net_force_down = self.F_N - self.m * self.g
            if not self._is_close(net_force_down, centripetal_force_O):
                self._log(f"Newton's 2nd law at O violated: F_N-mg={net_force_down:.3f}, m*v₀²/R={centripetal_force_O:.3f}")
                success = False
            else:
                self._log("✓ Newton's 2nd law at O holds")
                
        # 4. Newton's 3rd law: F_N' should equal normal force at C
        if self.F_N_prime is not None:
            # In step_3, F_N' = F_N (normal force from track on ball)
            # This is trivially true by definition, but we check if value is physically reasonable
            if self.F_N_prime < 0:
                self._log(f"Newton's 3rd law warning: F_N'={self.F_N_prime:.3f} is negative")
                # Not necessarily failure, but log it
                
        # 5. Minimum speed at highest point for circular motion
        # mg = m*v_1^2/R -> v_1 = sqrt(gR)
        v_1_min = math.sqrt(self.g * self.R)
        if self.v_1 is not None:
            if not self._is_close(self.v_1, v_1_min):
                self._log(f"Minimum speed condition violated: v_1={self.v_1:.3f}, sqrt(gR)={v_1_min:.3f}")
                success = False
            else:
                self._log("✓ Minimum speed condition holds")
                
        return success
        
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check t_1 (time to fall from C to ground)
        # 2R = (1/2)*g*t_1^2
        if self.t_1 is not None:
            t_1_calc = math.sqrt(4 * self.R / self.g)
            if not self._is_close(self.t_1, t_1_calc):
                self._log(f"t_1 inconsistency: {self.t_1:.3f} vs calculated {t_1_calc:.3f}")
                success = False
            else:
                self._log("✓ t_1 consistent with free fall from height 2R")
                
        # 2. Check x_1 = v_1 * t_1
        if self.x_1 is not None and self.v_1 is not None and self.t_1 is not None:
            x_1_calc = self.v_1 * self.t_1
            if not self._is_close(self.x_1, x_1_calc):
                self._log(f"x_1 inconsistency: {self.x_1:.3f} vs v_1*t_1={x_1_calc:.3f}")
                success = False
            else:
                self._log("✓ x_1 = v_1 * t_1 holds")
                
        # 3. Check t_2 (time to fall from C to N)
        # 2R - h0 = (1/2)*g*t_2^2
        if self.t_2 is not None:
            t_2_calc = math.sqrt(2 * (2 * self.R - self.h0) / self.g)
            if not self._is_close(self.t_2, t_2_calc):
                self._log(f"t_2 inconsistency: {self.t_2:.3f} vs calculated {t_2_calc:.3f}")
                success = False
            else:
                self._log("✓ t_2 consistent with free fall from height (2R-h0)")
                
        # 4. Check v_2 = L / t_2
        if self.v_2 is not None and self.t_2 is not None:
            v_2_calc = self.L / self.t_2
            if not self._is_close(self.v_2, v_2_calc):
                self._log(f"v_2 inconsistency: {self.v_2:.3f} vs L/t_2={v_2_calc:.3f}")
                success = False
            else:
                self._log("✓ v_2 = L / t_2 holds")
                
        # 5. Check x_2 = v_2 * t_1
        if self.x_2 is not None and self.v_2 is not None and self.t_1 is not None:
            x_2_calc = self.v_2 * self.t_1
            if not self._is_close(self.x_2, x_2_calc):
                self._log(f"x_2 inconsistency: {self.x_2:.3f} vs v_2*t_1={x_2_calc:.3f}")
                success = False
            else:
                self._log("✓ x_2 = v_2 * t_1 holds")
                
        # 6. Check range bounds
        if self.x_1 is not None and self.x_2 is not None:
            # x should be between x_1 and x_2
            if self.x is not None:
                # Try to parse range if it's a string
                if isinstance(self.x, str):
                    # Extract numbers from string like "1 to 2.68"
                    import re
                    nums = re.findall(r'\d+\.?\d*', self.x)
                    if len(nums) >= 2:
                        lower = float(nums[0])
                        upper = float(nums[1])
                        if not (self._is_close(lower, self.x_1) and self._is_close(upper, self.x_2)):
                            self._log(f"Range inconsistency: x={self.x}, but x_1={self.x_1}, x_2={self.x_2}")
                            success = False
                        else:
                            self._log("✓ Range bounds consistent")
                else:
                    # x might be a tuple or list
                    pass
                    
        return success
        
    def test_monotonicity(self) -> bool:
        """Perturb mass and check physical trends."""
        success = True
        
        try:
            # Store original values
            original_m = self.m
            
            # Test 1: Increase mass, normal forces should increase proportionally
            # F_N = m*(v_0^2/R + g) from Newton's 2nd law at O
            if self.F_N is not None and self.v_0 is not None:
                perturbed_m = original_m * 1.1  # +10%
                F_N_perturbed = perturbed_m * (self.v_0**2 / self.R + self.g)
                if F_N_perturbed <= self.F_N:
                    self._log(f"Monotonicity violation: F_N should increase with mass")
                    success = False
                else:
                    self._log("✓ F_N increases with mass (as expected)")
                    
            # Test 2: For F_N_prime = m*(v_C^2/R - g), check trend with mass
            if self.F_N_prime is not None and self.v_C is not None:
                perturbed_m = original_m * 1.1
                F_N_prime_perturbed = perturbed_m * (self.v_C**2 / self.R - self.g)
                if F_N_prime_perturbed <= self.F_N_prime:
                    self._log(f"Monotonicity violation: F_N' should increase with mass")
                    success = False
                else:
                    self._log("✓ F_N' increases with mass (as expected)")
                    
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}")
            success = False
            
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics checks and return results."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- {test_name} Test ---")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"✗ {test_name} FAILED")
                else:
                    self._log(f"✓ {test_name} passed")
            except Exception as e:
                self._log(f"✗ {test_name} ERROR: {str(e)}")
                self._log(traceback.format_exc())
                overall_success = False
                
        self._log(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs

# Example usage for testing
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'F_N': 12.2,
                'x': "1 to 2.68",
                'v_2': 6.0,
                't_2': 0.2,
                'F_N_prime': 2.2,
                'x_1': 1.0,
                'x_2': 2.6832,
                't_1': math.sqrt(0.2),  # ~0.4472
                'v_1': math.sqrt(5),  # ~2.236
                'v_0': math.sqrt(56),  # ~7.483 for scenario 2
                'v_C': 4.0,  # for scenario 1
            }
            
    # Create checker and run tests
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)