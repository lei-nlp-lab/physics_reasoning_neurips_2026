import sympy as sp
import math
import traceback
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', None))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', None))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', None))
        
        # Safe extraction of results using mapping from KB
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', None))
        self.x_prime = getattr(solution_instance, 'x_prime', solution_instance.results.get("x'", None))
        self.F_net = getattr(solution_instance, 'F_net', solution_instance.results.get('F_net', None))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', None))
        self.a_prime = getattr(solution_instance, 'a_prime', solution_instance.results.get("a'", None))
        
        self.logs = []
        
    def _log(self, msg, is_error=False):
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {msg}")
        
    def test_invariants(self):
        """Verify Newton's Laws and Energy Conservation from KB"""
        success = True
        
        try:
            # Symbolic setup
            m, g, theta, v, x = sp.symbols('m g theta v x', positive=True)
            a_sym = v**2 / (2*x)
            F_net_sym = m * a_sym
            F_sym = m*g*sp.sin(theta) + m*v**2/(2*x)
            a_prime_sym = g*sp.sin(theta)
            x_prime_sym = v**2 / (2*g*sp.sin(theta))
            
            # Test 1: Kinematic consistency (step 1)
            if self.a is not None and self.v is not None and self.x is not None:
                a_expected = self.v**2 / (2*self.x)
                if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Kinematic invariant failed: a={self.a}, expected {a_expected}", True)
                    success = False
                else:
                    self._log(f"Kinematic check passed: a = v²/(2x)")
            
            # Test 2: Newton's 2nd Law during force application (step 2)
            if self.F_net is not None and self.m is not None and self.a is not None:
                if not math.isclose(self.F_net, self.m * self.a, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Newton's 2nd Law (force period) failed: F_net={self.F_net}, m*a={self.m*self.a}", True)
                    success = False
                else:
                    self._log(f"Newton's 2nd Law (force period) check passed")
            
            # Test 3: Force decomposition (step 3)
            if self.F is not None and self.m is not None and self.g is not None and self.theta is not None and self.v is not None and self.x is not None:
                F_expected = self.m*self.g*math.sin(self.theta) + self.m*self.v**2/(2*self.x)
                if not math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Force decomposition failed: F={self.F}, expected {F_expected}", True)
                    success = False
                else:
                    self._log(f"Force decomposition check passed")
                    
            # Test 4: Newton's 2nd Law after force removal (step 4)
            if self.a_prime is not None and self.g is not None and self.theta is not None:
                a_prime_expected = self.g * math.sin(self.theta)
                if not math.isclose(self.a_prime, a_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Newton's 2nd Law (after force) failed: a'={self.a_prime}, expected {a_prime_expected}", True)
                    success = False
                else:
                    self._log(f"Newton's 2nd Law (after force) check passed")
            
            # Test 5: Energy-like relationship (step 5)
            if self.x_prime is not None and self.v is not None and self.a_prime is not None:
                x_prime_expected = self.v**2 / (2*self.a_prime)
                if not math.isclose(self.x_prime, x_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Energy-like relation failed: x'={self.x_prime}, expected {x_prime_expected}", True)
                    success = False
                else:
                    self._log(f"Energy-like relation check passed")
                    
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}\n{traceback.format_exc()}", True)
            success = False
            
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        success = True
        
        try:
            # Check that step-by-step results are consistent
            if all(val is not None for val in [self.a, self.v, self.x]):
                # From step 1: a should be v²/(2x)
                expected_a = self.v**2 / (2*self.x)
                if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency (step 1): a mismatch", True)
                    success = False
                    
            if all(val is not None for val in [self.F_net, self.m, self.a]):
                # From step 2: F_net should equal m*a
                if not math.isclose(self.F_net, self.m * self.a, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency (step 2): F_net mismatch", True)
                    success = False
                    
            if all(val is not None for val in [self.F, self.m, self.g, self.theta, self.v, self.x]):
                # From step 3: F should equal m*g*sinθ + m*v²/(2x)
                expected_F = self.m*self.g*math.sin(self.theta) + self.m*self.v**2/(2*self.x)
                if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency (step 3): F mismatch", True)
                    success = False
                    
            if all(val is not None for val in [self.a_prime, self.g, self.theta]):
                # From step 4: a' should equal g*sinθ
                expected_a_prime = self.g * math.sin(self.theta)
                if not math.isclose(self.a_prime, expected_a_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency (step 4): a' mismatch", True)
                    success = False
                    
            if all(val is not None for val in [self.x_prime, self.v, self.a_prime]):
                # From step 5: x' should equal v²/(2a')
                expected_x_prime = self.v**2 / (2*self.a_prime)
                if not math.isclose(self.x_prime, expected_x_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency (step 5): x' mismatch", True)
                    success = False
                    
            if all(val is not None for val in [self.x_prime, self.v, self.g, self.theta]):
                # From step 6: x' should equal v²/(2g*sinθ)
                expected_x_prime_final = self.v**2 / (2*self.g*math.sin(self.theta))
                if not math.isclose(self.x_prime, expected_x_prime_final, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency (step 6): final x' mismatch", True)
                    success = False
                    
            if success:
                self._log("All trace consistency checks passed")
                
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", True)
            success = False
            
        return success
    
    def test_monotonicity(self):
        """Perturb mass and check trends in results"""
        success = True
        
        try:
            # Store original values
            original_m = self.m
            
            if None in [original_m, self.g, self.theta, self.v, self.x, self.F, self.x_prime]:
                self._log("Skipping monotonicity test due to missing data")
                return True  # Skip, not fail
            
            # Create perturbed version (increase mass by 10%)
            perturbed_m = original_m * 1.1
            
            # Recalculate using formulas from KB
            # Formula for F from step 3
            F_perturbed = perturbed_m * self.g * math.sin(self.theta) + perturbed_m * self.v**2 / (2*self.x)
            
            # Formula for x' from step 6 (should be independent of m)
            x_prime_perturbed = self.v**2 / (2*self.g*math.sin(self.theta))
            
            # Check trends
            # F should increase proportionally with mass
            F_ratio_expected = perturbed_m / original_m  # Should be 1.1
            F_ratio_actual = F_perturbed / self.F
            
            if not math.isclose(F_ratio_actual, F_ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity test failed for F: ratio={F_ratio_actual}, expected={F_ratio_expected}", True)
                success = False
                
            # x' should remain constant (ratio = 1.0)
            x_prime_ratio = x_prime_perturbed / self.x_prime
            if not math.isclose(x_prime_ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity test failed for x': ratio={x_prime_ratio}, expected=1.0", True)
                success = False
                
            if success:
                self._log("Monotonicity tests passed: F scales with mass, x' independent of mass")
                
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}\n{traceback.format_exc()}", True)
            success = False
            
        return success
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []  # Reset logs
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Overall success
        overall_success = test1 and test2 and test3
        
        # Summary log
        summary = (f"Physics Checker Summary:\n"
                   f"  Invariants: {'PASS' if test1 else 'FAIL'}\n"
                   f"  Trace Consistency: {'PASS' if test2 else 'FAIL'}\n"
                   f"  Monotonicity: {'PASS' if test3 else 'FAIL'}\n"
                   f"  Overall: {'PASS' if overall_success else 'FAIL'}")
        self._log(summary)
        
        return overall_success, self.logs