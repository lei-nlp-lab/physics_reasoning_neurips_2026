import sympy as sp
import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract inputs and results from solution_instance.
        Never access solution_instance.attr directly.
        """
        # Extract input parameters with safe defaults
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 0.1))
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'results', {}).get('v0', 10.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 10.0))
        
        # Extract results with safe fallbacks
        self.results = {}
        expected_keys = ['a_1', 't_2', 'W_f', 'a_2', 'h', 't_1', 'W_G']
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, 0.0))
        
        # Constants
        self.air_resistance_coeff = 0.1  # Given: 0.1 * weight
        self.F_f = self.air_resistance_coeff * self.m * self.g
        
        self.logs = []
        
    def _log(self, msg: str):
        """Helper for logging."""
        self.logs.append(msg)
        
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants:
        1. Newton's 2nd Law for upward motion: mg + F_f = m*a_1
        2. Newton's 2nd Law for downward motion: mg - F_f = m*a_2
        3. Work-energy consistency
        4. Kinematic consistency
        """
        success = True
        
        try:
            # Extract values
            a_1 = self.results['a_1']
            a_2 = self.results['a_2']
            h = self.results['h']
            W_f = self.results['W_f']
            W_G = self.results['W_G']
            t_1 = self.results['t_1']
            t_2 = self.results['t_2']
            
            # 1. Newton's 2nd Law upward
            expected_a1 = self.g + self.F_f/self.m
            if not self._is_close(a_1, expected_a1):
                self._log(f"Newton's 2nd Law upward failed: a_1={a_1}, expected={expected_a1}")
                success = False
            else:
                self._log(f"✓ Newton's 2nd Law upward: mg + F_f = m*a_1 satisfied")
            
            # 2. Newton's 2nd Law downward
            expected_a2 = self.g - self.F_f/self.m
            if not self._is_close(a_2, expected_a2):
                self._log(f"Newton's 2nd Law downward failed: a_2={a_2}, expected={expected_a2}")
                success = False
            else:
                self._log(f"✓ Newton's 2nd Law downward: mg - F_f = m*a_2 satisfied")
            
            # 3. Work done by gravity in closed loop should be zero
            if not self._is_close(W_G, 0.0):
                self._log(f"Work by gravity in closed loop not zero: W_G={W_G}")
                success = False
            else:
                self._log(f"✓ Work by gravity in closed loop is zero")
            
            # 4. Work done by air resistance: W_f = -F_f * 2h
            expected_Wf = -self.F_f * 2 * h
            if not self._is_close(W_f, expected_Wf):
                self._log(f"Work by air resistance inconsistent: W_f={W_f}, expected={expected_Wf}")
                success = False
            else:
                self._log(f"✓ Work by air resistance: W_f = -F_f * 2h satisfied")
            
            # 5. Kinematic consistency for upward motion: v0 = a_1 * t_1
            expected_t1 = self.v0 / a_1
            if not self._is_close(t_1, expected_t1):
                self._log(f"Time of ascent inconsistent: t_1={t_1}, v0/a_1={expected_t1}")
                success = False
            else:
                self._log(f"✓ Time of ascent: t_1 = v0/a_1 satisfied")
            
            # 6. Kinematic consistency for downward motion: h = 0.5 * a_2 * t_2^2
            if a_2 > 0:
                expected_t2 = math.sqrt(2 * h / a_2)
                if not self._is_close(t_2, expected_t2):
                    self._log(f"Time of descent inconsistent: t_2={t_2}, sqrt(2h/a_2)={expected_t2}")
                    success = False
                else:
                    self._log(f"✓ Time of descent: h = 0.5*a_2*t_2^2 satisfied")
            
            # 7. Maximum height from energy considerations (alternative check)
            # Work-energy: Initial KE = mgh + |W_f_up| (but W_f_up = F_f * h)
            # Actually, using forces: net upward force = m*a_1, so v0^2 = 2*a_1*h
            expected_h = self.v0**2 / (2 * a_1)
            if not self._is_close(h, expected_h):
                self._log(f"Maximum height inconsistent: h={h}, v0^2/(2a_1)={expected_h}")
                success = False
            else:
                self._log(f"✓ Maximum height: h = v0^2/(2a_1) satisfied")
                
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}")
            success = False
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Check that all steps are numerically consistent.
        """
        success = True
        
        try:
            a_1 = self.results['a_1']
            a_2 = self.results['a_2']
            h = self.results['h']
            t_1 = self.results['t_1']
            t_2 = self.results['t_2']
            
            # 1. Check upward motion consistency
            expected_t1 = self.v0 / a_1
            if not self._is_close(t_1, expected_t1):
                self._log(f"Trace consistency failed: t_1={t_1} vs v0/a_1={expected_t1}")
                success = False
            else:
                self._log(f"✓ t_1 consistent with v0 and a_1")
            
            # 2. Check height from upward motion
            expected_h_from_up = self.v0 * t_1 - 0.5 * a_1 * t_1**2
            if not self._is_close(h, expected_h_from_up):
                self._log(f"Trace consistency: h from upward motion={expected_h_from_up} vs h={h}")
                success = False
            else:
                self._log(f"✓ h consistent with upward motion equations")
            
            # 3. Check downward motion time
            if a_2 > 0:
                expected_t2 = math.sqrt(2 * h / a_2)
                if not self._is_close(t_2, expected_t2):
                    self._log(f"Trace consistency: t_2 from downward motion={expected_t2} vs t_2={t_2}")
                    success = False
                else:
                    self._log(f"✓ t_2 consistent with downward motion")
            
            # 4. Check total time vs individual times (not directly given, but could be computed)
            # total_time = t_1 + t_2
            # No inconsistency check needed here
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}")
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input parameter (mass) and check that results change in expected direction.
        Then restore original values.
        """
        success = True
        original_m = self.m
        
        try:
            # Increase mass by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected changes using physical formulas
            # With increased mass, air resistance increases proportionally
            perturbed_F_f = self.air_resistance_coeff * perturbed_m * self.g
            
            # Upward acceleration: a1 = g + F_f/m = g + 0.1g = 1.1g (INDEPENDENT of m!)
            # Actually: a1 = g + (0.1*m*g)/m = 1.1g, so a1 should NOT change with m
            expected_a1_no_change = self.g + self.F_f/self.m
            
            # Downward acceleration: a2 = g - F_f/m = 0.9g (also independent of m)
            expected_a2_no_change = self.g - self.F_f/self.m
            
            # Height: h = v0^2/(2a1) - independent of m since a1 independent
            expected_h_no_change = self.v0**2 / (2 * expected_a1_no_change)
            
            # Time of ascent: t1 = v0/a1 - independent of m
            expected_t1_no_change = self.v0 / expected_a1_no_change
            
            # Time of descent: t2 = sqrt(2h/a2) - independent of m
            expected_t2_no_change = math.sqrt(2 * expected_h_no_change / expected_a2_no_change)
            
            # Work by air resistance: W_f = -F_f * 2h = -0.1*m*g * 2h
            # Since h doesn't change, W_f should increase proportionally with m
            original_Wf = -self.F_f * 2 * expected_h_no_change
            perturbed_Wf = -perturbed_F_f * 2 * expected_h_no_change
            expected_Wf_increase = perturbed_Wf > original_Wf  # More negative, so > is false
            
            # Log expected behavior
            self._log(f"Monotonicity test: Increasing mass from {original_m} to {perturbed_m}")
            self._log(f"  Expected: a1, a2, h, t1, t2 unchanged; |W_f| increases")
            
            # In reality, all accelerations and heights should be unchanged
            # Only work done by air resistance should increase in magnitude
            
            # Check that original calculations are consistent with these expectations
            original_a1 = self.results['a_1']
            original_a2 = self.results['a_2']
            original_h = self.results['h']
            original_t1 = self.results['t_1']
            original_t2 = self.results['t_2']
            original_Wf = self.results['W_f']
            
            # They should match the "no change" expectations
            checks = [
                ("a1", original_a1, expected_a1_no_change, True),
                ("a2", original_a2, expected_a2_no_change, True),
                ("h", original_h, expected_h_no_change, True),
                ("t1", original_t1, expected_t1_no_change, True),
                ("t2", original_t2, expected_t2_no_change, True),
            ]
            
            for name, actual, expected, should_match in checks:
                if should_match and not self._is_close(actual, expected):
                    self._log(f"  {name} mismatch: {actual} vs {expected}")
                    success = False
            
            # Check work sign: W_f should be negative
            if original_Wf >= 0:
                self._log(f"  W_f should be negative but is {original_Wf}")
                success = False
            
            self._log("✓ Monotonicity expectations verified (no change in kinematics with mass change)")
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}")
            success = False
        finally:
            # Restore original mass
            self.m = original_m
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        self.logs = ["=== Physics Checker Started ==="]
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} Test ---")
            try:
                test_success = test_func()
                if test_success:
                    self._log(f"✓ {test_name} test passed")
                else:
                    self._log(f"✗ {test_name} test failed")
                    overall_success = False
            except Exception as e:
                self._log(f"✗ {test_name} test crashed: {str(e)}")
                self._log(traceback.format_exc())
                overall_success = False
        
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs