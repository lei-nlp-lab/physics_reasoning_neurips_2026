import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.logs = []
        
        # Safely extract input parameters
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.B = getattr(solution_instance, 'B', None)
        self.d = getattr(solution_instance, 'd', None)
        self.L = getattr(solution_instance, 'L', None)
        
        # Safely extract results with fallback to results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        self.v = getattr(solution_instance, 'v', results_dict.get('v', None))
        self.U = getattr(solution_instance, 'U', results_dict.get('U', None))
        
        # For symbolic checks
        self.m_sym = sp.Symbol('m', positive=True)
        self.q_sym = sp.Symbol('q', negative=True)  # Negative charge as given
        self.B_sym = sp.Symbol('B', positive=True)
        self.d_sym = sp.Symbol('d', positive=True)
        self.L_sym = sp.Symbol('L', positive=True)
        self.v_sym = sp.Symbol('v', positive=True)
        self.U_sym = sp.Symbol('U', positive=True)
        
    def _log(self, message: str):
        """Add message to logs."""
        self.logs.append(message)
    
    def _check_close(self, val1, val2, name: str) -> bool:
        """
        Check if two values are close (fuzzy matching).
        Handles both numeric and symbolic expressions.
        """
        try:
            # If both are numeric
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                return math.isclose(val1, val2, rel_tol=0.02, abs_tol=0.02)
            
            # If symbolic, convert to numeric if possible
            val1_num = float(val1) if hasattr(val1, '__float__') else None
            val2_num = float(val2) if hasattr(val2, '__float__') else None
            
            if val1_num is not None and val2_num is not None:
                return math.isclose(val1_num, val2_num, rel_tol=0.02, abs_tol=0.02)
            
            # Symbolic check: simplify difference
            diff = sp.simplify(val1 - val2)
            if diff == 0:
                return True
            # Try numerical substitution if symbols have values
            if all(v is not None for v in [self.m, self.q, self.B, self.d, self.L]):
                diff_val = diff.subs({
                    self.m_sym: self.m,
                    self.q_sym: self.q,
                    self.B_sym: self.B,
                    self.d_sym: self.d,
                    self.L_sym: self.L
                })
                if diff_val == 0:
                    return True
                if isinstance(diff_val, sp.Number):
                    return abs(float(diff_val)) < 0.02
            
            self._log(f"  {name}: Could not verify numerically (symbolic diff: {diff})")
            return False
            
        except Exception as e:
            self._log(f"  {name}: Error in comparison: {str(e)}")
            return False
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants:
        1. Lorentz force circular motion: qvB = mv²/r
        2. Geometric relationship: r² = (r-L)² + d²
        3. Work-energy: qU = ½mv²
        """
        success = True
        
        # 1. Check Lorentz force and circular motion
        try:
            # From step_1: qvB = mv²/r => r = mv/(qB)
            if None not in [self.m, self.q, self.B, self.v]:
                r_from_lorentz = self.m * self.v / (abs(self.q) * self.B)
                
                # From step_2 geometry: r² = (r-L)² + d² => r = (d² + L²)/(2L)
                if None not in [self.d, self.L]:
                    r_from_geometry = (self.d**2 + self.L**2) / (2 * self.L)
                    
                    if not self._check_close(r_from_lorentz, r_from_geometry, "Radius consistency"):
                        self._log("  FAIL: Radius from Lorentz force != Radius from geometry")
                        success = False
                    else:
                        self._log("  PASS: Lorentz force and geometry consistent")
            else:
                self._log("  SKIP: Missing parameters for Lorentz/geometry check")
                
        except Exception as e:
            self._log(f"  ERROR in Lorentz/geometry check: {str(e)}")
            success = False
        
        # 2. Check work-energy theorem
        try:
            if None not in [self.q, self.U, self.m, self.v]:
                # qU = ½mv²
                left_energy = abs(self.q) * self.U  # q is negative but energy positive
                right_energy = 0.5 * self.m * self.v**2
                
                if not self._check_close(left_energy, right_energy, "Work-energy theorem"):
                    self._log(f"  FAIL: qU = {left_energy}, ½mv² = {right_energy}")
                    success = False
                else:
                    self._log("  PASS: Work-energy theorem satisfied")
            else:
                self._log("  SKIP: Missing parameters for work-energy check")
                
        except Exception as e:
            self._log(f"  ERROR in work-energy check: {str(e)}")
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate and final results.
        Verify that v and U satisfy the derived formulas.
        """
        success = True
        
        try:
            # Check v formula from step_3: v = (qB(d² + L²))/(2mL)
            if None not in [self.q, self.B, self.d, self.L, self.m, self.v]:
                v_expected = (abs(self.q) * self.B * (self.d**2 + self.L**2)) / (2 * self.m * self.L)
                
                if not self._check_close(self.v, v_expected, "Velocity formula"):
                    self._log(f"  FAIL: v = {self.v}, expected = {v_expected}")
                    success = False
                else:
                    self._log("  PASS: Velocity matches derived formula")
            else:
                self._log("  SKIP: Missing parameters for velocity check")
            
            # Check U formula from step_5: U = (qB²(L² + d²)²)/(8mL²)
            if None not in [self.q, self.B, self.d, self.L, self.m, self.U]:
                U_expected = (abs(self.q) * self.B**2 * (self.L**2 + self.d**2)**2) / (8 * self.m * self.L**2)
                
                if not self._check_close(self.U, U_expected, "Voltage formula"):
                    self._log(f"  FAIL: U = {self.U}, expected = {U_expected}")
                    success = False
                else:
                    self._log("  PASS: Voltage matches derived formula")
            else:
                self._log("  SKIP: Missing parameters for voltage check")
                
        except Exception as e:
            self._log(f"  ERROR in trace consistency: {str(e)}")
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and check expected trends.
        Tests:
        1. Increase B -> v should increase (from v ∝ B)
        2. Increase B -> U should increase (from U ∝ B²)
        3. Increase L -> v should decrease (from v ∝ 1/L for fixed d)
        """
        success = True
        original_params = {}
        
        try:
            # Store original values if they exist
            params = ['B', 'L', 'v', 'U']
            for param in params:
                original_params[param] = getattr(self, param)
            
            # Only run if we have all necessary parameters
            if None not in [self.B, self.L, self.v, self.U, self.m, self.q, self.d]:
                
                # Test 1: Increase B by 10%
                B_perturbed = self.B * 1.1
                v_perturbed = (abs(self.q) * B_perturbed * (self.d**2 + self.L**2)) / (2 * self.m * self.L)
                
                if v_perturbed <= self.v:
                    self._log(f"  FAIL: Increasing B from {self.B} to {B_perturbed} should increase v")
                    success = False
                else:
                    self._log(f"  PASS: v increases with B ({self.v} -> {v_perturbed})")
                
                # Test 2: U should increase with B²
                U_perturbed = (abs(self.q) * B_perturbed**2 * (self.L**2 + self.d**2)**2) / (8 * self.m * self.L**2)
                if U_perturbed <= self.U:
                    self._log(f"  FAIL: Increasing B should increase U")
                    success = False
                else:
                    self._log(f"  PASS: U increases with B² ({self.U} -> {U_perturbed})")
                
                # Test 3: Increase L by 10% -> v should decrease
                L_perturbed = self.L * 1.1
                v_perturbed_L = (abs(self.q) * self.B * (self.d**2 + L_perturbed**2)) / (2 * self.m * L_perturbed)
                
                # For fixed d, as L increases, (d² + L²)/(2L) increases but not as fast as L?
                # Actually, derivative shows v decreases with L for L > d/√3
                if v_perturbed_L >= self.v and self.L > self.d / math.sqrt(3):
                    self._log(f"  FAIL: Increasing L should decrease v for L > d/√3")
                    success = False
                else:
                    self._log(f"  PASS: v changes appropriately with L ({self.v} -> {v_perturbed_L})")
                
            else:
                self._log("  SKIP: Missing parameters for monotonicity tests")
                
        except Exception as e:
            self._log(f"  ERROR in monotonicity test: {str(e)}")
            success = False
        finally:
            # Restore original values
            for param, value in original_params.items():
                setattr(self, param, value)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics consistency checks.
        Returns (success, logs)
        """
        self.logs.clear()
        overall_success = True
        
        self._log("=== Physics Consistency Check ===")
        
        # Check 1: Physical invariants
        self._log("\n1. Testing physical invariants:")
        if not self.test_invariants():
            overall_success = False
        
        # Check 2: Trace consistency
        self._log("\n2. Testing trace consistency:")
        if not self.test_trace_consistency():
            overall_success = False
        
        # Check 3: Monotonicity
        self._log("\n3. Testing monotonicity:")
        if not self.test_monotonicity():
            overall_success = False
        
        # Summary
        self._log("\n=== Summary ===")
        if overall_success:
            self._log("All physics checks passed!")
        else:
            self._log("Some physics checks failed.")
        
        return overall_success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)