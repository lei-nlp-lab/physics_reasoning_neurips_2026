import sympy as sp
import math
import copy
import traceback
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely initialize from solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe extraction of input parameters with fallbacks
        self.mu1 = self._safe_get('mu1', 0.0)
        self.mu2 = self._safe_get('mu2', 0.0)
        self.g = self._safe_get('g', 9.81)
        self.l = self._safe_get('l', 1.0)  # table side length
        
        # Safe extraction of intermediate and final results
        self.a_min = self._safe_get('a', None)  # minimum acceleration
        self.x = self._safe_get('x', 0.0)  # tablecloth displacement
        self.x1 = self._safe_get('x1', 0.0)  # disk displacement on cloth
        self.a1 = self._safe_get('a1', 0.0)  # disk accel on cloth
        self.a2 = self._safe_get('a2', 0.0)  # disk decel on table
        self.v1_sq = self._safe_get('v1_sq', 0.0)  # v1^2
        self.x2 = self._safe_get('x2', 0.0)  # disk displacement on table
        
        # Symbolic variables for invariant checking
        self.mu1_sym = sp.symbols('mu1', positive=True)
        self.mu2_sym = sp.symbols('mu2', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        self.l_sym = sp.symbols('l', positive=True)
        
    def _safe_get(self, key: str, default: Any) -> Any:
        """Safely retrieve attribute from solution instance or results dict."""
        # Try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return val
        
        # Try results dict
        val = self.results.get(key, default)
        
        # Special handling for LaTeX keys
        if val is None and key in ['a', 'a1', 'a2', 'x', 'x1', 'x2', 'v1_sq']:
            latex_key = {
                'a': 'a', 'a1': 'a₁', 'a2': 'a₂',
                'x': 'x', 'x1': 'x₁', 'x2': 'x₂',
                'v1_sq': 'v₁²'
            }.get(key, key)
            val = self.results.get(latex_key, default)
        
        return float(val) if isinstance(val, (int, float, sp.Number)) else val
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify physical invariants:
        1. Newton's 2nd Law for a1 and a2
        2. Energy/work-energy theorem consistency
        3. Geometric constraints
        """
        logs = []
        success = True
        
        try:
            # 1. Check Newton's 2nd Law: a1 = μ1*g, a2 = μ2*g
            expected_a1 = self.mu1 * self.g
            expected_a2 = self.mu2 * self.g
            
            if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: a1 = {self.a1:.3f}, expected μ1*g = {expected_a1:.3f}")
                success = False
            else:
                logs.append(f"OK: a1 = μ1*g = {expected_a1:.3f}")
                
            if not math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: a2 = {self.a2:.3f}, expected μ2*g = {expected_a2:.3f}")
                success = False
            else:
                logs.append(f"OK: a2 = μ2*g = {expected_a2:.3f}")
            
            # 2. Check kinematic consistency from step 3: v1² = 2*a1*x1 = 2*a2*x2
            v1_sq_from_a1 = 2 * self.a1 * self.x1
            v1_sq_from_a2 = 2 * self.a2 * self.x2
            
            if not math.isclose(v1_sq_from_a1, v1_sq_from_a2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: 2*a1*x1 = {v1_sq_from_a1:.3f}, 2*a2*x2 = {v1_sq_from_a2:.3f}")
                success = False
            else:
                logs.append(f"OK: Kinematic consistency: 2*a1*x1 = 2*a2*x2 = {v1_sq_from_a1:.3f}")
            
            # Also check against stored v1_sq if available
            if self.v1_sq > 0:
                if not (math.isclose(self.v1_sq, v1_sq_from_a1, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.v1_sq, v1_sq_from_a2, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"FAIL: stored v1² = {self.v1_sq:.3f} doesn't match calculated")
                    success = False
            
            # 3. Check geometric constraint from step 7: x = l/2 + x1
            expected_x = self.l/2 + self.x1
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x = {self.x:.3f}, expected l/2 + x1 = {expected_x:.3f}")
                success = False
            else:
                logs.append(f"OK: Geometric constraint x = l/2 + x1 satisfied")
            
            # 4. Check no-fall condition from step 4: x2 ≤ l/2 - x1
            max_x2 = self.l/2 - self.x1
            if self.x2 > max_x2 + 1e-2:  # Slight tolerance
                logs.append(f"FAIL: x2 = {self.x2:.3f} > l/2 - x1 = {max_x2:.3f}")
                success = False
            else:
                logs.append(f"OK: No-fall condition x2 ≤ l/2 - x1 satisfied")
            
            # 5. Symbolic check of Newton's 2nd Law
            try:
                m = sp.symbols('m', positive=True)
                F1 = self.mu1_sym * m * self.g_sym
                a1_sym = sp.simplify(F1 / m)
                if a1_sym != self.mu1_sym * self.g_sym:
                    logs.append("WARN: Symbolic Newton check for a1 failed")
                
                F2 = self.mu2_sym * m * self.g_sym
                a2_sym = sp.simplify(F2 / m)
                if a2_sym != self.mu2_sym * self.g_sym:
                    logs.append("WARN: Symbolic Newton check for a2 failed")
            except:
                logs.append("WARN: Symbolic check skipped due to error")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Check consistency between intermediate steps and final answer.
        """
        logs = []
        success = True
        
        try:
            # 1. Check that the derived minimum acceleration formula matches
            expected_a_min = (2 + self.mu1/self.mu2) * self.mu1 * self.g
            
            if self.a_min is not None:
                if not math.isclose(self.a_min, expected_a_min, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a_min = {self.a_min:.3f}, expected formula gives {expected_a_min:.3f}")
                    success = False
                else:
                    logs.append(f"OK: a_min matches formula: {expected_a_min:.3f}")
            
            # 2. Check time consistency from steps 5-6: x = ½a t², x1 = ½a1 t²
            # Derive t from both equations
            if self.a_min is not None and self.a_min > 0 and self.a1 > 0:
                t_from_x = math.sqrt(2 * self.x / self.a_min) if self.x > 0 else 0
                t_from_x1 = math.sqrt(2 * self.x1 / self.a1) if self.x1 > 0 else 0
                
                if t_from_x > 0 and t_from_x1 > 0:
                    if not math.isclose(t_from_x, t_from_x1, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: Time inconsistency: t(x)={t_from_x:.3f}, t(x1)={t_from_x1:.3f}")
                        success = False
                    else:
                        logs.append(f"OK: Time consistency: t = {t_from_x:.3f}s")
            
            # 3. Check that using the derived a_min gives equality in no-fall condition
            # Solve for x2 from kinematic equations
            if self.a_min is not None and self.a_min > 0:
                # From geometric constraint: x = l/2 + x1
                # From x = ½a_min t² and x1 = ½a1 t², we get x/x1 = a_min/a1
                # So x1 = (a1/a_min) * x
                # But also x = l/2 + x1
                # So x = l/2 + (a1/a_min)*x
                # => x*(1 - a1/a_min) = l/2
                if abs(1 - self.a1/self.a_min) > 1e-9:
                    calc_x = (self.l/2) / (1 - self.a1/self.a_min)
                    calc_x1 = self.a1/self.a_min * calc_x
                    
                    # Then v1² = 2*a1*x1
                    calc_v1_sq = 2 * self.a1 * calc_x1
                    # And x2 = v1²/(2*a2)
                    calc_x2 = calc_v1_sq/(2*self.a2)
                    
                    # Check x1 + x2 = l/2 (for minimum a)
                    total = calc_x1 + calc_x2
                    if not math.isclose(total, self.l/2, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: At minimum a, x1+x2={total:.3f} ≠ l/2={self.l/2:.3f}")
                        success = False
                    else:
                        logs.append(f"OK: Minimum a gives x1+x2 = l/2 exactly")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb μ1 and check that minimum acceleration increases appropriately.
        """
        logs = []
        success = True
        
        try:
            # Save original value
            original_mu1 = self.mu1
            
            # Perturb μ1 by +10%
            perturbed_mu1 = original_mu1 * 1.1
            
            # Calculate original and perturbed minimum accelerations using formula
            a_min_original = (2 + original_mu1/self.mu2) * original_mu1 * self.g
            a_min_perturbed = (2 + perturbed_mu1/self.mu2) * perturbed_mu1 * self.g
            
            # Check monotonic increase
            if a_min_perturbed <= a_min_original:
                logs.append(f"FAIL: Monotonicity violated: a_min({perturbed_mu1:.3f}) = {a_min_perturbed:.3f} ≤ a_min({original_mu1:.3f}) = {a_min_original:.3f}")
                success = False
            else:
                logs.append(f"OK: Monotonicity: a_min increases with μ1")
            
            # Additional check: derivative wrt μ1 should be positive
            # da/dμ1 = (2 + 2*μ1/μ2) * g > 0
            derivative = (2 + 2*original_mu1/self.mu2) * self.g
            if derivative <= 0:
                logs.append(f"WARN: Derivative wrt μ1 is non-positive: {derivative:.3f}")
            
            # Restore original value
            self.mu1 = original_mu1
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Try to restore
            if 'original_mu1' in locals():
                self.mu1 = original_mu1
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and consolidated logs.
        """
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n1. Invariant Tests:")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n2. Trace Consistency Tests:")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n3. Monotonicity Tests:")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        # Summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs

# Optional: Example usage pattern
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'mu1': 0.3,
                'mu2': 0.2,
                'g': 9.81,
                'l': 2.0,
                'a': 8.829,  # (2 + 0.3/0.2)*0.3*9.81 = 8.829
                'a₁': 2.943,  # 0.3*9.81
                'a₂': 1.962,  # 0.2*9.81
                'x': 1.2,
                'x₁': 0.2,
                'x₂': 0.8,
                'v₁²': 1.1772  # 2*2.943*0.2 = 1.1772
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)