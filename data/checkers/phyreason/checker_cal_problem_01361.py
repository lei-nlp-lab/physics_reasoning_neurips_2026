import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safe attribute extraction using getattr with fallback to results dict.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 
                        getattr(solution_instance, 'results', {}).get('m', 1.0))
        self.g = getattr(solution_instance, 'g', 
                        getattr(solution_instance, 'results', {}).get('g', 9.8))
        self.R = getattr(solution_instance, 'R', 
                        getattr(solution_instance, 'results', {}).get('R', 1.0))
        
        # Extract results with safe defaults
        self.v0 = getattr(solution_instance, 'v0', 
                         getattr(solution_instance, 'results', {}).get('v_0', None))
        self.h = getattr(solution_instance, 'h', 
                        getattr(solution_instance, 'results', {}).get('h', None))
        
        # Intermediate velocity at B (given in problem statement)
        self.vB = 3 * math.sqrt(self.g * self.R)
        
        # Initialize logs
        self.logs = []
        
        # Convert to float if not already
        try:
            self.m = float(self.m)
            self.g = float(self.g)
            self.R = float(self.R)
            if self.v0 is not None:
                self.v0 = float(self.v0)
            if self.h is not None:
                self.h = float(self.h)
            self.vB = float(self.vB)
        except (TypeError, ValueError) as e:
            self.logs.append(f"Type conversion error: {str(e)}")
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy comparison for physics values."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test energy conservation invariants from KB."""
        success = True
        logs = []
        
        if self.v0 is None or self.h is None:
            logs.append("Missing v0 or h in results")
            return False, logs
        
        try:
            # Symbolic energy conservation between A and B (Step 1)
            K_A = 0.5 * self.m * self.v0**2
            U_A = self.m * self.g * 3 * self.R  # From step1 equation: 3mgR
            K_B = 0.5 * self.m * self.vB**2
            U_B = 0  # Reference point at B
            
            E_A = K_A + U_A
            E_B = K_B + U_B
            
            if not self._is_close(E_A, E_B):
                logs.append(f"Energy conservation A→B violated: E_A={E_A:.3f}, E_B={E_B:.3f}")
                success = False
            else:
                logs.append(f"✓ Energy conservation A→B verified: E_A={E_A:.3f} ≈ E_B={E_B:.3f}")
            
            # Symbolic energy conservation between A and highest point (Step 2)
            K_highest = 0  # At maximum height
            U_highest = self.m * self.g * self.h
            U_A_step2 = self.m * self.g * 2 * self.R  # From step2 equation: 2mgR
            
            E_A_step2 = K_A + U_A_step2
            E_highest = K_highest + U_highest
            
            if not self._is_close(E_A_step2, E_highest):
                logs.append(f"Energy conservation A→highest violated: E_A={E_A_step2:.3f}, E_h={E_highest:.3f}")
                success = False
            else:
                logs.append(f"✓ Energy conservation A→highest verified: E_A={E_A_step2:.3f} ≈ E_h={E_highest:.3f}")
            
            # Numerical fallback checks
            v0_expected = math.sqrt(3 * self.g * self.R)
            if not self._is_close(self.v0, v0_expected):
                logs.append(f"v0 mismatch: expected {v0_expected:.3f}, got {self.v0:.3f}")
                success = False
            else:
                logs.append(f"✓ v0 matches theoretical value: {self.v0:.3f} ≈ {v0_expected:.3f}")
                
            h_expected = 3.5 * self.R
            if not self._is_close(self.h, h_expected):
                logs.append(f"h mismatch: expected {h_expected:.3f}, got {self.h:.3f}")
                success = False
            else:
                logs.append(f"✓ h matches theoretical value: {self.h:.3f} ≈ {h_expected:.3f}")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        success = True
        logs = []
        
        if self.v0 is None or self.h is None:
            logs.append("Cannot check trace consistency: missing v0 or h")
            return False, logs
        
        try:
            # Check consistency between step1 and step2
            # From step1: v0^2 = vB^2 - 6gR
            v0_sq_from_step1 = self.vB**2 - 6 * self.g * self.R
            
            # From step2: h = (0.5*v0^2 + 2gR)/g
            h_from_step2 = (0.5 * self.v0**2 + 2 * self.g * self.R) / self.g
            
            if not self._is_close(self.v0**2, v0_sq_from_step1):
                logs.append(f"Trace inconsistency in v0: step1 gives v0²={v0_sq_from_step1:.3f}, actual v0²={self.v0**2:.3f}")
                success = False
            else:
                logs.append(f"✓ v0 consistent with step1 equation")
            
            if not self._is_close(self.h, h_from_step2):
                logs.append(f"Trace inconsistency in h: step2 gives h={h_from_step2:.3f}, actual h={self.h:.3f}")
                success = False
            else:
                logs.append(f"✓ h consistent with step2 equation")
            
            # Verify both equations simultaneously
            # Equation 1: 0.5*m*v0² + 3*m*g*R = 0.5*m*vB²
            lhs1 = 0.5 * self.m * self.v0**2 + 3 * self.m * self.g * self.R
            rhs1 = 0.5 * self.m * self.vB**2
            
            # Equation 2: 0.5*m*v0² + 2*m*g*R = m*g*h
            lhs2 = 0.5 * self.m * self.v0**2 + 2 * self.m * self.g * self.R
            rhs2 = self.m * self.g * self.h
            
            if not (self._is_close(lhs1, rhs1) and self._is_close(lhs2, rhs2)):
                logs.append(f"Simultaneous equation check failed: eq1({lhs1:.3f}≈{rhs1:.3f}), eq2({lhs2:.3f}≈{rhs2:.3f})")
                success = False
            else:
                logs.append("✓ Both step equations satisfied simultaneously")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test monotonic relationships by perturbing R."""
        success = True
        logs = []
        
        try:
            # Store original values
            R_orig = self.R
            g_orig = self.g
            
            # Perturb R by 1%
            self.R *= 1.01
            
            # Recalculate expected values using formulas
            v0_new_expected = math.sqrt(3 * self.g * self.R)
            h_new_expected = 3.5 * self.R
            
            # Restore original R
            self.R = R_orig
            
            # Check monotonicity: both should increase with R
            v0_orig = math.sqrt(3 * g_orig * R_orig)
            h_orig = 3.5 * R_orig
            
            if v0_new_expected <= v0_orig:
                logs.append(f"Monotonicity violation: v0 should increase with R")
                success = False
            else:
                logs.append(f"✓ v0 increases with R: {v0_orig:.3f} → {v0_new_expected:.3f}")
            
            if h_new_expected <= h_orig:
                logs.append(f"Monotonicity violation: h should increase with R")
                success = False
            else:
                logs.append(f"✓ h increases with R: {h_orig:.3f} → {h_new_expected:.3f}")
            
            # Symbolic derivative checks
            R_sym, g_sym = sp.symbols('R_sym g_sym', positive=True)
            v0_sym = sp.sqrt(3 * g_sym * R_sym)
            h_sym = 3.5 * R_sym
            
            # dv0/dR > 0
            dv0_dR = sp.diff(v0_sym, R_sym)
            dv0_dR_val = float(dv0_dR.subs({R_sym: R_orig, g_sym: g_orig}))
            
            # dh/dR > 0
            dh_dR = sp.diff(h_sym, R_sym)
            dh_dR_val = float(dh_dR.subs({R_sym: R_orig, g_sym: g_orig}))
            
            if dv0_dR_val <= 0:
                logs.append(f"Symbolic derivative dv0/dR = {dv0_dR_val:.3f} ≤ 0")
                success = False
            else:
                logs.append(f"✓ Symbolic dv0/dR = {dv0_dR_val:.3f} > 0")
            
            if dh_dR_val <= 0:
                logs.append(f"Symbolic derivative dh/dR = {dh_dR_val:.3f} ≤ 0")
                success = False
            else:
                logs.append(f"✓ Symbolic dh/dR = {dh_dR_val:.3f} > 0")
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        test_functions = [
            ("Energy Conservation Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"✗ {test_name} FAILED")
                else:
                    all_logs.append(f"✓ {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs

# Example usage (commented out):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.results = {'v_0': math.sqrt(3*9.8*1.0), 'h': 3.5}
#             self.m = 1.0
#             self.g = 9.8
#             self.R = 1.0
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)