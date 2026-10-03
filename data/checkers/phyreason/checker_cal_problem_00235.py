import math
import traceback
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input constants from results (they should be stored there)
        # Use defaults that match the problem's typical values to avoid division by zero
        self.v0 = self.results.get('v0', 1.0e6)
        self.m = self.results.get('m', 9.109e-31)
        self.e = self.results.get('e', 1.602e-19)
        self.L = self.results.get('L', 0.1)
        self.r = self.results.get('r', 0.01)
        self.E = self.results.get('E', 1.0e3)
        
        # Extract computed results with safe defaults
        self.B_min = self.results.get('B_min', 0.0)
        self.tantheta = self.results.get('tantheta', 0.0)
        self.y_m = self.results.get('y_m', 0.0)
        self.R_max = self.results.get('R_max', 0.0)
        self.T = self.results.get('T', 0.0)
        self.t = self.results.get('t', 0.0)
        self.a = self.results.get('a', 0.0)
        self.v_ym = self.results.get('v_ym', 0.0)
        self.B = self.results.get('B', 0.0)
        self.R = self.results.get('R', 0.0)
        
        # Initialize logs
        self.logs = []
        
    def log(self, message: str, status: str = "INFO"):
        """Add a log message with status."""
        self.logs.append(f"[{status}] {message}")
        
    def fuzzy_eq(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy equality check with relative and absolute tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Verify physical invariants based on KB theorems."""
        try:
            # 1. Check magnetic force provides centripetal force
            if self.fuzzy_eq(self.B_min, 0.0) or self.fuzzy_eq(self.v_ym, 0.0):
                self.log("Skipping magnetic force check due to zero values", "WARNING")
            else:
                # R = m*v_y/(eB)
                expected_R = self.m * self.v_ym / (abs(self.e) * self.B_min)
                if not self.fuzzy_eq(self.R_max, expected_R):
                    self.log(f"Centripetal force invariant failed: R_max={self.R_max}, expected={expected_R}", "ERROR")
                    return False
                self.log("Magnetic force centripetal invariant passed", "INFO")
            
            # 2. Check time consistency: L = v0 * t
            if self.v0 != 0:
                expected_t = self.L / self.v0
                if not self.fuzzy_eq(self.t, expected_t):
                    self.log(f"Time invariant failed: t={self.t}, expected L/v0={expected_t}", "ERROR")
                    return False
                self.log("Uniform motion time invariant passed", "INFO")
            
            # 3. Check period formula: T = 2πm/(eB)
            if not self.fuzzy_eq(self.B_min, 0.0):
                expected_T = 2 * math.pi * self.m / (abs(self.e) * self.B_min)
                if not self.fuzzy_eq(self.T, expected_T):
                    self.log(f"Period invariant failed: T={self.T}, expected 2πm/(eB)={expected_T}", "ERROR")
                    return False
                self.log("Circular motion period invariant passed", "INFO")
            
            # 4. Check electric field acceleration: a = Ee/m
            if self.m != 0:
                expected_a = self.E * abs(self.e) / self.m
                if not self.fuzzy_eq(self.a, expected_a):
                    self.log(f"Acceleration invariant failed: a={self.a}, expected Ee/m={expected_a}", "ERROR")
                    return False
                self.log("Electric field acceleration invariant passed", "INFO")
            
            # 5. Check energy conservation in electric field
            if not self.fuzzy_eq(self.v_ym, 0.0) and self.m != 0:
                ke_initial = 0.5 * self.m * self.v_ym**2
                work_done = -abs(self.e) * self.E * self.y_m
                if not self.fuzzy_eq(ke_initial, -work_done):
                    self.log(f"Energy conservation failed: KE_initial={ke_initial}, |work|={-work_done}", "ERROR")
                    return False
                self.log("Energy conservation in electric field passed", "INFO")
            
            return True
            
        except Exception as e:
            self.log(f"Invariants test failed with exception: {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            return False
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check B_min derivation: B_min = 2πmv0/(eL)
            if self.L != 0 and abs(self.e) != 0:
                expected_B_min = 2 * math.pi * self.m * self.v0 / (abs(self.e) * self.L)
                if not self.fuzzy_eq(self.B_min, expected_B_min):
                    self.log(f"B_min trace consistency failed: {self.B_min} vs {expected_B_min}", "ERROR")
                    return False
                self.log("B_min trace consistency passed", "INFO")
            
            # 2. Check v_ym derivation: v_ym = 2πrv0/L
            if self.L != 0:
                expected_v_ym = 2 * math.pi * self.r * self.v0 / self.L
                if not self.fuzzy_eq(self.v_ym, expected_v_ym):
                    self.log(f"v_ym trace consistency failed: {self.v_ym} vs {expected_v_ym}", "ERROR")
                    return False
                self.log("v_ym trace consistency passed", "INFO")
            
            # 3. Check tanθ = 2πr/L
            if self.L != 0:
                expected_tantheta = 2 * math.pi * self.r / self.L
                if not self.fuzzy_eq(self.tantheta, expected_tantheta):
                    self.log(f"tanθ trace consistency failed: {self.tantheta} vs {expected_tantheta}", "ERROR")
                    return False
                self.log("tanθ trace consistency passed", "INFO")
            
            # 4. Check y_m derivation: y_m = v_ym^2/(2a) with a = Ee/m
            if self.m != 0 and abs(self.e) != 0:
                expected_y_m = (self.v_ym**2) / (2 * (self.E * abs(self.e) / self.m))
                if not self.fuzzy_eq(self.y_m, expected_y_m):
                    self.log(f"y_m trace consistency failed: {self.y_m} vs {expected_y_m}", "ERROR")
                    return False
                self.log("y_m trace consistency passed", "INFO")
            
            # 5. Final formula: y_m = (2π²r²v0²m)/(EeL²)
            if self.L != 0 and abs(self.e) != 0:
                expected_y_m_final = (2 * (math.pi**2) * (self.r**2) * (self.v0**2) * self.m) / (self.E * abs(self.e) * (self.L**2))
                if not self.fuzzy_eq(self.y_m, expected_y_m_final):
                    self.log(f"Final y_m formula failed: {self.y_m} vs {expected_y_m_final}", "ERROR")
                    return False
                self.log("Final y_m formula consistency passed", "INFO")
            
            return True
            
        except Exception as e:
            self.log(f"Trace consistency test failed with exception: {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb inputs and check expected trends."""
        # Store original values
        original_v0 = self.v0
        original_m = self.m
        original_L = self.L
        original_r = self.r
        original_E = self.E
        
        try:
            # Test 1: Increase v0 → B_min should increase
            self.v0 *= 1.1
            if self.L != 0 and abs(self.e) != 0:
                expected_B_min_new = 2 * math.pi * self.m * self.v0 / (abs(self.e) * self.L)
                old_B_min = self.B_min
                if expected_B_min_new > old_B_min * 0.99:
                    self.log("B_min increases with v0 (monotonicity passed)", "INFO")
                else:
                    self.log("B_min monotonicity with v0 failed", "ERROR")
                    return False
            
            # Restore v0
            self.v0 = original_v0
            
            # Test 2: Increase m → B_min should increase
            self.m *= 1.1
            if self.L != 0 and abs(self.e) != 0:
                expected_B_min_new = 2 * math.pi * self.m * self.v0 / (abs(self.e) * self.L)
                old_B_min = self.B_min
                if expected_B_min_new > old_B_min * 0.99:
                    self.log("B_min increases with m (monotonicity passed)", "INFO")
                else:
                    self.log("B_min monotonicity with m failed", "ERROR")
                    return False
            
            # Restore m
            self.m = original_m
            
            # Test 3: Increase L → B_min should decrease
            self.L *= 1.1
            if self.L != 0 and abs(self.e) != 0:
                expected_B_min_new = 2 * math.pi * self.m * self.v0 / (abs(self.e) * self.L)
                old_B_min = self.B_min
                if expected_B_min_new < old_B_min * 1.01:
                    self.log("B_min decreases with L (monotonicity passed)", "INFO")
                else:
                    self.log("B_min monotonicity with L failed", "ERROR")
                    return False
            
            # Restore L
            self.L = original_L
            
            # Test 4: Increase r → tanθ should increase
            self.r *= 1.1
            if self.L != 0:
                expected_tantheta_new = 2 * math.pi * self.r / self.L
                old_tantheta = self.tantheta
                if expected_tantheta_new > old_tantheta * 0.99:
                    self.log("tanθ increases with r (monotonicity passed)", "INFO")
                else:
                    self.log("tanθ monotonicity with r failed", "ERROR")
                    return False
            
            # Restore r
            self.r = original_r
            
            # Test 5: Increase E → y_m should decrease
            self.E *= 1.1
            if self.L != 0 and abs(self.e) != 0:
                expected_y_m_new = (2 * (math.pi**2) * (self.r**2) * (self.v0**2) * self.m) / (self.E * abs(self.e) * (self.L**2))
                old_y_m = self.y_m
                if expected_y_m_new < old_y_m * 1.01:
                    self.log("y_m decreases with E (monotonicity passed)", "INFO")
                else:
                    self.log("y_m monotonicity with E failed", "ERROR")
                    return False
            
            # Restore E
            self.E = original_E
            
            return True
            
        except Exception as e:
            self.log(f"Monotonicity test failed with exception: {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            # Restore original values
            self.v0 = original_v0
            self.m = original_m
            self.L = original_L
            self.r = original_r
            self.E = original_E
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        success = test1 and test2 and test3
        
        if success:
            self.log("All physics checks passed!", "SUCCESS")
        else:
            self.log("Some physics checks failed", "FAILURE")
        
        return success, self.logs