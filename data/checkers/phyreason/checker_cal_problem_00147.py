import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract inputs from results dictionary (not from solution_instance attributes)
        self.m = results_dict.get('m', None)
        self.q = results_dict.get('q', None)
        self.v0 = results_dict.get('v0', None)
        self.d = results_dict.get('d', None)
        
        # Left region (triangle) results
        self.r = results_dict.get('r', None)
        self.B = results_dict.get('B', None)
        self.t1 = results_dict.get('t_1', None)
        self.CP = results_dict.get('CP', None)
        
        # Straight-line segment results
        self.DP = results_dict.get('DP', None)
        self.t2 = results_dict.get('t_2', None)
        
        # Right region results
        self.r_prime = results_dict.get('r_prime', None)
        self.B_prime = results_dict.get('B_prime', results_dict.get('B_2', None))
        self.T_prime = results_dict.get('T_prime', None)
        self.t3 = results_dict.get('t_3', None)
        
        # Alternative right region scenarios
        self.R = results_dict.get('R', None)
        self.B2 = results_dict.get('B2', results_dict.get('B_2', None))
        self.R_prime = results_dict.get('R_prime', None)
        self.B3 = results_dict.get('B_3', None)
        
        # Total time
        self.t_total = results_dict.get('t', None)
        
        # Store logs
        self.logs = []
        
    def log(self, message, success=None):
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(message)
    
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        try:
            # 1. Lorentz force = centripetal force for left region
            if all(v is not None for v in [self.m, self.q, self.v0, self.B, self.r]):
                expected_r = self.m * self.v0 / (abs(self.q) * self.B) if self.B != 0 else 0
                if not self.fuzzy_equal(self.r, expected_r):
                    self.log(f"Lorentz force invariant fails: r={self.r}, expected {expected_r}", False)
                    return False
                self.log("Lorentz force invariant holds (left region)", True)
            
            # 2. Period calculation for left region
            if all(v is not None for v in [self.r, self.v0, self.t1]):
                T_left = 2 * math.pi * self.r / self.v0
                expected_t1 = T_left * (60/360)  # 60° arc
                if not self.fuzzy_equal(self.t1, expected_t1):
                    self.log(f"Period invariant fails: t1={self.t1}, expected {expected_t1}", False)
                    return False
                self.log("Period invariant holds (left region)", True)
            
            # 3. Right region Lorentz force (if r' and B' exist)
            if all(v is not None for v in [self.r_prime, self.B_prime, self.m, self.q, self.v0]):
                expected_r_prime = self.m * self.v0 / (abs(self.q) * self.B_prime) if self.B_prime != 0 else 0
                if not self.fuzzy_equal(self.r_prime, expected_r_prime):
                    self.log(f"Lorentz force invariant fails (right): r'={self.r_prime}, expected {expected_r_prime}", False)
                    return False
                self.log("Lorentz force invariant holds (right region)", True)
            
            # 4. Right region period and arc time
            if all(v is not None for v in [self.r_prime, self.v0, self.t3]):
                T_right = 2 * math.pi * self.r_prime / self.v0
                # The ion traverses 300° (5/6 of full circle) in right region
                expected_t3 = (5/6) * T_right
                if not self.fuzzy_equal(self.t3, expected_t3):
                    self.log(f"Arc time invariant fails: t3={self.t3}, expected {expected_t3}", False)
                    return False
                self.log("Arc time invariant holds (right region)", True)
            
            # 5. Geometry invariants
            if self.r is not None and self.d is not None:
                if not self.fuzzy_equal(self.r, self.d/2):
                    self.log(f"Geometry invariant fails: r={self.r}, expected d/2={self.d/2}", False)
                    return False
                self.log("Geometry invariant holds (r = d/2)", True)
            
            if self.r_prime is not None and self.d is not None:
                if not self.fuzzy_equal(self.r_prime, 2*self.d):
                    self.log(f"Geometry invariant fails: r'={self.r_prime}, expected 2d={2*self.d}", False)
                    return False
                self.log("Geometry invariant holds (r' = 2d)", True)
            
            if self.CP is not None and self.d is not None:
                if not self.fuzzy_equal(self.CP, self.d):
                    self.log(f"Geometry invariant fails: CP={self.CP}, expected d={self.d}", False)
                    return False
                self.log("Geometry invariant holds (CP = d)", True)
            
            if self.DP is not None and self.CP is not None:
                expected_DP = self.CP * math.sin(math.radians(60))
                if not self.fuzzy_equal(self.DP, expected_DP):
                    self.log(f"Trigonometry invariant fails: DP={self.DP}, expected {expected_DP}", False)
                    return False
                self.log("Trigonometry invariant holds (DP = CP*sin60°)", True)
            
            # 6. Alternative scenario invariants
            if self.R is not None and self.d is not None:
                # From geometry: R + R*cos60° = d + (d/2)*cos60°
                lhs = self.R + self.R * math.cos(math.radians(60))
                rhs = self.d + (self.d/2) * math.cos(math.radians(60))
                if not self.fuzzy_equal(lhs, rhs):
                    self.log(f"Alternative geometry fails: {lhs} ≠ {rhs}", False)
                    return False
                self.log("Alternative geometry invariant holds", True)
            
            if self.R_prime is not None and self.d is not None:
                # From geometry: R' + R'*cos60° = d - (d/2)*cos60°
                lhs = self.R_prime + self.R_prime * math.cos(math.radians(60))
                rhs = self.d - (self.d/2) * math.cos(math.radians(60))
                if not self.fuzzy_equal(lhs, rhs):
                    self.log(f"Alternative geometry fails: {lhs} ≠ {rhs}", False)
                    return False
                self.log("Alternative geometry invariant holds", True)
            
            return True
        except Exception as e:
            self.log(f"Exception in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            success = True
            
            # 1. Check B formula
            if all(v is not None for v in [self.B, self.m, self.q, self.v0, self.d]):
                expected_B = 2 * self.m * self.v0 / (abs(self.q) * self.d)
                if not self.fuzzy_equal(self.B, expected_B):
                    self.log(f"B inconsistent: {self.B} ≠ {expected_B}", False)
                    success = False
                else:
                    self.log("B formula consistent", True)
            
            # 2. Check t1 formula
            if all(v is not None for v in [self.t1, self.d, self.v0]):
                expected_t1 = math.pi * self.d / (6 * self.v0)
                if not self.fuzzy_equal(self.t1, expected_t1):
                    self.log(f"t1 inconsistent: {self.t1} ≠ {expected_t1}", False)
                    success = False
                else:
                    self.log("t1 formula consistent", True)
            
            # 3. Check t2 formula
            if all(v is not None for v in [self.t2, self.d, self.v0]):
                expected_t2 = math.sqrt(3) * self.d / (2 * self.v0)
                if not self.fuzzy_equal(self.t2, expected_t2):
                    self.log(f"t2 inconsistent: {self.t2} ≠ {expected_t2}", False)
                    success = False
                else:
                    self.log("t2 formula consistent", True)
            
            # 4. Check t3 formula
            if all(v is not None for v in [self.t3, self.d, self.v0]):
                expected_t3 = 10 * math.pi * self.d / (3 * self.v0)
                if not self.fuzzy_equal(self.t3, expected_t3):
                    self.log(f"t3 inconsistent: {self.t3} ≠ {expected_t3}", False)
                    success = False
                else:
                    self.log("t3 formula consistent", True)
            
            # 5. Check total time formula
            if all(v is not None for v in [self.t_total, self.d, self.v0]):
                expected_t = (11*math.pi + 3*math.sqrt(3)) * self.d / (3 * self.v0)
                if not self.fuzzy_equal(self.t_total, expected_t):
                    self.log(f"Total time inconsistent: {self.t_total} ≠ {expected_t}", False)
                    success = False
                else:
                    self.log("Total time formula consistent", True)
            
            # 6. Check alternative B2 formula
            if all(v is not None for v in [self.B2, self.m, self.q, self.v0, self.d]):
                expected_B2 = 6 * self.m * self.v0 / (5 * abs(self.q) * self.d)
                if not self.fuzzy_equal(self.B2, expected_B2):
                    self.log(f"B2 inconsistent: {self.B2} ≠ {expected_B2}", False)
                    success = False
                else:
                    self.log("B2 formula consistent", True)
            
            # 7. Check alternative B3 formula
            if all(v is not None for v in [self.B3, self.m, self.q, self.v0, self.d]):
                expected_B3 = 2 * self.m * self.v0 / (abs(self.q) * self.d)
                if not self.fuzzy_equal(self.B3, expected_B3):
                    self.log(f"B3 inconsistent: {self.B3} ≠ {expected_B3}", False)
                    success = False
                else:
                    self.log("B3 formula consistent", True)
            
            # 8. Verify total time composition
            if all(v is not None for v in [self.t_total, self.t1, self.t2, self.t3]):
                expected_total = 2*(self.t1 + self.t2) + self.t3
                if not self.fuzzy_equal(self.t_total, expected_total):
                    self.log(f"Time composition fails: {self.t_total} ≠ 2*(t1+t2)+t3={expected_total}", False)
                    success = False
                else:
                    self.log("Time composition consistent", True)
            
            return success
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check expected trends."""
        try:
            if None in [self.m, self.q, self.v0, self.d, self.B, self.t_total]:
                self.log("Missing parameters for monotonicity test", False)
                return False
            
            # Store original values
            m_original = self.m
            
            # Test 1: Increase mass by 1%
            self.m = m_original * 1.01
            
            # Recalculate using formulas (not using solution instance)
            B_new = 2 * self.m * self.v0 / (abs(self.q) * self.d)
            t_new = (11*math.pi + 3*math.sqrt(3)) * self.d / (3 * self.v0)
            
            # Check trends
            # B should increase proportionally with m
            expected_B_ratio = 1.01
            actual_B_ratio = B_new / self.B if self.B != 0 else 0
            
            # t should remain constant (independent of m)
            t_ratio = t_new / self.t_total if self.t_total != 0 else 0
            
            # Restore original mass
            self.m = m_original
            
            if not self.fuzzy_equal(actual_B_ratio, expected_B_ratio):
                self.log(f"Monotonicity fails: B ratio {actual_B_ratio} ≠ {expected_B_ratio}", False)
                return False
            
            if not self.fuzzy_equal(t_ratio, 1.0):
                self.log(f"Monotonicity fails: t ratio {t_ratio} ≠ 1.0", False)
                return False
            
            self.log("Monotonicity test passed: B ∝ m, t independent of m", True)
            return True
        except Exception as e:
            self.log(f"Exception in test_monotonicity: {str(e)}", False)
            # Restore original mass in case of error
            if 'm_original' in locals():
                self.m = m_original
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        overall_success = True
        
        try:
            self.log("=== PhysicsChecker Starting ===")
            
            # Check if we have minimal inputs
            if None in [self.m, self.q, self.v0, self.d]:
                self.log("Missing input parameters", False)
                return False, self.logs
            
            # Run tests
            if not self.test_invariants():
                self.log("Invariants test failed", False)
                overall_success = False
            
            if not self.test_trace_consistency():
                self.log("Trace consistency test failed", False)
                overall_success = False
            
            if not self.test_monotonicity():
                self.log("Monotonicity test failed", False)
                overall_success = False
            
            self.log(f"=== PhysicsChecker Complete: {'PASS' if overall_success else 'FAIL'} ===")
            return overall_success, self.logs
            
        except Exception as e:
            self.log(f"Fatal error in run_all_tests: {str(e)}\n{traceback.format_exc()}", False)
            return False, self.logs