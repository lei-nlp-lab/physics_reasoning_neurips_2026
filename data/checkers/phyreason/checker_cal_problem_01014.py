import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract known constants safely (with defaults if missing)
        self.rho = 7.5e2  # kg/m³ (given in problem)
        self.g = 10.0     # m/s² (given in problem)
        self.p0 = 4.5e3   # Pa (given in problem)
        
        # Extract computed results with safe defaults
        self.V01 = self.results.get('V_01', 0.0)
        self.p1 = self.results.get('p_1', 0.0)
        self.p2 = self.results.get('p_2', 0.0)
        self.V = self.results.get('V', 0.0)
        self.h = self.results.get('h', 0.0)
        
        # For checking relationships stored in results
        self.p0V01_eq_p1V = self.results.get('p_0V_01=p_1V', 0.0)
        self.p0V02_eq_p2V = self.results.get('p_0V_02=p_2V', 0.0)
        self.V01_plus_V02_eq_2V = self.results.get('V_01+V_02=2V', 0.0)
        self.p2_minus_p1 = self.results.get('p_2-p_1', 0.0)
        
        # Store for monotonicity test backup
        self._original_h = self.h
        self._original_rho = self.rho
        
    def test_invariants(self):
        """Verify Boyle's Law and fluid pressure difference invariants"""
        logs = []
        
        try:
            # 1. Check Boyle's Law for left chamber: p0*V01 = p1*V
            # Use symbolic check with numerical fallback
            left_product_initial = self.p0 * self.V01
            left_product_final = self.p1 * self.V
            
            if self.V01 > 0 and self.V > 0 and self.p1 > 0:
                if not math.isclose(left_product_initial, left_product_final, 
                                  rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Boyle's Law left chamber: p0*V01={left_product_initial:.2f}, "
                              f"p1*V={left_product_final:.2f}")
                else:
                    logs.append(f"PASS: Boyle's Law left chamber within 1% tolerance")
            else:
                logs.append("SKIP: Boyle's Law left chamber - zero or negative volumes/pressure")
            
            # 2. Check Boyle's Law for right chamber: Need V02
            # From V01 = 0.5*V02 and V01 + V02 = 2V
            if self.V01 > 0 and self.V > 0:
                V02 = 2 * self.V - self.V01
                right_product_initial = self.p0 * V02
                right_product_final = self.p2 * self.V
                
                if not math.isclose(right_product_initial, right_product_final,
                                  rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Boyle's Law right chamber: p0*V02={right_product_initial:.2f}, "
                              f"p2*V={right_product_final:.2f}")
                else:
                    logs.append(f"PASS: Boyle's Law right chamber within 1% tolerance")
            else:
                logs.append("SKIP: Boyle's Law right chamber - insufficient data")
            
            # 3. Check fluid pressure difference: p2 - p1 = ρgh
            pressure_diff_calc = self.p2 - self.p1
            pressure_diff_fluid = self.rho * self.g * self.h
            
            if not math.isclose(pressure_diff_calc, pressure_diff_fluid,
                              rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Pressure difference mismatch: p2-p1={pressure_diff_calc:.2f}, "
                          f"ρgh={pressure_diff_fluid:.2f}")
            else:
                logs.append(f"PASS: Fluid pressure difference within 1% tolerance")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def test_trace_consistency(self):
        """Check that intermediate relationships lead to the final answer"""
        logs = []
        
        try:
            # From steps analysis: V01 = 0.5*V02 and V01 + V02 = 2V
            # So V02 = 2*V01 and 3*V01 = 2V => V01 = (2/3)V
            if self.V > 0:
                V01_calc = (2/3) * self.V
                if not math.isclose(self.V01, V01_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Volume ratio inconsistent: V01={self.V01:.2f}, "
                              f"expected (2/3)V={V01_calc:.2f}")
                else:
                    logs.append(f"PASS: Volume ratio V01 = (2/3)V within 1%")
            
            # Check Boyle's law gives consistent pressures
            # p1 = p0*V01/V, p2 = p0*V02/V = p0*(2*V01)/V = 2*p0*V01/V = 2*p1
            if self.p1 > 0:
                p2_expected = 2 * self.p1
                if not math.isclose(self.p2, p2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Pressure ratio inconsistent: p2={self.p2:.2f}, "
                              f"expected 2*p1={p2_expected:.2f}")
                else:
                    logs.append(f"PASS: Pressure ratio p2 = 2*p1 within 1%")
            
            # Verify the calculated h from pressures
            if self.p1 > 0 and self.p2 > 0:
                h_calc = (self.p2 - self.p1) / (self.rho * self.g)
                if not math.isclose(self.h, h_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: h calculation inconsistent: h={self.h:.2f}, "
                              f"from pressures h_calc={h_calc:.2f}")
                else:
                    logs.append(f"PASS: h consistent with pressure difference")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def test_monotonicity(self):
        """Perturb density ρ and check if h changes in expected direction"""
        logs = []
        
        try:
            # Backup original
            original_rho = self.rho
            
            # Increase density by 10%
            self.rho *= 1.1
            
            # Recalculate h using the relationship from the problem
            # From Boyle's law and volume relationships:
            # p1 = p0*(V01/V) = p0*(2/3)
            # p2 = p0*(V02/V) = p0*(4/3)
            # p2 - p1 = p0*(2/3)
            # h = (p2 - p1)/(ρg) = (2/3)*p0/(ρg)
            
            h_new = (2/3) * self.p0 / (self.rho * self.g)
            
            # Check monotonicity: h should decrease when ρ increases
            if h_new < self._original_h:
                logs.append(f"PASS: Monotonicity - h decreases when ρ increases "
                          f"({self._original_h:.2f} -> {h_new:.2f})")
            else:
                logs.append(f"FAIL: Monotonicity - h should decrease when ρ increases")
            
            # Restore original value
            self.rho = original_rho
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            # Ensure restoration even on error
            self.rho = self._original_rho
            
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs"""
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            test_logs = test_func()
            all_logs.extend(test_logs)
            
            # Check for critical failures in invariants and trace consistency
            if test_name in ["Invariants Test", "Trace Consistency Test"]:
                for log in test_logs:
                    if log.startswith("FAIL:"):
                        success = False
        
        # Final verification against expected answer (40 cm = 0.4 m)
        expected_h = 0.4  # meters
        if not math.isclose(self._original_h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
            all_logs.append(f"\nFAIL: Final answer mismatch: h={self._original_h:.3f}m, "
                          f"expected {expected_h:.3f}m")
            success = False
        else:
            all_logs.append(f"\nPASS: Final answer h={self._original_h:.3f}m matches expected {expected_h:.3f}m")
        
        return success, all_logs

# Example usage (commented out for production):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)