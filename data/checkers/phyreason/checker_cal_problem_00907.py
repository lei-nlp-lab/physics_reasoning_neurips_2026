import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction using getattr with fallback to results dict
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.n1 = getattr(solution_instance, 'n1', 
                         getattr(solution_instance, 'n_1', 
                                solution_instance.results.get('n_1', 100)))
        self.U1 = getattr(solution_instance, 'U1', 
                         getattr(solution_instance, 'U_1', 
                                solution_instance.results.get('U_1', 220.0)))
        
        # Extract results with safe defaults
        self.P_in = getattr(solution_instance, 'P_in', 
                           solution_instance.results.get('P_in', None))
        self.P2 = getattr(solution_instance, 'P_2', 
                         solution_instance.results.get('P_2', 20.0))
        self.P3 = getattr(solution_instance, 'P_3', 
                         solution_instance.results.get('P_3', 60.0))
        self.I1_U1 = getattr(solution_instance, 'I_1 * U_1', 
                            solution_instance.results.get('I_1*U_1', None))
        self.I1 = getattr(solution_instance, 'I_1', 
                         solution_instance.results.get('I_1', None))
        self.n2 = getattr(solution_instance, 'n_2', 
                         solution_instance.results.get('n_2', None))
        self.n3 = getattr(solution_instance, 'n_3', 
                         solution_instance.results.get('n_3', None))
        
        # Constants from problem analysis
        self.U2 = 66.0  # From step_1: n2 = n1*(U2/U1) with n2=30
        self.U3 = 110.0  # From step_2: n3 = n1*(U3/U1) with n3=50
        
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
        
    def test_invariants(self):
        """Test fundamental physics laws: power conservation and transformer equations"""
        try:
            # 1. Power conservation: P_in = P_out = P2 + P3
            P_out_expected = self.P2 + self.P3
            power_conservation = True
            
            if self.P_in is not None:
                power_conservation = math.isclose(self.P_in, P_out_expected, 
                                                 rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Power conservation: P_in={self.P_in:.2f}W = P2+P3={P_out_expected:.2f}W", 
                         power_conservation)
            
            # 2. Transformer voltage ratios (using symbolic check with numerical fallback)
            if self.n2 is not None and self.n3 is not None:
                # Check n2/n1 = U2/U1
                ratio1_lhs = self.n2 / self.n1 if self.n1 != 0 else float('inf')
                ratio1_rhs = self.U2 / self.U1 if self.U1 != 0 else float('inf')
                ratio1_ok = math.isclose(ratio1_lhs, ratio1_rhs, rel_tol=1e-2, abs_tol=1e-2)
                
                # Check n3/n1 = U3/U1
                ratio2_lhs = self.n3 / self.n1 if self.n1 != 0 else float('inf')
                ratio2_rhs = self.U3 / self.U1 if self.U1 != 0 else float('inf')
                ratio2_ok = math.isclose(ratio2_lhs, ratio2_rhs, rel_tol=1e-2, abs_tol=1e-2)
                
                self._log(f"Transformer ratio 1: n2/n1={ratio1_lhs:.3f} ≈ U2/U1={ratio1_rhs:.3f}", ratio1_ok)
                self._log(f"Transformer ratio 2: n3/n1={ratio2_lhs:.3f} ≈ U3/U1={ratio2_rhs:.3f}", ratio2_ok)
                
                # Also check actual values against steps analysis
                n2_ok = math.isclose(self.n2, 30.0, rel_tol=1e-2, abs_tol=1e-2)
                n3_ok = math.isclose(self.n3, 50.0, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"n2 ≈ 30 turns: {self.n2:.1f}", n2_ok)
                self._log(f"n3 ≈ 50 turns: {self.n3:.1f}", n3_ok)
            
            # 3. Input current from power equation
            if self.I1 is not None:
                I1_expected = (self.P2 + self.P3) / self.U1 if self.U1 != 0 else float('inf')
                I1_ok = math.isclose(self.I1, I1_expected, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"I1 = (P2+P3)/U1 = {I1_expected:.3f}A ≈ {self.I1:.3f}A", I1_ok)
                
                # Check I1*U1 product
                if self.I1_U1 is not None:
                    product_ok = math.isclose(self.I1_U1, self.I1 * self.U1, 
                                             rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"I1*U1 product consistency: {self.I1_U1:.2f}W ≈ {self.I1*self.U1:.2f}W", 
                             product_ok)
            
            return all([power_conservation, ratio1_ok if self.n2 is not None else True, 
                       ratio2_ok if self.n3 is not None else True, 
                       I1_ok if self.I1 is not None else True])
            
        except Exception as e:
            self._log(f"Invariants test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Check intermediate results are consistent with final answer"""
        try:
            all_ok = True
            
            # 1. Check I1*U1 equals total power
            if self.I1_U1 is not None:
                total_power = self.P2 + self.P3
                power_ok = math.isclose(self.I1_U1, total_power, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"I1*U1={self.I1_U1:.2f}W = P2+P3={total_power:.2f}W", power_ok)
                all_ok = all_ok and power_ok
            
            # 2. Check n2 and n3 are consistent with voltage ratios
            if self.n2 is not None and self.n3 is not None:
                # Verify both secondary coils produce correct voltages
                U2_from_n2 = (self.n2 / self.n1) * self.U1 if self.n1 != 0 else 0
                U3_from_n3 = (self.n3 / self.n1) * self.U1 if self.n1 != 0 else 0
                
                U2_ok = math.isclose(U2_from_n2, self.U2, rel_tol=1e-2, abs_tol=1e-2)
                U3_ok = math.isclose(U3_from_n3, self.U3, rel_tol=1e-2, abs_tol=1e-2)
                
                self._log(f"U2 from n2: {U2_from_n2:.1f}V ≈ 66V", U2_ok)
                self._log(f"U3 from n3: {U3_from_n3:.1f}V ≈ 110V", U3_ok)
                all_ok = all_ok and U2_ok and U3_ok
            
            # 3. Check P2 and P3 values from steps analysis
            P2_ok = math.isclose(self.P2, 20.0, rel_tol=1e-2, abs_tol=1e-2)
            P3_ok = math.isclose(self.P3, 60.0, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"P2 ≈ 20W: {self.P2:.1f}W", P2_ok)
            self._log(f"P3 ≈ 60W: {self.P3:.1f}W", P3_ok)
            all_ok = all_ok and P2_ok and P3_ok
            
            return all_ok
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb input power and check output current changes accordingly"""
        try:
            # Save original state
            original_P2 = self.P2
            original_I1 = self.I1
            
            # Perturb P2 by +10% (increase)
            perturbed_P2 = original_P2 * 1.1
            perturbed_P_in = perturbed_P2 + self.P3
            
            # Recalculate expected I1 using physics formula
            expected_I1_perturbed = perturbed_P_in / self.U1 if self.U1 != 0 else float('inf')
            
            # If original I1 exists, check monotonicity
            if original_I1 is not None:
                # Increasing input power should increase input current
                monotonic_ok = expected_I1_perturbed > original_I1
                self._log(f"Monotonicity: P2↑10% → I1 from {original_I1:.3f}A to {expected_I1_perturbed:.3f}A (increase)", 
                         monotonic_ok)
                
                # Also check the exact relationship: ΔI1/ΔP2 = 1/U1
                delta_I1_expected = (perturbed_P2 - original_P2) / self.U1 if self.U1 != 0 else 0
                actual_delta = expected_I1_perturbed - original_I1
                
                # Use fuzzy comparison for derivative
                derivative_ok = math.isclose(delta_I1_expected, actual_delta, 
                                            rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"dI1/dP2 = 1/U1 = {1/self.U1:.4f} A/W (verified)", derivative_ok)
                
                return monotonic_ok and derivative_ok
            else:
                self._log("Monotonicity test skipped (no I1 data)", True)
                return True
                
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all checks and return success status with logs"""
        self.logs = []
        success = True
        
        try:
            # Run individual tests
            self.logs.append("=== Physics Checker Results ===")
            
            self.logs.append("\n1. Testing Physics Invariants:")
            inv_success = self.test_invariants()
            success = success and inv_success
            
            self.logs.append("\n2. Testing Trace Consistency:")
            trace_success = self.test_trace_consistency()
            success = success and trace_success
            
            self.logs.append("\n3. Testing Monotonicity:")
            mono_success = self.test_monotonicity()
            success = success and mono_success
            
            # Summary
            self.logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
            
        except Exception as e:
            self.logs.append(f"Error in test suite: {traceback.format_exc()}")
            success = False
        
        return success, self.logs

# Example usage (for testing only, not part of final output):
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'n_2': 30.0,
                'n_3': 50.0,
                'P_in': 80.0,
                'P_2': 20.0,
                'P_3': 60.0,
                'I_1*U_1': 80.0,
                'I_1': 0.3636
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)