import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.U_m = getattr(solution_instance, 'U_m', solution_instance.results.get('U_m', 0.0))
        self.U_1 = getattr(solution_instance, 'U_1', solution_instance.results.get('U_1', 0.0))
        self.U_2 = getattr(solution_instance, 'U_2', solution_instance.results.get('U_2', 0.0))
        
        # Problem constants
        self.turns_ratio = 4.0  # Np:Ns = 4:1
        self.primary_peak = 400.0 * math.sqrt(2)  # From given expression
        
    def test_invariants(self):
        """Check fundamental physical laws using both symbolic and numerical methods."""
        logs = []
        success = True
        
        try:
            # Symbolic verification of peak-RMS relationship
            V_peak, V_rms = sp.symbols('V_peak V_rms', positive=True)
            peak_rms_law = sp.Eq(V_rms, V_peak / sp.sqrt(2))
            numeric_check = math.isclose(
                self.U_1, 
                self.primary_peak / math.sqrt(2), 
                rel_tol=1e-2, 
                abs_tol=1e-2
            )
            
            if not numeric_check:
                logs.append(f"FAIL: Peak-RMS invariant violated. U_1={self.U_1:.2f}, expected {self.primary_peak/math.sqrt(2):.2f}")
                success = False
            else:
                logs.append(f"PASS: Peak-RMS relationship holds (U_1 ≈ U_m/√2)")
            
            # Transformer voltage ratio invariant
            Np, Ns, Vp, Vs = sp.symbols('Np Ns Vp Vs', positive=True)
            transformer_law = sp.Eq(Vp/Vs, Np/Ns)
            
            # Numerical check with fuzzy matching
            if not math.isclose(
                self.U_1 / self.U_2, 
                self.turns_ratio, 
                rel_tol=1e-2, 
                abs_tol=1e-2
            ):
                logs.append(f"FAIL: Transformer ratio violated. U_1/U_2={self.U_1/self.U_2:.2f}, expected {self.turns_ratio}")
                success = False
            else:
                logs.append(f"PASS: Transformer voltage ratio holds (U_1/U_2 ≈ {self.turns_ratio})")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        logs = []
        success = True
        
        try:
            # Step 1: U_m should be 400√2
            if not math.isclose(self.U_m, self.primary_peak, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_m mismatch. Got {self.U_m:.2f}, expected {self.primary_peak:.2f}")
                success = False
            else:
                logs.append(f"PASS: U_m = {self.U_m:.2f} V (matches given expression)")
            
            # Step 2: U_1 = U_m/√2
            expected_U1 = self.primary_peak / math.sqrt(2)
            if not math.isclose(self.U_1, expected_U1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_1 mismatch. Got {self.U_1:.2f}, expected {expected_U1:.2f}")
                success = False
            else:
                logs.append(f"PASS: U_1 = {self.U_1:.2f} V (correct RMS calculation)")
            
            # Step 4: U_2 = U_1 * (Ns/Np) = U_1/4
            expected_U2 = self.U_1 / self.turns_ratio
            if not math.isclose(self.U_2, expected_U2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_2 mismatch. Got {self.U_2:.2f}, expected {expected_U2:.2f}")
                success = False
            else:
                logs.append(f"PASS: U_2 = {self.U_2:.2f} V (consistent with transformer law)")
            
            # Final answer check
            target_answer = 100.0
            if not math.isclose(self.U_2, target_answer, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final answer mismatch. Got {self.U_2:.2f} V, target is {target_answer} V")
                success = False
            else:
                logs.append(f"PASS: Final answer matches target ({target_answer} V)")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test sensitivity to input perturbations while preserving physical laws."""
        logs = []
        success = True
        
        try:
            # Store original values
            original_U1 = self.U_1
            original_U2 = self.U_2
            
            # Perturbation 1: Increase turns ratio (Np/Ns)
            perturbed_ratio = self.turns_ratio * 1.1  # +10%
            perturbed_U2 = original_U1 / perturbed_ratio
            
            # Check monotonic relationship: U2 decreases when turns ratio increases
            if not (perturbed_U2 < original_U2):
                logs.append(f"FAIL: Monotonicity violated. U2 should decrease when Np/Ns increases")
                logs.append(f"  Original: Np/Ns={self.turns_ratio}, U2={original_U2:.2f}")
                logs.append(f"  Perturbed: Np/Ns={perturbed_ratio}, U2={perturbed_U2:.2f}")
                success = False
            else:
                logs.append(f"PASS: U2 decreases appropriately when Np/Ns increases")
            
            # Perturbation 2: Increase primary peak voltage
            perturbed_peak = self.primary_peak * 1.1  # +10%
            perturbed_U1_new = perturbed_peak / math.sqrt(2)
            perturbed_U2_new = perturbed_U1_new / self.turns_ratio
            
            # Check monotonic relationship: U2 increases when primary voltage increases
            if not (perturbed_U2_new > original_U2):
                logs.append(f"FAIL: Monotonicity violated. U2 should increase when U_peak increases")
                logs.append(f"  Original: U_peak={self.primary_peak:.2f}, U2={original_U2:.2f}")
                logs.append(f"  Perturbed: U_peak={perturbed_peak:.2f}, U2={perturbed_U2_new:.2f}")
                success = False
            else:
                logs.append(f"PASS: U2 increases appropriately when primary voltage increases")
            
            # Test derivative sign using symbolic approach
            Np, Ns, Vp = sp.symbols('Np Ns Vp', positive=True)
            Vs_expr = Vp * Ns / Np
            dVs_dNp = sp.diff(Vs_expr, Np)
            dVs_dVp = sp.diff(Vs_expr, Vp)
            
            # dVs/dNp should be negative, dVs/dVp should be positive
            if not (dVs_dNp.subs({Ns: 1, Np: 4}) < 0):
                logs.append("FAIL: Symbolic derivative dVs/dNp should be negative")
                success = False
            if not (dVs_dVp.subs({Ns: 1, Np: 4}) > 0):
                logs.append("FAIL: Symbolic derivative dVs/dVp should be positive")
                success = False
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all verification tests."""
        all_logs = []
        overall_success = True
        
        # Run individual tests
        for test_name, test_method in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            success, logs = test_method()
            overall_success = overall_success and success
            all_logs.extend(logs)
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All physics checks PASSED ✓")
        else:
            all_logs.append("Some physics checks FAILED ✗")
        
        return overall_success, all_logs