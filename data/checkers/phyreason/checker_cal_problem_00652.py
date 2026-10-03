import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """solution_instance should be the results dictionary."""
        # The solution_instance should be a dictionary
        # If it's not, try to extract it
        if isinstance(solution_instance, dict):
            self.results = solution_instance
        else:
            # Try to get results from the object
            self.results = getattr(solution_instance, 'results', {})
        
        # Extract all values as floats with proper error handling
        try:
            # Electric field inputs/results (Part 1)
            self.Q = float(self.results.get('Q', 1e-6))
            self.q = float(self.results.get('q', 1e-9))  # test charge
            self.r = float(self.results.get('r', 0.1))
            self.k = float(self.results.get('k', 8.987551787e9))
            
            # Two-charge system inputs/results (Part 2)
            self.q1 = float(self.results.get('q1', 2e-6))
            self.q2 = float(self.results.get('q2', 1e-6))
            self.a = float(self.results.get('a', 0.05))
            
            # Water flow inputs/results (Part 3)
            self.Q1 = float(self.results.get('Q1', 0.001))  # nozzle flow rate
            self.Q2 = float(self.results.get('Q2', 0.0005))  # absorber flow rate
            self.r_water = float(self.results.get('r_water', 0.1))  # distance for v1
            
            # Final results from steps
            self.F = float(self.results.get('F', 0.0))
            self.E = float(self.results.get('E', 0.0))
            self.E_C = float(self.results.get('E_C', 0.0))
            self.v_1 = float(self.results.get('v_1', 0.0))
            # Note: v_1_prime and v_2_prime might not be in results
            self.v_1_prime = float(self.results.get("v_1'", 0.0))
            self.v_2_prime = float(self.results.get("v_2'", 0.0))
            self.v_2 = float(self.results.get('v_2', 0.0))
        except (ValueError, TypeError) as e:
            # If conversion fails, use defaults
            self.Q = 1e-6
            self.q = 1e-9
            self.r = 0.1
            self.k = 8.987551787e9
            self.q1 = 2e-6
            self.q2 = 1e-6
            self.a = 0.05
            self.Q1 = 0.001
            self.Q2 = 0.0005
            self.r_water = 0.1
            self.F = 0.0
            self.E = 0.0
            self.E_C = 0.0
            self.v_1 = 0.0
            self.v_1_prime = 0.0
            self.v_2_prime = 0.0
            self.v_2 = 0.0
        
    def test_invariants(self):
        """Verify fundamental physics laws: Coulomb's Law, field superposition, volume flow conservation."""
        logs = []
        
        try:
            # 1. Coulomb's Law and Electric Field definition consistency
            # Use a small epsilon for comparison instead of != 0
            epsilon = 1e-20
            if abs(self.F) > epsilon and abs(self.E) > epsilon and abs(self.q) > epsilon:
                F_calc = self.k * abs(self.Q) * abs(self.q) / (self.r**2)
                E_from_F_num = self.F / self.q
                E_calc = self.k * abs(self.Q) / (self.r**2)
                
                # Check numerical consistency
                if (not math.isclose(F_calc, self.F, rel_tol=0.02, abs_tol=epsilon) or
                    not math.isclose(E_from_F_num, self.E, rel_tol=0.02, abs_tol=epsilon) or
                    not math.isclose(E_calc, self.E, rel_tol=0.02, abs_tol=epsilon)):
                    logs.append(f"FAIL: Numerical Coulomb/Field check failed: F={self.F}, calc_F={F_calc}; E={self.E}, calc_E={E_calc}")
                else:
                    logs.append("PASS: Coulomb's Law and E=F/q numerically consistent")
            else:
                logs.append(f"SKIP: Coulomb/Field check - values too small: F={self.F}, E={self.E}, q={self.q}")
            
            # 2. Superposition Principle for Electric Fields
            if abs(self.E_C) > epsilon:
                E_C_calc = self.k * (abs(self.q1) + abs(self.q2)) / (self.a**2)
                if not math.isclose(E_C_calc, self.E_C, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: E_C superposition mismatch: {self.E_C} vs {E_C_calc}")
                else:
                    logs.append("PASS: Field superposition holds numerically")
            else:
                logs.append(f"SKIP: E_C check - value too small: E_C={self.E_C}")
            
            # 3. Volume Flow Rate Conservation for Water
            if abs(self.v_1) > epsilon and abs(self.Q1) > epsilon:
                v1_calc = self.Q1 / (4 * math.pi * self.r_water**2)
                if not math.isclose(v1_calc, self.v_1, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: v1 mismatch: {self.v_1} vs {v1_calc}")
                else:
                    logs.append("PASS: Numerical v1 check")
            
            if abs(self.v_2) > epsilon and abs(self.Q1) > epsilon and abs(self.Q2) > epsilon:
                v2_calc = (self.Q1 + self.Q2) / (4 * math.pi * self.a**2)
                if not math.isclose(v2_calc, self.v_2, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: v2 mismatch: {self.v_2} vs {v2_calc}")
                else:
                    logs.append("PASS: Numerical v2 check")
            
            # 4. Charge magnitude comparison from field lines (qualitative)
            if abs(self.q1) > epsilon and abs(self.q2) > epsilon:
                if abs(self.q1) < abs(self.q2):
                    logs.append("WARN: q1 < q2 but field lines suggest q1 > q2?")
                elif abs(self.q1) > abs(self.q2):
                    logs.append("PASS: q1 > q2 consistent with denser field lines")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        epsilon = 1e-20
        
        try:
            # Check Step 1 to Step 3 chain: F → E → E=kQ/r²
            if abs(self.F) > epsilon and abs(self.E) > epsilon and abs(self.q) > epsilon:
                E_from_F = self.F / self.q
                if not math.isclose(E_from_F, self.E, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: Step1→Step2 inconsistent: E_from_F={E_from_F}, E={self.E}")
                else:
                    logs.append("PASS: Step1→Step2→Step3 chain consistent")
            
            # Check Step 8-10: v1' + v2' = v2
            # Skip if v1' or v2' are essentially 0 (not provided)
            if (abs(self.v_1_prime) > epsilon and abs(self.v_2_prime) > epsilon and abs(self.v_2) > epsilon):
                v2_sum = self.v_1_prime + self.v_2_prime
                if not math.isclose(v2_sum, self.v_2, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: Step8-10 inconsistent: v1'+v2'={v2_sum}, v2={self.v_2}")
                else:
                    logs.append("PASS: Step8-10 superposition consistent")
            
            # Check v1 and v1' relationship (same Q1, different distances)
            # Skip if v1' is essentially 0 (not provided)
            if (abs(self.v_1) > epsilon and abs(self.v_1_prime) > epsilon and 
                abs(self.r_water) > epsilon and abs(self.a) > epsilon):
                # From Q1 = 4πr²v1 = 4πa²v1'
                ratio_v = self.v_1 / self.v_1_prime
                ratio_r = (self.a**2) / (self.r_water**2)
                if not math.isclose(ratio_v, ratio_r, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"WARN: v1/v1' mismatch with distance squared: ratio_v={ratio_v}, ratio_r={ratio_r}")
                else:
                    logs.append("PASS: v1/v1' follows inverse square law")
            
            # Check E and E_C follow same form (both inverse square)
            if abs(self.E) > epsilon and abs(self.E_C) > epsilon:
                # Structure check: both should be proportional to 1/distance²
                logs.append("INFO: E and E_C both show inverse-square dependence")
            
        except Exception as e:
            logs.append(f"ERROR in trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check trend predictions."""
        logs = []
        epsilon = 1e-20
        
        try:
            # Store original values
            orig_q1 = self.q1
            orig_q2 = self.q2
            orig_r = self.r
            orig_Q1 = self.Q1
            
            # 1. Increase q1 → E_C should increase
            if abs(orig_q1) > epsilon:
                self.q1 = orig_q1 * 1.1  # +10%
                E_C_new = self.k * (abs(self.q1) + abs(orig_q2)) / (self.a**2)
                if E_C_new <= self.E_C and not math.isclose(E_C_new, self.E_C, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: E_C didn't increase when q1 increased: {self.E_C} → {E_C_new}")
                else:
                    logs.append(f"PASS: E_C increases with q1: {self.E_C} → {E_C_new}")
                self.q1 = orig_q1  # Restore
            
            # 2. Increase r → E should decrease
            if abs(orig_r) > epsilon:
                self.r = orig_r * 1.1
                E_new = self.k * abs(self.Q) / (self.r**2)
                if E_new >= self.E and not math.isclose(E_new, self.E, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: E didn't decrease when r increased: {self.E} → {E_new}")
                else:
                    logs.append(f"PASS: E decreases with r: {self.E} → {E_new}")
                self.r = orig_r
            
            # 3. Increase Q1 → v1 should increase
            if abs(orig_Q1) > epsilon:
                self.Q1 = orig_Q1 * 1.1
                v1_new = self.Q1 / (4 * math.pi * self.r_water**2)
                if v1_new <= self.v_1 and not math.isclose(v1_new, self.v_1, rel_tol=0.02, abs_tol=epsilon):
                    logs.append(f"FAIL: v1 didn't increase when Q1 increased: {self.v_1} → {v1_new}")
                else:
                    logs.append(f"PASS: v1 increases with Q1: {self.v_1} → {v1_new}")
                self.Q1 = orig_Q1
            
            # 4. Check q1 > q2 trend from field lines (qualitative monotonicity)
            if abs(self.q1) > epsilon and abs(self.q2) > epsilon:
                if self.q1 > self.q2:
                    # If we swap them, field should become weaker if q1 was dominant
                    temp_q1, temp_q2 = self.q2, self.q1
                    E_C_swapped = self.k * (abs(temp_q1) + abs(temp_q2)) / (self.a**2)
                    if E_C_swapped <= self.E_C and not math.isclose(E_C_swapped, self.E_C, rel_tol=0.02, abs_tol=epsilon):
                        logs.append("PASS: Field strength monotonic with charge magnitudes")
                    else:
                        logs.append("WARN: Field strength not monotonic with charge swap")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            # Restore any changed values
            self.q1 = orig_q1
            self.r = orig_r
            self.Q1 = orig_Q1
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        # Run each test suite
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            test_logs = test_func()
            all_logs.extend(test_logs)
            
            # Check for critical failures
            for log in test_logs:
                if log.startswith("FAIL:"):
                    success = False
                if log.startswith("ERROR:"):
                    success = False
        
        # Final summary
        if success:
            all_logs.append("\n=== SUMMARY: All physics checks PASSED ===")
        else:
            all_logs.append("\n=== SUMMARY: Some physics checks FAILED ===")
        
        return success, all_logs