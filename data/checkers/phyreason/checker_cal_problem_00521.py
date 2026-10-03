import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using getattr/results.get"""
        self.solution = solution_instance
        
        # Get results dict safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs (from STEPS ANALYSIS)
        self.U = 50.0  # Rated voltage (V)
        self.I = 4.0   # Rated current (A)
        self.P_out = 120.0  # Rated output power (W)
        
        # Extract results with fallbacks
        self.P_total = self.results.get('P_total', 0.0)
        self.P_hot = self.results.get('P_hot', 0.0)
        self.R_in = self.results.get('R_in', 0.0)
        self.eta = self.results.get('η', 0.0)
        self.P = self.results.get('P', 0.0)  # Stalled power
        
        # Expected target values
        self.expected_R_in = 5.0
        self.expected_eta = 60.0  # percent
        self.expected_P_stalled = 500.0
        
        # Physics constants
        self.tol_rel = 1e-2
        self.tol_abs = 1e-2
        
    def test_invariants(self):
        """Test fundamental physics laws using fuzzy matching"""
        logs = []
        tests_passed = True
        
        try:
            # 1. Ohm's Law: P_total = U * I
            P_total_calc = self.U * self.I
            if not math.isclose(self.P_total, P_total_calc, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Ohm's Law: P_total = {self.P_total}, U*I = {P_total_calc}")
                tests_passed = False
            else:
                logs.append(f"PASS: Ohm's Law P_total = U*I")
            
            # 2. Conservation of Energy: P_total = P_out + P_hot
            P_sum = self.P_out + self.P_hot
            if not math.isclose(self.P_total, P_sum, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Energy Conservation: P_total = {self.P_total}, P_out+P_hot = {P_sum}")
                tests_passed = False
            else:
                logs.append(f"PASS: Energy Conservation P_total = P_out + P_hot")
            
            # 3. Joule's Law: P_hot = I² * R_in
            P_hot_calc = self.I**2 * self.R_in
            if not math.isclose(self.P_hot, P_hot_calc, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Joule's Law: P_hot = {self.P_hot}, I²*R_in = {P_hot_calc}")
                tests_passed = False
            else:
                logs.append(f"PASS: Joule's Law P_hot = I²*R_in")
            
            # 4. Efficiency formula: η = (P_out/P_total)*100%
            eta_calc = (self.P_out / self.P_total) * 100.0
            if not math.isclose(self.eta, eta_calc, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Efficiency: η = {self.eta}%, P_out/P_total*100 = {eta_calc}%")
                tests_passed = False
            else:
                logs.append(f"PASS: Efficiency η = (P_out/P_total)×100%")
            
            # 5. Stalled power: P = U² / R_in
            if self.R_in > 0:
                P_stalled_calc = self.U**2 / self.R_in
                if not math.isclose(self.P, P_stalled_calc, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                    logs.append(f"FAIL: Stalled Power: P = {self.P}, U²/R_in = {P_stalled_calc}")
                    tests_passed = False
                else:
                    logs.append(f"PASS: Stalled Power P = U²/R_in")
            else:
                logs.append("WARN: R_in is zero, cannot check stalled power")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            tests_passed = False
            
        return tests_passed, logs
    
    def test_trace_consistency(self):
        """Check intermediate calculations against final results"""
        logs = []
        tests_passed = True
        
        try:
            # Check each step from STEPS ANALYSIS
            # Step 4: P_total = 200W
            if not math.isclose(self.P_total, 200.0, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Step 4 P_total mismatch: {self.P_total} vs 200")
                tests_passed = False
            
            # Step 5: P_hot = 80W
            if not math.isclose(self.P_hot, 80.0, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Step 5 P_hot mismatch: {self.P_hot} vs 80")
                tests_passed = False
            
            # Step 7: R_in = 5Ω
            if not math.isclose(self.R_in, 5.0, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Step 7 R_in mismatch: {self.R_in} vs 5")
                tests_passed = False
            
            # Step 8: η = 60%
            if not math.isclose(self.eta, 60.0, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Step 8 η mismatch: {self.eta} vs 60")
                tests_passed = False
            
            # Step 9: P = 500W
            if not math.isclose(self.P, 500.0, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: Step 9 P mismatch: {self.P} vs 500")
                tests_passed = False
            
            if tests_passed:
                logs.append("PASS: All steps consistent with final results")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            tests_passed = False
            
        return tests_passed, logs
    
    def test_monotonicity(self):
        """Test that formulas respond correctly to parameter changes"""
        logs = []
        tests_passed = True
        
        try:
            # Test 1: If resistance increases, stalled power decreases
            original_R = self.R_in
            original_P = self.P
            
            # Perturb R_in upward by 10%
            perturbed_R = original_R * 1.1
            perturbed_P = self.U**2 / perturbed_R
            
            if not (perturbed_P < original_P):
                logs.append(f"FAIL: Monotonicity R↑: P should decrease but {perturbed_P} >= {original_P}")
                tests_passed = False
            else:
                logs.append(f"PASS: R↑ → P↓ monotonicity holds")
            
            # Test 2: If voltage increases, total power increases (with fixed current)
            original_P_total = self.P_total
            perturbed_U = self.U * 1.1
            perturbed_P_total = perturbed_U * self.I
            
            if not (perturbed_P_total > original_P_total):
                logs.append(f"FAIL: Monotonicity U↑: P_total should increase but {perturbed_P_total} <= {original_P_total}")
                tests_passed = False
            else:
                logs.append(f"PASS: U↑ → P_total↑ monotonicity holds")
            
            # Test 3: If output power increases (keeping total power fixed), efficiency increases
            if self.P_total > 0:
                original_eta = self.eta
                perturbed_P_out = self.P_out * 1.1
                perturbed_eta = (perturbed_P_out / self.P_total) * 100.0
                
                if not (perturbed_eta > original_eta):
                    logs.append(f"FAIL: Monotonicity P_out↑: η should increase but {perturbed_eta} <= {original_eta}")
                    tests_passed = False
                else:
                    logs.append(f"PASS: P_out↑ → η↑ monotonicity holds")
            
            # Test 4: If current increases, heating power increases quadratically
            original_P_hot = self.P_hot
            perturbed_I = self.I * 1.1
            perturbed_P_hot = perturbed_I**2 * self.R_in
            
            if not (perturbed_P_hot > original_P_hot):
                logs.append(f"FAIL: Monotonicity I↑: P_hot should increase but {perturbed_P_hot} <= {original_P_hot}")
                tests_passed = False
            else:
                logs.append(f"PASS: I↑ → P_hot↑ monotonicity holds")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            tests_passed = False
            
        return tests_passed, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        all_logs = []
        all_passed = True
        
        # Run invariant tests
        passed, logs = self.test_invariants()
        all_logs.extend(["=== Invariant Tests ==="] + logs)
        all_passed = all_passed and passed
        
        # Run trace consistency tests
        passed, logs = self.test_trace_consistency()
        all_logs.extend(["", "=== Trace Consistency Tests ==="] + logs)
        all_passed = all_passed and passed
        
        # Run monotonicity tests
        passed, logs = self.test_monotonicity()
        all_logs.extend(["", "=== Monotonicity Tests ==="] + logs)
        all_passed = all_passed and passed
        
        # Overall result
        if all_passed:
            all_logs.insert(0, "SUCCESS: All physics checks passed")
        else:
            all_logs.insert(0, "FAILURE: Some physics checks failed")
        
        return all_passed, all_logs