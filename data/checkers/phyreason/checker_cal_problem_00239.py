import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction with fallbacks
        self.solution = solution_instance
        
        # Inputs (safe access with defaults)
        self.Um = getattr(solution_instance, 'Um', 0.0)
        self.R = getattr(solution_instance, 'R', 0.0)
        self.T0 = getattr(solution_instance, 'T0', 0.0)
        self.C = getattr(solution_instance, 'C', 0.0)  # Heat capacity of gas
        
        # Results dictionary with safe access
        res = getattr(solution_instance, 'results', {})
        self.P = res.get('P', 0.0)
        self.T2 = res.get('T_2', 0.0)
        self.U1 = res.get('U_1', 0.0)
        self.t = res.get('t', 0.0)
        self.U2 = res.get('U_2', 0.0)
        self.Q = res.get('Q', 0.0)
        self.deltaU = res.get('DeltaU', 0.0)
        self.W = res.get('W', 0.0)
        self.fracU1_U2 = res.get('fracU_1U_2', 0.0)
        
        # Transformer constants
        self.n1 = 5.0
        self.n2 = 1.0
        
        self.logs = []
    
    def log(self, msg, status="INFO"):
        self.logs.append(f"[{status}] {msg}")
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        checks = []
        
        # 1. Transformer voltage ratio (ideal)
        expected_ratio = self.n1 / self.n2
        if not math.isclose(self.fracU1_U2, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Transformer ratio mismatch: {self.fracU1_U2} vs {expected_ratio}", "WARNING")
            checks.append(False)
        else:
            checks.append(True)
        
        # 2. RMS voltage calculation: U1 = Um/√2
        expected_U1 = self.Um / math.sqrt(2)
        if not math.isclose(self.U1, expected_U1, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"RMS voltage mismatch: {self.U1} vs {expected_U1}", "WARNING")
            checks.append(False)
        else:
            checks.append(True)
        
        # 3. Secondary voltage from ratio: U2 = U1 * (n2/n1)
        expected_U2 = self.U1 * (self.n2 / self.n1)
        if not math.isclose(self.U2, expected_U2, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Secondary voltage mismatch: {self.U2} vs {expected_U2}", "WARNING")
            checks.append(False)
        else:
            checks.append(True)
        
        # 4. Power dissipation: P = U2²/R
        if self.R != 0:
            expected_P = (self.U2 ** 2) / self.R
            if not math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Power mismatch: {self.P} vs {expected_P}", "WARNING")
                checks.append(False)
            else:
                checks.append(True)
        
        # 5. Isochoric gas law: T2/T0 = P2/P0 = 2 (since pressure doubles)
        expected_T2 = 2.0 * self.T0
        if not math.isclose(self.T2, expected_T2, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Final temperature mismatch: {self.T2} vs {expected_T2}", "WARNING")
            checks.append(False)
        else:
            checks.append(True)
        
        # 6. Heat absorbed: Q = C * (T2 - T0) = C * T0
        expected_Q = self.C * (self.T2 - self.T0)
        if not math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Heat absorbed mismatch: {self.Q} vs {expected_Q}", "WARNING")
            checks.append(False)
        else:
            checks.append(True)
        
        # 7. First law of thermodynamics: ΔU = Q - W (W=0 for isochoric)
        # Since W should be 0, ΔU should equal Q
        if not math.isclose(self.deltaU, self.Q, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"First law violation: ΔU={self.deltaU}, Q={self.Q}, W={self.W}", "WARNING")
            checks.append(False)
        else:
            checks.append(True)
        
        # 8. Work done in isochoric process: W = 0
        if not math.isclose(self.W, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Non-zero work in isochoric process: W={self.W}", "WARNING")
            checks.append(False)
        else:
            checks.append(True)
        
        # 9. Energy-power relation: ΔU = P * t
        if self.P != 0:
            expected_t = self.deltaU / self.P
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Time mismatch: {self.t} vs {expected_t}", "WARNING")
                checks.append(False)
            else:
                checks.append(True)
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        checks = []
        
        # Target answers (from problem statement):
        # 1. P = Um²/(50R)
        # 2. t = (50 C T0 R)/Um²
        
        # Check P
        if self.R != 0:
            expected_P = (self.Um ** 2) / (50 * self.R)
            if not math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Final P mismatch with target: {self.P} vs {expected_P}", "WARNING")
                checks.append(False)
            else:
                checks.append(True)
        
        # Check t
        if self.Um != 0:
            expected_t = (50 * self.C * self.T0 * self.R) / (self.Um ** 2)
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Final t mismatch with target: {self.t} vs {expected_t}", "WARNING")
                checks.append(False)
            else:
                checks.append(True)
        
        # Verify step-by-step consistency from steps analysis
        # Step 1: U1 = Um/√2
        if not math.isclose(self.U1, self.Um/math.sqrt(2), rel_tol=1e-2, abs_tol=1e-2):
            self.log("Step 1 (RMS) inconsistent", "WARNING")
            checks.append(False)
        
        # Step 2-3: U2 = U1/5
        if not math.isclose(self.U2, self.U1/5.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log("Step 2-3 (transformer) inconsistent", "WARNING")
            checks.append(False)
        
        # Step 4: P = U2²/R
        if self.R != 0:
            if not math.isclose(self.P, (self.U2**2)/self.R, rel_tol=1e-2, abs_tol=1e-2):
                self.log("Step 4 (power) inconsistent", "WARNING")
                checks.append(False)
        
        # Step 5: T2 = 2*T0
        if not math.isclose(self.T2, 2.0*self.T0, rel_tol=1e-2, abs_tol=1e-2):
            self.log("Step 5 (temperature) inconsistent", "WARNING")
            checks.append(False)
        
        # Step 6: Q = C*T0
        if not math.isclose(self.Q, self.C*self.T0, rel_tol=1e-2, abs_tol=1e-2):
            self.log("Step 6 (heat) inconsistent", "WARNING")
            checks.append(False)
        
        # Step 8: W = 0
        if not math.isclose(self.W, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log("Step 8 (work) inconsistent", "WARNING")
            checks.append(False)
        
        # Step 9-10: t = Q/P = (C*T0) / (Um²/(50R))
        if self.P != 0:
            if not math.isclose(self.t, self.Q/self.P, rel_tol=1e-2, abs_tol=1e-2):
                self.log("Step 9-10 (time) inconsistent", "WARNING")
                checks.append(False)
        
        return all(checks)
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        original_inputs = {'Um': self.Um, 'R': self.R, 'C': self.C, 'T0': self.T0}
        perturbations = [
            ('Um', 1.1, lambda old, new: (new**2)/(old**2)),  # P ∝ Um²
            ('R', 1.1, lambda old, new: old/new),            # P ∝ 1/R
            ('C', 1.1, lambda old, new: new/old),            # t ∝ C
            ('T0', 1.1, lambda old, new: new/old)            # t ∝ T0
        ]
        
        checks = []
        for param, factor, trend_func in perturbations:
            try:
                # Store original
                original = getattr(self, param)
                
                # Perturb
                setattr(self, param, original * factor)
                
                # Recompute dependent quantities using formulas
                U1_new = self.Um / math.sqrt(2)
                U2_new = U1_new * (self.n2 / self.n1)
                P_new = (U2_new ** 2) / self.R if self.R != 0 else 0.0
                T2_new = 2.0 * self.T0
                Q_new = self.C * (T2_new - self.T0)
                deltaU_new = Q_new  # W=0
                t_new = deltaU_new / P_new if P_new != 0 else 0.0
                
                # Compute original values for comparison
                P_orig = self.P
                t_orig = self.t
                
                # Check expected trend
                expected_ratio = trend_func(original, getattr(self, param))
                
                if param in ['Um', 'R']:
                    # Check power trend
                    if P_orig != 0 and P_new != 0:
                        actual_ratio = P_new / P_orig
                        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"Monotonicity failed for {param} on P: {actual_ratio} vs {expected_ratio}", "WARNING")
                            checks.append(False)
                        else:
                            checks.append(True)
                
                if param in ['C', 'T0']:
                    # Check time trend
                    if t_orig != 0 and t_new != 0:
                        actual_ratio = t_new / t_orig
                        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"Monotonicity failed for {param} on t: {actual_ratio} vs {expected_ratio}", "WARNING")
                            checks.append(False)
                        else:
                            checks.append(True)
                
                # Restore original value
                setattr(self, param, original)
                
            except Exception as e:
                self.log(f"Monotonicity test failed for {param}: {str(e)}", "ERROR")
                checks.append(False)
                # Restore original
                setattr(self, param, original_inputs[param])
        
        return all(checks)
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self.log(f"Running {test_name}...")
                if not test_func():
                    success = False
                    self.log(f"{test_name} failed", "ERROR")
                else:
                    self.log(f"{test_name} passed", "INFO")
            except Exception as e:
                success = False
                self.log(f"{test_name} crashed: {traceback.format_exc()}", "ERROR")
        
        return success, self.logs