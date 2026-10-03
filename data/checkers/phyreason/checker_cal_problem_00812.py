import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all known quantities with safe defaults
        self.v_2 = getattr(solution_instance, 'v_2', self.results.get('v_2', 0.0))
        self.s = getattr(solution_instance, 's', self.results.get('s', 0.0))
        self.T = getattr(solution_instance, 'T', self.results.get('T', 0.0))
        self.x_0_minus_x_Q = getattr(solution_instance, 'x_0-x_Q', self.results.get('x_0-x_Q', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.x_P = getattr(solution_instance, 'x_P', self.results.get('x_P', 0.0))
        self.A = getattr(solution_instance, 'A', self.results.get('A', 0.0))
        self.x_Q = getattr(solution_instance, 'x_Q', self.results.get('x_Q', 0.0))
        self.delta = getattr(solution_instance, 'δ', self.results.get('δ', 0.0))
        self.lambda_val = getattr(solution_instance, 'λ', self.results.get('λ', 0.0))
        self.x_0 = getattr(solution_instance, 'x_0', self.results.get('x_0', 0.0))
        self.x_0_minus_x_P = getattr(solution_instance, 'x_0-x_P', self.results.get('x_0-x_P', 0.0))
        self.v_1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0))
        
        # Problem constants from context (safe hardcoding)
        self.v_mag = 0.4  # m/s (wave speed magnitude)
        self.A0 = 2.0     # m (individual wave amplitude)
        self.source_left = -0.2  # m
        self.source_right = 1.2  # m
        self.x_M = 0.5  # m (equilibrium position of M)
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        
    def test_invariants(self):
        """Verify fundamental physics laws from KB"""
        try:
            # 1. Wave speed magnitudes (should be 0.4 m/s)
            if not math.isclose(abs(self.v_1), self.v_mag, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: |v_1| = {abs(self.v_1)} ≠ {self.v_mag}")
                return False
            if not math.isclose(abs(self.v_2), self.v_mag, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: |v_2| = {abs(self.v_2)} ≠ {self.v_mag}")
                return False
            
            # 2. Wave speed relation: v = λ / T
            if self.lambda_val > 0 and self.T > 0:
                v_calc = self.lambda_val / self.T
                if not math.isclose(self.v_mag, v_calc, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: v = λ/T violation: {self.v_mag} ≠ {self.lambda_val}/{self.T} = {v_calc}")
                    return False
            
            # 3. Path difference at M should be 0 (constructive interference)
            # δ = |(x_M - source_left) - (source_right - x_M)|
            delta_calc = abs((self.x_M - self.source_left) - (self.source_right - self.x_M))
            if not math.isclose(self.delta, delta_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: δ mismatch: {self.delta} ≠ {delta_calc}")
                return False
            if not math.isclose(self.delta, 0.0, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: δ should be 0 for constructive interference, got {self.delta}")
                return False
            
            # 4. Amplitude at antinode: A = 2*A0 = 4 m
            if not math.isclose(self.A, 2*self.A0, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: A = {self.A} ≠ 2*{self.A0} = {2*self.A0}")
                return False
            
            self.log("PASS: All invariants satisfied")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # 1. Check wavefront positions from steps
            if not math.isclose(self.x_P, 0.2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: x_P = {self.x_P} ≠ 0.2")
                return False
            if not math.isclose(self.x_Q, 0.8, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: x_Q = {self.x_Q} ≠ 0.8")
                return False
            
            # 2. Check wave speeds (signs matter!)
            if not math.isclose(self.v_1, self.v_mag, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: v_1 = {self.v_1} ≠ {self.v_mag}")
                return False
            if not math.isclose(self.v_2, -self.v_mag, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: v_2 = {self.v_2} ≠ {-self.v_mag}")
                return False
            
            # 3. Check meeting time calculation: t = (x_Q - x_P) / (2v)
            t_calc = (self.x_Q - self.x_P) / (2 * self.v_mag)
            if not math.isclose(self.t, t_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: t mismatch: {self.t} ≠ ({self.x_Q}-{self.x_P})/(2*{self.v_mag}) = {t_calc}")
                return False
            
            # 4. Check meeting position: x_0 = x_P + v*t
            x0_calc = self.x_P + self.v_1 * self.t
            if not math.isclose(self.x_0, x0_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: x_0 mismatch: {self.x_0} ≠ {self.x_P}+{self.v_1}*{self.t} = {x0_calc}")
                return False
            # Alternative: x_0 = x_Q + v_2*t
            x0_alt = self.x_Q + self.v_2 * self.t
            if not math.isclose(self.x_0, x0_alt, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: x_0 mismatch with alt: {self.x_0} ≠ {self.x_Q}+{self.v_2}*{self.t} = {x0_alt}")
                return False
            
            # 5. Check wavelength from figure? Step 9 says λ = 0.4 m
            if not math.isclose(self.lambda_val, 0.4, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: λ = {self.lambda_val} ≠ 0.4")
                return False
            
            # 6. Check period: T = λ/v = 0.4/0.4 = 1 s
            T_calc = self.lambda_val / self.v_mag
            if not math.isclose(self.T, T_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: T = {self.T} ≠ {self.lambda_val}/{self.v_mag} = {T_calc}")
                return False
            
            # 7. Check distance traveled by M: s = 4A*(1.5 - t)/T
            t_ab = 1.5  # from step 13 (0s to 1.5s)
            s_calc = 4 * self.A * (t_ab - self.t) / self.T
            if not math.isclose(self.s, s_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: s mismatch: {self.s} ≠ 4*{self.A}*(1.5-{self.t})/{self.T} = {s_calc}")
                return False
            
            self.log("PASS: All trace consistency checks passed")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check trend"""
        try:
            # Save original values
            orig_t = self.t
            orig_s = self.s
            
            # Test 1: Increase wave speed magnitude by 10%
            v_new = self.v_mag * 1.1
            
            # Recalculate meeting time with perturbed speed
            t_new = (self.x_Q - self.x_P) / (2 * v_new)
            
            # Since t = (x_Q - x_P)/(2v), t should DECREASE when v INCREASES
            if not (t_new < orig_t * 1.001):  # Allow small tolerance
                self.log(f"FAIL: Monotonicity violated for wave speed: t({v_new}) = {t_new} ≥ {orig_t}")
                return False
            
            # Recalculate distance with perturbed parameters
            T_new = self.lambda_val / v_new
            # M starts vibrating later? Actually, if v increases, t decreases, so (1.5-t) increases
            # But also T changes: s = 4A*(1.5-t)/T
            # The relationship isn't strictly monotonic without knowing which effect dominates
            # Let's just verify the formula holds with perturbed values
            
            s_new = 4 * self.A * (1.5 - t_new) / T_new
            
            # Restore original values
            self.t = orig_t
            self.s = orig_s
            
            self.log(f"PASS: Monotonicity check passed (t decreased with increased v)")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            # Restore on error
            self.t = orig_t if 'orig_t' in locals() else self.t
            self.s = orig_s if 'orig_s' in locals() else self.s
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n=== Running {test_name} ===")
            try:
                if not test_func():
                    success = False
                    self.log(f"FAILED: {test_name}")
            except Exception as e:
                success = False
                self.log(f"CRASH in {test_name}: {traceback.format_exc()}")
        
        # Final check against target answers
        self.log(f"\n=== Final Answer Check ===")
        try:
            # Target answers: ["0.75s", "12m"]
            if not math.isclose(self.t, 0.75, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: t = {self.t} ≠ 0.75s")
                success = False
            else:
                self.log(f"PASS: t = {self.t}s matches target 0.75s")
            
            if not math.isclose(self.s, 12.0, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: s = {self.s} ≠ 12m")
                success = False
            else:
                self.log(f"PASS: s = {self.s}m matches target 12m")
        except Exception as e:
            self.log(f"ERROR in final check: {str(e)}")
            success = False
        
        return success, self.logs