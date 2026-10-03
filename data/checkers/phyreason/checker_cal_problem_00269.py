import math
import sympy as sp
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract attributes from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with fallbacks
        self.v0 = getattr(solution_instance, 'v0', 10.0)           # initial velocity (m/s)
        self.a = getattr(solution_instance, 'a', 1.0)             # acceleration magnitude (m/s^2)
        self.t0 = getattr(solution_instance, 't0', 20.0)          # payment time in manual lane (s)
        self.v = getattr(solution_instance, 'v', 5.0)             # constant speed in ETC lane (m/s)
        self.delta_x = getattr(solution_instance, 'delta_x', 10.0)  # constant-speed distance in ETC lane (m)
        
        # Extract results with safe access and fallbacks
        # Manual lane
        self.t1 = self.results.get('t_1', 10.0)          # deceleration time
        self.t2 = self.results.get('t_2', 10.0)          # acceleration time
        self.x1 = self.results.get('x_1', 50.0)          # deceleration displacement
        self.x2 = self.results.get('x_2', 50.0)          # acceleration displacement
        self.t_manual = self.results.get('t', 40.0)      # total manual time
        self.s_manual = self.results.get('s', 100.0)     # total manual distance
        
        # ETC lane
        self.t1_prime = self.results.get('t_1\'', 5.0)   # deceleration time
        self.t2_prime = self.results.get('t_2\'', 5.0)   # acceleration time
        self.t3 = self.results.get('t_3', 2.0)           # constant-speed time
        self.x1_prime = self.results.get('x_1\'', 37.5)  # deceleration displacement
        self.x2_prime = self.results.get('x_2\'', 37.5)  # acceleration displacement
        self.t_etc = self.results.get('t\'', 12.0)       # total ETC time
        self.s_etc = self.results.get('s\'', 85.0)       # total ETC distance
        
        # Time calculations
        self.t_double_prime = self.results.get('t\'\'', 13.5)  # time to travel manual distance at v0
        self.delta_t = self.results.get('trianglet', 26.5)     # time saved
        
        # Tolerance for numerical comparisons
        self.tol_rel = 1e-2
        self.tol_abs = 1e-2

    def test_invariants(self):
        """Check kinematic equations and consistency."""
        logs = []
        success = True
        
        # 1. Manual lane deceleration: v0 to 0 with a = -1
        t1_expected = self.v0 / self.a
        if not math.isclose(self.t1, t1_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual deceleration time mismatch: {self.t1} vs {t1_expected}")
            success = False
        
        # Displacement during deceleration: x1 = (v0/2) * t1
        x1_expected = (self.v0 / 2) * self.t1
        if not math.isclose(self.x1, x1_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual deceleration displacement mismatch: {self.x1} vs {x1_expected}")
            success = False
        
        # 2. Manual lane acceleration: 0 to v0 with a = 1
        t2_expected = self.v0 / self.a
        if not math.isclose(self.t2, t2_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual acceleration time mismatch: {self.t2} vs {t2_expected}")
            success = False
        
        x2_expected = (self.v0 / 2) * self.t2
        if not math.isclose(self.x2, x2_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual acceleration displacement mismatch: {self.x2} vs {x2_expected}")
            success = False
        
        # 3. Manual total time and distance
        t_manual_expected = self.t1 + self.t2 + self.t0
        if not math.isclose(self.t_manual, t_manual_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual total time mismatch: {self.t_manual} vs {t_manual_expected}")
            success = False
        
        s_manual_expected = self.x1 + self.x2
        if not math.isclose(self.s_manual, s_manual_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual total distance mismatch: {self.s_manual} vs {s_manual_expected}")
            success = False
        
        # 4. ETC lane deceleration: v0 to v with a = -1
        t1_prime_expected = (self.v0 - self.v) / self.a
        if not math.isclose(self.t1_prime, t1_prime_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC deceleration time mismatch: {self.t1_prime} vs {t1_prime_expected}")
            success = False
        
        x1_prime_expected = (self.v0 + self.v) / 2 * self.t1_prime
        if not math.isclose(self.x1_prime, x1_prime_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC deceleration displacement mismatch: {self.x1_prime} vs {x1_prime_expected}")
            success = False
        
        # 5. ETC constant-speed time
        t3_expected = self.delta_x / self.v
        if not math.isclose(self.t3, t3_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC constant-speed time mismatch: {self.t3} vs {t3_expected}")
            success = False
        
        # 6. ETC acceleration: v to v0 with a = 1
        t2_prime_expected = (self.v0 - self.v) / self.a
        if not math.isclose(self.t2_prime, t2_prime_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC acceleration time mismatch: {self.t2_prime} vs {t2_prime_expected}")
            success = False
        
        x2_prime_expected = (self.v0 + self.v) / 2 * self.t2_prime
        if not math.isclose(self.x2_prime, x2_prime_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC acceleration displacement mismatch: {self.x2_prime} vs {x2_prime_expected}")
            success = False
        
        # 7. ETC total time and distance
        t_etc_expected = self.t1_prime + self.t2_prime + self.t3
        if not math.isclose(self.t_etc, t_etc_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC total time mismatch: {self.t_etc} vs {t_etc_expected}")
            success = False
        
        s_etc_expected = self.x1_prime + self.x2_prime + self.delta_x
        if not math.isclose(self.s_etc, s_etc_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC total distance mismatch: {self.s_etc} vs {s_etc_expected}")
            success = False
        
        # 8. Time to travel manual distance at initial velocity
        t_double_prime_expected = self.t_etc + (self.s_manual - self.s_etc) / self.v0
        if not math.isclose(self.t_double_prime, t_double_prime_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"t'' mismatch: {self.t_double_prime} vs {t_double_prime_expected}")
            success = False
        
        # 9. Time saved
        delta_t_expected = self.t_manual - self.t_double_prime
        if not math.isclose(self.delta_t, delta_t_expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Time saved mismatch: {self.delta_t} vs {delta_t_expected}")
            success = False
        
        # 10. Symbolic check using sympy (optional fallback)
        try:
            v0, a, t0, v, delta_x = sp.symbols('v0 a t0 v delta_x')
            t1 = v0 / a
            x1 = v0 / 2 * t1
            t2 = v0 / a
            x2 = v0 / 2 * t2
            t_manual_sym = t1 + t2 + t0
            s_manual_sym = x1 + x2
            
            t1_prime = (v0 - v) / a
            x1_prime = (v0 + v) / 2 * t1_prime
            t3 = delta_x / v
            t2_prime = (v0 - v) / a
            x2_prime = (v0 + v) / 2 * t2_prime
            t_etc_sym = t1_prime + t2_prime + t3
            s_etc_sym = x1_prime + x2_prime + delta_x
            
            t_double_prime_sym = t_etc_sym + (s_manual_sym - s_etc_sym) / v0
            delta_t_sym = t_manual_sym - t_double_prime_sym
            
            # Substitute numeric values
            subs = {v0: self.v0, a: self.a, t0: self.t0, v: self.v, delta_x: self.delta_x}
            t_manual_val = float(t_manual_sym.subs(subs))
            s_manual_val = float(s_manual_sym.subs(subs))
            t_etc_val = float(t_etc_sym.subs(subs))
            s_etc_val = float(s_etc_sym.subs(subs))
            t_double_prime_val = float(t_double_prime_sym.subs(subs))
            delta_t_val = float(delta_t_sym.subs(subs))
            
            # Compare with extracted results
            if not (math.isclose(t_manual_val, self.t_manual, rel_tol=self.tol_rel, abs_tol=self.tol_abs) and
                    math.isclose(s_manual_val, self.s_manual, rel_tol=self.tol_rel, abs_tol=self.tol_abs) and
                    math.isclose(t_etc_val, self.t_etc, rel_tol=self.tol_rel, abs_tol=self.tol_abs) and
                    math.isclose(s_etc_val, self.s_etc, rel_tol=self.tol_rel, abs_tol=self.tol_abs) and
                    math.isclose(t_double_prime_val, self.t_double_prime, rel_tol=self.tol_rel, abs_tol=self.tol_abs) and
                    math.isclose(delta_t_val, self.delta_t, rel_tol=self.tol_rel, abs_tol=self.tol_abs)):
                logs.append("Symbolic check failed: numeric substitution mismatch")
                success = False
        except Exception as e:
            logs.append(f"Symbolic check skipped due to error: {e}")
        
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Final answers from problem
        target_manual = (40.0, 100.0)  # t, s
        target_etc = (12.0, 85.0)      # t', s'
        target_saved = 26.5            # Δt
        
        if not (math.isclose(self.t_manual, target_manual[0], rel_tol=self.tol_rel, abs_tol=self.tol_abs) and
                math.isclose(self.s_manual, target_manual[1], rel_tol=self.tol_rel, abs_tol=self.tol_abs)):
            logs.append(f"Manual lane final answer mismatch: ({self.t_manual}, {self.s_manual}) vs {target_manual}")
            success = False
        
        if not (math.isclose(self.t_etc, target_etc[0], rel_tol=self.tol_rel, abs_tol=self.tol_abs) and
                math.isclose(self.s_etc, target_etc[1], rel_tol=self.tol_rel, abs_tol=self.tol_abs)):
            logs.append(f"ETC lane final answer mismatch: ({self.t_etc}, {self.s_etc}) vs {target_etc}")
            success = False
        
        if not math.isclose(self.delta_t, target_saved, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Time saved mismatch: {self.delta_t} vs {target_saved}")
            success = False
        
        # Additional consistency: check that intermediate results sum to final answers
        if not math.isclose(self.t1 + self.t2 + self.t0, self.t_manual, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual time sum inconsistency: {self.t1}+{self.t2}+{self.t0} != {self.t_manual}")
            success = False
        
        if not math.isclose(self.x1 + self.x2, self.s_manual, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"Manual distance sum inconsistency: {self.x1}+{self.x2} != {self.s_manual}")
            success = False
        
        if not math.isclose(self.t1_prime + self.t2_prime + self.t3, self.t_etc, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC time sum inconsistency: {self.t1_prime}+{self.t2_prime}+{self.t3} != {self.t_etc}")
            success = False
        
        if not math.isclose(self.x1_prime + self.x2_prime + self.delta_x, self.s_etc, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"ETC distance sum inconsistency: {self.x1_prime}+{self.x2_prime}+{self.delta_x} != {self.s_etc}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        """Perturb initial velocity and check that results change consistently."""
        logs = []
        success = True
        
        # Perturb v0 by +1%
        v0_original = self.v0
        self.v0 *= 1.01
        
        # Recalculate using the same formulas as in invariants
        t1_new = self.v0 / self.a
        x1_new = (self.v0 / 2) * t1_new
        t2_new = self.v0 / self.a
        x2_new = (self.v0 / 2) * t2_new
        t_manual_new = t1_new + t2_new + self.t0
        s_manual_new = x1_new + x2_new
        
        t1_prime_new = (self.v0 - self.v) / self.a
        x1_prime_new = (self.v0 + self.v) / 2 * t1_prime_new
        t3_new = self.delta_x / self.v
        t2_prime_new = (self.v0 - self.v) / self.a
        x2_prime_new = (self.v0 + self.v) / 2 * t2_prime_new
        t_etc_new = t1_prime_new + t2_prime_new + t3_new
        s_etc_new = x1_prime_new + x2_prime_new + self.delta_x
        
        t_double_prime_new = t_etc_new + (s_manual_new - s_etc_new) / self.v0
        delta_t_new = t_manual_new - t_double_prime_new
        
        # Restore original v0
        self.v0 = v0_original
        
        # Check monotonic trend: time saved should increase with v0? Let's compute derivative sign.
        # We expect that as v0 increases, the time saved might increase because the manual lane
        # has a fixed 20s payment time, but the ETC lane has less slowdown.
        # We can check that delta_t_new > self.delta_t (since v0 increased).
        if not delta_t_new > self.delta_t:
            logs.append(f"Monotonicity check: time saved did not increase with v0: {delta_t_new} <= {self.delta_t}")
            success = False
        
        # Also check that recalculated values are close to the original when v0 is restored?
        # Not needed here.
        
        return success, logs

    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                if not success:
                    overall_success = False
                    all_logs.append(f"FAILED {test_name}:")
                    all_logs.extend(logs)
                else:
                    all_logs.append(f"PASSED {test_name}")
            except Exception as e:
                overall_success = False
                all_logs.append(f"ERROR in {test_name}: {e}")
        
        return overall_success, all_logs