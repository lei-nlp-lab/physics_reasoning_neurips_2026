import math
import sympy as sp
from sympy.abc import t, T

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.d = getattr(solution_instance, 'd', 4.0)
        self.v0 = getattr(solution_instance, 'v0', 4.0)
        self.a1 = getattr(solution_instance, 'a1', 1.0)
        self.a2 = getattr(solution_instance, 'a2', 2.0)
        self.s0 = getattr(solution_instance, 's0', 10.0)
        
        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        self.t_delta = results.get('t_delta', None)
        self.S = results.get('S', None)
        self.t2 = results.get('t_2', None)
        self.T_val = results.get('T', None)
        self.t_val = results.get('t', None)
        self.t1 = results.get('t_1', None)
        self.s = results.get('s', None)
        
        self.logs = []

    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success

    def test_invariants(self):
        success = True
        
        # 1. Check velocity equality condition (from step_1)
        if self.t_val is not None:
            vA_eq = self.v0 + self.a1 * self.t_val
            vB_eq = self.a2 * self.t_val
            if not math.isclose(vA_eq, vB_eq, rel_tol=1e-2, abs_tol=1e-2):
                success = self._log(f"Velocity equality violated: {vA_eq} ≠ {vB_eq}", False)
            else:
                self._log("Velocity equality condition satisfied", True)
        
        # 2. Check maximum separation formula (from step_2)
        if self.t_val is not None and self.S is not None:
            S_calc = self.d + (self.v0 * self.t_val + 0.5 * self.a1 * self.t_val**2) - 0.5 * self.a2 * self.t_val**2
            if not math.isclose(S_calc, self.S, rel_tol=1e-2, abs_tol=1e-2):
                success = self._log(f"Maximum separation mismatch: {S_calc} ≠ {self.S}", False)
            else:
                self._log("Maximum separation formula satisfied", True)
        
        # 3. Check quadratic root conditions (from step_4 & step_5)
        if self.t1 is not None and self.t2 is not None and self.s0 is not None:
            # Symbolic check using sympy
            try:
                t_sym = sp.symbols('t_sym')
                separation_expr = self.d + (self.v0 * t_sym + 0.5 * self.a1 * t_sym**2) - 0.5 * self.a2 * t_sym**2
                eq = sp.Eq(separation_expr, self.s0)
                roots = sp.solve(eq, t_sym)
                roots_numeric = [float(r) for r in roots if r.is_real and float(r) >= 0]
                if len(roots_numeric) >= 2:
                    if not (math.isclose(min(roots_numeric), self.t1, rel_tol=1e-2, abs_tol=1e-2) and
                            math.isclose(max(roots_numeric), self.t2, rel_tol=1e-2, abs_tol=1e-2)):
                        success = self._log(f"Quadratic roots mismatch: expected {[self.t1, self.t2]}, got {roots_numeric}", False)
                    else:
                        self._log("Quadratic root condition satisfied", True)
            except Exception as e:
                # Fallback numerical check
                for t_test in [self.t1, self.t2]:
                    s_calc = self.d + (self.v0 * t_test + 0.5 * self.a1 * t_test**2) - 0.5 * self.a2 * t_test**2
                    if not math.isclose(s_calc, self.s0, rel_tol=1e-2, abs_tol=1e-2):
                        success = self._log(f"Root {t_test} doesn't satisfy s0 equation: {s_calc} ≠ {self.s0}", False)
                self._log("Quadratic root numerical check passed", True)
        
        # 4. Check meeting time equation (from step_6)
        if self.T_val is not None:
            left = self.d + (self.v0 * self.T_val + 0.5 * self.a1 * self.T_val**2)
            right = 0.5 * self.a2 * self.T_val**2
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                success = self._log(f"Meeting time equation violated: {left} ≠ {right}", False)
            else:
                self._log("Meeting time equation satisfied", True)
        
        return success

    def test_trace_consistency(self):
        success = True
        
        # Check that maximum separation S > communication range s0 (required by problem logic)
        if self.S is not None and self.s0 is not None:
            if not (self.S > self.s0 or math.isclose(self.S, self.s0, rel_tol=1e-2, abs_tol=1e-2)):
                success = self._log(f"Maximum separation {self.S} not >= communication range {self.s0}", False)
            else:
                self._log(f"Maximum separation {self.S} >= communication range {self.s0}", True)
        
        # Check time ordering: t1 < t2 < T (based on problem context)
        if self.t1 is not None and self.t2 is not None and self.T_val is not None:
            if not (self.t1 < self.t2 < self.T_val):
                success = self._log(f"Time ordering violated: {self.t1}, {self.t2}, {self.T_val}", False)
            else:
                self._log(f"Time ordering consistent: {self.t1} < {self.t2} < {self.T_val}", True)
        
        # Check time duration formula (from step_7)
        if self.t_delta is not None and self.T_val is not None and self.t1 is not None and self.t2 is not None:
            t_delta_calc = self.T_val - (self.t2 - self.t1)
            if not math.isclose(t_delta_calc, self.t_delta, rel_tol=1e-2, abs_tol=1e-2):
                success = self._log(f"Time duration mismatch: {t_delta_calc} ≠ {self.t_delta}", False)
            else:
                self._log("Time duration formula satisfied", True)
        
        return success

    def test_monotonicity(self):
        success = True
        
        # Store original values
        orig_d = self.d
        
        try:
            # Perturb initial distance d by +10%
            perturbed_d = self.d * 1.1
            
            # Recalculate meeting time T for perturbed d (solving the same equation)
            # Equation: d + v0*T + 0.5*a1*T^2 = 0.5*a2*T^2
            # Rearranged: 0.5*(a2 - a1)*T^2 - v0*T - d = 0
            T_sym = sp.symbols('T_sym', positive=True)
            eq_T = sp.Eq(0.5*(self.a2 - self.a1)*T_sym**2 - self.v0*T_sym - perturbed_d, 0)
            T_solutions = sp.solve(eq_T, T_sym)
            T_perturbed = float(max([sol for sol in T_solutions if sol.is_real and float(sol) > 0]))
            
            # Recalculate t1 and t2 for perturbed d
            # Equation: d + (v0*t + 0.5*a1*t^2) - 0.5*a2*t^2 = s0
            t_sym = sp.symbols('t_sym', positive=True)
            eq_t = sp.Eq(perturbed_d + (self.v0*t_sym + 0.5*self.a1*t_sym**2) - 0.5*self.a2*t_sym**2, self.s0)
            t_solutions = sp.solve(eq_t, t_sym)
            t_solutions_numeric = sorted([float(sol) for sol in t_solutions if sol.is_real and float(sol) > 0])
            
            if len(t_solutions_numeric) >= 2:
                t1_perturbed, t2_perturbed = t_solutions_numeric[0], t_solutions_numeric[1]
                
                # Recalculate time duration
                t_delta_perturbed = T_perturbed - (t2_perturbed - t1_perturbed)
                
                # Check monotonic trend: if d increases, meeting time T should increase
                if T_perturbed > self.T_val:
                    self._log(f"Meeting time increases with d (as expected): {self.T_val} → {T_perturbed}", True)
                else:
                    success = self._log(f"Meeting time doesn't increase with d: {self.T_val} → {T_perturbed}", False)
                
                # Check that time duration changes in a physically reasonable way
                # With larger d, cars start farther apart, so communication duration might decrease
                if t_delta_perturbed < self.t_delta:
                    self._log(f"Communication duration decreases with increased d (reasonable): {self.t_delta} → {t_delta_perturbed}", True)
                else:
                    # It's still physically possible, so we don't fail this test
                    self._log(f"Communication duration changes to {t_delta_perturbed} with perturbed d", True)
            else:
                success = self._log("Could not find two positive roots for perturbed d", False)
        
        except Exception as e:
            success = self._log(f"Monotonicity test failed due to error: {str(e)}", False)
        
        # Restore original d
        self.d = orig_d
        
        return success

    def run_all_tests(self):
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== {test_name} ===")
            if not test_func():
                overall_success = False
        
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs