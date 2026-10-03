import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.logs = []
        
        # Given constants (safe fallback)
        self.v1 = 80.0
        self.t1 = 1.5
        self.n = 100
        self.l = 1.0
        self.M = 10.0
        self.m = 5.0
        self.R0 = 9.5
        self.B = 0.1
        
        # Extract results safely
        self.I = self._get_value('I')
        self.R = self._get_value('R')
        self.t_3 = self._get_value('t_3')
        self.a = self._get_value('a')
        self.F_1 = self._get_value('F_1')
        self.a_prime = self._get_value('a_prime')
        self.t_2 = self._get_value('t_2')
        self.s = self._get_value('s')
        self.delta_q = self._get_value('delta_q')
        
    def _get_value(self, key):
        """Safely retrieve a value from solution_instance."""
        # First check results dictionary
        if hasattr(self.solution, 'results'):
            # Try direct key
            val = self.solution.results.get(key, None)
            if val is not None:
                return self._to_float(val)
            
            # Try alternative keys
            alt_map = {
                'a_prime': ["a'", "a′", "a\\'", "a_prime"],
                'delta_q': ["Δq", "\\Delta q", "Deltaq", "delta_q"],
                'I_prime': ["I'", "I′", "I\\'", "I_prime"],
                'F_1_prime': ["F_1'", "F_1′", "F_1\\'", "F_1_prime"],
            }
            if key in alt_map:
                for alt in alt_map[key]:
                    val = self.solution.results.get(alt, None)
                    if val is not None:
                        return self._to_float(val)
        
        # Then check direct attributes
        val = getattr(self.solution, key, None)
        if val is not None:
            return self._to_float(val)
        
        return None
    
    def _to_float(self, val):
        """Convert a value to float, handling strings and sympy expressions."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                # Try direct conversion
                return float(val)
            except ValueError:
                # Try sympy evaluation for expressions like "(sqrt(5)+3)/2"
                try:
                    import sympy as sp
                    expr = sp.sympify(val)
                    return float(expr.evalf())
                except:
                    pass
        # If it's already a float or convertible via float(), try
        try:
            return float(val)
        except:
            return val  # Return as-is if conversion fails
    
    def test_invariants(self):
        """Verify fundamental physics laws and consistency between steps."""
        logs = []
        success = True
        try:
            # 1. Acceleration from 0 to t1: a = v1 / t1
            if self.a is not None:
                expected_a = self.v1 / self.t1
                if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a = {self.a} ≠ v1/t1 = {expected_a}")
                    success = False
            
            # 2. Ampere force: F1 = n * B * I * l
            if self.F_1 is not None and self.I is not None:
                expected_F1 = self.n * self.B * self.I * self.l
                if not math.isclose(self.F_1, expected_F1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: F1 = {self.F_1} ≠ nBIl = {expected_F1}")
                    success = False
            
            # 3. Newton's second law: F1 = (M + m) * a
            if self.F_1 is not None and self.a is not None:
                expected_F1_newton = (self.M + self.m) * self.a
                if not math.isclose(self.F_1, expected_F1_newton, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: F1 = {self.F_1} ≠ (M+m)a = {expected_F1_newton}")
                    success = False
            
            # 4. Circuit condition: n²l²B²/(R0+R) = 10
            if self.R is not None:
                lhs = self.n**2 * self.l**2 * self.B**2 / (self.R0 + self.R)
                if not math.isclose(lhs, 10.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: n²l²B²/(R0+R) = {lhs} ≠ 10")
                    success = False
            
            # 5. Deceleration from t1 to t3: a' = 800 / m
            if self.a_prime is not None:
                expected_a_prime = 800.0 / self.m
                if not math.isclose(self.a_prime, expected_a_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a' = {self.a_prime} ≠ 800/m = {expected_a_prime}")
                    success = False
            
            # 6. Time to stop forward motion: t2 - t1 = v1 / a'
            if self.t_2 is not None and self.a_prime is not None:
                t2_minus_t1 = self.t_2 - self.t1
                expected_t2_minus_t1 = self.v1 / self.a_prime
                if not math.isclose(t2_minus_t1, expected_t2_minus_t1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t2-t1 = {t2_minus_t1} ≠ v1/a' = {expected_t2_minus_t1}")
                    success = False
            
            # 7. Displacement up to t2: s = 0.5 * v1 * t2
            if self.s is not None and self.t_2 is not None:
                expected_s = 0.5 * self.v1 * self.t_2
                if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: s = {self.s} ≠ 0.5*v1*t2 = {expected_s}")
                    success = False
            
            # 8. Quadratic condition for t3: (t3 - t2)² + (t3 - t2) - 1 = 0
            if self.t_3 is not None and self.t_2 is not None:
                x = self.t_3 - self.t_2
                quadratic_val = x**2 + x - 1
                if not math.isclose(quadratic_val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: (t3-t2)²+(t3-t2)-1 = {quadratic_val} ≠ 0")
                    success = False
            
            if success:
                logs.append("All invariants passed.")
            else:
                logs.append("Some invariants failed.")
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
            success = False
        self.logs.append(("invariants", logs))
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results and final answers."""
        logs = []
        success = True
        try:
            # Check current I
            if self.I is not None:
                expected_I = (self.M + self.m) * self.v1 / (self.n * self.l * self.B * self.t1)
                if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: I = {self.I} ≠ (M+m)v1/(n l B t1) = {expected_I}")
                    success = False
                else:
                    logs.append(f"OK: I = {self.I} A matches formula.")
            
            # Check coil resistance R
            if self.R is not None:
                expected_R = self.n**2 * self.l**2 * self.B**2 / 10.0 - self.R0
                if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: R = {self.R} Ω ≠ [n²l²B²/10 - R0] = {expected_R} Ω")
                    success = False
                else:
                    logs.append(f"OK: R = {self.R} Ω matches formula.")
            
            # Check time t3
            if self.t_3 is not None:
                expected_t3 = (math.sqrt(5) + 3) / 2
                if not math.isclose(self.t_3, expected_t3, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t3 = {self.t_3} s ≠ (√5+3)/2 ≈ {expected_t3} s")
                    success = False
                else:
                    logs.append(f"OK: t3 = {self.t_3} s matches target expression.")
            
            if success:
                logs.append("All trace consistency checks passed.")
            else:
                logs.append("Some trace consistency checks failed.")
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
            success = False
        self.logs.append(("trace", logs))
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []
        try:
            # Perturb mass m by +1 kg
            perturbed_m = self.m + 1.0
            
            # Effect on current I: I ∝ (M+m)
            if self.I is not None:
                new_I = (self.M + perturbed_m) * self.v1 / (self.n * self.l * self.B * self.t1)
                if new_I > self.I:
                    logs.append(f"MONOTONICITY: Increasing m → I increases ({self.I} → {new_I}), as expected.")
                else:
                    logs.append(f"WARNING: Increasing m did not increase I as expected.")
            
            # Effect on deceleration a': a' = 800/m (inverse relationship)
            if self.a_prime is not None:
                new_a_prime = 800.0 / perturbed_m
                if new_a_prime < self.a_prime:
                    logs.append(f"MONOTONICITY: Increasing m → a' decreases ({self.a_prime} → {new_a_prime}), as expected.")
                else:
                    logs.append(f"WARNING: Increasing m did not decrease a' as expected.")
            
            # Perturb magnetic field B (optional)
            perturbed_B = self.B * 1.1  # +10%
            if self.I is not None:
                # I ∝ 1/B from formula? Actually I = (M+m)v1/(n l B t1), so I ∝ 1/B
                new_I_B = (self.M + self.m) * self.v1 / (self.n * self.l * perturbed_B * self.t1)
                if new_I_B < self.I:
                    logs.append(f"MONOTONICITY: Increasing B → I decreases ({self.I} → {new_I_B}), as expected.")
                else:
                    logs.append(f"WARNING: Increasing B did not decrease I as expected.")
            
            if not logs:
                logs.append("Monotonicity checks passed.")
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
        self.logs.append(("monotonicity", logs))
        return True  # Monotonicity tests are advisory, not fatal
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        success = True
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            self.test_monotonicity()  # logs only, doesn't affect success
            if not inv_ok or not trace_ok:
                success = False
        except Exception as e:
            self.logs.append(("error", [f"Critical error in run_all_tests: {e}"]))
            success = False
        # Flatten logs for output
        flat_logs = []
        for category, log_list in self.logs:
            flat_logs.extend(log_list)
        return success, flat_logs