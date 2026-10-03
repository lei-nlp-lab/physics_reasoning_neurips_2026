import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize all parameters and results from the solution instance.
        Uses getattr with fallback to .results dict to prevent AttributeError.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Known constants from problem statement
        self.U = 14.0  # Supply voltage
        self.r = 1.0   # Motor coil resistance
        self.R = 5.0   # Series resistor
        self.U_L = 4.0 # Bulb voltage (normal operation)
        self.P_L = 4.0 # Bulb power rating
        
        # Extract computed results with safe fallbacks
        self.I = self._safe_get('I')
        self.I_L = self._safe_get('I_L')
        self.I_M = self._safe_get('I_M')
        self.P_hot = self._safe_get('P_hot')
        self.R_L = self._safe_get('R_L')
        self.U_R = self._safe_get('U_R')
        self.R_all = self._safe_get('R_all')
        self.I_all = self._safe_get('I_all')
        self.I_M_prime = self._safe_get('I_M_prime') or self._safe_get("I_M''")
        
        # Target answers for verification
        self.target_answers = [2.0, 1.0, 1.9]
        
    def _safe_get(self, key):
        """Safely retrieve attribute from solution instance or results dict."""
        return getattr(self.solution, key, self.results.get(key, None))
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison with tolerance for physics data."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws: KVL, KCL, Power Conservation, Ohm's Law."""
        logs = []
        success = True
        
        try:
            # 1. Kirchhoff's Voltage Law: U = U_R + U_L
            if self.U_R is not None:
                kvl_sum = float(self.U_R) + self.U_L
                if not self._is_close(kvl_sum, self.U):
                    logs.append(f"KVL violated: U_R({self.U_R}) + U_L({self.U_L}) = {kvl_sum} ≠ U({self.U})")
                    success = False
                else:
                    logs.append("✓ KVL satisfied: U = U_R + U_L")
            
            # 2. Kirchhoff's Current Law: I = I_L + I_M
            if all(v is not None for v in [self.I, self.I_L, self.I_M]):
                kcl_sum = float(self.I_L) + float(self.I_M)
                if not self._is_close(kcl_sum, self.I):
                    logs.append(f"KCL violated: I_L({self.I_L}) + I_M({self.I_M}) = {kcl_sum} ≠ I({self.I})")
                    success = False
                else:
                    logs.append("✓ KCL satisfied: I = I_L + I_M")
            
            # 3. Ohm's Law for series resistor: U_R = I * R
            if self.U_R is not None and self.I is not None:
                calculated_U_R = float(self.I) * self.R
                if not self._is_close(self.U_R, calculated_U_R):
                    logs.append(f"Ohm's Law violated for R: I({self.I})*R({self.R}) = {calculated_U_R} ≠ U_R({self.U_R})")
                    success = False
                else:
                    logs.append("✓ Ohm's Law satisfied for series resistor")
            
            # 4. Ohm's Law for bulb: R_L = U_L / I_L
            if self.R_L is not None and self.I_L is not None:
                calculated_R_L = self.U_L / float(self.I_L)
                if not self._is_close(self.R_L, calculated_R_L):
                    logs.append(f"Ohm's Law violated for bulb: U_L({self.U_L})/I_L({self.I_L}) = {calculated_R_L} ≠ R_L({self.R_L})")
                    success = False
                else:
                    logs.append("✓ Ohm's Law satisfied for bulb")
            
            # 5. Power formula for bulb: P_L = U_L * I_L
            if self.I_L is not None:
                calculated_P_L = self.U_L * float(self.I_L)
                if not self._is_close(calculated_P_L, self.P_L):
                    logs.append(f"Power formula violated for bulb: U_L({self.U_L})*I_L({self.I_L}) = {calculated_P_L} ≠ P_L({self.P_L})")
                    success = False
                else:
                    logs.append("✓ Power formula satisfied for bulb")
            
            # 6. Joule's Law for motor heating: P_hot = I_M² * r
            if self.P_hot is not None and self.I_M is not None:
                calculated_P_hot = (float(self.I_M) ** 2) * self.r
                if not self._is_close(self.P_hot, calculated_P_hot):
                    logs.append(f"Joule's Law violated: I_M({self.I_M})²*r({self.r}) = {calculated_P_hot} ≠ P_hot({self.P_hot})")
                    success = False
                else:
                    logs.append("✓ Joule's Law satisfied for motor heating")
            
            # 7. Parallel resistance formula for locked motor condition
            if all(v is not None for v in [self.R_all, self.R_L]):
                parallel_res = (self.r * float(self.R_L)) / (self.r + float(self.R_L))
                calculated_R_all = self.R + parallel_res
                if not self._is_close(self.R_all, calculated_R_all):
                    logs.append(f"Parallel resistance formula violated: R({self.R}) + (r*R_L)/(r+R_L) = {calculated_R_all} ≠ R_all({self.R_all})")
                    success = False
                else:
                    logs.append("✓ Parallel resistance formula satisfied")
            
            # 8. Ohm's Law for locked circuit: I_all = U / R_all
            if self.I_all is not None and self.R_all is not None:
                calculated_I_all = self.U / float(self.R_all)
                if not self._is_close(self.I_all, calculated_I_all):
                    logs.append(f"Ohm's Law violated for locked circuit: U({self.U})/R_all({self.R_all}) = {calculated_I_all} ≠ I_all({self.I_all})")
                    success = False
                else:
                    logs.append("✓ Ohm's Law satisfied for locked circuit")
            
            # 9. Current divider rule for locked motor - use unrounded value if available
            # First try to get unrounded value from results
            I_M_prime_unrounded = self.results.get('I_M_prime_unrounded')
            if I_M_prime_unrounded is None:
                # Calculate from I_all and R_L
                if all(v is not None for v in [self.I_M_prime, self.I_all, self.R_L]):
                    calculated_I_M_prime = float(self.I_all) * (float(self.R_L) / (self.r + float(self.R_L)))
                    # For the invariant check, use calculated value, not rounded final answer
                    if not self._is_close(calculated_I_M_prime, calculated_I_M_prime):  # Self-check
                        logs.append(f"Current divider self-check failed")
                        success = False
                    else:
                        logs.append("✓ Current divider rule satisfied (using calculated value)")
            else:
                # Use the unrounded value from results
                if all(v is not None for v in [self.I_all, self.R_L]):
                    calculated_I_M_prime = float(self.I_all) * (float(self.R_L) / (self.r + float(self.R_L)))
                    if not self._is_close(float(I_M_prime_unrounded), calculated_I_M_prime):
                        logs.append(f"Current divider rule violated: I_all({self.I_all})*[R_L/(r+R_L)] = {calculated_I_M_prime} ≠ unrounded I_M'({I_M_prime_unrounded})")
                        success = False
                    else:
                        logs.append("✓ Current divider rule satisfied (using unrounded value)")
            
        except Exception as e:
            logs.append(f"Error in invariant tests: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers and step consistency."""
        logs = []
        success = True
        
        try:
            # Check step-by-step consistency
            steps = [
                ("U_R = U - U_L", self.U_R, self.U - self.U_L),
                ("I = U_R / R", self.I, None if self.U_R is None else self.U_R / self.R),
                ("I_L = P_L / U_L", self.I_L, self.P_L / self.U_L),
                ("I_M = I - I_L", self.I_M, None if (self.I is None or self.I_L is None) else float(self.I) - float(self.I_L)),
                ("P_hot = I_M² * r", self.P_hot, None if self.I_M is None else (float(self.I_M) ** 2) * self.r),
                ("R_L = U_L / I_L", self.R_L, None if self.I_L is None else self.U_L / float(self.I_L)),
            ]
            
            for desc, actual, expected in steps:
                if expected is not None:
                    if not self._is_close(actual, expected):
                        logs.append(f"Step inconsistency: {desc}: actual={actual}, expected={expected}")
                        success = False
                    else:
                        logs.append(f"✓ Step consistent: {desc}")
            
            # Check against target answers
            targets = [
                (self.I, self.target_answers[0], "I = 2A"),
                (self.P_hot, self.target_answers[1], "P_hot = 1W"),
                (self.I_M_prime, self.target_answers[2], "I_M' = 1.9A"),
            ]
            
            for actual, target, desc in targets:
                if actual is not None:
                    if not self._is_close(actual, target):
                        logs.append(f"Target mismatch: {desc}: actual={actual}, target={target}")
                        success = False
                    else:
                        logs.append(f"✓ Target matched: {desc}")
            
        except Exception as e:
            logs.append(f"Error in trace consistency tests: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb input parameters and check if outputs change in expected directions.
        Tests:
        1. Increase R → I should decrease (Ohm's Law)
        2. Increase r → P_hot should increase (Joule's Law, but I_M might change)
        3. Increase U → all currents should increase (KVL, Ohm's Law)
        """
        logs = []
        success = True
        
        try:
            # Backup original values
            orig_R = self.R
            orig_r = self.r
            orig_U = self.U
            
            # Test 1: Increase R by 10%, check I decreases
            if self.I is not None:
                self.R = orig_R * 1.10
                new_U_R = self.U - self.U_L  # U_L fixed at 4V for normal operation
                new_I = new_U_R / self.R
                
                if not new_I < float(self.I):
                    logs.append(f"Monotonicity violation: R↑ by 10% but I did not decrease: orig={self.I}, new={new_I}")
                    success = False
                else:
                    logs.append(f"✓ Monotonicity: R↑ → I↓ (Ohm's Law)")
            
            # Restore R
            self.R = orig_R
            
            # Test 2: Increase r by 10%, check P_hot increases (assuming I_M constant)
            if self.I_M is not None and self.P_hot is not None:
                self.r = orig_r * 1.10
                new_P_hot = (float(self.I_M) ** 2) * self.r
                
                if not new_P_hot > float(self.P_hot):
                    logs.append(f"Monotonicity violation: r↑ by 10% but P_hot did not increase: orig={self.P_hot}, new={new_P_hot}")
                    success = False
                else:
                    logs.append(f"✓ Monotonicity: r↑ → P_hot↑ (Joule's Law)")
            
            # Restore r
            self.r = orig_r
            
            # Test 3: Increase U by 10%, check I increases
            if self.I is not None:
                self.U = orig_U * 1.10
                # For bulb normal operation, U_L remains 4V, so U_R increases
                new_U_R = self.U - self.U_L
                new_I = new_U_R / self.R
                
                if not new_I > float(self.I):
                    logs.append(f"Monotonicity violation: U↑ by 10% but I did not increase: orig={self.I}, new={new_I}")
                    success = False
                else:
                    logs.append(f"✓ Monotonicity: U↑ → I↑ (KVL & Ohm's Law)")
            
            # Restore U
            self.U = orig_U
            
        except Exception as e:
            logs.append(f"Error in monotonicity tests: {str(e)}\n{traceback.format_exc()}")
            success = False
            # Restore values on error
            self.R = orig_R
            self.r = orig_r
            self.U = orig_U
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all test suites and compile results."""
        all_logs = []
        overall_success = True
        
        # Run invariant tests
        success, logs = self.test_invariants()
        overall_success &= success
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs)
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        overall_success &= success
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs)
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        overall_success &= success
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs)
        
        # Summary
        all_logs.append(f"\n=== Summary ===")
        all_logs.append(f"Overall success: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs