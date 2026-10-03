import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr and .get() with defaults
        self.solution = solution_instance
        
        # Extract inputs and results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Wave parameters
        self.lambda_A = self.results.get('λ_A', 0.8)
        self.lambda_B = self.results.get('λ_B', 0.4)
        self.T_A = self.results.get('T_A', 0.8)
        self.T_B = self.results.get('T_B', 0.4)
        self.v = self.results.get('v', 1.0)
        
        # Positions
        self.x_P = -0.2
        self.x_Q = 0.1
        self.x_M = 0.3
        self.x_A_source = 0.9
        self.x_B_source = -0.6
        
        # Distances
        self.x_PM = self.results.get('x_PM', 0.5)
        
        # Times
        self.t_PM = self.results.get('t', 0.5)
        self.t_A_crest = self.results.get('t_A', 0.8)
        self.t_B_crest = self.results.get('t_B', 0.8)
        
        # Final answers
        self.answer1 = getattr(solution_instance, 'answer1', "0.5s")
        self.answer2 = getattr(solution_instance, 'answer2', "0.8s, 1.6s, 2.4s")
        
        # Parse answer2 into list of floats
        if isinstance(self.answer2, str):
            times_str = self.answer2.replace('s', '').replace(',', ' ').split()
            self.times_M = [float(t) for t in times_str]
        elif isinstance(self.answer2, list):
            self.times_M = self.answer2
        else:
            self.times_M = [0.8, 1.6, 2.4]
        
        self.logs = []
    
    def _log(self, message: str, success: bool = True) -> None:
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def test_invariants(self) -> None:
        """Test fundamental physical invariants."""
        
        # 1. Wave speed consistency check (v = λ_A / T_A)
        expected_v = self.lambda_A / self.T_A
        if math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Wave speed consistent: v = {self.v} ≈ λ_A/T_A = {expected_v}")
        else:
            self._log(f"Wave speed mismatch: v = {self.v}, λ_A/T_A = {expected_v}", False)
        
        # 2. Alternative wave speed check (v = λ_B / T_B)
        expected_v_B = self.lambda_B / self.T_B
        if math.isclose(self.v, expected_v_B, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Wave speed consistent with wave B: v = {self.v} ≈ λ_B/T_B = {expected_v_B}")
        else:
            self._log(f"Wave speed mismatch with wave B: v = {self.v}, λ_B/T_B = {expected_v_B}", False)
        
        # 3. Distance calculation check (x_PM = x_M - x_P)
        expected_x_PM = self.x_M - self.x_P
        if math.isclose(self.x_PM, expected_x_PM, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Distance x_PM correct: {self.x_PM} = {self.x_M} - ({self.x_P})")
        else:
            self._log(f"Distance x_PM incorrect: {self.x_PM} ≠ {expected_x_PM} = {self.x_M} - ({self.x_P})", False)
        
        # 4. Time-distance-speed relationship (t = x_PM / v)
        expected_t_PM = self.x_PM / self.v
        if math.isclose(self.t_PM, expected_t_PM, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Time t consistent: t = {self.t_PM} ≈ x_PM/v = {expected_t_PM}")
        else:
            self._log(f"Time t inconsistent: t = {self.t_PM}, x_PM/v = {expected_t_PM}", False)
        
        # 5. Constructive interference condition check for each time in answer2
        # CORRECTED: According to ground truth, wave A crests at M: t_A = m * T_A
        # Wave B crests at M: t_B = (n+2) * T_B
        # For constructive interference: t = m * T_A = (n+2) * T_B
        for t_val in self.times_M:
            # Check if t_val satisfies t_val = m * T_A for some integer m
            m_float = t_val / self.T_A
            m_int = round(m_float)
            is_A_crest = math.isclose(t_val, m_int * self.T_A, rel_tol=1e-2, abs_tol=1e-2)
            
            # Check if t_val satisfies t_val = (n+2) * T_B for some integer n
            n_plus2_float = t_val / self.T_B
            n_plus2_int = round(n_plus2_float)
            is_B_crest = math.isclose(t_val, n_plus2_int * self.T_B, rel_tol=1e-2, abs_tol=1e-2)
            
            # Also check that n = n_plus2_int - 2 is non-negative integer
            n_int = n_plus2_int - 2
            valid_n = n_int >= 0
            
            if is_A_crest and is_B_crest and valid_n:
                self._log(f"Time t={t_val}: constructive interference (m={m_int}, n={n_int})")
            else:
                self._log(f"Time t={t_val}: fails constructive interference check", False)
        
        # 6. Check if times are in increasing order
        if all(self.times_M[i] < self.times_M[i+1] for i in range(len(self.times_M)-1)):
            self._log(f"Times are in increasing order: {self.times_M}")
        else:
            self._log(f"Times not in increasing order: {self.times_M}", False)
    
    def test_trace_consistency(self) -> None:
        """Cross-check intermediate results with final answers."""
        
        # Check first answer (0.5s)
        expected_t_str = f"{self.t_PM}s"
        if self.answer1.strip() == expected_t_str.strip():
            self._log(f"First answer matches: '{self.answer1}'")
        else:
            self._log(f"First answer mismatch: got '{self.answer1}', expected '{expected_t_str}'", False)
        
        # Check second answer
        expected_times_str = ", ".join([f"{t}s" for t in self.times_M])
        if self.answer2.strip() == expected_times_str.strip():
            self._log(f"Second answer matches: '{self.answer2}'")
        else:
            self._log(f"Second answer mismatch: got '{self.answer2}', expected '{expected_times_str}'", False)
        
        # Check consistency between t_PM and first answer
        try:
            t_from_answer1 = float(self.answer1.replace('s', ''))
            if math.isclose(self.t_PM, t_from_answer1, rel_tol=1e-2, abs_tol=1e-2):
                self._log("t_PM consistent with first answer")
            else:
                self._log(f"t_PM={self.t_PM} inconsistent with first answer t={t_from_answer1}", False)
        except:
            self._log("Could not parse first answer", False)
    
    def test_monotonicity(self) -> None:
        """Perturb inputs and check if outputs change as expected."""
        
        # Save original values
        lambda_A_orig = self.lambda_A
        v_orig = self.v
        t_PM_orig = self.t_PM
        
        # Test 1: Increase wavelength of wave A by 10%
        self.lambda_A *= 1.1
        self.v = self.lambda_A / self.T_A  # Recalculate wave speed
        self.t_PM = self.x_PM / self.v  # Recalculate travel time
        
        # Expect: larger λ → larger v → smaller t (for fixed distance)
        if self.t_PM < t_PM_orig:
            self._log(f"Monotonicity check 1 PASS: λ_A↑ {lambda_A_orig:.3f}→{self.lambda_A:.3f} → v↑ {v_orig:.3f}→{self.v:.3f} → t↓ {t_PM_orig:.3f}→{self.t_PM:.3f}")
        else:
            self._log(f"Monotonicity check 1 FAIL: t should decrease but got {self.t_PM:.3f} ≥ {t_PM_orig:.3f}", False)
        
        # Restore original values
        self.lambda_A = lambda_A_orig
        self.v = v_orig
        self.t_PM = t_PM_orig
        
        # Test 2: Decrease period of wave A by 10%
        T_A_orig = self.T_A
        self.T_A *= 0.9
        self.v = self.lambda_A / self.T_A
        self.t_PM = self.x_PM / self.v
        
        # Expect: smaller T → larger v → smaller t
        if self.t_PM < t_PM_orig:
            self._log(f"Monotonicity check 2 PASS: T_A↓ {T_A_orig:.3f}→{self.T_A:.3f} → v↑ {v_orig:.3f}→{self.v:.3f} → t↓ {t_PM_orig:.3f}→{self.t_PM:.3f}")
        else:
            self._log(f"Monotonicity check 2 FAIL: t should decrease but got {self.t_PM:.3f} ≥ {t_PM_orig:.3f}", False)
        
        # Restore
        self.T_A = T_A_orig
        self.v = v_orig
        self.t_PM = t_PM_orig
        
        # Test 3: Increase distance x_PM by 10%
        x_PM_orig = self.x_PM
        self.x_PM *= 1.1
        self.t_PM = self.x_PM / self.v
        
        # Expect: larger distance → larger time
        if self.t_PM > t_PM_orig:
            self._log(f"Monotonicity check 3 PASS: x_PM↑ {x_PM_orig:.3f}→{self.x_PM:.3f} → t↑ {t_PM_orig:.3f}→{self.t_PM:.3f}")
        else:
            self._log(f"Monotonicity check 3 FAIL: t should increase but got {self.t_PM:.3f} ≤ {t_PM_orig:.3f}", False)
        
        # Restore
        self.x_PM = x_PM_orig
        self.t_PM = t_PM_orig
    
    def run_all_tests(self) -> tuple[bool, list[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        self.logs.append("=== PhysicsChecker Started ===")
        
        try:
            self.test_invariants()
        except Exception as e:
            self._log(f"test_invariants crashed: {str(e)}", False)
        
        try:
            self.test_trace_consistency()
        except Exception as e:
            self._log(f"test_trace_consistency crashed: {str(e)}", False)
        
        try:
            self.test_monotonicity()
        except Exception as e:
            self._log(f"test_monotonicity crashed: {str(e)}", False)
        
        # Determine overall success
        fail_count = sum(1 for log in self.logs if log.startswith("FAIL"))
        success = fail_count == 0
        
        self.logs.append(f"=== PhysicsChecker Finished: {'PASS' if success else 'FAIL'} ({fail_count} failures) ===")
        
        return success, self.logs