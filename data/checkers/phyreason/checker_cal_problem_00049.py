import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with safe fallbacks
        self.m = getattr(solution_instance, 'm', 1.0)  # default 1.0 if missing
        self.H = getattr(solution_instance, 'H', 10.0)
        self.h = getattr(solution_instance, 'h', 5.0)
        # lambda is a reserved word, so we expect 'lam' or 'lambda_'
        self.lam = getattr(solution_instance, 'lam', getattr(solution_instance, 'lambda_', 0.1))
        self.g = getattr(solution_instance, 'g', 9.8)
        self.h0 = getattr(solution_instance, 'h0', 0.5)
        self.N = getattr(solution_instance, 'N', 3)
        self.v = getattr(solution_instance, 'v', 0.0)  # initial velocity for impulse part
        
        # Extract results with safe fallbacks using the key mapping
        self.k = self._safe_get_result('k')
        self.F0 = self._safe_get_result('F_0')
        self.I = self._safe_get_result('I')
        
        # Intermediate velocities and heights (for trace consistency)
        self.v_down = self._safe_get_result('v_down')
        self.v_up = self._safe_get_result('v_up')
        self.a_down = self._safe_get_result('a_down')
        self.a_up = self._safe_get_result('a_up')
        self.v1 = self._safe_get_result('v_1')
        self.kv1 = self._safe_get_result('kv_1')
        self.h1 = self._safe_get_result('h_1')
        self.h2 = self._safe_get_result('h_2')
        self.h3 = self._safe_get_result('h_3')
        self.hN = self._safe_get_result('h_N')
        self.v2 = self._safe_get_result('v_2')
        self.v3 = self._safe_get_result('v_3')
        self.vN = self._safe_get_result('v_N')
        self.hv2 = self._safe_get_result('hv_2')
        self.hv3 = self._safe_get_result('hv_3')
        self.hvN = self._safe_get_result('hv_N')
        
        # Energy results
        self.KE_down = self._safe_get_result('1/2*m*v_down\'2')
        self.KE_up = self._safe_get_result('1/2*m*(k*v_F\')^2')
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for reporting
        self.logs = []
    
    def _safe_get_result(self, key):
        """Safely retrieve a result from the results dict or solution attributes."""
        # First try the results dictionary
        value = self.results.get(key, None)
        if value is not None:
            return value
        # Fallback to solution instance attribute (if exists)
        return getattr(self.solution, key, None)
    
    def _log(self, message, success=None):
        """Add a log message with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def _symbolic_check(self, expr1, expr2, symbols):
        """Check if two symbolic expressions are equivalent under given symbols."""
        try:
            diff = sp.simplify(expr1 - expr2)
            # Substitute numerical values to check if difference is zero
            sub_dict = {sym: getattr(self, str(sym)) for sym in symbols if hasattr(self, str(sym))}
            # If any symbol is missing, skip this check
            if all(sym in sub_dict for sym in symbols):
                numeric_diff = diff.subs(sub_dict)
                return abs(numeric_diff) < 1e-9  # symbolic exactness
            return False
        except Exception:
            return False
    
    def test_invariants(self):
        """Verify fundamental physics invariants: Newton's 2nd law, energy, and kinematic consistency."""
        success = True
        
        # 1. Newton's second law for accelerations
        # a_down should be (1 - λ) * g
        expected_a_down = (1 - self.lam) * self.g
        if self.a_down is not None:
            if not self._is_close(self.a_down, expected_a_down):
                self._log(f"a_down mismatch: expected {expected_a_down}, got {self.a_down}", False)
                success = False
            else:
                self._log("a_down satisfies Newton's second law (descent).", True)
        
        # a_up should be (1 + λ) * g
        expected_a_up = (1 + self.lam) * self.g
        if self.a_up is not None:
            if not self._is_close(self.a_up, expected_a_up):
                self._log(f"a_up mismatch: expected {expected_a_up}, got {self.a_up}", False)
                success = False
            else:
                self._log("a_up satisfies Newton's second law (ascent).", True)
        
        # 2. Kinematic consistency: v_down^2 = 2 * a_down * H
        if self.v_down is not None:
            v_down_sq_expected = 2 * expected_a_down * self.H
            v_down_sq_actual = self.v_down ** 2
            if not self._is_close(v_down_sq_actual, v_down_sq_expected):
                self._log(f"v_down kinematic mismatch: expected v_down^2={v_down_sq_expected}, got {v_down_sq_actual}", False)
                success = False
            else:
                self._log("v_down consistent with kinematic equation.", True)
        
        # v_up^2 = 2 * a_up * h
        if self.v_up is not None:
            v_up_sq_expected = 2 * expected_a_up * self.h
            v_up_sq_actual = self.v_up ** 2
            if not self._is_close(v_up_sq_actual, v_up_sq_expected):
                self._log(f"v_up kinematic mismatch: expected v_up^2={v_up_sq_expected}, got {v_up_sq_actual}", False)
                success = False
            else:
                self._log("v_up consistent with kinematic equation.", True)
        
        # 3. Coefficient of restitution k = v_up / v_down
        if all(v is not None for v in [self.v_up, self.v_down, self.k]):
            k_expected = self.v_up / self.v_down
            if not self._is_close(self.k, k_expected):
                self._log(f"k mismatch from velocities: expected {k_expected}, got {self.k}", False)
                success = False
            else:
                self._log("k matches velocity ratio.", True)
            
            # Also check the derived formula for k
            k_formula = math.sqrt((1 + self.lam) * self.h / ((1 - self.lam) * self.H))
            if not self._is_close(self.k, k_formula):
                self._log(f"k mismatch from formula: expected {k_formula}, got {self.k}", False)
                success = False
            else:
                self._log("k matches derived formula.", True)
        
        # 4. Energy check: Work-energy theorem for the first drop and rebound
        # Descent: loss due to air resistance
        # The net work done during descent: W_net = (mg - λmg) * H = m g (1-λ) H
        # This should equal the kinetic energy just before impact: 0.5 * m * v_down^2
        if self.v_down is not None:
            W_net_down = self.m * self.g * (1 - self.lam) * self.H
            KE_down_calc = 0.5 * self.m * self.v_down ** 2
            if not self._is_close(W_net_down, KE_down_calc):
                self._log(f"Work-energy mismatch during descent: expected work {W_net_down}, got KE {KE_down_calc}", False)
                success = False
            else:
                self._log("Work-energy theorem holds during descent.", True)
        
        # Ascent: work done against gravity and air resistance
        # The net work during ascent is negative: -mgh - λmgh = -mg(1+λ)h
        # Change in KE: 0 - 0.5 * m * v_up^2
        if self.v_up is not None:
            W_net_up = -self.m * self.g * (1 + self.lam) * self.h
            delta_KE_up = -0.5 * self.m * self.v_up ** 2
            if not self._is_close(W_net_up, delta_KE_up):
                self._log(f"Work-energy mismatch during ascent: expected work {W_net_up}, got ΔKE {delta_KE_up}", False)
                success = False
            else:
                self._log("Work-energy theorem holds during ascent.", True)
        
        return success
    
    def test_trace_consistency(self):
        """Check that intermediate results (heights, velocities) are consistent with the final answers."""
        success = True
        
        # 1. Check that the derived F0 matches the formula
        if self.F0 is not None:
            # Formula from target answer: F0 = 2 * m * g * (1-λ) * (H - h) / (h - h0)
            F0_expected = 2 * self.m * self.g * (1 - self.lam) * (self.H - self.h) / (self.h - self.h0)
            if not self._is_close(self.F0, F0_expected):
                self._log(f"F0 mismatch: expected {F0_expected}, got {self.F0}", False)
                success = False
            else:
                self._log("F0 matches target formula.", True)
        
        # 2. Check the recurrence for heights (h1, h2, h3, hN) if available
        # From step 15: h1 = (h/H) * h0 + (h/H) * v^2 / (2g(1-λ))
        if self.h1 is not None:
            h1_expected = (self.h / self.H) * self.h0 + (self.h / self.H) * (self.v ** 2) / (2 * self.g * (1 - self.lam))
            if not self._is_close(self.h1, h1_expected):
                self._log(f"h1 mismatch: expected {h1_expected}, got {self.h1}", False)
                success = False
            else:
                self._log("h1 matches recurrence formula.", True)
        
        # From step 18: h2 = (h/H)^2 * h0 + (h/H)^2 * v^2/(2g(1-λ)) + (h/H) * v^2/(2g(1-λ))
        if self.h2 is not None:
            term = (self.v ** 2) / (2 * self.g * (1 - self.lam))
            h2_expected = (self.h/self.H)**2 * self.h0 + (self.h/self.H)**2 * term + (self.h/self.H) * term
            if not self._is_close(self.h2, h2_expected):
                self._log(f"h2 mismatch: expected {h2_expected}, got {self.h2}", False)
                success = False
            else:
                self._log("h2 matches recurrence formula.", True)
        
        # 3. Check that the final impulse I matches the target formula
        if self.I is not None:
            # Formula: I = m * sqrt( 2g(1-λ)(H-h)(H^{N+1} - h^{N+1}) / ( h (H^N - h^N) ) )
            numerator = 2 * self.g * (1 - self.lam) * (self.H - self.h) * (self.H**(self.N+1) - self.h**(self.N+1))
            denominator = self.h * (self.H**self.N - self.h**self.N)
            if denominator == 0:
                self._log("Denominator in I formula is zero; cannot compute expected I.", False)
                success = False
            else:
                I_expected = self.m * math.sqrt(numerator / denominator)
                if not self._is_close(self.I, I_expected):
                    self._log(f"I mismatch: expected {I_expected}, got {self.I}", False)
                    success = False
                else:
                    self._log("I matches target formula.", True)
        
        # 4. Check that the velocities v1, v2, etc. are consistent with the heights
        # v1^2 - v^2 = 2 (1-λ) g h0
        if self.v1 is not None:
            v1_sq_expected = self.v**2 + 2 * (1 - self.lam) * self.g * self.h0
            if not self._is_close(self.v1**2, v1_sq_expected):
                self._log(f"v1 kinematic mismatch: expected v1^2={v1_sq_expected}, got {self.v1**2}", False)
                success = False
            else:
                self._log("v1 consistent with kinematic equation.", True)
        
        # kv1 should equal sqrt(2 (1+λ) g h1)
        if self.kv1 is not None and self.h1 is not None:
            kv1_expected = math.sqrt(2 * (1 + self.lam) * self.g * self.h1)
            if not self._is_close(self.kv1, kv1_expected):
                self._log(f"kv1 mismatch: expected {kv1_expected}, got {self.kv1}", False)
                success = False
            else:
                self._log("kv1 (rebound velocity after first impact) is consistent.", True)
        
        # Similarly for v2 and h1: v2^2 - v^2 = 2 (1-λ) g h1
        if self.v2 is not None and self.h1 is not None:
            v2_sq_expected = self.v**2 + 2 * (1 - self.lam) * self.g * self.h1
            if not self._is_close(self.v2**2, v2_sq_expected):
                self._log(f"v2 kinematic mismatch: expected v2^2={v2_sq_expected}, got {self.v2**2}", False)
                success = False
            else:
                self._log("v2 consistent with kinematic equation.", True)
        
        # And so on... We can add more checks if data is available.
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input parameter (λ) and check that the results change in a physically expected way.
        We'll perturb λ by a small amount (1e-3) and recalculate k, F0, I using the formulas.
        Then check monotonicity trends and restore the original λ.
        """
        success = True
        
        # Store original λ
        lam_original = self.lam
        
        # Perturbation amount (small relative change)
        delta = 1e-3
        self.lam = lam_original + delta
        
        # Recalculate expected values using formulas with perturbed λ
        k_perturbed = math.sqrt((1 + self.lam) * self.h / ((1 - self.lam) * self.H))
        F0_perturbed = 2 * self.m * self.g * (1 - self.lam) * (self.H - self.h) / (self.h - self.h0)
        # For I, we need to compute with perturbed λ and original N
        numerator = 2 * self.g * (1 - self.lam) * (self.H - self.h) * (self.H**(self.N+1) - self.h**(self.N+1))
        denominator = self.h * (self.H**self.N - self.h**self.N)
        I_perturbed = self.m * math.sqrt(numerator / denominator) if denominator != 0 else None
        
        # Restore original λ
        self.lam = lam_original
        
        # Get original values (from results or computed from formulas)
        k_original = self.k if self.k is not None else math.sqrt((1 + self.lam) * self.h / ((1 - self.lam) * self.H))
        F0_original = self.F0 if self.F0 is not None else 2 * self.m * self.g * (1 - self.lam) * (self.H - self.h) / (self.h - self.h0)
        if self.I is not None:
            I_original = self.I
        else:
            num_orig = 2 * self.g * (1 - self.lam) * (self.H - self.h) * (self.H**(self.N+1) - self.h**(self.N+1))
            denom_orig = self.h * (self.H**self.N - self.h**self.N)
            I_original = self.m * math.sqrt(num_orig / denom_orig) if denom_orig != 0 else None
        
        # Expected trends:
        # As λ increases (air resistance increases), k should decrease? Let's analyze:
        # k = sqrt( (1+λ)/(1-λ) * h/H ). Since (1+λ)/(1-λ) increases with λ, k increases with λ.
        # So k should increase when λ increases.
        if k_original is not None and k_perturbed is not None:
            if not (k_perturbed > k_original):
                self._log(f"Monotonicity fail for k: λ increased from {lam_original} to {lam_original+delta}, but k did not increase (original {k_original}, perturbed {k_perturbed}).", False)
                success = False
            else:
                self._log(f"k increases with λ (as expected).", True)
        
        # For F0: F0 = 2 m g (1-λ) (H-h)/(h-h0). So F0 decreases as λ increases.
        if F0_original is not None and F0_perturbed is not None:
            if not (F0_perturbed < F0_original):
                self._log(f"Monotonicity fail for F0: λ increased, but F0 did not decrease (original {F0_original}, perturbed {F0_perturbed}).", False)
                success = False
            else:
                self._log(f"F0 decreases with λ (as expected).", True)
        
        # For I: I = m * sqrt( 2g(1-λ)(H-h)(H^{N+1} - h^{N+1}) / ( h (H^N - h^N) ) )
        # The factor (1-λ) decreases with λ, so I should decrease as λ increases.
        if I_original is not None and I_perturbed is not None:
            if not (I_perturbed < I_original):
                self._log(f"Monotonicity fail for I: λ increased, but I did not decrease (original {I_original}, perturbed {I_perturbed}).", False)
                success = False
            else:
                self._log(f"I decreases with λ (as expected).", True)
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        overall_success = True
        
        try:
            self._log("Starting PhysicsChecker tests...")
            
            # Test invariants
            inv_success = self.test_invariants()
            overall_success = overall_success and inv_success
            self._log(f"Invariants test: {'PASS' if inv_success else 'FAIL'}")
            
            # Test trace consistency
            trace_success = self.test_trace_consistency()
            overall_success = overall_success and trace_success
            self._log(f"Trace consistency test: {'PASS' if trace_success else 'FAIL'}")
            
            # Test monotonicity
            mono_success = self.test_monotonicity()
            overall_success = overall_success and mono_success
            self._log(f"Monotonicity test: {'PASS' if mono_success else 'FAIL'}")
            
        except Exception as e:
            self._log(f"Exception during tests: {traceback.format_exc()}", False)
            overall_success = False
        
        self._log(f"Overall test result: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs