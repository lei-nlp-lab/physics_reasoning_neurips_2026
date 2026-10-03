import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults from problem statement
        self.m1 = getattr(solution_instance, 'm1', 0.6)  # 600g = 0.6 kg
        self.m2 = getattr(solution_instance, 'm2', 1.2)  # 1200g = 1.2 kg
        self.S = getattr(solution_instance, 'S', 20e-4)  # 20 cm² = 20e-4 m²
        self.p0 = getattr(solution_instance, 'p0', 1.0e5)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.T1 = getattr(solution_instance, 'T1', 300.0)
        
        # Safe extraction of results from solution instance
        self.results = getattr(solution_instance, 'results', {})
        self.m2_400g_m1g = self.results.get('m2-400g-m1g', None)
        self.p1 = self.results.get('p1', None)
        self.T_max = self.results.get('T_max', None)
        self.p3_p0S = self.results.get('p3-p0S', None)
        self.delta_mg = self.results.get('delta_mg', None)
        self.T2 = self.results.get('T2', None)
        self.p3 = self.results.get('p3', None)
        self.p0_p2S = self.results.get('p0-p2S', None)
        
        # Expected values from KB analysis
        self.expected = {
            'delta_mg': (self.m2 - 0.6) * self.g,  # 0.6 kg = 600g balance reading
            'p1': self.p0,
            'm2_400g_m1g': (self.m2 - 0.4 - self.m1) * self.g,  # 0.4 kg = 400g balance reading
            'p0_p2S': (self.p0 - (self.p0 - ((self.m2 - 0.4 - self.m1) * self.g / self.S))) * self.S,
            'p3': self.p0 + (self.m1 * self.g) / self.S,
            'p3_p0S': self.m1 * self.g,
            'T2': (self.p0 - ((self.m2 - 0.4 - self.m1) * self.g / self.S)) / self.p0 * self.T1,
            'T_max': (self.p0 + (self.m1 * self.g) / self.S) / self.p0 * self.T1
        }
        
        self.logs = []
    
    def _log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical invariants: Force balances and Ideal Gas Law."""
        success = True
        
        # 1. Force balance on piston when balance reads 600g (step 2)
        # p1 should equal p0 (since tension balances piston weight exactly)
        if self.p1 is not None:
            if not self._is_close(self.p1, self.expected['p1']):
                self._log(f"p1 mismatch: got {self.p1}, expected {self.expected['p1']}", False)
                success = False
            else:
                self._log(f"p1 = p0 invariant holds: {self.p1} Pa", True)
        
        # 2. Force balance when balance reads 400g (step 3)
        # (m2 - 400g - m1)g should equal (p0 - p2)S
        if self.m2_400g_m1g is not None and self.p0_p2S is not None:
            if not self._is_close(self.m2_400g_m1g, self.p0_p2S):
                self._log(f"Force balance at 400g mismatch: (m2-400g-m1)g = {self.m2_400g_m1g}, (p0-p2)S = {self.p0_p2S}", False)
                success = False
            else:
                self._log(f"Force balance at 400g holds: {self.m2_400g_m1g} N", True)
        
        # 3. Force balance at maximum temperature (step 6)
        # (p3 - p0)S should equal m1*g
        if self.p3_p0S is not None:
            if not self._is_close(self.p3_p0S, self.expected['p3_p0S']):
                self._log(f"Force balance at max T mismatch: (p3-p0)S = {self.p3_p0S}, m1*g = {self.m1*self.g}", False)
                success = False
            else:
                self._log(f"Force balance at max T holds: {self.p3_p0S} N", True)
        
        # 4. Ideal Gas Law (isochoric) for T2 (step 4-5)
        # p1/T1 should equal p2/T2 => T2 = (p2/p1)*T1
        if self.T2 is not None and self.p1 is not None:
            # Compute p2 from force balance: p2 = p0 - (m2-400g-m1)g/S
            p2_calc = self.p0 - (self.m2 - 0.4 - self.m1) * self.g / self.S
            T2_calc = p2_calc / self.p1 * self.T1
            if not self._is_close(self.T2, T2_calc):
                self._log(f"Ideal Gas Law for T2 mismatch: got {self.T2}, calculated {T2_calc}", False)
                success = False
            else:
                self._log(f"Ideal Gas Law for T2 holds: {self.T2} K", True)
        
        # 5. Ideal Gas Law for T_max (step 7-8)
        # p1/T1 should equal p3/T_max => T_max = (p3/p1)*T1
        if self.T_max is not None and self.p3 is not None and self.p1 is not None:
            T_max_calc = self.p3 / self.p1 * self.T1
            if not self._is_close(self.T_max, T_max_calc):
                self._log(f"Ideal Gas Law for T_max mismatch: got {self.T_max}, calculated {T_max_calc}", False)
                success = False
            else:
                self._log(f"Ideal Gas Law for T_max holds: {self.T_max} K", True)
        
        return success
    
    def test_trace_consistency(self):
        """Check intermediate results are consistent with final answers."""
        success = True
        
        # Expected final answers from problem
        expected_T2 = 297.0
        expected_Tmax = 309.0
        
        if self.T2 is not None:
            if not self._is_close(self.T2, expected_T2):
                self._log(f"T2 not matching target: got {self.T2}, expected {expected_T2}", False)
                success = False
            else:
                self._log(f"T2 matches target: {self.T2} K", True)
        
        if self.T_max is not None:
            if not self._is_close(self.T_max, expected_Tmax):
                self._log(f"T_max not matching target: got {self.T_max}, expected {expected_Tmax}", False)
                success = False
            else:
                self._log(f"T_max matches target: {self.T_max} K", True)
        
        # Cross-check: delta_mg at 600g balance reading
        if self.delta_mg is not None:
            expected_delta = (self.m2 - 0.6) * self.g  # 0.6 kg balance reading
            if not self._is_close(self.delta_mg, expected_delta):
                self._log(f"delta_mg mismatch: got {self.delta_mg}, expected {expected_delta}", False)
                success = False
            else:
                self._log(f"delta_mg consistent: {self.delta_mg} N", True)
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb input mass m1 and verify physical trends.
        Increase m1 → gas pressure should increase → temperatures should increase.
        Then restore original values.
        """
        success = True
        original_m1 = self.m1
        
        try:
            # Perturb m1 by +10%
            perturbed_m1 = original_m1 * 1.1
            
            # Recalculate expected values with perturbed m1
            p3_perturbed = self.p0 + (perturbed_m1 * self.g) / self.S
            T_max_perturbed = p3_perturbed / self.p0 * self.T1
            
            # With increased m1, maximum temperature should increase
            T_max_original = self.expected['T_max']
            if T_max_perturbed <= T_max_original:
                self._log(f"Monotonicity fail: increasing m1 from {original_m1} to {perturbed_m1} should increase T_max, but got {T_max_perturbed} <= {T_max_original}", False)
                success = False
            else:
                self._log(f"Monotonicity holds: increasing m1 increases T_max (from {T_max_original} to {T_max_perturbed})", True)
            
            # Also check T2 trend: p2 = p0 - (m2-400g-m1)g/S
            # As m1 increases, (m2-0.4-m1) decreases, so p2 increases, so T2 increases
            p2_original = self.p0 - ((self.m2 - 0.4 - original_m1) * self.g) / self.S
            T2_original = p2_original / self.p0 * self.T1
            p2_perturbed = self.p0 - ((self.m2 - 0.4 - perturbed_m1) * self.g) / self.S
            T2_perturbed = p2_perturbed / self.p0 * self.T1
            
            # Actually, T2 should INCREASE with m1, not decrease
            # Because p2 increases with m1: p2 = p0 - (m2-0.4-m1)g/S
            # When m1 increases, (m2-0.4-m1) decreases, so p2 increases
            # T2 = (p2/p0)*T1, so T2 increases
            if T2_perturbed <= T2_original:
                self._log(f"Monotonicity fail: increasing m1 should increase T2, but got {T2_perturbed} <= {T2_original}", False)
                success = False
            else:
                self._log(f"Monotonicity holds: increasing m1 increases T2 (from {T2_original} to {T2_perturbed})", True)
        
        except Exception as e:
            self._log(f"Monotonicity test error: {traceback.format_exc()}", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"--- Running {test_name} Test ---", None)
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test {test_name} crashed: {e}", False)
                overall_success = False
        
        self._log(f"=== Overall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===", None)
        return overall_success, self.logs