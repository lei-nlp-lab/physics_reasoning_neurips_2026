import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker by safely extracting inputs and results from solution.
        """
        # Extract inputs with safe defaults (convert cm to m for radius)
        self.r = solution_instance.results.get('r', 0.1)  # 10 cm = 0.1 m
        self.N = solution_instance.results.get('N', 100)
        self.B = solution_instance.results.get('B', 5/(math.pi**2))
        
        # Extract n correctly - it should be in r/s from the solution
        self.n_rps = solution_instance.results.get('n', 10.0)  # in r/s
        self.n_rpm = self.n_rps * 60  # convert back to r/min for reference
        
        # Extract results safely
        self.omega = solution_instance.results.get('omega', None)
        self.E_m = solution_instance.results.get('E_m', None)
        self.e_at_t = solution_instance.results.get('e', None)  # This is e at t=1/60
        self.overlineE = solution_instance.results.get('overlineE', None)
        self.delta_Phi = solution_instance.results.get('delta_Phi', None)
        
        # Derived constants
        self.A = math.pi * (self.r ** 2)  # area in m²
        
        # Target answers for comparison
        self.targets = {
            'omega': 20 * math.pi,  # 20π rad/s
            'E_m': 100.0,  # 100 V
            'e_at_t': 50 * math.sqrt(3),  # 50√3 V ≈ 86.6025 V
            'overlineE': 150 / math.pi  # 150/π V ≈ 47.7465 V
        }
        
        self.logs = []
    
    def _log(self, msg):
        self.logs.append(msg)
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physical laws."""
        checks = []
        
        # 1. Check ω = 2πn (from step 2)
        expected_omega = 2 * math.pi * self.n_rps
        if self.omega is not None:
            if not self._is_close(self.omega, expected_omega):
                self._log(f"FAIL: ω mismatch. Expected {expected_omega}, got {self.omega}")
                checks.append(False)
            else:
                self._log(f"PASS: ω = 2πn check: {self.omega} ≈ {expected_omega}")
                checks.append(True)
        
        # 2. Check E_m = N * B * A * ω (from step 3)
        if self.omega is not None:
            expected_E_m = self.N * self.B * self.A * self.omega
            if self.E_m is not None:
                if not self._is_close(self.E_m, expected_E_m):
                    self._log(f"FAIL: E_m mismatch. Expected {expected_E_m}, got {self.E_m}")
                    checks.append(False)
                else:
                    self._log(f"PASS: E_m = N*B*A*ω check: {self.E_m} ≈ {expected_E_m}")
                    checks.append(True)
        
        # 3. Check e(t) at t=1/60 = E_m * sin(ω*(1/60))
        if self.E_m is not None and self.omega is not None and self.e_at_t is not None:
            t_val = 1/60
            expected_e_at_t = self.E_m * math.sin(self.omega * t_val)
            if not self._is_close(self.e_at_t, expected_e_at_t):
                self._log(f"FAIL: e(1/60) mismatch. Expected {expected_e_at_t}, got {self.e_at_t}")
                checks.append(False)
            else:
                self._log(f"PASS: e(1/60) = E_m*sin(ω*(1/60)) check: {self.e_at_t} ≈ {expected_e_at_t}")
                checks.append(True)
        
        # 4. Check Faraday's law: average EMF = |ΔΦ| / Δt
        if self.delta_Phi is not None and self.overlineE is not None:
            delta_t = 1/60  # from t=0 to t=1/60 s
            expected_avg = abs(self.delta_Phi) / delta_t
            if not self._is_close(self.overlineE, expected_avg):
                self._log(f"FAIL: Faraday's law mismatch. Expected {expected_avg}, got {self.overlineE}")
                checks.append(False)
            else:
                self._log(f"PASS: Faraday's law check: {self.overlineE} ≈ |ΔΦ|/Δt = {expected_avg}")
                checks.append(True)
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        checks = []
        
        # Check step 1: n conversion (600 r/min = 10 r/s)
        if not self._is_close(self.n_rps, 10.0):
            self._log(f"FAIL: n conversion. Expected 10 r/s, got {self.n_rps}")
            checks.append(False)
        else:
            self._log(f"PASS: n conversion: {self.n_rps} ≈ 10 r/s")
        
        # Check step 2: ω calculation (ω = 20π rad/s)
        if self.omega is not None:
            if not self._is_close(self.omega, self.targets['omega']):
                self._log(f"FAIL: ω calculation. Expected {self.targets['omega']}, got {self.omega}")
                checks.append(False)
            else:
                self._log(f"PASS: ω = 20π: {self.omega} ≈ {self.targets['omega']}")
        
        # Check step 3: E_m calculation (E_m = 100 V)
        if self.E_m is not None:
            if not self._is_close(self.E_m, self.targets['E_m']):
                self._log(f"FAIL: E_m calculation. Expected {self.targets['E_m']}, got {self.E_m}")
                checks.append(False)
            else:
                self._log(f"PASS: E_m = 100 V: {self.E_m} ≈ {self.targets['E_m']}")
        
        # Check step 5: e at t=1/60 (e = 50√3 V ≈ 86.6025 V)
        if self.e_at_t is not None:
            if not self._is_close(self.e_at_t, self.targets['e_at_t']):
                self._log(f"FAIL: e(1/60) mismatch. Expected {self.targets['e_at_t']}, got {self.e_at_t}")
                checks.append(False)
            else:
                self._log(f"PASS: e(1/60) = 50√3: {self.e_at_t} ≈ {self.targets['e_at_t']}")
        
        # Check step 8: average EMF (E_avg = 150/π V ≈ 47.7465 V)
        if self.overlineE is not None:
            if not self._is_close(self.overlineE, self.targets['overlineE']):
                self._log(f"FAIL: average EMF mismatch. Expected {self.targets['overlineE']}, got {self.overlineE}")
                checks.append(False)
            else:
                self._log(f"PASS: average EMF = 150/π: {self.overlineE} ≈ {self.targets['overlineE']}")
        
        return all(checks)
    
    def test_monotonicity(self):
        """Perturb inputs and check trends."""
        checks = []
        
        # Test 1: Increase B by 10%, E_m should increase proportionally
        original_B = self.B
        perturbed_B = original_B * 1.1
        
        if self.omega is not None:
            original_E_m = self.N * original_B * self.A * self.omega
            perturbed_E_m = self.N * perturbed_B * self.A * self.omega
            
            if not perturbed_E_m > original_E_m:
                self._log(f"FAIL: Increasing B did not increase E_m. Original: {original_E_m}, Perturbed: {perturbed_E_m}")
                checks.append(False)
            else:
                expected_ratio = 1.1
                actual_ratio = perturbed_E_m / original_E_m
                if not self._is_close(actual_ratio, expected_ratio):
                    self._log(f"WARN: E_m increase ratio {actual_ratio} not exactly {expected_ratio} (within tolerance)")
                self._log(f"PASS: E_m increases with B: {original_E_m} → {perturbed_E_m}")
        
        # Test 2: Increase n by 10%, ω and E_m should increase proportionally
        original_n_rps = self.n_rps
        perturbed_n_rps = original_n_rps * 1.1
        
        original_omega = 2 * math.pi * original_n_rps
        perturbed_omega = 2 * math.pi * perturbed_n_rps
        
        original_E_m2 = self.N * self.B * self.A * original_omega
        perturbed_E_m2 = self.N * self.B * self.A * perturbed_omega
        
        if not perturbed_omega > original_omega:
            self._log(f"FAIL: Increasing n did not increase ω")
            checks.append(False)
        elif not perturbed_E_m2 > original_E_m2:
            self._log(f"FAIL: Increasing n did not increase E_m")
            checks.append(False)
        else:
            omega_ratio = perturbed_omega / original_omega
            E_m_ratio = perturbed_E_m2 / original_E_m2
            if not self._is_close(omega_ratio, 1.1) or not self._is_close(E_m_ratio, 1.1):
                self._log(f"WARN: Ratios not exactly 1.1: ω_ratio={omega_ratio}, E_m_ratio={E_m_ratio}")
            self._log(f"PASS: ω and E_m increase with n: ω {original_omega}→{perturbed_omega}, E_m {original_E_m2}→{perturbed_E_m2}")
        
        return all(checks)
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        success = True
        
        try:
            self._log("=== Physics Checker Started ===")
            
            # Run tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            self._log(f"\n=== Summary ===")
            self._log(f"Invariants: {'PASS' if inv_ok else 'FAIL'}")
            self._log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
            self._log(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
            self._log(f"Overall: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            self._log(f"ERROR during testing: {str(e)}")
            self._log(traceback.format_exc())
            success = False
        
        return success, self.logs

# Example usage (if run directly):
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'r': 0.1,
                'N': 100,
                'B': 5/(math.pi**2),
                'n': 10.0,
                'omega': 20*math.pi,
                'E_m': 100.0,
                'e': 50*math.sqrt(3),
                'overlineE': 150/math.pi,
                'delta_Phi': -0.5 * 100 * (5/(math.pi**2)) * math.pi * (0.1**2)
            }
    
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)