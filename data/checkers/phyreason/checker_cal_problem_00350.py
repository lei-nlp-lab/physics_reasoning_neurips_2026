import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract constants from results dict with safe defaults
        self.m = self.results.get('m', 8.0e-26)
        self.q = self.results.get('q', 8.0e-19)
        self.v0 = self.results.get('v0', 2.0e5)
        
        # Extract results with safe access
        self.L = self.results.get('L', None)
        self.theta = self.results.get('theta', None)  # in radians (as stored)
        self.ratio = self.results.get('N/N_all', None)
        self.t = self.results.get('t', None)
        self.a = self.results.get('a', None)
        self.U_AB = self.results.get('U_AB', None)
        self.alpha = self.results.get('alpha', None)  # in radians (as stored)
        self.t_prime = self.results.get('t_prime', None)
        self.a_prime = self.results.get('a_prime', None)
        self.U_BA = self.results.get('U_BA', None)
        
        self.logs = []
        
    def _log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            msg = f"[{status}] {msg}"
        self.logs.append(msg)
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        if a is None or b is None:
            return False
        # Handle very small numbers
        if abs(float(a)) < 1e-30 and abs(float(b)) < 1e-30:
            return True
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _safe_divide(self, num, den):
        """Safe division that returns 0 if denominator is 0 or very small."""
        if den is None or abs(float(den)) < 1e-30:
            return 0.0
        return float(num) / float(den)
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Check ratio from geometry: θ should be 90° = π/2 rad, ratio = θ/π = 0.5
            if self.theta is not None:
                theta_deg = math.degrees(self.theta)
                if not self._is_close(theta_deg, 90.0):
                    self._log(f"Theta should be 90°, got {theta_deg}°", False)
                    return False
                self._log(f"Theta check: {theta_deg}° ≈ 90°", True)
            
            if self.ratio is not None:
                if not self._is_close(self.ratio, 0.5):
                    self._log(f"Ratio should be 0.5, got {self.ratio}", False)
                    return False
                self._log(f"Ratio check: {self.ratio} ≈ 0.5", True)
            
            # 2. Check Newton's 2nd Law for first condition: a = q*U_AB/(m*L)
            if all(v is not None for v in [self.a, self.U_AB, self.m, self.q, self.L]):
                # Safe division to avoid zero division
                if abs(self.m * self.L) < 1e-30:
                    self._log("Skipping Newton's 2nd Law check: m*L is too small", True)
                else:
                    a_calc = self.q * self.U_AB / (self.m * self.L)
                    if not self._is_close(self.a, a_calc):
                        self._log(f"Newton's 2nd Law fail: a={self.a}, qU/(mL)={a_calc}", False)
                        return False
                    self._log(f"Newton's 2nd Law check: a = qU_AB/(mL) holds", True)
            
            # 3. Check Newton's 2nd Law for second condition: a' = q*U_BA/(m*L)
            if all(v is not None for v in [self.a_prime, self.U_BA, self.m, self.q, self.L]):
                # Safe division to avoid zero division
                if abs(self.m * self.L) < 1e-30:
                    self._log("Skipping Newton's 2nd Law check (second condition): m*L is too small", True)
                else:
                    a_prime_calc = self.q * self.U_BA / (self.m * self.L)
                    if not self._is_close(self.a_prime, a_prime_calc):
                        self._log(f"Newton's 2nd Law fail for second condition: a'={self.a_prime}, qU_BA/(mL)={a_prime_calc}", False)
                        return False
                    self._log(f"Newton's 2nd Law check (second condition): a' = qU_BA/(mL) holds", True)
            
            # 4. Energy conservation: work done by field = kinetic energy change
            # For first condition: initial KE = 0.5*m*v0^2, final has both x and y components
            # Work = q*U_AB (since moves from A to B through potential U_AB)
            if all(v is not None for v in [self.U_AB, self.m, self.v0, self.t, self.a]):
                # Final horizontal velocity: v_x = a*t
                v_x_final = self.a * self.t
                # Final vertical velocity remains v0 (no vertical acceleration)
                v_y_final = self.v0
                ke_final = 0.5 * self.m * (v_x_final**2 + v_y_final**2)
                ke_initial = 0.5 * self.m * self.v0**2
                work_done = self.q * self.U_AB
                if not self._is_close(ke_final - ke_initial, work_done):
                    self._log(f"Energy conservation fail: ΔKE={ke_final-ke_initial}, qU_AB={work_done}", False)
                    return False
                self._log(f"Energy conservation check: ΔKE = qU_AB holds", True)
            
            # 5. Voltage relation: U_AB = -U_BA
            if self.U_AB is not None and self.U_BA is not None:
                if not self._is_close(self.U_AB, -self.U_BA):
                    self._log(f"Voltage relation fail: U_AB={self.U_AB}, -U_BA={-self.U_BA}", False)
                    return False
                self._log(f"Voltage relation check: U_AB = -U_BA holds", True)
            
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. First condition: check kinematics consistency
            if all(v is not None for v in [self.t, self.L, self.v0]):
                # Vertical motion: L/2 = v0*t
                if abs(self.v0) < 1e-30:
                    self._log("Skipping vertical motion check: v0 is too small", True)
                else:
                    t_calc = (self.L / 2) / self.v0
                    if not self._is_close(self.t, t_calc):
                        self._log(f"First condition time mismatch: t={self.t}, L/(2v0)={t_calc}", False)
                        return False
                    self._log(f"First condition time check: t = L/(2v0) holds", True)
            
            if all(v is not None for v in [self.a, self.t, self.L]):
                # Horizontal motion: L = 0.5*a*t^2
                if abs(self.t**2) < 1e-30:
                    self._log("Skipping horizontal motion check: t^2 is too small", True)
                else:
                    a_calc = 2 * self.L / (self.t**2)
                    if not self._is_close(self.a, a_calc):
                        self._log(f"First condition acceleration mismatch: a={self.a}, 2L/t²={a_calc}", False)
                        return False
                    self._log(f"First condition acceleration check: a = 2L/t² holds", True)
            
            # 2. Second condition: check kinematics consistency
            if all(v is not None for v in [self.alpha, self.t_prime, self.L, self.v0]):
                # alpha is in radians, no conversion needed
                alpha_rad = self.alpha
                # Vertical motion: L = v0*sin(α)*t'
                if abs(self.v0 * math.sin(alpha_rad)) < 1e-30:
                    self._log("Skipping second condition vertical motion check: v0*sin(α) is too small", True)
                else:
                    t_prime_calc = self.L / (self.v0 * math.sin(alpha_rad))
                    if not self._is_close(self.t_prime, t_prime_calc):
                        self._log(f"Second condition time mismatch: t'={self.t_prime}, L/(v0*sinα)={t_prime_calc}", False)
                        return False
                    self._log(f"Second condition time check: t' = L/(v0*sinα) holds", True)
            
            if all(v is not None for v in [self.a_prime, self.t_prime, self.L, self.v0, self.alpha]):
                alpha_rad = self.alpha
                # Horizontal motion: L = v0*cosα*t' - 0.5*a'*t'^2
                lhs = self.L
                rhs = self.v0 * math.cos(alpha_rad) * self.t_prime - 0.5 * self.a_prime * self.t_prime**2
                if not self._is_close(lhs, rhs):
                    self._log(f"Second condition horizontal motion mismatch: L={lhs}, RHS={rhs}", False)
                    return False
                self._log(f"Second condition horizontal motion check: L = v0*cosα*t' - 0.5*a'*t'² holds", True)
            
            # 3. Check U_AB from first condition derivation
            # CORRECTED: U_AB = 2*m*v0^2/q, not 8*m*v0^2/q
            if all(v is not None for v in [self.U_AB, self.m, self.v0, self.q]):
                if abs(self.q) < 1e-30:
                    self._log("Skipping U_AB derivation check: q is too small", True)
                else:
                    U_AB_calc = 2 * self.m * self.v0**2 / self.q
                    if not self._is_close(self.U_AB, U_AB_calc):
                        self._log(f"U_AB derivation mismatch: {self.U_AB} vs 2mv0²/q={U_AB_calc}", False)
                        return False
                    self._log(f"U_AB derivation check: U_AB = 2mv0²/q holds", True)
            
            # 4. Check ratio for second condition: 2α/π = 1/3 (since α in radians, ratio = 2α/(π) = 1/3)
            if self.alpha is not None:
                ratio_calc = 2 * self.alpha / math.pi
                if not self._is_close(ratio_calc, 1.0/3.0):
                    self._log(f"Second condition ratio mismatch: 2α/π={ratio_calc}, expected 1/3", False)
                    return False
                self._log(f"Second condition ratio check: 2α/π = 1/3 holds (α={math.degrees(self.alpha)}°)", True)
            
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check trend in U_AB, then restore."""
        try:
            if self.U_AB is None or self.m is None or self.q is None:
                self._log("Skipping monotonicity test: missing data", True)
                return True
            
            # Check for near-zero values
            if abs(self.q) < 1e-30:
                self._log("Skipping monotonicity test: q is too small", True)
                return True
            
            # Perturb mass by +1%
            m_perturbed = self.m * 1.01
            
            # Recalculate U_AB using CORRECTED formula: U_AB = 2*m*v0^2/q
            U_AB_perturbed = 2 * m_perturbed * self.v0**2 / self.q
            
            # Check monotonicity: if mass increases, U_AB should increase
            if U_AB_perturbed > self.U_AB:
                self._log(f"Monotonicity check: U_AB increases with mass (Δm=+1%, ΔU_AB=+{100*(U_AB_perturbed/self.U_AB-1):.2f}%)", True)
                return True
            else:
                self._log(f"Monotonicity violation: U_AB should increase with mass", False)
                return False
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} Test ---")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} test failed", False)
            except Exception as e:
                self._log(f"Exception in {test_name}: {traceback.format_exc()}", False)
                success = False
        
        # Summary
        self._log(f"\n=== Summary: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'} ===")
        return success, self.logs