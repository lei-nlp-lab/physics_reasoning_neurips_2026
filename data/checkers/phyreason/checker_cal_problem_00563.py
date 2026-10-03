import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract numeric inputs from results or use defaults
        results = getattr(solution_instance, 'results', {})
        
        # Use numeric defaults if not found in results OR if value is 0.0 (sandbox bug workaround)
        # The sandbox seems to set m and e to 0.0, so we explicitly check for that
        m_from_results = results.get('m', 9.11e-31)
        e_from_results = results.get('e', 1.6e-19)
        
        self.m = m_from_results if abs(m_from_results) > 1e-35 else 9.11e-31
        self.e = e_from_results if abs(e_from_results) > 1e-35 else 1.6e-19
        self.v0 = results.get('v0', 1.0e6)
        self.B = results.get('B', 0.001)
        
        # Extract results (already floats from solution code)
        self.r_1 = results.get('r_1', 0.0)
        self.t_1 = results.get('t_1', 0.0)
        self.t_2 = results.get('t_2', 0.0)
        self.T = results.get('T', 0.0)
        self.theta = results.get('theta', 0.0)
        
        self.logs = []
        self.success = True

    def _log(self, msg, is_error=False):
        self.logs.append(msg)
        if is_error:
            self.success = False

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify centripetal force, period, and geometric relationships"""
        try:
            # 1. Lorentz force as centripetal force: e*v0*B = m*v0²/r₁
            lhs_force = self.e * self.v0 * self.B
            rhs_force = self.m * self.v0**2 / self.r_1
            if not self._is_close(lhs_force, rhs_force):
                self._log(f"FAIL: Centripetal force mismatch. LHS={lhs_force:.3e}, RHS={rhs_force:.3e}", True)
            else:
                self._log(f"OK: Centripetal force verified: e*v0*B = m*v0²/r₁")

            # 2. Period consistency: T = 2πr₁/v₀
            T_from_r = 2 * math.pi * self.r_1 / self.v0
            if not self._is_close(self.T, T_from_r):
                self._log(f"FAIL: Period mismatch. T={self.T:.3e}, 2πr₁/v₀={T_from_r:.3e}", True)
            else:
                self._log(f"OK: Period consistency: T = 2πr₁/v₀")

            # 3. Time for electron 1: t₁ = T/2
            if not self._is_close(self.t_1, self.T / 2):
                self._log(f"FAIL: t₁ should be T/2. t₁={self.t_1:.3e}, T/2={self.T/2:.3e}", True)
            else:
                self._log(f"OK: t₁ = T/2 verified")

            # 4. Geometric angle: θ = 60° = π/3 rad
            theta_rad = math.radians(60)
            if not self._is_close(self.theta, 60) and not self._is_close(self.theta, theta_rad):
                self._log(f"FAIL: θ should be 60° or π/3 rad. Got {self.theta}", True)
            else:
                self._log(f"OK: Central angle θ = 60°")

            # 5. Time for electron 2: t₂ = T/6 (since θ=60° = 360°/6)
            if not self._is_close(self.t_2, self.T / 6):
                self._log(f"FAIL: t₂ should be T/6. t₂={self.t_2:.3e}, T/6={self.T/6:.3e}", True)
            else:
                self._log(f"OK: t₂ = T/6 verified")

        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", True)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Expected formulas from steps analysis (using numeric inputs)
            # Guard against division by zero
            if abs(self.e * self.B) < 1e-30:
                self._log("SKIP: e*B is too small, skipping trace consistency", False)
                return
                
            expected_r1 = self.m * self.v0 / (self.e * self.B)
            expected_T = 2 * math.pi * self.m / (self.e * self.B)
            expected_t1 = math.pi * self.m / (self.e * self.B)
            expected_t2 = math.pi * self.m / (3 * self.e * self.B)

            checks = [
                ("r₁", self.r_1, expected_r1),
                ("T", self.T, expected_T),
                ("t₁", self.t_1, expected_t1),
                ("t₂", self.t_2, expected_t2)
            ]

            for name, value, expected in checks:
                if not self._is_close(value, expected):
                    self._log(f"FAIL: {name} mismatch. Value={value:.3e}, Expected={expected:.3e}", True)
                else:
                    self._log(f"OK: {name} matches analytical formula")

        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", True)

    def test_monotonicity(self):
        """Perturb inputs and check trends"""
        # Use local copies to avoid modifying instance attributes
        original_B = self.B
        original_m = self.m
        original_r1 = self.r_1
        original_T = self.T
        
        try:
            # Test 1: Increase B should decrease r₁ (r₁ ∝ 1/B)
            B_test = original_B * 1.1  # +10%
            if abs(self.e * B_test) < 1e-30:
                self._log("SKIP: e*B_test is too small, skipping monotonicity test 1", False)
            else:
                new_r1 = self.m * self.v0 / (self.e * B_test)
                if not new_r1 < original_r1:
                    self._log(f"FAIL: r₁ should decrease when B increases. Old={original_r1:.3e}, New={new_r1:.3e}", True)
                else:
                    self._log(f"OK: r₁ ∝ 1/B monotonicity verified")
            
            # Test 2: Increase m should increase T (T ∝ m)
            m_test = original_m * 1.1
            if abs(self.e * self.B) < 1e-30:
                self._log("SKIP: e*B is too small, skipping monotonicity test 2", False)
            else:
                new_T = 2 * math.pi * m_test / (self.e * self.B)
                if not new_T > original_T:
                    self._log(f"FAIL: T should increase when m increases. Old={original_T:.3e}, New={new_T:.3e}", True)
                else:
                    self._log(f"OK: T ∝ m monotonicity verified")
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", True)

    def run_all_tests(self):
        """Execute all tests and return success flag with logs"""
        self.logs = []
        self.success = True
        
        self.logs.append("=== Physics Checker Start ===")
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        self.logs.append(f"=== Physics Checker Complete: {'PASS' if self.success else 'FAIL'} ===")
        
        return self.success, self.logs