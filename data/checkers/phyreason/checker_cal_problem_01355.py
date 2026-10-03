import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract inputs and results safely from solution_instance."""
        self.solution = solution_instance
        
        # Input parameters (with defaults if missing)
        self.m = getattr(solution_instance, 'm', 50.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.h = getattr(solution_instance, 'h', 1.4)
        self.H = getattr(solution_instance, 'H', 1.8)
        self.v_P = getattr(solution_instance, 'v_P', 6.0)
        self.l = getattr(solution_instance, 'l', 7.0)
        
        # Results from solution (using results dict as primary source)
        results = getattr(solution_instance, 'results', {})
        self.v_B = results.get('v_B', getattr(solution_instance, 'v_B', 0.0))
        self.v_C = results.get('v_C', getattr(solution_instance, 'v_C', 0.0))
        self.mu = results.get('mu', getattr(solution_instance, 'mu', 0.0))
        self.s = results.get('s', getattr(solution_instance, 's', 0.0))
        self.x = results.get('x', getattr(solution_instance, 'x', 0.0))
        
        # Initialize logs
        self.logs = []

    def _log(self, msg, success):
        """Append message and success flag to logs."""
        self.logs.append((msg, success))
        return success

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Safe fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Verify energy conservation and work-energy theorem across segments."""
        try:
            # 1. Energy conservation from P to B (curved track, no friction)
            # ΔKE + ΔPE = 0 → (1/2)m(v_B² - v_P²) - mgh = 0
            left = 0.5*self.m*(self.v_B**2 - self.v_P**2)
            right = self.m*self.g*self.h
            if not self._is_close(left, right):
                return self._log(
                    f"Energy conservation P→B failed: ΔKE={left:.2f}, -ΔPE={right:.2f}",
                    False
                )
            
            # 2. Energy conservation from C to Q (curved track, no friction)
            # (1/2)mv_C² - mgH = 0
            left = 0.5*self.m*self.v_C**2
            right = self.m*self.g*self.H
            if not self._is_close(left, right):
                return self._log(
                    f"Energy conservation C→Q failed: KE_C={left:.2f}, PE_Q={right:.2f}",
                    False
                )
            
            # 3. Work-energy theorem on B→C (horizontal with friction)
            # W_friction = ΔKE → -μ m g l = (1/2)m(v_C² - v_B²)
            w_fric = -self.mu * self.m * self.g * self.l
            delta_ke = 0.5*self.m*(self.v_C**2 - self.v_B**2)
            if not self._is_close(w_fric, delta_ke):
                return self._log(
                    f"Work-energy B→C failed: W_fric={w_fric:.2f}, ΔKE={delta_ke:.2f}",
                    False
                )
            
            # 4. Overall work-energy from P to final stop (on BC)
            # Initial energy = (1/2)m v_P² + mgh
            # Work by friction = -μ m g s
            # Final energy = 0
            initial_energy = 0.5*self.m*self.v_P**2 + self.m*self.g*self.h
            work_fric_total = -self.mu * self.m * self.g * self.s
            if not self._is_close(work_fric_total, -initial_energy):
                return self._log(
                    f"Overall work-energy failed: W_fric_total={work_fric_total:.2f}, "
                    f"-E_initial={-initial_energy:.2f}",
                    False
                )
            
            # All passed
            self._log("All invariant tests passed", True)
            return True
            
        except Exception as e:
            return self._log(f"Invariant test error: {str(e)}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Step 1: v_B from v_P and h
            v_B_calc = math.sqrt(self.v_P**2 + 2*self.g*self.h)
            if not self._is_close(self.v_B, v_B_calc):
                return self._log(
                    f"Trace consistency v_B failed: expected {v_B_calc:.2f}, got {self.v_B:.2f}",
                    False
                )
            
            # Step 2: v_C from H
            v_C_calc = math.sqrt(2*self.g*self.H)
            if not self._is_close(self.v_C, v_C_calc):
                return self._log(
                    f"Trace consistency v_C failed: expected {v_C_calc:.2f}, got {self.v_C:.2f}",
                    False
                )
            
            # Step 3: μ from v_B, v_C, and l
            mu_calc = (self.v_B**2 - self.v_C**2) / (2*self.g*self.l)
            if not self._is_close(self.mu, mu_calc):
                return self._log(
                    f"Trace consistency μ failed: expected {mu_calc:.3f}, got {self.mu:.3f}",
                    False
                )
            
            # Step 4: s from overall work-energy
            s_calc = (0.5*self.v_P**2 + self.g*self.h) / (self.mu*self.g)
            if not self._is_close(self.s, s_calc):
                return self._log(
                    f"Trace consistency s failed: expected {s_calc:.2f}, got {self.s:.2f}",
                    False
                )
            
            # Step 5: x from s and l
            # Note: s = total distance on BC, l = length of BC (7m)
            # The athlete travels B→C→B→stop, so s = 2l + x
            x_calc = self.s - 2*self.l
            if not self._is_close(self.x, x_calc):
                return self._log(
                    f"Trace consistency x failed: expected {x_calc:.2f}, got {self.x:.2f}",
                    False
                )
            
            # All passed
            self._log("All trace consistency tests passed", True)
            return True
            
        except Exception as e:
            return self._log(f"Trace consistency test error: {str(e)}", False)

    def test_monotonicity(self):
        """Perturb input h and verify expected trend in results."""
        original_h = self.h
        original_v_B = self.v_B
        original_mu = self.mu
        original_s = self.s
        original_x = self.x
        
        try:
            # Increase h by 10% and recalculate expected results
            new_h = original_h * 1.10
            
            # Recalculate expected values using formulas
            new_v_B = math.sqrt(self.v_P**2 + 2*self.g*new_h)
            new_v_C = math.sqrt(2*self.g*self.H)  # unchanged
            new_mu = (new_v_B**2 - new_v_C**2) / (2*self.g*self.l)
            new_s = (0.5*self.v_P**2 + self.g*new_h) / (new_mu*self.g)
            new_x = new_s - 2*self.l
            
            # Check monotonicity: increasing h should increase v_B and μ
            # s and x may increase or decrease depending on parameters
            if not (new_v_B > original_v_B):
                return self._log("Monotonicity: v_B should increase with h", False)
            if not (new_mu > original_mu):
                return self._log("Monotonicity: μ should increase with h", False)
            # Don't test s and x monotonicity - they're not guaranteed to increase
            
            self._log("Monotonicity test passed", True)
            return True
            
        except Exception as e:
            return self._log(f"Monotonicity test error: {str(e)}", False)

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        success = True
        
        # Run tests in sequence
        tests = [
            ("Invariant tests", self.test_invariants),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self.logs.append((f"{test_name} crashed: {traceback.format_exc()}", False))
                success = False
        
        # Also verify final answers match target
        if not self._is_close(self.v_B, 8.0):
            self.logs.append((f"v_B mismatch: expected 8.0, got {self.v_B:.2f}", False))
            success = False
        if not self._is_close(self.v_C, 6.0):
            self.logs.append((f"v_C mismatch: expected 6.0, got {self.v_C:.2f}", False))
            success = False
        if not self._is_close(self.mu, 0.2):
            self.logs.append((f"μ mismatch: expected 0.2, got {self.mu:.3f}", False))
            success = False
        if not self._is_close(self.x, 2.0):
            self.logs.append((f"x mismatch: expected 2.0, got {self.x:.2f}", False))
            success = False
        
        if success:
            self.logs.append(("All tests passed successfully", True))
        
        return success, self.logs