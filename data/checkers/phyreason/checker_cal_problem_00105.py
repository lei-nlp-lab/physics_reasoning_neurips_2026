import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all inputs and results from the solution_instance.
        Uses getattr and .get() to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        
        # Extract inputs safely
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Physics parameters
        self.L = self.inputs.get('L', None)
        self.m = self.inputs.get('m', None)
        self.R0 = self.inputs.get('R0', None)
        self.B = self.inputs.get('B', None)
        
        # Results mapping (using safe access)
        self.v0 = self.results.get('$v_0$', None)
        self.q1 = self.results.get('$q_1$', None)
        self.q = self.results.get('$q$', None)
        self.v1 = self.results.get('$v_1$', None)
        self.Q1 = self.results.get('$Q_1$', None)
        self.Q_R1 = self.results.get('$Q_R_1$', None)
        self.v2 = self.results.get('$v_2$', None)
        self.Q2 = self.results.get('$Q_2$', None)
        self.Q_R1_prime = self.results.get('$Q_R_1prime$', None)
        self.Q_R1_all = self.results.get('$Q_R_1,all$', None)
        self.R_all = self.results.get('$R_all$', None)
        self.R_all_prime = self.results.get('$R_allprime$', None)
        self.hatE = self.results.get('$hatE$', None)
        
        # Track logs for all tests
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def _check_numeric(self, val1, val2, context=""):
        """
        Fuzzy numerical comparison with 2% tolerance.
        Returns True if values match within tolerance.
        """
        if val1 is None or val2 is None:
            self._log(f"{context}: Missing value for comparison", False)
            return False
        
        try:
            # Convert to float if they aren't already
            v1 = float(val1) if not isinstance(val1, (int, float)) else val1
            v2 = float(val2) if not isinstance(val2, (int, float)) else val2
            
            if math.isclose(v1, v2, rel_tol=0.02, abs_tol=0.02):
                self._log(f"{context}: {v1:.4e} ≈ {v2:.4e}", True)
                return True
            else:
                self._log(f"{context}: {v1:.4e} ≠ {v2:.4e} (diff: {abs(v1-v2)/v2*100:.2f}%)", False)
                return False
        except Exception as e:
            self._log(f"{context}: Numeric check failed: {e}", False)
            return False
    
    def _check_symbolic(self, expr1, expr2, context=""):
        """
        Symbolic equality check using sympy.
        Returns True if expressions are mathematically equivalent.
        """
        try:
            # Parse if strings, otherwise use as-is
            if isinstance(expr1, str):
                expr1 = sp.sympify(expr1.replace('\\', ''))
            if isinstance(expr2, str):
                expr2 = sp.sympify(expr2.replace('\\', ''))
            
            # Simplify difference
            diff = sp.simplify(expr1 - expr2)
            if diff == 0:
                self._log(f"{context}: Symbolic match ✓", True)
                return True
            else:
                self._log(f"{context}: Symbolic mismatch: {expr1} ≠ {expr2}", False)
                return False
        except Exception as e:
            self._log(f"{context}: Symbolic check failed: {e}", False)
            return False
    
    def test_invariants(self):
        """Test physical invariants: Energy conservation and impulse-momentum."""
        self._log("=== Testing Physical Invariants ===")
        success = True
        
        # 1. Energy conservation during entry (step_9)
        if all(v is not None for v in [self.m, self.v0, self.v1, self.Q1]):
            try:
                K0 = 0.5 * self.m * self.v0**2
                K1 = 0.5 * self.m * self.v1**2
                energy_diff = K0 - K1
                
                if math.isclose(energy_diff, self.Q1, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Energy conservation (entry): ΔK = {energy_diff:.4e} = Q1 = {self.Q1:.4e}", True)
                else:
                    self._log(f"Energy conservation (entry) violated: ΔK={energy_diff:.4e}, Q1={self.Q1:.4e}", False)
                    success = False
            except Exception as e:
                self._log(f"Energy conservation (entry) failed: {e}", False)
                success = False
        
        # 2. Energy conservation during leaving (step_13)
        if all(v is not None for v in [self.m, self.v1, self.Q2]):
            try:
                K1 = 0.5 * self.m * self.v1**2
                
                if math.isclose(K1, self.Q2, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Energy conservation (leaving): K1 = {K1:.4e} = Q2 = {self.Q2:.4e}", True)
                else:
                    self._log(f"Energy conservation (leaving) violated: K1={K1:.4e}, Q2={self.Q2:.4e}", False)
                    success = False
            except Exception as e:
                self._log(f"Energy conservation (leaving) failed: {e}", False)
                success = False
        
        # 3. Impulse-momentum during entry (step_4)
        if all(v is not None for v in [self.B, self.q, self.L, self.m, self.v0]):
            try:
                # Impulse from magnetic force: -B * q * L
                impulse = -self.B * self.q * self.L
                # Change in momentum: m*(v1 - v0) but v1 not available in step_4
                # Actually step_4 gives: -B*q*L = m*(v0/2 - v0) = -m*v0/2
                momentum_change = -self.m * self.v0 / 2
                
                if math.isclose(impulse, momentum_change, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Impulse-momentum (entry step4): J = {impulse:.4e} = Δp = {momentum_change:.4e}", True)
                else:
                    self._log(f"Impulse-momentum (entry step4) violated: J={impulse:.4e}, Δp={momentum_change:.4e}", False)
                    success = False
            except Exception as e:
                self._log(f"Impulse-momentum (entry step4) failed: {e}", False)
                success = False
        
        # 4. Impulse-momentum during entry (step_7)
        if all(v is not None for v in [self.B, self.L, self.R_all, self.m, self.v0, self.v1]):
            try:
                impulse = -(self.B**2 * self.L**3) / self.R_all
                momentum_change = self.m * (self.v1 - self.v0)
                
                if math.isclose(impulse, momentum_change, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Impulse-momentum (entry step7): J = {impulse:.4e} = Δp = {momentum_change:.4e}", True)
                else:
                    self._log(f"Impulse-momentum (entry step7) violated: J={impulse:.4e}, Δp={momentum_change:.4e}", False)
                    success = False
            except Exception as e:
                self._log(f"Impulse-momentum (entry step7) failed: {e}", False)
                success = False
        
        # 5. Impulse-momentum during leaving (step_12)
        if all(v is not None for v in [self.B, self.L, self.R_all_prime, self.m, self.v1, self.v2]):
            try:
                impulse = -(self.B**2 * self.L**3) / self.R_all_prime
                momentum_change = self.m * (self.v2 - self.v1)
                
                if math.isclose(impulse, momentum_change, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Impulse-momentum (leaving): J = {impulse:.4e} = Δp = {momentum_change:.4e}", True)
                else:
                    self._log(f"Impulse-momentum (leaving) violated: J={impulse:.4e}, Δp={momentum_change:.4e}", False)
                    success = False
            except Exception as e:
                self._log(f"Impulse-momentum (leaving) failed: {e}", False)
                success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        self._log("=== Testing Trace Consistency ===")
        success = True
        
        # Check that all necessary values are present
        required = [self.L, self.m, self.R0, self.B, self.v0, self.Q_R1_all]
        if any(v is None for v in required):
            self._log("Missing required values for trace consistency", False)
            return False
        
        # 1. Check q1 formula (step_2)
        expected_q1 = (self.B * self.L**2) / (4 * self.R0)
        if self.q1 is not None:
            if not self._check_numeric(self.q1, expected_q1, "q1 consistency"):
                success = False
        else:
            self._log("q1 not available for check", False)
            success = False
        
        # 2. Check total q (step_3)
        expected_q = 2 * expected_q1
        if self.q is not None:
            if not self._check_numeric(self.q, expected_q, "q consistency"):
                success = False
        else:
            # Calculate from q1 if available
            if self.q1 is not None:
                expected_q_from_q1 = 2 * self.q1
                if not self._check_numeric(expected_q, expected_q_from_q1, "q from q1"):
                    success = False
        
        # 3. Check v0 from impulse-momentum (step_5)
        # From step_4: -B*q*L = m*(v0/2 - v0) = -m*v0/2 => v0 = 2*B*q*L/m
        if self.q is not None:
            expected_v0_from_q = 2 * self.B * self.q * self.L / self.m
            if not self._check_numeric(self.v0, expected_v0_from_q, "v0 from q"):
                success = False
        
        # 4. Check R_all (step_6)
        expected_R_all = (5/3) * self.R0
        if self.R_all is not None:
            if not self._check_numeric(self.R_all, expected_R_all, "R_all"):
                success = False
        
        # 5. Check v1 (step_8)
        expected_v1 = (2 * self.B**2 * self.L**3) / (5 * self.m * self.R0)
        if self.v1 is not None:
            if not self._check_numeric(self.v1, expected_v1, "v1"):
                success = False
        
        # 6. Check Q1 (step_9)
        expected_Q1 = (21 * self.B**4 * self.L**6) / (50 * self.m * self.R0**2)
        if self.Q1 is not None:
            if not self._check_numeric(self.Q1, expected_Q1, "Q1"):
                success = False
        
        # 7. Check Q_R1 (step_10)
        if self.Q1 is not None and self.Q_R1 is not None:
            expected_Q_R1 = (2/15) * self.Q1
            if not self._check_numeric(self.Q_R1, expected_Q_R1, "Q_R1 from Q1"):
                success = False
        
        # 8. Check R_all_prime (step_11)
        expected_R_all_prime = (5/2) * self.R0
        if self.R_all_prime is not None:
            if not self._check_numeric(self.R_all_prime, expected_R_all_prime, "R_all_prime"):
                success = False
        
        # 9. Check v2 (step_12)
        if self.v2 is not None:
            if not self._check_numeric(self.v2, 0, "v2 = 0"):
                success = False
        
        # 10. Check Q2 (step_13)
        expected_Q2 = (2 * self.B**4 * self.L**6) / (25 * self.m * self.R0**2)
        if self.Q2 is not None:
            if not self._check_numeric(self.Q2, expected_Q2, "Q2"):
                success = False
        
        # 11. Check Q_R1_prime (step_13)
        if self.Q2 is not None and self.Q_R1_prime is not None:
            expected_Q_R1_prime = (4/5) * self.Q2
            if not self._check_numeric(self.Q_R1_prime, expected_Q_R1_prime, "Q_R1_prime from Q2"):
                success = False
        
        # 12. Check Q_R1_all (step_14,15)
        if self.Q_R1 is not None and self.Q_R1_prime is not None and self.Q_R1_all is not None:
            expected_sum = self.Q_R1 + self.Q_R1_prime
            if not self._check_numeric(self.Q_R1_all, expected_sum, "Q_R1_all sum"):
                success = False
        
        # 13. Final answer check
        expected_v0 = (self.B**2 * self.L**3) / (self.m * self.R0)
        expected_Q_R1_all = (3 * self.B**4 * self.L**6) / (25 * self.m * self.R0**2)
        
        if not self._check_numeric(self.v0, expected_v0, "Final v0"):
            success = False
        if not self._check_numeric(self.Q_R1_all, expected_Q_R1_all, "Final Q_R1_all"):
            success = False
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb mass and check trend.
        v0 ∝ 1/m, Q_R1_all ∝ 1/m
        """
        self._log("=== Testing Monotonicity ===")
        
        if any(v is None for v in [self.L, self.m, self.R0, self.B, self.v0, self.Q_R1_all]):
            self._log("Missing values for monotonicity test", False)
            return False
        
        try:
            # Save original values
            original_m = self.m
            
            # Perturb mass by +10%
            self.m = original_m * 1.10
            
            # Recalculate expected values using formulas
            new_v0 = (self.B**2 * self.L**3) / (self.m * self.R0)
            new_Q_R1_all = (3 * self.B**4 * self.L**6) / (25 * self.m * self.R0**2)
            
            # Check trends
            v0_ratio = new_v0 / self.v0
            expected_v0_ratio = original_m / self.m  # v0 ∝ 1/m
            
            Q_ratio = new_Q_R1_all / self.Q_R1_all
            expected_Q_ratio = original_m / self.m  # Q ∝ 1/m
            
            # Restore original mass
            self.m = original_m
            
            # Check with tolerance
            v0_success = math.isclose(v0_ratio, expected_v0_ratio, rel_tol=0.02, abs_tol=0.02)
            Q_success = math.isclose(Q_ratio, expected_Q_ratio, rel_tol=0.02, abs_tol=0.02)
            
            if v0_success:
                self._log(f"v0 monotonicity: {v0_ratio:.4f} ≈ {expected_v0_ratio:.4f} (m↑10% → v0↓)", True)
            else:
                self._log(f"v0 monotonicity failed: {v0_ratio:.4f} ≠ {expected_v0_ratio:.4f}", False)
            
            if Q_success:
                self._log(f"Q_R1_all monotonicity: {Q_ratio:.4f} ≈ {expected_Q_ratio:.4f} (m↑10% → Q↓)", True)
            else:
                self._log(f"Q_R1_all monotonicity failed: {Q_ratio:.4f} ≠ {expected_Q_ratio:.4f}", False)
            
            return v0_success and Q_success
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {e}", False)
            # Restore mass even if error occurs
            if 'original_m' in locals():
                self.m = original_m
            return False
    
    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []
        
        try:
            # Run tests
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            # Overall success
            overall_success = inv_success and trace_success and mono_success
            
            # Summary
            self._log("=" * 50)
            self._log(f"SUMMARY:")
            self._log(f"  Invariants: {'PASS' if inv_success else 'FAIL'}")
            self._log(f"  Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
            self._log(f"  Monotonicity: {'PASS' if mono_success else 'FAIL'}")
            self._log(f"  OVERALL: {'PASS' if overall_success else 'FAIL'}")
            
            return overall_success, self.logs
            
        except Exception as e:
            self._log(f"Test suite crashed: {traceback.format_exc()}", False)
            return False, self.logs

# Example usage (commented out for safety):
# checker = PhysicsChecker(solution_instance)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)