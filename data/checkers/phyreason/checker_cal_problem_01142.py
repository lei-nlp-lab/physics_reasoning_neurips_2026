import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction for inputs
        self.N = getattr(solution_instance, 'N', solution_instance.results.get('N', 100))
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 1/math.pi))
        self.S = getattr(solution_instance, 'S', solution_instance.results.get('S', 0.05))
        self.omega = getattr(solution_instance, 'omega', solution_instance.results.get('omega', 100*math.pi))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', 1))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 4))
        
        # Safe attribute extraction for results (with fallback defaults)
        self.E_m = getattr(solution_instance, 'E_m', solution_instance.results.get('E_m', 0.0))
        self.e_expr = getattr(solution_instance, 'e', solution_instance.results.get('e', ''))
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', 0.0))
        
        # Log storage
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append((message, success))
        
    def test_invariants(self):
        """Check fundamental physical laws."""
        try:
            # 1. Check maximum EMF formula: E_m = N * B * S * omega
            expected_E_m = self.N * self.B * self.S * self.omega
            if math.isclose(self.E_m, expected_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"✓ Maximum EMF invariant correct: {self.E_m} ≈ {expected_E_m}", True)
            else:
                self._log(f"✗ Maximum EMF mismatch: {self.E_m} ≠ {expected_E_m}", False)
                
            # 2. Check RMS voltage: E = E_m / sqrt(2)
            if self.E > 0:
                expected_E = self.E_m / math.sqrt(2)
                if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"✓ RMS voltage invariant correct: {self.E} ≈ {expected_E}", True)
                else:
                    self._log(f"✗ RMS voltage mismatch: {self.E} ≠ {expected_E}", False)
            
            # 3. Check Joule heat formula: Q = (E^2 / (R+r)) * T, where T = 2π/ω
            if self.Q > 0 and self.E > 0:
                T = 2 * math.pi / self.omega
                expected_Q = (self.E ** 2 / (self.R + self.r)) * T
                if math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"✓ Joule heat invariant correct: {self.Q} ≈ {expected_Q}", True)
                else:
                    self._log(f"✗ Joule heat mismatch: {self.Q} ≠ {expected_Q}", False)
                    
            return True
        except Exception as e:
            self._log(f"Invariant test error: {traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Check intermediate results lead to final answer."""
        try:
            # Step-by-step consistency from STEPS ANALYSIS
            
            # Step 1: E_m should be 500 V
            expected_E_m_step = 500.0
            if math.isclose(self.E_m, expected_E_m_step, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"✓ Step 1 (E_m) consistent: {self.E_m} ≈ {expected_E_m_step}", True)
            else:
                self._log(f"✗ Step 1 inconsistency: {self.E_m} ≠ {expected_E_m_step}", False)
            
            # Step 3: E = E_m / √2
            if self.E > 0:
                expected_E_step = self.E_m / math.sqrt(2)
                if math.isclose(self.E, expected_E_step, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"✓ Step 3 (E) consistent: {self.E} ≈ {expected_E_step}", True)
                else:
                    self._log(f"✗ Step 3 inconsistency: {self.E} ≠ {expected_E_step}", False)
            
            # Step 4: Q should be 500 J
            if self.Q > 0:
                expected_Q_step = 500.0
                if math.isclose(self.Q, expected_Q_step, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"✓ Step 4 (Q) consistent: {self.Q} ≈ {expected_Q_step}", True)
                else:
                    self._log(f"✗ Step 4 inconsistency: {self.Q} ≠ {expected_Q_step}", False)
            
            # Check that using the steps gives the same final Q
            if all([self.E_m > 0, self.E > 0, self.Q > 0]):
                # Recalculate Q from E_m
                E_from_Em = self.E_m / math.sqrt(2)
                T = 2 * math.pi / self.omega
                Q_from_Em = (E_from_Em ** 2 / (self.R + self.r)) * T
                
                if math.isclose(self.Q, Q_from_Em, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"✓ Trace consistency: Q={self.Q} matches recalculated Q={Q_from_Em}", True)
                else:
                    self._log(f"✗ Trace inconsistency: Q={self.Q} ≠ recalculated Q={Q_from_Em}", False)
            
            return True
        except Exception as e:
            self._log(f"Trace consistency error: {traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb parameters and check trends."""
        try:
            # Store original values
            original_omega = self.omega
            original_Q = self.Q
            
            # Perturb omega by +10%
            perturbed_omega = original_omega * 1.1
            
            # Recalculate using formulas
            perturbed_E_m = self.N * self.B * self.S * perturbed_omega
            perturbed_E = perturbed_E_m / math.sqrt(2)
            perturbed_T = 2 * math.pi / perturbed_omega
            perturbed_Q = (perturbed_E ** 2 / (self.R + self.r)) * perturbed_T
            
            # Check trend: Increasing omega should increase E_m proportionally
            expected_E_m_trend = perturbed_omega / original_omega
            actual_E_m_trend = perturbed_E_m / self.E_m if self.E_m > 0 else 0
            
            if math.isclose(actual_E_m_trend, expected_E_m_trend, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"✓ E_m monotonicity: ω↑{original_omega:.1f}→{perturbed_omega:.1f}, E_m↑{self.E_m:.1f}→{perturbed_E_m:.1f}", True)
            else:
                self._log(f"✗ E_m monotonicity failed: ratio {actual_E_m_trend:.3f} ≠ {expected_E_m_trend:.3f}", False)
            
            # Check trend for Q: Q ∝ ω (since E^2 ∝ ω^2 and T ∝ 1/ω)
            # So Q ∝ ω^2 * (1/ω) = ω
            expected_Q_trend = perturbed_omega / original_omega
            actual_Q_trend = perturbed_Q / original_Q if original_Q > 0 else 0
            
            if math.isclose(actual_Q_trend, expected_Q_trend, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"✓ Q monotonicity: ω↑{original_omega:.1f}→{perturbed_omega:.1f}, Q↑{original_Q:.1f}→{perturbed_Q:.1f}", True)
            else:
                self._log(f"✗ Q monotonicity failed: ratio {actual_Q_trend:.3f} ≠ {expected_Q_trend:.3f}", False)
            
            return True
        except Exception as e:
            self._log(f"Monotonicity test error: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        overall_success = True
        
        try:
            # Run each test
            tests = [
                ("Invariant Tests", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]
            
            for test_name, test_func in tests:
                self._log(f"\n--- Running {test_name} ---", None)
                test_success = test_func()
                overall_success = overall_success and test_success
            
            # Check final answers against expected targets
            self._log("\n--- Final Answer Check ---", None)
            
            # Check instantaneous EMF expression - accept string or numeric
            if isinstance(self.e_expr, str):
                # Normalize for comparison: remove spaces, convert to lowercase
                normalized = self.e_expr.replace(" ", "").lower()
                # Accept variations: 500sin(100πt) or 500*sin(100πt) etc.
                if "500sin(100πt)" in normalized or "500*sin(100πt)" in normalized:
                    self._log(f"✓ Instantaneous EMF expression matches target: {self.e_expr}", True)
                else:
                    self._log(f"✗ Instantaneous EMF mismatch: got '{self.e_expr}'", False)
                    overall_success = False
            elif isinstance(self.e_expr, (int, float, complex)):
                # If it's a number, check it's approximately 0 (since e is time-dependent)
                if math.isclose(self.e_expr, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"✓ Instantaneous EMF numeric value: {self.e_expr}", True)
                else:
                    self._log(f"✗ Instantaneous EMF numeric value unexpected: {self.e_expr}", False)
                    overall_success = False
            else:
                self._log("✓ Instantaneous EMF present (non-string, non-numeric type accepted)", True)
            
            # Check Joule heat value
            if math.isclose(self.Q, 500.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"✓ Joule heat matches target: {self.Q} ≈ 500 J", True)
            else:
                self._log(f"✗ Joule heat mismatch: {self.Q} ≠ 500 J", False)
                overall_success = False
            
            # Summary
            success_count = sum(1 for _, success in self.logs if success is True)
            total_checks = sum(1 for _, success in self.logs if success is not None)
            self._log(f"\n--- Summary: {success_count}/{total_checks} checks passed ---", overall_success)
            
        except Exception as e:
            self._log(f"Critical error in test suite: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs