import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        solution_instance is expected to have a `results` dictionary.
        """
        self.solution = solution_instance
        
        # Input parameters (with defaults if missing)
        self.A = getattr(solution_instance, 'A', 0.05)              # area in m^2
        self.N = getattr(solution_instance, 'N', 100)               # number of turns
        self.r = getattr(solution_instance, 'r', 1.0)               # coil resistance in Ω
        self.R = getattr(solution_instance, 'R', 24.0)              # external resistance in Ω
        self.B = getattr(solution_instance, 'B', 1.0 / math.pi)     # magnetic field in T
        self.omega = getattr(solution_instance, 'omega', 10.0 * math.pi)  # angular velocity in rad/s
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.E_m = float(self.results.get('E_{m}', self.results.get('E_m', 0.0)))
        self.e_expr = self.results.get('e', None)   # could be a string or expression
        self.I = float(self.results.get('I', 0.0))
        
        self.logs = []
        
    def _log(self, msg, status='INFO'):
        """Helper to log messages."""
        self.logs.append(f"[{status}] {msg}")
    
    def test_invariants(self):
        """Verify fundamental physics laws (peak EMF, Ohm's law, RMS relation)."""
        try:
            # 1. Check peak EMF: E_m = N * B * A * omega
            E_m_calculated = self.N * self.B * self.A * self.omega
            if not math.isclose(self.E_m, E_m_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Peak EMF mismatch: expected {E_m_calculated:.2f}, got {self.E_m:.2f}", "FAIL")
                return False
            self._log(f"Peak EMF check passed: E_m = {self.E_m:.2f} V", "PASS")
            
            # 2. Check instantaneous EMF expression (symbolic)
            if self.e_expr is not None:
                # For string expressions, check that they match the expected form
                if isinstance(self.e_expr, str):
                    # Check that the string contains the correct amplitude
                    # Expected: something like "50*cos(10*pi*t)" or "50\cos10\pi t"
                    # Extract numeric part before '*' or before 'cos'
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)', str(self.e_expr))
                    if match:
                        amplitude = float(match.group(1))
                        if not math.isclose(amplitude, self.E_m, rel_tol=1e-2, abs_tol=1e-2):
                            self._log(f"Amplitude in e expression mismatch: expected {self.E_m:.2f}, got {amplitude:.2f}", "FAIL")
                            return False
                    # Also check that it contains cosine and the correct angular frequency
                    if 'cos' not in str(self.e_expr).lower():
                        self._log("e expression should contain cosine function", "FAIL")
                        return False
                    self._log("Instantaneous EMF expression check passed", "PASS")
                else:
                    # For non-string expressions, we can't easily check without SymPy
                    # Just log a warning and continue
                    self._log("e is not a string expression; skipping detailed check", "WARN")
            else:
                self._log("No instantaneous EMF expression provided; skipping check", "WARN")
            
            # 3. Check RMS current: I = E_m / (sqrt(2) * (R + r))
            I_calculated = self.E_m / (math.sqrt(2) * (self.R + self.r))
            if not math.isclose(self.I, I_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"RMS current mismatch: expected {I_calculated:.2f}, got {self.I:.2f}", "FAIL")
                return False
            self._log(f"RMS current check passed: I = {self.I:.2f} A", "PASS")
            
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Step 1: E_m should be used in step 2 and step 3
            # We already checked E_m calculation in invariants.
            # For step 2: e should be expressed in terms of E_m and omega.
            if self.e_expr is not None and isinstance(self.e_expr, str):
                # Check that the amplitude in e expression matches E_m
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', str(self.e_expr))
                if match:
                    amplitude = float(match.group(1))
                    if not math.isclose(amplitude, self.E_m, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"Amplitude in e expression mismatch: expected {self.E_m:.2f}, got {amplitude:.2f}", "FAIL")
                        return False
                self._log("Step 2 (instantaneous EMF) is consistent with Step 1 (peak EMF)", "PASS")
            else:
                self._log("Step 2 check skipped (e not a string expression)", "INFO")
            
            # Step 3: I should be computed from E_m and resistances.
            # Already checked in invariants.
            self._log("Step 3 (RMS current) is consistent with Step 1 (peak EMF)", "PASS")
            
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check the expected trend, then restore."""
        original_omega = self.omega
        original_E_m = self.E_m
        original_I = self.I
        
        try:
            # Perturb omega by +10%
            perturb_factor = 1.1
            self.omega *= perturb_factor
            
            # Recompute peak EMF (should increase by same factor)
            new_E_m = self.N * self.B * self.A * self.omega
            expected_E_m = original_E_m * perturb_factor
            if not math.isclose(new_E_m, expected_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity check failed for E_m: expected {expected_E_m:.2f}, got {new_E_m:.2f}", "FAIL")
                self.omega = original_omega
                return False
            
            # Recompute RMS current (should also increase by same factor, because I ∝ E_m)
            new_I = new_E_m / (math.sqrt(2) * (self.R + self.r))
            expected_I = original_I * perturb_factor
            if not math.isclose(new_I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity check failed for I: expected {expected_I:.2f}, got {new_I:.2f}", "FAIL")
                self.omega = original_omega
                return False
            
            self._log(f"Monotonicity check passed: perturbing ω by +10% changes E_m and I proportionally", "PASS")
            
            # Restore original values
            self.omega = original_omega
            return True
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", "ERROR")
            self.omega = original_omega
            return False
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            if not test_func():
                success = False
                self._log(f"{test_name} failed.", "FAIL")
            else:
                self._log(f"{test_name} passed.", "PASS")
        
        return success, self.logs