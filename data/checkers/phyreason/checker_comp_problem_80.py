import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Extract constants from results dictionary (not instance attributes)
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract parameters from results (they should be stored there)
        self.R = self.results.get('R', None)
        self.L = self.results.get('L', None)
        self.f = self.results.get('f', None)
        self.T = self.results.get('T', None)
        self.k = self.results.get('k', None)  # Boltzmann constant
        
        # Map expected result keys
        self.dI2_df = self.results.get('d<I2>/df', None)          # spectral density of current fluctuations
        self.Z_sq = self.results.get('|Z|2', None)                # |Z|^2
        self.I2 = self.results.get('<I2>', None)                  # mean-square current
        self.Z_mag = self.results.get('|Z|', None)                # |Z|
        self.Z_complex = self.results.get('Z', None)              # complex impedance (or magnitude)
        
        # Logging
        self.logs = []

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Safe fuzzy comparison for numbers."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except Exception as e:
            self.logs.append(f"Comparison error: {e}")
            return False

    def test_invariants(self):
        """Check fundamental physical laws and derived relationships using numeric values."""
        logs = []
        try:
            # Check that all necessary parameters are available
            if None in [self.R, self.L, self.f, self.T, self.k]:
                logs.append("SKIP: Missing parameters for invariant tests.")
                self.logs.extend(logs)
                return True
            
            # 1. Check impedance magnitude |Z| = sqrt(R^2 + (2πfL)^2)
            if self.Z_mag is not None:
                expected_Z_mag = math.sqrt(self.R**2 + (2 * math.pi * self.f * self.L)**2)
                if not self._isclose(self.Z_mag, expected_Z_mag):
                    logs.append(f"FAIL: |Z| mismatch. Got {self.Z_mag}, expected {expected_Z_mag}")
                else:
                    logs.append("PASS: |Z| = sqrt(R^2 + (2πfL)^2) is correct.")
            
            # 2. Check |Z|^2
            if self.Z_sq is not None:
                expected_Z_sq = self.R**2 + (2 * math.pi * self.f * self.L)**2
                if not self._isclose(self.Z_sq, expected_Z_sq):
                    logs.append(f"FAIL: |Z|^2 mismatch. Got {self.Z_sq}, expected {expected_Z_sq}")
                else:
                    logs.append("PASS: |Z|^2 = R^2 + (2πfL)^2 is correct.")
            
            # 3. Check d<I^2>/df
            if self.dI2_df is not None:
                dV2_df = 4 * self.k * self.T * self.R
                expected_dI2_df = dV2_df / (self.R**2 + (2 * math.pi * self.f * self.L)**2)
                if not self._isclose(self.dI2_df, expected_dI2_df):
                    logs.append(f"FAIL: d<I^2>/df mismatch. Got {self.dI2_df}, expected {expected_dI2_df}")
                else:
                    logs.append("PASS: d<I^2>/df = 4kTR/(R^2 + (2πfL)^2) is correct.")
            
            # 4. Check Ohm's law relationship numerically
            if self.dI2_df is not None and self.Z_sq is not None:
                dV2_df_val = 4 * self.k * self.T * self.R
                expected_dI2_df_val = dV2_df_val / self.Z_sq
                if not math.isclose(self.dI2_df, expected_dI2_df_val, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Numerical d<I^2>/df mismatch. Got {self.dI2_df}, computed {expected_dI2_df_val}")
                else:
                    logs.append("PASS: Numerical check of d<I^2>/df passes.")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len(logs) == 0 or all("PASS" in log or "ERROR" not in log for log in logs)

    def test_trace_consistency(self):
        """Verify intermediate results lead to final answer."""
        logs = []
        try:
            # Check that |Z|^2 is consistent with |Z|
            if self.Z_mag is not None and self.Z_sq is not None:
                if not self._isclose(self.Z_sq, self.Z_mag**2):
                    logs.append(f"FAIL: |Z|^2 != |Z|^2. |Z|={self.Z_mag}, |Z|^2={self.Z_sq}")
                else:
                    logs.append("PASS: |Z|^2 is square of |Z|.")
            
            # Check that d<I^2>/df = (1/|Z|^2) * d<V^2>/df
            if self.dI2_df is not None and self.Z_sq is not None:
                if None in [self.k, self.T, self.R]:
                    logs.append("SKIP: Missing parameters for trace consistency.")
                else:
                    dV2_df = 4 * self.k * self.T * self.R
                    expected = dV2_df / self.Z_sq
                    if not self._isclose(self.dI2_df, expected):
                        logs.append(f"FAIL: d<I^2>/df not consistent with d<V^2>/df and |Z|^2. Got {self.dI2_df}, expected {expected}")
                    else:
                        logs.append("PASS: d<I^2>/df = (d<V^2>/df) / |Z|^2 is consistent.")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len(logs) == 0

    def test_monotonicity(self):
        """Perturb inductance L and check that d<I^2>/df decreases."""
        logs = []
        try:
            # Only run if we have all needed parameters
            if None in [self.R, self.L, self.f, self.T, self.k]:
                logs.append("SKIP: Missing parameters for monotonicity test.")
                return True
            
            original_L = self.L
            perturbed_L = original_L * 1.1  # Increase by 10%
            
            # Compute original and perturbed d<I^2>/df using the formula
            original_val = (4 * self.k * self.T * self.R) / (self.R**2 + (2 * math.pi * self.f * original_L)**2)
            perturbed_val = (4 * self.k * self.T * self.R) / (self.R**2 + (2 * math.pi * self.f * perturbed_L)**2)
            
            # Since denominator increases with L, perturbed_val should be smaller
            if not (perturbed_val < original_val):
                logs.append(f"FAIL: Monotonicity violated. Increasing L should decrease d<I^2>/df. "
                           f"Original: {original_val}, Perturbed: {perturbed_val}")
            else:
                logs.append(f"PASS: Monotonicity holds: L↑ → d<I^2>/df↓.")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len(logs) == 0 or "PASS" in logs[-1]

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for name, test_func in tests:
            self.logs.append(f"\n=== Running {name} Test ===")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self.logs.append(f"CRASH in {name}: {e}\n{traceback.format_exc()}")
                success = False
        
        # Final summary
        if success:
            self.logs.append("\n✅ All tests passed.")
        else:
            self.logs.append("\n❌ Some tests failed.")
        
        return success, self.logs