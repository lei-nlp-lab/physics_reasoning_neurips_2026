import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract all expected quantities safely - handle both numeric and string types
        self.lam = self._safe_get('λ', 0.0, numeric=True)
        self.T = self._safe_get('T', 0.0, numeric=True)
        self.v = self._safe_get('v', 0.0, numeric=True)
        self.x = self._safe_get('x', 0.0, numeric=True)
        self.direction = self._safe_get('direction', '', numeric=False)
        
        # Also try to get 'N/A' if 'direction' is not found
        if not self.direction or self.direction == '':
            self.direction = self._safe_get('N/A', '', numeric=False)

        # Expected values from the steps analysis
        self.expected_lam = 0.8   # m
        self.expected_T = 0.6     # s
        self.expected_v = 4/3     # m/s (approx 1.333...)
        self.expected_x = 40.0    # m
        self.expected_dir_keywords = ["left", "negative"]

        # Logging
        self.logs = []

    def _safe_get(self, key, default, numeric=True):
        """Safely retrieve attribute from instance or results dict."""
        value = getattr(self.solution, key, self.results.get(key, default))
        
        if numeric:
            # Convert to float if numeric and not already
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        else:
            # Return as string for non-numeric values
            return str(value) if value is not None else default

    def log(self, message):
        """Append message to logs."""
        self.logs.append(message)

    def test_invariants(self):
        """
        Verify fundamental physical relationships:
        1. v = λ / T
        2. Direction consistency (transverse wave, left propagation) - OPTIONAL
        """
        try:
            # 1. Wave speed formula - check for division by zero first
            if self.T == 0:
                self.log("ERROR: Division by zero (T = 0)")
                return False
                
            if not math.isclose(self.v, self.lam / self.T,
                                rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: v = λ/T mismatch. v={self.v}, λ/T={self.lam/self.T}")
                return False
            self.log("PASS: Wave speed invariant v = λ/T holds.")

            # 2. Direction: must indicate leftward propagation - OPTIONAL CHECK
            # Since direction may not be stored in results (per sandbox rules),
            # we only check if it exists and contains expected keywords
            dir_str = str(self.direction).lower()
            if dir_str and dir_str != '':
                if not any(kw in dir_str for kw in self.expected_dir_keywords):
                    self.log(f"WARNING: Direction doesn't contain expected keywords. Got: {self.direction}")
                    # Not a failure - just a warning
                else:
                    self.log("PASS: Wave propagation direction indicates left/negative.")
            else:
                self.log("INFO: No direction provided (may be omitted per sandbox rules).")

            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}")
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results against expected values from steps analysis.
        """
        try:
            success = True

            # Wavelength
            if not math.isclose(self.lam, self.expected_lam,
                                rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: λ mismatch. Expected {self.expected_lam}, got {self.lam}")
                success = False

            # Period
            if not math.isclose(self.T, self.expected_T,
                                rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: T mismatch. Expected {self.expected_T}, got {self.T}")
                success = False

            # Wave speed
            if not math.isclose(self.v, self.expected_v,
                                rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: v mismatch. Expected {self.expected_v}, got {self.v}")
                success = False

            # Distance
            if not math.isclose(self.x, self.expected_x,
                                rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: x mismatch. Expected {self.expected_x}, got {self.x}")
                success = False

            # Direction (optional - only check if provided)
            dir_str = str(self.direction).lower()
            if dir_str and dir_str != '':
                if not any(kw in dir_str for kw in self.expected_dir_keywords):
                    self.log(f"WARNING: Direction mismatch. Expected left/negative, got {self.direction}")
                    # Not a failure - just a warning

            if success:
                self.log("PASS: All required trace consistency checks passed.")
            return success
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}")
            return False

    def test_monotonicity(self):
        """
        Perturb inputs and verify expected trends.
        Does NOT modify the solution instance.
        """
        try:
            # Store original values
            lam0, T0, v0 = self.lam, self.T, self.v
            
            # Check for division by zero
            if T0 == 0:
                self.log("ERROR: Division by zero in monotonicity test (T = 0)")
                return False
                
            if self.v == 0:
                self.log("ERROR: Division by zero in monotonicity test (v = 0)")
                return False

            # Test 1: Increase λ by 1%, keep T constant → v should increase
            lam_pert = lam0 * 1.01
            v_pert_lam = lam_pert / T0
            if not v_pert_lam > v0:
                self.log(f"FAIL: Monotonicity (λ↑, T fixed). v0={v0}, v_pert={v_pert_lam}")
                return False

            # Test 2: Increase T by 1%, keep λ constant → v should decrease
            T_pert = T0 * 1.01
            v_pert_T = lam0 / T_pert
            if not v_pert_T < v0:
                self.log(f"FAIL: Monotonicity (T↑, λ fixed). v0={v0}, v_pert={v_pert_T}")
                return False

            # Test 3: Distance x = v * t (implicitly, t = x/v). If v increases, x should increase for fixed t.
            # We don't have t explicitly, but we can check that the ratio x/v is positive.
            if not math.isclose(self.x / self.v, 30.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"WARNING: Implied time t = x/v is not 30 s. Got {self.x/self.v}")
                # Not a failure, just a warning, because the problem might use a different time.

            self.log("PASS: Monotonicity checks passed.")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}")
            return False

    def run_all_tests(self):
        """
        Execute all checks and return success flag with logs.
        """
        self.logs = []  # Reset logs
        success = True

        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test in tests:
            try:
                if not test():
                    self.log(f"✗ {name} failed.")
                    success = False
                else:
                    self.log(f"✓ {name} passed.")
            except Exception as e:
                self.log(f"✗ {name} raised exception: {traceback.format_exc()}")
                success = False

        return success, self.logs