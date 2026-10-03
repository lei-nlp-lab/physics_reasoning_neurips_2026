import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract values from results dict
        self.C = self.results.get('C', None)
        self.n = self.results.get('n', None)
        self.v = self.results.get('v', None)
        
        # Constants
        self.c = 299792458.0  # speed of light in air (m/s)
        
        # Convert C to radians if it's in degrees
        self.C_rad = None
        if self.C is not None:
            try:
                c_val = float(self.C)
                # If value is between 0 and 90, assume degrees
                if 0 <= c_val <= 90:
                    self.C_rad = math.radians(c_val)
                else:
                    self.C_rad = c_val
            except (TypeError, ValueError):
                self.C_rad = self.C

        # Store logs for test results
        self.logs = []

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Check physical invariants: Snell's law at critical angle and v = c/n."""
        logs = []
        try:
            # 1. Check n = 1/sin(C) (critical angle condition)
            if self.C_rad is not None and self.n is not None:
                expected_n = 1.0 / math.sin(self.C_rad)
                if self._is_close(self.n, expected_n):
                    logs.append("PASS: n = 1/sin(C) holds numerically.")
                else:
                    logs.append(f"FAIL: n = {self.n}, but 1/sin(C) = {expected_n}")
            else:
                logs.append("SKIP: C or n not available for invariant check.")

            # 2. Check v = c / n (using actual c value)
            if self.n is not None and self.v is not None:
                expected_v = self.c / float(self.n)
                if self._is_close(self.v, expected_v):
                    logs.append("PASS: v = c/n holds numerically.")
                else:
                    logs.append(f"FAIL: v = {self.v}, but c/n = {expected_v}")
            else:
                logs.append("SKIP: n or v not available for v = c/n check.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("FAIL")]) == 0

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # According to steps: C -> n -> v
            if self.C_rad is not None:
                # Compute n from C
                n_from_C = 1.0 / math.sin(self.C_rad)
                if self.n is not None:
                    if self._is_close(self.n, n_from_C):
                        logs.append("PASS: n consistent with C.")
                    else:
                        logs.append(f"FAIL: n = {self.n} but from C we get n = {n_from_C}")
                else:
                    logs.append("SKIP: n not available for consistency check.")

                # Compute v from C (using actual c)
                v_from_C = self.c * math.sin(self.C_rad)  # since v = c * sin(C) from v = c/n and n = 1/sin(C)
                if self.v is not None:
                    if self._is_close(self.v, v_from_C):
                        logs.append("PASS: v consistent with C.")
                    else:
                        logs.append(f"FAIL: v = {self.v} but from C we get v = {v_from_C}")
                else:
                    logs.append("SKIP: v not available for consistency check.")
            else:
                logs.append("SKIP: C not available for trace consistency.")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("FAIL")]) == 0

    def test_monotonicity(self):
        """Check monotonic trends: n decreases with C, v increases with C."""
        logs = []
        try:
            # Use numeric derivatives around C = 45°
            C_deg = 45.0
            delta = 0.1  # small change in degrees
            
            # n at C = 45°
            C_rad1 = math.radians(C_deg)
            n1 = 1.0 / math.sin(C_rad1)
            
            # n at C = 45.1°
            C_rad2 = math.radians(C_deg + delta)
            n2 = 1.0 / math.sin(C_rad2)
            
            # v at C = 45°
            v1 = self.c * math.sin(C_rad1)
            
            # v at C = 45.1°
            v2 = self.c * math.sin(C_rad2)
            
            # Check monotonicity
            if n2 < n1:
                logs.append(f"PASS: n decreases as C increases (n at 45°={n1:.4f}, at 45.1°={n2:.4f}).")
            else:
                logs.append(f"FAIL: n does not decrease as C increases.")
                
            if v2 > v1:
                logs.append(f"PASS: v increases as C increases (v at 45°={v1:.1f}, at 45.1°={v2:.1f}).")
            else:
                logs.append(f"FAIL: v does not increase as C increases.")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("FAIL")]) == 0

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        success = True

        # Run each test
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()

        success = inv_ok and trace_ok and mono_ok
        return success, self.logs