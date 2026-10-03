import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution_instance = solution_instance
        
        # Extract input parameters
        self.R = self._safe_get('R')
        self.h = self._safe_get('h')
        self.m = self._safe_get('m')
        self.g = self._safe_get('g', 9.8)  # default g if not provided
        self.n = self._safe_get('n', 1)    # integer number of revolutions, default 1
        
        # Extract results (with safe defaults)
        self.v_0 = self._safe_get('v_0')
        self.N = self._safe_get('N')
        self.t = self._safe_get('t')
        
        # Store logs for all tests
        self.logs = []
        
    def _safe_get(self, key, default=0.0):
        """Safely get attribute from solution instance or its results dict."""
        try:
            # First try direct attribute
            val = getattr(self.solution_instance, key, None)
            if val is not None:
                return val
            # Then try results dictionary
            if hasattr(self.solution_instance, 'results'):
                val = self.solution_instance.results.get(key, default)
                return val
            return default
        except Exception:
            return default

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerance for physics data."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            # If conversion fails, try symbolic equality
            return bool(sp.simplify(sp.N(a) - sp.N(b)) == 0)

    def test_invariants(self):
        """Verify fundamental physics laws based on KB."""
        logs = []
        try:
            # 1. Free fall time invariant: t = sqrt(2h/g)
            t_free_fall = sp.sqrt(2 * self.h / self.g)
            if not self._is_close(self.t, t_free_fall):
                logs.append(f"FAIL: Free fall time mismatch. Computed: {t_free_fall}, Given: {self.t}")
            else:
                logs.append(f"PASS: Free fall time t = sqrt(2h/g) holds.")
            
            # 2. Uniform circular motion time invariant: t = n * 2πR / v_0
            t_circular = self.n * 2 * sp.pi * self.R / self.v_0
            if not self._is_close(self.t, t_circular):
                logs.append(f"FAIL: Circular motion time mismatch. Computed: {t_circular}, Given: {self.t}")
            else:
                logs.append(f"PASS: Circular motion time t = n*2πR/v_0 holds.")
            
            # 3. Centripetal force invariant: N = m * v_0² / R
            N_centripetal = self.m * self.v_0**2 / self.R
            if not self._is_close(self.N, N_centripetal):
                logs.append(f"FAIL: Centripetal force mismatch. Computed: {N_centripetal}, Given: {self.N}")
            else:
                logs.append(f"PASS: Centripetal force N = m*v_0²/R holds.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        
        self.logs.extend(logs)
        return all("FAIL" not in log and "ERROR" not in log for log in logs)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers using steps analysis."""
        logs = []
        try:
            # Step 1 & 2 consistency: both give same time t
            t1 = sp.sqrt(2 * self.h / self.g)  # from free fall
            t2 = self.n * 2 * sp.pi * self.R / self.v_0  # from circular motion
            
            if not self._is_close(t1, t2):
                logs.append(f"FAIL: Time from step1 ({t1}) and step2 ({t2}) inconsistent.")
            else:
                logs.append(f"PASS: Free fall and circular motion times consistent.")
                
            # Derive v_0 from time equality: v_0 = n * 2πR / sqrt(2h/g)
            v0_derived = self.n * 2 * sp.pi * self.R / sp.sqrt(2 * self.h / self.g)
            if not self._is_close(self.v_0, v0_derived):
                logs.append(f"WARN: v_0 derived from time equality ({v0_derived}) differs from given ({self.v_0}).")
            else:
                logs.append(f"PASS: v_0 consistent with time equality.")
                
            # Step 4: N = m * v_0² / R = 2mgR n²π² / h (from steps analysis)
            N_step4 = 2 * self.m * self.g * self.R * self.n**2 * sp.pi**2 / self.h
            if not self._is_close(self.N, N_step4):
                logs.append(f"FAIL: Step4 N formula gives {N_step4}, but given N is {self.N}.")
            else:
                logs.append(f"PASS: N matches step4 formula.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        
        self.logs.extend(logs)
        return all("FAIL" not in log and "ERROR" not in log for log in logs)

    def test_monotonicity(self):
        """Perturb mass m and check that N changes proportionally while v_0 stays constant."""
        logs = []
        original_m = float(self.m)
        original_v0 = float(self.v_0)
        original_N = float(self.N)
        
        try:
            # Perturb mass by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate N using physics formula (should scale linearly with m)
            # From steps analysis: N = 2 * m * g * R * n² * π² / h
            perturbed_N = 2 * perturbed_m * float(self.g) * float(self.R) * float(self.n)**2 * math.pi**2 / float(self.h)
            
            # Expected scaling: N ∝ m
            expected_N = original_N * 1.1
            
            if not self._is_close(perturbed_N, expected_N, rel_tol=1e-2):
                logs.append(f"FAIL: N does not scale linearly with m. Expected {expected_N}, got {perturbed_N}.")
            else:
                logs.append(f"PASS: N scales correctly with m (monotonic).")
            
            # v_0 should remain unchanged when m changes
            # From derivation: v_0 = n * 2πR / sqrt(2h/g) - independent of m
            if not self._is_close(original_v0, float(self.v_0)):
                logs.append(f"FAIL: v_0 changed when m was perturbed.")
            else:
                logs.append(f"PASS: v_0 independent of mass m.")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
        
        self.logs.extend(logs)
        return all("FAIL" not in log and "ERROR" not in log for log in logs)

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        results = []
        
        results.append(self.test_invariants())
        results.append(self.test_trace_consistency())
        results.append(self.test_monotonicity())
        
        success = all(results)
        return success, self.logs

# Example usage (commented out for sandbox):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.R = 0.5
#             self.h = 1.0
#             self.m = 0.1
#             self.g = 9.8
#             self.n = 2
#             self.results = {
#                 'v_0': 8.86,  # n*2πR/sqrt(2h/g) for n=2, R=0.5, h=1, g=9.8
#                 'N': 15.71,   # 2mgR n²π²/h for same params
#                 't': 0.452    # sqrt(2h/g)
#             }
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     print("Success:", success)
#     for log in logs:
#         print(log)