import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all input parameters and computed results from the solution instance.
        """
        self.solution = solution_instance
        self.logs = []

        # Extract input parameters with safe fallbacks (use defaults if not provided)
        self.m = self._safe_get('m', 1000.0)
        self.M = self._safe_get('M', 5.972e24)
        self.r = self._safe_get('r', 6.371e6)
        self.G = self._safe_get('G', 6.67430e-11)

        # Extract computed results with safe fallbacks
        self.T = self._safe_get('T', None)
        self.T_Q = self._safe_get('T_Q', None)
        self.r_Q = self._safe_get('r_Q', None)
        self.d = self._safe_get('d', None)
        self.N_A = self._safe_get('N/A', 0.0)

        # Store numerical values for testing (all floats)
        self.num_vals = {
            'm': float(self.m),
            'M': float(self.M),
            'r': float(self.r),
            'G': float(self.G),
            'T': float(self.T) if self.T is not None else None,
            'T_Q': float(self.T_Q) if self.T_Q is not None else None,
            'r_Q': float(self.r_Q) if self.r_Q is not None else None,
            'd': float(self.d) if self.d is not None else None
        }

    def _safe_get(self, key, default):
        """
        Safely retrieve an attribute from the solution instance or its results dictionary.
        """
        try:
            # First try results dictionary
            if hasattr(self.solution, 'results'):
                val = self.solution.results.get(key, default)
                if val is not None:
                    return val
            # Then try direct attribute
            val = getattr(self.solution, key, default)
            return val
        except (AttributeError, KeyError):
            return default

    def test_invariants(self):
        """
        Verify fundamental physics invariants: Kepler's Third Law and gravitational force.
        """
        logs = []
        try:
            # 1. Check Kepler's Third Law for satellite P: T^2 ∝ r^3
            T = self.num_vals['T']
            r = self.num_vals['r']
            M = self.num_vals['M']
            G = self.num_vals['G']
            
            if T is not None:
                T_sq = T**2
                expected_T_sq = (4 * math.pi**2 / (G * M)) * r**3
                if not math.isclose(T_sq, expected_T_sq, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Kepler's Third Law violated for P. T^2={T_sq:.3e}, expected {expected_T_sq:.3e}")
                else:
                    logs.append("PASS: Kepler's Third Law holds for satellite P.")

            # 2. Check Kepler's Third Law between P and Q: T^2 / r^3 = T_Q^2 / r_Q^3
            T_Q = self.num_vals['T_Q']
            r_Q = self.num_vals['r_Q']
            
            if all(v is not None for v in [T, r, T_Q, r_Q]):
                ratio_P = T**2 / r**3
                ratio_Q = T_Q**2 / r_Q**3
                if not math.isclose(ratio_P, ratio_Q, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Kepler's Third Law not consistent between P and Q. Ratios: {ratio_P:.3e} vs {ratio_Q:.3e}")
                else:
                    logs.append("PASS: Kepler's Third Law consistent between P and Q.")

            # 3. Check gravitational force law (optional: force should be centripetal)
            if T is not None:
                v = 2 * math.pi * r / T
                F_grav = G * self.num_vals['m'] * M / r**2
                F_cent = self.num_vals['m'] * v**2 / r
                if not math.isclose(F_grav, F_cent, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Centripetal force mismatch. F_grav={F_grav:.3e}, F_cent={F_cent:.3e}")
                else:
                    logs.append("PASS: Gravitational force equals centripetal force for P.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        return logs

    def test_trace_consistency(self):
        """
        Check consistency between intermediate results and final answers.
        """
        logs = []
        try:
            T = self.num_vals['T']
            T_Q = self.num_vals['T_Q']
            r = self.num_vals['r']
            r_Q = self.num_vals['r_Q']
            d = self.num_vals['d']

            # 1. Check that T matches the expected formula numerically
            if T is not None:
                expected_T = 2 * math.pi * math.sqrt(r**3 / (self.num_vals['G'] * self.num_vals['M']))
                if not math.isclose(T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T value {T:.3e} differs from expected {expected_T:.3e}")
                else:
                    logs.append("PASS: T value matches expected formula numerically.")

            # 2. Check that T_Q = 8 * T
            if T is not None and T_Q is not None:
                if not math.isclose(T_Q, 8 * T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T_Q {T_Q:.3e} ≠ 8*T {8*T:.3e}")
                else:
                    logs.append("PASS: T_Q = 8*T numerically.")

            # 3. Check that r_Q = 4 * r
            if r_Q is not None:
                if not math.isclose(r_Q, 4 * r, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: r_Q {r_Q:.3e} ≠ 4*r {4*r:.3e}")
                else:
                    logs.append("PASS: r_Q = 4*r numerically.")

            # 4. Check that d = r + r_Q
            if d is not None and r_Q is not None:
                if not math.isclose(d, r + r_Q, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: d {d:.3e} ≠ r + r_Q {r + r_Q:.3e}")
                else:
                    logs.append("PASS: d = r + r_Q numerically.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        return logs

    def test_monotonicity(self):
        """
        Perturb input parameters and verify expected trends.
        """
        logs = []
        try:
            r = self.num_vals['r']
            M = self.num_vals['M']
            G = self.num_vals['G']

            # Test 1: Perturb r and check that d changes proportionally (d = 5r)
            original_r = r
            perturbed_r = original_r * 1.1
            original_d = 5 * original_r
            perturbed_d = 5 * perturbed_r
            if not math.isclose(perturbed_d / original_d, perturbed_r / original_r, rel_tol=1e-2):
                logs.append("FAIL: Monotonicity w.r.t. r violated. d should scale linearly with r.")
            else:
                logs.append("PASS: d scales linearly with r (d = 5r).")

            # Test 2: Perturb M and check that T changes as T ∝ M^{-1/2}
            original_M = M
            perturbed_M = original_M * 1.1
            original_T = 2 * math.pi * math.sqrt(original_r**3 / (G * original_M))
            perturbed_T = 2 * math.pi * math.sqrt(original_r**3 / (G * perturbed_M))
            expected_ratio = math.sqrt(original_M / perturbed_M)
            actual_ratio = perturbed_T / original_T
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
                logs.append(f"FAIL: T does not scale as M^(-1/2). Ratio: {actual_ratio:.4f}, expected {expected_ratio:.4f}")
            else:
                logs.append("PASS: T scales as M^{-1/2}.")

            # Test 3: Perturb r and check T ∝ r^{3/2}
            original_T = 2 * math.pi * math.sqrt(original_r**3 / (G * original_M))
            perturbed_T = 2 * math.pi * math.sqrt(perturbed_r**3 / (G * original_M))
            expected_ratio_T = (perturbed_r / original_r) ** 1.5
            actual_ratio_T = perturbed_T / original_T
            if not math.isclose(actual_ratio_T, expected_ratio_T, rel_tol=1e-2):
                logs.append(f"FAIL: T does not scale as r^(3/2). Ratio: {actual_ratio_T:.4f}, expected {expected_ratio_T:.4f}")
            else:
                logs.append("PASS: T scales as r^{3/2}.")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
        return logs

    def run_all_tests(self):
        """
        Execute all checks and return a tuple (success: bool, logs: list).
        """
        all_logs = []
        all_logs.append("=== PhysicsChecker started ===")

        # Run each test suite
        all_logs.extend(self.test_invariants())
        all_logs.extend(self.test_trace_consistency())
        all_logs.extend(self.test_monotonicity())

        # Determine overall success: no lines starting with "FAIL" or "ERROR"
        success = True
        for log in all_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break

        all_logs.append(f"=== PhysicsChecker finished: {'SUCCESS' if success else 'FAILED'} ===")
        return success, all_logs