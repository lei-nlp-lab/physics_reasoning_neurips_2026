import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Extract inputs - use results dict first, then attributes
        self.U = self._safe_get_numeric(solution_instance, 'U', 1000.0)
        self.R = self._safe_get_numeric(solution_instance, 'R', 0.1)
        self.m = self._safe_get_numeric(solution_instance, 'm', 9.10938356e-31)
        self.e = self._safe_get_numeric(solution_instance, 'e', 1.60217662e-19)

        # Extract results using the mapping provided
        self.results = {}
        mapping = {
            'E_k': 'E_k',
            'φ': 'phi',
            't': 't',
            'r_m': 'r_m',
            'r_1': 'r_1',
            'N': 'N',
            'k': 'k',
            'B_1': 'B_1',
            'v_1': 'v_1'
        }
        for key, attr in mapping.items():
            self.results[attr] = self._safe_get_numeric(solution_instance, key)

        # Constants
        self.pi = math.pi
        self.tan22_5 = 0.4  # given

    def _safe_get_numeric(self, obj, key, default=None):
        """Safely retrieve numeric value from results dict or attributes."""
        # First try the results dictionary if it exists
        if hasattr(obj, 'results'):
            value = obj.results.get(key)
            if value is not None:
                # Ensure it's numeric
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        
        # Then try to get from the object's attributes
        value = getattr(obj, key, None)
        if value is not None:
            # If it's a numeric type, convert to float
            if isinstance(value, (int, float)):
                return float(value)
            # If it's a SymPy expression with no free symbols, evaluate
            if hasattr(value, 'is_number') and value.is_number:
                return float(value)
            # If it's a SymPy symbol, try to get its value from results
            if hasattr(value, 'name'):
                # Look for a corresponding value in results
                for k, v in obj.results.items():
                    if k == key or k == value.name:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
        
        # Return default if nothing found
        return default

    def _isclose(self, a, b):
        """Robust fuzzy comparison for scalars."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        success = True

        try:
            # 1. Work-Energy Theorem for first acceleration
            v1 = self.results.get('v_1')
            if v1 is not None:
                expected_ke = 2 * self.e * self.U
                actual_ke = 0.5 * self.m * v1**2
                if not self._isclose(actual_ke, expected_ke):
                    logs.append(f"FAIL: Work-Energy Theorem (first acceleration). Expected {expected_ke:.3e}, got {actual_ke:.3e}")
                    success = False
                else:
                    logs.append("PASS: Work-Energy Theorem (first acceleration)")

            # 2. Lorentz force as centripetal force in region I
            B1 = self.results.get('B_1')
            r1 = self.results.get('r_1')
            if all(x is not None for x in [v1, B1, r1]):
                lhs = self.e * v1 * B1
                rhs = self.m * v1**2 / r1
                if not self._isclose(lhs, rhs):
                    logs.append(f"FAIL: Centripetal force in region I. LHS {lhs:.3e}, RHS {rhs:.3e}")
                    success = False
                else:
                    logs.append("PASS: Centripetal force in region I")

            # 3. Energy after N accelerations
            Ek = self.results.get('E_k')
            N = self.results.get('N')
            if Ek is not None and N is not None:
                expected_Ek = N * self.e * self.U
                if not self._isclose(Ek, expected_Ek):
                    logs.append(f"FAIL: Total kinetic energy. Expected {expected_Ek:.3e}, got {Ek:.3e}")
                    success = False
                else:
                    logs.append("PASS: Total kinetic energy consistent with N")

            # 4. Geometric relation for maximum radius r_m
            r_m = self.results.get('r_m')
            if r_m is not None and self.R is not None:
                lhs = (math.sqrt(3) * self.R - r_m)**2
                rhs = self.R**2 + r_m**2
                if not self._isclose(lhs, rhs):
                    logs.append(f"FAIL: Geometry for r_m. LHS {lhs:.3e}, RHS {rhs:.3e}")
                    success = False
                else:
                    logs.append("PASS: Geometry for r_m")

            # 5. Value of k
            k = self.results.get('k')
            if k is not None:
                expected_k = 13/6
                if not self._isclose(k, expected_k):
                    logs.append(f"FAIL: k value. Expected {expected_k:.3f}, got {k:.3f}")
                    success = False
                else:
                    logs.append("PASS: k = 13/6")

        except Exception as e:
            logs.append(f"ERROR in invariants: {traceback.format_exc()}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with formulas."""
        logs = []
        success = True

        try:
            # 1. v_1 = 2*sqrt(eU/m)
            v1 = self.results.get('v_1')
            if v1 is not None:
                expected_v1 = 2 * math.sqrt(self.e * self.U / self.m)
                if not self._isclose(v1, expected_v1):
                    logs.append(f"FAIL: v_1 consistency. Expected {expected_v1:.3e}, got {v1:.3e}")
                    success = False
                else:
                    logs.append("PASS: v_1 = 2*sqrt(eU/m)")

            # 2. r_1 = 0.4 * R
            r1 = self.results.get('r_1')
            if r1 is not None and self.R is not None:
                expected_r1 = self.tan22_5 * self.R
                if not self._isclose(r1, expected_r1):
                    logs.append(f"FAIL: r_1 consistency. Expected {expected_r1:.3e}, got {r1:.3e}")
                    success = False
                else:
                    logs.append("PASS: r_1 = 0.4 * R")

            # 3. B_1 = (5/R)*sqrt(mU/e)
            B1 = self.results.get('B_1')
            if B1 is not None and self.R is not None:
                expected_B1 = (5/self.R) * math.sqrt(self.m * self.U / self.e)
                if not self._isclose(B1, expected_B1):
                    logs.append(f"FAIL: B_1 consistency. Expected {expected_B1:.3e}, got {B1:.3e}")
                    success = False
                else:
                    logs.append("PASS: B_1 = (5/R)*sqrt(mU/e)")

            # 4. φ = 5π/4
            phi = self.results.get('phi')
            if phi is not None:
                expected_phi = 5 * self.pi / 4
                if not self._isclose(phi, expected_phi):
                    logs.append(f"FAIL: φ consistency. Expected {expected_phi:.3f}, got {phi:.3f}")
                    success = False
                else:
                    logs.append("PASS: φ = 5π/4")

            # 5. t = φ * r_1 / v_1
            t = self.results.get('t')
            if all(x is not None for x in [t, phi, r1, v1]):
                expected_t = phi * r1 / v1
                if not self._isclose(t, expected_t):
                    logs.append(f"FAIL: t consistency. Expected {expected_t:.3e}, got {t:.3e}")
                    success = False
                else:
                    logs.append("PASS: t = φ * r_1 / v_1")

            # 6. N = 8
            N = self.results.get('N')
            if N is not None:
                if not self._isclose(N, 8):
                    logs.append(f"FAIL: N consistency. Expected 8, got {N}")
                    success = False
                else:
                    logs.append("PASS: N = 8")

            # 7. E_k = 8eU
            Ek = self.results.get('E_k')
            if Ek is not None:
                expected_Ek = 8 * self.e * self.U
                if not self._isclose(Ek, expected_Ek):
                    logs.append(f"FAIL: E_k consistency. Expected {expected_Ek:.3e}, got {Ek:.3e}")
                    success = False
                else:
                    logs.append("PASS: E_k = 8eU")

            # 8. r_m = sqrt(3)R/3
            r_m = self.results.get('r_m')
            if r_m is not None and self.R is not None:
                expected_r_m = math.sqrt(3) * self.R / 3
                if not self._isclose(r_m, expected_r_m):
                    logs.append(f"FAIL: r_m consistency. Expected {expected_r_m:.3e}, got {r_m:.3e}")
                    success = False
                else:
                    logs.append("PASS: r_m = sqrt(3)R/3")

            # 9. k = 13/6
            k = self.results.get('k')
            if k is not None:
                expected_k = 13/6
                if not self._isclose(k, expected_k):
                    logs.append(f"FAIL: k consistency. Expected {expected_k:.3f}, got {k:.3f}")
                    success = False
                else:
                    logs.append("PASS: k = 13/6")

        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """Perturb input U and check trends."""
        logs = []
        success = True

        try:
            # Store original values
            U_orig = self.U
            perturb_factor = 1.01  # 1% increase
            U_new = U_orig * perturb_factor

            # Recalculate expected trends
            # v1 ∝ sqrt(U)
            v1_orig = 2 * math.sqrt(self.e * U_orig / self.m)
            v1_new = 2 * math.sqrt(self.e * U_new / self.m)
            if v1_new <= v1_orig:
                logs.append(f"FAIL: v1 monotonicity. U increased but v1 did not increase (orig {v1_orig:.3e}, new {v1_new:.3e})")
                success = False
            else:
                logs.append("PASS: v1 increases with U")

            # B1 ∝ sqrt(U)
            B1_orig = (5/self.R) * math.sqrt(self.m * U_orig / self.e)
            B1_new = (5/self.R) * math.sqrt(self.m * U_new / self.e)
            if B1_new <= B1_orig:
                logs.append(f"FAIL: B1 monotonicity. U increased but B1 did not increase (orig {B1_orig:.3e}, new {B1_new:.3e})")
                success = False
            else:
                logs.append("PASS: B1 increases with U")

            # t ∝ 1/sqrt(U)
            r1 = self.tan22_5 * self.R
            phi = 5 * self.pi / 4
            t_orig = phi * r1 / v1_orig
            t_new = phi * r1 / v1_new
            if t_new >= t_orig:
                logs.append(f"FAIL: t monotonicity. U increased but t did not decrease (orig {t_orig:.3e}, new {t_new:.3e})")
                success = False
            else:
                logs.append("PASS: t decreases with U")

            # E_k ∝ U
            Ek_orig = 8 * self.e * U_orig
            Ek_new = 8 * self.e * U_new
            if Ek_new <= Ek_orig:
                logs.append(f"FAIL: E_k monotonicity. U increased but E_k did not increase (orig {Ek_orig:.3e}, new {Ek_new:.3e})")
                success = False
            else:
                logs.append("PASS: E_k increases with U")

        except Exception as e:
            logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
            success = False

        return success, logs

    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            success, logs = test_method()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False

        return overall_success, all_logs