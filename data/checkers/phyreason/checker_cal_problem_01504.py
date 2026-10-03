import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary inputs and results from the solution instance.
        """
        # Helper to safely get attributes, falling back to results dictionary
        def safe_get(attr, default=None):
            # First try direct attribute
            val = getattr(solution_instance, attr, None)
            if val is not None:
                return val
            # Then try results dictionary (if it exists)
            if hasattr(solution_instance, 'results'):
                val = solution_instance.results.get(attr, None)
                if val is not None:
                    return val
                # Special handling for F_N' (which might be stored as "F_N'")
                if attr == 'F_N_prime':
                    val = solution_instance.results.get("F_N'", None)
                    if val is not None:
                        return val
            return default

        # Extract given constants (with defaults as per problem)
        self.m = safe_get('m', 2.0)
        self.R = safe_get('R', 0.6)
        self.g = safe_get('g', 10.0)

        # Extract results (some may be computed later)
        self.F_N = safe_get('F_N')
        self.E_p = safe_get('E_p')
        self.F_N_prime = safe_get('F_N_prime')
        self.v_C = safe_get('v_C')
        self.W = safe_get('W')

        # Intermediate computed values (will be set in tests)
        self.v_B = None
        self.v_C_expected = None
        self.v_B_expected = None
        self.E_p_expected = None
        self.W_expected = None

        # Logs for reporting
        self.logs = []

    def _log(self, message, success=None):
        """Add a message to the logs, optionally with success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physics laws and given conditions."""
        try:
            # 1. Condition at point C: mg = m * v_C^2 / R  => v_C = sqrt(g*R)
            self.v_C_expected = math.sqrt(self.g * self.R)
            if self.v_C is not None:
                c1 = self._is_close(self.v_C, self.v_C_expected)
                self._log(f"v_C: expected ≈{self.v_C_expected:.3f}, got {self.v_C}", c1)
            else:
                c1 = False
                self._log("v_C not provided", False)

            # 2. Condition at point B: F_N - m*g = m * v_B^2 / R, with F_N = 7*m*g
            #    => v_B^2 = (7mg - mg) * R / m = 6gR
            self.v_B_expected = math.sqrt(6 * self.g * self.R)
            # Compute v_B from provided F_N if available, otherwise use expected
            if self.F_N is not None:
                v_B_from_FN = math.sqrt((self.F_N - self.m * self.g) * self.R / self.m)
                c2 = self._is_close(v_B_from_FN, self.v_B_expected)
                self._log(f"v_B from F_N: expected ≈{self.v_B_expected:.3f}, got ≈{v_B_from_FN:.3f}", c2)
                self.v_B = v_B_from_FN
            else:
                c2 = False
                self._log("F_N not provided, cannot check v_B from normal force", False)
                self.v_B = self.v_B_expected  # fallback for later tests

            # 3. Newton's third law: F_N (on object) and F_N' (on track) equal in magnitude
            if self.F_N is not None and self.F_N_prime is not None:
                c3 = self._is_close(self.F_N, self.F_N_prime)
                self._log(f"Newton's third law: F_N={self.F_N}, F_N'={self.F_N_prime}", c3)
            else:
                c3 = False
                if self.F_N is None:
                    self._log("F_N missing", False)
                if self.F_N_prime is None:
                    self._log("F_N' missing", False)

            # 4. Conservation of mechanical energy from A to B (smooth): E_p = 0.5 * m * v_B^2
            self.E_p_expected = 0.5 * self.m * self.v_B_expected**2
            if self.E_p is not None:
                c4 = self._is_close(self.E_p, self.E_p_expected)
                self._log(f"Elastic potential energy: expected ≈{self.E_p_expected:.3f}, got {self.E_p}", c4)
            else:
                c4 = False
                self._log("E_p not provided", False)

            # 5. Work-energy theorem from B to C: -2mgR - W = 0.5*m*v_C^2 - 0.5*m*v_B^2
            #    Rearranged: W = 0.5*m*(v_B^2 - v_C^2) - 2*m*g*R
            self.W_expected = 0.5 * self.m * (self.v_B_expected**2 - self.v_C_expected**2) - 2 * self.m * self.g * self.R
            if self.W is not None:
                c5 = self._is_close(self.W, self.W_expected)
                self._log(f"Work against resistance: expected ≈{self.W_expected:.3f}, got {self.W}", c5)
            else:
                c5 = False
                self._log("W not provided", False)

            success = all([c1, c2, c3, c4, c5])
            self._log(f"Invariants test overall: {'PASS' if success else 'FAIL'}")
            return success

        except Exception as e:
            self._log(f"Error in test_invariants: {e}\n{traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results for consistency."""
        try:
            # Compute v_B from both normal force and energy, if possible
            v_B_from_FN = None
            if self.F_N is not None:
                v_B_from_FN = math.sqrt((self.F_N - self.m * self.g) * self.R / self.m)

            v_B_from_Ep = None
            if self.E_p is not None:
                v_B_from_Ep = math.sqrt(2 * self.E_p / self.m)

            checks = []
            logs = []

            # Compare v_B from different sources
            if v_B_from_FN is not None and v_B_from_Ep is not None:
                c = self._is_close(v_B_from_FN, v_B_from_Ep)
                checks.append(c)
                logs.append(f"v_B from F_N ≈{v_B_from_FN:.3f} vs from E_p ≈{v_B_from_Ep:.3f}: {'OK' if c else 'MISMATCH'}")
            else:
                logs.append("Cannot compare v_B from F_N and E_p (missing data)")

            # Check that v_C matches the condition at the top
            if self.v_C is not None:
                v_C_condition = math.sqrt(self.g * self.R)
                c = self._is_close(self.v_C, v_C_condition)
                checks.append(c)
                logs.append(f"v_C ≈{self.v_C:.3f} vs condition sqrt(gR)≈{v_C_condition:.3f}: {'OK' if c else 'MISMATCH'}")
            else:
                logs.append("v_C not provided")

            # Check that W matches energy difference
            if self.W is not None and self.v_B is not None and self.v_C is not None:
                W_from_energy = 0.5 * self.m * (self.v_B**2 - self.v_C**2) - 2 * self.m * self.g * self.R
                c = self._is_close(self.W, W_from_energy)
                checks.append(c)
                logs.append(f"W from energy ≈{W_from_energy:.3f} vs provided {self.W}: {'OK' if c else 'MISMATCH'}")
            else:
                logs.append("Cannot compute W from energy (missing v_B or v_C)")

            for log in logs:
                self._log(log, None)

            success = all(checks) if checks else False
            self._log(f"Trace consistency overall: {'PASS' if success else 'INCOMPLETE'}")
            return success

        except Exception as e:
            self._log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """
        Perturb inputs (mass m and radius R) and verify that results change
        in the expected direction using physical formulas.
        """
        try:
            # Base values computed from formulas
            v_C_base = math.sqrt(self.g * self.R)
            v_B_base = math.sqrt(6 * self.g * self.R)
            E_p_base = 0.5 * self.m * v_B_base**2
            W_base = 0.5 * self.m * (v_B_base**2 - v_C_base**2) - 2 * self.m * self.g * self.R

            checks = []
            logs = []

            # Perturb mass m by +1%
            delta = 0.01
            m_new = self.m * (1 + delta)
            v_C_new_m = math.sqrt(self.g * self.R)          # independent of m
            v_B_new_m = math.sqrt(6 * self.g * self.R)      # independent of m
            E_p_new_m = 0.5 * m_new * v_B_new_m**2         # proportional to m
            W_new_m = 0.5 * m_new * (v_B_new_m**2 - v_C_new_m**2) - 2 * m_new * self.g * self.R  # proportional to m

            c1 = self._is_close(v_C_new_m, v_C_base)  # should be equal
            c2 = self._is_close(v_B_new_m, v_B_base)  # should be equal
            c3 = (E_p_new_m > E_p_base)  # should increase
            c4 = (W_new_m > W_base)      # should increase

            if c1 and c2 and c3 and c4:
                logs.append(f"Perturb m +1%: v_C unchanged, v_B unchanged, E_p increased, W increased (OK)")
                checks.append(True)
            else:
                logs.append(f"Perturb m +1%: unexpected changes: v_C_eq={c1}, v_B_eq={c2}, E_p_inc={c3}, W_inc={c4}")
                checks.append(False)

            # Perturb radius R by +1%
            R_new = self.R * (1 + delta)
            v_C_new_R = math.sqrt(self.g * R_new)
            v_B_new_R = math.sqrt(6 * self.g * R_new)
            E_p_new_R = 0.5 * self.m * v_B_new_R**2
            W_new_R = 0.5 * self.m * (v_B_new_R**2 - v_C_new_R**2) - 2 * self.m * self.g * R_new

            c5 = (v_C_new_R > v_C_base)  # sqrt increases
            c6 = (v_B_new_R > v_B_base)  # sqrt increases
            c7 = (E_p_new_R > E_p_base)  # increases with R
            c8 = (W_new_R > W_base)      # increases with R (since W_base = 0.5*g*R*m)

            if c5 and c6 and c7 and c8:
                logs.append(f"Perturb R +1%: v_C increased, v_B increased, E_p increased, W increased (OK)")
                checks.append(True)
            else:
                logs.append(f"Perturb R +1%: unexpected changes: v_C_inc={c5}, v_B_inc={c6}, E_p_inc={c7}, W_inc={c8}")
                checks.append(False)

            for log in logs:
                self._log(log, None)

            success = all(checks)
            self._log(f"Monotonicity test overall: {'PASS' if success else 'FAIL'}")
            return success

        except Exception as e:
            self._log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # reset logs
        overall = True

        try:
            inv = self.test_invariants()
            overall &= inv
            # Wait a moment to allow logs to be written
        except Exception as e:
            self._log(f"Invariants test crashed: {e}", False)
            overall = False

        try:
            trace = self.test_trace_consistency()
            overall &= trace
        except Exception as e:
            self._log(f"Trace consistency test crashed: {e}", False)
            overall = False

        try:
            mono = self.test_monotonicity()
            overall &= mono
        except Exception as e:
            self._log(f"Monotonicity test crashed: {e}", False)
            overall = False

        self._log(f"All tests completed. Overall: {'PASS' if overall else 'FAIL'}")
        return overall, self.logs