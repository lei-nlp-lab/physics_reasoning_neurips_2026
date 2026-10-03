import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Inputs (with defaults from problem statement)
        self.v = self._safe_get(solution_instance, 'v', 6.0)
        self.L1 = self._safe_get(solution_instance, 'L1', 10.0)
        self.m = self._safe_get(solution_instance, 'm', 3.0)
        self.g = self._safe_get(solution_instance, 'g', 10.0)

        # Results (use None if not present)
        self.W = self._safe_get(solution_instance, 'W', None)
        self.W_prime = self._safe_get(solution_instance, "W'", None)   # W'
        self.W_dprime = self._safe_get(solution_instance, "W''", None) # W''
        self.a = self._safe_get(solution_instance, 'a', None)
        self.t = self._safe_get(solution_instance, 't', None)
        self.L = self._safe_get(solution_instance, 'L', None)
        self.a_prime = self._safe_get(solution_instance, "a'", None)   # a'
        self.t1 = self._safe_get(solution_instance, 't_1', None)
        self.L_prime = self._safe_get(solution_instance, "L'", None)   # L'
        self.L_dprime = self._safe_get(solution_instance, "L''", None) # L''

        # Derived coefficients (inferred from accelerations)
        self.mu = self.a / self.g if self.a is not None and self.g != 0 else None
        self.mu_prime = self.a_prime / self.g if self.a_prime is not None and self.g != 0 else None

        # Storage for logs
        self.logs = []

    def _safe_get(self, obj, key, default):
        """
        Safely retrieve an attribute from the solution instance.
        Tries obj.key, then obj.results.get(key).
        """
        try:
            val = getattr(obj, key, None)
            if val is None:
                val = obj.results.get(key, default)
            return val
        except AttributeError:
            return default

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Fuzzy comparison for floats or SymPy numbers.
        """
        try:
            a_float = float(a)
            b_float = float(b)
            return math.isclose(a_float, b_float, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """
        Verify physical invariants: Newton's laws and work-energy theorem.
        """
        logs = []
        try:
            # --- First part (initial sliding until speed matches belt) ---
            if self.a is not None and self.L is not None:
                # Kinematics: v^2 = 2*a*L  =>  a = v^2/(2*L)
                expected_a = self.v**2 / (2 * self.L)
                if not self._is_close(self.a, expected_a):
                    logs.append(f"FAIL: a = {self.a} but v^2/(2*L) = {expected_a}")
                else:
                    logs.append(f"PASS: kinematics a = v^2/(2*L)")

            if self.a is not None and self.t is not None:
                # Kinematics: t = v/a
                expected_t = self.v / self.a
                if not self._is_close(self.t, expected_t):
                    logs.append(f"FAIL: t = {self.t} but v/a = {expected_t}")
                else:
                    logs.append(f"PASS: t = v/a")

            if self.W is not None:
                # Work-energy: W = ΔK = 0.5*m*v^2 (starts from rest)
                expected_W = 0.5 * self.m * self.v**2
                if not self._is_close(self.W, expected_W):
                    logs.append(f"FAIL: W = {self.W} but 0.5*m*v^2 = {expected_W}")
                else:
                    logs.append(f"PASS: work-energy for W")

            # --- Second part (full slide over L1 with same μ) ---
            if self.W_prime is not None and self.a is not None:
                # W' = μ*m*g*L1 = a * m * L1
                expected_W_prime = self.a * self.m * self.L1
                if not self._is_close(self.W_prime, expected_W_prime):
                    logs.append(f"FAIL: W' = {self.W_prime} but a*m*L1 = {expected_W_prime}")
                else:
                    logs.append(f"PASS: work for W'")

            if self.L_prime is not None and self.a is not None and self.t1 is not None:
                # L' = 0.5 * a * t1^2
                expected_L_prime = 0.5 * self.a * self.t1**2
                if not self._is_close(self.L_prime, expected_L_prime):
                    logs.append(f"FAIL: L' = {self.L_prime} but 0.5*a*t1^2 = {expected_L_prime}")
                else:
                    logs.append(f"PASS: kinematics for L'")

            # --- Third part (different μ') ---
            if self.a_prime is not None and self.L_dprime is not None:
                # L'' = v^2/(2*a')
                expected_L_dprime = self.v**2 / (2 * self.a_prime)
                if not self._is_close(self.L_dprime, expected_L_dprime):
                    logs.append(f"FAIL: L'' = {self.L_dprime} but v^2/(2*a') = {expected_L_dprime}")
                else:
                    logs.append(f"PASS: kinematics for L''")

            if self.W_dprime is not None and self.a_prime is not None:
                # W'' = μ'*m*g*L1 = a' * m * L1
                expected_W_dprime = self.a_prime * self.m * self.L1
                if not self._is_close(self.W_dprime, expected_W_dprime):
                    logs.append(f"FAIL: W'' = {self.W_dprime} but a'*m*L1 = {expected_W_dprime}")
                else:
                    logs.append(f"PASS: work for W''")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")

        self.logs.extend(logs)
        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        try:
            # Check that the same a is used in first and second parts? Actually they are separate.
            # We'll check internal consistency of each part.

            # First part: if we have a, t, L, W, they should be consistent.
            if all(x is not None for x in [self.a, self.t, self.L, self.W]):
                # From a and t: L = 0.5*a*t^2
                L_from_a_t = 0.5 * self.a * self.t**2
                if not self._is_close(self.L, L_from_a_t):
                    logs.append(f"TRACE FAIL (first part): L={self.L} but 0.5*a*t^2={L_from_a_t}")
                else:
                    logs.append(f"TRACE PASS: L consistent with a and t")

                # From a and L: W = μ*m*g*L = a * m * L
                W_from_a_L = self.a * self.m * self.L
                if not self._is_close(self.W, W_from_a_L):
                    logs.append(f"TRACE FAIL (first part): W={self.W} but a*m*L={W_from_a_L}")
                else:
                    logs.append(f"TRACE PASS: W consistent with a and L")

            # Second part: if we have a, t1, L_prime, W_prime
            if all(x is not None for x in [self.a, self.t1, self.L_prime, self.W_prime]):
                # L' from a and t1
                L_prime_from_a_t1 = 0.5 * self.a * self.t1**2
                if not self._is_close(self.L_prime, L_prime_from_a_t1):
                    logs.append(f"TRACE FAIL (second part): L'={self.L_prime} but 0.5*a*t1^2={L_prime_from_a_t1}")
                else:
                    logs.append(f"TRACE PASS: L' consistent with a and t1")

                # W' from a and L1
                W_prime_from_a_L1 = self.a * self.m * self.L1
                if not self._is_close(self.W_prime, W_prime_from_a_L1):
                    logs.append(f"TRACE FAIL (second part): W'={self.W_prime} but a*m*L1={W_prime_from_a_L1}")
                else:
                    logs.append(f"TRACE PASS: W' consistent with a and L1")

            # Third part: if we have a_prime, L_dprime, W_dprime
            if all(x is not None for x in [self.a_prime, self.L_dprime, self.W_dprime]):
                # L'' from a' and v
                L_dprime_from_a_prime = self.v**2 / (2 * self.a_prime)
                if not self._is_close(self.L_dprime, L_dprime_from_a_prime):
                    logs.append(f"TRACE FAIL (third part): L''={self.L_dprime} but v^2/(2*a')={L_dprime_from_a_prime}")
                else:
                    logs.append(f"TRACE PASS: L'' consistent with a' and v")

                # W'' from a' and L1
                W_dprime_from_a_prime_L1 = self.a_prime * self.m * self.L1
                if not self._is_close(self.W_dprime, W_dprime_from_a_prime_L1):
                    logs.append(f"TRACE FAIL (third part): W''={self.W_dprime} but a'*m*L1={W_dprime_from_a_prime_L1}")
                else:
                    logs.append(f"TRACE PASS: W'' consistent with a' and L1")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")

        self.logs.extend(logs)
        return logs

    def test_monotonicity(self):
        """
        Perturb mass m and check that outputs scale proportionally (since a and a' are independent of m).
        """
        logs = []
        try:
            original_m = self.m
            if original_m is None:
                logs.append("SKIP monotonicity: mass m not available.")
                return logs

            # Perturbation factor
            factor = 1.1
            perturbed_m = original_m * factor

            # Recompute expected outputs using formulas (assuming same a, a', etc.)
            # First part: W = 0.5 * m * v^2
            if self.W is not None:
                expected_W_new = 0.5 * perturbed_m * self.v**2
                expected_ratio = expected_W_new / self.W
                if not self._is_close(expected_ratio, factor):
                    logs.append(f"MONOTONICITY FAIL (W): ratio={expected_ratio}, expected {factor}")
                else:
                    logs.append(f"MONOTONICITY PASS (W): scales linearly with m")

            # Second part: W' = a * m * L1
            if self.W_prime is not None and self.a is not None:
                expected_W_prime_new = self.a * perturbed_m * self.L1
                expected_ratio_prime = expected_W_prime_new / self.W_prime
                if not self._is_close(expected_ratio_prime, factor):
                    logs.append(f"MONOTONICITY FAIL (W'): ratio={expected_ratio_prime}, expected {factor}")
                else:
                    logs.append(f"MONOTONICITY PASS (W'): scales linearly with m")

            # Third part: W'' = a' * m * L1
            if self.W_dprime is not None and self.a_prime is not None:
                expected_W_dprime_new = self.a_prime * perturbed_m * self.L1
                expected_ratio_dprime = expected_W_dprime_new / self.W_dprime
                if not self._is_close(expected_ratio_dprime, factor):
                    logs.append(f"MONOTONICITY FAIL (W''): ratio={expected_ratio_dprime}, expected {factor}")
                else:
                    logs.append(f"MONOTONICITY PASS (W''): scales linearly with m")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")

        self.logs.extend(logs)
        return logs

    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs).
        """
        self.logs = []
        self.logs.append("=== PhysicsChecker started ===")

        # Run tests
        inv_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        mono_logs = self.test_monotonicity()

        # Determine overall success: no "FAIL" or "ERROR" in logs
        all_logs = self.logs
        success = not any("FAIL" in log or "ERROR" in log for log in all_logs)

        self.logs.append(f"=== PhysicsChecker finished: {'SUCCESS' if success else 'FAILED'} ===")
        return success, self.logs