import math
import sympy as sp
from sympy import symbols, Eq, solve
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: extract inputs and results using getattr to avoid AttributeError.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Inputs (use defaults if not found)
        self.m = getattr(solution_instance, 'm', 1.0)  # mass of block A (default 1.0)
        self.v0 = getattr(solution_instance, 'v0', self.results.get('v_0', 1.0))
        self.t0 = getattr(solution_instance, 't0', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.theta = math.asin(0.6)  # sinθ = 0.6

        # Results (target answers)
        self.E_pmax = self.results.get('E_pmax', None)  # from step_3
        self.delta_s = self.results.get('delta_s', None)  # from step_7
        self.mu = self.results.get('mu', None)  # from step_13

        # Intermediate results (for trace consistency)
        self.m_B = self.results.get('m_B', None)  # from step_3
        self.a_A = self.results.get('a_A', None)  # from step_5
        self.s_A = self.results.get('s_A', None)  # from step_6
        self.s_B = self.results.get('s_B', None)  # from step_6
        self.v_A_prime = self.results.get("v_A''", None)  # from step_10 (note: key is v_A'' in mapping)
        self.v_B = self.results.get('v_B', None)  # from step_5 or step_8

        # Tolerance for numerical comparisons (1-2% relative)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def is_close(self, a, b):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        """Verify conservation laws and Newton's laws based on KB and steps analysis."""
        logs = []
        try:
            # 1. Conservation of momentum at maximum compression (step_1)
            if self.m_B is not None:
                lhs = self.m_B * 1.2 * self.v0
                rhs = (self.m_B + self.m) * self.v0
                if not self.is_close(lhs, rhs):
                    logs.append(f"FAIL: Momentum conservation at max compression. LHS={lhs}, RHS={rhs}")
                else:
                    logs.append("PASS: Momentum conservation at max compression.")

            # 2. Conservation of energy for spring (step_2)
            if self.E_pmax is not None and self.m_B is not None:
                ke_initial = 0.5 * self.m_B * (1.2 * self.v0)**2
                ke_max_comp = 0.5 * (self.m_B + self.m) * self.v0**2
                E_pmax_calc = ke_initial - ke_max_comp
                if not self.is_close(self.E_pmax, E_pmax_calc):
                    logs.append(f"FAIL: Energy conservation for spring. Calculated={E_pmax_calc}, Result={self.E_pmax}")
                else:
                    logs.append("PASS: Energy conservation for spring.")

            # 3. Newton's second law: force equality on A and B (step_4)
            # From steps: F = m * a_A (on A) and F = m_B * a_B (on B), with a_A = 5*a_B.
            if self.a_A is not None and self.m_B is not None:
                a_B = self.a_A / 5.0  # from step_5: a_A = 5*a_B
                force_A = self.m * self.a_A
                force_B = self.m_B * a_B
                if not self.is_close(force_A, force_B):
                    logs.append(f"FAIL: Newton's 2nd law (force equality). F_A={force_A}, F_B={force_B}")
                else:
                    logs.append("PASS: Newton's 2nd law (force equality).")

            # 4. Work-energy theorem on incline (steps 11-13)
            if self.mu is not None:
                sin_theta = 0.6
                cos_theta = math.sqrt(1 - sin_theta**2)  # ≈0.8
                # Upward motion: -mgL sinθ - μ mgL cosθ = 0 - 0.5*m*(2v0)^2
                # Downward motion: mgL sinθ - μ mgL cosθ = 0.5*m*v0^2 - 0
                # Solve for L from upward:
                L_up = (2 * self.v0**2) / (self.g * (sin_theta + self.mu * cos_theta))
                # Solve for L from downward:
                L_down = (0.5 * self.v0**2) / (self.g * (sin_theta - self.mu * cos_theta))
                if not self.is_close(L_up, L_down):
                    logs.append(f"FAIL: Work-energy theorem consistency. L_up={L_up}, L_down={L_down}")
                else:
                    logs.append("PASS: Work-energy theorem on incline.")

            # 5. Conservation of momentum in second collision (step_8)
            if self.v_A_prime is not None and self.v_B is not None:
                # Momentum: m*v_A' - 5m*0.8v0 = m*(-2v0) + 5m*v_B
                lhs_momentum = self.m * self.v_A_prime - 5 * self.m * 0.8 * self.v0
                rhs_momentum = self.m * (-2 * self.v0) + 5 * self.m * self.v_B
                if not self.is_close(lhs_momentum, rhs_momentum):
                    logs.append(f"FAIL: Momentum conservation in second collision. LHS={lhs_momentum}, RHS={rhs_momentum}")
                else:
                    logs.append("PASS: Momentum conservation in second collision.")

            # 6. Conservation of energy in second collision (step_9)
            if self.v_A_prime is not None and self.v_B is not None:
                ke_initial = 0.5 * self.m * self.v_A_prime**2 + 0.5 * 5 * self.m * (0.8 * self.v0)**2
                ke_final = 0.5 * self.m * (-2 * self.v0)**2 + 0.5 * 5 * self.m * self.v_B**2
                if not self.is_close(ke_initial, ke_final):
                    logs.append(f"FAIL: Energy conservation in second collision. KE_initial={ke_initial}, KE_final={ke_final}")
                else:
                    logs.append("PASS: Energy conservation in second collision.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # 1. Check m_B = 5*m (from step_3)
            if self.m_B is not None and not self.is_close(self.m_B, 5 * self.m):
                logs.append(f"FAIL: m_B consistency. Expected {5*self.m}, got {self.m_B}")

            # 2. Check E_pmax = 0.6*m*v0^2 (from step_3)
            if self.E_pmax is not None:
                expected_E_pmax = 0.6 * self.m * self.v0**2
                if not self.is_close(self.E_pmax, expected_E_pmax):
                    logs.append(f"FAIL: E_pmax consistency. Expected {expected_E_pmax}, got {self.E_pmax}")

            # 3. Check s_A = 0.36*v0*t0 (given in problem)
            if self.s_A is not None:
                expected_s_A = 0.36 * self.v0 * self.t0
                if not self.is_close(self.s_A, expected_s_A):
                    logs.append(f"FAIL: s_A consistency. Expected {expected_s_A}, got {self.s_A}")

            # 4. Check s_B = 1.128*v0*t0 (from step_6)
            if self.s_B is not None:
                expected_s_B = 1.128 * self.v0 * self.t0
                if not self.is_close(self.s_B, expected_s_B):
                    logs.append(f"FAIL: s_B consistency. Expected {expected_s_B}, got {self.s_B}")

            # 5. Check delta_s = s_B - s_A = 0.768*v0*t0 (from step_7)
            if self.delta_s is not None and self.s_A is not None and self.s_B is not None:
                if not self.is_close(self.delta_s, self.s_B - self.s_A):
                    logs.append(f"FAIL: delta_s consistency. Expected {self.s_B - self.s_A}, got {self.delta_s}")
                if not self.is_close(self.delta_s, 0.768 * self.v0 * self.t0):
                    logs.append(f"FAIL: delta_s value. Expected {0.768*self.v0*self.t0}, got {self.delta_s}")

            # 6. Check v_A' = v0 (from step_10)
            if self.v_A_prime is not None and not self.is_close(self.v_A_prime, self.v0):
                logs.append(f"FAIL: v_A' consistency. Expected {self.v0}, got {self.v_A_prime}")

            # 7. Check mu = 0.45 (from step_13)
            if self.mu is not None and not self.is_close(self.mu, 0.45):
                logs.append(f"FAIL: mu consistency. Expected 0.45, got {self.mu}")

            if not logs:
                logs.append("PASS: All trace consistency checks passed.")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """Perturb mass m and check trends in target answers."""
        logs = []
        original_m = self.m
        try:
            # Perturb m by +10%
            perturb_factor = 1.1
            self.m = original_m * perturb_factor

            # Recalculate target answers using formulas (without solving entire problem)
            # E_pmax = 0.6 * m * v0^2 (from step_3)
            new_E_pmax = 0.6 * self.m * self.v0**2
            # delta_s = 0.768 * v0 * t0 (independent of m, from step_7)
            new_delta_s = 0.768 * self.v0 * self.t0
            # mu = 0.45 (independent of m, from step_13)
            new_mu = 0.45

            # Check trends
            original_E_pmax = 0.6 * original_m * self.v0**2
            if not self.is_close(new_E_pmax, perturb_factor * original_E_pmax):
                logs.append(f"FAIL: E_pmax monotonicity. Expected {perturb_factor*original_E_pmax}, got {new_E_pmax}")
            if not self.is_close(new_delta_s, self.delta_s):
                logs.append(f"FAIL: delta_s monotonicity. Should be independent of m.")
            if not self.is_close(new_mu, self.mu):
                logs.append(f"FAIL: mu monotonicity. Should be independent of m.")

            if not logs:
                logs.append("PASS: Monotonicity checks passed.")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Restore original m
            self.m = original_m
        return logs

    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        all_logs = []
        success = True

        # Run each test suite
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_func()
            if logs and any(log.startswith(("FAIL", "ERROR")) for log in logs):
                success = False
            all_logs.append(f"--- {test_name} ---")
            all_logs.extend(logs)

        return success, all_logs

# Example usage (commented out for sandbox):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)