import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safely extract inputs with defaults from problem statement
        self.R = getattr(solution_instance, 'R', 0.45)
        self.M = getattr(solution_instance, 'M', 5.0)
        self.L = getattr(solution_instance, 'L', 0.75)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Safely extract results from solution_instance.results or attributes
        results = getattr(solution_instance, 'results', {})
        self.v_B = getattr(solution_instance, 'v_B', results.get('v_B', None))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', None))
        # Handle F_N'' key: might be stored as "F_N''" or "F_N_prime"
        self.F_N_prime = getattr(solution_instance, 'F_N_prime', 
                                 results.get("F_N''", results.get('F_N_prime', None)))
        self.Q = getattr(solution_instance, 'Q', results.get('Q', None))
        self.W_f = getattr(solution_instance, 'W_f', results.get('W_f', None))
        self.v = getattr(solution_instance, 'v', results.get('v', None))
        self.v_total = getattr(solution_instance, 'v_total', 
                               results.get('v_{total}', results.get('v_total', None)))

        # Store original inputs for monotonicity test
        self._original_inputs = {
            'R': self.R, 'M': self.M, 'L': self.L, 'm': self.m, 'g': self.g
        }
        self._original_results = {
            'v_B': self.v_B, 'F_N': self.F_N, 'F_N_prime': self.F_N_prime,
            'Q': self.Q, 'W_f': self.W_f, 'v': self.v, 'v_total': self.v_total
        }

        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def _isclose(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # 1. Conservation of mechanical energy (step 1): mgR = 0.5*m*v_B^2
        if self.v_B is not None:
            expected_v_B_sq = 2 * self.g * self.R
            if not self._isclose(self.v_B**2, expected_v_B_sq):
                logs.append(f"Energy invariant failed: v_B^2 = {self.v_B**2}, expected {expected_v_B_sq}")
                success = False
        else:
            logs.append("Missing v_B for energy invariant")
            success = False

        # 2. Newton's second law at B (step 2): F_N = m*g + m*v_B^2/R
        if self.F_N is not None and self.v_B is not None:
            expected_F_N = self.m * (self.g + self.v_B**2 / self.R)
            if not self._isclose(self.F_N, expected_F_N):
                logs.append(f"Newton's second law failed: F_N = {self.F_N}, expected {expected_F_N}")
                success = False
        else:
            logs.append("Missing F_N or v_B for Newton's second law")
            success = False

        # 3. Newton's third law (step 3): F_N' = F_N
        if self.F_N_prime is not None and self.F_N is not None:
            if not self._isclose(self.F_N_prime, self.F_N):
                logs.append(f"Newton's third law failed: F_N' = {self.F_N_prime}, F_N = {self.F_N}")
                success = False
        else:
            logs.append("Missing F_N' or F_N for Newton's third law")
            success = False

        # 4. Conservation of momentum (step 4): m*v_B = (m+M)*v_total
        if self.v_B is not None and self.v_total is not None:
            lhs = self.m * self.v_B
            rhs = (self.m + self.M) * self.v_total
            if not self._isclose(lhs, rhs):
                logs.append(f"Momentum conservation failed: m*v_B = {lhs}, (m+M)*v_total = {rhs}")
                success = False
        else:
            logs.append("Missing v_B or v_total for momentum conservation")
            success = False

        # 5. Heat from energy loss (step 5): Q = 0.5*m*v_B^2 - 0.5*(m+M)*v_total^2
        if self.Q is not None and self.v_B is not None and self.v_total is not None:
            expected_Q = 0.5 * self.m * self.v_B**2 - 0.5 * (self.m + self.M) * self.v_total**2
            if not self._isclose(self.Q, expected_Q):
                logs.append(f"Heat energy failed: Q = {self.Q}, expected {expected_Q}")
                success = False
        else:
            logs.append("Missing Q, v_B, or v_total for heat energy check")
            success = False

        # 6. Work-energy theorem for block on cart (step 8): -W_f = 0.5*m*v^2 - 0.5*m*v_B^2
        if self.W_f is not None and self.v is not None and self.v_B is not None:
            lhs = -self.W_f
            rhs = 0.5 * self.m * self.v**2 - 0.5 * self.m * self.v_B**2
            if not self._isclose(lhs, rhs):
                logs.append(f"Work-energy theorem failed: -W_f = {lhs}, ΔK = {rhs}")
                success = False
        else:
            logs.append("Missing W_f, v, or v_B for work-energy theorem")
            success = False

        # 7. Consistency of heat and work (steps 6 & 7): Q ≈ W_f
        if self.Q is not None and self.W_f is not None:
            if not self._isclose(self.Q, self.W_f):
                logs.append(f"Heat-work consistency failed: Q = {self.Q}, W_f = {self.W_f}")
                success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Check that intermediate results lead to final answers
        # Target answers: F_N ≈ 30, Q ≈ 3.75, v ≈ sqrt(6)/2 ≈ 1.2247
        if self.F_N is not None:
            if not self._isclose(self.F_N, 30.0):
                logs.append(f"F_N inconsistent with target: {self.F_N} vs 30.0")
                success = False

        if self.Q is not None:
            if not self._isclose(self.Q, 3.75):
                logs.append(f"Q inconsistent with target: {self.Q} vs 3.75")
                success = False

        if self.v is not None:
            target_v = math.sqrt(6) / 2
            if not self._isclose(self.v, target_v):
                logs.append(f"v inconsistent with target: {self.v} vs {target_v}")
                success = False

        # Cross-check v_B from energy and used in momentum
        if self.v_B is not None:
            expected_v_B = math.sqrt(2 * self.g * self.R)  # from step 1
            if not self._isclose(self.v_B, expected_v_B):
                logs.append(f"v_B trace inconsistency: {self.v_B} vs {expected_v_B}")
                success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Perturb mass m by +10%
        m_original = self.m
        m_perturbed = m_original * 1.1

        # Recalculate expected results using step formulas
        v_B_new = math.sqrt(2 * self.g * self.R)  # step 1
        F_N_new = m_perturbed * (self.g + v_B_new**2 / self.R)  # step 2
        v_total_new = (m_perturbed * v_B_new) / (m_perturbed + self.M)  # step 4
        Q_new = 0.5 * m_perturbed * v_B_new**2 - 0.5 * (m_perturbed + self.M) * v_total_new**2  # step 5
        W_f_new = Q_new  # assumption from steps 6 & 7
        v_new = math.sqrt(v_B_new**2 - 2 * W_f_new / m_perturbed)  # from step 8

        # Restore original m (important for safety)
        self.m = m_original

        # Check monotonic trends: if m increases, F_N should increase
        if self.F_N is not None:
            if not F_N_new > self.F_N or self._isclose(F_N_new, self.F_N):
                logs.append(f"Monotonicity failed for F_N: {F_N_new} not > {self.F_N}")
                success = False

        # v_total should increase with m (since derivative positive)
        if self.v_total is not None:
            if not v_total_new > self.v_total or self._isclose(v_total_new, self.v_total):
                logs.append(f"Monotonicity failed for v_total: {v_total_new} not > {self.v_total}")
                success = False

        # Q: compute original Q from formula for comparison
        if self.v_B is not None and self.v_total is not None:
            Q_original_formula = 0.5 * self.m * self.v_B**2 - 0.5 * (self.m + self.M) * self.v_total**2
            if not Q_new > Q_original_formula or self._isclose(Q_new, Q_original_formula):
                logs.append(f"Monotonicity failed for Q: {Q_new} not > {Q_original_formula}")
                success = False

        # v: compute original v from formula for comparison
        if self.v_B is not None and self.W_f is not None:
            v_original_formula = math.sqrt(self.v_B**2 - 2 * self.W_f / self.m)
            # Trend: as m increases, v_new vs v_original? Not straightforward, so skip.

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                if logs:
                    all_logs.extend(logs)
                else:
                    all_logs.append("All checks passed.")
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {e}")
                overall_success = False

        return overall_success, all_logs