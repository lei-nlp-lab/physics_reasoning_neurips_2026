import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        solution_instance is expected to have a .results dict or attributes.
        """
        # Inputs (with safe defaults)
        self.v0 = getattr(solution_instance, 'v0', 5.0)  # initial speed m/s
        self.a_mag = getattr(solution_instance, 'a_mag', 0.4)  # deceleration magnitude m/s²
        self.a = -self.a_mag  # acceleration is negative for deceleration
        self.t1 = getattr(solution_instance, 't1', 20.0)  # 20 seconds for first part
        self.t2 = getattr(solution_instance, 't2', 2.0)   # 2 seconds for second part

        # Results from solution (safe extraction)
        self.x1 = getattr(solution_instance, 'x_1', getattr(solution_instance, 'results', {}).get('x_1', None))
        self.x2 = getattr(solution_instance, 'x_2', getattr(solution_instance, 'results', {}).get('x_2', None))

        # Store original for restoration after perturbations
        self._original_a_mag = self.a_mag
        self._original_v0 = self.v0
        self.logs = []

    def _log(self, msg: str) -> None:
        """Helper to append logs."""
        self.logs.append(msg)

    def _compute_stopping_time(self) -> float:
        """Time until car comes to rest: t_stop = v0 / a_mag."""
        if self.a_mag <= 0:
            return float('inf')
        return self.v0 / self.a_mag

    def _compute_x1_theoretical(self) -> float:
        """
        Distance to stop: x1 = v0² / (2 * a_mag)
        This is the distance until the car stops, which happens before 20s.
        """
        if self.a_mag <= 0:
            return float('inf')
        return self.v0**2 / (2 * self.a_mag)

    def _compute_x2_theoretical(self) -> float:
        """
        Distance in the last 2 seconds before stopping.
        Let t_stop = v0 / a_mag.
        Distance from (t_stop - 2) to t_stop:
        x2 = v0*(t_stop) + 0.5*a*t_stop² - [v0*(t_stop-2) + 0.5*a*(t_stop-2)²]
        Simplifies to: x2 = 2 * a_mag  (independent of v0 for constant deceleration).
        """
        t_stop = self._compute_stopping_time()
        if t_stop < self.t2:
            # Car stops in less than 2 seconds, so the last 2 seconds are not fully defined.
            # In this problem, t_stop = 12.5s > 2s, so we proceed.
            return 0.0
        # Using the simplified formula derived from kinematics
        return 2.0 * self.a_mag

    def test_invariants(self) -> bool:
        """
        Verify energy/work and kinematics consistency.
        Returns True if all checks pass.
        """
        success = True

        # 1. Check x1 using the velocity-displacement formula
        if self.x1 is not None:
            x1_expected = self._compute_x1_theoretical()
            if not math.isclose(self.x1, x1_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL invariant x1: got {self.x1}, expected {x1_expected}")
                success = False
            else:
                self._log(f"PASS invariant x1: {self.x1} ≈ {x1_expected}")

        # 2. Check x2 using the last-2-seconds formula
        if self.x2 is not None:
            x2_expected = self._compute_x2_theoretical()
            if not math.isclose(self.x2, x2_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL invariant x2: got {self.x2}, expected {x2_expected}")
                success = False
            else:
                self._log(f"PASS invariant x2: {self.x2} ≈ {x2_expected}")

        # 3. Symbolic check: v² = v0² + 2*a*x
        # For x1, final velocity is 0.
        if self.x1 is not None:
            lhs = 0.0
            rhs = self.v0**2 + 2 * self.a * self.x1
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL symbolic v² equation: 0 ≠ {rhs}")
                success = False
            else:
                self._log(f"PASS symbolic v² equation: 0 = {rhs}")

        # 4. Time to stop consistency
        t_stop = self._compute_stopping_time()
        if t_stop < self.t1:
            # Car stops before 20s, so x1 should be distance to stop
            x1_from_t_stop = self.v0 * t_stop + 0.5 * self.a * t_stop**2
            if self.x1 is not None and not math.isclose(self.x1, x1_from_t_stop, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL t_stop consistency: x1 {self.x1} ≠ {x1_from_t_stop} from t_stop {t_stop}")
                success = False
            else:
                self._log(f"PASS t_stop consistency: x1 = {x1_from_t_stop}")

        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Here we have two steps, but both are independent final answers.
        We'll check that the two distances are physically plausible.
        """
        success = True

        # x1 should be positive and greater than x2 (since x2 is part of the stopping distance)
        if self.x1 is not None and self.x2 is not None:
            if self.x1 <= 0 or self.x2 <= 0:
                self._log(f"FAIL trace: distances must be positive: x1={self.x1}, x2={self.x2}")
                success = False
            elif self.x2 > self.x1:
                self._log(f"FAIL trace: x2 ({self.x2}) > x1 ({self.x1}) is impossible")
                success = False
            else:
                self._log(f"PASS trace: x1={self.x1}, x2={self.x2} are positive and ordered")

        # Check that the car is still moving during the last 2 seconds
        t_stop = self._compute_stopping_time()
        if t_stop < self.t2:
            self._log(f"FAIL trace: car stops in {t_stop}s, last {self.t2}s not fully defined")
            success = False
        else:
            self._log(f"PASS trace: car stops in {t_stop}s > {self.t2}s")

        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check expected trends.
        We'll perturb a_mag and v0 separately, then restore.
        """
        success = True

        # 1. Perturb deceleration magnitude a_mag (increase by 10%)
        perturbed_a_mag = self.a_mag * 1.1
        original_a_mag = self.a_mag
        self.a_mag = perturbed_a_mag
        self.a = -self.a_mag

        # Recompute expected values
        x1_new = self._compute_x1_theoretical()
        x2_new = self._compute_x2_theoretical()

        # Trend: if |a| increases, x1 should decrease, x2 should increase (since x2 = 2*a_mag)
        if self.x1 is not None and x1_new >= self.x1:
            self._log(f"FAIL monotonicity a_mag↑: x1 should decrease, but {x1_new} >= {self.x1}")
            success = False
        if self.x2 is not None and x2_new <= self.x2:
            self._log(f"FAIL monotonicity a_mag↑: x2 should increase, but {x2_new} <= {self.x2}")
            success = False

        # Restore a_mag
        self.a_mag = original_a_mag
        self.a = -self.a_mag

        # 2. Perturb initial velocity v0 (increase by 10%)
        perturbed_v0 = self.v0 * 1.1
        original_v0 = self.v0
        self.v0 = perturbed_v0

        x1_new_v0 = self._compute_x1_theoretical()
        x2_new_v0 = self._compute_x2_theoretical()  # x2 independent of v0

        # Trend: if v0 increases, x1 should increase (quadratically), x2 unchanged
        if self.x1 is not None and x1_new_v0 <= self.x1:
            self._log(f"FAIL monotonicity v0↑: x1 should increase, but {x1_new_v0} <= {self.x1}")
            success = False
        if self.x2 is not None and not math.isclose(x2_new_v0, self.x2, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL monotonicity v0↑: x2 should be invariant, but {x2_new_v0} ≠ {self.x2}")
            success = False

        # Restore v0
        self.v0 = original_v0

        if success:
            self._log("PASS monotonicity tests: trends match expectations")
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all checks and return (success, logs).
        """
        self.logs = ["=== Physics Auditor Report ==="]

        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()

        success = inv_ok and trace_ok and mono_ok
        self._log(f"\nSummary: invariants={inv_ok}, trace={trace_ok}, monotonicity={mono_ok}")
        self._log(f"Overall: {'PASS' if success else 'FAIL'}")

        return success, self.logs