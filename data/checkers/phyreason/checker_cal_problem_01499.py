import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs using getattr and .get()
        self.R = getattr(solution_instance, 'R', 1.0)                 # radius in meters
        self.theta = getattr(solution_instance, 'theta', 37.0)       # degrees
        self.M = getattr(solution_instance, 'M', 1.0)                # board mass in kg
        self.m = getattr(solution_instance, 'm', 1.0)                # block mass in kg
        self.v0 = getattr(solution_instance, 'v0', 1.2)              # initial velocity in m/s
        self.mu = getattr(solution_instance, 'mu', 0.2)              # friction coefficient
        self.g = getattr(solution_instance, 'g', 10.0)               # gravity in m/s^2

        # Extract results safely (assuming they are stored in a dictionary 'results')
        results = getattr(solution_instance, 'results', {})
        self.v_C = results.get('v_C', 0.0)
        self.Q = results.get('Q', 0.0)
        self.v_B = results.get('v_B', 0.0)
        self.a_1 = results.get('a_1', 0.0)
        self.a_2 = results.get('a_2', 0.0)
        self.v = results.get('v', 0.0)

        # Convert theta to radians for calculations
        self.theta_rad = math.radians(self.theta)
        self.sin_theta = math.sin(self.theta_rad)
        self.cos_theta = math.cos(self.theta_rad)

        # Store logs for reporting
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((success, msg))

    def _is_close(self, a, b):
        """Fuzzy numerical matching with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Check fundamental physical laws."""
        success = True

        # 1. Energy conservation from B to C (smooth track, only gravity does work)
        # Expected: v_C^2 = v_B^2 + 2gR(1 + sinθ)
        v_C_sq_expected = self.v_B**2 + 2 * self.g * self.R * (1 + self.sin_theta)
        if not self._is_close(self.v_C**2, v_C_sq_expected):
            msg = f"Energy conservation B->C failed: v_C^2 = {self.v_C**2:.3f}, expected {v_C_sq_expected:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log("Energy conservation B->C satisfied.", True)

        # 2. Momentum conservation during sliding (block + board system, no external horizontal force)
        # Expected: m * v_C = (m + M) * v
        momentum_initial = self.m * self.v_C
        momentum_final = (self.m + self.M) * self.v
        if not self._is_close(momentum_initial, momentum_final):
            msg = f"Momentum conservation failed: initial p = {momentum_initial:.3f}, final p = {momentum_final:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log("Momentum conservation during sliding satisfied.", True)

        # 3. Energy dissipation during sliding
        # Expected: Q = 0.5 * m * v_C^2 - 0.5 * (m + M) * v^2
        Q_expected = 0.5 * self.m * self.v_C**2 - 0.5 * (self.m + self.M) * self.v**2
        if not self._is_close(self.Q, Q_expected):
            msg = f"Energy dissipation mismatch: Q = {self.Q:.3f}, expected {Q_expected:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log("Energy dissipation (heat) consistent with work-energy theorem.", True)

        # 4. Newton's second law for block and board
        a1_expected = self.mu * self.g
        a2_expected = self.mu * self.m * self.g / self.M
        if not self._is_close(self.a_1, a1_expected):
            msg = f"Block acceleration mismatch: a_1 = {self.a_1:.3f}, expected {a1_expected:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log("Block acceleration (Newton's second law) satisfied.", True)

        if not self._is_close(self.a_2, a2_expected):
            msg = f"Board acceleration mismatch: a_2 = {self.a_2:.3f}, expected {a2_expected:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log("Board acceleration (Newton's second law) satisfied.", True)

        return success

    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        success = True

        # 1. Velocity at B from projectile condition: v_B = v0 / sinθ
        v_B_expected = self.v0 / self.sin_theta
        if not self._is_close(self.v_B, v_B_expected):
            msg = f"Velocity at B mismatch: v_B = {self.v_B:.3f}, expected {v_B_expected:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Velocity at B consistent with tangential entry: v_B = {self.v_B:.3f} m/s", True)

        # 2. Common velocity v from kinematics (should match momentum conservation)
        # Using a_1 and a_2, solve for v: v = v_C * a_2 / (a_1 + a_2)
        if self.a_1 + self.a_2 != 0:
            v_expected = self.v_C * self.a_2 / (self.a_1 + self.a_2)
            if not self._is_close(self.v, v_expected):
                msg = f"Common velocity mismatch: v = {self.v:.3f}, expected {v_expected:.3f}"
                self._log(msg, False)
                success = False
            else:
                self._log(f"Common velocity consistent with kinematics: v = {self.v:.3f} m/s", True)
        else:
            self._log("Skipping common velocity check because a_1 + a_2 = 0", False)
            success = False

        # 3. Check that v_C and Q match the target answers (6 m/s and 9 J)
        if not self._is_close(self.v_C, 6.0):
            msg = f"v_C = {self.v_C:.3f} m/s, but expected 6.0 m/s"
            self._log(msg, False)
            success = False
        else:
            self._log(f"v_C matches target: {self.v_C:.3f} m/s", True)

        if not self._is_close(self.Q, 9.0):
            msg = f"Q = {self.Q:.3f} J, but expected 9.0 J"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Q matches target: {self.Q:.3f} J", True)

        return success

    def test_monotonicity(self):
        """Perturb mass m and check trends."""
        success = True

        # Original values
        m_original = self.m
        M = self.M
        v0 = self.v0
        mu = self.mu
        g = self.g
        R = self.R
        sin_theta = self.sin_theta

        # Perturb m by +10%
        m_perturbed = m_original * 1.10

        # Recalculate expected v_C (independent of m, only depends on v0, θ, g, R)
        v_B_perturbed = v0 / sin_theta
        v_C_perturbed = math.sqrt(v_B_perturbed**2 + 2 * g * R * (1 + sin_theta))

        # Recalculate a_1 and a_2
        a1_perturbed = mu * g
        a2_perturbed = mu * m_perturbed * g / M

        # Recalculate common velocity v using momentum conservation: v = (m * v_C) / (m + M)
        v_perturbed = (m_perturbed * v_C_perturbed) / (m_perturbed + M)

        # Recalculate heat Q
        Q_perturbed = 0.5 * m_perturbed * v_C_perturbed**2 - 0.5 * (m_perturbed + M) * v_perturbed**2

        # Check trends:
        # 1. v should increase with m (since derivative positive)
        if v_perturbed <= self.v:
            msg = f"Monotonicity (v vs m) failed: v_original = {self.v:.3f}, v_perturbed = {v_perturbed:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Common velocity increases with m (as expected): {self.v:.3f} -> {v_perturbed:.3f}", True)

        # 2. Q should increase with m (since derivative positive)
        if Q_perturbed <= self.Q:
            msg = f"Monotonicity (Q vs m) failed: Q_original = {self.Q:.3f}, Q_perturbed = {Q_perturbed:.3f}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Heat generated increases with m (as expected): {self.Q:.3f} -> {Q_perturbed:.3f}", True)

        # Restore original m (just in case)
        self.m = m_original
        return success

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs.clear()
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' raised exception: {e}\n{traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs