import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Do NOT directly access solution_instance.attr; use getattr with fallback.
        """
        # Input parameters (safely extract with defaults)
        self.theta = math.radians(53)  # θ=53°
        self.mA = getattr(solution_instance, 'mA', 2.0)  # kg
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.mB = getattr(solution_instance, 'mB', 1.6)  # kg
        self.mC = getattr(solution_instance, 'mC', 1.0)  # kg
        self.mD = getattr(solution_instance, 'mD', 0.4)  # kg
        self.k1 = getattr(solution_instance, 'k1', 140.0)  # N/m
        self.k2 = getattr(solution_instance, 'k2', 100.0)  # N/m
        self.alpha = math.radians(74)  # α=74°
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²

        # Trigonometric values (given: sin37°=0.6, cos37°=0.8)
        self.sin37 = 0.6
        self.cos37 = 0.8
        # sin53° = cos37° = 0.8, cos53° = sin37° = 0.6
        self.sin53 = self.cos37
        self.cos53 = self.sin37

        # Extract results from solution_instance.results with fallback defaults
        # Use the key mapping provided
        results = getattr(solution_instance, 'results', {})
        self.f = getattr(solution_instance, 'f', results.get('f', 0.0))
        self.delta_x_3 = getattr(solution_instance, 'delta_x_3', results.get('delta_x_3', 0.0))
        self.delta_x_1 = getattr(solution_instance, 'delta_x_1', results.get('delta_x_1', 0.0))
        self.T = getattr(solution_instance, 'T', results.get('T', 0.0))
        self.G_B = getattr(solution_instance, 'G_B', results.get('G_B', 0.0))
        self.h = getattr(solution_instance, 'h', results.get('h', 0.0))
        self.T_total = getattr(solution_instance, 'T_total', results.get('T_total', 0.0))
        self.delta_x_2 = getattr(solution_instance, 'delta_x_2', results.get('delta_x_2', 0.0))

        # Store logs for reporting
        self.logs = []

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical matching with 1-2% tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify physical invariants:
        1. Force equilibrium on block A (Newton's 1st law).
        2. Force equilibrium on block B (T_total = G_B in equilibrium).
        3. Hooke's law for spring compressions/extensions.
        """
        logs = []
        try:
            # 1. Force equilibrium on block A (parallel to incline)
            # Friction force: f = μ * mA * g * cosθ
            f_expected = self.mu * self.mA * self.g * self.cos53
            if not self._isclose(self.f, f_expected):
                logs.append(f"FAIL: Friction force mismatch. Expected {f_expected:.3f}, got {self.f:.3f}")

            # Net force on A: mA*g*sinθ - T - f = 0 (just about to slide down, friction up incline)
            net_force_A = self.mA * self.g * self.sin53 - self.T - self.f
            if not self._isclose(net_force_A, 0.0):
                logs.append(f"FAIL: Block A not in equilibrium. Net force = {net_force_A:.3f} N")

            # 2. Force equilibrium on block B (vertical direction)
            # T_total = 2T * cos(α/2) (from force composition on movable pulley)
            T_total_expected = 2 * self.T * math.cos(self.alpha / 2)
            if not self._isclose(self.T_total, T_total_expected):
                logs.append(f"FAIL: T_total mismatch. Expected {T_total_expected:.3f}, got {self.T_total:.3f}")

            # Gravity on B: G_B = mB * g
            G_B_expected = self.mB * self.g
            if not self._isclose(self.G_B, G_B_expected):
                logs.append(f"FAIL: G_B mismatch. Expected {G_B_expected:.3f}, got {self.G_B:.3f}")

            # In equilibrium, T_total should balance G_B (since no other vertical forces on B)
            if not self._isclose(self.T_total, self.G_B):
                logs.append(f"FAIL: Block B not in equilibrium. T_total={self.T_total:.3f} != G_B={self.G_B:.3f}")

            # 3. Hooke's law checks for springs
            # Spring 2 compression due to C: Δx_1 = mC * g / k2
            delta_x_1_expected = self.mC * self.g / self.k2
            if not self._isclose(self.delta_x_1, delta_x_1_expected):
                logs.append(f"FAIL: Δx_1 mismatch. Expected {delta_x_1_expected:.3f}, got {self.delta_x_1:.3f}")

            # Spring 1 elongation due to (C+D): Δx_2 = (mC + mD) * g / k1
            delta_x_2_expected = (self.mC + self.mD) * self.g / self.k1
            if not self._isclose(self.delta_x_2, delta_x_2_expected):
                logs.append(f"FAIL: Δx_2 mismatch. Expected {delta_x_2_expected:.3f}, got {self.delta_x_2:.3f}")

            # Spring 2 elongation due to D: Δx_3 = mD * g / k2
            delta_x_3_expected = self.mD * self.g / self.k2
            if not self._isclose(self.delta_x_3, delta_x_3_expected):
                logs.append(f"FAIL: Δx_3 mismatch. Expected {delta_x_3_expected:.3f}, got {self.delta_x_3:.3f}")

            # 4. Energy conservation (optional: gravitational potential vs spring energy)
            # Total spring potential energy: U_spring = 0.5*k1*Δx_2² + 0.5*k2*(Δx_1² + Δx_3²)
            U_spring = 0.5 * self.k1 * self.delta_x_2**2 + 0.5 * self.k2 * (self.delta_x_1**2 + self.delta_x_3**2)
            # Gravitational potential change: ΔU_grav = mB*g*h + mC*g*(h - Δx_2) + mD*g*(h - Δx_2 - Δx_3)
            # But since the system is in equilibrium, we can check that the virtual work is zero.
            # Simpler: Check that the sum of forces on each mass is zero (already done).
            if len(logs) == 0:
                logs.append("PASS: All invariants satisfied.")
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        self.logs.extend(logs)
        return len(logs) == 1 and "PASS" in logs[0]

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        Ensure the calculated h matches the sum of displacements.
        """
        logs = []
        try:
            # h = Δx_1 + Δx_2 + Δx_3 (from step 15)
            h_expected = self.delta_x_1 + self.delta_x_2 + self.delta_x_3
            if not self._isclose(self.h, h_expected):
                logs.append(f"FAIL: Height h mismatch. Expected {h_expected:.3f}, got {self.h:.3f}")

            # T should match the target answer 10 N (within tolerance)
            if not self._isclose(self.T, 10.0):
                logs.append(f"FAIL: Tension T mismatch. Expected 10.0, got {self.T:.3f}")

            # h should match the target answer 0.24 m (within tolerance)
            if not self._isclose(self.h, 0.24):
                logs.append(f"FAIL: Height h mismatch. Expected 0.24, got {self.h:.3f}")

            if len(logs) == 0:
                logs.append("PASS: Trace consistency verified.")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        self.logs.extend(logs)
        return len(logs) == 1 and "PASS" in logs[0]

    def test_monotonicity(self):
        """
        Perturb an input (mass mA) and recalculate T and h to verify trends.
        Restore original value after test.
        """
        logs = []
        original_mA = self.mA
        try:
            # Increase mA by 0.1 kg (5%)
            self.mA += 0.1
            # Recalculate T and h using formulas (not from solution_instance)
            # T = mA*g*sinθ - μ*mA*g*cosθ
            T_perturbed = self.mA * self.g * self.sin53 - self.mu * self.mA * self.g * self.cos53
            # Recalculate spring displacements (they depend on mC, mD, not mA, so unchanged)
            delta_x_1 = self.mC * self.g / self.k2
            delta_x_2 = (self.mC + self.mD) * self.g / self.k1
            delta_x_3 = self.mD * self.g / self.k2
            h_perturbed = delta_x_1 + delta_x_2 + delta_x_3

            # Check monotonicity: T should increase with mA (since dT/dmA = g*(sinθ - μ*cosθ) > 0)
            if T_perturbed <= self.T:
                logs.append(f"FAIL: T does not increase with mA. T_original={self.T:.3f}, T_perturbed={T_perturbed:.3f}")

            # h should remain unchanged (independent of mA)
            if not self._isclose(h_perturbed, self.h):
                logs.append(f"FAIL: h changed with mA perturbation. Expected {self.h:.3f}, got {h_perturbed:.3f}")

            if len(logs) == 0:
                logs.append("PASS: Monotonicity test passed.")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
        finally:
            # Restore original mA
            self.mA = original_mA

        self.logs.extend(logs)
        return len(logs) == 1 and "PASS" in logs[0]

    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []
        success = True

        # Run each test, stop if any fails
        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False

        return success, self.logs