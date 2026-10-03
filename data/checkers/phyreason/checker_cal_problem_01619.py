import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Inputs (with safe defaults)
        self.L = getattr(solution_instance, 'L', 3.0)
        self.m = getattr(solution_instance, 'm', 0.1)
        self.mu = getattr(solution_instance, 'mu', 0.4)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        self.theta = math.radians(self.theta_deg)
        self.H = getattr(solution_instance, 'H', 0.45)
        self.R = getattr(solution_instance, 'R', 0.4)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)  # computed as sqrt(1-0.6^2)=0.8

        # Results (safely extract from results dict or attributes)
        # Use solution_instance.results if available, else fallback to attributes
        results = getattr(solution_instance, 'results', {})
        self.v_D = getattr(solution_instance, 'v_D', results.get('v_D', 0.0))
        self.v_y = getattr(solution_instance, 'v_y', results.get('v_y', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', results.get('v_B', 0.0))
        self.W_f = getattr(solution_instance, 'W_f', results.get('W_f', 0.0))
        self.v_A = getattr(solution_instance, 'v_A', results.get('v_A', 0.0))
        self.s = getattr(solution_instance, 's', results.get('s', 0.0))
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))

        # For symbolic computations
        self.symbols = {}
        self._init_symbols()

    def _init_symbols(self):
        """Define symbolic variables for invariant checks."""
        m, g, R, theta, v_D, v_B, v_A, v_y, mu, s, H = sp.symbols('m g R theta v_D v_B v_A v_y mu s H')
        self.symbols.update(locals())
        # Remove the local variables we don't want to store
        del m, g, R, theta, v_D, v_B, v_A, v_y, mu, s, H

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Centripetal condition at D.
        2. Work-energy on conveyor belt.
        3. Free-fall vertical velocity.
        4. Tangential velocity direction at B.
        5. Energy conservation from B to D (with friction).
        """
        logs = []
        try:
            # 1. At top D: mg = m*v_D^2 / R  => v_D^2 = g*R
            expected_vD_sq = self.g * self.R
            actual_vD_sq = self.v_D**2
            if self._isclose(actual_vD_sq, expected_vD_sq):
                logs.append("PASS: Centripetal condition at D: v_D^2 ≈ g*R")
            else:
                logs.append(f"FAIL: Centripetal condition at D: v_D^2={actual_vD_sq}, g*R={expected_vD_sq}")

            # 2. Work by friction on conveyor: μ*m*g*s = 0.5*m*v_A^2
            left = self.mu * self.m * self.g * self.s
            right = 0.5 * self.m * self.v_A**2
            if self._isclose(left, right):
                logs.append("PASS: Work-energy on conveyor belt: μmg*s = 0.5*m*v_A^2")
            else:
                logs.append(f"FAIL: Work-energy on conveyor: μmg*s={left}, 0.5*m*v_A^2={right}")

            # 3. Free fall from height H: v_y^2 = 2*g*H
            expected_vy_sq = 2 * self.g * self.H
            actual_vy_sq = self.v_y**2
            if self._isclose(actual_vy_sq, expected_vy_sq):
                logs.append("PASS: Vertical velocity from free fall: v_y^2 ≈ 2gH")
            else:
                logs.append(f"FAIL: Vertical velocity: v_y^2={actual_vy_sq}, 2gH={expected_vy_sq}")

            # 4. At B, velocity components: v_y = v_B * sinθ and v_A = v_B * cosθ
            # Check v_y = v_B * sinθ
            expected_vy_from_vB = self.v_B * self.sin37
            if self._isclose(self.v_y, expected_vy_from_vB):
                logs.append("PASS: v_y = v_B * sinθ")
            else:
                logs.append(f"FAIL: v_y={self.v_y}, v_B*sinθ={expected_vy_from_vB}")
            # Check v_A = v_B * cosθ
            expected_vA_from_vB = self.v_B * self.cos37
            if self._isclose(self.v_A, expected_vA_from_vB):
                logs.append("PASS: v_A = v_B * cosθ")
            else:
                logs.append(f"FAIL: v_A={self.v_A}, v_B*cosθ={expected_vA_from_vB}")

            # 5. Energy conservation from B to D: -mgR(1+cosθ) - W_f = 0.5*m*(v_D^2 - v_B^2)
            delta_K = 0.5 * self.m * (self.v_D**2 - self.v_B**2)
            delta_U = self.m * self.g * self.R * (1 + self.cos37)
            left_energy = -delta_U - self.W_f
            if self._isclose(left_energy, delta_K):
                logs.append("PASS: Energy conservation from B to D (with friction)")
            else:
                logs.append(f"FAIL: Energy B->D: -ΔU - W_f={left_energy}, ΔK={delta_K}")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results using the step-by-step formulas.
        """
        logs = []
        try:
            # 1. v_D from circular motion: v_D = sqrt(g*R)
            v_D_expected = math.sqrt(self.g * self.R)
            if self._isclose(self.v_D, v_D_expected):
                logs.append(f"PASS: v_D = sqrt(gR) = {v_D_expected}")
            else:
                logs.append(f"FAIL: v_D={self.v_D}, expected {v_D_expected}")

            # 2. v_y from free fall: v_y = sqrt(2gH)
            v_y_expected = math.sqrt(2 * self.g * self.H)
            if self._isclose(self.v_y, v_y_expected):
                logs.append(f"PASS: v_y = sqrt(2gH) = {v_y_expected}")
            else:
                logs.append(f"FAIL: v_y={self.v_y}, expected {v_y_expected}")

            # 3. v_B from v_y and sinθ: v_B = v_y / sinθ
            v_B_expected = self.v_y / self.sin37
            if self._isclose(self.v_B, v_B_expected):
                logs.append(f"PASS: v_B = v_y / sinθ = {v_B_expected}")
            else:
                logs.append(f"FAIL: v_B={self.v_B}, expected {v_B_expected}")

            # 4. v_A from v_y and tanθ: v_A = v_y / tanθ
            tan37 = self.sin37 / self.cos37
            v_A_expected = self.v_y / tan37
            if self._isclose(self.v_A, v_A_expected):
                logs.append(f"PASS: v_A = v_y / tanθ = {v_A_expected}")
            else:
                logs.append(f"FAIL: v_A={self.v_A}, expected {v_A_expected}")

            # 5. s from v_A and friction: s = v_A^2 / (2*μ*g)
            s_expected = self.v_A**2 / (2 * self.mu * self.g)
            if self._isclose(self.s, s_expected):
                logs.append(f"PASS: s = v_A^2/(2μg) = {s_expected}")
            else:
                logs.append(f"FAIL: s={self.s}, expected {s_expected}")

            # 6. W_f from energy B to D: W_f = -0.5*m*(v_D^2 - v_B^2) - m*g*R*(1+cosθ)
            W_f_expected = -0.5 * self.m * (self.v_D**2 - self.v_B**2) - self.m * self.g * self.R * (1 + self.cos37)
            if self._isclose(self.W_f, W_f_expected):
                logs.append(f"PASS: W_f = {W_f_expected}")
            else:
                logs.append(f"FAIL: W_f={self.W_f}, expected {W_f_expected}")

            # 7. Conveyor belt speed v should equal v_A (since block accelerates from 0 to v_A)
            if self._isclose(self.v, self.v_A):
                logs.append(f"PASS: Conveyor belt speed v = v_A = {self.v_A}")
            else:
                logs.append(f"FAIL: v={self.v}, v_A={self.v_A}")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """
        Perturb input parameters and check trends in results.
        """
        logs = []
        try:
            # Test 1: Increase μ, s should decrease (s = v_A^2/(2μg), v_A independent of μ)
            original_mu = self.mu
            perturbed_mu = original_mu * 1.1  # +10%
            s_perturbed = self.v_A**2 / (2 * perturbed_mu * self.g)
            if s_perturbed < self.s:
                logs.append(f"PASS: Increasing μ from {original_mu} to {perturbed_mu} decreases s from {self.s} to {s_perturbed}")
            else:
                logs.append(f"FAIL: Increasing μ did not decrease s. s_original={self.s}, s_perturbed={s_perturbed}")

            # Test 2: Increase m, W_f should increase proportionally (W_f ∝ m)
            original_m = self.m
            perturbed_m = original_m * 1.1  # +10%
            # Recompute W_f using the formula with perturbed m, other quantities unchanged
            W_f_perturbed = -0.5 * perturbed_m * (self.v_D**2 - self.v_B**2) - perturbed_m * self.g * self.R * (1 + self.cos37)
            if W_f_perturbed > self.W_f:
                logs.append(f"PASS: Increasing m from {original_m} to {perturbed_m} increases W_f from {self.W_f} to {W_f_perturbed}")
            else:
                logs.append(f"FAIL: Increasing m did not increase W_f. W_f_original={self.W_f}, W_f_perturbed={W_f_perturbed}")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        logs = []
        logs.append("=== PhysicsChecker started ===")
        
        # Run individual test suites
        inv_logs = self.test_invariants()
        logs.extend(inv_logs)
        
        trace_logs = self.test_trace_consistency()
        logs.extend(trace_logs)
        
        mono_logs = self.test_monotonicity()
        logs.extend(mono_logs)
        
        # Determine overall success: no failures in invariants and trace consistency
        success = True
        for log in inv_logs + trace_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        logs.append(f"=== PhysicsChecker finished: {'PASS' if success else 'FAIL'} ===")
        return success, logs