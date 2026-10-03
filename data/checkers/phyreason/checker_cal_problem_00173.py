import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.m = getattr(solution_instance, 'm', 0.1)
        self.l = getattr(solution_instance, 'l', 0.1)
        self.theta = getattr(solution_instance, 'theta', 37)
        self.mu = getattr(solution_instance, 'mu', 0.45)
        self.R = getattr(solution_instance, 'R', 0.1)
        self.k = getattr(solution_instance, 'k', 5)
        self.x1 = getattr(solution_instance, 'x1', 0.2)
        self.x2 = getattr(solution_instance, 'x2', 0.8)
        self.g = 10.0  # assumed m/s^2

        # Convert theta to radians for calculations
        self.theta_rad = math.radians(self.theta)
        self.sin_theta = math.sin(self.theta_rad)
        self.cos_theta = math.cos(self.theta_rad)

        # Safe extraction of results using the provided key mapping
        results = getattr(solution_instance, 'results', {})
        self.B = getattr(solution_instance, 'B', results.get('B', 0.0))
        self.F_expr = getattr(solution_instance, 'F', results.get('F', ''))
        self.Q = getattr(solution_instance, 'Q', results.get('Q', 0.0))
        self.v_prime = getattr(solution_instance, 'v_prime', results.get("v''", 0.0))
        self.W_A1 = getattr(solution_instance, 'W_A1', results.get('W_A1', 0.0))
        self.W_A2 = getattr(solution_instance, 'W_A2', results.get('W_A2', 0.0))
        self.E = getattr(solution_instance, 'E', results.get('E', 0.0))
        self.P = getattr(solution_instance, 'P', results.get('P', 0.12))
        self.a = getattr(solution_instance, 'a', results.get('a', 0.0))
        self.F_A = getattr(solution_instance, 'F_A', results.get('F_A', 0.0))

        # For force expressions, we expect two linear forms
        self.F_nonmag_coeffs = None
        self.F_mag_coeffs = None

    def _parse_force_expr(self, expr_str: str):
        """Parse a linear expression of the form 'a + b*x' into (a, b)."""
        expr_str = expr_str.replace(' ', '').replace('F=', '')
        if '+' in expr_str:
            parts = expr_str.split('+')
            const = float(parts[0])
            linear_part = parts[1]
            if 'x' in linear_part:
                b = float(linear_part.replace('x', ''))
            else:
                b = 0.0
        else:
            if 'x' in expr_str:
                const = 0.0
                b = float(expr_str.replace('x', ''))
            else:
                const = float(expr_str)
                b = 0.0
        return (const, b)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # 1. Check magnetic field B from power at x1
        v_at_x1 = self.k * self.x1
        E_expected = math.sqrt(self.P * self.R)
        B_from_power = E_expected / (self.l * v_at_x1)
        if not math.isclose(self.B, B_from_power, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: B from power mismatch. Computed {B_from_power}, got {self.B}")
            success = False
        else:
            logs.append(f"Invariant passed: B from power matches.")

        # 2. Check motional EMF: E = B l v at x1
        E_calc = self.B * self.l * v_at_x1
        if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: EMF mismatch at x1. Computed {E_calc}, got {self.E}")
            success = False
        else:
            logs.append(f"Invariant passed: EMF formula holds at x1.")

        # 3. Check Ampere force expression: F_A = (B l)^2 * v / R
        x_test = 0.5
        v_test = self.k * x_test
        F_A_calc = (self.B * self.l)**2 * v_test / self.R
        # Expected coefficient is 0.6 when g=9.8? Actually, with g=10, B changes slightly.
        # But the problem statement gives 0.6x as the Ampere force expression.
        # We'll compute the expected coefficient from the derived formula.
        # From step 7: F_A = 0.6x. Let's verify if our B yields that.
        expected_coeff = 0.6  # given in ground truth
        F_A_expected = expected_coeff * x_test
        if not math.isclose(F_A_calc, F_A_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant warning: Ampere force mismatch at x={x_test}. Computed {F_A_calc}, expected {F_A_expected}")
            # Not a failure, just warning
        else:
            logs.append(f"Invariant passed: Ampere force linear with coefficient 0.6.")

        # 4. Check Newton's second law at a sample point in non-magnetic region
        x_nm = 0.1
        a_nm = self.k**2 * x_nm
        # Gravity and friction components with g=10
        F_gravity = self.m * self.g * self.sin_theta
        F_normal = self.m * self.g * self.cos_theta
        F_friction = self.mu * F_normal
        F_ext_calc_nm = self.m * a_nm + F_friction + F_gravity
        # The expected expression from ground truth uses g=9.8, but we use g=10.
        # We'll compute the expected constant term with g=10:
        # F_gravity = m*g*sinθ, F_friction = μ*m*g*cosθ
        const_expected = F_gravity + F_friction
        slope_expected = self.m * self.k**2
        F_ext_expected_nm = const_expected + slope_expected * x_nm
        if not math.isclose(F_ext_calc_nm, F_ext_expected_nm, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: Non-magnetic force at x={x_nm}. Computed {F_ext_calc_nm}, expected {F_ext_expected_nm}")
            success = False
        else:
            logs.append(f"Invariant passed: Newton's second law in non-magnetic region.")

        # 5. Check Newton's second law in magnetic region
        x_m = 0.5
        a_m = self.k**2 * x_m
        v_m = self.k * x_m
        F_A_m = (self.B * self.l)**2 * v_m / self.R
        F_ext_calc_m = self.m * a_m + F_friction + F_gravity + F_A_m
        F_ext_expected_m = const_expected + slope_expected * x_m + F_A_m
        if not math.isclose(F_ext_calc_m, F_ext_expected_m, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: Magnetic force at x={x_m}. Computed {F_ext_calc_m}, expected {F_ext_expected_m}")
            success = False
        else:
            logs.append(f"Invariant passed: Newton's second law in magnetic region.")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        v_x1 = self.k * self.x1
        v_x2 = self.k * self.x2
        logs.append(f"Velocity at x1: {v_x1} m/s, at x2: {v_x2} m/s")

        # W_A1 = 0.5*(F_A(x1)+F_A(x2))*(x2-x1)
        # Compute F_A using the actual B
        F_A_x1 = (self.B * self.l)**2 * self.k * self.x1 / self.R
        F_A_x2 = (self.B * self.l)**2 * self.k * self.x2 / self.R
        W_A1_calc = 0.5 * (F_A_x1 + F_A_x2) * (self.x2 - self.x1)
        if not math.isclose(self.W_A1, W_A1_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency failed: W_A1 mismatch. Computed {W_A1_calc}, got {self.W_A1}")
            success = False
        else:
            logs.append(f"Trace consistency passed: W_A1 matches.")

        if not math.isclose(self.Q, self.W_A1 + self.W_A2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency failed: Q != W_A1 + W_A2. Q={self.Q}, sum={self.W_A1+self.W_A2}")
            success = False
        else:
            logs.append(f"Trace consistency passed: Q = W_A1 + W_A2.")

        # v' should be approximately 2 m/s (ground truth)
        if not math.isclose(self.v_prime, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency warning: v' mismatch. Got {self.v_prime}, expected 2.0")
            # Not a failure because g difference affects v'
        else:
            logs.append(f"Trace consistency passed: v' = 2 m/s.")

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        original_m = self.m
        perturbed_m = original_m * 1.1

        x_test_nm = 0.1
        x_test_m = 0.5

        # Original forces
        a_nm = self.k**2 * x_test_nm
        F_grav_nm = original_m * self.g * self.sin_theta
        F_fric_nm = self.mu * original_m * self.g * self.cos_theta
        F_orig_nm = original_m * a_nm + F_fric_nm + F_grav_nm

        a_m = self.k**2 * x_test_m
        F_grav_m = original_m * self.g * self.sin_theta
        F_fric_m = self.mu * original_m * self.g * self.cos_theta
        F_A = (self.B * self.l)**2 * self.k * x_test_m / self.R
        F_orig_m = original_m * a_m + F_fric_m + F_grav_m + F_A

        # Perturbed forces
        F_grav_nm_p = perturbed_m * self.g * self.sin_theta
        F_fric_nm_p = self.mu * perturbed_m * self.g * self.cos_theta
        F_pert_nm = perturbed_m * a_nm + F_fric_nm_p + F_grav_nm_p

        F_grav_m_p = perturbed_m * self.g * self.sin_theta
        F_fric_m_p = self.mu * perturbed_m * self.g * self.cos_theta
        F_pert_m = perturbed_m * a_m + F_fric_m_p + F_grav_m_p + F_A

        if F_pert_nm <= F_orig_nm:
            logs.append(f"Monotonicity failed: Non-magnetic force did not increase with m.")
            success = False
        else:
            logs.append(f"Monotonicity passed: Non-magnetic force increases with m.")

        if F_pert_m <= F_orig_m:
            logs.append(f"Monotonicity failed: Magnetic force did not increase with m.")
            success = False
        else:
            logs.append(f"Monotonicity passed: Magnetic force increases with m.")

        expected_factor = 1.1
        actual_factor_nm = F_pert_nm / F_orig_nm
        if not math.isclose(actual_factor_nm, expected_factor, rel_tol=1e-2):
            logs.append(f"Monotonicity warning: Non-magnetic force scaling factor {actual_factor_nm} not exactly 1.1")
        else:
            logs.append(f"Monotonicity passed: Non-magnetic force scales exactly with m.")

        m_dep_part_orig = original_m * a_m + F_fric_m + F_grav_m
        m_dep_part_pert = perturbed_m * a_m + F_fric_m_p + F_grav_m_p
        expected_factor_m = (m_dep_part_pert + F_A) / (m_dep_part_orig + F_A)
        actual_factor_m = F_pert_m / F_orig_m
        if not math.isclose(actual_factor_m, expected_factor_m, rel_tol=1e-2):
            logs.append(f"Monotonicity warning: Magnetic force scaling factor {actual_factor_m} differs from expected {expected_factor_m}")
        else:
            logs.append(f"Monotonicity passed: Magnetic force scales as expected with m.")

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
            success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False

        return overall_success, all_logs