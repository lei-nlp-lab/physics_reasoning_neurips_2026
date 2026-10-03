import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract inputs with defaults
        self.h = getattr(solution_instance, 'h', None)
        if self.h is None:
            self.h = getattr(solution_instance, 'input_h', 1.0)  # Default for testing

        self.v0 = getattr(solution_instance, 'v0', None)
        if self.v0 is None:
            self.v0 = getattr(solution_instance, 'input_v0', 1.0)  # Default for testing

        self.g = getattr(solution_instance, 'g', 9.8)

        # Extract results safely using mapping
        self.result_v = self.results.get('v', None)  # Velocity when farthest from slope
        self.result_v_prime = self.results.get('v\'', None)  # Landing velocity
        self.result_tanalpha = self.results.get('tanalpha', None)  # tanα at landing
        self.result_tanbeta = self.results.get('tanβ', None)  # tanβ for right-angle hit
        self.result_tangamma = self.results.get('tanγ', None)  # tanγ for right-angle hit
        self.result_x1 = self.results.get('x_1', None)  # Horizontal displacement for right-angle hit
        self.result_y1 = self.results.get('y_1', None)  # Vertical displacement for right-angle hit
        self.result_t = self.results.get('t', None)  # Time of flight for normal landing
        self.result_theta = self.results.get('theta', None)  # Slope angle θ

        # Symbolic variables for physics laws
        self.v0_sym = sp.Symbol('v0', positive=True)
        self.g_sym = sp.Symbol('g', positive=True)
        self.h_sym = sp.Symbol('h', positive=True)
        self.t_sym = sp.Symbol('t', positive=True)
        self.theta_sym = sp.Symbol('theta', positive=True)

        # Derived quantities
        if self.result_theta is None:
            # θ = arctan(4/3) ≈ 53.13°
            self.theta_val = math.atan(4/3) if self.h is None else math.atan(4/3)
        else:
            self.theta_val = float(self.result_theta)

        self.cos_theta = math.cos(self.theta_val)
        self.sin_theta = math.sin(self.theta_val)
        self.tan_theta = math.tan(self.theta_val)

    def test_invariants(self):
        logs = []
        try:
            # 1. Energy conservation for projectile (no air resistance)
            # Initial energy: E0 = 0.5*m*v0^2 + m*g*4h (height from D to ground)
            # At any point: E = 0.5*m*v^2 + m*g*y
            # Since height changes, we need to check energy conservation between specific points

            if self.result_v is not None and self.v0 is not None:
                # Check v = v0 / cosθ (step 2)
                expected_v = self.v0 / self.cos_theta
                if not math.isclose(self.result_v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Energy invariant (v = v0/cosθ) failed: {self.result_v} vs {expected_v}")

            if self.result_v_prime is not None and self.result_t is not None and self.v0 is not None:
                # Energy from launch to landing
                # Initial height = 4h, final height = y_landing (unknown but can use geometry)
                # From step 7: t = 8v0/(3g)
                t_land = 8*self.v0/(3*self.g)
                x_land = self.v0 * t_land
                y_land = 4*self.h - 0.5*self.g*t_land**2  # Height above ground

                E_initial = 0.5*self.v0**2 + self.g*4*self.h
                E_final = 0.5*self.result_v_prime**2 + self.g*y_land

                if not math.isclose(E_initial, E_final, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Energy conservation (launch to landing) failed: {E_initial} vs {E_final}")

            # 2. Newton's Laws: Horizontal velocity constant (vx = v0)
            if self.result_v_prime is not None and self.v0 is not None:
                # vx at landing should equal v0
                # From step 10: tanα = gt/v0, so vy = g*t
                t_land = 8*self.v0/(3*self.g)  # From step 7
                vy_land = self.g * t_land
                vx_land = self.v0
                v_mag = math.sqrt(vx_land**2 + vy_land**2)
                if not math.isclose(self.result_v_prime, v_mag, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Velocity magnitude inconsistent: {self.result_v_prime} vs {v_mag}")

        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Cross-check intermediate results with final answers
            # 1. Check v = 5v0/3 (step 3)
            if self.result_v is not None and self.v0 is not None:
                expected_v = (5/3) * self.v0
                if not math.isclose(self.result_v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency (v) failed: {self.result_v} vs {expected_v}")

            # 2. Check v' = (√73/3)v0 and tanα = 8/3 (step 9 & 11)
            if self.result_v_prime is not None and self.v0 is not None:
                expected_v_prime = (math.sqrt(73)/3) * self.v0
                if not math.isclose(self.result_v_prime, expected_v_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency (v') failed: {self.result_v_prime} vs {expected_v_prime}")

            if self.result_tanalpha is not None:
                if not math.isclose(self.result_tanalpha, 8/3, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency (tanα) failed: {self.result_tanalpha} vs {8/3}")

            # 3. Check right-angle hit geometry (steps 12-16)
            if self.result_x1 is not None and self.result_y1 is not None and self.h is not None:
                # From step 16: x1 = 96h/41, y1 = 36h/41
                expected_x1 = (96/41) * self.h
                expected_y1 = (36/41) * self.h
                if not math.isclose(self.result_x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency (x1) failed: {self.result_x1} vs {expected_x1}")
                if not math.isclose(self.result_y1, expected_y1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency (y1) failed: {self.result_y1} vs {expected_y1}")

                # Check slope condition: y1/x1 should equal tanγ = 3/8 (from step 13-14)
                if self.result_x1 != 0:
                    tan_gamma_calc = self.result_y1 / self.result_x1
                    expected_tan_gamma = 3/8
                    if not math.isclose(tan_gamma_calc, expected_tan_gamma, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"Trace consistency (tanγ) failed: {tan_gamma_calc} vs {expected_tan_gamma}")

        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
        return logs

    def test_monotonicity(self):
        logs = []
        try:
            # Perturb an input (v0) and check if results change monotonically
            # Since v0 appears linearly in most formulas, increasing v0 should increase v, v', etc.
            if self.v0 is not None:
                # Perturb by 10%
                v0_perturbed = self.v0 * 1.1

                # Recalculate expected results using formulas
                v_perturbed = v0_perturbed / self.cos_theta  # v = v0/cosθ
                v_prime_perturbed = (math.sqrt(73)/3) * v0_perturbed  # v' = (√73/3)v0

                # Check monotonicity with original results
                if self.result_v is not None:
                    if not (v_perturbed > self.result_v):
                        logs.append(f"Monotonicity (v) failed: {v_perturbed} ≤ {self.result_v}")
                if self.result_v_prime is not None:
                    if not (v_prime_perturbed > self.result_v_prime):
                        logs.append(f"Monotonicity (v') failed: {v_prime_perturbed} ≤ {self.result_v_prime}")

                # For right-angle hit, x1 and y1 are proportional to h, not v0
                # But from geometry, if v0 increases, h might adjust? We skip for simplicity.

        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
        return logs

    def run_all_tests(self):
        all_logs = []
        try:
            all_logs.extend(self.test_invariants())
            all_logs.extend(self.test_trace_consistency())
            all_logs.extend(self.test_monotonicity())

            success = len(all_logs) == 0
            return success, all_logs
        except Exception as e:
            all_logs.append(f"Run all tests error: {str(e)}")
            return False, all_logs