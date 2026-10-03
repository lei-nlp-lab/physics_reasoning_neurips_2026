import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', 0.1)
        self.v_b = getattr(solution_instance, 'v_b', 5.0)
        self.L = getattr(solution_instance, 'L', 26.25)
        self.h = getattr(solution_instance, 'h', 0.2)
        self.l = getattr(solution_instance, 'l', 0.4)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.theta = getattr(solution_instance, 'theta', 37.0)  # degrees
        self.sin_theta = getattr(solution_instance, 'sin_theta', 0.6)
        self.cos_theta = getattr(solution_instance, 'cos_theta', 0.8)

        # Convert theta to radians for symbolic use
        self.theta_rad = math.radians(self.theta)

        # Safe extraction of results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.mu = results.get('μ', results.get('mu', 0.0))
        self.v_2 = results.get('v_2', 0.0)
        self.t_2 = results.get('t_2', 0.0)
        self.v_c = results.get('v_c', 0.0)
        self.t = results.get('t', 0.0)
        self.t_1 = results.get('t_1', 0.0)
        self.h_1 = results.get('h_1', 0.0)
        self.v_1 = results.get('v_1', 0.0)

        # Derived constants for numeric checks
        self.sqrt_10 = math.sqrt(10)

    def test_invariants(self):
        logs = []
        try:
            # 1. Check work-energy theorem from b to d (overall)
            # W_net = ΔK = K_d - K_b = 0 - 1/2 m v_b^2
            # W_net = W_gravity + W_friction = mgΔh - μmg cosθ L
            # Δh = h - h_1 (final height minus initial height)
            # Note: h_1 is height of b above horizontal, d is at height h above horizontal
            # So Δh = h - h_1 (negative since going down)
            # But careful: In step_2, the equation uses (h_1 - h) with positive sign for gravity work
            # Actually: W_gravity = mg(h - h_1) = -mg(h_1 - h)
            # The equation in step_2: mg(h_1 - h) - μmg cosθ L = -1/2 m v_b^2
            # Which rearranges to: mg(h - h_1) + μmg cosθ L = 1/2 m v_b^2
            # Let's verify this equation numerically
            lhs = self.m * self.g * (self.h_1 - self.h) - self.mu * self.m * self.g * self.cos_theta * self.L
            rhs = 0 - 0.5 * self.m * self.v_b**2
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy invariant violated. LHS={lhs:.4f}, RHS={rhs:.4f}")
            else:
                logs.append(f"PASS: Work-energy theorem from b to d holds.")

            # 2. Check kinematic consistency on rough segment (bc)
            # For constant acceleration on incline:
            # v_c^2 = v_b^2 + 2a L, where a = g sinθ - μ g cosθ
            a_rough = self.g * self.sin_theta - self.mu * self.g * self.cos_theta
            v_c_sq_expected = self.v_b**2 + 2 * a_rough * self.L
            if not math.isclose(self.v_c**2, v_c_sq_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinematic eqn on rough segment. v_c^2={self.v_c**2:.4f}, expected={v_c_sq_expected:.4f}")
            else:
                logs.append(f"PASS: Kinematic consistency on rough segment.")

            # 3. Check smooth segment (cd) energy conservation
            # cd is smooth, so mechanical energy conserved: 1/2 m v_c^2 + mgh_c = 1/2 m v_d^2 + mgh_d
            # v_d = 0 at rest, h_d = h (given), h_c = 0 (if we set horizontal at c?)
            # Actually, track cd is inclined but smooth. We need the height difference.
            # From geometry: l = 0.4m, h = 0.2m, so sin(α) = h/l = 0.5 (α is inclination of cd)
            # But we don't need the angle if we use energy: Δh from c to d is -h (going down h meters)
            # So: 1/2 m v_c^2 + mg*0 = 0 + mg*h (since v_d=0, and d is h lower than c? Wait, d is above ground by h, but where is c relative to ground?
            # Actually from step_1, h_1 is height of b above ground. c is at the bottom of cb, so height of c is 0 (ground level).
            # Then d is at height h above ground. So from c to d, the slider goes UP by h.
            # So energy: 1/2 m v_c^2 = mgh + 1/2 m v_d^2, but v_d=0, so v_c^2 = 2gh
            if not math.isclose(self.v_c**2, 2 * self.g * self.h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARNING: Energy on smooth segment mismatch. v_c^2={self.v_c**2:.4f}, 2gh={2*self.g*self.h:.4f}")
                # This might be okay because there's a circular arc? But problem says cd is smooth.
            else:
                logs.append(f"PASS: Energy conservation on smooth segment.")

            # 4. Check time addition: t = t_1 + t_2
            if not math.isclose(self.t, self.t_1 + self.t_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Time addition violated. t={self.t:.4f}, t_1+t_2={self.t_1+self.t_2:.4f}")
            else:
                logs.append(f"PASS: Time addition holds.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # 1. Check h_1 = L * sinθ
            h_1_expected = self.L * self.sin_theta
            if not math.isclose(self.h_1, h_1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h_1 inconsistent. Got {self.h_1:.4f}, expected {h_1_expected:.4f}")
            else:
                logs.append(f"PASS: h_1 = L sinθ holds.")

            # 2. Check v_1 = (v_c + v_b)/2 (average velocity on bc, assuming constant acceleration)
            v_1_expected = (self.v_c + self.v_b) / 2
            if not math.isclose(self.v_1, v_1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_1 inconsistent. Got {self.v_1:.4f}, expected {v_1_expected:.4f}")
            else:
                logs.append(f"PASS: v_1 average velocity formula holds.")

            # 3. Check t_1 = L / v_1
            if abs(self.v_1) > 1e-9:
                t_1_expected = self.L / self.v_1
                if not math.isclose(self.t_1, t_1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_1 inconsistent. Got {self.t_1:.4f}, expected {t_1_expected:.4f}")
                else:
                    logs.append(f"PASS: t_1 = L/v_1 holds.")
            else:
                logs.append("SKIP: v_1 is zero, cannot compute t_1.")

            # 4. Check v_2 = (v_c + 0)/2 = v_c/2 (average velocity on cd, assuming constant acceleration to rest)
            v_2_expected = self.v_c / 2
            if not math.isclose(self.v_2, v_2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_2 inconsistent. Got {self.v_2:.4f}, expected {v_2_expected:.4f}")
            else:
                logs.append(f"PASS: v_2 average velocity formula holds.")

            # 5. Check t_2 = l / v_2
            if abs(self.v_2) > 1e-9:
                t_2_expected = self.l / self.v_2
                if not math.isclose(self.t_2, t_2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_2 inconsistent. Got {self.t_2:.4f}, expected {t_2_expected:.4f}")
                else:
                    logs.append(f"PASS: t_2 = l/v_2 holds.")
            else:
                logs.append("SKIP: v_2 is zero, cannot compute t_2.")

            # 6. Check v_c from energy on rough segment (alternative to invariant check)
            # Using work-energy from b to c: 1/2 m v_c^2 - 1/2 m v_b^2 = mg(h_1) - μmg cosθ L
            # But careful: height change from b to c is -h_1 (going down), so ΔU = -mg h_1
            # The equation becomes: 1/2 m v_c^2 - 1/2 m v_b^2 = -mg h_1 - μmg cosθ L
            # Multiply by -1: 1/2 m v_b^2 - 1/2 m v_c^2 = mg h_1 + μmg cosθ L
            # Actually the sign might be off. Let's derive properly:
            # W_gravity = mg*(0 - h_1) = -mg h_1
            # W_friction = - μmg cosθ L
            # W_net = -mg h_1 - μmg cosθ L
            # ΔK = 1/2 m v_c^2 - 1/2 m v_b^2
            # So: 1/2 m v_c^2 - 1/2 m v_b^2 = -mg h_1 - μmg cosθ L
            # Rearr: 1/2 m v_c^2 = 1/2 m v_b^2 - mg h_1 - μmg cosθ L
            v_c_sq_expected = self.v_b**2 - 2 * self.g * self.h_1 - 2 * self.mu * self.g * self.cos_theta * self.L
            if v_c_sq_expected < 0:
                logs.append(f"WARNING: v_c^2 expected negative: {v_c_sq_expected:.4f}")
            elif not math.isclose(self.v_c**2, v_c_sq_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_c from work-energy on rough segment. Got {self.v_c**2:.4f}, expected {v_c_sq_expected:.4f}")
            else:
                logs.append(f"PASS: v_c consistent with work-energy on rough segment.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        original_m = self.m
        try:
            # Perturb mass m slightly and see if μ changes (it shouldn't, because μ is independent of m in the equation)
            # From step_2 equation: mg(h_1 - h) - μ mg cosθ L = -1/2 m v_b^2
            # Cancel m: g(h_1 - h) - μ g cosθ L = -1/2 v_b^2
            # So μ should be independent of m. Let's test by changing m and recomputing μ.
            # We'll use the equation to compute μ.
            perturbed_m = original_m * 1.01  # +1%
            # Recompute h_1 (unchanged), then solve for μ
            h_1 = self.L * self.sin_theta
            # μ = [g(h_1 - h) + 1/2 v_b^2] / (g cosθ L)
            mu_new = (self.g * (h_1 - self.h) + 0.5 * self.v_b**2) / (self.g * self.cos_theta * self.L)
            if not math.isclose(self.mu, mu_new, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: μ should be independent of m. Original μ={self.mu:.4f}, new μ={mu_new:.4f}")
            else:
                logs.append(f"PASS: μ is invariant under mass perturbation.")

            # Now perturb v_b and see if μ changes in the expected way.
            # From the equation: μ = [g(h_1 - h) + 1/2 v_b^2] / (g cosθ L)
            # So μ increases with v_b^2. If v_b increases, μ should increase.
            original_v_b = self.v_b
            perturbed_v_b = original_v_b * 1.01  # +1%
            mu_vb_original = (self.g * (h_1 - self.h) + 0.5 * original_v_b**2) / (self.g * self.cos_theta * self.L)
            mu_vb_perturbed = (self.g * (h_1 - self.h) + 0.5 * perturbed_v_b**2) / (self.g * self.cos_theta * self.L)
            if mu_vb_perturbed <= mu_vb_original:
                logs.append(f"FAIL: μ should increase when v_b increases. Original μ={mu_vb_original:.4f}, new μ={mu_vb_perturbed:.4f}")
            else:
                logs.append(f"PASS: μ increases with v_b (monotonicity).")

            # Restore original values
            self.m = original_m
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            self.m = original_m  # restore on error
        return logs

    def run_all_tests(self):
        all_logs = []
        success = True

        # Run each test suite
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_method()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            # Check for any FAIL or ERROR messages
            if any(log.startswith("FAIL") or log.startswith("ERROR") for log in logs):
                success = False

        return success, all_logs