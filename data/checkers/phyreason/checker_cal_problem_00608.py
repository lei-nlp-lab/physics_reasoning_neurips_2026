import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Constants from problem statement
        self.g = 10.0  # m/s^2
        self.theta = math.radians(37)  # radians, but we'll use sin=0.6, cos=0.8
        self.sin_theta = 0.6
        self.cos_theta = 0.8  # since sin37=0.6, cos37=0.8
        self.l1_given = 32.0  # m
        self.l2_given = 51.2  # m
        self.t_total_given = 10.4  # s
        self.m = 60.0  # kg (mass, though it cancels in many checks)

        # Safely extract results from solution_instance
        # Use getattr and .get() to avoid AttributeError
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results with fallback defaults (0.0 for safety)
        self.v_m = getattr(solution_instance, 'v_m', self.results.get('v_m', 0.0))
        self.t_2 = getattr(solution_instance, 't_2', self.results.get('t_2', 0.0))
        self.mu = getattr(solution_instance, 'mu', self.results.get('mu', 0.0))

        # Validate extracted values are numeric
        try:
            self.v_m = float(self.v_m)
            self.t_2 = float(self.t_2)
            self.mu = float(self.mu)
        except (TypeError, ValueError):
            # If conversion fails, set to 0.0 to avoid crashes
            self.v_m = 0.0
            self.t_2 = 0.0
            self.mu = 0.0

        # For monotonicity test, store original inputs to restore later
        self.original_l1 = self.l1_given
        self.original_l2 = self.l2_given

    def test_invariants(self):
        """Verify physical invariants: kinematic equations and work-energy theorem."""
        logs = []
        try:
            # 1. Check kinematic equations from steps 1 and 2
            # Step 1: l1 = (v_m/2) * (t_total - t_2)
            t1_calc = self.t_total_given - self.t_2
            l1_calc = (self.v_m / 2.0) * t1_calc
            if not math.isclose(l1_calc, self.l1_given, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL Step1 kinematics: l1_calc={l1_calc:.3f}, given={self.l1_given:.3f}")
            else:
                logs.append(f"PASS Step1 kinematics: l1 consistent")

            # Step 2: l2 = (v_m/2) * t_2
            l2_calc = (self.v_m / 2.0) * self.t_2
            if not math.isclose(l2_calc, self.l2_given, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL Step2 kinematics: l2_calc={l2_calc:.3f}, given={self.l2_given:.3f}")
            else:
                logs.append(f"PASS Step2 kinematics: l2 consistent")

            # 2. Check work-energy theorem (net work = 0 from start to rest)
            # Net work = m*g*l1*sinθ - μ*m*g*(l1*cosθ + l2) = 0
            work_gravity = self.m * self.g * self.l1_given * self.sin_theta
            work_friction = self.mu * self.m * self.g * (self.l1_given * self.cos_theta + self.l2_given)
            net_work = work_gravity - work_friction
            if not math.isclose(net_work, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL Work-energy: net_work={net_work:.3f} != 0 (gravity={work_gravity:.3f}, friction={work_friction:.3f})")
            else:
                logs.append(f"PASS Work-energy: net work zero (μ={self.mu:.3f})")

            # 3. Symbolic check with SymPy (fallback to numerical if needed)
            try:
                m, g, l1, l2, mu, sin_th, cos_th = sp.symbols('m g l1 l2 mu sin_th cos_th', positive=True)
                expr = m*g*l1*sin_th - mu*m*g*(l1*cos_th + l2)
                # Substitute values
                sub_dict = {
                    m: self.m,
                    g: self.g,
                    l1: self.l1_given,
                    l2: self.l2_given,
                    mu: self.mu,
                    sin_th: self.sin_theta,
                    cos_th: self.cos_theta
                }
                expr_val = expr.subs(sub_dict).evalf()
                if not math.isclose(float(expr_val), 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL Sympy work-energy: expr={expr_val:.3f}")
                else:
                    logs.append(f"PASS Sympy work-energy check")
            except Exception as e:
                logs.append(f"Sympy fallback failed: {e}, using numeric check above.")

        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # From step1 and step2, we can solve for v_m and t_2 independently
            # Equations:
            #   l1 = (v_m/2) * (t_total - t_2)   --> (1)
            #   l2 = (v_m/2) * t_2               --> (2)
            # From (2): v_m = 2*l2 / t_2
            # Substitute into (1): l1 = (l2 / t_2) * (t_total - t_2)
            # => l1 * t_2 = l2 * (t_total - t_2)
            # => t_2 = (l2 * t_total) / (l1 + l2)
            t_2_calc = (self.l2_given * self.t_total_given) / (self.l1_given + self.l2_given)
            if not math.isclose(t_2_calc, self.t_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL t_2 consistency: calc={t_2_calc:.3f}, result={self.t_2:.3f}")
            else:
                logs.append(f"PASS t_2 consistency")

            # Then v_m = 2*l2 / t_2
            v_m_calc = 2.0 * self.l2_given / t_2_calc
            if not math.isclose(v_m_calc, self.v_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL v_m consistency: calc={v_m_calc:.3f}, result={self.v_m:.3f}")
            else:
                logs.append(f"PASS v_m consistency")

            # Alternatively, from the two equations, v_m = 2*(l1 + l2)/t_total
            v_m_alt = 2.0 * (self.l1_given + self.l2_given) / self.t_total_given
            if not math.isclose(v_m_alt, self.v_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL v_m alternate: calc={v_m_alt:.3f}, result={self.v_m:.3f}")
            else:
                logs.append(f"PASS v_m alternate formula")

        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
        return logs

    def test_monotonicity(self):
        """Perturb an input (l1) and check trends in v_m, t_2, and μ."""
        logs = []
        try:
            # Store original
            l1_original = self.l1_given

            # Perturb l1 by +10% (small but noticeable)
            perturbation = 0.1  # 10%
            self.l1_given = l1_original * (1.0 + perturbation)

            # Recalculate using derived formulas (without using solution's internal methods)
            # Formulas derived from kinematics:
            #   t_2_new = (l2 * t_total) / (l1_new + l2)
            #   v_m_new = 2*(l1_new + l2) / t_total
            #   μ_new = (l1_new * sinθ) / (l1_new * cosθ + l2)
            t_2_new = (self.l2_given * self.t_total_given) / (self.l1_given + self.l2_given)
            v_m_new = 2.0 * (self.l1_given + self.l2_given) / self.t_total_given
            mu_new = (self.l1_given * self.sin_theta) / (self.l1_given * self.cos_theta + self.l2_given)

            # Check trends (monotonicity)
            # As l1 increases, v_m should increase (since v_m = 2*(l1+l2)/t)
            if not v_m_new > self.v_m:
                logs.append(f"FAIL monotonicity v_m: increased l1 but v_m did not increase (new={v_m_new:.3f}, old={self.v_m:.3f})")
            else:
                logs.append(f"PASS v_m monotonic: increases with l1")

            # As l1 increases, t_2 should decrease (since t_2 = l2*t/(l1+l2))
            if not t_2_new < self.t_2:
                logs.append(f"FAIL monotonicity t_2: increased l1 but t_2 did not decrease (new={t_2_new:.3f}, old={self.t_2:.3f})")
            else:
                logs.append(f"PASS t_2 monotonic: decreases with l1")

            # As l1 increases, μ should increase (derivative positive)
            if not mu_new > self.mu:
                logs.append(f"FAIL monotonicity μ: increased l1 but μ did not increase (new={mu_new:.3f}, old={self.mu:.3f})")
            else:
                logs.append(f"PASS μ monotonic: increases with l1")

            # Restore original l1
            self.l1_given = l1_original

        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
            # Ensure restoration on error
            self.l1_given = self.original_l1
        return logs

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        logs = []
        success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                test_logs = test_method()
                if any("FAIL" in log for log in test_logs):
                    success = False
                logs.append(f"--- {test_name} ---")
                logs.extend(test_logs)
            except Exception as e:
                success = False
                logs.append(f"Error in {test_name}: {traceback.format_exc()}")

        # Summary
        logs.append(f"\nOverall result: {'PASS' if success else 'FAIL'}")
        return success, logs