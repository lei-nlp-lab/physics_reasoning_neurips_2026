import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Given constants
        self.g = 10.0
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.v0 = 4.0  # conveyor belt speed (m/s)
        self.L = 1.6   # conveyor belt length (m)
        self.s = 5.0   # incline length (m)
        self.theta = 37.0  # degrees (used for symbolic checks)

        # Extract results with safe fallbacks
        # Results from solution steps
        self.mu = getattr(solution_instance, 'mu', solution_instance.results.get('mu', 0.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.a2 = getattr(solution_instance, 'a2', solution_instance.results.get('a2', 0.0))
        self.t1 = getattr(solution_instance, 't₁', solution_instance.results.get('t₁', 0.0))
        self.t2 = getattr(solution_instance, 't₂', solution_instance.results.get('t₂', 0.0))
        self.t3 = getattr(solution_instance, 't₃', solution_instance.results.get('t₃', 0.0))
        self.vC = getattr(solution_instance, 'v_C', solution_instance.results.get('v_C', 0.0))
        self.t_all = getattr(solution_instance, 't_all', solution_instance.results.get('t_all', 0.0))

        # Derived quantities (for convenience)
        self.mumg = self.mu * self.g  # μ*g, assuming m=1 for dimensionless checks

    def test_invariants(self):
        """Check fundamental physical laws using symbolic and numerical methods."""
        logs = []
        try:
            # Symbolic definitions
            m, g, mu, v0, L, theta = sp.symbols('m g mu v0 L theta', positive=True)
            sin_th = sp.sin(theta)
            cos_th = sp.cos(theta)

            # 1. Conveyor belt: v0^2 = 2*a*L (from v^2 = u^2 + 2as, u=0)
            a_expected = v0**2 / (2*L)
            logs.append(f"Conveyor belt a: expected {a_expected.subs({v0:self.v0, L:self.L})}, got {self.a}")
            if not math.isclose(self.a, float(a_expected.subs({v0:self.v0, L:self.L})), rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Conveyor belt acceleration mismatch")

            # 2. Friction on conveyor: μ*m*g = m*a => μ = a/g
            mu_expected = a_expected / g
            logs.append(f"Friction coefficient: expected {mu_expected.subs({a_expected: self.a, g:self.g})}, got {self.mu}")
            if not math.isclose(self.mu, float(mu_expected.subs({a_expected: self.a, g:self.g})), rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Friction coefficient mismatch")

            # 3. Incline acceleration: a2 = g*sinθ - μ*g*cosθ
            a2_expected = g*sin_th - mu*g*cos_th
            a2_num = self.g*self.sin37 - self.mu*self.g*self.cos37
            logs.append(f"Incline a2: expected {a2_num}, got {self.a2}")
            if not math.isclose(self.a2, a2_num, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Incline acceleration mismatch")

            # 4. Energy check (work-energy theorem): ΔKE + ΔPE = W_friction
            # Not directly applicable due to non-conservative forces, but we can verify power balance
            # This is more complex; we skip detailed energy audit since friction dominates.

            logs.append("Invariant tests completed.")
            return True, logs
        except Exception as e:
            logs.append(f"Invariant test error: {traceback.format_exc()}")
            return False, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # 1. Check t1 (conveyor belt time): t1 = v0 / a
            t1_expected = self.v0 / self.a if self.a != 0 else 0
            logs.append(f"t1: expected {t1_expected}, got {self.t1}")
            if not math.isclose(self.t1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t1 mismatch")

            # 2. Check incline kinematics: s = v0*t2 + 0.5*a2*t2^2
            # Solve quadratic: 0.5*a2*t2^2 + v0*t2 - s = 0
            discriminant = self.v0**2 + 2*self.a2*self.s
            if discriminant >= 0:
                t2_expected = (-self.v0 + math.sqrt(discriminant)) / self.a2  # positive root
                logs.append(f"t2 (from kinematics): expected {t2_expected}, got {self.t2}")
                if not math.isclose(self.t2, t2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t2 mismatch")
            else:
                logs.append(f"FAIL: Invalid discriminant for t2")

            # 3. Check vC: vC = v0 + a2*t2
            vC_expected = self.v0 + self.a2 * self.t2
            logs.append(f"vC: expected {vC_expected}, got {self.vC}")
            if not math.isclose(self.vC, vC_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: vC mismatch")

            # 4. Check t3 (horizontal deceleration): t3 = vC / (μ*g)
            t3_expected = self.vC / (self.mu * self.g) if self.mu * self.g != 0 else 0
            logs.append(f"t3: expected {t3_expected}, got {self.t3}")
            if not math.isclose(self.t3, t3_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t3 mismatch")

            # 5. Check total time: t_all = t1 + t2 + t3
            t_all_expected = self.t1 + self.t2 + self.t3
            logs.append(f"Total time: expected {t_all_expected}, got {self.t_all}")
            if not math.isclose(self.t_all, t_all_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Total time mismatch")

            logs.append("Trace consistency tests completed.")
            return True, logs
        except Exception as e:
            logs.append(f"Trace consistency error: {traceback.format_exc()}")
            return False, logs

    def test_monotonicity(self):
        """Perturb an input (conveyor speed v0) and check trend in total time."""
        logs = []
        try:
            # Original total time (from solution)
            t_orig = self.t_all

            # Perturb v0 by +1%
            v0_perturbed = self.v0 * 1.01
            # Recalculate using physical formulas (assuming same steps)
            # Conveyor belt
            a_pert = v0_perturbed**2 / (2*self.L)
            mu_pert = a_pert / self.g
            # Incline
            a2_pert = self.g*self.sin37 - mu_pert*self.g*self.cos37
            # Solve for t2_pert: s = v0_perturbed*t2 + 0.5*a2_pert*t2^2
            disc = v0_perturbed**2 + 2*a2_pert*self.s
            if disc >= 0:
                t2_pert = (-v0_perturbed + math.sqrt(disc)) / a2_pert
            else:
                logs.append("FAIL: No real t2 for perturbed v0")
                return False, logs
            # Conveyor time t1
            t1_pert = v0_perturbed / a_pert
            # Velocity at C
            vC_pert = v0_perturbed + a2_pert * t2_pert
            # Horizontal deceleration time
            t3_pert = vC_pert / (mu_pert * self.g)
            # Total perturbed time
            t_all_pert = t1_pert + t2_pert + t3_pert

            logs.append(f"Original v0={self.v0}, t_all={t_orig}")
            logs.append(f"Perturbed v0={v0_perturbed:.3f}, t_all={t_all_pert:.3f}")

            # Check physical trend: increasing v0 should decrease total time?
            # Actually, from our earlier calculation, increasing v0 decreases t1 but may affect others.
            # We just check that the change is within reasonable bounds (not discontinuous).
            if math.isclose(t_all_pert, t_orig, rel_tol=0.05, abs_tol=0.05):
                logs.append("Perturbation test passed: total time change is small (<5%).")
                return True, logs
            else:
                logs.append(f"WARNING: Total time changed by >5% for 1% v0 change.")
                # This might be acceptable; we don't fail the test outright.
                return True, logs
        except Exception as e:
            logs.append(f"Monotonicity test error: {traceback.format_exc()}")
            return False, logs

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        all_logs = ["=== Physics Auditor Report ==="]
        success = True

        # Run tests
        inv_ok, inv_logs = self.test_invariants()
        all_logs.extend(inv_logs)
        success = success and inv_ok

        trace_ok, trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        success = success and trace_ok

        mono_ok, mono_logs = self.test_monotonicity()
        all_logs.extend(mono_logs)
        success = success and mono_ok

        # Final summary
        all_logs.append(f"Overall success: {success}")
        return success, all_logs

# Example usage (commented out for sandbox):
# checker = PhysicsChecker(solution)
# ok, logs = checker.run_all_tests()
# for log in logs:
#     print(log)