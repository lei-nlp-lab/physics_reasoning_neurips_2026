import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Inputs from problem statement (with fallbacks)
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 60.0))
        self.h1 = getattr(solution_instance, 'h1', getattr(solution_instance, 'inputs', {}).get('h1', 3.2))
        self.h2 = getattr(solution_instance, 'h2', getattr(solution_instance, 'inputs', {}).get('h2', 5.0))
        self.t_contact = getattr(solution_instance, 't', getattr(solution_instance, 'inputs', {}).get('t', 1.2))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))

        # Results (with fallbacks to results dictionary)
        self.v1 = getattr(solution_instance, 'v_1', getattr(solution_instance, 'results', {}).get('v_1', None))
        self.v2 = getattr(solution_instance, 'v_2', getattr(solution_instance, 'results', {}).get('v_2', None))
        self.F_net = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', None))

        # Derived quantities (for internal checks)
        self.v1_expected = math.sqrt(2 * self.g * self.h1)  # downward, magnitude
        self.v2_expected = math.sqrt(2 * self.g * self.h2)  # upward, magnitude

    def test_invariants(self):
        """Verify energy conservation and impulse-momentum theorem."""
        logs = []
        success = True

        # 1. Energy conservation during free fall (before contact)
        try:
            # Initial mechanical energy at height h1
            E_initial = self.m * self.g * self.h1
            # Kinetic energy just before contact (using reported v1)
            if self.v1 is not None:
                K_before = 0.5 * self.m * (self.v1 ** 2)
                if not math.isclose(E_initial, K_before, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Energy conservation before contact violated: E_initial={E_initial:.1f}J, K_before={K_before:.1f}J")
                    success = False
                else:
                    logs.append("✓ Energy conservation before contact holds.")
        except Exception as e:
            logs.append(f"Energy conservation check failed: {e}")
            success = False

        # 2. Energy conservation during rise (after contact)
        try:
            # Kinetic energy just after contact (using reported v2)
            if self.v2 is not None:
                K_after = 0.5 * self.m * (self.v2 ** 2)
                # Potential energy at maximum height h2
                E_final = self.m * self.g * self.h2
                if not math.isclose(K_after, E_final, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Energy conservation after contact violated: K_after={K_after:.1f}J, E_final={E_final:.1f}J")
                    success = False
                else:
                    logs.append("✓ Energy conservation after contact holds.")
        except Exception as e:
            logs.append(f"Energy after-contact check failed: {e}")
            success = False

        # 3. Impulse-momentum theorem
        try:
            if self.v1 is not None and self.v2 is not None and self.F_net is not None:
                # Define symbols for symbolic check
                m, g, t, v1, v2, F = sp.symbols('m g t v1 v2 F')
                # Impulse-momentum: F_net * t = m*v2 - m*(-v1) = m*(v2 + v1)
                # Where F_net = F (force by net) - m*g (weight)
                eq = sp.Eq((F - m * g) * t, m * (v2 + v1))
                # Substitute values
                eq_sub = eq.subs({
                    m: self.m,
                    g: self.g,
                    t: self.t_contact,
                    v1: self.v1,
                    v2: self.v2,
                    F: self.F_net
                })
                # Check if equation holds numerically
                lhs = float((self.F_net - self.m * self.g) * self.t_contact)
                rhs = float(self.m * (self.v2 + self.v1))
                if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Impulse-momentum violated: (F-mg)*t={lhs:.1f}Ns, m(v2+v1)={rhs:.1f}Ns")
                    success = False
                else:
                    logs.append("✓ Impulse-momentum theorem holds.")
        except Exception as e:
            logs.append(f"Impulse-momentum check failed: {e}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results (v1, v2) with expected values and final force."""
        logs = []
        success = True

        # Check v1 against kinematic expectation
        if self.v1 is not None:
            if not math.isclose(self.v1, self.v1_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"v1 inconsistent: reported {self.v1:.1f}m/s, expected {self.v1_expected:.1f}m/s")
                success = False
            else:
                logs.append("✓ v1 matches kinematic expectation.")
        else:
            logs.append("v1 missing; cannot check.")
            success = False

        # Check v2 against kinematic expectation
        if self.v2 is not None:
            if not math.isclose(self.v2, self.v2_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"v2 inconsistent: reported {self.v2:.1f}m/s, expected {self.v2_expected:.1f}m/s")
                success = False
            else:
                logs.append("✓ v2 matches kinematic expectation.")
        else:
            logs.append("v2 missing; cannot check.")
            success = False

        # Recompute F from v1, v2 and compare
        if self.v1 is not None and self.v2 is not None and self.F_net is not None:
            F_recomputed = self.m * (self.v2 + self.v1) / self.t_contact + self.m * self.g
            if not math.isclose(self.F_net, F_recomputed, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"F inconsistent: reported {self.F_net:.0f}N, recomputed {F_recomputed:.0f}N")
                success = False
            else:
                logs.append("✓ F matches recomputed value from v1 and v2.")
        else:
            logs.append("Missing v1, v2, or F; cannot recompute F.")
            success = False

        return success, logs

    def test_monotonicity(self):
        """Perturb mass and check force response; restore original mass."""
        logs = []
        success = True
        original_m = self.m

        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            # Compute expected velocities (independent of mass)
            v1_expected = math.sqrt(2 * self.g * self.h1)
            v2_expected = math.sqrt(2 * self.g * self.h2)
            # Compute expected force with perturbed mass
            F_perturbed = perturbed_m * (v2_expected + v1_expected) / self.t_contact + perturbed_m * self.g
            # Compute original expected force (for comparison)
            F_original_expected = original_m * (v2_expected + v1_expected) / self.t_contact + original_m * self.g

            # Check if force increases proportionally with mass (linear relationship)
            expected_ratio = 1.1  # because mass increased by 10%
            actual_ratio = F_perturbed / F_original_expected
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.01):
                logs.append(f"Monotonicity violated: mass +10% → force ratio {actual_ratio:.3f}, expected {expected_ratio}")
                success = False
            else:
                logs.append("✓ Force scales linearly with mass (monotonicity holds).")

        except Exception as e:
            logs.append(f"Monotonicity test failed: {e}")
            success = False
        finally:
            # Restore original mass (important!)
            self.m = original_m

        return success, logs

    def run_all_tests(self):
        """Execute all tests and return aggregated results."""
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
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"!!! {test_name} crashed: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs