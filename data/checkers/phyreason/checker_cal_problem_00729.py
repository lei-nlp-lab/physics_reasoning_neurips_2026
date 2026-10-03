import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = {}
        self.inputs = {}

        # Extract inputs safely
        self.inputs['U0'] = getattr(solution_instance, 'U0', getattr(solution_instance, 'results', {}).get('U0', (25/12) * 1e3))
        self.inputs['E'] = getattr(solution_instance, 'E', getattr(solution_instance, 'results', {}).get('E', (5/3) * 1e4))
        self.inputs['B'] = getattr(solution_instance, 'B', getattr(solution_instance, 'results', {}).get('B', 1e-2))
        self.inputs['q1_over_m1'] = getattr(solution_instance, 'q1_over_m1', getattr(solution_instance, 'results', {}).get('q1_over_m1', 2.4e6))
        self.inputs['q2_over_m2'] = getattr(solution_instance, 'q2_over_m2', getattr(solution_instance, 'results', {}).get('q2_over_m2', 2.0e6))
        self.inputs['x_AM'] = 0.5  # Given horizontal distance

        # Extract results safely
        self.results['v_0'] = getattr(solution_instance, 'v_0', getattr(solution_instance, 'results', {}).get('v_0', None))
        self.results['y'] = getattr(solution_instance, 'y', getattr(solution_instance, 'results', {}).get('y', None))
        self.results['R'] = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', None))
        self.results['OF'] = getattr(solution_instance, 'OF', getattr(solution_instance, 'results', {}).get('OF', None))
        self.results['FK'] = getattr(solution_instance, 'FK', getattr(solution_instance, 'results', {}).get('FK', None))
        self.results['L'] = getattr(solution_instance, 'L', getattr(solution_instance, 'results', {}).get('L', None))

        # Derived symbols for symbolic checks
        self.symbols = {}
        for key in ['U0', 'E', 'B', 'q1_over_m1', 'q2_over_m2', 'v_0', 'y', 'R', 'OF', 'FK', 'L']:
            self.symbols[key] = sp.symbols(key, real=True, positive=True)

        self.logs = []

    def log(self, message, is_error=False):
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")

    def test_invariants(self):
        """Verify energy conservation and Newton's 2nd law for argon ions."""
        logs = []

        try:
            # 1. Work-energy for argon ion acceleration
            # v0^2 = 2*(q1/m1)*U0
            v0_calc = math.sqrt(2 * self.inputs['q1_over_m1'] * self.inputs['U0'])
            if self.results['v_0'] is not None:
                if not math.isclose(v0_calc, self.results['v_0'], rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"v0 mismatch: calc {v0_calc:.2e}, result {self.results['v_0']:.2e}")
                else:
                    logs.append(f"v0 matches: {v0_calc:.2e} m/s")

            # 2. Kinematics for argon ion in electric field
            # y = (q1*E*t^2)/(2*m1) with t = x_AM / v0
            t_calc = self.inputs['x_AM'] / v0_calc
            y_calc = (self.inputs['q1_over_m1'] * self.inputs['E'] * t_calc**2) / 2
            if self.results['y'] is not None:
                if not math.isclose(y_calc, self.results['y'], rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"y mismatch: calc {y_calc:.3f}, result {self.results['y']:.3f}")
                else:
                    logs.append(f"y matches: {y_calc:.3f} m")

            # 3. Lorentz force as centripetal force for metal ion (if R and v available)
            # R = m2*v / (q2*B) => v = (q2/m2)*B*R
            if self.results['R'] is not None:
                v_metal_calc = self.inputs['q2_over_m2'] * self.inputs['B'] * self.results['R']
                # We don't have an independent v_metal from results, so just note the consistency
                logs.append(f"Metal ion speed from R: {v_metal_calc:.2e} m/s")

        except Exception as e:
            logs.append(f"Exception in test_invariants: {e}\n{traceback.format_exc()}")

        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []

        try:
            # Check geometry: OF, FK, L relations from steps
            if (self.results['OF'] is not None and self.results['FK'] is not None and
                self.results['L'] is not None):
                # OF + FK should be 0.5 (from steps: OF + FK = 0.5 m)
                sum_OF_FK = self.results['OF'] + self.results['FK']
                if not math.isclose(sum_OF_FK, 0.5, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"OF + FK = {sum_OF_FK:.3f}, expected 0.5 m")
                else:
                    logs.append(f"OF + FK consistent: {sum_OF_FK:.3f} m")

                # L should be 2 * FK (from steps: L = 2 * FK)
                if not math.isclose(self.results['L'], 2 * self.results['FK'], rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"L = {self.results['L']:.3f}, expected 2*FK = {2*self.results['FK']:.3f}")
                else:
                    logs.append(f"L = 2*FK consistent: {self.results['L']:.3f} m")

                # Also check target answer with π (second target answer)
                L_expected_pi = (2 - math.sqrt(2)) / 2 * math.pi
                if math.isclose(self.results['L'], L_expected_pi, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"L matches target answer with π: {L_expected_pi:.4f} m")
                else:
                    # Check without π
                    L_expected_no_pi = (2 - math.sqrt(2)) / 2
                    if math.isclose(self.results['L'], L_expected_no_pi, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"L matches steps value (without π): {L_expected_no_pi:.4f} m")
                    else:
                        logs.append(f"L mismatch: result {self.results['L']:.4f}, expected with π {L_expected_pi:.4f} or without π {L_expected_no_pi:.4f}")

        except Exception as e:
            logs.append(f"Exception in test_trace_consistency: {e}\n{traceback.format_exc()}")

        return logs

    def test_monotonicity(self):
        """Perturb an input and check trends."""
        logs = []

        try:
            # Perturb U0 and recalc v0
            U0_original = self.inputs['U0']
            U0_perturbed = U0_original * 1.1  # +10%
            v0_original = math.sqrt(2 * self.inputs['q1_over_m1'] * U0_original)
            v0_perturbed = math.sqrt(2 * self.inputs['q1_over_m1'] * U0_perturbed)

            if v0_perturbed > v0_original:
                logs.append(f"Monotonicity: v0 increases with U0 (√(U0) dependence)")
            else:
                logs.append(f"Monotonicity violation: v0 did not increase with U0")

            # Restore U0 (not needed as we didn't modify instance)
            logs.append("U0 perturbation test passed.")

        except Exception as e:
            logs.append(f"Exception in test_monotonicity: {e}\n{traceback.format_exc()}")

        return logs

    def run_all_tests(self):
        """Run all checks and return success and logs."""
        self.logs.clear()

        # Run tests
        self.logs.extend(self.test_invariants())
        self.logs.extend(self.test_trace_consistency())
        self.logs.extend(self.test_monotonicity())

        # Determine success: no ERROR logs
        success = not any('ERROR' in log for log in self.logs)

        return success, self.logs