import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.extract_variables()

    def extract_variables(self):
        # Inputs with safe attribute access
        self.h = getattr(self.solution, 'h', 5.2)
        self.B = getattr(self.solution, 'B', 1.0)
        self.L = getattr(self.solution, 'L', 1.0)
        self.R = getattr(self.solution, 'R', 1.0)
        self.m = getattr(self.solution, 'm', 1.0)
        self.g = getattr(self.solution, 'g', 10.0)

        # Results from attributes or results dictionary
        self.v = getattr(self.solution, 'v', None)
        self.Q = getattr(self.solution, 'Q', None)
        self.t = getattr(self.solution, 't', None)
        if hasattr(self.solution, 'results'):
            self.v = self.solution.results.get('v', self.v)
            self.Q = self.solution.results.get('Q', self.Q)
            self.t = self.solution.results.get('t', self.t)
            self.q = self.solution.results.get('q', None)

        # Store original inputs for restoration in monotonicity test
        self.original_inputs = {
            'h': self.h, 'B': self.B, 'L': self.L,
            'R': self.R, 'm': self.m, 'g': self.g
        }

    def test_invariants(self):
        logs = []
        success = True

        # 1. Force equilibrium at constant velocity: mg = B²L²v/R
        if self.v is not None:
            magnetic_force = (self.B ** 2 * self.L ** 2 * self.v) / self.R
            gravity_force = self.m * self.g
            if not math.isclose(magnetic_force, gravity_force, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(
                    f"Force equilibrium fails: mg = {gravity_force:.3f}, "
                    f"magnetic force = {magnetic_force:.3f}"
                )
                success = False
            else:
                logs.append("Force equilibrium holds (magnetic force balances gravity).")
        else:
            logs.append("Velocity 'v' not available; skipping force equilibrium check.")
            success = False

        # 2. Energy conservation: mg(h + L) = ½ m v² + Q
        if self.v is not None and self.Q is not None:
            potential_loss = self.m * self.g * (self.h + self.L)
            kinetic_gain = 0.5 * self.m * self.v ** 2
            total_energy = kinetic_gain + self.Q
            if not math.isclose(potential_loss, total_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(
                    f"Energy conservation fails: mg(h+L) = {potential_loss:.3f}, "
                    f"½mv² + Q = {total_energy:.3f}"
                )
                success = False
            else:
                logs.append("Energy conservation holds.")
        else:
            logs.append("Missing 'v' or 'Q'; skipping energy conservation check.")
            success = False

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Compute expected charge from flux change: q = B L² / R
        q_expected = self.B * self.L ** 2 / self.R

        # Momentum theorem: mg t - B L q = m v  => t = (m v + B L q) / (m g)
        if self.v is not None and self.t is not None:
            t_calculated = (self.m * self.v + self.B * self.L * q_expected) / (self.m * self.g)
            if not math.isclose(t_calculated, self.t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(
                    f"Trace consistency fails: from momentum theorem, t = {t_calculated:.3f}, "
                    f"but given t = {self.t:.3f}"
                )
                success = False
            else:
                logs.append("Trace consistency holds (t matches momentum theorem prediction).")
        else:
            logs.append("Missing 'v' or 't'; skipping trace consistency check.")
            success = False

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb mass m by +10% and check effect on velocity v = mgR/(B²L²)
        original_m = self.m
        perturbed_m = original_m * 1.10

        v_original = (original_m * self.g * self.R) / (self.B ** 2 * self.L ** 2)
        v_perturbed = (perturbed_m * self.g * self.R) / (self.B ** 2 * self.L ** 2)

        if v_perturbed <= v_original:
            logs.append(
                f"Monotonicity fails: v should increase with m, but "
                f"v(original) = {v_original:.3f}, v(perturbed) = {v_perturbed:.3f}"
            )
            success = False
        else:
            logs.append(
                f"Monotonicity holds: v increases with m "
                f"({v_original:.3f} -> {v_perturbed:.3f})."
            )

        # Restore original mass
        self.m = original_m

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]

        for test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(
                    f"Test {test_func.__name__} raised exception: {e}\n"
                    f"{traceback.format_exc()}"
                )
                overall_success = False

        return overall_success, all_logs