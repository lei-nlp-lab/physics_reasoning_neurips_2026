import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Use getattr to avoid AttributeError in sandbox environments.
        """
        # Input parameters from the problem
        self.m = getattr(solution_instance, 'm', 5.0)  # mass in kg
        self.h = getattr(solution_instance, 'h', 1.80) # free-fall height in m
        self.d = getattr(solution_instance, 'd', 0.03) # compression distance in m
        self.g = getattr(solution_instance, 'g', 10.0) # gravity in m/s²

        # Results calculated by the solution
        # Access via results dictionary or direct attribute, default to 0.0 if missing
        self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 0.0))

        # Store for monotonicity test backup
        self._original_inputs = {'m': self.m, 'h': self.h, 'd': self.d, 'g': self.g}
        self._original_results = {'v': self.v, 't': self.t, 'F': self.F}

        # Tolerance for numerical comparisons (robust to SFT noise)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def test_invariants(self):
        """
        Verify physics invariants:
        1. Energy conservation during free fall: m*g*h = 0.5*m*v²
        2. Kinematics during compression: v = a*t and d = 0.5*v*t (uniform deceleration)
        3. Impulse-momentum: F*t = m*v (since gravity neglected during compression)
        Returns list of error messages, empty if all pass.
        """
        logs = []

        # 1. Energy conservation: gravitational potential energy -> kinetic energy
        potential_energy = self.m * self.g * self.h
        kinetic_energy = 0.5 * self.m * self.v**2
        if not math.isclose(potential_energy, kinetic_energy, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Energy conservation violated: PE={potential_energy:.3f} J, KE={kinetic_energy:.3f} J")

        # 2. Kinematics during compression: d = (v_initial + v_final)/2 * t, with v_final=0
        expected_t = 2 * self.d / self.v if abs(self.v) > 1e-12 else 0.0
        if not math.isclose(self.t, expected_t, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Compression time mismatch: expected t={expected_t:.4f} s, got t={self.t:.4f} s")

        # Also check deceleration: a = v / t, and d = 0.5 * a * t²
        if abs(self.t) > 1e-12:
            a = self.v / self.t
            d_from_a = 0.5 * a * self.t**2
            if not math.isclose(self.d, d_from_a, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Compression distance mismatch: expected d={d_from_a:.5f} m, got d={self.d:.5f} m")

        # 3. Impulse-momentum theorem: F*t = m*v (net force is F since gravity neglected)
        impulse = self.F * self.t
        momentum_change = self.m * self.v
        if not math.isclose(impulse, momentum_change, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Impulse-momentum mismatch: F*t={impulse:.3f} N·s, m*v={momentum_change:.3f} N·s")

        # 4. SymPy symbolic check for energy conservation (alternative formulation)
        m, g, h, v = sp.symbols('m g h v', positive=True)
        energy_eq = sp.Eq(m*g*h, sp.Rational(1,2)*m*v**2)
        # Substitute known values
        energy_sub = energy_eq.subs({m: self.m, g: self.g, h: self.h, v: self.v})
        if not energy_sub:
            logs.append("Symbolic energy conservation check failed.")

        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer and input parameters.
        Steps from the problem's step-by-step analysis.
        """
        logs = []

        # Step 1: Free fall velocity v = sqrt(2*g*h)
        expected_v = math.sqrt(2 * self.g * self.h)
        if not math.isclose(self.v, expected_v, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Velocity mismatch: expected v={expected_v:.3f} m/s, got v={self.v:.3f} m/s")

        # Step 2: Compression time t = 2*d / v
        if abs(self.v) > 1e-12:
            expected_t = 2 * self.d / self.v
            if not math.isclose(self.t, expected_t, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Time mismatch: expected t={expected_t:.4f} s, got t={self.t:.4f} s")

        # Step 3: Average force F = m*v / t
        if abs(self.t) > 1e-12:
            expected_F = self.m * self.v / self.t
            if not math.isclose(self.F, expected_F, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Force mismatch: expected F={expected_F:.1f} N, got F={self.F:.1f} N")

        # Consistency between all three: F should also equal m*v²/(2*d) via kinematics
        expected_F_alt = self.m * self.v**2 / (2 * self.d)
        if not math.isclose(self.F, expected_F_alt, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Force from kinematics mismatch: expected F={expected_F_alt:.1f} N, got F={self.F:.1f} N")

        return logs

    def test_monotonicity(self):
        """
        Perturb an input (height h) and verify that force F changes in the expected direction.
        Since F ∝ v ∝ √h, increasing h should increase F.
        After check, restore original values.
        """
        logs = []

        # Perturbation factor (small increase)
        perturb_factor = 1.1  # 10% increase
        original_h = self.h
        perturbed_h = original_h * perturb_factor

        # Recalculate velocity and force using formulas (not calling solution methods)
        perturbed_v = math.sqrt(2 * self.g * perturbed_h)
        perturbed_t = 2 * self.d / perturbed_v if abs(perturbed_v) > 1e-12 else 0.0
        perturbed_F = self.m * perturbed_v / perturbed_t if abs(perturbed_t) > 1e-12 else 0.0

        # Check monotonicity: force should increase with height
        if not (perturbed_F > self.F or math.isclose(perturbed_F, self.F, rel_tol=self.rel_tol, abs_tol=self.abs_tol)):
            logs.append(f"Monotonicity violation: increasing h from {original_h} to {perturbed_h} should increase F, but got F={self.F} -> {perturbed_F}")

        # Restore original values (important!)
        self.h = original_h
        # Note: self.v, self.t, self.F are unchanged as we only read them; we didn't modify.

        return logs

    def run_all_tests(self):
        """
        Execute all tests and return a tuple (success_bool, list_of_logs).
        Success means no errors in any test.
        """
        all_logs = []

        # Run each test and collect logs
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                logs = test_method()
                if logs:
                    all_logs.append(f"{test_name} errors:")
                    all_logs.extend([f"  - {msg}" for msg in logs])
            except Exception as e:
                all_logs.append(f"{test_name} crashed: {str(e)}")

        success = len(all_logs) == 0
        return success, all_logs