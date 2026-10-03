import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.I = getattr(solution_instance, 'I', None)
        self.d = getattr(solution_instance, 'd', None)
        self.m = getattr(solution_instance, 'm', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.x1 = results.get('x1', None)
        self.x2 = results.get('x2', None)
        self.pattern = results.get('pattern', None)
        self.k = results.get('k', None)
        self.omega = results.get('omega', None)
        self.T = results.get('T', None)

    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---- Interface checks ----
    def test_interface_has_x1(self):
        if self.x1 is None:
            return (False, "SKIP: x1 not provided")
        return (True, "x1 is present")

    def test_interface_has_x2(self):
        if self.x2 is None:
            return (False, "SKIP: x2 not provided")
        return (True, "x2 is present")

    def test_interface_has_pattern(self):
        if self.pattern is None:
            return (False, "SKIP: pattern not provided")
        if not isinstance(self.pattern, str):
            return (False, "pattern should be a string")
        return (True, "pattern is a string")

    def test_interface_has_k(self):
        if self.k is None:
            return (False, "SKIP: k not provided")
        return (True, "k is present")

    def test_interface_has_omega(self):
        if self.omega is None:
            return (False, "SKIP: omega not provided")
        return (True, "omega is present")

    def test_interface_has_T(self):
        if self.T is None:
            return (False, "SKIP: T not provided")
        return (True, "T is present")

    # ---- Invariant checks (theorem-based, use math.isclose) ----
    def test_invariant_x1_sign(self):
        if self.x1 is None:
            return (False, "SKIP: x1 not available")
        if self.x1 >= 0:
            return (False, f"x1 should be negative, got {self.x1}")
        return (True, "x1 is negative")

    def test_invariant_x2_sign(self):
        if self.x2 is None:
            return (False, "SKIP: x2 not available")
        if self.x2 <= 0:
            return (False, f"x2 should be positive, got {self.x2}")
        return (True, "x2 is positive")

    def test_invariant_x1_x2_symmetry(self):
        if self.x1 is None or self.x2 is None:
            return (False, "SKIP: x1 or x2 not available")
        if not math.isclose(self.x1, -self.x2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"x1 and x2 are not symmetric: x1={self.x1}, x2={self.x2}")
        return (True, "x1 ≈ -x2")

    def test_invariant_x_magnitude_given_d(self):
        if self.x1 is None or self.x2 is None or self.d is None:
            return (False, "SKIP: x1, x2 or d not available")
        expected = self.d / math.sqrt(3)
        if not math.isclose(abs(self.x1), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"|x1| = {abs(self.x1)} not close to d/√3 = {expected}")
        if not math.isclose(abs(self.x2), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"|x2| = {abs(self.x2)} not close to d/√3 = {expected}")
        return (True, "x1 and x2 magnitudes match d/√3")

    def test_invariant_force_constant_expression(self):
        if self.k is None or self.mu0 is None or self.I is None or self.d is None:
            return (False, "SKIP: k, mu0, I, d not all available")
        expected_k = self.mu0 * self.I**2 / (math.pi * self.d**2)
        if not math.isclose(self.k, expected_k, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"k={self.k} not close to expected {expected_k}")
        return (True, "k matches μ0 I^2/(π d^2)")

    def test_invariant_omega_relation(self):
        if self.omega is None or self.k is None or self.m is None:
            return (False, "SKIP: omega, k, m not all available")
        expected_omega = math.sqrt(self.k / self.m) if self.m != 0 else float('inf')
        if self.m == 0 or not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega={self.omega} not close to sqrt(k/m)={expected_omega}")
        return (True, "omega ≈ sqrt(k/m)")

    def test_invariant_T_omega_relation(self):
        if self.T is None or self.omega is None:
            return (False, "SKIP: T or omega not available")
        if self.omega == 0:
            return (False, "omega is zero, cannot compute period")
        expected_T = 2 * math.pi / self.omega
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T={self.T} not close to 2π/ω={expected_T}")
        return (True, "T ≈ 2π/ω")

    # ---- Modeling / trace checks ----
    def test_modeling_pattern_contains_keywords(self):
        if self.pattern is None:
            return (False, "SKIP: pattern not available")
        keywords = ['circle', 'zero', 'symmetric', 'wire']
        found = [kw for kw in keywords if kw.lower() in self.pattern.lower()]
        if len(found) == 0:
            return (False, f"pattern does not contain any of {keywords}")
        return (True, f"pattern contains keywords: {found}")

    def test_modeling_force_direction_negative(self):
        # Force per unit length on middle wire is -k x, so k should be positive
        if self.k is None:
            return (False, "SKIP: k not available")
        if self.k <= 0:
            return (False, f"k should be positive, got {self.k}")
        return (True, "k > 0, consistent with restoring force opposite to displacement")

    def test_modeling_period_expression(self):
        # Check that T is consistent with derived expression from given values
        if self.T is None or self.mu0 is None or self.I is None or self.d is None or self.m is None:
            return (False, "SKIP: T, mu0, I, d, m not all available")
        expected_T = 2 * math.pi * self.d / self.I * math.sqrt(math.pi * self.m / self.mu0)
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T={self.T} not close to expected {expected_T}")
        return (True, "T matches 2π d/I sqrt(π m/μ0)")

    # ---- Monotonicity checks (perturb one input, compare derived output) ----
    def test_monotonicity_x_magnitude_increases_with_d(self):
        # Zero positions: |x| = d/√3.  Increasing d should increase |x|.
        if self.d is None:
            return (False, "SKIP: d not available")
        baseline_d = self.d
        perturbed_d = baseline_d * 1.1
        baseline_mag = baseline_d / math.sqrt(3)
        perturbed_mag = perturbed_d / math.sqrt(3)
        if not (perturbed_mag > baseline_mag):
            return (False, f"Monotonicity fail: d increased by 10% but predicted |x| did not increase")
        return (True, "|x| increases with d (monotonicity holds)")

    def test_monotonicity_T_increases_with_d(self):
        # T = 2π d/I sqrt(π m/μ0).  Increasing d increases T.
        if self.d is None or self.I is None or self.mu0 is None or self.m is None:
            return (False, "SKIP: d, I, mu0, m not all available")
        baseline_T = 2 * math.pi * self.d / self.I * math.sqrt(math.pi * self.m / self.mu0)
        perturbed_d = self.d * 1.1
        perturbed_T = 2 * math.pi * perturbed_d / self.I * math.sqrt(math.pi * self.m / self.mu0)
        if not (perturbed_T > baseline_T):
            return (False, f"Monotonicity fail: d increased but predicted T did not increase")
        return (True, "T increases with d (monotonicity holds)")

    def test_monotonicity_T_decreases_with_I(self):
        # T = 2π d/I sqrt(π m/μ0).  Increasing I decreases T.
        if self.d is None or self.I is None or self.mu0 is None or self.m is None:
            return (False, "SKIP: d, I, mu0, m not all available")
        baseline_T = 2 * math.pi * self.d / self.I * math.sqrt(math.pi * self.m / self.mu0)
        perturbed_I = self.I * 1.1
        perturbed_T = 2 * math.pi * self.d / perturbed_I * math.sqrt(math.pi * self.m / self.mu0)
        if not (perturbed_T < baseline_T):
            return (False, f"Monotonicity fail: I increased but predicted T did not decrease")
        return (True, "T decreases with I (monotonicity holds)")

    # ---- Single weak answer sanity (no isclose, no re-derivation, only finiteness) ----
    def test_answer_sanity_finite(self):
        # Check that all available numerical results are finite and not NaN
        numerics = [self.x1, self.x2, self.k, self.omega, self.T]
        names = ['x1', 'x2', 'k', 'omega', 'T']
        for val, name in zip(numerics, names):
            if val is None:
                continue
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")
        return (True, "All available numeric results are finite")