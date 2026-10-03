import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: first try direct attribute, then results dict, then fallback to problem constants
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract constants (in SI units)
        self.d = getattr(solution_instance, 'd', self.results.get('d', 0.08))           # plate separation: 8 cm = 0.08 m
        self.L = getattr(solution_instance, 'L', self.results.get('L', 0.25))           # plate length: 25 cm = 0.25 m
        self.v0 = getattr(solution_instance, 'v0', self.results.get('v0', 0.5))         # initial velocity: 0.5 m/s
        self.Delta_d = getattr(solution_instance, 'Delta_d', self.results.get('Delta_d', 4/300))  # 4/3 cm = 4/300 m ≈ 0.01333 m
        self.g = getattr(solution_instance, 'g', self.results.get('g', 10.0))           # gravity: 10 m/s²

        # Extract results with safe fallbacks
        self.E_ratio = getattr(solution_instance, 'E_ratio', self.results.get("E'/E", None))
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.t1 = getattr(solution_instance, 't1', self.results.get('t1', None))
        self.t2 = getattr(solution_instance, 't2', self.results.get('t2', None))
        self.t = getattr(solution_instance, 't', self.results.get('t', None))
        self.E = getattr(solution_instance, 'E', self.results.get('E', None))
        self.E_prime = getattr(solution_instance, 'E_prime', self.results.get("E'", None))

        # For symbolic computations
        self.symbols = self._define_symbols()
        self.logs = []

    def _define_symbols(self):
        """Define SymPy symbols for physical quantities."""
        d, L, v0, Delta_d, g, U = sp.symbols('d L v0 Delta_d g U', positive=True)
        E = U / d
        E_prime = U / (d - Delta_d)
        E_ratio = E_prime / E
        a = E_ratio * g - g
        t2 = sp.sqrt(d / a)  # from d/2 = (1/2)*a*t2^2
        t = L / v0
        t1 = t - t2
        return {
            'd': d, 'L': L, 'v0': v0, 'Delta_d': Delta_d, 'g': g, 'U': U,
            'E': E, 'E_prime': E_prime, 'E_ratio': E_ratio,
            'a': a, 't2': t2, 't': t, 't1': t1
        }

    def _log(self, msg, success):
        """Append a test result to logs."""
        self.logs.append((success, msg))

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physics laws: force balance and kinematics."""
        success = True

        # 1. Check that initial motion is uniform (net force zero) => qE = mg
        # Since we don't have q or m, we check that the acceleration after plate move is consistent with Newton's second law.
        # From steps: a = (E'/E)*g - g. So check that a matches this formula.
        if self.E_ratio is not None and self.a is not None:
            expected_a = (self.E_ratio - 1) * self.g
            if not self._isclose(self.a, expected_a):
                self._log(f"Newton's Second Law fail: a={self.a}, expected (E'/E - 1)*g = {expected_a}", False)
                success = False
            else:
                self._log("Newton's Second Law satisfied for acceleration after plate move.", True)
        else:
            self._log("Skipping Newton's Second Law check due to missing E_ratio or a.", None)

        # 2. Check kinematic consistency: t1 + t2 = L/v0
        if all(x is not None for x in [self.t1, self.t2, self.t]):
            total_t = self.t1 + self.t2
            if not self._isclose(total_t, self.t):
                self._log(f"Time consistency fail: t1+t2={total_t}, t=L/v0={self.t}", False)
                success = False
            else:
                self._log("Kinematic time consistency satisfied.", True)

        # 3. Check that t2 is consistent with vertical motion: d/2 = (1/2)*a*t2^2
        if self.t2 is not None and self.a is not None:
            expected_t2 = math.sqrt(self.d / self.a)  # from d = a*t2^2
            if not self._isclose(self.t2, expected_t2):
                self._log(f"Vertical motion kinematics fail: t2={self.t2}, expected sqrt(d/a)={expected_t2}", False)
                success = False
            else:
                self._log("Vertical motion kinematics satisfied.", True)

        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True

        # 1. Check E'/E = d/(d - Δd)
        expected_ratio = self.d / (self.d - self.Delta_d)
        if self.E_ratio is not None:
            if not self._isclose(self.E_ratio, expected_ratio):
                self._log(f"E'/E ratio mismatch: {self.E_ratio} vs expected {expected_ratio}", False)
                success = False
            else:
                self._log("E'/E ratio correct.", True)

        # 2. Check t = L/v0
        expected_t = self.L / self.v0
        if self.t is not None:
            if not self._isclose(self.t, expected_t):
                self._log(f"Uniform motion time mismatch: t={self.t}, expected L/v0={expected_t}", False)
                success = False
            else:
                self._log("Uniform motion time correct.", True)

        # 3. Check t1 = t - t2
        if all(x is not None for x in [self.t1, self.t, self.t2]):
            expected_t1 = self.t - self.t2
            if not self._isclose(self.t1, expected_t1):
                self._log(f"t1 mismatch: {self.t1} vs expected t - t2 = {expected_t1}", False)
                success = False
            else:
                self._log("t1 correctly derived from t and t2.", True)

        return success

    def test_monotonicity(self):
        """Perturb Δd and check that results change in the expected direction."""
        success = True

        # Perturb Δd by +10% (but ensure d - Δd remains positive)
        perturbed_Delta_d = self.Delta_d * 1.1
        if perturbed_Delta_d >= self.d:
            perturbed_Delta_d = self.Delta_d * 0.9  # try decreasing instead

        # Recompute using formulas from steps (numerical fallback)
        new_ratio = self.d / (self.d - perturbed_Delta_d)
        new_a = (new_ratio - 1) * self.g
        new_t2 = math.sqrt(self.d / new_a) if new_a > 0 else None
        new_t = self.L / self.v0
        new_t1 = new_t - new_t2 if new_t2 is not None else None

        # Expected trends:
        # If Δd increases, E'/E increases (since denominator decreases).
        # Then a increases (because a = (E'/E - 1)*g).
        # Then t2 decreases (since t2 = sqrt(d/a)).
        # Then t1 increases (since t1 = t - t2, and t is constant).
        if self.E_ratio is not None:
            if perturbed_Delta_d > self.Delta_d and new_ratio <= self.E_ratio:
                self._log(f"Monotonicity fail: Δd increased but E'/E did not increase ({self.E_ratio} -> {new_ratio})", False)
                success = False
            elif perturbed_Delta_d < self.Delta_d and new_ratio >= self.E_ratio:
                self._log(f"Monotonicity fail: Δd decreased but E'/E did not decrease ({self.E_ratio} -> {new_ratio})", False)
                success = False
            else:
                self._log("E'/E monotonicity with Δd holds.", True)

        if self.a is not None and new_a is not None:
            if perturbed_Delta_d > self.Delta_d and new_a <= self.a:
                self._log(f"Monotonicity fail: Δd increased but a did not increase ({self.a} -> {new_a})", False)
                success = False
            elif perturbed_Delta_d < self.Delta_d and new_a >= self.a:
                self._log(f"Monotonicity fail: Δd decreased but a did not decrease ({self.a} -> {new_a})", False)
                success = False
            else:
                self._log("Acceleration monotonicity with Δd holds.", True)

        if self.t2 is not None and new_t2 is not None:
            if perturbed_Delta_d > self.Delta_d and new_t2 >= self.t2:
                self._log(f"Monotonicity fail: Δd increased but t2 did not decrease ({self.t2} -> {new_t2})", False)
                success = False
            elif perturbed_Delta_d < self.Delta_d and new_t2 <= self.t2:
                self._log(f"Monotonicity fail: Δd decreased but t2 did not increase ({self.t2} -> {new_t2})", False)
                success = False
            else:
                self._log("t2 monotonicity with Δd holds.", True)

        if self.t1 is not None and new_t1 is not None:
            if perturbed_Delta_d > self.Delta_d and new_t1 <= self.t1:
                self._log(f"Monotonicity fail: Δd increased but t1 did not increase ({self.t1} -> {new_t1})", False)
                success = False
            elif perturbed_Delta_d < self.Delta_d and new_t1 >= self.t1:
                self._log(f"Monotonicity fail: Δd decreased but t1 did not decrease ({self.t1} -> {new_t1})", False)
                success = False
            else:
                self._log("t1 monotonicity with Δd holds.", True)

        return success

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{name}' crashed: {traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs