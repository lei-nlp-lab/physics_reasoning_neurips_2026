import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for a charged particle in a 3D isotropic harmonic oscillator
    with uniform magnetic and electric fields. Checks frequencies, invariants, weak/strong limits.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.B = getattr(solution_instance, 'B', None)
        self.omega0 = getattr(solution_instance, 'omega0', None)
        self.E = getattr(solution_instance, 'E', None)  # electric field – not used for frequencies

        # Extract results from solution_instance.results dict (expected keys)
        results = getattr(solution_instance, 'results', {})
        self.omega_z = results.get('omega_z', None)
        self.omega_plus = results.get('omega_plus', None)
        self.omega_minus = results.get('omega_minus', None)
        self.omega_plus_weak = results.get('omega_plus_weak', None)
        self.omega_minus_weak = results.get('omega_minus_weak', None)
        self.omega_plus_strong = results.get('omega_plus_strong', None)
        self.omega_minus_strong = results.get('omega_minus_strong', None)

        # Derived quantities (cached only if givens are present)
        self._omega_c = None
        if self.m is not None and self.q is not None and self.B is not None and self.m != 0:
            self._omega_c = self.q * self.B / self.m

    # ------------------------------------------------------------------
    # Interface tests – check that required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_omega_z_present(self):
        if self.omega_z is None:
            return (False, "omega_z missing from results")
        if not isinstance(self.omega_z, (int, float)):
            return (False, f"omega_z is not numeric: {type(self.omega_z)}")
        return (True, "")

    def test_interface_omega_plus_present(self):
        if self.omega_plus is None:
            return (False, "omega_plus missing from results")
        if not isinstance(self.omega_plus, (int, float)):
            return (False, f"omega_plus is not numeric: {type(self.omega_plus)}")
        return (True, "")

    def test_interface_omega_minus_present(self):
        if self.omega_minus is None:
            return (False, "omega_minus missing from results")
        if not isinstance(self.omega_minus, (int, float)):
            return (False, f"omega_minus is not numeric: {type(self.omega_minus)}")
        return (True, "")

    def test_interface_omega_plus_weak_present(self):
        if self.omega_plus_weak is None:
            return (False, "omega_plus_weak missing from results")
        if not isinstance(self.omega_plus_weak, (int, float)):
            return (False, f"omega_plus_weak is not numeric: {type(self.omega_plus_weak)}")
        return (True, "")

    def test_interface_omega_minus_weak_present(self):
        if self.omega_minus_weak is None:
            return (False, "omega_minus_weak missing from results")
        if not isinstance(self.omega_minus_weak, (int, float)):
            return (False, f"omega_minus_weak is not numeric: {type(self.omega_minus_weak)}")
        return (True, "")

    def test_interface_omega_plus_strong_present(self):
        if self.omega_plus_strong is None:
            return (False, "omega_plus_strong missing from results")
        if not isinstance(self.omega_plus_strong, (int, float)):
            return (False, f"omega_plus_strong is not numeric: {type(self.omega_plus_strong)}")
        return (True, "")

    def test_interface_omega_minus_strong_present(self):
        if self.omega_minus_strong is None:
            return (False, "omega_minus_strong missing from results")
        if not isinstance(self.omega_minus_strong, (int, float)):
            return (False, f"omega_minus_strong is not numeric: {type(self.omega_minus_strong)}")
        return (True, "")

    def test_interface_givens_present(self):
        missing = []
        if self.m is None:
            missing.append('m')
        if self.q is None:
            missing.append('q')
        if self.B is None:
            missing.append('B')
        if self.omega0 is None:
            missing.append('omega0')
        if missing:
            return (False, f"Missing required given(s): {', '.join(missing)}")
        return (True, "")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_omega_z_equals_omega0(self):
        if self.omega_z is None or self.omega0 is None:
            return (False, "SKIP: omega_z or omega0 not available")
        if math.isclose(self.omega_z, self.omega0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_z {self.omega_z} != omega0 {self.omega0}")

    def test_invariant_omega_plus_minus_product(self):
        if self.omega_plus is None or self.omega_minus is None or self.omega0 is None:
            return (False, "SKIP: omega_plus, omega_minus, or omega0 not available")
        product = self.omega_plus * self.omega_minus
        expected = self.omega0 ** 2
        if math.isclose(product, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_plus*omega_minus = {product}, expected {expected}")

    def test_invariant_omega_plus_minus_difference(self):
        if self.omega_plus is None or self.omega_minus is None or self._omega_c is None:
            return (False, "SKIP: omega_plus, omega_minus, or omega_c not available")
        diff = self.omega_plus - self.omega_minus
        if math.isclose(diff, self._omega_c, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_plus - omega_minus = {diff}, expected omega_c = {self._omega_c}")

    def test_invariant_weak_plus(self):
        if self.omega_plus_weak is None or self.omega0 is None or self._omega_c is None:
            return (False, "SKIP: omega_plus_weak, omega0, or omega_c not available")
        expected = self.omega0 + self._omega_c / 2
        if math.isclose(self.omega_plus_weak, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_plus_weak = {self.omega_plus_weak}, expected weak approx {expected}")

    def test_invariant_weak_minus(self):
        if self.omega_minus_weak is None or self.omega0 is None or self._omega_c is None:
            return (False, "SKIP: omega_minus_weak, omega0, or omega_c not available")
        expected = self.omega0 - self._omega_c / 2
        if math.isclose(self.omega_minus_weak, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_minus_weak = {self.omega_minus_weak}, expected weak approx {expected}")

    def test_invariant_strong_plus(self):
        if self.omega_plus_strong is None or self.omega0 is None or self._omega_c is None or self._omega_c == 0:
            return (False, "SKIP: omega_plus_strong, omega0, or omega_c not available or omega_c=0")
        expected = self._omega_c + self.omega0**2 / self._omega_c
        if math.isclose(self.omega_plus_strong, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_plus_strong = {self.omega_plus_strong}, expected strong approx {expected}")

    def test_invariant_strong_minus(self):
        if self.omega_minus_strong is None or self.omega0 is None or self._omega_c is None or self._omega_c == 0:
            return (False, "SKIP: omega_minus_strong, omega0, or omega_c not available or omega_c=0")
        expected = self.omega0**2 / self._omega_c
        if math.isclose(self.omega_minus_strong, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_minus_strong = {self.omega_minus_strong}, expected strong approx {expected}")

    # ------------------------------------------------------------------
    # Modeling tests – check that exact formulas are correctly implemented
    # ------------------------------------------------------------------
    def test_modeling_exact_omega_plus(self):
        if self.omega_plus is None or self.omega0 is None or self._omega_c is None:
            return (False, "SKIP: omega_plus, omega0, or omega_c not available")
        expected = math.sqrt(self.omega0**2 + (self._omega_c/2)**2) + self._omega_c/2
        if math.isclose(self.omega_plus, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_plus = {self.omega_plus}, expected exact = {expected}")

    def test_modeling_exact_omega_minus(self):
        if self.omega_minus is None or self.omega0 is None or self._omega_c is None:
            return (False, "SKIP: omega_minus, omega0, or omega_c not available")
        expected = math.sqrt(self.omega0**2 + (self._omega_c/2)**2) - self._omega_c/2
        if math.isclose(self.omega_minus, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "")
        else:
            return (False, f"omega_minus = {self.omega_minus}, expected exact = {expected}")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb an input and compare perturbed output to baseline
    # ------------------------------------------------------------------
    def test_monotonicity_omega_plus_increases_with_B(self):
        # Baseline: solution's computed omega_plus
        if self.omega_plus is None or self.omega0 is None or self._omega_c is None:
            return (False, "SKIP: omega_plus, omega0, or omega_c not available")
        wc = self._omega_c
        # Perturbation: increase wc by 1% (or at least a small nonzero amount)
        eps = max(1e-6, 0.01 * wc) if wc != 0 else 1e-6
        # Compute expected omega_plus for perturbed wc using exact formula
        def f(x):
            return math.sqrt(self.omega0**2 + (x/2)**2) + x/2
        perturbed_omega_plus = f(wc + eps)
        if perturbed_omega_plus > self.omega_plus:
            return (True, "")
        else:
            return (False, f"omega_plus should increase with B: baseline {self.omega_plus}, "
                           f"expected after +{eps} {perturbed_omega_plus}")

    def test_monotonicity_omega_minus_decreases_with_B(self):
        if self.omega_minus is None or self.omega0 is None or self._omega_c is None:
            return (False, "SKIP: omega_minus, omega0, or omega_c not available")
        wc = self._omega_c
        eps = max(1e-6, 0.01 * wc) if wc != 0 else 1e-6
        def f(x):
            return math.sqrt(self.omega0**2 + (x/2)**2) - x/2
        perturbed_omega_minus = f(wc + eps)
        if perturbed_omega_minus < self.omega_minus:
            return (True, "")
        else:
            return (False, f"omega_minus should decrease with B: baseline {self.omega_minus}, "
                           f"expected after +{eps} {perturbed_omega_minus}")

    # ------------------------------------------------------------------
    # Answer sanity test – weak check (no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        freqs = [self.omega_z, self.omega_plus, self.omega_minus,
                 self.omega_plus_weak, self.omega_minus_weak,
                 self.omega_plus_strong, self.omega_minus_strong]
        non_none = [f for f in freqs if f is not None]
        if not non_none:
            return (False, "SKIP: no frequencies provided")
        for f in non_none:
            if f <= 0:
                return (False, f"Frequency {f} is not positive")
        return (True, "")

    # ------------------------------------------------------------------
    # Run all tests in a safe try/except loop
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all callable methods starting with "test_"
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)