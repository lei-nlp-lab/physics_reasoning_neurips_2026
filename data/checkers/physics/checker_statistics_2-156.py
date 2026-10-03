import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.T = self.givens.get('T', None)
        self.P = self.givens.get('P', None)
        self.d = self.givens.get('d', None)
        self.m = self.givens.get('m', None)
        self.r_target = self.givens.get('r_target', None)

        # Extract results safely
        self.results = getattr(solution_instance, 'results', None) or {}
        self.lambda_mp = self.results.get('lambda_mp', None)
        self.v_mean = self.results.get('v_mean', None)
        self.D = self.results.get('D', None)
        self.t = self.results.get('t', None)

        # Physical constants
        self.k_B = 1.380649e-23
        self.pi = math.pi
        self.sqrt2 = math.sqrt(2)

    def _all_givens_present(self, keys):
        for key in keys:
            if getattr(self, key, None) is None:
                return False
        return True

    def _all_results_present(self, keys):
        for key in keys:
            if getattr(self, key, None) is None:
                return False
        return True

    # ------------- Interface Tests -------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist."""
        required = ['lambda_mp', 'v_mean', 'D', 't']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_results_positive(self):
        """Check that all results are positive."""
        if not self._all_results_present(['lambda_mp', 'v_mean', 'D', 't']):
            return (False, "SKIP: some results missing")
        for name in ['lambda_mp', 'v_mean', 'D', 't']:
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric")
            if val <= 0:
                return (False, f"{name} = {val} is not positive")
        return (True, "All results positive")

    def test_interface_results_finite(self):
        """Check that all results are finite."""
        if not self._all_results_present(['lambda_mp', 'v_mean', 'D', 't']):
            return (False, "SKIP: some results missing")
        for name in ['lambda_mp', 'v_mean', 'D', 't']:
            val = getattr(self, name)
            if not math.isfinite(val):
                return (False, f"{name} = {val} is not finite")
        return (True, "All results finite")

    # ------------- Invariant / Theorem Tests -------------
    def test_invariant_lambda_mp_formula(self):
        """Check lambda_mp computed from kinetic theory for given T,P,d."""
        if not self._all_givens_present(['T', 'P', 'd']) or self.lambda_mp is None:
            return (False, "SKIP: missing givens or lambda_mp")
        T, P, d = self.T, self.P, self.d
        expected_lambda = self.k_B * T / (self.sqrt2 * self.pi * d**2 * P)
        if math.isclose(self.lambda_mp, expected_lambda, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"lambda_mp matches theory: {expected_lambda}")
        else:
            return (False, f"lambda_mp {self.lambda_mp} != expected {expected_lambda}")

    def test_invariant_v_mean_formula(self):
        """Check v_mean from kinetic theory for given T,m."""
        if not self._all_givens_present(['T', 'm']) or self.v_mean is None:
            return (False, "SKIP: missing givens or v_mean")
        T, m = self.T, self.m
        expected_v = math.sqrt(8 * self.k_B * T / (self.pi * m))
        if math.isclose(self.v_mean, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_mean matches theory: {expected_v}")
        else:
            return (False, f"v_mean {self.v_mean} != expected {expected_v}")

    def test_invariant_D_formula(self):
        """Check D = (1/3) * lambda_mp * v_mean."""
        if None in (self.lambda_mp, self.v_mean, self.D):
            return (False, "SKIP: missing lambda_mp, v_mean, or D")
        expected_D = (1/3) * self.lambda_mp * self.v_mean
        if math.isclose(self.D, expected_D, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"D matches (1/3)*lambda*v: {expected_D}")
        else:
            return (False, f"D {self.D} != expected {expected_D}")

    def test_invariant_einstein_relation(self):
        """Check RMS displacement = sqrt(6 D t) with r_target."""
        if None in (self.r_target, self.D, self.t):
            return (False, "SKIP: missing r_target, D, or t")
        derived_D = self.r_target**2 / (6 * self.t)
        if math.isclose(self.D, derived_D, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"Einstein relation holds: D={self.D}, derived D={derived_D}")
        else:
            return (False, f"Einstein relation fails: D={self.D} vs derived D={derived_D}")

    def test_invariant_diffusive_regime(self):
        """Check lambda_mp << r_target."""
        if None in (self.lambda_mp, self.r_target):
            return (False, "SKIP: missing lambda_mp or r_target")
        ratio = self.lambda_mp / self.r_target
        if ratio < 0.01:
            return (True, f"Diffusive regime: lambda/r = {ratio}")
        else:
            return (False, f"Not diffusive: lambda/r = {ratio}")

    # ------------- Modeling / Trace Consistency Tests -------------
    def test_modeling_time_from_D_and_r(self):
        """Check that t computed from D and r_target matches self.t."""
        if None in (self.D, self.r_target, self.t):
            return (False, "SKIP: missing D, r_target, or t")
        derived_t = self.r_target**2 / (6 * self.D)
        if math.isclose(self.t, derived_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t matches derived value {derived_t}")
        else:
            return (False, f"t={self.t} deviates from derived t={derived_t}")

    def test_modeling_time_via_lambda_v(self):
        """Check t = r^2/(2 λ v) derived from kinetic theory."""
        if None in (self.r_target, self.lambda_mp, self.v_mean, self.t):
            return (False, "SKIP: missing r_target, lambda_mp, v_mean, or t")
        # From D = (1/3) λ v and t = r^2/(6D) => t = r^2/(2 λ v)
        denominator = 2 * self.lambda_mp * self.v_mean
        if denominator == 0:
            return (False, "SKIP: zero denominator")
        derived_t = self.r_target**2 / denominator
        if math.isclose(self.t, derived_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t matches derived t={derived_t} from r^2/(2λv)")
        else:
            return (False, f"t={self.t} deviates from derived t={derived_t}")

    # ------------- Monotonicity Tests -------------
    def test_monotonicity_time_increases_with_distance(self):
        """If r_target increases, t should increase (t ∝ r^2)."""
        if None in (self.r_target, self.D, self.t):
            return (False, "SKIP: missing r_target, D, or t")
        baseline_t = self.t
        perturbed_r = self.r_target * 1.05
        perturbed_t = perturbed_r**2 / (6 * self.D)
        if perturbed_t > baseline_t:
            return (True, f"t increases when r increases: baseline t={baseline_t}, perturbed t={perturbed_t}")
        else:
            return (False, f"Monotonicity broken: perturbed t={perturbed_t} <= baseline t={baseline_t}")

    def test_monotonicity_time_decreases_with_temperature(self):
        """If T increases, D increases and t decreases (via kinetic theory)."""
        required_givens = ['T', 'P', 'd', 'm', 'r_target']
        if not self._all_givens_present(required_givens) or self.t is None:
            return (False, "SKIP: missing givens or t")
        baseline_t = self.t
        perturbed_T = self.T * 1.10
        perturbed_lambda = self.k_B * perturbed_T / (self.sqrt2 * self.pi * self.d**2 * self.P)
        perturbed_v = math.sqrt(8 * self.k_B * perturbed_T / (self.pi * self.m))
        perturbed_D = (1/3) * perturbed_lambda * perturbed_v
        perturbed_t = self.r_target**2 / (6 * perturbed_D)
        if perturbed_t < baseline_t:
            return (True, f"t decreases when T increases: baseline t={baseline_t}, perturbed t={perturbed_t}")
        else:
            return (False, f"Monotonicity broken: perturbed t={perturbed_t} >= baseline t={baseline_t}")

    # ------------- Weak Answer Sanity Tests (only one) -------------
    def test_answer_sanity_order_of_magnitude(self):
        """Check that t is within broad expected range (hours to days)."""
        if self.t is None:
            return (False, "SKIP: t not provided")
        if 1e3 < self.t < 1e7:
            return (True, f"t = {self.t} is within plausible range")
        else:
            return (False, f"t = {self.t} is not in [1e3, 1e7] seconds")

    # ------------- Run All Tests -------------
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

# Note: The checker assumes solution_instance has .givens and .results (dict-like).