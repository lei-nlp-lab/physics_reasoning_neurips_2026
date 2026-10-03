import math
import sys

class PhysicsChecker:
    """
    PhysicsChecker for N identical spinless bosons in a box problem.
    Extracts relevant quantities from solution_instance and runs a series
    of physics sanity, interface, and modeling tests.
    """
    def __init__(self, solution_instance):
        # Safe extraction from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Emit keys from pseudocode contract (safe .get with None)
        self.n_eps = self.results.get('n_eps', None)
        self.z = self.results.get('z', None)
        self.e_mu = self.results.get('e^{-beta mu}', None)   # key exactly as emitted
        self.E_B = self.results.get('E_B', None)
        self.Delta_E = self.results.get('Delta_E', None)

        # Raw givens (may be missing)
        self.N = self.givens.get('N', None)
        self.m = self.givens.get('m', None)
        self.V = self.givens.get('V', None)
        self.T = self.givens.get('T', None)
        self.h = self.givens.get('h', None)
        self.k = self.givens.get('k', None)
        self.mu = self.givens.get('mu', None)

        # Derived quantities (only if all needed givens present)
        self.lambda_db = None
        if (self.h is not None and self.m is not None and
            self.k is not None and self.T is not None and
            self.T > 0 and self.m > 0 and self.h > 0 and self.k > 0):
            try:
                self.lambda_db = self.h / math.sqrt(2.0 * math.pi * self.m * self.k * self.T)
            except (ZeroDivisionError, ValueError):
                pass

        self.d = None
        if self.N is not None and self.V is not None and self.N > 0 and self.V > 0:
            try:
                self.d = (self.V / self.N) ** (1.0 / 3.0)
            except (ZeroDivisionError, ValueError):
                pass

        # For numeric checks, attempt to convert emitted quantities to floats if possible
        self._numeric_z = None
        self._numeric_e_mu = None
        self._numeric_E_B = None
        self._numeric_Delta_E = None
        self._try_numeric()

    def _try_numeric(self):
        """Attempt to convert emitted quantities to numeric values."""
        for attr in ['z', 'e_mu', 'E_B', 'Delta_E']:
            val = getattr(self, attr, None)
            if val is not None:
                try:
                    float_val = float(val)
                    setattr(self, f'_numeric_{attr}', float_val)
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # Interface tests  – required keys exist
    # ------------------------------------------------------------------
    def test_interface_n_eps_exists(self):
        if self.n_eps is None:
            return (False, "SKIP: n_eps not found in results")
        return (True, "n_eps key present")

    def test_interface_z_exists(self):
        if self.z is None:
            return (False, "SKIP: z not found in results")
        return (True, "z key present")

    def test_interface_e_mu_exists(self):
        if self.e_mu is None:
            return (False, "SKIP: e^{-beta mu} not found in results")
        return (True, "e^{-beta mu} key present")

    def test_interface_E_B_exists(self):
        if self.E_B is None:
            return (False, "SKIP: E_B not found in results")
        return (True, "E_B key present")

    def test_interface_Delta_E_exists(self):
        if self.Delta_E is None:
            return (False, "SKIP: Delta_E not found in results")
        return (True, "Delta_E key present")

    # ------------------------------------------------------------------
    # Invariant tests  – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_mu_nonpositive(self):
        if self.mu is None:
            return (False, "SKIP: mu not provided")
        if self.mu > 0:
            return (False, f"mu must be non-positive, got mu={self.mu}")
        return (True, f"mu <= 0 satisfied (mu={self.mu})")

    def test_invariant_fugacity_positive(self):
        if self._numeric_z is None:
            return (False, "SKIP: z not numeric")
        if self._numeric_z <= 0:
            return (False, f"fugacity z must be positive, got {self._numeric_z}")
        return (True, f"z > 0 satisfied (z={self._numeric_z})")

    def test_invariant_e_mu_positive(self):
        if self._numeric_e_mu is None:
            return (False, "SKIP: e^{-beta mu} not numeric")
        if self._numeric_e_mu <= 0:
            return (False, rf"e^{{-beta mu}} must be positive, got {self._numeric_e_mu}")
        return (True, rf"e^{{-beta mu}} > 0 satisfied")

    def test_invariant_inverse_relation(self):
        # Check that z * e^{-beta mu} ≈ 1 (they are inverses)
        if self._numeric_z is None or self._numeric_e_mu is None:
            return (False, "SKIP: z or e^{-beta mu} not available numerically")
        product = self._numeric_z * self._numeric_e_mu
        if not math.isclose(product, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, rf"z * e^{{-beta mu}} = {product}, expected 1.0")
        return (True, rf"z * e^{{-beta mu}} = {product}, close to 1.0")

    def test_invariant_z_via_lambda_d(self):
        # If lambda and d are known, check z = (lambda/d)^3
        if self._numeric_z is None or self.lambda_db is None or self.d is None:
            return (False, "SKIP: insufficient data for lambda/d relation")
        expected_z = (self.lambda_db / self.d) ** 3
        if not math.isclose(self._numeric_z, expected_z, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"z = {self._numeric_z}, but (lambda/d)^3 = {expected_z}")
        return (True, f"z ≈ (lambda/d)^3 = {expected_z}")

    def test_invariant_e_mu_via_d_lambda(self):
        # e^{-beta mu} = (d/lambda)^3
        if self._numeric_e_mu is None or self.lambda_db is None or self.d is None:
            return (False, "SKIP: insufficient data for d/lambda relation")
        expected_e_mu = (self.d / self.lambda_db) ** 3
        if not math.isclose(self._numeric_e_mu, expected_e_mu, rel_tol=1e-2, abs_tol=1e-2):
            return (False, rf"e^{{-beta mu}} = {self._numeric_e_mu}, but (d/lambda)^3 = {expected_e_mu}")
        return (True, rf"e^{{-beta mu}} ≈ (d/lambda)^3 = {expected_e_mu}")

    def test_invariant_delta_E_positive(self):
        # First order quantum correction should be positive (bosons have higher energy than classical)
        if self._numeric_Delta_E is None:
            return (False, "SKIP: Delta_E not numeric")
        if self._numeric_Delta_E <= 0:
            return (False, f"Delta_E must be positive, got {self._numeric_Delta_E}")
        return (True, f"Delta_E = {self._numeric_Delta_E} > 0")

    # ------------------------------------------------------------------
    # Modeling/trace tests – based on explanation steps and expected form
    # ------------------------------------------------------------------
    def test_modeling_n_eps_structure(self):
        """Check that n_eps string contains expected functional form: sqrt(eps) / (exp(...)-1)"""
        if self.n_eps is None:
            return (False, "SKIP: n_eps not available")
        n_str = str(self.n_eps).replace(" ", "")
        # Must contain sqrt, exp, and a subtraction of 1 somewhere
        has_sqrt = "sqrt" in n_str or "√" in n_str
        has_exp = "exp" in n_str
        has_minus_one = "-1" in n_str or "- 1" in n_str
        if not (has_sqrt and has_exp and has_minus_one):
            return (False, "n_eps does not contain expected structure sqrt(eps)/[exp(...)-1]")
        return (True, "n_eps contains sqrt, exp, and '-1'")

    def test_modeling_z_structure_classical_limit(self):
        """z should be small (<<1) in the classical limit condition"""
        if self._numeric_z is None:
            return (False, "SKIP: z not numeric")
        if self._numeric_z >= 1:
            return (False, f"z = {self._numeric_z}, not in classical limit (require z << 1)")
        return (True, f"z = {self._numeric_z} < 1, consistent with classical limit")

    def test_modeling_E_B_contains_three_half_NkT(self):
        """E_B expression should contain the factor 3/2 N k T"""
        if self.E_B is None:
            return (False, "SKIP: E_B not available")
        eb_str = str(self.E_B).replace(" ", "")
        # Check for variations: "3/2", "1.5", "N", "k", "T"
        has_numerator = any(x in eb_str for x in ["3/2", "1.5", "3/2*N", "3*N"])
        has_kT = "kT" in eb_str or "k*T" in eb_str
        if not (has_numerator and has_kT):
            return (False, "E_B expression does not contain obvious (3/2)N k T factor")
        return (True, "E_B appears to contain (3/2)N k T factor")

    def test_modeling_Delta_E_structure(self):
        """Delta_E should be proportional to (3/2)N k T * (lambda^3/d^3) / (4 sqrt(2))"""
        if self.Delta_E is None:
            return (False, "SKIP: Delta_E not available")
        de_str = str(self.Delta_E).replace(" ", "")
        # Check for factor 1/(4 sqrt(2)) or 1/(4*√2) etc.
        has_numerical_factor = any(x in de_str.lower() for x in ["1/(4*sqrt(2))", "1/(4*√2)", "0.17678", "0.17678*"])
        if not has_numerical_factor:
            return (False, "Delta_E does not appear to contain the expected 1/(4√2) factor")
        return (True, "Delta_E contains 1/(4√2) factor")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_delta_E_increases_with_lambda_d_ratio(self):
        """
        Delta_E is proportional to (lambda/d)^3. 
        We perturb d by a small factor and check that the predicted Delta_E changes in the correct direction.
        """
        # We need lambda and d to compute baseline and perturbed values
        if self.lambda_db is None or self.d is None or self.N is None or self.k is None or self.T is None:
            return (False, "SKIP: missing required data (lambda, d, N, k, T) for monotonicity test")

        # Baseline: compute the theoretical Delta_E using the known expression
        # from physics: Delta_E = (3/2) * N * k * T * (1/(4*sqrt(2))) * (lambda/d)^3
        try:
            factor = (1.0 / (4.0 * math.sqrt(2.0)))
            baseline_Delta_E = (3.0/2.0) * self.N * self.k * self.T * factor * (self.lambda_db / self.d) ** 3
        except (ZeroDivisionError, ValueError):
            return (False, "SKIP: numerical error computing baseline")

        # Perturb: decrease d by 10% → increase (lambda/d) ratio
        d_perturbed = self.d * 0.9
        perturbed_Delta_E = (3.0/2.0) * self.N * self.k * self.T * factor * (self.lambda_db / d_perturbed) ** 3

        # If the relation is monotonic increasing, perturbed > baseline
        if perturbed_Delta_E <= baseline_Delta_E:
            return (False, f"Monotonicity fails: perturbed Delta_E = {perturbed_Delta_E} <= baseline {baseline_Delta_E}")
        return (True, f"Delta_E increases when d decreases (baseline {baseline_Delta_E}, perturbed {perturbed_Delta_E})")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: check that Delta_E is finite and not NaN.
        This is the only answer check; no re-derivation or isclose.
        """
        if self._numeric_Delta_E is None:
            return (False, "SKIP: Delta_E not numeric")
        if math.isnan(self._numeric_Delta_E) or math.isinf(self._numeric_Delta_E):
            return (False, f"Delta_E is not a finite number (got {self._numeric_Delta_E})")
        return (True, f"Delta_E = {self._numeric_Delta_E} is finite")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all methods starting with "test_"
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