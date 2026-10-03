import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for electromagnetic wave propagation in a free-electron medium.
    Checks model consistency, invariant relations, and basic sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract both results and givens
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}

        # --- Required output keys from [emit] contract ---
        self.omega_p = results.get('omega_p', None)      # plasma frequency
        self.K = results.get('K', None)                  # wavenumber
        self.B = results.get('B', None)                  # magnetic field (string or expression)
        self.lambda_ = results.get('lambda', None)       # wavelength (avoid shadowing)

        # --- Input quantities (from givens) ---
        self.omega = givens.get('omega', None)           # angular frequency
        self.n = givens.get('n', None)                   # free electron density
        self.e = givens.get('e', None)                   # elementary charge
        self.m = givens.get('m', None)                   # electron mass
        self.epsilon0 = givens.get('epsilon0', None)     # vacuum permittivity
        self.mu0 = givens.get('mu0', None)               # vacuum permeability
        self.E0 = givens.get('E0', None)                 # amplitude of electric field (optional)

        # --- Derived constants ---
        self.pi = math.pi

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _is_numeric(self, x):
        """Check if x is a numeric value (not None, not string/symbol)."""
        if x is None:
            return False
        if isinstance(x, (int, float, complex)):
            return True
        return False

    def _speed_of_light(self):
        """Return c if epsilon0 and mu0 are given, else None."""
        if self.epsilon0 is not None and self.mu0 is not None:
            return 1.0 / math.sqrt(self.epsilon0 * self.mu0)
        return None

    def _safe_close(self, a, b):
        """Fuzzy comparison with 1% tolerance. Only for invariant/modeling tests."""
        if not (self._is_numeric(a) and self._is_numeric(b)):
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    # ------------------------------------------------------------------
    # Interface checks (key existence, type)
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four [emit] keys exist and are not None."""
        missing = []
        for key in ['omega_p', 'K', 'B', 'lambda']:
            if getattr(self, key, None) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing required results keys: {missing}")
        return (True, "All required keys are present")

    def test_interface_B_format(self):
        """Check that B is a string containing the expected form (e_x direction, negative sign)."""
        if self.B is None:
            return (False, "SKIP: B not available")
        if not isinstance(self.B, str):
            return (False, f"B should be a string, got {type(self.B)}")
        if '-K/omega' not in self.B and '-K/ω' not in self.B:
            return (False, "B string does not contain '-K/omega' form")
        if 'e_x' not in self.B and 'x' not in self.B:
            return (False, "B string does not contain e_x direction")
        return (True, "B format is correct")

    def test_interface_lambda_type(self):
        """Check that lambda is numeric or at least not None."""
        if self.lambda_ is None:
            return (False, "SKIP: lambda not available")
        if self._is_numeric(self.lambda_):
            return (True, "lambda is numeric")
        elif isinstance(self.lambda_, str):
            return (True, "lambda is a string expression")
        else:
            return (False, f"lambda has unexpected type {type(self.lambda_)}")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based physical laws)
    # ------------------------------------------------------------------
    def test_invariant_dispersion_relation(self):
        """Check K^2 = (omega^2 - omega_p^2) / c^2."""
        if not (self._is_numeric(self.K) and self._is_numeric(self.omega) and self._is_numeric(self.omega_p)):
            return (False, "SKIP: missing numeric K, omega, or omega_p")
        c = self._speed_of_light()
        if c is None:
            return (False, "SKIP: cannot compute c (provide epsilon0 and mu0)")
        K_sq = self.K ** 2
        expected_K_sq = (self.omega ** 2 - self.omega_p ** 2) / (c ** 2)
        if self._safe_close(K_sq, expected_K_sq):
            return (True, f"K^2 matches dispersion: {K_sq} ≈ {expected_K_sq}")
        else:
            return (False, f"K^2 = {K_sq}, expected {expected_K_sq}")

    def test_invariant_propagation_condition(self):
        """Check that omega^2 > omega_p^2 (positive K^2)."""
        if not (self._is_numeric(self.omega) and self._is_numeric(self.omega_p)):
            return (False, "SKIP: numeric omega or omega_p missing")
        if self.omega ** 2 <= self.omega_p ** 2 - 1e-12:
            return (False, f"omega^2 = {self.omega**2} is not > omega_p^2 = {self.omega_p**2}")
        return (True, "omega^2 > omega_p^2 holds")

    def test_invariant_plasma_frequency_definition(self):
        """Check omega_p^2 = n e^2 / (m epsilon0)."""
        if not (self._is_numeric(self.omega_p) and
                self._is_numeric(self.n) and
                self._is_numeric(self.e) and
                self._is_numeric(self.m) and
                self._is_numeric(self.epsilon0)):
            return (False, "SKIP: missing required numeric values for plasma frequency")
        expected = self.n * self.e ** 2 / (self.m * self.epsilon0)
        if self._safe_close(self.omega_p ** 2, expected):
            return (True, f"omega_p^2 matches definition: {self.omega_p**2} ≈ {expected}")
        else:
            return (False, f"omega_p^2 = {self.omega_p**2}, expected {expected}")

    # ------------------------------------------------------------------
    # Modeling / trace quality checks
    # ------------------------------------------------------------------
    def test_modeling_wavelength_from_K(self):
        """Check that lambda = 2π / K (if both numeric)."""
        if not (self._is_numeric(self.lambda_) and self._is_numeric(self.K)):
            return (False, "SKIP: lambda or K not numeric")
        expected = 2 * self.pi / self.K
        if self._safe_close(self.lambda_, expected):
            return (True, f"lambda matches 2π/K: {self.lambda_} ≈ {expected}")
        else:
            return (False, f"lambda = {self.lambda_}, expected {expected}")

    def test_modeling_dispersion_alt_form(self):
        """Alternative form: omega^2 = omega_p^2 + c^2 K^2."""
        if not (self._is_numeric(self.omega) and self._is_numeric(self.omega_p) and self._is_numeric(self.K)):
            return (False, "SKIP: numeric values missing")
        c = self._speed_of_light()
        if c is None:
            return (False, "SKIP: cannot compute c")
        expected = self.omega_p ** 2 + (c * self.K) ** 2
        if self._safe_close(self.omega ** 2, expected):
            return (True, f"omega^2 matches omega_p^2 + c^2 K^2: {self.omega**2} ≈ {expected}")
        else:
            return (False, f"omega^2 = {self.omega**2}, expected {expected}")

    # ------------------------------------------------------------------
    # Monotonicity checks (directional trend via explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_lambda_decreases_with_omega(self):
        """Check that dλ/dω < 0: lambda decreases when omega increases."""
        if not (self._is_numeric(self.lambda_) and self._is_numeric(self.omega) and self._is_numeric(self.omega_p)):
            return (False, "SKIP: numeric lambda, omega, or omega_p missing")
        # compute perturbed lambda using +1% in omega
        omega_pert = self.omega * 1.01
        c = self._speed_of_light()
        if c is None:
            return (False, "SKIP: cannot compute c")
        # check that perturbed omega still > omega_p
        if omega_pert ** 2 <= self.omega_p ** 2:
            return (False, "Perturbed omega too small – cannot propagate")
        K_pert = math.sqrt((omega_pert**2 - self.omega_p**2) / (c**2))
        lambda_pert = 2 * self.pi / K_pert
        if lambda_pert < self.lambda_:
            return (True, f"lambda decreased: {self.lambda_} -> {lambda_pert}")
        else:
            return (False, f"lambda did not decrease: {self.lambda_} -> {lambda_pert}")

    def test_monotonicity_K_increases_with_omega(self):
        """Check that dK/dω > 0: K increases when omega increases."""
        if not (self._is_numeric(self.K) and self._is_numeric(self.omega) and self._is_numeric(self.omega_p)):
            return (False, "SKIP: numeric K, omega, or omega_p missing")
        omega_pert = self.omega * 1.01
        c = self._speed_of_light()
        if c is None:
            return (False, "SKIP: cannot compute c")
        K_pert_sq = (omega_pert**2 - self.omega_p**2) / (c**2)
        if K_pert_sq <= 0:
            return (False, "Perturbed K^2 negative – not propagating")
        K_pert = math.sqrt(K_pert_sq)
        if K_pert > self.K:
            return (True, f"K increased: {self.K} -> {K_pert}")
        else:
            return (False, f"K did not increase: {self.K} -> {K_pert}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, at most one method)
    # ------------------------------------------------------------------
    def test_answer_sanity_lambda_positive(self):
        """Check that lambda is positive (if numeric)."""
        if self.lambda_ is None:
            return (False, "SKIP: lambda not available")
        if self._is_numeric(self.lambda_):
            if self.lambda_ > 0:
                return (True, "lambda positive")
            else:
                return (False, f"lambda is not positive: {self.lambda_}")
        else:
            # string expression - can't check numerically, but acceptable
            return (True, "lambda expression present (numerical positivity not verified)")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)