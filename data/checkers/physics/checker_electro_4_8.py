import math
import traceback

class PhysicsChecker:
    """
    Auditor for Maxwell's equations symmetry problem.
    Checks solution correctness under charge reversal, parity, and time reversal.
    """

    def __init__(self, solution_instance):
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract each expected key with None default
        self.maxwell_eqs = self.results.get('maxwell_eqs', None)
        self.E_prime = self.results.get('E_prime', None)           # charge reversal
        self.B_prime = self.results.get('B_prime', None)
        self.rho_prime = self.results.get('rho_prime', None)       # parity
        self.j_prime = self.results.get('j_prime', None)
        self.E_prime_p = self.results.get('E_prime_p', None)
        self.B_prime_p = self.results.get('B_prime_p', None)
        self.rho_prime_T = self.results.get('rho_prime_T', None)   # time reversal
        self.j_prime_T = self.results.get('j_prime_T', None)
        self.E_prime_T = self.results.get('E_prime_T', None)
        self.B_prime_T = self.results.get('B_prime_T', None)

    def _sign_factor(self, expr_str, variable):
        """
        Determine the factor relating transformed field to original field.
        e.g. '-E' returns -1, 'E' returns 1.
        Returns None if input is None.
        """
        if expr_str is None:
            return None
        normalized = expr_str.replace(" ", "")
        idx = normalized.find(variable)
        if idx == -1:
            return None   # unrecognized
        # Check if there is a minus sign immediately before the variable
        if idx > 0 and normalized[idx-1] == '-':
            return -1
        return 1

    # ------------------------------------------------------------------
    # Interface tests (key existence and basic type)
    # ------------------------------------------------------------------
    def test_interface_maxwell_eqs(self):
        if self.maxwell_eqs is None:
            return (False, "SKIP: maxwell_eqs not provided")
        if not isinstance(self.maxwell_eqs, str):
            return (False, "maxwell_eqs should be a string")
        if len(self.maxwell_eqs) == 0:
            return (False, "maxwell_eqs string is empty")
        return (True, "maxwell_eqs present and non-empty")

    def test_interface_charge_reversal_keys(self):
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: charge reversal keys missing")
        return (True, "E_prime and B_prime provided")

    def test_interface_parity_keys(self):
        keys = ['rho_prime', 'j_prime', 'E_prime_p', 'B_prime_p']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"SKIP: parity keys missing: {missing}")
        return (True, "All parity transformation keys present")

    def test_interface_time_reversal_keys(self):
        keys = ['rho_prime_T', 'j_prime_T', 'E_prime_T', 'B_prime_T']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"SKIP: time reversal keys missing: {missing}")
        return (True, "All time reversal keys present")

    # ------------------------------------------------------------------
    # Invariant tests (static consistency)
    # ------------------------------------------------------------------
    def test_invariant_charge_reversal_E(self):
        factor = self._sign_factor(self.E_prime, 'E')
        if factor is None:
            return (False, "SKIP: E_prime not available")
        if factor == -1:
            return (True, "E_prime correctly flips E sign")
        else:
            return (False, "E_prime should be = -E, got factor 1")

    def test_invariant_charge_reversal_B(self):
        factor = self._sign_factor(self.B_prime, 'B')
        if factor is None:
            return (False, "SKIP: B_prime not available")
        if factor == -1:
            return (True, "B_prime correctly flips B sign")
        else:
            return (False, "B_prime should be = -B, got factor 1")

    def test_invariant_parity_rho(self):
        factor = self._sign_factor(self.rho_prime, 'rho')
        if factor is None:
            return (False, "SKIP: rho_prime not available")
        if factor == 1:
            return (True, "rho_prime unchanged under parity (scalar)")
        else:
            return (False, "rho_prime should be unchanged (factor 1)")

    def test_invariant_parity_j(self):
        factor = self._sign_factor(self.j_prime, 'j')
        if factor is None:
            return (False, "SKIP: j_prime not available")
        if factor == -1:
            return (True, "j_prime flips sign under parity (polar vector)")
        else:
            return (False, "j_prime should be = -j, got factor 1")

    def test_invariant_parity_E(self):
        factor = self._sign_factor(self.E_prime_p, 'E')
        if factor is None:
            return (False, "SKIP: E_prime_p not available")
        if factor == -1:
            return (True, "E flips under parity (polar vector)")
        else:
            return (False, "E should flip sign under parity")

    def test_invariant_parity_B(self):
        factor = self._sign_factor(self.B_prime_p, 'B')
        if factor is None:
            return (False, "SKIP: B_prime_p not available")
        if factor == 1:
            return (True, "B unchanged under parity (axial vector)")
        else:
            return (False, "B should be unchanged under parity")

    def test_invariant_time_reversal_rho(self):
        factor = self._sign_factor(self.rho_prime_T, 'rho')
        if factor is None:
            return (False, "SKIP: rho_prime_T not available")
        if factor == 1:
            return (True, "rho unchanged under time reversal")
        else:
            return (False, "rho should be unchanged under T")

    def test_invariant_time_reversal_j(self):
        factor = self._sign_factor(self.j_prime_T, 'j')
        if factor is None:
            return (False, "SKIP: j_prime_T not available")
        if factor == -1:
            return (True, "j flips under time reversal")
        else:
            return (False, "j should flip sign under T")

    def test_invariant_time_reversal_E(self):
        factor = self._sign_factor(self.E_prime_T, 'E')
        if factor is None:
            return (False, "SKIP: E_prime_T not available")
        if factor == 1:
            return (True, "E even under time reversal")
        else:
            return (False, "E should be unchanged under T")

    def test_invariant_time_reversal_B(self):
        factor = self._sign_factor(self.B_prime_T, 'B')
        if factor is None:
            return (False, "SKIP: B_prime_T not available")
        if factor == -1:
            return (True, "B odd under time reversal")
        else:
            return (False, "B should flip sign under T")

    # ------------------------------------------------------------------
    # Modeling quality tests (cross-check consistency)
    # ------------------------------------------------------------------
    def test_modeling_charge_reversal_linearity(self):
        """Both E and B must flip under charge reversal (linearity)."""
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: missing charge reversal fields")
        e_factor = self._sign_factor(self.E_prime, 'E')
        b_factor = self._sign_factor(self.B_prime, 'B')
        if e_factor == -1 and b_factor == -1:
            return (True, "Charge reversal: both fields flip consistently")
        else:
            return (False, "Under charge reversal, both E and B must flip sign")

    def test_modeling_parity_consistency(self):
        """Cross-check parity flips: rho no, j yes, E yes, B no."""
        keys = ['rho_prime', 'j_prime', 'E_prime_p', 'B_prime_p']
        if any(getattr(self, k) is None for k in keys):
            return (False, "SKIP: parity keys incomplete")
        rho_f = self._sign_factor(self.rho_prime, 'rho')
        j_f = self._sign_factor(self.j_prime, 'j')
        e_f = self._sign_factor(self.E_prime_p, 'E')
        b_f = self._sign_factor(self.B_prime_p, 'B')
        expected = (1, -1, -1, 1)
        actual = (rho_f, j_f, e_f, b_f)
        if actual == expected:
            return (True, "Parity transformations consistent")
        else:
            return (False, f"Expected {expected}, got {actual}")

    def test_modeling_time_reversal_consistency(self):
        """Cross-check T flips: rho no, j yes, E no, B yes."""
        keys = ['rho_prime_T', 'j_prime_T', 'E_prime_T', 'B_prime_T']
        if any(getattr(self, k) is None for k in keys):
            return (False, "SKIP: T keys incomplete")
        rho_f = self._sign_factor(self.rho_prime_T, 'rho')
        j_f = self._sign_factor(self.j_prime_T, 'j')
        e_f = self._sign_factor(self.E_prime_T, 'E')
        b_f = self._sign_factor(self.B_prime_T, 'B')
        expected = (1, -1, 1, -1)
        actual = (rho_f, j_f, e_f, b_f)
        if actual == expected:
            return (True, "Time reversal transformations consistent")
        else:
            return (False, f"Expected {expected}, got {actual}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input)
    # ------------------------------------------------------------------
    def test_monotonicity_charge_reversal_perturb(self):
        """
        Perturbation: flip sign of sources (multiply by -1).
        Baseline: arbitrary field values (E0=3.14, B0=2.71).
        Expected perturbed fields: -E0, -B0.
        Check that the solution's transformation rule yields the expected sign flip.
        """
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: charge reversal fields missing")
        # arbitrary baseline
        E0 = 3.14
        B0 = 2.71
        e_factor = self._sign_factor(self.E_prime, 'E')
        b_factor = self._sign_factor(self.B_prime, 'B')
        if e_factor is None or b_factor is None:
            return (False, "Cannot determine sign factor from E_prime/B_prime")
        perturbed_E = e_factor * E0
        perturbed_B = b_factor * B0
        expected_E = -E0
        expected_B = -B0
        # use isclose for robustness (although it's exact factors)
        if math.isclose(perturbed_E, expected_E, rel_tol=1e-9) and math.isclose(perturbed_B, expected_B, rel_tol=1e-9):
            return (True, "Charge reversal perturbation: fields correctly flip sign")
        else:
            return (False, f"Perturbed fields E'={perturbed_E}, B'={perturbed_B}, expected -E0={expected_E}, -B0={expected_B}")

    def test_monotonicity_parity_perturb(self):
        """
        Perturbation: apply space inversion (x -> -x).
        Baseline arbitrary: rho0=5, j0= (1,2,3), E0=(4,5,6), B0=(7,8,9)
        Expected under parity: rho unchanged, j flips, E flips, B unchanged.
        We only check sign factors, not full vector.
        """
        keys = ['rho_prime', 'j_prime', 'E_prime_p', 'B_prime_p']
        if any(getattr(self, k) is None for k in keys):
            return (False, "SKIP: parity keys missing")
        rho0 = 5.0
        j0 = 1.0   # one component, direction doesn't matter
        E0 = 4.0
        B0 = 7.0
        rho_f = self._sign_factor(self.rho_prime, 'rho')
        j_f = self._sign_factor(self.j_prime, 'j')
        e_f = self._sign_factor(self.E_prime_p, 'E')
        b_f = self._sign_factor(self.B_prime_p, 'B')
        if None in (rho_f, j_f, e_f, b_f):
            return (False, "Cannot determine sign factors for parity")
        perturbed_rho = rho_f * rho0
        perturbed_j = j_f * j0
        perturbed_E = e_f * E0
        perturbed_B = b_f * B0
        # expected: rho unchanged, j, E flip, B unchanged
        if (math.isclose(perturbed_rho, rho0) and
            math.isclose(perturbed_j, -j0) and
            math.isclose(perturbed_E, -E0) and
            math.isclose(perturbed_B, B0)):
            return (True, "Parity perturbation: all quantities transform correctly")
        else:
            return (False, "Parity perturbation mismatch")

    def test_monotonicity_time_reversal_perturb(self):
        """
        Perturbation: apply time reversal (t -> -t).
        Baseline arbitrary: same as above.
        Expected: rho unchanged, j flips, E unchanged, B flips.
        """
        keys = ['rho_prime_T', 'j_prime_T', 'E_prime_T', 'B_prime_T']
        if any(getattr(self, k) is None for k in keys):
            return (False, "SKIP: time reversal keys missing")
        rho0 = 5.0
        j0 = 1.0
        E0 = 4.0
        B0 = 7.0
        rho_f = self._sign_factor(self.rho_prime_T, 'rho')
        j_f = self._sign_factor(self.j_prime_T, 'j')
        e_f = self._sign_factor(self.E_prime_T, 'E')
        b_f = self._sign_factor(self.B_prime_T, 'B')
        if None in (rho_f, j_f, e_f, b_f):
            return (False, "Cannot determine sign factors for time reversal")
        perturbed_rho = rho_f * rho0
        perturbed_j = j_f * j0
        perturbed_E = e_f * E0
        perturbed_B = b_f * B0
        if (math.isclose(perturbed_rho, rho0) and
            math.isclose(perturbed_j, -j0) and
            math.isclose(perturbed_E, E0) and
            math.isclose(perturbed_B, -B0)):
            return (True, "Time reversal perturbation: all quantities transform correctly")
        else:
            return (False, "Time reversal perturbation mismatch")

    # ------------------------------------------------------------------
    # Weak answer sanity test (no re-derivation, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_maxwell_eqs(self):
        """Weak sanity: check that the Maxwell equations string contains relevant physics terms."""
        if self.maxwell_eqs is None:
            return (False, "SKIP: maxwell_eqs not provided")
        text = self.maxwell_eqs.lower()
        keywords = ['nabla', 'div', 'curl', 'rho', 'epsilon', 'grad', 'mu0', 'd/dt']
        if any(kw in text for kw in keywords):
            return (True, "maxwell_eqs contains recognizable terms")
        else:
            return (False, "maxwell_eqs lacks expected physics keywords")

    # ------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)