import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the magnetic dipole interaction problem of two electrons.
    Focuses on operator expression, eigenvalues, degeneracies, and physical invariants.
    """

    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract all expected keys with None defaults
        self.E_over_A_operator = self.results.get('E_over_A_operator')
        self.eigenvalue_singlet = self.results.get('eigenvalue_singlet')
        self.eigenvalue_triplet_pm = self.results.get('eigenvalue_triplet_pm')
        self.eigenvalue_triplet_0 = self.results.get('eigenvalue_triplet_0')
        self.degeneracy_singlet = self.results.get('degeneracy_singlet')
        self.degeneracy_triplet_pm = self.results.get('degeneracy_triplet_pm')
        self.degeneracy_triplet_0 = self.results.get('degeneracy_triplet_0')

    def _is_close(self, a, b):
        """Fuzzy comparison for modeling/invariant tests (1-2% tolerance)."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def _safe_skip(self, key):
        """Return True if key is missing."""
        val = getattr(self, key, None)
        return val is None

    # ---------- Interface tests ----------
    def test_interface_required_keys_present(self):
        """All required result keys must exist and be not None."""
        required_keys = [
            'E_over_A_operator', 'eigenvalue_singlet', 'eigenvalue_triplet_pm',
            'eigenvalue_triplet_0', 'degeneracy_singlet', 'degeneracy_triplet_pm',
            'degeneracy_triplet_0'
        ]
        for key in required_keys:
            if getattr(self, key, None) is None:
                return (False, f"Missing or None: {key}")
        return (True, "All required keys present")

    def test_interface_operator_type(self):
        """E_over_A_operator must be a string."""
        val = self.E_over_A_operator
        if val is None:
            return (False, "SKIP: E_over_A_operator not available")
        if not isinstance(val, str):
            return (False, f"Expected string, got {type(val)}")
        return (True, "E_over_A_operator is a string")

    def test_interface_numeric_types(self):
        """Eigenvalues should be numeric, degeneracies integer."""
        if self.eigenvalue_singlet is None or self.eigenvalue_triplet_pm is None or self.eigenvalue_triplet_0 is None:
            return (False, "SKIP: Missing eigenvalue")
        if not isinstance(self.eigenvalue_singlet, (int, float)):
            return (False, f"eigenvalue_singlet not numeric")
        if not isinstance(self.eigenvalue_triplet_pm, (int, float)):
            return (False, f"eigenvalue_triplet_pm not numeric")
        if not isinstance(self.eigenvalue_triplet_0, (int, float)):
            return (False, f"eigenvalue_triplet_0 not numeric")
        # degeneracies must be integers
        for deg in [self.degeneracy_singlet, self.degeneracy_triplet_pm, self.degeneracy_triplet_0]:
            if deg is None:
                return (False, "SKIP: Missing degeneracy")
            if not isinstance(deg, int):
                return (False, f"Degeneracy {deg} not int")
        return (True, "All eigenvalues numeric, degeneracies int")

    # ---------- Invariant tests (theorem-based) ----------
    def test_invariant_trace_zero(self):
        """Sum of (eigenvalue * degeneracy) should be zero (traceless operator)."""
        if any(x is None for x in [self.eigenvalue_singlet, self.eigenvalue_triplet_pm, self.eigenvalue_triplet_0,
                                   self.degeneracy_singlet, self.degeneracy_triplet_pm, self.degeneracy_triplet_0]):
            return (False, "SKIP: Missing data for trace")
        trace = (self.eigenvalue_singlet * self.degeneracy_singlet +
                 self.eigenvalue_triplet_pm * self.degeneracy_triplet_pm +
                 self.eigenvalue_triplet_0 * self.degeneracy_triplet_0)
        if not self._is_close(trace, 0.0):
            return (False, f"Trace = {trace}, expected 0")
        return (True, f"Trace = {trace} ~ 0")

    def test_invariant_dimension_total(self):
        """Sum of degeneracies must equal 4 (two spin-1/2)."""
        if any(deg is None for deg in [self.degeneracy_singlet, self.degeneracy_triplet_pm, self.degeneracy_triplet_0]):
            return (False, "SKIP: Missing degeneracies")
        total_dim = self.degeneracy_singlet + self.degeneracy_triplet_pm + self.degeneracy_triplet_0
        if total_dim != 4:
            return (False, f"Total dimension = {total_dim}, expected 4")
        return (True, f"Total dimension = {total_dim} = 4")

    def test_invariant_ordering(self):
        """Verify expected ordering: triplet_pm < singlet < triplet_0."""
        if any(x is None for x in [self.eigenvalue_singlet, self.eigenvalue_triplet_pm, self.eigenvalue_triplet_0]):
            return (False, "SKIP: Missing eigenvalue")
        if not (self.eigenvalue_triplet_pm < self.eigenvalue_singlet < self.eigenvalue_triplet_0):
            return (False, f"Ordering fails: {self.eigenvalue_triplet_pm} < {self.eigenvalue_singlet} < {self.eigenvalue_triplet_0} not satisfied")
        return (True, f"Ordering: {self.eigenvalue_triplet_pm} < {self.eigenvalue_singlet} < {self.eigenvalue_triplet_0}")

    def test_invariant_singlet_zero(self):
        """Singlet eigenvalue should be zero by formula (S=0,M=0)."""
        if self.eigenvalue_singlet is None:
            return (False, "SKIP: Missing eigenvalue_singlet")
        if not self._is_close(self.eigenvalue_singlet, 0.0):
            return (False, f"Singlet eigenvalue = {self.eigenvalue_singlet}, expected 0")
        return (True, f"Singlet eigenvalue = {self.eigenvalue_singlet} ~ 0")

    def test_invariant_triplet_pm_minus_half(self):
        """Triplet M=±1 eigenvalue should be -0.5 (S=1, M=±1)."""
        if self.eigenvalue_triplet_pm is None:
            return (False, "SKIP: Missing eigenvalue_triplet_pm")
        if not self._is_close(self.eigenvalue_triplet_pm, -0.5):
            return (False, f"Triplet ±1 eigenvalue = {self.eigenvalue_triplet_pm}, expected -0.5")
        return (True, f"Triplet ±1 eigenvalue = {self.eigenvalue_triplet_pm} ~ -0.5")

    def test_invariant_triplet_0_one(self):
        """Triplet M=0 eigenvalue should be 1 (S=1, M=0)."""
        if self.eigenvalue_triplet_0 is None:
            return (False, "SKIP: Missing eigenvalue_triplet_0")
        if not self._is_close(self.eigenvalue_triplet_0, 1.0):
            return (False, f"Triplet 0 eigenvalue = {self.eigenvalue_triplet_0}, expected 1")
        return (True, f"Triplet 0 eigenvalue = {self.eigenvalue_triplet_0} ~ 1")

    # ---------- Modeling/trace tests ----------
    def test_modeling_operator_form(self):
        """Operator expression must match the derived form (S^2 - 3 S_z^2)/2."""
        if self.E_over_A_operator is None:
            return (False, "SKIP: No operator expression")
        # Normalize: remove spaces
        expected = "(S^2 - 3 S_z^2)/2"
        cleaned_input = self.E_over_A_operator.replace(" ", "")
        cleaned_expected = expected.replace(" ", "")
        if cleaned_input == cleaned_expected:
            return (True, f"Operator expression matches '{expected}'")
        else:
            return (False, f"Operator expression '{self.E_over_A_operator}' does not match expected '{expected}'")

    def test_trace_eigenvalue_formula(self):
        """Re-derive eigenvalues from formula (S(S+1)-3M^2)/2 for each state category."""
        if any(x is None for x in [self.eigenvalue_singlet, self.eigenvalue_triplet_pm, self.eigenvalue_triplet_0]):
            return (False, "SKIP: Missing eigenvalues for formula check")
        # Singlet: S=0, M=0 -> (0 - 0)/2 = 0
        if not self._is_close(self.eigenvalue_singlet, 0.0):
            return (False, f"Singlet not matching formula: got {self.eigenvalue_singlet}")
        # Triplet ±1: S=1, M=±1 -> (2 - 3)/2 = -0.5
        if not self._is_close(self.eigenvalue_triplet_pm, -0.5):
            return (False, f"Triplet ±1 not matching formula: got {self.eigenvalue_triplet_pm}")
        # Triplet 0: S=1, M=0 -> (2 - 0)/2 = 1
        if not self._is_close(self.eigenvalue_triplet_0, 1.0):
            return (False, f"Triplet 0 not matching formula: got {self.eigenvalue_triplet_0}")
        return (True, "All eigenvalues match formula (S(S+1)-3M^2)/2")

    def test_trace_degeneracy_assignment(self):
        """Degeneracies must correspond to number of M states for each S category."""
        if any(deg is None for deg in [self.degeneracy_singlet, self.degeneracy_triplet_pm, self.degeneracy_triplet_0]):
            return (False, "SKIP: Missing degeneracies")
        # Singlet: S=0, M=0 -> 1 state
        if self.degeneracy_singlet != 1:
            return (False, f"Singlet degeneracy = {self.degeneracy_singlet}, expected 1")
        # Triplet ±1: M=±1 -> 2 states
        if self.degeneracy_triplet_pm != 2:
            return (False, f"Triplet ±1 degeneracy = {self.degeneracy_triplet_pm}, expected 2")
        # Triplet 0: M=0 -> 1 state
        if self.degeneracy_triplet_0 != 1:
            return (False, f"Triplet 0 degeneracy = {self.degeneracy_triplet_0}, expected 1")
        return (True, "Degeneracies match expectation")

    # ---------- Monotonicity tests (explicit perturbation of a physical parameter) ----------
    def test_monotonicity_M_increases_decreases_eigenvalue(self):
        """
        For triplet (S=1), baseline: M=0 -> eigenvalue = triplet_0.
        Perturb input M to maximize (M=±1) -> eigenvalue = triplet_pm.
        Verify that increasing |M| decreases the eigenvalue.
        """
        if self.eigenvalue_triplet_0 is None or self.eigenvalue_triplet_pm is None:
            return (False, "SKIP: Missing triplet eigenvalues")
        # Baseline: M=0 eigenvalue
        baseline_val = self.eigenvalue_triplet_0
        # Perturbed: M=±1 eigenvalue
        perturbed_val = self.eigenvalue_triplet_pm
        # For this system (S=1, formula gives eigenvalue = (2 - 3M^2)/2),
        # increasing |M| from 0 to 1 decreases the eigenvalue.
        if not (perturbed_val < baseline_val):
            return (False, f"Perturbing M from 0 to ±1 should decrease eigenvalue: baseline {baseline_val}, perturbed {perturbed_val}")
        return (True, f"Baseline M=0 {baseline_val}, perturbed M=±1 {perturbed_val}: value decreased as expected")

    def test_monotonicity_S_increases_increases_eigenvalue(self):
        """
        For fixed M=0, baseline: S=0 (singlet) -> eigenvalue = singlet.
        Perturb input S to S=1 (triplet0) -> eigenvalue = triplet_0.
        Verify that increasing S increases the eigenvalue.
        """
        if self.eigenvalue_singlet is None or self.eigenvalue_triplet_0 is None:
            return (False, "SKIP: Missing singlet/triplet0 eigenvalues")
        baseline_val = self.eigenvalue_singlet
        perturbed_val = self.eigenvalue_triplet_0
        # For M=0, formula gives eigenvalue = S(S+1)/2. So larger S gives larger value.
        if not (perturbed_val > baseline_val):
            return (False, f"Perturbing S from 0 to 1 should increase eigenvalue: baseline {baseline_val}, perturbed {perturbed_val}")
        return (True, f"Baseline S=0 {baseline_val}, perturbed S=1 {perturbed_val}: value increased as expected")

    # ---------- Weak answer sanity (only one) ----------
    def test_answer_sanity_broad_range(self):
        """Eigenvalues must be finite and within a reasonable broad range (no precise matching)."""
        if any(x is None for x in [self.eigenvalue_singlet, self.eigenvalue_triplet_pm, self.eigenvalue_triplet_0]):
            return (False, "SKIP: Missing eigenvalues")
        vals = [self.eigenvalue_singlet, self.eigenvalue_triplet_pm, self.eigenvalue_triplet_0]
        for v in vals:
            if not math.isfinite(v):
                return (False, f"Eigenvalue {v} is not finite")
            if v < -1 or v > 2:  # expect -0.5, 0, 1 so within [-1,2]
                return (False, f"Eigenvalue {v} outside broad range [-1,2]")
        return (True, "Eigenvalues finite and within broad range")

    def run_all_tests(self):
        """Discover and run all test_ methods, returning (success, logs)."""
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