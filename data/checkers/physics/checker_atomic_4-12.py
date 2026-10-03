import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the threshold energy problem for particle production
    in proton-proton collisions.
    """

    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.m = givens.get('m', None)          # proton rest mass
        self.M = givens.get('M', None)          # produced particle rest mass

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.E1 = results.get('E1', None)       # fixed‑target energy
        self.E_coll = results.get('E_coll', None)  # collider energy

        # Optional metadata
        self.explanation = getattr(solution_instance, 'explanation_steps', None)

    # -----------------------------------------------------------------
    # Interface checks
    # -----------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.E1 is None:
            return (False, "Missing required result 'E1'")
        if self.E_coll is None:
            return (False, "Missing required result 'E_coll'")
        return (True, "Both required result keys present")

    def test_interface_givens_available(self):
        if self.m is None:
            return (False, "Missing given 'm' (proton rest mass)")
        if self.M is None:
            return (False, "Missing given 'M' (produced particle rest mass)")
        return (True, "Givens 'm' and 'M' available")

    def test_interface_givens_positive(self):
        if self.m is None or self.M is None:
            return (False, "SKIP: givens not available to check positivity")
        if self.m <= 0:
            return (False, f"Proton mass m={self.m} must be positive")
        if self.M <= 0:
            return (False, f"Produced mass M={self.M} must be positive")
        return (True, "Rest masses are positive")

    # -----------------------------------------------------------------
    # Theorem‑based invariant checks (physics sanity)
    # -----------------------------------------------------------------
    def test_invariant_threshold_fixed_target(self):
        """Check that s = 2m^2 + 2E1*m matches (2m+M)^2."""
        if any(x is None for x in (self.m, self.M, self.E1)):
            return (False, "SKIP: missing m, M or E1")
        s_computed = 2 * self.m**2 + 2 * self.E1 * self.m
        s_threshold = (2 * self.m + self.M) ** 2
        if math.isclose(s_computed, s_threshold, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s_fixed = {s_computed:.4e}, s_threshold = {s_threshold:.4e}")
        else:
            return (False, f"s_fixed = {s_computed:.4e}, s_threshold = {s_threshold:.4e}, "
                           f"difference too large")

    def test_invariant_threshold_collider(self):
        """Check that s = 4*E_coll^2 matches (2m+M)^2."""
        if any(x is None for x in (self.m, self.M, self.E_coll)):
            return (False, "SKIP: missing m, M or E_coll")
        s_computed = 4 * self.E_coll ** 2
        s_threshold = (2 * self.m + self.M) ** 2
        if math.isclose(s_computed, s_threshold, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s_collider = {s_computed:.4e}, s_threshold = {s_threshold:.4e}")
        else:
            return (False, f"s_collider = {s_computed:.4e}, s_threshold = {s_threshold:.4e}, "
                           f"difference too large")

    def test_invariant_energy_above_mass(self):
        """Threshold energies must be > rest mass (proton)."""
        if self.E1 is not None and self.m is not None:
            if not self.E1 > self.m:
                return (False, f"Fixed‑target E1={self.E1} must be > m={self.m}")
        if self.E_coll is not None and self.m is not None:
            if not self.E_coll > self.m:
                return (False, f"Collider E_coll={self.E_coll} must be > m={self.m}")
        return (True, "Energies are above rest‑mass threshold")

    # -----------------------------------------------------------------
    # Modeling/trace checks (re‑derived from theorem)
    # -----------------------------------------------------------------
    def test_modeling_fixed_target_formula(self):
        """
        Check that E1 = m + 2M + M^2/(2m), derived from threshold condition.
        """
        if any(x is None for x in (self.m, self.M, self.E1)):
            return (False, "SKIP: missing m, M or E1")
        expected_E1 = self.m + 2 * self.M + self.M**2 / (2 * self.m)
        if math.isclose(self.E1, expected_E1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E1 = {self.E1:.4e}, expected = {expected_E1:.4e}")
        else:
            return (False, f"E1 = {self.E1:.4e}, expected = {expected_E1:.4e}, "
                           f"deviation > 1%")

    def test_modeling_collider_formula(self):
        """
        Check that E_coll = m + M/2, derived from threshold condition.
        """
        if any(x is None for x in (self.m, self.M, self.E_coll)):
            return (False, "SKIP: missing m, M or E_coll")
        expected_E_coll = self.m + self.M / 2
        if math.isclose(self.E_coll, expected_E_coll, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_coll = {self.E_coll:.4e}, expected = {expected_E_coll:.4e}")
        else:
            return (False, f"E_coll = {self.E_coll:.4e}, expected = {expected_E_coll:.4e}, "
                           f"deviation > 1%")

    # -----------------------------------------------------------------
    # Weak answer sanity (only one method)
    # -----------------------------------------------------------------
    def test_answer_sanity_finite_positive_and_ordering(self):
        """
        Check that energies are finite positive numbers and that
        fixed‑target energy > collider energy.
        """
        # Check finite and positive
        for name, val in [('E1', self.E1), ('E_coll', self.E_coll)]:
            if val is None:
                continue
            if val <= 0 or not math.isfinite(val):
                return (False, f"{name} = {val} is not a finite positive number")
        # Check ordering
        if self.E1 is not None and self.E_coll is not None:
            if self.E1 <= self.E_coll:
                return (False, f"Fixed‑target E1={self.E1} should be > collider E_coll={self.E_coll}")
        return (True, "Energies finite, positive, and E1 > E_coll")

    # -----------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of input)
    # -----------------------------------------------------------------
    def test_monotonicity_fixed_target_with_M(self):
        """
        If M increases by 10%, the fixed‑target energy should increase.
        We compute the expected E1 using the analytic formula.
        """
        if any(x is None for x in (self.m, self.M)):
            return (False, "SKIP: missing m or M")
        # baseline
        E1_base = self.m + 2*self.M + self.M**2/(2*self.m)
        # perturb M upward by 10%
        M_prime = self.M * 1.1
        E1_prime = self.m + 2*M_prime + M_prime**2/(2*self.m)
        if E1_prime > E1_base:
            return (True, f"10% increase in M: E1={E1_base:.4e} -> {E1_prime:.4e} (increased)")
        else:
            return (False, f"Monotonicity violated: E1 should increase with M, "
                           f"but {E1_prime:.4e} <= {E1_base:.4e}")

    def test_monotonicity_collider_with_M(self):
        """
        If M increases by 10%, the collider energy should increase.
        """
        if any(x is None for x in (self.m, self.M)):
            return (False, "SKIP: missing m or M")
        E_coll_base = self.m + self.M/2
        M_prime = self.M * 1.1
        E_coll_prime = self.m + M_prime/2
        if E_coll_prime > E_coll_base:
            return (True, f"10% increase in M: E_coll={E_coll_base:.4e} -> {E_coll_prime:.4e} (increased)")
        else:
            return (False, f"Monotonicity violated: E_coll should increase with M, "
                           f"but {E_coll_prime:.4e} <= {E_coll_base:.4e}")

    # -----------------------------------------------------------------
    # Runner
    # -----------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e} {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)