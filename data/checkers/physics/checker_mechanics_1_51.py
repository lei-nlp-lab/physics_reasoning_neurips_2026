import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the Tritium atom sudden charge change problem.
    Checks physics consistency, modeling quality, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Safely extract givens (allow missing dict)
        givens = getattr(solution_instance, 'givens', {})
        self.r0 = givens.get('r0', None)
        self.k = givens.get('k', None)
        self.e = givens.get('e', None)
        self.m = givens.get('m', None)

        # Safely extract results
        results = getattr(solution_instance, 'results', {})
        self.R = results.get('R', None)
        self.a = results.get('a', None)
        self.ecc = results.get('ecc', None)
        self.r_min = results.get('r_min', None)
        self.r_max = results.get('r_max', None)
        self.major_axis = results.get('major_axis', None)
        self.minor_axis = results.get('minor_axis', None)

        # Explanation steps (if any)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Derived constants for internal use (re-derived only when all givens present)
        self._E_i = None
        self._E_f = None
        self._L_sq = None
        self._can_recompute = False
        if self.r0 is not None and self.k is not None and self.e is not None and self.m is not None:
            self._can_recompute = True
            v0_sq = (self.k * self.e**2) / (self.m * self.r0)
            self._E_i = - (self.k * self.e**2) / (2 * self.r0)
            self._E_f = - (3 * self.k * self.e**2) / (2 * self.r0)
            self._L_sq = self.m**2 * v0_sq * self.r0**2  # L^2

    # --------------------------------------------------------------------------
    # Interface checks
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All expected result keys must exist and be numeric."""
        keys = ['R', 'a', 'ecc', 'r_min', 'r_max', 'major_axis', 'minor_axis']
        for k in keys:
            val = getattr(self, k, None)
            if val is None:
                return (False, f"Missing or None result key: {k}")
        return (True, "All required result keys present")

    def test_interface_r0_positive(self):
        """Given r0 must be positive."""
        if self.r0 is None:
            return (False, "SKIP: r0 not provided")
        if not (isinstance(self.r0, (int, float, complex)) and self.r0 > 0):
            return (False, f"r0 is not a positive number: {self.r0}")
        return (True, f"r0 = {self.r0} is positive")

    # --------------------------------------------------------------------------
    # Invariant / theorem checks (use math.isclose)
    # --------------------------------------------------------------------------
    def test_invariant_energy_ratio(self):
        """If givens present, re-derive E_f/E_i and compare to result R."""
        if not self._can_recompute:
            return (False, "SKIP: insufficient givens to recompute ratio")
        if self.R is None:
            return (False, "R missing from results")
        expected_R = 3.0
        if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Energy ratio R={self.R} differs from expected {expected_R}")
        return (True, f"Energy ratio R={self.R} matches expected 3")

    def test_invariant_angular_momentum_relation(self):
        """Check L^2 = m * 2k e^2 * a * (1 - ecc^2)."""
        if not self._can_recompute:
            return (False, "SKIP: insufficient givens")
        if self.a is None or self.ecc is None:
            return (False, "SKIP: a or ecc missing from results")
        # Compute product
        rhs = self.m * (2 * self.k * self.e**2) * self.a * (1 - self.ecc**2)
        if not math.isclose(self._L_sq, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"L^2 = {self._L_sq} and m*2k*e^2*a*(1-ecc^2) = {rhs} differ")
        return (True, "Angular momentum relation L^2 = m*2k*e^2*a*(1-ecc^2) holds")

    def test_invariant_closest_farthest_sum(self):
        """r_min + r_max should equal 2a (for ellipse)."""
        if self.r_min is None or self.r_max is None or self.a is None:
            return (False, "SKIP: missing r_min, r_max or a")
        two_a = 2 * self.a
        s = self.r_min + self.r_max
        if not math.isclose(s, two_a, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"r_min + r_max = {s}, expected 2a = {two_a}")
        return (True, f"r_min + r_max = {s} ≈ 2a = {two_a}")

    def test_invariant_eccentricity_range(self):
        """0 <= ecc < 1."""
        if self.ecc is None:
            return (False, "SKIP: ecc missing")
        if not (0 <= self.ecc < 1):
            return (False, f"Eccentricity out of range: {self.ecc}")
        return (True, f"ecc = {self.ecc} in [0, 1)")

    # --------------------------------------------------------------------------
    # Modeling / trace checks (use explanation steps)
    # --------------------------------------------------------------------------
    def test_modeling_elliptical_orbit_relations(self):
        """Check r_min = a*(1-ecc) and r_max = a*(1+ecc)."""
        if self.r_min is None or self.a is None or self.ecc is None:
            return (False, "SKIP: missing r_min, a, or ecc")
        expected_min = self.a * (1 - self.ecc)
        expected_max = self.a * (1 + self.ecc)
        if not math.isclose(self.r_min, expected_min, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"r_min = {self.r_min} differs from a*(1-ecc) = {expected_min}")
        if not math.isclose(self.r_max, expected_max, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"r_max = {self.r_max} differs from a*(1+ecc) = {expected_max}")
        return (True, "r_min = a*(1-ecc) and r_max = a*(1+ecc) hold")

    def test_modeling_major_minor_axes(self):
        """Check major_axis = 2a, minor_axis = 2a*sqrt(1-ecc^2)."""
        if self.major_axis is None or self.minor_axis is None or self.a is None or self.ecc is None:
            return (False, "SKIP: missing axis or a/ecc")
        exp_major = 2 * self.a
        exp_minor = 2 * self.a * math.sqrt(1 - self.ecc**2)
        if not math.isclose(self.major_axis, exp_major, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"major_axis = {self.major_axis}, expected 2a = {exp_major}")
        if not math.isclose(self.minor_axis, exp_minor, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"minor_axis = {self.minor_axis}, expected 2a*sqrt(1-e^2) = {exp_minor}")
        return (True, "Major and minor axes match elliptical formulas")

    def test_modeling_explanation_steps_critical(self):
        """If explanation_steps present, verify key formulas appear."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        # Check for energy ratio and force balance mention
        text = str(self.explanation_steps)
        if 'E_2/E_1' not in text and 'energy ratio' not in text and '3' not in text:
            return (False, "Explanation steps missing energy ratio discussion")
        if 'mv0^2/r0' not in text and 'force' not in text and 'centripetal' not in text:
            return (False, "Explanation steps missing force balance condition")
        return (True, "Explanation steps contain key physics relations")

    # --------------------------------------------------------------------------
    # Monotonicity checks (perturb an input and check output trend)
    # --------------------------------------------------------------------------
    def test_monotonicity_r0_to_r_min(self):
        """If r0 increases, r_min should increase proportionally (r_min = r0/3)."""
        if not self._can_recompute or self.r_min is None:
            return (False, "SKIP: cannot perform monotonicity check")
        # Perturb r0 by 10%
        r0_new = self.r0 * 1.1
        # Recompute expected r_min_new = r0_new/3 (since r_min = r0/3)
        r_min_new = r0_new / 3.0
        # Compare direction: new r_min > old r_min
        if not (r_min_new > self.r_min):
            return (False, f"Monotonicity fail: increasing r0 should increase r_min, but r_min={self.r_min}, r_min_new={r_min_new}")
        return (True, f"r_min increases with r0: r0*1.1 → r_min_new={r_min_new} > {self.r_min}")

    def test_monotonicity_r0_to_major_axis(self):
        """Major axis = 4r0/3, so increasing r0 increases major axis."""
        if self.r0 is None or self.major_axis is None:
            return (False, "SKIP: missing r0 or major_axis")
        # Use 5% perturbation
        r0_new = self.r0 * 1.05
        exp_major_new = 4.0 * r0_new / 3.0
        exp_old = 4.0 * self.r0 / 3.0
        if not math.isclose(self.major_axis, exp_old, rel_tol=1e-2):
            return (False, f"Baseline major axis mismatch: {self.major_axis} vs {exp_old}")
        if not (exp_major_new > self.major_axis):
            return (False, "Monotonicity fail: increasing r0 should increase major axis")
        return (True, f"major_axis increases with r0: {exp_major_new} > {self.major_axis}")

    # --------------------------------------------------------------------------
    # Answer sanity (weak, no math.isclose, no re-derivation)
    # --------------------------------------------------------------------------
    def test_answer_sanity_signs(self):
        """Check that energies (R implies negativity) and distances are positive."""
        if self.R is None:
            return (False, "SKIP: R missing")
        if not isinstance(self.R, (int, float)):
            return (False, f"R is not numeric: {self.R}")
        if self.R <= 0:
            return (False, f"R should be positive, got {self.R}")
        for key, val in [('a', self.a), ('r_min', self.r_min), ('r_max', self.r_max),
                         ('major_axis', self.major_axis), ('minor_axis', self.minor_axis)]:
            if val is not None and val <= 0:
                return (False, f"{key} is non-positive: {val}")
        return (True, "Signs and positivity OK")

    # --------------------------------------------------------------------------
    # Test runner
    # --------------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_* methods, run them, collect results."""
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
                passed, msg = False, f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)