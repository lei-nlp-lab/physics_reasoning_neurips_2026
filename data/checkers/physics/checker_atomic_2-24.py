import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the semiempirical mass formula problem.
    Checks coefficients, signs, magnitudes, and physical monotonicity trends.
    """

    def __init__(self, solution_instance):
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.a_v = self.results.get('a_v', None)
        self.a_s = self.results.get('a_s', None)
        self.a_e = self.results.get('a_e', None)
        self.a_a = self.results.get('a_a', None)
        self.a_p = self.results.get('a_p', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        required = ['a_v', 'a_s', 'a_e', 'a_a', 'a_p']
        missing = [k for k in required if k not in self.results]
        non_numeric = []
        for k in required:
            val = self.results.get(k, None)
            if val is not None and not isinstance(val, (int, float)):
                non_numeric.append(k)
        if missing:
            return (False, f"Missing keys: {missing}")
        if non_numeric:
            return (False, f"Non-numeric values for keys: {non_numeric}")
        return (True, "All required keys present and numeric")

    def test_interface_a_v_exists(self):
        if self.a_v is None:
            return (False, "SKIP: a_v not available")
        return (True, "a_v exists")

    def test_interface_a_s_exists(self):
        if self.a_s is None:
            return (False, "SKIP: a_s not available")
        return (True, "a_s exists")

    def test_interface_a_e_exists(self):
        if self.a_e is None:
            return (False, "SKIP: a_e not available")
        return (True, "a_e exists")

    def test_interface_a_a_exists(self):
        if self.a_a is None:
            return (False, "SKIP: a_a not available")
        return (True, "a_a exists")

    def test_interface_a_p_exists(self):
        if self.a_p is None:
            return (False, "SKIP: a_p not available")
        return (True, "a_p exists")

    # ------------------------------------------------------------------
    # Invariant tests (signs, basic inequalities)
    # ------------------------------------------------------------------
    def test_invariant_a_v_positive(self):
        if self.a_v is None:
            return (False, "SKIP: a_v not available")
        return (True, f"a_v = {self.a_v} > 0") if self.a_v > 0 else (False, f"a_v = {self.a_v} <= 0")

    def test_invariant_a_s_positive(self):
        if self.a_s is None:
            return (False, "SKIP: a_s not available")
        return (True, f"a_s = {self.a_s} > 0") if self.a_s > 0 else (False, f"a_s = {self.a_s} <= 0")

    def test_invariant_a_e_positive(self):
        if self.a_e is None:
            return (False, "SKIP: a_e not available")
        return (True, f"a_e = {self.a_e} > 0") if self.a_e > 0 else (False, f"a_e = {self.a_e} <= 0")

    def test_invariant_a_a_positive(self):
        if self.a_a is None:
            return (False, "SKIP: a_a not available")
        return (True, f"a_a = {self.a_a} > 0") if self.a_a > 0 else (False, f"a_a = {self.a_a} <= 0")

    def test_invariant_a_p_positive(self):
        if self.a_p is None:
            return (False, "SKIP: a_p not available")
        return (True, f"a_p = {self.a_p} > 0") if self.a_p > 0 else (False, f"a_p = {self.a_p} <= 0")

    def test_invariant_volume_greater_than_surface(self):
        if self.a_v is None or self.a_s is None:
            return (False, "SKIP: a_v or a_s not available")
        if self.a_v > self.a_s:
            return (True, f"a_v ({self.a_v}) > a_s ({self.a_s})")
        else:
            return (False, f"a_v <= a_s: {self.a_v} <= {self.a_s}")

    def test_invariant_surface_greater_than_coulomb(self):
        if self.a_s is None or self.a_e is None:
            return (False, "SKIP: a_s or a_e not available")
        if self.a_s > self.a_e:
            return (True, f"a_s ({self.a_s}) > a_e ({self.a_e})")
        else:
            return (False, f"a_s <= a_e: {self.a_s} <= {self.a_e}")

    def test_invariant_asymmetry_greater_than_volume(self):
        if self.a_a is None or self.a_v is None:
            return (False, "SKIP: a_a or a_v not available")
        if self.a_a > self.a_v:
            return (True, f"a_a ({self.a_a}) > a_v ({self.a_v})")
        else:
            return (False, f"a_a <= a_v: {self.a_a} <= {self.a_v}")

    def test_invariant_pairing_reasonable_magnitude(self):
        if self.a_p is None:
            return (False, "SKIP: a_p not available")
        if 5.0 <= self.a_p <= 20.0:
            return (True, f"a_p = {self.a_p} in [5,20]")
        else:
            return (False, f"a_p = {self.a_p} outside typical range [5,20]")

    # ------------------------------------------------------------------
    # Modeling tests (check coefficients against typical ranges)
    # ------------------------------------------------------------------
    def test_modeling_a_v_range(self):
        if self.a_v is None:
            return (False, "SKIP: a_v not available")
        if 15.0 <= self.a_v <= 16.0:
            return (True, f"a_v = {self.a_v} within [15,16]")
        return (False, f"a_v = {self.a_v} outside [15,16]")

    def test_modeling_a_s_range(self):
        if self.a_s is None:
            return (False, "SKIP: a_s not available")
        if 17.0 <= self.a_s <= 19.0:
            return (True, f"a_s = {self.a_s} within [17,19]")
        return (False, f"a_s = {self.a_s} outside [17,19]")

    def test_modeling_a_e_range(self):
        if self.a_e is None:
            return (False, "SKIP: a_e not available")
        if 0.7 <= self.a_e <= 0.8:
            return (True, f"a_e = {self.a_e} within [0.7,0.8]")
        return (False, f"a_e = {self.a_e} outside [0.7,0.8]")

    def test_modeling_a_a_range(self):
        if self.a_a is None:
            return (False, "SKIP: a_a not available")
        if 90.0 <= self.a_a <= 95.0:
            return (True, f"a_a = {self.a_a} within [90,95]")
        return (False, f"a_a = {self.a_a} outside [90,95]")

    def test_modeling_a_p_range(self):
        if self.a_p is None:
            return (False, "SKIP: a_p not available")
        if 10.0 <= self.a_p <= 12.0:
            return (True, f"a_p = {self.a_p} within [10,12]")
        return (False, f"a_p = {self.a_p} outside [10,12]")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation with percentage change)
    # ------------------------------------------------------------------
    def test_monotonicity_coulomb_increases_with_Z(self):
        """For fixed A, increasing Z by 10% makes Coulomb term more negative."""
        if self.a_e is None:
            return (False, "SKIP: a_e not available")
        A = 100.0
        Z_baseline = 40.0
        Z_perturbed = Z_baseline * 1.1   # 10% increase
        coulomb_baseline = -self.a_e * Z_baseline**2 / (A**(1.0/3.0))
        coulomb_perturbed = -self.a_e * Z_perturbed**2 / (A**(1.0/3.0))
        if coulomb_perturbed < coulomb_baseline:
            return (True, f"Coulomb becomes more negative: base={coulomb_baseline:.3f}, perturbed={coulomb_perturbed:.3f}")
        else:
            return (False, f"Coulomb did not become more negative: base={coulomb_baseline:.3f}, perturbed={coulomb_perturbed:.3f}")

    def test_monotonicity_asymmetry_penalty_increases_with_deviation(self):
        """For fixed A, increasing deviation from Z=A/2 by 10% makes asymmetry term more negative."""
        if self.a_a is None:
            return (False, "SKIP: a_a not available")
        A = 100.0
        Z_center = 50.0
        # baseline: at center (no deviation) -> asymmetry = 0
        Z_baseline = Z_center
        # perturb: move Z away by 10% of center value (i.e., 5)
        Z_perturbed = Z_center * 0.9   # 10% reduction in Z (move left)
        asym_baseline = -self.a_a * (A - 2*Z_baseline)**2 / A
        asym_perturbed = -self.a_a * (A - 2*Z_perturbed)**2 / A
        if asym_perturbed < asym_baseline:
            return (True, f"Asymmetry becomes more negative: base={asym_baseline:.3f}, perturbed={asym_perturbed:.3f}")
        else:
            return (False, f"Asymmetry did not become more negative: base={asym_baseline:.3f}, perturbed={asym_perturbed:.3f}")

    def test_monotonicity_pairing_magnitude_decreases_with_A(self):
        """For fixed pairing coefficient, increasing A by 10% reduces pairing magnitude."""
        if self.a_p is None:
            return (False, "SKIP: a_p not available")
        A_baseline = 100.0
        A_perturbed = A_baseline * 1.1   # 10% increase
        mag_baseline = self.a_p / math.sqrt(A_baseline)
        mag_perturbed = self.a_p / math.sqrt(A_perturbed)
        if mag_perturbed < mag_baseline:
            return (True, f"Pairing magnitude decreases: base={mag_baseline:.4f}, perturbed={mag_perturbed:.4f}")
        else:
            return (False, f"Pairing magnitude did not decrease: base={mag_baseline:.4f}, perturbed={mag_perturbed:.4f}")

    def test_monotonicity_volume_linear_in_A(self):
        """For fixed coefficients, increasing A by 10% increases volume term."""
        if self.a_v is None:
            return (False, "SKIP: a_v not available")
        A_baseline = 50.0
        A_perturbed = A_baseline * 1.1   # 10% increase
        vol_baseline = self.a_v * A_baseline
        vol_perturbed = self.a_v * A_perturbed
        if vol_perturbed > vol_baseline:
            return (True, f"Volume term increases: base={vol_baseline:.2f}, perturbed={vol_perturbed:.2f}")
        else:
            return (False, f"Volume term did not increase: base={vol_baseline:.2f}, perturbed={vol_perturbed:.2f}")

    # ------------------------------------------------------------------
    # Answer sanity (weak check – only one method)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs(self):
        """All coefficients must be positive (sign check)."""
        coeffs = {'a_v': self.a_v, 'a_s': self.a_s, 'a_e': self.a_e,
                  'a_a': self.a_a, 'a_p': self.a_p}
        for name, val in coeffs.items():
            if val is None:
                return (False, f"SKIP: {name} not available")
            if val <= 0:
                return (False, f"{name} = {val} is not positive")
        return (True, "All five coefficients are positive")

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