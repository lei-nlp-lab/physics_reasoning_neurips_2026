import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {}) or {}
        # Expected keys from pseudocode emit contract
        self.expected_keys = [
            'lambda_nm_1', 'lambda_A_angstrom_1',
            'lambda_nm_2', 'lambda_A_angstrom_2',
            'lambda_nm_3', 'lambda_A_angstrom_3'
        ]
        # Predefine physical constants (derived from problem givens)
        self.d_nm = 4000.0          # grating spacing in nm (d = 1/250 mm = 4000 nm)
        self.sin_theta = 0.5        # sin(30°)
        self.visible_min_nm = 380.0
        self.visible_max_nm = 750.0
        # Tolerance for numerical checks (lenient for physics invariants)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def _get(self, key, default=None):
        """Safely retrieve a value from results."""
        return self.results.get(key, default)

    # ---------- Interface Tests ----------
    def test_interface_required_keys_exist(self):
        """All required keys must be present in results."""
        missing = [key for key in self.expected_keys if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_values_not_none(self):
        """Each key must have a non-None value."""
        none_keys = [key for key in self.expected_keys if self._get(key) is None]
        if none_keys:
            return (False, f"Keys with None value: {none_keys}")
        return (True, "All values are non-None")

    def test_interface_values_numeric(self):
        """Values must be numeric (int, float, or sympy rational)."""
        bad_keys = []
        for key in self.expected_keys:
            val = self._get(key)
            if val is None:
                continue
            if not isinstance(val, (int, float, sp.Number)):
                bad_keys.append(key)
        if bad_keys:
            return (False, f"Non-numeric values for keys: {bad_keys}")
        return (True, "All values are numeric type")

    # ---------- Invariant Tests (theorem-based) ----------
    def test_invariant_grating_equation_m3(self):
        """Check d*sinθ = m*λ for order m=3 (lambda_nm_3)."""
        lam = self._get('lambda_nm_3')
        if lam is None:
            return (False, "SKIP: lambda_nm_3 not available")
        m = 3.0
        expected = self.d_nm * self.sin_theta  # 2000 nm
        actual = m * lam
        if math.isclose(actual, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return (True, f"d sinθ ≈ mλ for m=3: {actual} ≈ {expected}")
        return (False, f"d sinθ = {expected}, but m*λ = {actual} for m=3")

    def test_invariant_grating_equation_m4(self):
        """Check d*sinθ = m*λ for order m=4."""
        lam = self._get('lambda_nm_2')
        if lam is None:
            return (False, "SKIP: lambda_nm_2 not available")
        m = 4.0
        expected = self.d_nm * self.sin_theta
        actual = m * lam
        if math.isclose(actual, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return (True, f"d sinθ ≈ mλ for m=4: {actual} ≈ {expected}")
        return (False, f"d sinθ = {expected}, but m*λ = {actual} for m=4")

    def test_invariant_grating_equation_m5(self):
        """Check d*sinθ = m*λ for order m=5."""
        lam = self._get('lambda_nm_1')
        if lam is None:
            return (False, "SKIP: lambda_nm_1 not available")
        m = 5.0
        expected = self.d_nm * self.sin_theta
        actual = m * lam
        if math.isclose(actual, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return (True, f"d sinθ ≈ mλ for m=5: {actual} ≈ {expected}")
        return (False, f"d sinθ = {expected}, but m*λ = {actual} for m=5")

    def test_invariant_visible_range(self):
        """All wavelengths in nm must be within visible range (380–750 nm)."""
        nm_keys = ['lambda_nm_1', 'lambda_nm_2', 'lambda_nm_3']
        out_of_range = []
        for key in nm_keys:
            lam = self._get(key)
            if lam is None:
                out_of_range.append(f"{key}: missing")
                continue
            if lam < self.visible_min_nm or lam > self.visible_max_nm:
                out_of_range.append(f"{key}={lam} nm")
        if out_of_range:
            return (False, f"Wavelength(s) outside visible range: {out_of_range}")
        return (True, "All wavelengths in visible range")

    def test_invariant_order_integers(self):
        """Check that the deduced m = 2000/λ_nm is close to integer (3,4,5)."""
        nm_keys_m = [('lambda_nm_1', 5), ('lambda_nm_2', 4), ('lambda_nm_3', 3)]
        bad = []
        for key, expected_m in nm_keys_m:
            lam = self._get(key)
            if lam is None:
                bad.append(f"{key}: missing")
                continue
            m_calc = 2000.0 / lam  # because d sinθ = 2000 nm
            if not math.isclose(m_calc, float(expected_m), rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                bad.append(f"{key} gives m={m_calc:.2f} (expected {expected_m})")
        if bad:
            return (False, f"Order integer check failed: {bad}")
        return (True, "All orders correspond to integer m")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_conversion_nm_to_angstrom(self):
        """Check that λ_angstrom = λ_nm * 10 for each order."""
        errors = []
        for i in range(1, 4):
            nm_key = f'lambda_nm_{i}'
            a_key = f'lambda_A_angstrom_{i}'
            lam_nm = self._get(nm_key)
            lam_A = self._get(a_key)
            if lam_nm is None or lam_A is None:
                errors.append(f"Missing {nm_key} or {a_key}")
                continue
            expected_A = lam_nm * 10.0
            if not math.isclose(lam_A, expected_A, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                errors.append(f"{a_key}={lam_A} vs expected {expected_A} from {nm_key}={lam_nm}")
        if errors:
            return (False, f"Conversion errors: {errors}")
        return (True, "All nm-to-Å conversions consistent")

    def test_modeling_grating_constant_consistency(self):
        """
        Verify that for each wavelength, the product m*λ is constant and equal to d*sinθ (=2000 nm).
        This tests the core modeling step: λ = 2000/m.
        """
        pairs = [('lambda_nm_1', 5), ('lambda_nm_2', 4), ('lambda_nm_3', 3)]
        errors = []
        ref_product = self.d_nm * self.sin_theta  # 2000 nm
        for key, m in pairs:
            lam = self._get(key)
            if lam is None:
                errors.append(f"Missing {key}")
                continue
            product = m * lam
            if not math.isclose(product, ref_product, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                errors.append(f"{key} (m={m}): m*λ = {product} ≠ {ref_product}")
        if errors:
            return (False, f"Grating constant inconsistency: {errors}")
        return (True, "All m*λ products match d*sinθ")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_wavelength_decreases_with_increasing_m(self):
        """
        Perturb the diffraction order m from 3 to 4 to 5.
        Verify that λ decreases as m increases (i.e., λ ∝ 1/m).
        This uses the discrete perturbation of the integer input m.
        """
        lam3 = self._get('lambda_nm_3')  # m=3
        lam4 = self._get('lambda_nm_2')  # m=4
        lam5 = self._get('lambda_nm_1')  # m=5
        if any(lam is None for lam in (lam3, lam4, lam5)):
            return (False, "SKIP: missing wavelengths for monotonicity check")
        # Baseline: λ for m=3 is the longest. As we perturb m upward, λ should decrease.
        if lam3 <= lam4 or lam4 <= lam5:
            return (False, f"Monotonicity fails: λ(m=3)={lam3}, λ(m=4)={lam4}, λ(m=5)={lam5}; expected strictly decreasing")
        # Further check: the ratio should be approximately m_high/m_low
        if not math.isclose(lam3 / lam4, 4.0/3.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return (False, f"Ratio λ(m=3)/λ(m=4) = {lam3/lam4:.4f} not close to 4/3 = 1.3333")
        if not math.isclose(lam4 / lam5, 5.0/4.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return (False, f"Ratio λ(m=4)/λ(m=5) = {lam4/lam5:.4f} not close to 5/4 = 1.25")
        return (True, "Monotonicity and inverse proportionality to m verified")

    # ---------- Answer Sanity Tests (weak, at most one) ----------
    def test_answer_sanity_wavelengths_valid(self):
        """
        Weak sanity: all wavelengths must be positive, finite, and not NaN.
        No hardcoded expected values; just basic validity.
        """
        for key in self.expected_keys:
            val = self._get(key)
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                return (False, f"NaN value for {key}")
            if not (isinstance(val, (int, float, sp.Number)) and val > 0):
                return (False, f"Non-positive or invalid value for {key}: {val}")
        return (True, "All wavelengths are positive and not NaN")

    # ---------- Runner ----------
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