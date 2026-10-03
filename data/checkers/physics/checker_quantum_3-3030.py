import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all required quantities from solution_instance.
        # Prefer the 'results' dict if present; fall back to direct attributes.
        # Use None for missing values — never numeric defaults.
        results = getattr(solution_instance, 'results', {})

        # Clebsch-Gordan coefficient for j=3/2, m=3/2
        self.cg13 = results.get('cg13', getattr(solution_instance, 'cg13', None))

        # Coefficients for j=3/2, m=1/2
        self.a = results.get('a', getattr(solution_instance, 'a', None))
        self.b = results.get('b', getattr(solution_instance, 'b', None))

        # Relative rates for I=1 resonance (K- p channel)
        self.rate_Kp_Sigma_minus = results.get('rate_Kp_Sigma-',
                                               getattr(solution_instance, 'rate_Kp_Sigma-', None))
        self.rate_Kp_Sigma_plus = results.get('rate_Kp_Sigma+',
                                              getattr(solution_instance, 'rate_Kp_Sigma+', None))
        self.rate_Kp_Sigma0 = results.get('rate_Kp_Sigma0',
                                          getattr(solution_instance, 'rate_Kp_Sigma0', None))

        # Relative rates for I=1 resonance (K- n channel)
        self.rate_Kn_Sigma_minus = results.get('rate_Kn_Sigma-',
                                               getattr(solution_instance, 'rate_Kn_Sigma-', None))
        self.rate_Kn_Sigma0 = results.get('rate_Kn_Sigma0',
                                          getattr(solution_instance, 'rate_Kn_Sigma0', None))

        # Rate string for I=0 resonance (K- p only)
        self.rate_Kp_all = results.get('rate_Kp_all',
                                       getattr(solution_instance, 'rate_Kp_all', None))

        # Additional derived numeric quantity: the squared sum a^2 + b^2 (if both available)
        self._squared_sum = None
        if self.a is not None and self.b is not None:
            try:
                self._squared_sum = float(self.a)**2 + float(self.b)**2
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Interface tests: check that required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_cg13_present(self):
        if self.cg13 is None:
            return (False, "SKIP: cg13 not available")
        if not isinstance(self.cg13, (int, float)):
            return (False, "cg13 should be numeric")
        return (True, f"cg13 = {self.cg13}")

    def test_interface_a_present(self):
        if self.a is None:
            return (False, "SKIP: a not available")
        if not isinstance(self.a, (int, float)):
            return (False, "a should be numeric")
        return (True, f"a = {self.a}")

    def test_interface_b_present(self):
        if self.b is None:
            return (False, "SKIP: b not available")
        if not isinstance(self.b, (int, float)):
            return (False, "b should be numeric")
        return (True, f"b = {self.b}")

    def test_interface_rate_Kp_Sigma_minus(self):
        if self.rate_Kp_Sigma_minus is None:
            return (False, "SKIP: rate_Kp_Sigma- not available")
        if not isinstance(self.rate_Kp_Sigma_minus, (int, float)):
            return (False, "rate_Kp_Sigma- should be numeric")
        return (True, f"rate_Kp_Sigma- = {self.rate_Kp_Sigma_minus}")

    def test_interface_rate_Kp_Sigma_plus(self):
        if self.rate_Kp_Sigma_plus is None:
            return (False, "SKIP: rate_Kp_Sigma+ not available")
        if not isinstance(self.rate_Kp_Sigma_plus, (int, float)):
            return (False, "rate_Kp_Sigma+ should be numeric")
        return (True, f"rate_Kp_Sigma+ = {self.rate_Kp_Sigma_plus}")

    def test_interface_rate_Kp_Sigma0(self):
        if self.rate_Kp_Sigma0 is None:
            return (False, "SKIP: rate_Kp_Sigma0 not available")
        if not isinstance(self.rate_Kp_Sigma0, (int, float)):
            return (False, "rate_Kp_Sigma0 should be numeric")
        return (True, f"rate_Kp_Sigma0 = {self.rate_Kp_Sigma0}")

    def test_interface_rate_Kn_Sigma_minus(self):
        if self.rate_Kn_Sigma_minus is None:
            return (False, "SKIP: rate_Kn_Sigma- not available")
        if not isinstance(self.rate_Kn_Sigma_minus, (int, float)):
            return (False, "rate_Kn_Sigma- should be numeric")
        return (True, f"rate_Kn_Sigma- = {self.rate_Kn_Sigma_minus}")

    def test_interface_rate_Kn_Sigma0(self):
        if self.rate_Kn_Sigma0 is None:
            return (False, "SKIP: rate_Kn_Sigma0 not available")
        if not isinstance(self.rate_Kn_Sigma0, (int, float)):
            return (False, "rate_Kn_Sigma0 should be numeric")
        return (True, f"rate_Kn_Sigma0 = {self.rate_Kn_Sigma0}")

    def test_interface_rate_Kp_all(self):
        if self.rate_Kp_all is None:
            return (False, "SKIP: rate_Kp_all not available")
        if not isinstance(self.rate_Kp_all, str):
            return (False, "rate_Kp_all should be a string")
        return (True, f"rate_Kp_all = {self.rate_Kp_all}")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_cg_orthonormality_j32_m12(self):
        """For j=3/2, m=1/2, the Clebsch-Gordan coefficients satisfy a^2 + b^2 = 1."""
        if self.a is None or self.b is None:
            return (False, "SKIP: a or b missing, cannot check orthonormality")
        try:
            total = float(self.a)**2 + float(self.b)**2
        except (TypeError, ValueError):
            return (False, "a or b are non-numeric")
        if math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"a^2 + b^2 = {total:.4f} ≈ 1")
        else:
            return (False, f"a^2 + b^2 = {total:.4f} ≠ 1")

    def test_invariant_cg_j32_m32_is_one(self):
        """The Clebsch-Gordan coefficient for j=3/2, m=3/2 must be exactly 1."""
        if self.cg13 is None:
            return (False, "SKIP: cg13 missing")
        # Exact value 1 is known from theory, not a numeric default.
        expected = 1.0
        if math.isclose(float(self.cg13), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"cg13 = {self.cg13}")
        else:
            return (False, f"cg13 = {self.cg13} ≠ 1")

    def test_invariant_rates_I1_squared_sum_Kp(self):
        """
        For I=1 resonance, K-p initial state is |1,0> (normalized).
        The squared magnitudes of the isospin Clebsch-Gordan coefficients
        for the three final states must sum to 1.
        From physics: |<Sigma- pi+|1,0>|^2 = 1/2, |<Sigma+ pi-|1,0>|^2 = 1/2,
        |<Sigma0 pi0|1,0>|^2 = 0 → sum = 1.
        The rates are proportional to these squares. We check that the sum of
        (rate / normalization) equals 1, where normalization = 2 (since rates are 1,1,0).
        """
        if any(x is None for x in [self.rate_Kp_Sigma_minus, self.rate_Kp_Sigma_plus, self.rate_Kp_Sigma0]):
            return (False, "SKIP: one of the Kp rates missing")
        try:
            r1 = float(self.rate_Kp_Sigma_minus)
            r2 = float(self.rate_Kp_Sigma_plus)
            r3 = float(self.rate_Kp_Sigma0)
        except (TypeError, ValueError):
            return (False, "One of the rates is non-numeric")
        # The theoretical squared CGs are 0.5, 0.5, 0, which sum to 1.
        # The rates given are 1,1,0; they are already proportional.
        # Check that the sum of rates is 2 (since 1+1+0=2).
        total = r1 + r2 + r3
        if math.isclose(total, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Sum of Kp rates = {total} (expected 2)")
        else:
            return (False, f"Sum of Kp rates = {total} ≠ 2")

    # ------------------------------------------------------------------
    # Modeling tests: check that specific coefficients and rate ratios
    # match the expected physics values (using math.isclose)
    # ------------------------------------------------------------------
    def test_modeling_cg_j32_m12_values(self):
        """Check the two CG coefficients for j=3/2, m=1/2: a = sqrt(2/3), b = sqrt(1/3)."""
        if self.a is None or self.b is None:
            return (False, "SKIP: a or b missing")
        expected_a = math.sqrt(2/3)
        expected_b = math.sqrt(1/3)
        try:
            a_val = float(self.a)
            b_val = float(self.b)
        except (TypeError, ValueError):
            return (False, "a or b non-numeric")
        a_ok = math.isclose(a_val, expected_a, rel_tol=1e-2, abs_tol=1e-2)
        b_ok = math.isclose(b_val, expected_b, rel_tol=1e-2, abs_tol=1e-2)
        if a_ok and b_ok:
            return (True, f"a ≈ {expected_a:.4f} (got {a_val:.4f}), b ≈ {expected_b:.4f} (got {b_val:.4f})")
        else:
            msg = ""
            if not a_ok:
                msg += f"a = {a_val:.4f} (expected ~{expected_a:.4f}); "
            if not b_ok:
                msg += f"b = {b_val:.4f} (expected ~{expected_b:.4f})"
            return (False, msg)

    def test_modeling_rates_I1_Kp_ratios(self):
        """
        For I=1 resonance, the K-p reaction rates should be in the ratio
        Sigma- pi+ : Sigma+ pi- : Sigma0 pi0 = 1 : 1 : 0.
        Check that rates are equal (nonzero) and the third is zero.
        """
        if any(x is None for x in [self.rate_Kp_Sigma_minus, self.rate_Kp_Sigma_plus, self.rate_Kp_Sigma0]):
            return (False, "SKIP: Kp rates missing")
        try:
            r1 = float(self.rate_Kp_Sigma_minus)
            r2 = float(self.rate_Kp_Sigma_plus)
            r3 = float(self.rate_Kp_Sigma0)
        except (TypeError, ValueError):
            return (False, "Non-numeric Kp rate")
        # Check r1 == r2 (nonzero)
        if not math.isclose(r1, r2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Rate Sigma- pi+ ({r1}) != Rate Sigma+ pi- ({r2})")
        # Check r3 is 0 (or very small)
        if not math.isclose(r3, 0, abs_tol=1e-2):
            return (False, f"Rate Sigma0 pi0 is not 0 (got {r3})")
        # Optionally check positivity
        if r1 <= 0:
            return (False, f"Rate Sigma- pi+ is {r1} (should be positive)")
        return (True, f"Rates: {r1}:{r2}:{r3} ≈ 1:1:0")

    def test_modeling_rates_I1_Kn_ratios(self):
        """For I=1 resonance, K-n reaction rates should be equal (1:1)."""
        if self.rate_Kn_Sigma_minus is None or self.rate_Kn_Sigma0 is None:
            return (False, "SKIP: Kn rates missing")
        try:
            r1 = float(self.rate_Kn_Sigma_minus)
            r2 = float(self.rate_Kn_Sigma0)
        except (TypeError, ValueError):
            return (False, "Non-numeric Kn rate")
        if not math.isclose(r1, r2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Rate Sigma- pi0 ({r1}) != Rate Sigma0 pi- ({r2})")
        return (True, f"Rates: {r1}:{r2} ≈ 1:1")

    def test_modeling_rates_I0_Kp_ratios(self):
        """
        For I=0 resonance, all three K-p reaction rates should be equal.
        The result rate_Kp_all is expected as a string '1:1:1'.
        """
        if self.rate_Kp_all is None:
            return (False, "SKIP: rate_Kp_all missing")
        if not isinstance(self.rate_Kp_all, str):
            return (False, "rate_Kp_all should be a string")
        # Remove spaces, split by ':'
        parts = [p.strip() for p in self.rate_Kp_all.split(':')]
        if len(parts) != 3:
            return (False, f"rate_Kp_all format incorrect: '{self.rate_Kp_all}' (expected '1:1:1')")
        try:
            r1, r2, r3 = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return (False, "rate_Kp_all parts are not numeric")
        # Check they are equal (isclose)
        if (math.isclose(r1, r2, rel_tol=1e-2, abs_tol=1e-2) and
            math.isclose(r2, r3, rel_tol=1e-2, abs_tol=1e-2)):
            return (True, f"Rates are equal: {r1}:{r2}:{r3}")
        else:
            return (False, f"Rates are not equal: {r1}:{r2}:{r3} (expected equal)")

    # ------------------------------------------------------------------
    # Monotonicity test: explicit perturbation of an input parameter
    # ------------------------------------------------------------------
    def test_monotonicity_a_squared_sum(self):
        """
        Check that if we perturb the coefficient 'a' (an input in the CG decomposition)
        by +10%, the squared sum a^2 + b^2 increases relative to the baseline.
        This tests the monotonic relationship: larger 'a' -> larger squared sum.
        """
        if self.a is None or self.b is None:
            return (False, "SKIP: a or b missing")
        try:
            a_orig = float(self.a)
            b_orig = float(self.b)
        except (TypeError, ValueError):
            return (False, "a or b is non-numeric")
        # Baseline: squared sum
        baseline_sum = a_orig**2 + b_orig**2
        # Perturb a by +10%
        a_perturbed = a_orig * 1.1
        # Keep b unchanged (as if we only perturb a)
        perturbed_sum = a_perturbed**2 + b_orig**2
        # The perturbed sum should be larger (since a^2 increases)
        if perturbed_sum > baseline_sum + 1e-9:  # allow tiny numerical tolerance
            return (True, f"Baseline sum={baseline_sum:.4f}, after +10% increase in a: sum={perturbed_sum:.4f} > baseline")
        else:
            return (False, f"Perturbed sum not greater: baseline={baseline_sum:.4f}, perturbed={perturbed_sum:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity test: final-answer plausibility only
    # ------------------------------------------------------------------
    def test_answer_sanity_rates_non_negative(self):
        """All numeric rates must be non-negative."""
        rates = [self.rate_Kp_Sigma_minus, self.rate_Kp_Sigma_plus, self.rate_Kp_Sigma0,
                 self.rate_Kn_Sigma_minus, self.rate_Kn_Sigma0]
        names = ['rate_Kp_Sigma-', 'rate_Kp_Sigma+', 'rate_Kp_Sigma0',
                 'rate_Kn_Sigma-', 'rate_Kn_Sigma0']
        for rate, name in zip(rates, names):
            if rate is None:
                continue
            try:
                val = float(rate)
            except (TypeError, ValueError):
                return (False, f"{name} is non-numeric")
            if val < 0:
                return (False, f"{name} is negative: {val}")
        return (True, "All numeric rates are non-negative")

    # ------------------------------------------------------------------
    # Main test runner
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
                passed, msg = False, f"{name} CRASHED: {e} ({traceback.format_exc()})"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)