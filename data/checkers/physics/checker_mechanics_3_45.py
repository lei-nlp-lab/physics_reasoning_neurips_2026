import math
import logging

class PhysicsChecker:
    """
    Audits a solution for the motion of a charged particle in constant
    electromagnetic fields (E=(a,0,0), B=(0,0,b)).  Checks:
      - interface completeness
      - invariant eigenvalue relations
      - modeling consistency of exponent expressions and boundedness condition
      - monotonic behaviour of the exponent magnitude with respect to field strength
      - weak sanity of derived quantities
    """

    def __init__(self, solution_instance):
        # ---- extract givens ----
        givens = getattr(solution_instance, 'givens', {})
        self.e = givens.get('e', None)          # charge
        self.m = givens.get('m', None)          # rest mass
        self.c = givens.get('c', None)          # speed of light
        self.a = givens.get('a', None)          # E_x component
        self.b = givens.get('b', None)          # B_z component
        # optional vector representations
        self.E = givens.get('E', None)
        self.B = givens.get('B', None)

        # ---- extract results ----
        results = getattr(solution_instance, 'results', {})
        self.lambda1 = results.get('lambda1', None)
        self.lambda2 = results.get('lambda2', None)
        self.lambda3 = results.get('lambda3', None)
        self.D = results.get('D', None)
        self.bounded_threshold = results.get('bounded_threshold', None)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _is_numeric(self, val):
        """Return True if val is a number (int, float, complex)."""
        return isinstance(val, (int, float, complex))

    def _as_float(self, val):
        """Try to convert a value to float.  If it fails or is complex, return None."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, complex):
            return None          # complex cannot be safely converted to float
        # SymPy handling: try numeric evaluation
        try:
            if hasattr(val, 'evalf'):
                num = val.evalf()
                if num.is_Float or num.is_Integer:
                    return float(num)
                else:
                    return None
        except Exception:
            return None
        return None

    def _as_complex(self, val):
        """Try to convert a value to complex.  Accepts int, float, complex, sympy complex."""
        if isinstance(val, (int, float)):
            return complex(val, 0)
        if isinstance(val, complex):
            return val
        # SymPy: try to evaluate and convert
        try:
            if hasattr(val, 'evalf'):
                num = val.evalf()
                if num.is_Float or num.is_Integer:
                    return complex(num, 0)
                # If it has an imaginary part, it may be a sympy Add of I*
                # Attempt complex conversion via meth
                real_part = float(num.as_real_imag()[0])
                imag_part = float(num.as_real_imag()[1])
                return complex(real_part, imag_part)
        except Exception:
            return None
        return None

    def _sqrt_radicand(self, a, b, c):
        """Return radicand a**2 - c**2 * b**2.  If any is None, return None."""
        if None in (a, b, c):
            return None
        return a**2 - c**2 * b**2

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------

    def test_interface_lambda1_present(self):
        if self.lambda1 is None:
            return False, "lambda1 is missing (None)"
        return True, "lambda1 is present"

    def test_interface_lambda2_present(self):
        if self.lambda2 is None:
            return False, "lambda2 is missing (None)"
        return True, "lambda2 is present"

    def test_interface_lambda3_present(self):
        if self.lambda3 is None:
            return False, "lambda3 is missing (None)"
        return True, "lambda3 is present"

    def test_interface_D_present(self):
        if self.D is None:
            return False, "D (discriminant) is missing (None)"
        return True, "D is present"

    def test_interface_bounded_threshold_present(self):
        if self.bounded_threshold is None:
            return False, "bounded_threshold is missing (None)"
        return True, "bounded_threshold is present"

    def test_interface_givens_available(self):
        """Check that essential givens (e,m,c,a,b) are all provided."""
        missing = []
        for name, val in [('e', self.e), ('m', self.m), ('c', self.c),
                          ('a', self.a), ('b', self.b)]:
            if val is None:
                missing.append(name)
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All essential givens present"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------

    def test_invariant_lambda1_zero(self):
        if not self._is_numeric(self.lambda1):
            return True, "SKIP: lambda1 not numeric"
        lam = float(self.lambda1)
        if math.isclose(lam, 0.0, abs_tol=1e-6):
            return True, "lambda1 ≈ 0 (as expected)"
        return False, f"lambda1 = {lam}, expected 0"

    def test_invariant_lambda_sum(self):
        """lambda_2 + lambda_3 should be zero."""
        lam2 = self._as_complex(self.lambda2)
        lam3 = self._as_complex(self.lambda3)
        if lam2 is None or lam3 is None:
            return True, "SKIP: lambda2 or lambda3 not numeric/complex"
        s = lam2 + lam3
        # tolerance on magnitude of sum
        if abs(s) < 1e-6:
            return True, "lambda2 + lambda3 ≈ 0"
        return False, f"lambda2 + lambda3 = {s}, expected 0"

    def test_invariant_lambda_product(self):
        """lambda_2 * lambda_3 should equal - (e/(mc))^2 * (a^2 - c^2 b^2)."""
        lam2 = self._as_complex(self.lambda2)
        lam3 = self._as_complex(self.lambda3)
        if lam2 is None or lam3 is None:
            return True, "SKIP: lambda2 or lambda3 not numeric/complex"
        if None in (self.e, self.m, self.c, self.a, self.b):
            return True, "SKIP: missing givens for product computation"
        expected_product = - (self.e**2 / (self.m**2 * self.c**2)) * (self.a**2 - self.c**2 * self.b**2)
        product = lam2 * lam3
        # product should be real; compare real part
        actual_real = product.real
        if math.isclose(actual_real, expected_product, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"lambda2*lambda3 ≈ {product.real:.6e}, expected {expected_product:.6e}"
        return False, f"lambda2*lambda3 real part = {actual_real:.6e}, expected {expected_product:.6e}"

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------

    def test_modeling_D_definition(self):
        """D should equal a**2 - c**2 * b**2."""
        if self.D is None:
            return True, "SKIP: D is None"
        expected_D = self._sqrt_radicand(self.a, self.b, self.c)
        if expected_D is None:
            return True, "SKIP: missing givens to compute expected D"
        if math.isclose(self.D, expected_D, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"D ≈ {self.D:.6e}, matches expected {expected_D:.6e}"
        return False, f"D = {self.D}, expected {expected_D}"

    def test_modeling_bounded_threshold_string(self):
        """Check that bounded_threshold is a non‑empty string and starts with 'a^2 < c^2 b^2'."""
        if self.bounded_threshold is None:
            return True, "SKIP: bounded_threshold is None"
        if not isinstance(self.bounded_threshold, str):
            return False, f"bounded_threshold is not a string: {type(self.bounded_threshold)}"
        if 'a^2 < c^2 b^2' not in self.bounded_threshold:
            return False, f"bounded_threshold does not contain expected condition: {self.bounded_threshold}"
        return True, f"bounded_threshold correctly contains condition"

    def test_modeling_lambda2_vs_lambda3_signs(self):
        """lambda2 and lambda3 should be opposite signs (if real) or complex conjugates."""
        lam2 = self._as_complex(self.lambda2)
        lam3 = self._as_complex(self.lambda3)
        if lam2 is None or lam3 is None:
            return True, "SKIP: lambda2 or lambda3 not numeric/complex"
        # For complex numbers, check they are conjugates (within tolerance)
        if isinstance(lam2, complex) and isinstance(lam3, complex):
            conj = lam2.conjugate()
            if math.isclose(lam3.real, conj.real, rel_tol=1e-2) and math.isclose(lam3.imag, conj.imag, abs_tol=1e-2):
                return True, "lambda2 and lambda3 are complex conjugates (as expected)"
            else:
                return False, f"lambda2 = {lam2}, lambda3 = {lam3} are not complex conjugates"
        # Real case: opposite sign
        if math.isclose(lam2, -lam3, rel_tol=1e-2):
            return True, f"lambda2 ≈ -lambda3"
        return False, f"lambda2 = {lam2}, lambda3 = {lam3}"

    # ------------------------------------------------------------------
    # Monotonicity test (perturbation)
    # ------------------------------------------------------------------

    def test_monotonicity_lambda2_magnitude_vs_a(self):
        """
        Perturb a by +5% and check that the magnitude of lambda2 changes in the
        expected direction (decreases if a^2 < c^2 b^2, increases otherwise).
        """
        # Need numeric a, b, c, e, m and numeric lambda2
        if None in (self.a, self.b, self.c, self.e, self.m):
            return True, "SKIP: missing givens"
        lam2 = self._as_complex(self.lambda2)
        if lam2 is None:
            return True, "SKIP: lambda2 not numeric/complex"

        radicand = self.a**2 - self.c**2 * self.b**2
        if radicand == 0:
            return True, "SKIP: degenerate case, radicand = 0"

        # Baseline magnitude of lambda2
        base_mag = abs(lam2)

        # Perturb a by +5%
        a_pert = self.a * 1.05
        rad_pert = a_pert**2 - self.c**2 * self.b**2

        # If radicand changes sign, skip (transition region)
        if (radicand > 0 and rad_pert < 0) or (radicand < 0 and rad_pert > 0):
            return True, "SKIP: perturbation crosses sign boundary; monotonicity not defined"

        # Compute perturbed magnitude
        # lambda2_pert = (e/(mc)) * sqrt(rad_pert)
        const = self.e / (self.m * self.c)
        # sqrt of negative gives imaginary; we want absolute value
        if rad_pert < 0:
            pert_mag = abs(const * (1j * math.sqrt(-rad_pert)))
        else:
            pert_mag = abs(const * math.sqrt(rad_pert))

        # Check direction
        if radicand < 0:
            # both negative: increasing a -> magnitude decreases
            if pert_mag <= base_mag * 1.001:   # allow small slack
                return True, f"a increased by 5%: mag decreased from {base_mag:.6e} to {pert_mag:.6e}"
            else:
                return False, f"monotonicity failure: a+5% increased mag from {base_mag:e} to {pert_mag:e} (should decrease)"
        else:
            # both positive: increasing a -> magnitude increases
            if pert_mag >= base_mag * 0.999:
                return True, f"a increased by 5%: mag increased from {base_mag:.6e} to {pert_mag:.6e}"
            else:
                return False, f"monotonicity failure: a+5% decreased mag from {base_mag:e} to {pert_mag:e} (should increase)"

    # ------------------------------------------------------------------
    # Answer sanity (weak, at most one)
    # ------------------------------------------------------------------

    def test_answer_sanity_lambdas_not_nan(self):
        """
        Very weak check: if lambdas are numeric, they should not be NaN or infinite.
        Does NOT use math.isclose or re‑derive the answer.
        """
        sane = True
        msgs = []
        for name, val in [('lambda1', self.lambda1), ('lambda2', self.lambda2), ('lambda3', self.lambda3)]:
            if val is None:
                continue
            if self._is_numeric(val):
                if math.isnan(float(val.real)) if isinstance(val, complex) else math.isnan(float(val)):
                    sane = False
                    msgs.append(f"{name} is NaN")
                if math.isinf(float(val.real)) if isinstance(val, complex) else math.isinf(float(val)):
                    sane = False
                    msgs.append(f"{name} is infinite")
        if not sane:
            return False, "; ".join(msgs)
        return True, "All numeric lambdas are finite (no NaN/inf)"

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    def run_all_tests(self):
        logs = []
        all_pass = True
        # discover all test_ methods
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {e}"
                import traceback
                msg += "\n" + traceback.format_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)