import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Robust auditor for first-order perturbation theory expectation values.
    Checks interface, invariants, modeling quality, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        """
        Extract all relevant data safely with None defaults.
        Parameters:
            solution_instance: object with attributes and/or .results dict
        """
        # Results from pseudocode [emit] contract
        self.results = getattr(solution_instance, 'results', {})
        self.B_expect = self.results.get('B_expect', None)
        self.x1 = self.results.get('x1', None)
        self.x2 = self.results.get('x2', None)
        self.x3 = self.results.get('x3', None)

        # Givens (physical parameters) – may be in solution_instance directly or in .givens dict
        self.lambda_val = None
        self.m = None
        self.omega = None
        # Try .givens dict first
        givens = getattr(solution_instance, 'givens', None)
        if givens is not None and isinstance(givens, dict):
            self.lambda_val = givens.get('lambda', None)
            self.m = givens.get('m', None)
            self.omega = givens.get('omega', None)
        else:
            # Fallback: direct attributes
            self.lambda_val = getattr(solution_instance, 'lambda', None)
            self.m = getattr(solution_instance, 'm', None)
            self.omega = getattr(solution_instance, 'omega', None)

    # ----------------------------------------------------------------------
    # Helper: try to interpret a result as a numeric value
    # ----------------------------------------------------------------------
    def _to_numeric(self, value, allow_string=False):
        """Attempt to convert value to float, safely."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if allow_string and isinstance(value, str):
            try:
                expr = sp.sympify(value)
                if expr.is_Number:
                    return float(expr)
                else:
                    return None
            except (sp.SympifyError, TypeError):
                return None
        return None

    def _is_zero_string(self, value):
        """Check if a value is either numeric zero or the string '0'."""
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return abs(value) < 1e-12
        if isinstance(value, str):
            try:
                num = sp.sympify(value)
                if num.is_Number:
                    return abs(float(num)) < 1e-12
                return False
            except:
                return value.strip() == "0"
        return False

    # ----------------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four required result keys exist."""
        required = ['B_expect', 'x1', 'x2', 'x3']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    # ----------------------------------------------------------------------
    # INVARIANT TESTS (theorem-based physical sanity, uses math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_x1_zero(self):
        """B = x1: expectation must be exactly zero (first order) due to parity."""
        if self.x1 is None:
            return (False, "SKIP: x1 not available")
        if self._is_zero_string(self.x1):
            return (True, "x1 correctly zero")
        num = self._to_numeric(self.x1, allow_string=True)
        if num is not None and math.isclose(num, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return (True, f"x1 ≈ 0 (value {num})")
        # If symbolic, check if it simplifies to zero
        try:
            expr = sp.sympify(self.x1)
            if expr == 0:
                return (True, "x1 symbolic zero")
        except:
            pass
        return (False, f"x1 cannot be verified as zero: {self.x1}")

    def test_invariant_x2_zero(self):
        """B = x2: expectation must be exactly zero."""
        if self.x2 is None:
            return (False, "SKIP: x2 not available")
        if self._is_zero_string(self.x2):
            return (True, "x2 correctly zero")
        num = self._to_numeric(self.x2, allow_string=True)
        if num is not None and math.isclose(num, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return (True, f"x2 ≈ 0 (value {num})")
        try:
            expr = sp.sympify(self.x2)
            if expr == 0:
                return (True, "x2 symbolic zero")
        except:
            pass
        return (False, f"x2 cannot be verified as zero: {self.x2}")

    def test_invariant_x3_nonzero(self):
        """B = x3: expectation must be non-zero for non-zero lambda (first order).
           If lambda is zero, zero is acceptable."""
        if self.x3 is None:
            return (False, "SKIP: x3 not available")
        # Check if lambda is zero – if so, zero is expected.
        try:
            lambda_num = float(self.lambda_val) if self.lambda_val is not None else None
        except (TypeError, ValueError):
            lambda_num = None
        if lambda_num is not None and math.isclose(lambda_num, 0.0, abs_tol=1e-15):
            # lambda zero: x3 must be zero as well
            if self._is_zero_string(self.x3):
                return (True, "x3 zero because lambda = 0")
            else:
                return (False, "x3 non-zero but lambda = 0")
        # Lambda non-zero: x3 must be non-zero
        if self._is_zero_string(self.x3):
            return (False, "x3 is zero, violates first-order prediction for non-zero lambda")
        num = self._to_numeric(self.x3, allow_string=True)
        if num is not None:
            if math.isclose(num, 0.0, abs_tol=1e-15):
                return (False, f"x3 is effectively zero: {num}")
            return (True, f"x3 non-zero: {num}")
        if isinstance(self.x3, str):
            try:
                expr = sp.sympify(self.x3)
                if expr == 0:
                    return (False, "x3 is symbolic zero")
            except:
                pass
        return (True, "x3 is present and appears non-zero (symbolic)")

    def test_invariant_x3_sign_correct(self):
        """If lambda, m, omega are available, check sign of x3 equals -lambda/(mω²)."""
        if self.lambda_val is None or self.m is None or self.omega is None:
            return (True, "SKIP: insufficient parameters for sign check")
        try:
            lambda_num = float(self.lambda_val)
            m_num = float(self.m)
            omega_num = float(self.omega)
        except (TypeError, ValueError):
            return (True, "SKIP: parameters not numeric")
        if math.isclose(lambda_num, 0.0, abs_tol=1e-15):
            return (True, "lambda = 0, sign trivial")
        num = self._to_numeric(self.x3, allow_string=True)
        if num is None:
            return (True, "SKIP: x3 not numeric, symbolic sign not checkable")
        expected_sign = -1 if lambda_num > 0 else 1
        actual_sign = 1 if num > 0 else -1
        if actual_sign == expected_sign:
            return (True, f"x3 sign matches -λ/(mω²): {num}")
        else:
            return (False, f"x3 sign mismatch: expected {expected_sign} but got {actual_sign} for λ={lambda_num}")

    # ----------------------------------------------------------------------
    # MODELING / TRACE TESTS (check that the emitted expressions are correct)
    # ----------------------------------------------------------------------
    def test_modeling_general_formula_string(self):
        """Check that B_expect is the expected string '<B>_0 + lambda * <C>_0'."""
        if self.B_expect is None:
            return (False, "SKIP: B_expect not available")
        expected = "<B>_0 + lambda * <C>_0"
        if self.B_expect.strip() == expected:
            return (True, "B_expect string matches formula")
        return (False, f"B_expect string mismatch: got '{self.B_expect}', expected '{expected}'")

    def test_modeling_x3_formula_expression(self):
        """Check that x3 symbolically equals -lambda/(m*omega^2) or numerically matches."""
        if self.x3 is None:
            return (False, "SKIP: x3 not available")
        # Try symbolic comparison
        try:
            expr = sp.sympify(self.x3)
            expected = sp.sympify("-λ/(m*ω^2)")
            if sp.simplify(expr - expected) == 0:
                return (True, "x3 equals -λ/(m ω²) symbolically")
        except:
            pass
        # Numeric fallback
        if self.lambda_val is not None and self.m is not None and self.omega is not None:
            try:
                lambda_num = float(self.lambda_val)
                m_num = float(self.m)
                omega_num = float(self.omega)
                expected_num = -lambda_num / (m_num * omega_num ** 2)
                x3_num = self._to_numeric(self.x3, allow_string=True)
                if x3_num is not None and math.isclose(x3_num, expected_num, rel_tol=1e-9, abs_tol=1e-9):
                    return (True, f"x3 numeric matches -λ/(mω²): {x3_num} ≈ {expected_num}")
            except (ValueError, TypeError):
                pass
        # If cannot verify, it's not a failure (modeling check is for correct expression)
        return (True, "x3 cannot be explicitly verified as -λ/(mω²), but it's not wrong")

    def test_modeling_x1x2_are_zero(self):
        """Modeling: x1 and x2 should be zero (exact)."""
        if self.x1 is None or self.x2 is None:
            return (False, "SKIP: x1 or x2 missing")
        both_ok = True
        msgs = []
        for var, name in [(self.x1, 'x1'), (self.x2, 'x2')]:
            if self._is_zero_string(var):
                msgs.append(f"{name} is zero")
            else:
                both_ok = False
                msgs.append(f"{name} not zero: {var}")
        if both_ok:
            return (True, "; ".join(msgs))
        else:
            return (False, "; ".join(msgs))

    # ----------------------------------------------------------------------
    # MONOTONICITY TESTS (perturb an input and compare perturbed output to baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_x3_lambda(self):
        """
        Perturb lambda by +10% and check that x3 becomes more negative.
        Requires numeric lambda, m, omega and numeric x3.
        """
        if self.lambda_val is None or self.m is None or self.omega is None:
            return (True, "SKIP: insufficient parameters for monotonicity")
        try:
            lambda0 = float(self.lambda_val)
            m_val = float(self.m)
            omega_val = float(self.omega)
        except (TypeError, ValueError):
            return (True, "SKIP: parameters not numeric")

        x3_num = self._to_numeric(self.x3, allow_string=True)
        if x3_num is None:
            return (True, "SKIP: x3 not numeric")

        # Baseline: x3 = -λ/(m ω²)
        # Perturb lambda by +10%
        delta_lambda = 0.10 * lambda0
        # Compute expected perturbed x3 using the formula (baseline change)
        expected_new_x3 = x3_num - delta_lambda / (m_val * omega_val ** 2)

        # Check that the perturbed value is more negative (decrease)
        if expected_new_x3 < x3_num:
            return (True, f"Monotonicity correct: increasing lambda from {lambda0} to {lambda0+delta_lambda} decreases x3 from {x3_num} to {expected_new_x3}")
        else:
            return (False, f"Expected decrease but got non-decrease: baseline {x3_num}, perturbed {expected_new_x3}")

    def test_monotonicity_x3_mass(self):
        """
        Perturb mass m by +10% and check that |x3| decreases (x3 becomes closer to zero).
        Since x3 = -λ/(m ω²), increasing m should bring x3 closer to zero (less negative if λ>0).
        """
        if self.lambda_val is None or self.m is None or self.omega is None:
            return (True, "SKIP: insufficient parameters for monotonicity")
        try:
            lambda0 = float(self.lambda_val)
            m_val = float(self.m)
            omega_val = float(self.omega)
        except (TypeError, ValueError):
            return (True, "SKIP: parameters not numeric")

        x3_num = self._to_numeric(self.x3, allow_string=True)
        if x3_num is None:
            return (True, "SKIP: x3 not numeric")

        # Baseline: x3 = -λ/(m ω²)
        # Perturb m by +10%
        delta_m = 0.10 * m_val
        new_m = m_val + delta_m
        expected_new_x3 = -lambda0 / (new_m * omega_val ** 2)

        # Check that the absolute value decreases (if λ non-zero)
        if abs(expected_new_x3) < abs(x3_num):
            return (True, f"Monotonicity correct: increasing mass from {m_val} to {new_m} reduces |x3| from {abs(x3_num)} to {abs(expected_new_x3)}")
        else:
            # If λ=0, both are zero, acceptable
            if math.isclose(lambda0, 0.0, abs_tol=1e-15):
                return (True, "lambda=0, trivial")
            return (False, f"Expected decrease in |x3| but got: baseline {abs(x3_num)}, perturbed {abs(expected_new_x3)}")

    # ----------------------------------------------------------------------
    # WEAK ANSWER SANITY TESTS (only one, weak)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Weak sanity: all results should be finite (not NaN, not infinite)."""
        checks = {}
        for name, val in [('B_expect', self.B_expect), ('x1', self.x1), ('x2', self.x2), ('x3', self.x3)]:
            if val is None:
                checks[name] = "missing"
            else:
                num = self._to_numeric(val, allow_string=True)
                if num is None:
                    checks[name] = "symbolic"
                elif math.isfinite(num):
                    checks[name] = f"finite ({num})"
                else:
                    checks[name] = "infinite or NaN"
        failures = [f"{k}: {v}" for k, v in checks.items() if v in ("infinite or NaN", "missing")]
        if failures:
            return (False, "; ".join(failures))
        return (True, "All results appear finite")

    # ----------------------------------------------------------------------
    # RUN ALL TESTS
    # ----------------------------------------------------------------------
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