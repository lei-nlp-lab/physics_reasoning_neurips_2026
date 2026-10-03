import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for the quantum harmonic oscillator Gaussian wavefunction problem.
    Checks:
    - Interface: required result keys exist with proper types.
    - Invariants: <x>=0, <p>=0, V(x) from TISE, P(p) normalisation.
    - Modeling/trace: V(x) quadratic, P(p) Gaussian form.
    - Monotonicity: P(p) decreases with |p|.
    - Weak answer sanity: finite real outputs.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Required result keys (from [emit] contract)
        self.mean_x = results.get('mean_x', None)
        self.mean_p = results.get('mean_p', None)
        self.V_x = results.get('V_x', None)          # expected as string or sympy expression
        self.P_p_dp = results.get('P_p_dp', None)    # expected as string

        # Givens (may be provided as attributes of solution_instance)
        self.gamma = getattr(solution_instance, 'gamma', None)
        self.m = getattr(solution_instance, 'm', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.psi_x = getattr(solution_instance, 'psi_x', None)   # string or sympy expression

    # ----------------------------------------------------------------------
    # Interface checks: existence and basic type/format
    # ----------------------------------------------------------------------
    def test_interface_mean_x_exists(self):
        """Check that mean_x is present and is a number."""
        if self.mean_x is None:
            return (False, "SKIP: mean_x is missing")
        if not isinstance(self.mean_x, (int, float, complex)):
            return (False, f"mean_x has unexpected type: {type(self.mean_x)}")
        return (True, f"mean_x = {self.mean_x}")

    def test_interface_mean_p_exists(self):
        """Check that mean_p is present and is a number."""
        if self.mean_p is None:
            return (False, "SKIP: mean_p is missing")
        if not isinstance(self.mean_p, (int, float, complex)):
            return (False, f"mean_p has unexpected type: {type(self.mean_p)}")
        return (True, f"mean_p = {self.mean_p}")

    def test_interface_V_x_exists(self):
        """Check that V_x is present and is a string or sympy expression."""
        if self.V_x is None:
            return (False, "SKIP: V_x is missing")
        if not (isinstance(self.V_x, str) or isinstance(self.V_x, sp.Expr)):
            return (False, f"V_x has unexpected type: {type(self.V_x)}")
        return (True, f"V_x = {self.V_x}")

    def test_interface_P_p_dp_exists(self):
        """Check that P_p_dp is present and is a string or sympy expression."""
        if self.P_p_dp is None:
            return (False, "SKIP: P_p_dp is missing")
        if not (isinstance(self.P_p_dp, str) or isinstance(self.P_p_dp, sp.Expr)):
            return (False, f"P_p_dp has unexpected type: {type(self.P_p_dp)}")
        return (True, f"P_p_dp = {self.P_p_dp}")

    # ----------------------------------------------------------------------
    # Invariant checks: theorem‑based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_mean_x_zero_by_symmetry(self):
        """Expect <x>=0 from odd integrand."""
        if self.mean_x is None:
            return (False, "SKIP: mean_x not available")
        if not math.isclose(self.mean_x, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"mean_x = {self.mean_x} not close to 0")
        return (True, f"mean_x ≈ 0 within tolerance")

    def test_invariant_mean_p_zero_by_symmetry(self):
        """Expect <p>=0 from odd integrand."""
        if self.mean_p is None:
            return (False, "SKIP: mean_p not available")
        if not math.isclose(self.mean_p, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"mean_p = {self.mean_p} not close to 0")
        return (True, f"mean_p ≈ 0 within tolerance")

    def test_invariant_V_x_derived_from_TISE(self):
        """
        Re‑derive V(x) from TISE using the given ψ(x) and E (if available).
        Compare symbolic expression with reported V_x.
        """
        if self.gamma is None or self.m is None or self.hbar is None or self.psi_x is None:
            return (False, "SKIP: insufficient givens (need gamma, m, hbar, psi_x)")
        if self.V_x is None:
            return (False, "SKIP: V_x not available for comparison")

        x = sp.symbols('x', real=True)
        gamma_sym = sp.symbols('gamma', positive=True)
        m_sym = sp.symbols('m', positive=True)
        hbar_sym = sp.symbols('hbar', positive=True)

        # Build given ψ(x) as sympy expression
        psi_expr = (gamma_sym**2 / sp.pi)**(sp.Rational(1,4)) * sp.exp(-gamma_sym**2 * x**2 / 2)

        # Compute ψ''
        psi_second = sp.diff(psi_expr, x, 2)

        # TISE: -ħ²/(2m) ψ'' + V ψ = E ψ  ->  V = E + (ħ²/(2m)) (ψ''/ψ)
        E_expr = hbar_sym**2 * gamma_sym**2 / (2 * m_sym)
        V_derived = E_expr + (hbar_sym**2/(2*m_sym)) * (psi_second / psi_expr)
        V_derived_simplified = sp.simplify(V_derived)

        # Expected: V(x) = (ħ² γ⁴ x²)/(2m)
        V_expected = (hbar_sym**2 * gamma_sym**4 * x**2) / (2*m_sym)

        # Compare with reported V_x
        if isinstance(self.V_x, str):
            try:
                V_reported = sp.sympify(self.V_x)
            except Exception as e:
                return (False, f"Cannot sympify V_x string: {e}")
        else:
            V_reported = self.V_x

        # Use numerical substitution to compare
        subs_dict = {gamma_sym: 1.0, m_sym: 1.0, hbar_sym: 1.0, x: 1.0}
        try:
            derived_val = float(V_derived_simplified.subs(subs_dict).evalf())
            reported_val = float(V_reported.subs(subs_dict).evalf())
        except Exception as e:
            return (False, f"Numerical substitution failed: {e}")

        if not math.isclose(derived_val, reported_val, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"V(x) values differ: derived={derived_val}, reported={reported_val}")
        return (True, f"V_x derived from TISE matches reported: {V_reported}")

    def test_invariant_P_p_normalization(self):
        """
        Check that ∫ P(p) dp = 1 (sympy integration of the expression).
        """
        if self.P_p_dp is None or self.hbar is None or self.gamma is None:
            return (False, "SKIP: missing P_p_dp or hbar/gamma")
        if isinstance(self.P_p_dp, str):
            try:
                P_expr = sp.sympify(self.P_p_dp)
            except Exception as e:
                return (False, f"Cannot sympify P_p_dp: {e}")
        else:
            P_expr = self.P_p_dp
        p = sp.symbols('p', real=True)
        try:
            integral = sp.integrate(P_expr, (p, -sp.oo, sp.oo))
        except Exception as e:
            return (False, f"Integration failed: {e}")
        subs = {sp.symbols('gamma'): 1.0, sp.symbols('hbar'): 1.0}
        try:
            numeric_integral = float(integral.subs(subs).evalf())
        except Exception as e:
            return (False, f"Numerical evaluation failed: {e}")
        if not math.isclose(numeric_integral, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"∫ P(p) dp = {numeric_integral}, not close to 1")
        return (True, f"P(p) integrates to 1 within tolerance")

    # ----------------------------------------------------------------------
    # Modeling/trace quality checks (at least two required)
    # ----------------------------------------------------------------------
    def test_trace_V_x_quadratic_and_even(self):
        """Check that V_x expression is quadratic and even (no odd powers)."""
        if self.V_x is None:
            return (False, "SKIP: V_x not available")
        if isinstance(self.V_x, str):
            try:
                V_expr = sp.sympify(self.V_x)
            except Exception as e:
                return (False, f"Cannot sympify V_x string: {e}")
        else:
            V_expr = self.V_x
        x = sp.symbols('x', real=True)
        expanded = sp.expand(V_expr)
        # Check evenness: V(x) == V(-x)
        diff = sp.simplify(V_expr.subs(x, -x) - V_expr)
        if diff != 0:
            return (False, f"V(x) is not even: V(x)-V(-x) = {diff}")
        # Check only even powers of x appear: use Poly
        poly = sp.Poly(expanded, x)
        monoms = poly.monoms()
        for monom in monoms:
            if len(monom) == 1 and monom[0] % 2 == 1:  # odd power
                return (False, f"V(x) contains odd power x^{monom[0]}")
        return (True, "V(x) is even and contains no odd powers")

    def test_trace_P_p_gaussian_form(self):
        """
        Verify that P(p) has the form A * exp(-b * p^2) with correct structure.
        The expected form (from Fourier transform) is:
        sqrt(1/(pi * hbar^2 * gamma^2)) * exp(-p^2/(hbar^2 * gamma^2))
        We check that it is a Gaussian (exp of -c * p^2) with c > 0.
        """
        if self.P_p_dp is None:
            return (False, "SKIP: P_p_dp not available")
        if isinstance(self.P_p_dp, str):
            try:
                P_expr = sp.sympify(self.P_p_dp)
            except Exception as e:
                return (False, f"Cannot sympify P_p_dp: {e}")
        else:
            P_expr = self.P_p_dp
        p = sp.symbols('p', real=True)
        # Extract factor multiplying the exponential
        # Write as product: factor * exp(-c * p^2)
        # We cannot directly extract the structure, but we can check that P_expr involves exp(-c * p^2)
        # and that the factor is positive and finite.
        try:
            # Substitute a small p to check factor positive
            subs = {sp.symbols('gamma'): 1.0, sp.symbols('hbar'): 1.0}
            val_at_zero = float(P_expr.subs({p: 0}).subs(subs).evalf())
            val_at_one = float(P_expr.subs({p: 1}).subs(subs).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate P(p) numerically: {e}")
        # Check that P(0) > P(1) (monotonic decreasing) and both positive
        if val_at_zero <= 0 or val_at_one <= 0:
            return (False, f"P(p) values are not positive: P(0)={val_at_zero}, P(1)={val_at_one}")
        if val_at_zero <= val_at_one:
            return (False, "P(p) does not decrease with p")
        # Check Gaussian exponent: P(p) should be proportional to exp(-p^2/(hbar^2 gamma^2))
        # For a quick structural test, check logarithmic ratio
        try:
            log_ratio = math.log(val_at_zero / val_at_one)
            # Expected exponent difference: (1^2 - 0^2)/(hbar^2 gamma^2) = 1/(hbar^2 gamma^2) ≈ 1 if hbar=gamma=1
            expected_log_ratio = 1.0  # since 1/(1*1)=1
        except Exception:
            return (False, "Could not compute logarithmic ratio")
        if not math.isclose(log_ratio, expected_log_ratio, rel_tol=2e-2, abs_tol=2e-2):
            return (False, f"Exponent of Gaussian does not match expected: log(P(0)/P(1))={log_ratio}, expected ~1")
        return (True, "P(p) has Gaussian form with correct exponent")

    # ----------------------------------------------------------------------
    # Monotonicity check: perturb input p and compare output P(p)
    # ----------------------------------------------------------------------
    def test_monotonicity_P_p_decreases_with_abs_p(self):
        """
        Perturb p from a baseline (p=0.5) to a larger value (p=0.75)
        and verify that P decreases.
        """
        if self.P_p_dp is None or self.hbar is None or self.gamma is None:
            return (False, "SKIP: missing P_p_dp or parameters")
        if isinstance(self.P_p_dp, str):
            try:
                P_expr = sp.sympify(self.P_p_dp)
            except Exception as e:
                return (False, f"Cannot sympify P_p_dp: {e}")
        else:
            P_expr = self.P_p_dp
        p = sp.symbols('p', real=True)
        subs = {sp.symbols('gamma'): 1.0, sp.symbols('hbar'): 1.0}
        # Baseline at p=0.5
        try:
            baseline = float(P_expr.subs({p: 0.5}).subs(subs).evalf())
            perturbed = float(P_expr.subs({p: 0.75}).subs(subs).evalf())
        except Exception as e:
            return (False, f"Could not evaluate P(p) numerically: {e}")
        if baseline <= perturbed:
            return (False, f"P(0.5)={baseline} <= P(0.75)={perturbed}, expected decrease")
        return (True, f"P decreases from baseline (p=0.5) to perturbed (p=0.75): {baseline} > {perturbed}")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (only one, secondary)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_reals(self):
        """Check that mean_x, mean_p are finite real numbers and V_x is real symbolically."""
        issues = []
        if self.mean_x is not None:
            if not isinstance(self.mean_x, (int, float)):
                issues.append("mean_x is not a number")
            elif math.isinf(self.mean_x) or math.isnan(self.mean_x):
                issues.append("mean_x is inf or NaN")
        if self.mean_p is not None:
            if not isinstance(self.mean_p, (int, float)):
                issues.append("mean_p is not a number")
            elif math.isinf(self.mean_p) or math.isnan(self.mean_p):
                issues.append("mean_p is inf or NaN")
        if self.V_x is not None:
            try:
                if isinstance(self.V_x, str):
                    V_expr = sp.sympify(self.V_x)
                else:
                    V_expr = self.V_x
                x = sp.symbols('x', real=True)
                im = sp.im(V_expr)
                if im != 0:
                    issues.append("V_x has non-zero imaginary part")
            except Exception:
                issues.append("V_x could not be parsed/sympified")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Numbers finite, V_x real")

    # ----------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)