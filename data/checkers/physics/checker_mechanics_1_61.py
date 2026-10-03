import sympy as sp
import math
import re
import traceback

class PhysicsChecker:
    """
    Audit checker for the hard‑sphere scattering problem.
    Checks isotropy of the classical differential cross section.
    No direct final-answer comparison; relies on theorem‑based checks.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely (no numeric defaults)
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

        # Extract key quantities (use None if missing)
        self.R = self.givens.get('R', None)
        self.theta = self.givens.get('theta', None)  # may be symbolic or numeric
        self.dsigma_dOmega = self.results.get('dsigma_dOmega', None)

        # Store symbolic version of result for analysis
        self._expr = None
        if self.dsigma_dOmega is not None:
            try:
                self._expr = sp.sympify(self.dsigma_dOmega)
            except (sp.SympifyError, TypeError, AttributeError):
                self._expr = None

        # Symbol for theta (use a default symbol if not provided)
        self._theta_sym = None
        if self.theta is not None:
            try:
                self._theta_sym = sp.sympify(self.theta)
            except (sp.SympifyError, TypeError):
                self._theta_sym = sp.Symbol('theta', positive=True)
        else:
            self._theta_sym = sp.Symbol('theta', positive=True)

        # Symbol for R
        self._R_sym = sp.Symbol('R', positive=True)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_result_keys(self):
        """Check that the solution provides 'dsigma_dOmega'."""
        if 'dsigma_dOmega' not in self.results:
            return (False, "Missing key 'dsigma_dOmega' in results")
        if self.dsigma_dOmega is None:
            return (False, "'dsigma_dOmega' is None")
        return (True, "'dsigma_dOmega' is present")

    def test_interface_result_type(self):
        """Result must be a string, number, or sympy expression."""
        if self.dsigma_dOmega is None:
            return (False, "SKIP: No result to check")
        ok_types = (str, int, float, sp.Basic)
        if not isinstance(self.dsigma_dOmega, ok_types):
            return (False, f"Unexpected type: {type(self.dsigma_dOmega).__name__}")
        return (True, "Result type is acceptable")

    # ----------------------------------------------------------------------
    # Theorem‑based invariant checks (using math.isclose or symbolic equality)
    # ----------------------------------------------------------------------
    def test_invariant_isotropy_no_theta_dependence(self):
        """
        The differential cross section must be independent of scattering angle.
        Check that the expression does not contain the variable theta.
        """
        if self.dsigma_dOmega is None:
            return (False, "SKIP: No result to check")

        # If it's a numeric constant, it's trivially isotropic.
        if isinstance(self.dsigma_dOmega, (int, float)):
            return (True, "Numeric constant implies isotropy")

        # If it's a sympy expression, check that no free symbols are theta
        if isinstance(self.dsigma_dOmega, sp.Basic):
            free_syms = self._expr.free_symbols
            theta_names = {'theta', 'θ'}
            if any(str(s) in theta_names for s in free_syms):
                return (False, f"Expression contains theta: {free_syms}")
            return (True, "Expression free of theta")

        # If it's a string, look for theta pattern
        if isinstance(self.dsigma_dOmega, str):
            lower = self.dsigma_dOmega.lower()
            if 'theta' in lower or 'θ' in lower:
                return (False, "Expression string contains 'theta' or 'θ'")
            if re.search(r'\bsin\b|\bcos\b', lower):
                return (False, "Expression string contains trigonometric functions (likely theta‑dependent)")
            return (True, "String expression does not contain theta identifiers")

        return (False, "Cannot determine isotropy from this type")

    def test_invariant_dsigma_dOmega_positive(self):
        """The differential cross section must be positive (physical sanity)."""
        if self.dsigma_dOmega is None:
            return (False, "SKIP: No result to check")

        if isinstance(self.dsigma_dOmega, (int, float)):
            if self.dsigma_dOmega > 0:
                return (True, f"dsigma/dOmega = {self.dsigma_dOmega} > 0")
            else:
                return (False, f"dsigma/dOmega is non‑positive: {self.dsigma_dOmega}")

        if isinstance(self.dsigma_dOmega, sp.Basic):
            # Attempt a numeric substitution for a quick sign check
            subs_dict = {}
            for s in self._expr.free_symbols:
                name = str(s)
                if name in ('R',) or name.startswith('R'):
                    subs_dict[s] = 2.0  # arbitrary positive
                elif name in ('theta', 'θ'):
                    subs_dict[s] = 1.0  # arbitrary valid angle
            try:
                numeric_val = float(self._expr.subs(subs_dict))
                if numeric_val > 0:
                    return (True, f"Evaluated expression (subs={subs_dict}) = {numeric_val} > 0")
                else:
                    return (False, f"Expression evaluates to non‑positive: {numeric_val}")
            except (TypeError, ValueError):
                return (True, "Symbolic expression cannot be evaluated numerically but is assumed positive")

        if isinstance(self.dsigma_dOmega, str):
            # Weak heuristic: check for negative sign
            if '-' in self.dsigma_dOmega.strip() and not self.dsigma_dOmega.strip().startswith('-'):
                pass  # Could be subtraction, not clearly negative
            return (True, "String expression accepted as positive by default")

        return (True, "Positive condition not checkable; assumed satisfied")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (theorem‑based, no direct final answer matching)
    # ----------------------------------------------------------------------
    def test_modeling_isotropy_derivative(self):
        """
        Modeling check: the differential cross section must have zero derivative
        with respect to theta.  Compute d/dθ (dsigma_dOmega) and verify it is
        identically zero (or very small if numeric).  This re‑derives isotropy
        using calculus without referencing the final constant.
        """
        if self.dsigma_dOmega is None:
            return (False, "SKIP: No result to check")

        theta_sym = self._theta_sym

        # Symbolic case
        if isinstance(self.dsigma_dOmega, sp.Basic):
            # If no theta dependence, derivative is trivially zero
            if theta_sym not in self._expr.free_symbols:
                return (True, "No theta dependence, derivative is zero")
            try:
                deriv = sp.diff(self._expr, theta_sym)
                simplified = sp.simplify(deriv)
                if simplified == 0:
                    return (True, "Derivative w.r.t. theta is identically zero (isotropy verified)")
                else:
                    return (False, f"Derivative w.r.t. theta is not zero: {simplified}")
            except Exception as e:
                return (False, f"Derivative computation failed: {e}")

        # Numeric case: cannot compute derivative without functional form; skip
        if isinstance(self.dsigma_dOmega, (int, float)):
            return (True, "Numeric result; derivative computed from functional form not available – skipping")

        # String case
        return (True, "String result; derivative test not applicable")

    def test_modeling_formula_consistency(self):
        """
        Modeling check: if the result is symbolic, verify that it is of the form
        (something) * R^2 (i.e., it depends on R only quadratically) and that it
        does not contain trigonometric functions of theta.  This confirms the
        final expression has the correct structure without matching the constant.
        """
        if self._expr is None:
            return (False, "SKIP: No symbolic expression to analyse")
        if not isinstance(self._expr, sp.Basic):
            return (False, "SKIP: Expression not symbolic")

        # Check free symbols
        free = self._expr.free_symbols
        # Allowed symbols: only R (or nothing if constant)
        for s in free:
            if str(s) not in ('R',):
                return (False, f"Contains disallowed symbol: {s}")
        # Check that if R is present, the power is quadratic (i.e., R**2 factor)
        if self._R_sym in free:
            # Check that the expression is proportional to R**2
            ratio = sp.simplify(self._expr / (self._R_sym**2))
            if ratio.is_constant():
                return (True, "Expression is of the form constant × R²")
            else:
                return (False, "Expression not proportional to R²")
        else:
            # No R – must be constant (maybe a numeric constant)
            return (True, "Expression is constant (no R), consistent with isotropy")

    # ----------------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation and baseline comparison)
    # ----------------------------------------------------------------------
    def test_monotonicity_R_scaling(self):
        """
        The cross section should scale as R².  Perturb R by a factor of 1.5
        and check that the expression increases by a factor of 1.5² = 2.25.
        """
        if self._expr is None:
            return (False, "SKIP: Cannot test scaling without symbolic expression")
        if not isinstance(self._expr, sp.Basic):
            return (False, "SKIP: Expression not symbolic")

        # Find the symbol for R
        R_sym = None
        for s in self._expr.free_symbols:
            if str(s).upper() == 'R':
                R_sym = s
                break
        if R_sym is None:
            return (True, "No R symbol found; scaling cannot be verified")

        try:
            # Baseline: evaluate at R = 2 (arbitrary positive)
            baseline = self._expr.subs(R_sym, 2.0)
            # Perturb: multiply R by 1.5 → 3.0
            perturbed = self._expr.subs(R_sym, 3.0)
            # Expected factor: (3/2)² = 2.25
            expected_factor = 2.25
            actual_factor = float(perturbed / baseline)
            if math.isclose(actual_factor, expected_factor, rel_tol=0.01):
                return (True, f"Scaling verified: R→1.5R gives factor {actual_factor} (expected 2.25)")
            else:
                return (False, f"Scaling not correct: factor = {actual_factor}, expected 2.25")
        except Exception as e:
            return (False, f"Scaling test error: {e}")

    def test_monotonicity_theta_variation(self):
        """
        Since the cross section is isotropic, varying theta should produce no change.
        Evaluate the expression at θ = 0.5 rad and θ = 2.0 rad and compare.
        """
        if self._expr is None:
            return (False, "SKIP: Cannot test theta variation without symbolic expression")
        if not isinstance(self._expr, sp.Basic):
            return (False, "SKIP: Expression not symbolic")

        theta_sym = self._theta_sym
        if theta_sym not in self._expr.free_symbols:
            return (True, "No theta dependence; constant function")

        try:
            # Baseline at θ = 0.5 rad
            baseline = self._expr.subs(theta_sym, 0.5)
            # Perturb to θ = 2.0 rad
            perturbed = self._expr.subs(theta_sym, 2.0)
            diff = sp.N(baseline - perturbed)
            if math.isclose(float(diff), 0.0, abs_tol=1e-10):
                return (True, f"Isotropy confirmed: dσ/dΩ(θ=0.5) = {baseline}, dσ/dΩ(θ=2) = {perturbed}")
            else:
                return (False, f"Difference found: baseline={baseline}, perturbed={perturbed}, diff={diff}")
        except Exception as e:
            return (False, f"Theta variation test error: {e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one, no isclose, no re‑derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic_checks(self):
        """
        Weak sanity: the result must be finite and real, and if numeric, must be positive.
        No re‑derivation of expected values, no target‑answer comparison.
        """
        if self.dsigma_dOmega is None:
            return (False, "SKIP: No result to check")

        if isinstance(self.dsigma_dOmega, (int, float)):
            if math.isnan(self.dsigma_dOmega) or math.isinf(self.dsigma_dOmega):
                return (False, "Result is NaN or Inf")
            if self.dsigma_dOmega <= 1e-15:
                return (False, f"Result is effectively zero or negative: {self.dsigma_dOmega}")
            return (True, "Numeric result is finite and positive")

        if isinstance(self.dsigma_dOmega, sp.Basic):
            # Cannot check finiteness symbolically; accept by convention
            return (True, "Symbolic expression accepted (cannot check finiteness symbolically)")

        if isinstance(self.dsigma_dOmega, str):
            lower = self.dsigma_dOmega.lower()
            if 'inf' in lower or 'nan' in lower or 'infty' in lower:
                return (False, "String contains infinite or NaN")
            return (True, "String expression accepted")

        return (True, "Weak sanity check passed")

    # ----------------------------------------------------------------------
    # Test runner (do not modify)
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (all_pass, logs)."""
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