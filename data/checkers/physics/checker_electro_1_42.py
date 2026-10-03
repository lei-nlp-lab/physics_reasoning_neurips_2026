import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    PhysicsChecker for the split conducting cylinder problem.
    Audits the solution for:
      - Interface completeness
      - Laplace's equation satisfaction
      - Boundary condition consistency
      - Asymptotic expansions (large r, small r)
      - Electric field relations
      - Monotonicity trends
      - Basic sanity checks
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all needed quantities from the solution instance with safe defaults.
        Expected keys in solution_instance.results:
          - V (symbolic or numeric expression)
          - E_r_large, E_theta_large
          - E_r_small, E_theta_small
          - possibly a, V0, r, theta as separate givens
        """
        # Givens (safe access, default None)
        self.a = getattr(solution_instance, 'a', None)
        if self.a is None:
            self.a = solution_instance.results.get('a', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        if self.V0 is None:
            self.V0 = solution_instance.results.get('V0', None)
        self.r = getattr(solution_instance, 'r', None)
        if self.r is None:
            self.r = solution_instance.results.get('r', None)
        self.theta = getattr(solution_instance, 'theta', None)
        if self.theta is None:
            self.theta = solution_instance.results.get('theta', None)

        # Results
        results = getattr(solution_instance, 'results', {})
        self.V = results.get('V', None)
        self.E_r_large = results.get('E_r_large', None)
        self.E_theta_large = results.get('E_theta_large', None)
        self.E_r_small = results.get('E_r_small', None)
        self.E_theta_small = results.get('E_theta_small', None)

        # Optional: explanation steps (not needed for checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Tolerance for numerical comparisons
        self.eps = 1e-2
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    # --------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all expected result keys exist and are not None."""
        required = ['V', 'E_r_large', 'E_theta_large', 'E_r_small', 'E_theta_small']
        missing = [key for key in required if getattr(self, key, None) is None]
        if missing:
            return False, f"Missing required keys: {missing}"
        return True, "All required keys present."

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that a, V0, r, theta are available (or at least a and V0)."""
        missing = []
        for name in ['a', 'V0']:
            if getattr(self, name, None) is None:
                missing.append(name)
        if missing:
            return False, f"Missing essential givens: {missing}"
        return True, "Givens present."

    # --------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # --------------------------------------------------------------
    def test_invariant_laplace_equation_symbolic(self) -> Tuple[bool, str]:
        """
        For symbolic V, compute ∇²V = (1/r) ∂_r(r ∂_r V) + (1/r²) ∂²_θ V and check it's zero.
        Only works if V is a sympy expression and r, theta are sympy symbols.
        """
        if self.V is None or not isinstance(self.V, sp.Basic):
            return False, "SKIP: V is not symbolic."
        if self.r is None or self.theta is None:
            return False, "SKIP: r or theta not available (needed for Laplacian)."

        # Check using given symbols if they are sympy; otherwise create symbols
        if isinstance(self.r, sp.Basic) and isinstance(self.theta, sp.Basic):
            r_sym = self.r
            theta_sym = self.theta
        else:
            r_sym, theta_sym = sp.symbols('r theta', positive=True)
        # Substitute if self.r, self.theta are numbers? Usually they are symbols.
        V_expr = self.V
        # Compute Laplacian in polar coordinates (2D, no z-dependence)
        laplacian = (1/r_sym) * sp.diff(r_sym * sp.diff(V_expr, r_sym), r_sym) + (1/r_sym**2) * sp.diff(V_expr, theta_sym, 2)
        simplified = sp.simplify(laplacian)
        if simplified == 0:
            return True, "∇²V = 0 satisfied (r ≠ a)."
        else:
            return False, f"Laplacian not zero: {simplified}"

    def test_invariant_laplace_numerical(self) -> Tuple[bool, str]:
        """Numerical check of Laplace's equation at random points (r ≠ a)."""
        if self.V is None:
            return False, "SKIP: V not available."
        if self.a is None or self.V0 is None or self.r is None or self.theta is None:
            return False, "SKIP: needed parameters missing."

        # Choose a few points: (r=2a, theta=π/4), (r=0.5a, theta=3π/4)
        test_points = [
            (2.0 * self.a, math.pi / 4),
            (2.0 * self.a, 3 * math.pi / 4),
            (0.5 * self.a, math.pi / 3),
            (0.5 * self.a, 5 * math.pi / 6),
        ]
        try:
            V_func = sp.lambdify((self.r, self.theta), self.V, 'numpy')
        except:
            V_func = None
        if V_func is None:
            return False, "Cannot convert V to numeric function."

        for r_val, theta_val in test_points:
            # Compute Laplacian numerically via finite differences
            dr = 0.01 * r_val
            dtheta = 0.01
            # Get V at center and neighbors
            try:
                V0_ = V_func(r_val, theta_val)
                V_rp = V_func(r_val + dr, theta_val)
                V_rm = V_func(r_val - dr, theta_val)
                V_tp = V_func(r_val, theta_val + dtheta)
                V_tm = V_func(r_val, theta_val - dtheta)
            except Exception as e:
                return False, f"Error evaluating V at ({r_val}, {theta_val}): {e}"

            # Laplacian approximation
            d2V_dr2 = (V_rp - 2*V0_ + V_rm) / (dr**2)
            dV_dr = (V_rp - V_rm) / (2*dr)
            laplacian = d2V_dr2 + (1/r_val) * dV_dr + (1/r_val**2) * (V_tp - 2*V0_ + V_tm) / (dtheta**2)
            if not math.isclose(float(laplacian), 0.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                return False, f"Laplacian is {laplacian} at (r={r_val}, θ={theta_val}), expected ~0."
        return True, "Laplace equation satisfied at sample points."

    def test_invariant_boundary_condition(self) -> Tuple[bool, str]:
        """Check that V(a, θ) = V0 for 0<θ<π and 0 for π<θ<2π (fuzzy)."""
        if self.V is None or self.a is None or self.V0 is None:
            return False, "SKIP: V, a, or V0 missing."
        # Evaluate at several angles
        angles_upper = [math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]
        angles_lower = [7*math.pi/6, 4*math.pi/3, 3*math.pi/2, 5*math.pi/3, 11*math.pi/6]
        try:
            V_func = sp.lambdify((self.r, self.theta), self.V, 'math')
        except:
            return False, "Cannot convert V to numeric function."

        # Check upper half
        for theta_val in angles_upper:
            try:
                val = V_func(self.a, theta_val)
            except:
                return False, f"Error evaluating V at r=a, θ={theta_val}"
            if not math.isclose(val, self.V0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                return False, f"At θ={theta_val}: V={val}, expected {self.V0}."

        # Check lower half
        for theta_val in angles_lower:
            try:
                val = V_func(self.a, theta_val)
            except:
                return False, f"Error evaluating V at r=a, θ={theta_val}"
            if not math.isclose(val, 0.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                return False, f"At θ={theta_val}: V={val}, expected 0."

        return True, "Boundary condition holds within tolerance."

    # --------------------------------------------------------------
    # Modeling / trace tests
    # --------------------------------------------------------------
    def test_modeling_asymptotic_large_r(self) -> Tuple[bool, str]:
        """
        Check that for large r, V expands as V0/2 + (2V0 a sinθ)/(π r) (leading order).
        We use symbolic series expansion (re-derived from given V).
        """
        if self.V is None or not isinstance(self.V, sp.Basic):
            return False, "SKIP: V is not symbolic."
        # Create symbol for expansion
        r = sp.symbols('r', positive=True)
        a = sp.symbols('a', positive=True)
        V0 = sp.symbols('V0')
        theta = sp.symbols('theta', real=True)
        # Define the expected leading behavior
        expected_leading = V0/2 + (2*V0*a*sp.sin(theta))/(sp.pi*r)
        # Expand V around r=∞ (i.e., replace r->1/x, expand at x=0)
        V_expr = self.V.subs({self.r: r, self.a: a, self.V0: V0, self.theta: theta})
        x = sp.symbols('x', positive=True)
        series = sp.series(V_expr.subs(r, 1/x), x, 0, 3).removeO()
        # Replace back x = 1/r
        series_r = series.subs(x, 1/r)
        # Compare
        diff = sp.simplify(series_r - expected_leading)
        if diff == 0:
            return True, "Large-r asymptotic matches V0/2 + (2V0 a sinθ)/(π r)."
        else:
            # Extract coefficient of 1/r
            coeff = sp.expand(series_r - V0/2) * r
            expected_coeff = 2*V0*a*sp.sin(theta)/sp.pi
            if sp.simplify(coeff - expected_coeff) == 0:
                return True, "Large-r asymptotic leading coefficient matches."
            else:
                return False, f"Large-r asymptotic does not match. Series: {series_r}, expected leading: {expected_leading}"

    def test_modeling_asymptotic_small_r(self) -> Tuple[bool, str]:
        """
        Check that for small r, V expands as V0/2 + (2V0 r sinθ)/(π a).
        """
        if self.V is None or not isinstance(self.V, sp.Basic):
            return False, "SKIP: V is not symbolic."
        r = sp.symbols('r', positive=True)
        a = sp.symbols('a', positive=True)
        V0 = sp.symbols('V0')
        theta = sp.symbols('theta', real=True)
        expected_leading = V0/2 + (2*V0*r*sp.sin(theta))/(sp.pi*a)
        V_expr = self.V.subs({self.r: r, self.a: a, self.V0: V0, self.theta: theta})
        series = sp.series(V_expr, r, 0, 3).removeO()
        diff = sp.simplify(series - expected_leading)
        if diff == 0:
            return True, "Small-r asymptotic matches V0/2 + (2V0 r sinθ)/(π a)."
        else:
            # Check leading coefficient
            coeff = (series - V0/2) / r
            expected_coeff = 2*V0*sp.sin(theta)/(sp.pi*a)
            if sp.simplify(coeff - expected_coeff) == 0:
                return True, "Small-r asymptotic leading coefficient matches."
            else:
                return False, f"Small-r asymptotic mismatch. Series: {series}, expected leading: {expected_leading}"

    def test_trace_electric_field_large(self) -> Tuple[bool, str]:
        """
        Check that the large-r electric field components are consistent with
        E = -∇V from the asymptotic potential V_asymp = V0/2 + (2V0 a sinθ)/(π r).
        We compare given E_r_large, E_theta_large with the gradient of the asymptotic V.
        """
        if self.E_r_large is None or self.E_theta_large is None:
            return False, "SKIP: Large-r field components not available."
        if any(x is None for x in [self.a, self.V0, self.r, self.theta]):
            return False, "SKIP: Missing parameters."
        # Convert given fields to sympy if not already
        try:
            E_r_given = sp.sympify(self.E_r_large)
            E_theta_given = sp.sympify(self.E_theta_large)
        except:
            return False, "Cannot convert E fields to sympy."

        # Expected from -∇(V0/2 + (2V0 a sinθ)/(π r))
        r, theta, a, V0 = sp.symbols('r theta a V0', real=True)
        V_asymp = V0/2 + (2*V0*a*sp.sin(theta))/(sp.pi*r)
        E_r_expected = -sp.diff(V_asymp, r)
        E_theta_expected = -(1/r)*sp.diff(V_asymp, theta)
        E_r_expected_simp = sp.simplify(E_r_expected)
        E_theta_expected_simp = sp.simplify(E_theta_expected)
        diff_r = sp.simplify(E_r_given - E_r_expected_simp)
        diff_theta = sp.simplify(E_theta_given - E_theta_expected_simp)
        if diff_r == 0 and diff_theta == 0:
            return True, "Large-r field matches gradient of asymptotic potential."
        else:
            # Maybe the given expression uses different notation; try numeric check
            return False, f"Large-r field mismatch: ΔE_r = {diff_r}, ΔE_θ = {diff_theta}"

    def test_trace_electric_field_small(self) -> Tuple[bool, str]:
        """
        Check consistency of small-r field with asymptotic potential.
        V_asymp = V0/2 + (2V0 r sinθ)/(π a)
        """
        if self.E_r_small is None or self.E_theta_small is None:
            return False, "SKIP: Small-r field components not available."
        if any(x is None for x in [self.a, self.V0, self.r, self.theta]):
            return False, "SKIP: Missing parameters."
        try:
            E_r_given = sp.sympify(self.E_r_small)
            E_theta_given = sp.sympify(self.E_theta_small)
        except:
            return False, "Cannot convert E fields to sympy."

        r, theta, a, V0 = sp.symbols('r theta a V0', real=True)
        V_asymp = V0/2 + (2*V0*r*sp.sin(theta))/(sp.pi*a)
        E_r_expected = -sp.diff(V_asymp, r)
        E_theta_expected = -(1/r)*sp.diff(V_asymp, theta)
        E_r_expected_simp = sp.simplify(E_r_expected)
        E_theta_expected_simp = sp.simplify(E_theta_expected)
        diff_r = sp.simplify(E_r_given - E_r_expected_simp)
        diff_theta = sp.simplify(E_theta_given - E_theta_expected_simp)
        if diff_r == 0 and diff_theta == 0:
            return True, "Small-r field matches gradient of asymptotic potential."
        else:
            return False, f"Small-r field mismatch: ΔE_r = {diff_r}, ΔE_θ = {diff_theta}"

    # --------------------------------------------------------------
    # Monotonicity tests (trends via perturbation)
    # --------------------------------------------------------------
    def test_monotonicity_field_strength_large_r(self) -> Tuple[bool, str]:
        """
        For fixed θ = π/2, E_r_large ∝ 1/r². Increase r by 10% should decrease |E_r| by ~17%.
        """
        if self.E_r_large is None or self.a is None or self.V0 is None:
            return False, "SKIP: Missing data for large-r field monotonicity."
        r0 = 10 * self.a  # large enough
        theta0 = math.pi / 2
        try:
            E_func = sp.lambdify((self.r, self.theta), self.E_r_large, 'math')
        except:
            return False, "Cannot evaluate E_r_large numerically."
        try:
            E0 = E_func(r0, theta0)
            E_pert = E_func(r0 * 1.1, theta0)
        except Exception as e:
            return False, f"Evaluation error: {e}"
        if E0 == 0:
            return False, "Baseline field is zero, cannot test monotonicity."
        ratio = E_pert / E0
        # Expected ratio: (r0/r0*1.1)^2 = 1/1.21 ≈ 0.8264
        if math.isclose(ratio, 1/1.21, rel_tol=self.rel_tol):
            return True, f"E_r decreases appropriately: ratio {ratio} (expected ~0.826)."
        else:
            return False, f"Monotonicity fail: ratio {ratio} != expected ~0.826."

    def test_monotonicity_field_theta_sign(self) -> Tuple[bool, str]:
        """
        Check that E_theta_large = -(2V0 a cosθ)/(π r²). For 0<θ<π/2, cosθ>0 => E_theta negative.
        For π/2<θ<π, cosθ<0 => E_theta positive. Check sign consistency.
        This is a sign monotonicity, not a true perturbation test, but it checks trend across angles.
        To satisfy the perturbation requirement, we'll also perturb θ slightly and check that E_theta changes sign appropriately.
        """
        if self.E_theta_large is None or self.a is None or self.V0 is None:
            return False, "SKIP: Missing data."
        r0 = 5 * self.a
        try:
            E_func = sp.lambdify((self.r, self.theta), self.E_theta_large, 'math')
        except:
            return False, "Cannot evaluate E_theta_large."
        angles = [math.pi/4, 3*math.pi/4]  # one in each half
        for theta_val in angles:
            try:
                val = E_func(r0, theta_val)
            except:
                continue
            if theta_val < math.pi/2:
                if val >= 0:
                    return False, f"E_theta should be negative for θ={theta_val}, got {val}."
            else:
                if val <= 0:
                    return False, f"E_theta should be positive for θ={theta_val}, got {val}."
        # Also perturb θ by small amount and check response
        try:
            val_center = E_func(r0, math.pi/4)
            val_pert = E_func(r0, math.pi/4 + 0.1)  # move toward π/2
        except:
            return False, "Error evaluating perturbed field."
        # For θ in (0, π/2), E_theta is negative and becomes less negative as θ increases (closer to zero at π/2)
        if val_center < val_pert:  # less negative = larger value
            pass
        else:
            return False, f"Monotonicity: E_theta should increase with θ near π/4, but {val_center} -> {val_pert}."
        return True, "Sign and monotonicity of E_theta_large consistent with formula."

    # --------------------------------------------------------------
    # Answer sanity (weak, final-answer related)
    # --------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """
        Weak check: all computed quantities are not None and produce finite numeric values
        when evaluated at a sample point. No comparison against expected values.
        """
        if self.V is None:
            return False, "V is missing."
        if self.a is None or self.V0 is None:
            return False, "SKIP: Cannot evaluate without a and V0."
        # Try to evaluate V at a sample point (r=2a, θ=π/4)
        try:
            V_func = sp.lambdify((self.r, self.theta), self.V, 'numpy')
            val = V_func(2*self.a, math.pi/4)
            if not math.isfinite(val):
                return False, f"V evaluated to non-finite value: {val}"
        except Exception as e:
            return False, f"Error evaluating V: {e}"
        # Additional very weak checks: V should be between 0 and V0 for physical reason, but we only check finiteness.
        return True, "V is finite at a sample point."

    # --------------------------------------------------------------
    # Main runner
    # --------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods, execute them in try/except, collect logs.
        Returns (all_pass, logs_list).
        """
        logs = []
        all_pass = True
        # Sort methods to ensure deterministic order
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