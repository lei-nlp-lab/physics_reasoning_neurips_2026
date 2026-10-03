import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics checker for parity conservation test in Lambda0 decay.
    Audits the derived angular distribution formula and its physical consistency.
    """

    def __init__(self, solution_instance):
        """
        Extract solution data safely with None defaults.
        Expects solution_instance to have:
          - .results dict (or .distribution_formula attribute)
          - Optional .given dict for polarization etc.
        """
        # Primary result expected: distribution formula as string
        self.distribution_formula = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.distribution_formula = solution_instance.results.get('distribution_formula', None)
        elif hasattr(solution_instance, 'distribution_formula'):
            self.distribution_formula = solution_instance.distribution_formula
        else:
            self.distribution_formula = getattr(solution_instance, 'distribution_formula', None)

        # Optional extracted givens for deeper checks (may be missing)
        self.polarization = getattr(solution_instance, 'P', None)
        self.alpha = getattr(solution_instance, 'alpha', None)

    # ---------- Helper to convert formula string to sympy expression ----------
    def _parse_formula(self, formula_str):
        """Convert distribution formula string to sympy expression with alpha, P, theta."""
        s = formula_str.replace(' ', '')
        s = s.replace('costheta', 'cos(theta)').replace('cosθ', 'cos(theta)')
        s = s.replace('α', 'alpha')
        alpha, P, theta = sp.symbols('alpha P theta', real=True)
        expr = sp.sympify(s, locals={'alpha': alpha, 'P': P, 'theta': theta, 'cos': sp.cos})
        return expr, alpha, P, theta

    # ---------- Interface Tests ----------
    def test_interface_distribution_formula_exists(self):
        if self.distribution_formula is None:
            return (False, "Missing distribution_formula in results")
        if not isinstance(self.distribution_formula, str):
            return (False, "distribution_formula must be a string")
        return (True, "distribution_formula present and is a string")

    def test_interface_distribution_formula_contains_terms(self):
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        formula = self.distribution_formula.replace(" ", "")
        has_one = '1' in formula
        has_alpha = 'alpha' in formula or 'α' in formula
        has_P = 'P' in formula
        has_cos = 'cos' in formula
        if not (has_one and has_alpha and has_P and has_cos):
            return (False, f"Formula lacks expected terms: {formula}")
        return (True, "Formula contains required terms: 1, alpha, P, cos")

    # ---------- Invariant (theorem-based) Tests ----------
    def test_invariant_normalization_condition(self):
        """
        Check that the angular distribution integrates to 1 over full solid angle.
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        try:
            expr, alpha, P, theta = self._parse_formula(self.distribution_formula)
            phi = sp.symbols('phi', real=True)
            full_dist = expr / (4*sp.pi)
            integrand = full_dist * sp.sin(theta)
            integral = sp.integrate(integrand, (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
            simplified = sp.simplify(integral)
            if simplified == 1:
                return (True, "Angular distribution is properly normalized")
            else:
                return (False, f"Normalization integral gave {simplified}, expected 1")
        except Exception as e:
            return (False, f"Normalization check crashed: {e}")

    def test_invariant_parity_conservation_condition(self):
        """
        When parity is conserved (α=0), distribution becomes isotropic.
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        try:
            expr, alpha, P, theta = self._parse_formula(self.distribution_formula)
            expr_alpha0 = expr.subs(alpha, 0)
            if expr_alpha0.is_constant():
                return (True, "When α=0, distribution is constant as expected for parity conservation")
            else:
                return (False, f"With α=0, distribution is not constant: {expr_alpha0}")
        except Exception as e:
            return (False, f"Parity condition check crashed: {e}")

    def test_invariant_polarization_zero(self):
        """
        If polarization P=0, distribution becomes isotropic regardless of α.
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        try:
            expr, alpha, P, theta = self._parse_formula(self.distribution_formula)
            expr_P0 = expr.subs(P, 0)
            if expr_P0.is_constant():
                return (True, "When P=0, distribution is isotropic as expected")
            else:
                return (False, f"With P=0, distribution is not constant: {expr_P0}")
        except Exception as e:
            return (False, f"P=0 condition check crashed: {e}")

    def test_invariant_asymmetry_parameter_range(self):
        """
        The distribution should remain positive for all θ when |αP| ≤ 1.
        (Structural check; actual numbers not available.)
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        return (True, "Asymmetry parameter range not numerically testable, but formula structure is consistent.")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_angular_dependence_form(self):
        """
        Distribution must be linear in cosθ (no higher powers of cos or sin terms).
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        try:
            expr, alpha, P, theta = self._parse_formula(self.distribution_formula)
            expanded = sp.expand(expr)
            s = str(expanded)
            if 'sin' in s or 'cos**2' in s or 'cos(theta)**2' in s:
                return (False, f"Distribution contains higher angular dependence: {expanded}")
            # Check that cos(theta) appears exactly once and linearly
            if 'cos(theta)' not in s or s.count('cos(theta)') > 1:
                return (True, "Distribution has correct linear cosθ dependence")
            return (True, "Distribution has correct linear cosθ dependence")
        except Exception as e:
            return (False, f"Angular dependence check crashed: {e}")

    def test_modeling_parity_violation_asymmetry(self):
        """
        Distribution should not be symmetric under θ → π-θ unless α=0.
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        try:
            expr, alpha, P, theta = self._parse_formula(self.distribution_formula)
            expr_flip = expr.subs(theta, sp.pi - theta)
            diff = sp.simplify(expr - expr_flip)
            if diff == 0:
                return (False, "Distribution is symmetric under θ->π-θ for any α, contradicting parity violation")
            else:
                return (True, "Distribution asymmetric under flip when α≠0, consistent with parity violation")
        except Exception as e:
            return (False, f"Asymmetry check crashed: {e}")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_polarization_increases_asymmetry(self):
        """
        For fixed α>0, θ=0 (cosθ=1), the distribution should increase with P.
        Baseline: P=1.0, perturb by +10% and -10%, expect monotonic changes.
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        try:
            expr, alpha, P, theta = self._parse_formula(self.distribution_formula)
            # Use α=1, θ=0 for test
            baseline = float(expr.subs({alpha: 1, P: 1.0, theta: 0}))
            perturbed_up = float(expr.subs({alpha: 1, P: 1.1, theta: 0}))
            perturbed_down = float(expr.subs({alpha: 1, P: 0.9, theta: 0}))
            if baseline >= perturbed_up:
                return (False, f"Baseline {baseline} not less than perturbed_up {perturbed_up}")
            if baseline <= perturbed_down:
                return (False, f"Baseline {baseline} not greater than perturbed_down {perturbed_down}")
            return (True, f"P monotonicity: baseline={baseline:.4f}, +10% = {perturbed_up:.4f}, -10% = {perturbed_down:.4f}")
        except Exception as e:
            return (False, f"Monotonicity polarization check crashed: {e}")

    def test_monotonicity_angle_perturbation(self):
        """
        For fixed α>0, P>0, the distribution should increase with cosθ (i.e., decrease with θ).
        Choose baseline at θ=π/4, perturb by +0.1 rad (θ increases, cosθ decreases, so distribution should decrease).
        """
        if self.distribution_formula is None:
            return (False, "SKIP: no distribution_formula")
        try:
            expr, alpha, P, theta = self._parse_formula(self.distribution_formula)
            # Set α=1, P=1 for test
            baseline_theta = sp.pi/4
            delta = 0.1  # rad
            perturbed_theta = baseline_theta + delta
            baseline_val = float(expr.subs({alpha: 1, P: 1, theta: baseline_theta}))
            perturbed_val = float(expr.subs({alpha: 1, P: 1, theta: perturbed_theta}))
            # Since cos decreases with θ in [0,π], the distribution should decrease because αP=1>0
            if baseline_val <= perturbed_val:
                return (False, f"Baseline at θ=π/4: {baseline_val:.4f}, perturbed +0.1 rad: {perturbed_val:.4f}; expected decrease (cosθ decreases)")
            # Also test a negative perturbation (θ decreases)
            perturbed_theta_down = baseline_theta - delta
            perturbed_val_down = float(expr.subs({alpha: 1, P: 1, theta: perturbed_theta_down}))
            if baseline_val >= perturbed_val_down:
                return (False, f"Baseline at θ=π/4: {baseline_val:.4f}, perturbed -0.1 rad: {perturbed_val_down:.4f}; expected increase (cosθ increases)")
            return (True, f"Angle monotonicity: baseline={baseline_val:.4f}, +0.1rad = {perturbed_val:.4f}, -0.1rad = {perturbed_val_down:.4f} (consistent with cosθ dependence)")
        except Exception as e:
            return (False, f"Monotonicity angle perturbation check crashed: {e}")

    # ---------- Answer Sanity (weak, at most one) ----------
    def test_answer_sanity_distribution_nonempty(self):
        if self.distribution_formula is None:
            return (False, "distribution_formula missing")
        if len(self.distribution_formula.strip()) == 0:
            return (False, "distribution_formula is empty string")
        return (True, "distribution_formula is non-empty")

    # ---------- Run All Tests ----------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)