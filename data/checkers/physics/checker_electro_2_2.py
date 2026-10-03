import sympy as sp
import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits a solution for the magnetic field of a long cylindrical conductor
    with uniform current density and a hollow inner region.

    Expected result keys (from pseudocode [emit] contract):
        - B_phi1 (r < a)
        - B_phi2 (a < r < b)
        - B_phi3 (r > b)
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens (safe defaults: None)
        self.I = getattr(solution_instance, 'I', None)
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)

        # Extract results (from dict or attribute)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}
        self.B_phi1 = results.get('B_phi1', None)
        self.B_phi2 = results.get('B_phi2', None)
        self.B_phi3 = results.get('B_phi3', None)

        # Radial variable (may be provided as Symbol or numeric, default to Symbol)
        self.r = getattr(solution_instance, 'r', sp.Symbol('r', positive=True))

        # Sample numeric values for evaluation (SI units)
        self.sample_I = 1.0
        self.sample_a = 0.01
        self.sample_b = 0.02
        self.sample_mu0 = 4 * math.pi * 1e-7
        self.sample_r_inside = 0.005
        self.sample_r_conductor = 0.015
        self.sample_r_outside = 0.03

    # -------------------------------------------------------------------
    # Helper: evaluate an expression with given parameters (substitution)
    # -------------------------------------------------------------------
    def _eval_expr_at_params(self, expr: Any, r_val: float,
                             I_val: float = None, a_val: float = None,
                             b_val: float = None, mu0_val: float = None) -> float:
        """Evaluate expression at r_val, substituting provided parameter values.
        If a parameter is None, use the sample default."""
        if expr is None:
            return None
        if not isinstance(expr, sp.Expr):
            try:
                return float(expr)
            except (TypeError, ValueError):
                return None
        # Build substitution dict
        subs = {}
        if I_val is None:
            I_val = self.sample_I
        if a_val is None:
            a_val = self.sample_a
        if b_val is None:
            b_val = self.sample_b
        if mu0_val is None:
            mu0_val = self.sample_mu0

        # Only substitute if the symbol exists in the expression
        for sym, val in [(self.I, I_val), (self.a, a_val), (self.b, b_val),
                         (self.mu0, mu0_val), (self.r, r_val)]:
            if sym is not None and sym in expr.free_symbols:
                subs[sym] = val
        try:
            val = float(expr.subs(subs).evalf())
            return val
        except Exception:
            return None

    # -------------------------------------------------------------------
    # Interface tests
    # -------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all three magnetic field results exist."""
        missing = []
        if self.B_phi1 is None:
            missing.append('B_phi1')
        if self.B_phi2 is None:
            missing.append('B_phi2')
        if self.B_phi3 is None:
            missing.append('B_phi3')
        if missing:
            return False, f"Missing result keys: {', '.join(missing)}"
        return True, "All required result keys present"

    # -------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # -------------------------------------------------------------------
    def test_invariant_ampere_law_inside(self) -> Tuple[bool, str]:
        """For r < a, B should be exactly zero."""
        if self.B_phi1 is None:
            return False, "SKIP: B_phi1 not available"
        # Symbolic check
        try:
            expr = sp.sympify(self.B_phi1)
            if sp.simplify(expr) == 0:
                return True, "B_phi1 is symbolically zero"
        except Exception:
            pass
        # Numeric fallback
        val = self._eval_expr_at_params(self.B_phi1, self.sample_r_inside)
        if val is None:
            return False, "Cannot evaluate B_phi1 numerically"
        if math.isclose(val, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            return True, f"B_phi1 = {val} ≈ 0 (inside)"
        return False, f"B_phi1 = {val} is not zero"

    def test_invariant_ampere_law_outside(self) -> Tuple[bool, str]:
        """For r > b, B_phi should equal mu0*I/(2*pi*r)."""
        if self.B_phi3 is None:
            return False, "SKIP: B_phi3 not available"
        # Build expected expression
        expected = self.sample_mu0 * self.sample_I / (2 * math.pi * self.sample_r_outside)
        val = self._eval_expr_at_params(self.B_phi3, self.sample_r_outside)
        if val is None:
            return False, "Cannot evaluate B_phi3 numerically"
        if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"B_phi3 = {val} ≈ {expected} (outside)"
        return False, f"B_phi3 = {val}, expected {expected}"

    def test_invariant_ampere_law_conductor(self) -> Tuple[bool, str]:
        """For a<r<b, B_phi = mu0*I/(2πr) * (r^2 - a^2)/(b^2 - a^2)."""
        if self.B_phi2 is None:
            return False, "SKIP: B_phi2 not available"
        r = self.sample_r_conductor
        expected = (self.sample_mu0 * self.sample_I / (2 * math.pi * r)) * (
            (r**2 - self.sample_a**2) / (self.sample_b**2 - self.sample_a**2))
        val = self._eval_expr_at_params(self.B_phi2, r)
        if val is None:
            return False, "Cannot evaluate B_phi2 numerically"
        if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"B_phi2 = {val} ≈ {expected} (inside conductor)"
        return False, f"B_phi2 = {val}, expected {expected}"

    # -------------------------------------------------------------------
    # Modeling / trace tests
    # -------------------------------------------------------------------
    def test_modeling_enclosed_current_inside_conductor(self) -> Tuple[bool, str]:
        """
        From B_phi2, compute I_enc = (2πr B_phi2)/μ0 and check
        it matches I*(r^2 - a^2)/(b^2 - a^2).
        """
        if self.B_phi2 is None:
            return False, "SKIP: B_phi2 not available"
        r_val = self.sample_r_conductor
        B_val = self._eval_expr_at_params(self.B_phi2, r_val)
        if B_val is None:
            return False, "Cannot evaluate B_phi2 for enclosed current check"
        I_enc_computed = (2 * math.pi * r_val * B_val) / self.sample_mu0
        I_enc_expected = self.sample_I * (r_val**2 - self.sample_a**2) / (self.sample_b**2 - self.sample_a**2)
        if math.isclose(I_enc_computed, I_enc_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"I_enc computed = {I_enc_computed}, expected {I_enc_expected}"
        return False, f"I_enc mismatch: {I_enc_computed} vs {I_enc_expected}"

    def test_modeling_enclosed_current_outside(self) -> Tuple[bool, str]:
        """
        From B_phi3, compute I_enc = (2πr B_phi3)/μ0 and check it equals I.
        """
        if self.B_phi3 is None:
            return False, "SKIP: B_phi3 not available"
        r_val = self.sample_r_outside
        B_val = self._eval_expr_at_params(self.B_phi3, r_val)
        if B_val is None:
            return False, "Cannot evaluate B_phi3 for enclosed current check"
        I_enc_computed = (2 * math.pi * r_val * B_val) / self.sample_mu0
        if math.isclose(I_enc_computed, self.sample_I, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"I_enc computed = {I_enc_computed} ≈ I = {self.sample_I}"
        return False, f"I_enc computed = {I_enc_computed}, expected {self.sample_I}"

    # -------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # -------------------------------------------------------------------
    def test_monotonicity_B_scales_with_current_outside(self) -> Tuple[bool, str]:
        """Perturb I by +5% and check B_phi3 scales proportionally."""
        if self.B_phi3 is None:
            return False, "SKIP: B_phi3 not available"
        r_val = self.sample_r_outside
        baseline_I = self.sample_I
        perturbed_I = baseline_I * 1.05   # +5%

        B_base = self._eval_expr_at_params(self.B_phi3, r_val, I_val=baseline_I)
        B_pert = self._eval_expr_at_params(self.B_phi3, r_val, I_val=perturbed_I)
        if B_base is None or B_pert is None:
            return False, "Cannot evaluate B_phi3 for monotonicity check"

        # Expected ratio: B_pert/B_base = perturbed_I/baseline_I = 1.05
        ratio = B_pert / B_base if B_base != 0 else 0
        expected_ratio = 1.05
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
            return True, (f"Base I={baseline_I}, B={B_base:.6e}; "
                          f"Perturbed I={perturbed_I}, B={B_pert:.6e}; "
                          f"Ratio={ratio:.4f} ≈ {expected_ratio}")
        return False, f"Ratio {ratio:.4f} does not match expected {expected_ratio}"

    def test_monotonicity_B_scales_with_permeability_outside(self) -> Tuple[bool, str]:
        """Perturb mu0 by +5% and check B_phi3 scales proportionally."""
        if self.B_phi3 is None:
            return False, "SKIP: B_phi3 not available"
        r_val = self.sample_r_outside
        baseline_mu0 = self.sample_mu0
        perturbed_mu0 = baseline_mu0 * 1.05   # +5%

        B_base = self._eval_expr_at_params(self.B_phi3, r_val, mu0_val=baseline_mu0)
        B_pert = self._eval_expr_at_params(self.B_phi3, r_val, mu0_val=perturbed_mu0)
        if B_base is None or B_pert is None:
            return False, "Cannot evaluate B_phi3 for monotonicity check"

        ratio = B_pert / B_base if B_base != 0 else 0
        expected_ratio = 1.05
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
            return True, (f"Base mu0={baseline_mu0:.6e}, B={B_base:.6e}; "
                          f"Perturbed mu0={perturbed_mu0:.6e}, B={B_pert:.6e}; "
                          f"Ratio={ratio:.4f} ≈ {expected_ratio}")
        return False, f"Ratio {ratio:.4f} does not match expected {expected_ratio}"

    def test_monotonicity_B_scales_with_current_inside_conductor(self) -> Tuple[bool, str]:
        """Perturb I by +5% and check B_phi2 scales proportionally."""
        if self.B_phi2 is None:
            return False, "SKIP: B_phi2 not available"
        r_val = self.sample_r_conductor
        baseline_I = self.sample_I
        perturbed_I = baseline_I * 1.05   # +5%

        B_base = self._eval_expr_at_params(self.B_phi2, r_val, I_val=baseline_I)
        B_pert = self._eval_expr_at_params(self.B_phi2, r_val, I_val=perturbed_I)
        if B_base is None or B_pert is None:
            return False, "Cannot evaluate B_phi2 for monotonicity check"

        ratio = B_pert / B_base if B_base != 0 else 0
        expected_ratio = 1.05
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
            return True, (f"Base I={baseline_I}, B={B_base:.6e}; "
                          f"Perturbed I={perturbed_I}, B={B_pert:.6e}; "
                          f"Ratio={ratio:.4f} ≈ {expected_ratio}")
        return False, f"Ratio {ratio:.4f} does not match expected {expected_ratio}"

    # -------------------------------------------------------------------
    # Weak answer sanity (only one, no derivation, no isclose)
    # -------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak check: B_phi1 must be exactly zero, B_phi2 and B_phi3 must be
        finite and positive (for typical parameters). No re-derivation.
        """
        # Check B_phi1 zero
        if self.B_phi1 is None:
            return False, "SKIP: B_phi1 not available"
        try:
            val1 = float(sp.N(self.B_phi1))
        except Exception:
            return False, "Cannot convert B_phi1 to float"
        if not math.isfinite(val1) or abs(val1) > 1e-12:
            return False, f"B_phi1 = {val1} is not zero"

        # Check B_phi2 finite and positive
        if self.B_phi2 is None:
            return False, "SKIP: B_phi2 not available"
        val2 = self._eval_expr_at_params(self.B_phi2, self.sample_r_conductor)
        if val2 is None or not math.isfinite(val2) or val2 <= 0:
            return False, f"B_phi2 = {val2} is not finite positive"

        # Check B_phi3 finite and positive
        if self.B_phi3 is None:
            return False, "SKIP: B_phi3 not available"
        val3 = self._eval_expr_at_params(self.B_phi3, self.sample_r_outside)
        if val3 is None or not math.isfinite(val3) or val3 <= 0:
            return False, f"B_phi3 = {val3} is not finite positive"

        return True, "Weak sanity passed: B_phi1=0, B_phi2>0, B_phi3>0"

    # -------------------------------------------------------------------
    # Main runner
    # -------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Discover all test methods
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
                # Optionally include traceback for debugging
                # logs.append(traceback.format_exc())
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)