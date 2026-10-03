import sympy as sp
import math
from typing import Any, Dict, List, Tuple, Optional, Callable

class PhysicsChecker:
    """Audits the finite-nucleus effect solution for atomic physics."""

    def __init__(self, solution_instance: Any):
        # ---------- safe extraction of solution_results ----------
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results: Dict[str, Any] = solution_instance.results
        else:
            self.results = {}

        # Extract key result quantities (safe defaults = None)
        self.V_out: Optional[Any] = self.results.get('V_out', None)          # expected: -Z/r (sympy/str)
        self.V_in: Optional[Any] = self.results.get('V_in', None)            # expected: -Z/(2R)*(3 - (r/R)^2)
        self.delta_E_formula: Optional[Any] = self.results.get('delta_E_formula', None)  # (4/5)*Z^2*|E_0|*(R/a0)^2
        self.delta_E_cm: Optional[Any] = self.results.get('delta_E_cm', None)            # numerical value

        # ---------- extract givens from solution_instance ----------
        # Try 'givens' attribute, then fallback to attributes directly
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
        else:
            givens = {}
        self.Z: Optional[float] = givens.get('Z', None) or getattr(solution_instance, 'Z', None)
        self.A: Optional[float] = givens.get('A', None) or getattr(solution_instance, 'A', None)
        self.r0: Optional[float] = givens.get('r0', None) or getattr(solution_instance, 'r0', None)
        self.a0: Optional[float] = givens.get('a0', None) or getattr(solution_instance, 'a0', None)

        # Physical constants (not to be patched as defaults)
        self.hartree_to_cm = 219474.63  # 1 Hartree in cm^-1
        self.bohr_radius_m = 0.529e-10  # m
        self.default_r0 = 1.2e-15      # m

        # Define independent variables (symbolic)
        self.r = sp.Symbol('r', real=True, positive=True)
        self.R_sym = sp.Symbol('R', real=True, positive=True)
        self.Z_sym = sp.Symbol('Z', real=True, positive=True)
        self.a0_sym = sp.Symbol('a0', real=True, positive=True)
        self.E0_sym = sp.Symbol('|E_0|', real=True, positive=True)  # magnitude

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------
    def _safe_sympify(self, expr: Any) -> Optional[sp.Expr]:
        """Convert expression to SymPy, return None if impossible."""
        if expr is None:
            return None
        try:
            if isinstance(expr, str):
                return sp.sympify(expr)
            elif isinstance(expr, sp.Basic):
                return expr
            else:
                return sp.sympify(str(expr))
        except Exception:
            return None

    def _evaluate_expr(self, expr_sympy: sp.Expr,
                       subs_dict: Dict[sp.Symbol, float]) -> Optional[float]:
        """Substitute and evaluate, return None on failure."""
        try:
            val = expr_sympy.subs(subs_dict).evalf()
            return float(val)
        except Exception:
            return None

    def _get_R_from_givens(self) -> Optional[float]:
        """Compute R = r0 * A^(1/3) if givens available."""
        if None in (self.r0, self.A):
            return None
        return self.r0 * (self.A ** (1/3))

    def _get_baseline_subs(self) -> Dict[sp.Symbol, float]:
        """Return substitution dict for standard numeric check."""
        subs = {}
        if self.Z is not None:
            subs[self.Z_sym] = self.Z
        if self.a0 is not None:
            subs[self.a0_sym] = self.a0
        # Use symbolic R if not numeric; prefer numeric from givens
        R_val = self._get_R_from_givens()
        if R_val is not None:
            subs[self.R_sym] = R_val
        # For E0 magnitude: E0 = Z^2/(2) in atomic units
        if self.Z is not None:
            subs[self.E0_sym] = self.Z**2 / 2.0
        return subs

    # -----------------------------------------------------------------
    # Interface tests (keys must exist)
    # -----------------------------------------------------------------
    def test_interface_V_out_exists(self) -> Tuple[bool, str]:
        if self.V_out is None:
            return (False, "Missing 'V_out' in results")
        return (True, "V_out present")

    def test_interface_V_in_exists(self) -> Tuple[bool, str]:
        if self.V_in is None:
            return (False, "Missing 'V_in' in results")
        return (True, "V_in present")

    def test_interface_delta_E_formula_exists(self) -> Tuple[bool, str]:
        if self.delta_E_formula is None:
            return (False, "Missing 'delta_E_formula' in results")
        return (True, "delta_E_formula present")

    def test_interface_delta_E_cm_exists(self) -> Tuple[bool, str]:
        if self.delta_E_cm is None:
            return (False, "Missing 'delta_E_cm' in results")
        return (True, "delta_E_cm present")

    # -----------------------------------------------------------------
    # Invariant (physics theorem) tests
    # -----------------------------------------------------------------
    def test_invariant_V_out_form(self) -> Tuple[bool, str]:
        """V_out should be -Z/r (point charge) for r>R."""
        if self.V_out is None:
            return (False, "SKIP: V_out missing")
        expr = self._safe_sympify(self.V_out)
        if expr is None:
            return (False, "V_out not convertible to SymPy expression")
        # Check that after substitution r=2R, V_out = -Z/(2R)
        Z_test = 82
        R_test = 7.0e-15
        subs = {self.Z_sym: Z_test, self.R_sym: R_test, self.r: 2*R_test}
        expected = -Z_test / (2*R_test)
        computed = self._evaluate_expr(expr, subs)
        if computed is None:
            return (False, "Could not evaluate V_out")
        if math.isclose(computed, expected, rel_tol=1e-2):
            return (True, f"V_out(2R) ≈ {computed:.3e}, expected {expected:.3e}")
        else:
            return (False, f"V_out(2R) = {computed:.3e}, expected {expected:.3e}")

    def test_invariant_V_in_continuity(self) -> Tuple[bool, str]:
        """At r=R, V_in should equal V_out = -Z/R."""
        if self.V_in is None or self.V_out is None:
            return (False, "SKIP: V_in or V_out missing")
        expr_in = self._safe_sympify(self.V_in)
        expr_out = self._safe_sympify(self.V_out)
        if expr_in is None or expr_out is None:
            return (False, "Could not sympify expressions")
        Z_test = 82
        R_test = 7.0e-15
        subs = {self.Z_sym: Z_test, self.R_sym: R_test, self.r: R_test}
        val_in = self._evaluate_expr(expr_in, subs)
        val_out = self._evaluate_expr(expr_out, subs)
        if val_in is None or val_out is None:
            return (False, "Evaluation failure")
        if math.isclose(val_in, val_out, rel_tol=1e-2):
            return (True, f"V_in(R) ≈ {val_in:.3e}, V_out(R) ≈ {val_out:.3e}")
        else:
            return (False, f"V_in(R) = {val_in:.3e}, V_out(R) = {val_out:.3e}")

    def test_invariant_V_in_quadratic_ratio(self) -> Tuple[bool, str]:
        """V_in(0)/V_in(R) should be 3/2."""
        if self.V_in is None:
            return (False, "SKIP: V_in missing")
        expr = self._safe_sympify(self.V_in)
        if expr is None:
            return (False, "Cannot parse V_in")
        Z_test = 82
        R_test = 7.0e-15
        subs0 = {self.Z_sym: Z_test, self.R_sym: R_test, self.r: 0}
        subsR = {self.Z_sym: Z_test, self.R_sym: R_test, self.r: R_test}
        val0 = self._evaluate_expr(expr, subs0)
        valR = self._evaluate_expr(expr, subsR)
        if val0 is None or valR is None or valR == 0:
            return (False, "Evaluation error or division by zero")
        ratio = val0 / valR
        expected_ratio = 3/2
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"V_in(0)/V_in(R) ≈ {ratio:.3f}, expected {expected_ratio}")
        else:
            return (False, f"Ratio = {ratio:.3f}, expected {expected_ratio}")

    def test_invariant_delta_E_positive(self) -> Tuple[bool, str]:
        """Delta E should be positive (finite nucleus reduces binding)."""
        if self.delta_E_cm is None:
            return (False, "SKIP: delta_E_cm missing")
        try:
            val = float(self.delta_E_cm)
        except (TypeError, ValueError):
            return (False, "delta_E_cm not numeric")
        if val > 0:
            return (True, f"delta_E_cm = {val:.3e} > 0")
        else:
            return (False, f"delta_E_cm = {val:.3e} <= 0 (invalid)")

    # -----------------------------------------------------------------
    # Modeling quality & trace checks
    # -----------------------------------------------------------------
    def test_modeling_delta_E_formula_dependence(self) -> Tuple[bool, str]:
        """Check the scaling: delta_E ~ Z^4 * (R/a0)^2.
           Use baseline values to verify order 0.3 Hartree."""
        if self.delta_E_formula is None:
            return (False, "SKIP: delta_E_formula missing")
        expr = self._safe_sympify(self.delta_E_formula)
        if expr is None:
            return (False, "Cannot parse delta_E_formula")
        # Use standard numerical substitution
        subs = self._get_baseline_subs()
        if self.Z is None or self.a0 is None or self._get_R_from_givens() is None:
            return (False, "SKIP: missing givens (Z, a0, R) to evaluate formula")
        computed = self._evaluate_expr(expr, subs)
        if computed is None:
            return (False, "Evaluation of delta_E_formula failed")
        # Expected: delta_E = (2/5)*Z^4*(R/a0)^2  (in Hartree)
        Z_val = self.Z
        R_val = self._get_R_from_givens()
        a0_val = self.a0
        expected = (2.0/5.0) * (Z_val**4) * ((R_val/a0_val)**2)
        if math.isclose(computed, expected, rel_tol=1e-2):
            return (True, f"delta_E_formula ≈ {computed:.3e} Hartree, expected {expected:.3e}")
        else:
            return (False, f"delta_E_formula = {computed:.3e} Hartree, expected {expected:.3e}")

    def test_modeling_wavefunction_approximation(self) -> Tuple[bool, str]:
        """Check that R/a0 << 1 (valid approximation)."""
        R_val = self._get_R_from_givens()
        a0_val = self.a0
        if R_val is None or a0_val is None:
            return (False, "SKIP: R or a0 not available")
        ratio = R_val / a0_val
        if ratio < 0.01:
            return (True, f"R/a0 = {ratio:.3e} << 1, approximation valid")
        else:
            return (False, f"R/a0 = {ratio:.3e} not << 1, approximation questionable")

    def test_modeling_delta_E_cm_consistency(self) -> Tuple[bool, str]:
        """Compute expected delta_E in cm^-1 using formula and conversion, compare with provided delta_E_cm."""
        if self.delta_E_cm is None or self.delta_E_formula is None:
            return (False, "SKIP: required keys missing")
        # First get delta_E in Hartree from formula
        expr_formula = self._safe_sympify(self.delta_E_formula)
        if expr_formula is None:
            return (False, "delta_E_formula not parseable")
        subs = self._get_baseline_subs()
        if self.Z is None or self.a0 is None or self._get_R_from_givens() is None:
            return (False, "SKIP: missing givens for numeric evaluation")
        delta_E_hartree = self._evaluate_expr(expr_formula, subs)
        if delta_E_hartree is None:
            return (False, "Evaluation of delta_E_formula failed")
        expected_cm = delta_E_hartree * self.hartree_to_cm
        try:
            provided_cm = float(self.delta_E_cm)
        except (TypeError, ValueError):
            return (False, "delta_E_cm not numeric")
        if math.isclose(provided_cm, expected_cm, rel_tol=1e-1):  # 10% tolerance due to rounding
            return (True, f"delta_E_cm ≈ {provided_cm:.3e}, expected {expected_cm:.3e}")
        else:
            return (False, f"delta_E_cm = {provided_cm:.3e}, expected from formula {expected_cm:.3e}")

    # -----------------------------------------------------------------
    # Monotonicity tests (perturb inputs)
    # -----------------------------------------------------------------
    def test_monotonicity_increase_with_Z(self) -> Tuple[bool, str]:
        """Increase Z by 10% should increase delta_E (since Z^4)."""
        if self.delta_E_formula is None:
            return (False, "SKIP: delta_E_formula missing")
        expr = self._safe_sympify(self.delta_E_formula)
        if expr is None:
            return (False, "Parse error")
        base_subs = self._get_baseline_subs()
        if self.Z is None or self.a0 is None or self._get_R_from_givens() is None:
            return (False, "SKIP: missing givens")
        perturbed_subs = base_subs.copy()
        Z_pert = self.Z * 1.1
        perturbed_subs[self.Z_sym] = Z_pert
        # Need to update E0 magnitude accordingly? The formula likely contains |E_0| which is Z^2/2, so we must update it.
        # We'll update E0_sym as well if present in subs.
        # Since we may have used E0_sym in formula, we need to correct.
        # Better: Instead, evaluate directly by substituting symbolic expression and then compute.
        # To be safe, we'll replace any occurrence of |E_0| symbol with Z^2/2 manually.
        # We'll create a version of expr where |E_0| is replaced by (Z^2/2).
        # But we don't know if formula uses that symbol exactly; the pseudocode says delta_E_formula = (4/5)*Z^2*|E_0|*(R/a0)^2.
        # So we assume it uses the symbol |E_0|. We'll substitute that symbol.
        # For base case, we already set base_subs[self.E0_sym] = self.Z**2/2.
        # For perturbed, we set perturbed_subs[self.E0_sym] = Z_pert**2/2.
        perturbed_subs[self.E0_sym] = Z_pert**2 / 2.0
        base_val = self._evaluate_expr(expr, base_subs)
        perturbed_val = self._evaluate_expr(expr, perturbed_subs)
        if base_val is None or perturbed_val is None:
            return (False, "Evaluation error")
        if perturbed_val > base_val:
            return (True, f"Z increased 10%: delta_E from {base_val:.3e} to {perturbed_val:.3e} (increase)")
        else:
            return (False, f"Z increased but delta_E did not increase: {base_val:.3e} -> {perturbed_val:.3e}")

    def test_monotonicity_increase_with_R(self) -> Tuple[bool, str]:
        """Increase R by 10% should increase delta_E (since (R/a0)^2)."""
        if self.delta_E_formula is None:
            return (False, "SKIP: delta_E_formula missing")
        expr = self._safe_sympify(self.delta_E_formula)
        if expr is None:
            return (False, "Parse error")
        R_val = self._get_R_from_givens()
        if R_val is None or self.Z is None or self.a0 is None:
            return (False, "SKIP: missing givens")
        # Build base and perturbed subs with R perturbed
        base_subs = self._get_baseline_subs()
        perturbed_subs = base_subs.copy()
        R_pert = R_val * 1.1
        perturbed_subs[self.R_sym] = R_pert
        base_val = self._evaluate_expr(expr, base_subs)
        perturbed_val = self._evaluate_expr(expr, perturbed_subs)
        if base_val is None or perturbed_val is None:
            return (False, "Evaluation error")
        if perturbed_val > base_val:
            return (True, f"R increased 10%: delta_E from {base_val:.3e} to {perturbed_val:.3e} (increase)")
        else:
            return (False, f"R increased but delta_E did not increase: {base_val:.3e} -> {perturbed_val:.3e}")

    # -----------------------------------------------------------------
    # Weak answer sanity (only one such test)
    # -----------------------------------------------------------------
    def test_answer_sanity_delta_E_cm_order(self) -> Tuple[bool, str]:
        """Check that delta_E_cm lies in a reasonable range (10^4 - 10^6 cm^{-1})."""
        if self.delta_E_cm is None:
            return (False, "SKIP: delta_E_cm missing")
        try:
            val = float(self.delta_E_cm)
        except (TypeError, ValueError):
            return (False, "delta_E_cm not numeric")
        if 1e4 <= val <= 1e6:
            return (True, f"delta_E_cm = {val:.3e} in plausible range [1e4,1e6] cm^{-1}")
        else:
            return (False, f"delta_E_cm = {val:.3e} outside plausible range [1e4,1e6]")

    # -----------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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