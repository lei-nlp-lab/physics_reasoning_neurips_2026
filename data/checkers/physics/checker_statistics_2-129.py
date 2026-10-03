import math
import sympy as sp
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    PhysicsAuditor for the ideal gas chemical potential problem (3D/2D equilibrium).
    Follows robustness rules: no numeric defaults, fuzzy matching for invariants/modeling,
    symbolic & numerical fallback, monotonicity tests with perturbations.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens and results safely (default None)
        self.givens: Optional[Dict[str, Any]] = getattr(solution_instance, 'givens', None)
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)

        # Fallback if results is None (use empty dict)
        if self.results is None:
            self.results = {}

        # Givens with None defaults (no numeric fallback values)
        self.N = None
        self.V = None
        self.A = None
        self.T = None
        self.m = None
        self.h = None
        self.k = None
        self.epsilon0 = None
        self.p = None

        if self.givens is not None:
            self.N = self.givens.get('N', None)
            self.V = self.givens.get('V', None)
            self.A = self.givens.get('A', None)
            self.T = self.givens.get('T', None)
            self.m = self.givens.get('m', None)
            self.h = self.givens.get('h', None)
            self.k = self.givens.get('k', None)
            self.epsilon0 = self.givens.get('epsilon0', None)
            self.p = self.givens.get('p', None)

        # Results from [emit] contract
        self.mu_3D = self.results.get('mu_3D', None)
        self.mu_2D = self.results.get('mu_2D', None)
        self.n = self.results.get('n', None)

    # ------------------------------------------------------------------
    # Helper: safely convert to SymPy expression
    # ------------------------------------------------------------------
    @staticmethod
    def _sympy(val: Any) -> Optional[sp.Expr]:
        if val is None:
            return None
        if isinstance(val, sp.Expr):
            return val
        try:
            ex = sp.sympify(val)
            return ex
        except (sp.SympifyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist (mu_3D, mu_2D, n)."""
        if self.results is None:
            return False, "SKIP: results is None"
        missing = []
        for key in ['mu_3D', 'mu_2D', 'n']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "All required keys present"

    def test_interface_given_keys(self) -> Tuple[bool, str]:
        """Check that important given keys exist (optional but recommended)."""
        if self.givens is None:
            return False, "SKIP: givens dictionary is None"
        expected = {'N', 'V', 'A', 'T', 'm', 'h', 'k', 'epsilon0', 'p'}
        missing = expected - set(self.givens.keys())
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All expected givens present"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_mu_3D_form(self) -> Tuple[bool, str]:
        """Check that mu_3D has the theoretical form: mu_3D = -k T * ln( V / (N * lambda^3) )."""
        expr = self._sympy(self.mu_3D)
        if expr is None:
            return False, "SKIP: mu_3D is not available or not interpretable"

        N, V, T, m, h, k = sp.symbols('N V T m h k', positive=True)
        lam = h / sp.sqrt(2 * sp.pi * m * k * T)   # thermal wavelength
        expected = -k * T * sp.log(V / (N * lam**3))

        diff = sp.simplify(expr - expected)
        if diff == 0:
            return True, "mu_3D matches theoretical form exactly"
        else:
            return self._numeric_fallback_mu3D(expr, expected)

    def _numeric_fallback_mu3D(self, expr: sp.Expr, expected: sp.Expr) -> Tuple[bool, str]:
        """If symbolic comparison fails, try numeric point check using available givens."""
        subs_dict = {}
        for sym_name in ['N', 'V', 'T', 'm', 'h', 'k']:
            sym = sp.Symbol(sym_name)
            val = getattr(self, sym_name, None)
            if val is not None:
                try:
                    subs_dict[sym] = float(sp.N(val))
                except (TypeError, ValueError):
                    return False, "Cannot perform numeric fallback: non-numeric given value"
        if not subs_dict:
            return False, "SKIP: no numeric givens available for numeric fallback"

        needed = [sp.Symbol('N'), sp.Symbol('V'), sp.Symbol('T'), sp.Symbol('m'), sp.Symbol('h'), sp.Symbol('k')]
        if any(s not in subs_dict for s in needed):
            return False, "SKIP: missing numeric values for some required symbols"

        try:
            val_expr = float(expr.subs(subs_dict).evalf())
            val_expected = float(expected.subs(subs_dict).evalf())
            if math.isclose(val_expr, val_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, "mu_3D numerically matches theoretical form (within 1%)"
            else:
                return False, f"mu_3D numerical mismatch: got {val_expr}, expected {val_expected}"
        except Exception as e:
            return False, f"Numerical fallback crashed: {e}"

    def test_invariant_mu_2D_form(self) -> Tuple[bool, str]:
        """Check that mu_2D has the theoretical form: mu_2D = -k T * ln( A / (N * lambda^2) ) - epsilon0."""
        expr = self._sympy(self.mu_2D)
        if expr is None:
            return False, "SKIP: mu_2D not available"

        N, A, T, m, h, k, eps0 = sp.symbols('N A T m h k epsilon0', positive=True)
        lam = h / sp.sqrt(2 * sp.pi * m * k * T)
        expected = -k * T * sp.log(A / (N * lam**2)) - eps0

        diff = sp.simplify(expr - expected)
        if diff == 0:
            return True, "mu_2D matches theoretical form exactly"
        else:
            return self._numeric_fallback_mu2D(expr, expected)

    def _numeric_fallback_mu2D(self, expr: sp.Expr, expected: sp.Expr) -> Tuple[bool, str]:
        subs_dict = {}
        for sym_name in ['N', 'A', 'T', 'm', 'h', 'k', 'epsilon0']:
            sym = sp.Symbol(sym_name)
            val = getattr(self, sym_name, None)
            if val is not None:
                try:
                    subs_dict[sym] = float(sp.N(val))
                except (TypeError, ValueError):
                    return False, "Cannot perform numeric fallback: non-numeric given value"
        if not subs_dict:
            return False, "SKIP: no numeric givens available"

        needed = [sp.Symbol('N'), sp.Symbol('A'), sp.Symbol('T'), sp.Symbol('m'), sp.Symbol('h'), sp.Symbol('k'), sp.Symbol('epsilon0')]
        if any(s not in subs_dict for s in needed):
            return False, "SKIP: missing numeric values for required symbols"

        try:
            val_expr = float(expr.subs(subs_dict).evalf())
            val_expected = float(expected.subs(subs_dict).evalf())
            if math.isclose(val_expr, val_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, "mu_2D numerically matches theoretical form (within 1%)"
            else:
                return False, f"mu_2D numerical mismatch: got {val_expr}, expected {val_expected}"
        except Exception as e:
            return False, f"Numerical fallback crashed: {e}"

    def test_invariant_equilibrium(self) -> Tuple[bool, str]:
        """Check that the equilibrium condition holds symbolically: mu_3D == mu_2D."""
        mu3 = self._sympy(self.mu_3D)
        mu2 = self._sympy(self.mu_2D)
        if mu3 is None or mu2 is None:
            return False, "SKIP: mu_3D or mu_2D missing"

        if sp.simplify(mu3 - mu2) == 0:
            return True, "mu_3D and mu_2D are symbolically equal"

        # Avoid numeric fallback because N_v and N_s need distinction.
        return False, "Equilibrium not symbolically equal (cannot verify numerically without separate N_v, N_s)."

    # ------------------------------------------------------------------
    # Modeling / trace tests (at least two)
    # ------------------------------------------------------------------
    def test_modeling_n_expression(self) -> Tuple[bool, str]:
        """Check that n has the theoretical form: n = (p/(kT)) * sqrt(h^2/(2πmkT)) * exp(ε0/(kT))."""
        n_expr = self._sympy(self.n)
        if n_expr is None:
            return False, "SKIP: n not available"

        p, T, m, h, k, eps0 = sp.symbols('p T m h k epsilon0', positive=True)
        expected = (p/(k*T)) * sp.sqrt(h**2/(2*sp.pi*m*k*T)) * sp.exp(eps0/(k*T))

        diff = sp.simplify(n_expr - expected)
        if diff == 0:
            return True, "n matches theoretical form exactly"
        else:
            return self._numeric_fallback_n(n_expr, expected)

    def _numeric_fallback_n(self, expr: sp.Expr, expected: sp.Expr) -> Tuple[bool, str]:
        subs_dict = {}
        for sym_name in ['p', 'T', 'm', 'h', 'k', 'epsilon0']:
            sym = sp.Symbol(sym_name)
            val = getattr(self, sym_name, None)
            if val is not None:
                try:
                    subs_dict[sym] = float(sp.N(val))
                except (TypeError, ValueError):
                    continue
        if not subs_dict:
            return False, "SKIP: no numeric givens available"

        needed = [sp.Symbol('p'), sp.Symbol('T'), sp.Symbol('m'), sp.Symbol('h'), sp.Symbol('k'), sp.Symbol('epsilon0')]
        if any(s not in subs_dict for s in needed):
            return False, "SKIP: missing numeric values for required symbols"

        try:
            val_n = float(expr.subs(subs_dict).evalf())
            val_expected = float(expected.subs(subs_dict).evalf())
            if math.isclose(val_n, val_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, "n numerically matches theoretical form (within 1%)"
            else:
                return False, f"n numerical mismatch: got {val_n}, expected {val_expected}"
        except Exception as e:
            return False, f"Numerical fallback crashed: {e}"

    def test_modeling_mu_2D_binding_energy_shift(self) -> Tuple[bool, str]:
        """Modeling check: mu_2D should equal (mu_2D without binding) - epsilon0."""
        mu2 = self._sympy(self.mu_2D)
        if mu2 is None:
            return False, "SKIP: mu_2D not available"

        # Build mu_2D without binding: -kT ln(A/(N lambda^2))
        N, A, T, m, h, k = sp.symbols('N A T m h k', positive=True)
        lam = h / sp.sqrt(2 * sp.pi * m * k * T)
        mu2_no_binding = -k * T * sp.log(A / (N * lam**2))

        # The claim: mu_2D = mu2_no_binding - epsilon0
        eps0 = sp.Symbol('epsilon0')
        expected = mu2_no_binding - eps0

        diff = sp.simplify(mu2 - expected)
        if diff == 0:
            return True, "mu_2D correctly includes binding energy shift"
        else:
            # Numeric fallback
            subs_dict = {}
            for sym_name in ['N', 'A', 'T', 'm', 'h', 'k', 'epsilon0']:
                sym = sp.Symbol(sym_name)
                val = getattr(self, sym_name, None)
                if val is not None:
                    try:
                        subs_dict[sym] = float(sp.N(val))
                    except (TypeError, ValueError):
                        continue
            if not subs_dict:
                return False, "SKIP: no numeric givens for numeric fallback"

            needed = [sp.Symbol('N'), sp.Symbol('A'), sp.Symbol('T'), sp.Symbol('m'), sp.Symbol('h'), sp.Symbol('k'), sp.Symbol('epsilon0')]
            if any(s not in subs_dict for s in needed):
                return False, "SKIP: missing numeric values for required symbols"

            try:
                val_mu2 = float(mu2.subs(subs_dict).evalf())
                val_expected = float(expected.subs(subs_dict).evalf())
                if math.isclose(val_mu2, val_expected, rel_tol=1e-2, abs_tol=1e-2):
                    return True, "mu_2D numerically satisfies binding energy shift (within 1%)"
                else:
                    return False, f"mu_2D binding shift mismatch: got {val_mu2}, expected {val_expected}"
            except Exception as e:
                return False, f"Numeric fallback crashed: {e}"

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_mu3D_with_N(self) -> Tuple[bool, str]:
        """μ_3D should increase with N (since ln(V/(Nλ^3)) decreases when N increases)."""
        expr = self._sympy(self.mu_3D)
        if expr is None:
            return False, "SKIP: mu_3D not available"

        # Numeric perturbation: we need a numeric value for N, V, T, m, h, k.
        subs_base = {}
        for sym_name in ['N', 'V', 'T', 'm', 'h', 'k']:
            sym = sp.Symbol(sym_name)
            val = getattr(self, sym_name, None)
            if val is not None:
                try:
                    subs_base[sym] = float(sp.N(val))
                except (TypeError, ValueError):
                    continue
        if not subs_base:
            return False, "SKIP: no numeric givens for monotonicity test"

        if sp.Symbol('N') not in subs_base:
            return False, "SKIP: N not available"

        try:
            base_val = float(expr.subs(subs_base).evalf())
            subs_perturb = subs_base.copy()
            N0 = subs_base[sp.Symbol('N')]
            subs_perturb[sp.Symbol('N')] = N0 * 1.1  # increase by 10%
            perturb_val = float(expr.subs(subs_perturb).evalf())
            if perturb_val > base_val:
                return True, f"μ_3D increased when N increased (10%): {base_val:.4e} -> {perturb_val:.4e}"
            else:
                return False, f"μ_3D did not increase when N increased: {base_val:.4e} -> {perturb_val:.4e}"
        except Exception as e:
            return False, f"Numeric monotonicity crashed: {e}"

    def test_monotonicity_n_with_p(self) -> Tuple[bool, str]:
        """n should increase with p (proportionality factor)."""
        expr = self._sympy(self.n)
        if expr is None:
            return False, "SKIP: n not available"

        subs_base = {}
        for sym_name in ['p', 'T', 'm', 'h', 'k', 'epsilon0']:
            sym = sp.Symbol(sym_name)
            val = getattr(self, sym_name, None)
            if val is not None:
                try:
                    subs_base[sym] = float(sp.N(val))
                except (TypeError, ValueError):
                    continue
        if not subs_base:
            return False, "SKIP: no numeric givens for monotonicity test"

        if sp.Symbol('p') not in subs_base:
            return False, "SKIP: p not available"

        try:
            base_val = float(expr.subs(subs_base).evalf())
            subs_perturb = subs_base.copy()
            p0 = subs_base[sp.Symbol('p')]
            subs_perturb[sp.Symbol('p')] = p0 * 1.1
            perturb_val = float(expr.subs(subs_perturb).evalf())
            if perturb_val > base_val:
                return True, f"n increased when p increased (10%): {base_val:.4e} -> {perturb_val:.4e}"
            else:
                return False, f"n did not increase when p increased: {base_val:.4e} -> {perturb_val:.4e}"
        except Exception as e:
            return False, f"Numeric monotonicity crashed: {e}"

    def test_monotonicity_n_with_epsilon0(self) -> Tuple[bool, str]:
        """n should increase with ε0 (exponential factor)."""
        expr = self._sympy(self.n)
        if expr is None:
            return False, "SKIP: n not available"

        subs_base = {}
        for sym_name in ['p', 'T', 'm', 'h', 'k', 'epsilon0']:
            sym = sp.Symbol(sym_name)
            val = getattr(self, sym_name, None)
            if val is not None:
                try:
                    subs_base[sym] = float(sp.N(val))
                except (TypeError, ValueError):
                    continue
        if not subs_base:
            return False, "SKIP: no numeric givens for monotonicity test"

        if sp.Symbol('epsilon0') not in subs_base:
            return False, "SKIP: epsilon0 not available"

        try:
            base_val = float(expr.subs(subs_base).evalf())
            subs_perturb = subs_base.copy()
            e0 = subs_base[sp.Symbol('epsilon0')]
            subs_perturb[sp.Symbol('epsilon0')] = e0 * 1.1
            perturb_val = float(expr.subs(subs_perturb).evalf())
            if perturb_val > base_val:
                return True, f"n increased when ε0 increased (10%): {base_val:.4e} -> {perturb_val:.4e}"
            else:
                return False, f"n did not increase when ε0 increased: {base_val:.4e} -> {perturb_val:.4e}"
        except Exception as e:
            return False, f"Numeric monotonicity crashed: {e}"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Weak sanity: n should be positive and finite (if numeric). No re-derivation."""
        n_val = self.n
        if n_val is None:
            return False, "SKIP: n is None"
        try:
            num = float(sp.N(n_val))
        except (TypeError, ValueError, sp.SympifyError):
            # If symbolic, cannot evaluate
            return True, "n is symbolic; numeric positivity cannot be verified (skip)"
        if num > 0 and not math.isnan(num) and not math.isinf(num):
            return True, f"n is positive and finite: {num:.4e}"
        else:
            return False, f"n is not positive/finite: {num}"

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
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