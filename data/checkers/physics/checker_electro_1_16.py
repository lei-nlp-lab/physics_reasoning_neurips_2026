import math
import sympy as sp
from typing import Any, Callable, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for the spherical conductor Gauss's law problem.
    Checks interface, invariants, modeling quality, and monotonicity.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens safely (default None)
        givens = getattr(solution_instance, 'givens', None) or {}
        self.Q = givens.get('Q', None)
        self.r1 = givens.get('r1', None)
        self.r2 = givens.get('r2', None)
        self.r3 = givens.get('r3', None)
        self.epsilon0 = givens.get('epsilon0', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', None) or {}
        self.E1_expr = results.get('E1', None)
        self.E2_expr = results.get('E2', None)
        self.V_r1_expr = results.get('V_r1', None)

        # Sympy symbols for expression manipulation
        self.r_sym = sp.Symbol('r', positive=True, real=True)
        self.Q_sym = sp.Symbol('Q', positive=True, real=True)
        self.eps_sym = sp.Symbol('epsilon0', positive=True, real=True)
        self.pi_sym = sp.Symbol('pi', positive=True, real=True)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _to_sympy(self, expr: Any) -> Optional[sp.Expr]:
        """Convert a string or number to a sympy expression."""
        if expr is None:
            return None
        if isinstance(expr, sp.Expr):
            return expr
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception:
                return None
        if isinstance(expr, (int, float)):
            return sp.nsimplify(expr)
        return None

    def _eval_numeric(self, expr: sp.Expr, r_val: float, Q_val: float, eps_val: float) -> Optional[float]:
        """Evaluate a sympy numerical expression at given parameters."""
        subs = {
            self.r_sym: r_val,
            self.Q_sym: Q_val,
            self.eps_sym: eps_val,
            self.pi_sym: math.pi,
        }
        # Also map any other symbols that match known names
        for s in expr.free_symbols:
            name = str(s)
            if name in ('r',):
                subs.setdefault(s, r_val)
            elif name in ('Q',):
                subs.setdefault(s, Q_val)
            elif name in ('epsilon0', 'varepsilon0', 'epsilon_0', 'varepsilon_0'):
                subs.setdefault(s, eps_val)
            elif name in ('pi', 'Pi', 'PI'):
                subs.setdefault(s, math.pi)
        try:
            val = expr.subs(subs).evalf()
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_givens_exist(self) -> Tuple[bool, str]:
        missing = [name for name in ('Q', 'r1', 'r2', 'r3', 'epsilon0')
                   if getattr(self, name, None) is None]
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All givens present."

    def test_interface_results_exist(self) -> Tuple[bool, str]:
        missing = [key for key in ('E1_expr', 'E2_expr', 'V_r1_expr')
                   if getattr(self, key, None) is None]
        if missing:
            return False, f"Missing results: {', '.join(k.replace('_expr', '') for k in missing)}"
        return True, "All result keys present."

    def test_interface_givens_numeric(self) -> Tuple[bool, str]:
        for name in ('Q', 'r1', 'r2', 'r3', 'epsilon0'):
            val = getattr(self, name, None)
            if val is not None:
                try:
                    float(val)
                except (TypeError, ValueError):
                    return False, f"'{name}' is not numeric."
        return True, "All givens are numeric convertible."

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_gauss_outside(self) -> Tuple[bool, str]:
        """E1*4πr² = Q/ε0 for r > r3."""
        if any(x is None for x in (self.E1_expr, self.Q, self.epsilon0, self.r3)):
            return False, "SKIP: missing data."
        sym = self._to_sympy(self.E1_expr)
        if sym is None:
            return False, "E1 not sympy-convertible."
        try:
            Q_val, eps_val, r3_val = float(self.Q), float(self.epsilon0), float(self.r3)
        except Exception as e:
            return False, f"Conversion error: {e}"
        r_test = r3_val * 2.0
        expected = Q_val / (4.0 * math.pi * eps_val * r_test**2)
        val = self._eval_numeric(sym, r_test, Q_val, eps_val)
        if val is None:
            return False, "Numerical evaluation failed."
        if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Gauss holds: E={val:.6e}, expected={expected:.6e}."
        else:
            return False, f"Mismatch: E={val:.6e}, expected={expected:.6e}."

    def test_invariant_gauss_between(self) -> Tuple[bool, str]:
        """E2*4πr² = Q/ε0 for r1 < r < r2."""
        if any(x is None for x in (self.E2_expr, self.Q, self.epsilon0, self.r1, self.r2)):
            return False, "SKIP: missing data."
        sym = self._to_sympy(self.E2_expr)
        if sym is None:
            return False, "E2 not sympy-convertible."
        try:
            Q_val, eps_val, r1_val, r2_val = float(self.Q), float(self.epsilon0), float(self.r1), float(self.r2)
        except Exception as e:
            return False, f"Conversion error: {e}"
        r_test = (r1_val + r2_val) / 2.0
        expected = Q_val / (4.0 * math.pi * eps_val * r_test**2)
        val = self._eval_numeric(sym, r_test, Q_val, eps_val)
        if val is None:
            return False, "Numerical evaluation failed."
        if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Gauss holds: E={val:.6e}, expected={expected:.6e}."
        else:
            return False, f"Mismatch: E={val:.6e}, expected={expected:.6e}."

    def test_invariant_potential_integral_structure(self) -> Tuple[bool, str]:
        """V_r1 must be sum of two integrals (or at least contain two integrals)."""
        if self.V_r1_expr is None:
            return False, "SKIP: V_r1 missing."
        sym = self._to_sympy(self.V_r1_expr)
        if sym is None:
            return False, "V_r1 not sympy-convertible."
        from sympy import Integral, Add
        if isinstance(sym, Add):
            integrals = [arg for arg in sym.args if isinstance(arg, Integral)]
            if len(integrals) >= 2:
                return True, f"Expression contains {len(integrals)} integrals."
        elif isinstance(sym, Integral):
            return True, "Expression is a single integral (expecting sum of two?)."
        # Fallback string check
        if 'integral' in str(sym).lower() or 'Integral' in str(sym):
            return True, "Integral pattern found (lenient check)."
        return False, "No clear integral structure in V_r1."

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_fields_have_r(self) -> Tuple[bool, str]:
        """Both E1 and E2 expressions must depend on radial coordinate r."""
        for name, expr in [('E1', self.E1_expr), ('E2', self.E2_expr)]:
            if expr is None:
                continue
            sym = self._to_sympy(expr)
            if sym is None:
                # Check string
                if 'r' not in str(expr):
                    return False, f"{name} does not contain 'r'."
            else:
                if 'r' not in [str(s) for s in sym.free_symbols]:
                    return False, f"{name} does not depend on r."
        return True, "Both fields depend on r."

    def test_modeling_E1_E2_same_form(self) -> Tuple[bool, str]:
        """For same r, E1 and E2 should be equal (both Q/(4π ε0 r²))."""
        if any(x is None for x in (self.E1_expr, self.E2_expr, self.Q, self.epsilon0)):
            return False, "SKIP: missing data."
        sym1 = self._to_sympy(self.E1_expr)
        sym2 = self._to_sympy(self.E2_expr)
        if sym1 is None or sym2 is None:
            return False, "Cannot convert expressions."
        try:
            Q_val, eps_val = float(self.Q), float(self.epsilon0)
        except Exception as e:
            return False, f"Conversion error: {e}"
        r_test = 2.0  # arbitrary, but must be in both domains? We ignore domain for form check.
        v1 = self._eval_numeric(sym1, r_test, Q_val, eps_val)
        v2 = self._eval_numeric(sym2, r_test, Q_val, eps_val)
        if v1 is None or v2 is None:
            return False, "Numeric evaluation failed."
        if math.isclose(v1, v2, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E1={v1:.6e}, E2={v2:.6e} at r={r_test}."
        else:
            return False, f"Different: E1={v1:.6e}, E2={v2:.6e}."

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare baseline vs perturbed output)
    # ------------------------------------------------------------------
    def test_monotonicity_E_outside_decreases_with_r(self) -> Tuple[bool, str]:
        """Increase r by factor 1.5, E should decrease."""
        # This test explicitly chooses a baseline r (just outside r3) and a perturbed r (larger).
        if any(x is None for x in (self.E1_expr, self.Q, self.epsilon0, self.r3)):
            return False, "SKIP: missing data."
        sym = self._to_sympy(self.E1_expr)
        if sym is None:
            return False, "E1 not convertible."
        try:
            Q_val, eps_val, r3_val = float(self.Q), float(self.epsilon0), float(self.r3)
        except Exception as e:
            return False, f"Conversion error: {e}"

        # Baseline r
        baseline_r = r3_val * 1.2
        baseline_E = self._eval_numeric(sym, baseline_r, Q_val, eps_val)

        # Perturbed r (increase by 50%)
        perturbed_r = baseline_r * 1.5
        perturbed_E = self._eval_numeric(sym, perturbed_r, Q_val, eps_val)

        if baseline_E is None or perturbed_E is None:
            return False, "Numerical evaluation failed."

        if perturbed_E < baseline_E:
            return (True,
                    f"Baseline E={baseline_E:.6e} at r={baseline_r:.4e}; "
                    f"perturbed E={perturbed_E:.6e} at r={perturbed_r:.4e} (decreased).")
        else:
            return (False,
                    f"E did not decrease: baseline={baseline_E:.6e}, perturbed={perturbed_E:.6e}.")

    def test_monotonicity_E_between_decreases_with_r(self) -> Tuple[bool, str]:
        """Increase r by a factor (0.3*(r2-r1) to 0.6*(r2-r1)), E should decrease."""
        if any(x is None for x in (self.E2_expr, self.Q, self.epsilon0, self.r1, self.r2)):
            return False, "SKIP: missing data."
        sym = self._to_sympy(self.E2_expr)
        if sym is None:
            return False, "E2 not convertible."
        try:
            Q_val, eps_val, r1_val, r2_val = float(self.Q), float(self.epsilon0), float(self.r1), float(self.r2)
        except Exception as e:
            return False, f"Conversion error: {e}"

        # Baseline: 30% from r1
        baseline_r = r1_val + 0.3 * (r2_val - r1_val)
        baseline_E = self._eval_numeric(sym, baseline_r, Q_val, eps_val)

        # Perturbed: 60% from r1 (larger r)
        perturbed_r = r1_val + 0.6 * (r2_val - r1_val)
        perturbed_E = self._eval_numeric(sym, perturbed_r, Q_val, eps_val)

        if baseline_E is None or perturbed_E is None:
            return False, "Numerical evaluation failed."

        if perturbed_E < baseline_E:
            return (True,
                    f"Baseline E={baseline_E:.6e} at r={baseline_r:.4e}; "
                    f"perturbed E={perturbed_E:.6e} at r={perturbed_r:.4e} (decreased).")
        else:
            return (False,
                    f"E did not decrease: baseline={baseline_E:.6e}, perturbed={perturbed_E:.6e}.")

    def test_monotonicity_V_increases_with_Q(self) -> Tuple[bool, str]:
        """Increase Q by 10% and verify V_r1 increases."""
        # This test explicitly perturbs Q (baseline vs perturbed) and compares V.
        if any(x is None for x in (self.V_r1_expr, self.Q, self.epsilon0, self.r1, self.r2, self.r3)):
            return False, "SKIP: missing data."
        sym = self._to_sympy(self.V_r1_expr)
        if sym is None:
            return False, "V_r1 not convertible."

        try:
            Q_base = float(self.Q)
            eps_val = float(self.epsilon0)
            r1_val = float(self.r1)
            r2_val = float(self.r2)
            r3_val = float(self.r3)
        except Exception as e:
            return False, f"Conversion error: {e}"

        # Helper to evaluate V_r1 numerically (handles symbolic integrals)
        def eval_V(Q):
            subs = {
                self.Q_sym: Q,
                self.eps_sym: eps_val,
                self.pi_sym: math.pi,
            }
            # Map free symbols (excluding integration variable)
            for s in sym.free_symbols:
                name = str(s)
                if name == 'r':
                    continue  # integration variable
                if name == 'Q':
                    subs[s] = Q
                elif name in ('epsilon0', 'varepsilon0', 'epsilon_0', 'varepsilon_0'):
                    subs[s] = eps_val
                elif name in ('pi', 'Pi', 'PI'):
                    subs[s] = math.pi
                elif name in ('r1',):
                    subs[s] = r1_val
                elif name in ('r2',):
                    subs[s] = r2_val
                elif name in ('r3',):
                    subs[s] = r3_val
            try:
                # Try direct evaluation (if expression is already integrated)
                val = sym.subs(subs).evalf()
                return float(val)
            except Exception:
                # Fallback: evaluate each integral numerically
                from sympy import Integral, Add, oo
                def eval_integral(integral):
                    if not isinstance(integral, Integral):
                        return None
                    integrand, (var, a, b) = integral.args[0], integral.args[1]
                    # Substitute constants
                    f = integrand.subs(subs)
                    a_num = a.subs(subs).evalf()
                    b_num = b.subs(subs).evalf()
                    try:
                        res = sp.integrate(f, (var, a_num, b_num))
                        return float(res.evalf())
                    except Exception:
                        return None

                integrals = [arg for arg in sym.args if isinstance(arg, Integral)] if isinstance(sym, Add) else [sym] if isinstance(sym, Integral) else []
                total = 0.0
                for i in integrals:
                    val = eval_integral(i)
                    if val is None:
                        return None
                    total += val
                return total

        V_base = eval_V(Q_base)
        if V_base is None:
            return False, "Could not evaluate V at baseline Q."

        Q_perturbed = Q_base * 1.1   # 10% increase
        V_pert = eval_V(Q_perturbed)
        if V_pert is None:
            return False, "Could not evaluate V at perturbed Q."

        if V_pert > V_base:
            return (True,
                    f"Baseline V={V_base:.6e} at Q={Q_base:.4e}; "
                    f"perturbed V={V_pert:.6e} at Q={Q_perturbed:.4e} (increased).")
        else:
            return (False,
                    f"V did not increase: baseline={V_base:.6e}, perturbed={V_pert:.6e}.")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one method)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign(self) -> Tuple[bool, str]:
        """For positive Q, all fields and potential should be positive (or at least non-negative)."""
        if self.Q is None:
            return False, "SKIP: Q missing."
        try:
            Q_val = float(self.Q)
        except:
            return False, "Q not numeric."
        if Q_val <= 0:
            return False, "SKIP: Q not positive."
        # Check fields at a sample point
        for name, expr in [('E1', self.E1_expr), ('E2', self.E2_expr)]:
            if expr is None:
                continue
            sym = self._to_sympy(expr)
            if sym is None:
                continue
            try:
                eps_val = float(self.epsilon0) if self.epsilon0 is not None else 8.854e-12
                val = self._eval_numeric(sym, r_val=1.0, Q_val=Q_val, eps_val=eps_val)
                if val is not None and val < 0:
                    return False, f"{name} negative ({val})."
            except:
                pass
        # Potential sign: check that V_r1 is positive when evaluated
        if self.V_r1_expr is not None:
            symV = self._to_sympy(self.V_r1_expr)
            if symV is not None:
                try:
                    eps_val = float(self.epsilon0) if self.epsilon0 is not None else 8.854e-12
                    r1_val = float(self.r1) if self.r1 is not None else 0.1
                    r2_val = float(self.r2) if self.r2 is not None else 0.5
                    r3_val = float(self.r3) if self.r3 is not None else 1.0
                    from sympy import Integral, Add
                    def eval_V_simple(Q):
                        subs = {
                            self.Q_sym: Q,
                            self.eps_sym: eps_val,
                            self.pi_sym: math.pi,
                        }
                        for s in symV.free_symbols:
                            name = str(s)
                            if name == 'r':
                                continue
                            if name == 'Q':
                                subs[s] = Q
                            elif name in ('epsilon0', 'varepsilon0', 'epsilon_0', 'varepsilon_0'):
                                subs[s] = eps_val
                            elif name in ('pi', 'Pi', 'PI'):
                                subs[s] = math.pi
                            elif name in ('r1',):
                                subs[s] = r1_val
                            elif name in ('r2',):
                                subs[s] = r2_val
                            elif name in ('r3',):
                                subs[s] = r3_val
                        try:
                            val = symV.subs(subs).evalf()
                            return float(val)
                        except Exception:
                            integrals = [arg for arg in symV.args if isinstance(arg, Integral)] if isinstance(symV, Add) else [symV] if isinstance(symV, Integral) else []
                            total = 0.0
                            for i in integrals:
                                integrand, (var, a, b) = i.args[0], i.args[1]
                                f = integrand.subs(subs)
                                a_num = a.subs(subs).evalf()
                                b_num = b.subs(subs).evalf()
                                try:
                                    total += float(sp.integrate(f, (var, a_num, b_num)).evalf())
                                except:
                                    return None
                            return total
                    val = eval_V_simple(Q_val)
                    if val is not None and val < 0:
                        return False, f"V_r1 negative ({val})."
                except:
                    pass
        return True, "Sign check passed (no negative values detected)."

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
                passed, msg = False, f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs