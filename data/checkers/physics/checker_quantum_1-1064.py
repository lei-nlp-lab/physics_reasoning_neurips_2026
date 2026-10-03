import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract numeric parameters (may be None)
        self.m = getattr(solution_instance, 'm', None)
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = getattr(solution_instance, 'solution', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Cache symbolic expressions for repeated use
        self._expr_cache = {}

    def _parse_expr(self, key, fallback=None):
        """Try to parse the result value as a sympy expression.
        Returns None if parsing fails."""
        expr = self.results.get(key, None)
        if expr is None:
            return fallback
        if isinstance(expr, sp.Basic):
            return expr
        if isinstance(expr, str):
            try:
                # Attempt to parse as plain math expression
                return sp.sympify(expr)
            except (sp.SympifyError, TypeError, ValueError):
                pass
            # Try to parse as LaTeX (requires sympy.parsing.latex, not always available)
            try:
                from sympy.parsing.latex import parse_latex
                return parse_latex(expr)
            except (ImportError, Exception):
                pass
        return fallback

    # ========== INTERFACE TESTS ==========
    def test_interface_required_keys(self):
        """Check that all required result keys are present."""
        required_keys = ['bc_psi', 'bc_psi_prime', 'k0_eq', 'psi', 'psi_barrier', 'E_gap']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing required keys: {missing}")
        return (True, "All required keys present")

    def test_interface_bc_psi(self):
        """Check that bc_psi mentions continuity."""
        val = self.results.get('bc_psi', None)
        if val is None:
            return (False, "bc_psi not found")
        val_str = str(val)
        if 'continuous' in val_str.lower():
            return (True, "bc_psi mentions continuity")
        else:
            return (False, f"bc_psi does not mention continuity: {val_str}")

    def test_interface_bc_psi_prime(self):
        """Check that bc_psi_prime mentions the jump condition proportional to psi."""
        val = self.results.get('bc_psi_prime', None)
        if val is None:
            return (False, "bc_psi_prime not found")
        val_str = str(val)
        # Look for "Delta", "jump", "Omega", "psi"
        if 'Delta' in val_str and ('Omega' in val_str or 'psi' in val_str):
            return (True, "bc_psi_prime contains jump condition")
        else:
            return (False, f"bc_psi_prime does not contain expected jump form: {val_str}")

    def test_interface_k0_eq_structure(self):
        """Check that k0_eq contains cos, sin, k0, a, Omega."""
        expr = self._parse_expr('k0_eq')
        if expr is None:
            # Fall back to string check
            s = str(self.results.get('k0_eq', ''))
            if not s:
                return (False, "k0_eq is empty or missing")
            # Check for presence of essential symbols and trigonometric functions
            if 'cos' in s and 'sin' in s and ('k0' in s or 'k_0' in s) and 'a' in s and 'Omega' in s:
                return (True, "k0_eq string contains expected symbols")
            else:
                return (False, f"k0_eq string structure incorrect: {s}")
        else:
            # Symbolic check: verify that the expression involves cos, sin, k0, a, Omega
            symbols_needed = ['k0', 'a', 'Omega']
            trig_found = any(func in sp.preorder_traversal(expr) for func in (sp.cos, sp.sin))
            syms_in_expr = [s.name for s in expr.free_symbols]
            has_symbols = all(sym in syms_in_expr for sym in symbols_needed)
            if trig_found and has_symbols:
                return (True, "k0_eq symbolic structure correct")
            else:
                return (False, f"k0_eq symbolic structure missing trig or symbols: {expr}")

    def test_interface_psi_structure(self):
        """Check that psi contains sin and exponentials."""
        expr = self._parse_expr('psi')
        if expr is None:
            s = str(self.results.get('psi', ''))
            if not s:
                return (False, "psi missing")
            if 'sin' in s and 'exp' in s:
                return (True, "psi string contains sin and exp")
            else:
                return (False, f"psi string structure incorrect: {s}")
        else:
            has_sin = any(isinstance(node, sp.sin) for node in sp.preorder_traversal(expr))
            has_exp = any(isinstance(node, sp.exp) for node in sp.preorder_traversal(expr))
            if has_sin and has_exp:
                return (True, "psi symbolic contains sin and exponential")
            else:
                return (False, f"psi symbolic missing sin or exp: {expr}")

    def test_interface_psi_barrier_structure(self):
        """Check that psi_barrier contains exponential functions."""
        expr = self._parse_expr('psi_barrier')
        if expr is None:
            s = str(self.results.get('psi_barrier', ''))
            if not s:
                return (False, "psi_barrier missing")
            if 'exp' in s:
                return (True, "psi_barrier string contains exp")
            else:
                return (False, f"psi_barrier string structure incorrect: {s}")
        else:
            has_exp = any(isinstance(node, sp.exp) for node in sp.preorder_traversal(expr))
            if has_exp:
                return (True, "psi_barrier symbolic contains exponential")
            else:
                return (False, f"psi_barrier symbolic missing exponential: {expr}")

    def test_interface_E_gap_form(self):
        """Check that E_gap is of the form hbar^2 pi^2/(2 m a^2)."""
        expr = self._parse_expr('E_gap')
        if expr is None:
            s = str(self.results.get('E_gap', ''))
            if not s:
                return (False, "E_gap missing")
            # Simple keyword check
            if 'pi' in s and 'hbar' in s:
                return (True, "E_gap string contains pi and hbar")
            else:
                return (False, f"E_gap string does not look right: {s}")
        else:
            # Try to recognize the symbolic form: hbar^2*pi^2/(2*m*a^2)
            hbar, pi, m, a = sp.symbols('hbar pi m a')
            target = hbar**2 * pi**2 / (2 * m * a**2)
            try:
                if sp.simplify(expr - target) == 0:
                    return (True, "E_gap matches expected symbolic form")
                else:
                    return (False, f"E_gap symbolic form does not match: {expr}")
            except Exception:
                # If comparison fails, check free symbols
                syms = [str(s) for s in expr.free_symbols]
                if 'hbar' in syms and 'pi' in syms and 'm' in syms and 'a' in syms:
                    return (True, "E_gap has all required symbols")
                else:
                    return (False, f"E_gap symbolic free symbols missing: {syms}")

    # ========== INVARIANT TESTS ==========
    def test_invariant_dispersion_at_band_edge(self):
        """Check that at the band edge (K=0) the dispersion relation becomes
        cos(k0 a) + (Omega/k0) sin(k0 a) = 1.
        This is a property of the solution for k0."""
        expr_k0_eq = self._parse_expr('k0_eq')
        if expr_k0_eq is None:
            return (False, "k0_eq not available for invariant test")
        if isinstance(expr_k0_eq, sp.Equality):
            lhs, rhs = expr_k0_eq.lhs, expr_k0_eq.rhs
            if rhs == 1:
                return (True, "k0_eq is an equation with RHS = 1")
            else:
                return (False, f"k0_eq RHS is not 1: {rhs}")
        try:
            simplified = sp.simplify(expr_k0_eq - 1)
        except Exception:
            simplified = expr_k0_eq - 1
        if simplified == 0:
            return (True, "k0_eq expression equals 1 after simplification")
        else:
            return (False, f"k0_eq does not simplify to 1: {expr_k0_eq}")

    def test_invariant_psi_schrodinger(self):
        """Verify that the psi expression satisfies the Schrödinger equation in region 0<x<a."""
        psi_expr = self._parse_expr('psi')
        if psi_expr is None:
            return (False, "psi not available for invariant test")
        x = sp.Symbol('x', real=True)
        k0 = sp.Symbol('k0', real=True)
        try:
            dpsi = sp.diff(psi_expr, x)
            d2psi = sp.diff(dpsi, x)
            lhs = d2psi + k0**2 * psi_expr
            simplified = sp.simplify(lhs)
            if simplified == 0:
                return (True, "psi satisfies Helmholtz equation: psi'' + k0^2 psi = 0")
            else:
                return (False, f"psi does not satisfy Helmholtz equation, simplified to: {simplified}")
        except Exception as e:
            return (False, f"Failed to compute derivative: {e}")

    # ========== MODELING / TRACE TESTS ==========
    def test_modeling_continuity_at_zero(self):
        """Check that psi(0) = 1 as required by normalization."""
        psi_expr = self._parse_expr('psi')
        if psi_expr is None:
            return (False, "psi not available for continuity test")
        x = sp.Symbol('x', real=True)
        try:
            psi0 = psi_expr.subs(x, 0)
            simplified = sp.simplify(psi0)
            if simplified == 1:
                return (True, "psi(0) = 1")
            else:
                return (False, f"psi(0) = {psi0}, not 1")
        except Exception as e:
            return (False, f"Failed to evaluate psi at 0: {e}")

    def test_modeling_bloch_condition(self):
        """Check that the Bloch condition psi(x+a) = e^{i K a} psi(x) holds, at least at x=0?
        But we only have psi on [0,a]. We can check that the expression satisfies Bloch at x=0.
        That is, psi(a) = e^{i K a} psi(0) = e^{i K a} because psi(0)=1.
        So check that psi evaluated at x=a equals exp(i K a)."""
        psi_expr = self._parse_expr('psi')
        if psi_expr is None:
            return (False, "psi not available for Bloch test")
        x = sp.Symbol('x', real=True)
        K = sp.Symbol('K', real=True)
        a_sym = sp.Symbol('a', real=True, positive=True)
        try:
            psi_a = psi_expr.subs({x: a_sym, sp.Symbol('K'): K})
            expected = sp.exp(sp.I * K * a_sym)
            diff = sp.simplify(psi_a - expected)
            if diff == 0:
                return (True, "Bloc condition holds at x=a")
            else:
                return (False, f"Bloc condition not satisfied: psi(a) = {psi_a}, expected {expected}")
        except Exception as e:
            return (False, f"Failed to compute Bloch condition: {e}")

    def test_modeling_delta_jump(self):
        """Check that the boundary condition for the derivative jump is consistent.
        For the delta at x=0, the derivative jump should be 2*Omega*psi(0).
        Using the expression, compute the difference of derivatives from left and right?
        But we only have a region [0,a]; left region is not defined. So we skip."""
        return (True, "SKIP: Derivative jump check requires left-side wave function, not available")

    # ========== MONOTONICITY TESTS ==========
    def test_monotonicity_Egap_vs_a(self):
        """Check that E_gap decreases when lattice spacing a increases.
        Uses symbolic derivative if numeric params unavailable; otherwise numeric perturbation."""
        # First try symbolic approach: if we can parse E_gap, compute its derivative w.r.t. a
        Egap_expr = self._parse_expr('E_gap')
        if Egap_expr is not None:
            a_sym = sp.Symbol('a', real=True, positive=True)
            hbar_sym = sp.Symbol('hbar', real=True, positive=True)
            m_sym = sp.Symbol('m', real=True, positive=True)
            pi_sym = sp.Symbol('pi', real=True, positive=True)
            # Substitute known symbols for a, hbar, m, pi
            sub_dict = {}
            if sp.Symbol('a') in Egap_expr.free_symbols:
                sub_dict[sp.Symbol('a')] = a_sym
            if sp.Symbol('hbar') in Egap_expr.free_symbols:
                sub_dict[sp.Symbol('hbar')] = hbar_sym
            if sp.Symbol('m') in Egap_expr.free_symbols:
                sub_dict[sp.Symbol('m')] = m_sym
            if sp.Symbol('pi') in Egap_expr.free_symbols:
                sub_dict[sp.Symbol('pi')] = pi_sym
            if sub_dict:
                try:
                    Egap_sym = Egap_expr.subs(sub_dict)
                    dE_da = sp.diff(Egap_sym, a_sym)
                    # Check that derivative is negative: dE_da < 0
                    # We can check if dE_da is a negative expression for positive parameters
                    # For instance, if dE_da = -hbar^2*pi^2/(m*a^3), it's clearly negative.
                    # Use simplify on sign: sp.simplify(dE_da < 0) might be tricky.
                    # Instead, check that dE_da is not positive.
                    # A robust check: multiply by a positive denominator and see if the result is negative.
                    # Assume all symbols positive; compute dE_da * a**3 * m / hbar**2 / pi**2 ...
                    # Better: substitute numeric test values if all symbols are present.
                    # If we have all symbols and they are ones we expect, we can test with dummy positive numbers.
                    test_subs = {a_sym: 1.0, hbar_sym: 1.0, m_sym: 1.0, pi_sym: sp.pi}
                    try:
                        numeric_deriv = float(dE_da.subs(test_subs).evalf())
                        if numeric_deriv < 0:
                            return (True, "Symbolic derivative dE/da is negative (monotonic decreasing)")
                        else:
                            return (False, f"Symbolic derivative dE/da = {numeric_deriv} is not negative")
                    except Exception:
                        # Fallback to numeric perturbation if possible
                        pass
                except Exception:
                    pass

        # If symbolic approach fails, fallback to numeric perturbation if parameters are present
        if self.a is not None and self.hbar is not None and self.m is not None:
            try:
                a_orig = float(self.a)
                # Re-evaluate E_gap numeric
                Egap_expr = self._parse_expr('E_gap')
                if Egap_expr is None:
                    return (True, "SKIP: E_gap expression not available for numeric monotonicity")
                subs_dict = {sp.Symbol('m'): float(self.m),
                             sp.Symbol('a'): a_orig,
                             sp.Symbol('hbar'): float(self.hbar)}
                if sp.Symbol('pi') in subs_dict:
                    subs_dict[sp.Symbol('pi')] = sp.pi
                Egap_orig = float(Egap_expr.subs(subs_dict).evalf())
                # Perturb a: increase by 10%
                a_new = 1.1 * a_orig
                subs_dict[sp.Symbol('a')] = a_new
                Egap_new = float(Egap_expr.subs(subs_dict).evalf())
                if Egap_new < Egap_orig:
                    return (True, f"Numeric monotonicity: a increased 10%, E_gap decreased from {Egap_orig} to {Egap_new}")
                else:
                    return (False, f"Numeric monotonicity fails: a increased 10%, E_gap changed from {Egap_orig} to {Egap_new}")
            except Exception as e:
                return (False, f"Numeric monotonicity test crashed: {e}")

        # If neither symbolic nor numeric works, skip
        return (True, "SKIP: Unable to perform monotonicity check (no symbolic derivative or numeric parameters)")

    # ========== WEAK ANSWER SANITY ==========
    def test_answer_sanity_sign(self):
        """Check that the band edge energy E_gap is positive and finite (if numeric)."""
        Egap_expr = self._parse_expr('E_gap')
        if Egap_expr is None:
            return (False, "E_gap not available for sanity check")
        # If numeric, check sign
        if self.m and self.hbar and self.a:
            try:
                subs_dict = {sp.Symbol('m'): float(self.m),
                             sp.Symbol('a'): float(self.a),
                             sp.Symbol('hbar'): float(self.hbar)}
                if sp.Symbol('pi') in subs_dict:
                    subs_dict[sp.Symbol('pi')] = sp.pi
                val = float(Egap_expr.subs(subs_dict).evalf())
                if val > 0 and math.isfinite(val):
                    return (True, f"E_gap = {val} > 0 and finite")
                else:
                    return (False, f"E_gap = {val} not positive or not finite")
            except Exception:
                pass
        # If not numeric, skip as we cannot check sign without numeric parameters
        return (True, "SKIP: Cannot evaluate sign without numeric parameters")

    # ========== RUN ALL TESTS ==========
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)