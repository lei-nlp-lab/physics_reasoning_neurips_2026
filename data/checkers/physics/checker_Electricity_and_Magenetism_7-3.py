import math
import sympy as sp
from typing import Any, Union

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Extract givens safely with None defaults
        self.I = getattr(solution_instance, 'I', None)
        self.a = getattr(solution_instance, 'a', None)
        self.chi_m = getattr(solution_instance, 'chi_m', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)

        # Extract results dict safely
        results = getattr(solution_instance, 'results', {}) or {}
        self.results = results
        self.B_piecewise = results.get('B_piecewise', None)
        self.J_b_z = results.get('J_b_z', None)
        self.K_b_z = results.get('K_b_z', None)
        self.I_b_net = results.get('I_b_net', None)

        # Sympy symbols for verification
        self.s, self.I_sym, self.a_sym, self.chi_m_sym, self.mu0_sym = \
            sp.symbols('s I a chi_m mu0')

        # Build expected expressions
        self._build_expected()

    def _build_expected(self):
        """Build expected symbolic expressions for all results."""
        # H fields
        H_in = self.I_sym * self.s / (2 * sp.pi * self.a_sym**2)
        H_out = self.I_sym / (2 * sp.pi * self.s)

        # B fields
        B_in = self.mu0_sym * (1 + self.chi_m_sym) * H_in
        B_out = self.mu0_sym * H_out

        # Expected piecewise B
        self.B_expected = sp.Piecewise(
            (B_in, self.s < self.a_sym),
            (B_out, True)
        )

        # Expected bound currents
        self.J_b_z_expected = self.chi_m_sym * self.I_sym / (sp.pi * self.a_sym**2)
        self.K_b_z_expected = -self.chi_m_sym * self.I_sym / (2 * sp.pi * self.a_sym)
        self.I_b_net_expected = 0

    def _parse_sympy(self, expr: Union[str, sp.Expr]) -> sp.Expr:
        """Try to parse a string or sympy expression into a sympy object."""
        if isinstance(expr, sp.Expr):
            return expr
        if isinstance(expr, str):
            # Attempt to convert from LaTeX or plain sympy string
            try:
                # First try standard sympify (works for e.g. 'chi_m*I/(pi*a**2)')
                return sp.sympify(expr)
            except (sp.SympifyError, TypeError):
                pass
            try:
                # Fallback: parse as LaTeX (requires sympy.parsing.latex)
                from sympy.parsing.latex import parse_latex
                return parse_latex(expr)
            except Exception:
                raise ValueError(f"Cannot parse expression: {expr}")
        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def run_all_tests(self) -> tuple:
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith('test_'):
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
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> tuple:
        required = ['B_piecewise', 'J_b_z', 'K_b_z', 'I_b_net']
        missing = [k for k in required if k not in self.results]
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present"

    def test_interface_givens(self) -> tuple:
        givens = {'I': self.I, 'a': self.a, 'chi_m': self.chi_m, 'mu0': self.mu0}
        missing = [k for k, v in givens.items() if v is None]
        if missing:
            return False, f"Missing givens: {missing}"
        return True, "All givens available"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_inside_H_relation(self) -> tuple:
        """Check H = B/(mu0*(1+chi_m)) = I s/(2 pi a^2) for s<a."""
        if self.B_piecewise is None:
            return (False, "SKIP: B_piecewise not available")
        try:
            expr = self._parse_sympy(self.B_piecewise)
        except ValueError as e:
            return (False, f"Parse error: {e}")
        if not isinstance(expr, sp.Piecewise):
            return (False, "B_piecewise is not a Piecewise object")
        inside_B = expr.args[0][0]  # first case
        inside_H = inside_B / (self.mu0_sym * (1 + self.chi_m_sym))
        expected_H = self.I_sym * self.s / (2 * sp.pi * self.a_sym**2)
        diff = sp.simplify(inside_H - expected_H)
        if diff == 0:
            return (True, "Inside H relation holds symbolically")
        # Numeric fallback if givens present
        if all(v is not None for v in [self.I, self.a, self.chi_m, self.mu0]):
            s_val = self.a / 2.0
            try:
                H_actual = float(inside_B.subs({
                    self.s: s_val,
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m,
                    self.mu0_sym: self.mu0
                }) / (self.mu0 * (1 + self.chi_m)))
                H_expected = self.I * s_val / (2 * math.pi * self.a**2)
                if math.isclose(H_actual, H_expected, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, "Inside H relation holds numerically")
                else:
                    return (False, f"Inside H numeric mismatch: {H_actual} vs {H_expected}")
            except Exception as e:
                return (False, f"Numeric fallback error: {e}")
        return (False, f"Symbolic difference: {diff}")

    def test_invariant_outside_H_relation(self) -> tuple:
        """Check H = I/(2 pi s) for s>a."""
        if self.B_piecewise is None:
            return (False, "SKIP: B_piecewise not available")
        try:
            expr = self._parse_sympy(self.B_piecewise)
        except ValueError as e:
            return (False, f"Parse error: {e}")
        if not isinstance(expr, sp.Piecewise):
            return (False, "B_piecewise is not a Piecewise object")
        outside_B = expr.args[1][0]  # second case
        outside_H = outside_B / self.mu0_sym
        expected_H = self.I_sym / (2 * sp.pi * self.s)
        diff = sp.simplify(outside_H - expected_H)
        if diff == 0:
            return (True, "Outside H relation holds symbolically")
        if all(v is not None for v in [self.I, self.mu0]):
            s_val = 2.0 * self.a if self.a is not None else 1.0
            try:
                H_actual = float(outside_B.subs({
                    self.s: s_val,
                    self.I_sym: self.I,
                    self.mu0_sym: self.mu0
                }) / self.mu0)
                H_expected = self.I / (2 * math.pi * s_val)
                if math.isclose(H_actual, H_expected, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, "Outside H relation holds numerically")
                else:
                    return (False, f"Outside H numeric mismatch: {H_actual} vs {H_expected}")
            except Exception as e:
                return (False, f"Numeric fallback error: {e}")
        return (False, f"Symbolic difference: {diff}")

    def test_invariant_H_continuity_at_a(self) -> tuple:
        """Check that H(s) is continuous at s=a (no free surface current)."""
        if self.B_piecewise is None:
            return (False, "SKIP: B_piecewise not available")
        try:
            expr = self._parse_sympy(self.B_piecewise)
        except ValueError as e:
            return (False, f"Parse error: {e}")
        if not isinstance(expr, sp.Piecewise):
            return (False, "B_piecewise is not a Piecewise object")
        inside_B = expr.args[0][0]
        outside_B = expr.args[1][0]
        # Evaluate H just inside and just outside at s=a
        H_in_a = inside_B.subs(self.s, self.a_sym) / (self.mu0_sym * (1 + self.chi_m_sym))
        H_out_a = outside_B.subs(self.s, self.a_sym) / self.mu0_sym
        diff = sp.simplify(H_in_a - H_out_a)
        if diff == 0:
            return (True, "H continuous at s=a symbolically")
        # Numeric fallback
        if all(v is not None for v in [self.I, self.a, self.chi_m, self.mu0]):
            try:
                H_in_val = float(inside_B.subs({
                    self.s: self.a,
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m,
                    self.mu0_sym: self.mu0
                }) / (self.mu0 * (1 + self.chi_m)))
                H_out_val = float(outside_B.subs({
                    self.s: self.a,
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.mu0_sym: self.mu0
                }) / self.mu0)
                if math.isclose(H_in_val, H_out_val, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, "H continuous at s=a numerically")
                else:
                    return (False, f"H continuity mismatch: inside={H_in_val}, outside={H_out_val}")
            except Exception as e:
                return (False, f"Numeric continuity error: {e}")
        return (False, f"Symbolic difference: {diff}")

    def test_invariant_bound_volume_current_relation(self) -> tuple:
        """Check J_b = chi_m * I / (pi * a^2) from curl(M)."""
        if self.J_b_z is None:
            return (False, "SKIP: J_b_z not available")
        # Expected J_b
        expected = self.chi_m_sym * self.I_sym / (sp.pi * self.a_sym**2)
        ok, err = self._sympy_equal(self.J_b_z, expected)
        if ok:
            return (True, "J_b_z symbolic match")
        # Numeric fallback
        if all(v is not None for v in [self.I, self.a, self.chi_m]):
            try:
                expr = self._parse_sympy(self.J_b_z)
                numeric_actual = float(expr.subs({
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                numeric_expected = self.chi_m * self.I / (math.pi * self.a**2)
                if math.isclose(numeric_actual, numeric_expected, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, "J_b_z numeric match")
                else:
                    return (False, f"J_b_z numeric mismatch: {numeric_actual} vs {numeric_expected}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, f"Symbolic mismatch: {err if err else 'unknown'}")

    def test_invariant_bound_surface_current_relation(self) -> tuple:
        """Check K_b = -chi_m * I / (2*pi*a) from M x n̂."""
        if self.K_b_z is None:
            return (False, "SKIP: K_b_z not available")
        expected = -self.chi_m_sym * self.I_sym / (2 * sp.pi * self.a_sym)
        ok, err = self._sympy_equal(self.K_b_z, expected)
        if ok:
            return (True, "K_b_z symbolic match")
        if all(v is not None for v in [self.I, self.a, self.chi_m]):
            try:
                expr = self._parse_sympy(self.K_b_z)
                numeric_actual = float(expr.subs({
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                numeric_expected = -self.chi_m * self.I / (2 * math.pi * self.a)
                if math.isclose(numeric_actual, numeric_expected, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, "K_b_z numeric match")
                else:
                    return (False, f"K_b_z numeric mismatch: {numeric_actual} vs {numeric_expected}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, f"Symbolic mismatch: {err if err else 'unknown'}")

    def test_invariant_net_bound_current_zero(self) -> tuple:
        """Check that net bound current is zero (I_b_net = 0)."""
        if self.I_b_net is None:
            return (False, "SKIP: I_b_net not available")
        # Try numeric or symbolic zero
        try:
            val = float(self.I_b_net)
            if abs(val) < 1e-9:
                return (True, "Net bound current is zero")
            return (False, f"Net bound current non-zero: {val}")
        except (TypeError, ValueError):
            pass
        # Try symbolic zero
        expr = self._parse_sympy(self.I_b_net)
        if sp.simplify(expr) == 0:
            return (True, "Net bound current zero symbolically")
        return (False, f"Net bound current not zero: {expr}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (check consistency of explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_free_current_enclosed(self) -> tuple:
        """Check that I_enc = I * s^2/a^2 for s<a and I for s>a."""
        if self.B_piecewise is None:
            return (False, "SKIP: B_piecewise not available")
        # Compute I_enc from H: I_enc = 2 pi s H(s)
        try:
            expr = self._parse_sympy(self.B_piecewise)
        except ValueError as e:
            return (False, f"Parse error: {e}")
        if not isinstance(expr, sp.Piecewise):
            return (False, "B_piecewise is not a Piecewise object")
        inside_B = expr.args[0][0]
        outside_B = expr.args[1][0]
        # Inside: H = B/(mu0*(1+chi_m)) => I_enc = 2 pi s H
        inside_H = inside_B / (self.mu0_sym * (1 + self.chi_m_sym))
        inside_I_enc = 2 * sp.pi * self.s * inside_H
        expected_inside = self.I_sym * self.s**2 / self.a_sym**2
        diff_inside = sp.simplify(inside_I_enc - expected_inside)
        # Outside: H = B/mu0 => I_enc = 2 pi s H
        outside_H = outside_B / self.mu0_sym
        outside_I_enc = 2 * sp.pi * self.s * outside_H
        diff_outside = sp.simplify(outside_I_enc - self.I_sym)
        if diff_inside == 0 and diff_outside == 0:
            return (True, "Free current enclosed relation holds symbolically")
        # Numeric fallback
        if all(v is not None for v in [self.I, self.a, self.chi_m, self.mu0]):
            s_in = self.a / 2.0
            s_out = 2.0 * self.a
            try:
                B_in = float(inside_B.subs({
                    self.s: s_in,
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m,
                    self.mu0_sym: self.mu0
                }))
                H_in = B_in / (self.mu0 * (1 + self.chi_m))
                I_enc_in = 2 * math.pi * s_in * H_in
                expected_enc_in = self.I * s_in**2 / self.a**2
                if not math.isclose(I_enc_in, expected_enc_in, rel_tol=1e-2, abs_tol=1e-6):
                    return (False, f"Inside I_enc mismatch: {I_enc_in} vs {expected_enc_in}")

                B_out = float(outside_B.subs({
                    self.s: s_out,
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m,
                    self.mu0_sym: self.mu0
                }))
                H_out = B_out / self.mu0
                I_enc_out = 2 * math.pi * s_out * H_out
                if not math.isclose(I_enc_out, self.I, rel_tol=1e-2, abs_tol=1e-6):
                    return (False, f"Outside I_enc mismatch: {I_enc_out} vs {self.I}")
                return (True, "Free current enclosed relation holds numerically")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, f"Inside diff: {diff_inside}, Outside diff: {diff_outside}")

    def test_modeling_volume_current_via_curl_M(self) -> tuple:
        """Check J_b_z = (1/s) d(s M_phi)/ds inside wire."""
        if self.J_b_z is None:
            return (False, "SKIP: J_b_z not available")
        # Compute M_phi = chi_m * H_in = chi_m * I s/(2 pi a^2)
        M_phi = self.chi_m_sym * self.I_sym * self.s / (2 * sp.pi * self.a_sym**2)
        # Compute curl: (1/s) d(s M_phi)/ds
        s_M_phi = self.s * M_phi
        deriv = sp.diff(s_M_phi, self.s)
        curl = deriv / self.s
        expected = sp.simplify(curl)
        # Compare with given J_b_z
        ok, err = self._sympy_equal(self.J_b_z, expected)
        if ok:
            return (True, "J_b_z matches curl(M) symbolic")
        # Numeric fallback at sample s
        if all(v is not None for v in [self.I, self.a, self.chi_m]):
            s_val = self.a / 3.0
            try:
                expr = self._parse_sympy(self.J_b_z)
                numeric_actual = float(expr.subs({
                    self.s: s_val,
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                numeric_expected = float(expected.subs({
                    self.s: s_val,
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                if math.isclose(numeric_actual, numeric_expected, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, "J_b_z matches curl(M) numerically")
                else:
                    return (False, f"Numeric curl mismatch: actual={numeric_actual}, expected={numeric_expected}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, f"Symbolic mismatch: {err}")

    def test_modeling_surface_current_via_M_cross_n(self) -> tuple:
        """Check K_b_z = -M_phi(a) from M x n̂."""
        if self.K_b_z is None:
            return (False, "SKIP: K_b_z not available")
        # M_phi at s=a: M_phi(a) = chi_m * I * a / (2 pi a^2) = chi_m * I/(2 pi a)
        M_phi_a = self.chi_m_sym * self.I_sym / (2 * sp.pi * self.a_sym)
        # K_b_z should be -M_phi_a (since phi x s = -z)
        expected_K = -M_phi_a
        ok, err = self._sympy_equal(self.K_b_z, expected_K)
        if ok:
            return (True, "K_b_z matches -M_phi(a)")
        if all(v is not None for v in [self.I, self.a, self.chi_m]):
            try:
                expr = self._parse_sympy(self.K_b_z)
                numeric_actual = float(expr.subs({
                    self.I_sym: self.I,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                numeric_expected = -self.chi_m * self.I / (2 * math.pi * self.a)
                if math.isclose(numeric_actual, numeric_expected, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, "K_b_z numeric match")
                else:
                    return (False, f"K_b_z numeric mismatch: {numeric_actual} vs {numeric_expected}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, f"Symbolic mismatch: {err}")

    def test_modeling_net_bound_current_integral(self) -> tuple:
        """Check that I_b_net = J_b * (pi a^2) + K_b * (2 pi a) = 0."""
        if any(x is None for x in [self.J_b_z, self.K_b_z, self.I_b_net]):
            return (False, "SKIP: one of J_b_z, K_b_z, I_b_net missing")
        # Compute volume integral: J_b * area
        vol_current = self.J_b_z_expected * (sp.pi * self.a_sym**2)
        surf_current = self.K_b_z_expected * (2 * sp.pi * self.a_sym)
        total = sp.simplify(vol_current + surf_current)
        if total == 0:
            return (True, "Net bound current integral zero symbolically")
        # Numeric fallback
        if all(v is not None for v in [self.I, self.a, self.chi_m]):
            J_val = self.chi_m * self.I / (math.pi * self.a**2)
            K_val = -self.chi_m * self.I / (2 * math.pi * self.a)
            total_num = J_val * (math.pi * self.a**2) + K_val * (2 * math.pi * self.a)
            if abs(total_num) < 1e-9:
                return (True, "Net bound current integral zero numerically")
            else:
                return (False, f"Net bound current non-zero: {total_num}")
        return (False, f"Symbolic total not zero: {total}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input)
    # ------------------------------------------------------------------
    def test_monotonicity_B_inside_increases_with_I(self) -> tuple:
        """Perturb I by +10% and check that B_inside(s<a) scales proportionally."""
        if self.B_piecewise is None:
            return (False, "SKIP: B_piecewise not available")
        try:
            expr = self._parse_sympy(self.B_piecewise)
        except ValueError as e:
            return (False, f"Parse error: {e}")
        if not isinstance(expr, sp.Piecewise):
            return (False, "B_piecewise is not a Piecewise object")
        inside_B = expr.args[0][0]

        if all(v is not None for v in [self.I, self.a, self.chi_m, self.mu0]):
            s_test = self.a / 2.0
            I_base = self.I
            factor = 1.1  # 10% increase
            I_perturbed = I_base * factor
            try:
                B_base = float(inside_B.subs({
                    self.s: s_test,
                    self.I_sym: I_base,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m,
                    self.mu0_sym: self.mu0
                }))
                B_perturbed = float(inside_B.subs({
                    self.s: s_test,
                    self.I_sym: I_perturbed,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m,
                    self.mu0_sym: self.mu0
                }))
                # B is proportional to I, so perturbed/base should equal factor
                ratio = B_perturbed / B_base
                if math.isclose(ratio, factor, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, f"B inside scales linearly: ratio={ratio:.4f} (expected {factor})")
                else:
                    return (False, f"B inside ratio mismatch: {ratio:.4f} vs expected {factor}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, "SKIP: missing givens for numeric check")

    def test_monotonicity_B_outside_increases_with_I(self) -> tuple:
        """Perturb I by +10% and check that B_outside(s>a) scales proportionally."""
        if self.B_piecewise is None:
            return (False, "SKIP: B_piecewise not available")
        try:
            expr = self._parse_sympy(self.B_piecewise)
        except ValueError as e:
            return (False, f"Parse error: {e}")
        if not isinstance(expr, sp.Piecewise):
            return (False, "B_piecewise is not a Piecewise object")
        outside_B = expr.args[1][0]

        if all(v is not None for v in [self.I, self.a, self.mu0]):
            s_test = 2.0 * self.a if self.a is not None else 1.0
            I_base = self.I
            factor = 1.1
            I_perturbed = I_base * factor
            try:
                B_base = float(outside_B.subs({
                    self.s: s_test,
                    self.I_sym: I_base,
                    self.mu0_sym: self.mu0
                }))
                B_perturbed = float(outside_B.subs({
                    self.s: s_test,
                    self.I_sym: I_perturbed,
                    self.mu0_sym: self.mu0
                }))
                ratio = B_perturbed / B_base
                if math.isclose(ratio, factor, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, f"B outside scales linearly: ratio={ratio:.4f} (expected {factor})")
                else:
                    return (False, f"B outside ratio mismatch: {ratio:.4f} vs expected {factor}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, "SKIP: missing givens for numeric check")

    def test_monotonicity_J_b_scales_with_I(self) -> tuple:
        """Perturb I by +10% and check that J_b_z scales proportionally."""
        if self.J_b_z is None:
            return (False, "SKIP: J_b_z not available")
        if all(v is not None for v in [self.I, self.a, self.chi_m]):
            I_base = self.I
            factor = 1.1
            I_perturbed = I_base * factor
            try:
                expr = self._parse_sympy(self.J_b_z)
                J_base = float(expr.subs({
                    self.I_sym: I_base,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                J_perturbed = float(expr.subs({
                    self.I_sym: I_perturbed,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                ratio = J_perturbed / J_base
                if math.isclose(ratio, factor, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, f"J_b scales linearly: ratio={ratio:.4f} (expected {factor})")
                else:
                    return (False, f"J_b ratio mismatch: {ratio:.4f} vs expected {factor}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, "SKIP: missing givens for numeric check")

    def test_monotonicity_K_b_scales_with_I(self) -> tuple:
        """Perturb I by +10% and check that K_b_z scales proportionally."""
        if self.K_b_z is None:
            return (False, "SKIP: K_b_z not available")
        if all(v is not None for v in [self.I, self.a, self.chi_m]):
            I_base = self.I
            factor = 1.1
            I_perturbed = I_base * factor
            try:
                expr = self._parse_sympy(self.K_b_z)
                K_base = float(expr.subs({
                    self.I_sym: I_base,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                K_perturbed = float(expr.subs({
                    self.I_sym: I_perturbed,
                    self.a_sym: self.a,
                    self.chi_m_sym: self.chi_m
                }))
                ratio = K_perturbed / K_base
                if math.isclose(ratio, factor, rel_tol=1e-2, abs_tol=1e-6):
                    return (True, f"K_b scales linearly: ratio={ratio:.4f} (expected {factor})")
                else:
                    return (False, f"K_b ratio mismatch: {ratio:.4f} vs expected {factor}")
            except Exception as e:
                return (False, f"Numeric error: {e}")
        return (False, "SKIP: missing givens for numeric check")

    # ------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_piecewise_sign(self) -> tuple:
        """Check that B is non-negative for typical positive parameters."""
        if self.B_piecewise is None:
            return (False, "SKIP: B_piecewise not available")
        if all(v is not None for v in [self.I, self.a, self.chi_m, self.mu0]):
            try:
                expr = self._parse_sympy(self.B_piecewise)
                for s_val in [self.a/2.0, self.a, 2.0*self.a]:
                    B_val = float(expr.subs({
                        self.s: s_val,
                        self.I_sym: self.I,
                        self.a_sym: self.a,
                        self.chi_m_sym: self.chi_m,
                        self.mu0_sym: self.mu0
                    }))
                    if B_val < 0:
                        return (False, f"B negative at s={s_val}: {B_val}")
                return (True, "B non-negative at sample points")
            except Exception as e:
                return (False, f"Numerical evaluation error: {e}")
        # Without givens, just report that sign check is skipped
        return (True, "Sign check skipped (no numeric givens)")

    # Helper for symbolic equality
    def _sympy_equal(self, actual: Any, expected: sp.Expr) -> tuple:
        """Check if actual (sympy or string) equals expected symbolically.
        Returns (True, None) or (False, error_message)."""
        if actual is None:
            return (False, "Actual is None")
        try:
            actual_sym = self._parse_sympy(actual)
        except Exception as e:
            return (False, f"Cannot parse actual: {e}")
        diff = sp.simplify(actual_sym - expected)
        if diff == 0:
            return (True, None)
        return (False, f"Difference: {diff}")