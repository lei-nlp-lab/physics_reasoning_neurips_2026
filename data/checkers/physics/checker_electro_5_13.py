import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.analysis_id = getattr(solution_instance, 'analysis_id', 'unknown')
        givens = getattr(solution_instance, 'givens', {})
        self.q = givens.get('q') if isinstance(givens, dict) else getattr(solution_instance, 'q', None)
        self.l = givens.get('l') if isinstance(givens, dict) else getattr(solution_instance, 'l', None)
        self.R = givens.get('R') if isinstance(givens, dict) else getattr(solution_instance, 'R', None)
        self.v = givens.get('v') if isinstance(givens, dict) else getattr(solution_instance, 'v', None)
        self.epsilon0 = givens.get('epsilon0') if isinstance(givens, dict) else getattr(solution_instance, 'epsilon0', None)
        self.mu0 = givens.get('mu0') if isinstance(givens, dict) else getattr(solution_instance, 'mu0', None)
        self.c = givens.get('c') if isinstance(givens, dict) else getattr(solution_instance, 'c', None)

        self.results = getattr(solution_instance, 'results', {})
        self.E_r_inside = self.results.get('E_r_inside')
        self.E_r_outside = self.results.get('E_r_outside')
        self.B_phi_inside = self.results.get('B_phi_inside')
        self.B_phi_outside = self.results.get('B_phi_outside')
        self.u_inside = self.results.get('u_inside')
        self.u_outside = self.results.get('u_outside')
        self.g_z_inside = self.results.get('g_z_inside')
        self.g_z_outside = self.results.get('g_z_outside')

        # Sample radial coordinates (avoid division by zero later)
        self.r_in = 0.5 * (self.R if self.R is not None else 1.0)
        self.r_out = 2.0 * (self.R if self.R is not None else 1.0)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        required_keys = [
            'E_r_inside', 'E_r_outside', 'B_phi_inside', 'B_phi_outside',
            'u_inside', 'u_outside', 'g_z_inside', 'g_z_outside'
        ]
        missing = [k for k in required_keys if self.results.get(k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_givens_present(self):
        missing_givens = []
        for attr in ['q', 'l', 'R', 'v', 'epsilon0', 'mu0', 'c']:
            if getattr(self, attr) is None:
                missing_givens.append(attr)
        if missing_givens:
            return (False, f"Missing givens: {missing_givens}")
        return (True, "All essential givens present.")

    # ------------------------------------------------------------------
    # Invariant tests
    # ------------------------------------------------------------------
    def test_invariant_gauss_law_inside(self):
        if any(x is None for x in [self.E_r_inside, self.q, self.l, self.R, self.epsilon0]):
            return (False, "SKIP: missing data for Gauss law inside")
        r = self.r_in
        try:
            E_sym = sp.sympify(self.E_r_inside) if not isinstance(self.E_r_inside, sp.Expr) else self.E_r_inside
            sym_r = sp.Symbol('r')
            lhs_sym = (1/sym_r) * sp.diff(sym_r * E_sym, sym_r)
            lhs_val = float(lhs_sym.subs(sym_r, r))
            rho = float(self.q / (math.pi * self.l * self.R**2))
            rhs_val = float(rho / self.epsilon0)
            if math.isclose(lhs_val, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Gauss inside: LHS={lhs_val:.3e}, RHS={rhs_val:.3e}")
            else:
                return (False, f"Gauss inside mismatch: LHS={lhs_val:.3e}, RHS={rhs_val:.3e}")
        except Exception as e:
            return (False, f"test_invariant_gauss_law_inside crashed: {e}")

    def test_invariant_gauss_law_outside(self):
        if any(x is None for x in [self.E_r_outside, self.q, self.l, self.R, self.epsilon0]):
            return (False, "SKIP: missing data for Gauss law outside")
        r = self.r_out
        try:
            E_sym = sp.sympify(self.E_r_outside) if not isinstance(self.E_r_outside, sp.Expr) else self.E_r_outside
            sym_r = sp.Symbol('r')
            lhs_sym = (1/sym_r) * sp.diff(sym_r * E_sym, sym_r)
            lhs_val = float(lhs_sym.subs(sym_r, r))
            if math.isclose(lhs_val, 0, abs_tol=1e-2):
                return (True, f"Gauss outside zero: LHS={lhs_val:.3e}")
            else:
                return (False, f"Gauss outside non‑zero: LHS={lhs_val:.3e}")
        except Exception as e:
            return (False, f"test_invariant_gauss_law_outside crashed: {e}")

    def test_invariant_ampere_law_inside(self):
        if any(x is None for x in [self.B_phi_inside, self.q, self.l, self.R, self.v, self.mu0]):
            return (False, "SKIP: missing data for Ampere inside")
        r = self.r_in
        try:
            B_sym = sp.sympify(self.B_phi_inside) if not isinstance(self.B_phi_inside, sp.Expr) else self.B_phi_inside
            sym_r = sp.Symbol('r')
            lhs_sym = (1/sym_r) * sp.diff(sym_r * B_sym, sym_r)
            lhs_val = float(lhs_sym.subs(sym_r, r))
            rho = float(self.q / (math.pi * self.l * self.R**2))
            J = float(rho * self.v)
            rhs_val = float(self.mu0 * J)
            if math.isclose(lhs_val, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Ampere inside: LHS={lhs_val:.3e}, RHS={rhs_val:.3e}")
            else:
                return (False, f"Ampere inside mismatch: LHS={lhs_val:.3e}, RHS={rhs_val:.3e}")
        except Exception as e:
            return (False, f"test_invariant_ampere_law_inside crashed: {e}")

    def test_invariant_ampere_law_outside(self):
        if any(x is None for x in [self.B_phi_outside, self.q, self.l, self.R, self.v, self.mu0]):
            return (False, "SKIP: missing data for Ampere outside")
        r = self.r_out
        try:
            B_sym = sp.sympify(self.B_phi_outside) if not isinstance(self.B_phi_outside, sp.Expr) else self.B_phi_outside
            sym_r = sp.Symbol('r')
            lhs_sym = (1/sym_r) * sp.diff(sym_r * B_sym, sym_r)
            lhs_val = float(lhs_sym.subs(sym_r, r))
            if math.isclose(lhs_val, 0, abs_tol=1e-2):
                return (True, f"Ampere outside zero: LHS={lhs_val:.3e}")
            else:
                return (False, f"Ampere outside non‑zero: LHS={lhs_val:.3e}")
        except Exception as e:
            return (False, f"test_invariant_ampere_law_outside crashed: {e}")

    def test_invariant_boundary_conditions(self):
        if any(x is None for x in [self.E_r_inside, self.E_r_outside,
                                    self.B_phi_inside, self.B_phi_outside,
                                    self.q, self.l, self.R]):
            return (False, "SKIP: missing data for boundary check")
        r = self.R
        try:
            E_inside_at_R = float(sp.sympify(self.E_r_inside).subs(sp.Symbol('r'), r))
            E_outside_at_R = float(sp.sympify(self.E_r_outside).subs(sp.Symbol('r'), r))
            B_inside_at_R = float(sp.sympify(self.B_phi_inside).subs(sp.Symbol('r'), r))
            B_outside_at_R = float(sp.sympify(self.B_phi_outside).subs(sp.Symbol('r'), r))
            ok = True
            msg = ""
            if not math.isclose(E_inside_at_R, E_outside_at_R, rel_tol=1e-2, abs_tol=1e-2):
                ok = False
                msg += f"E_r mismatch at R: inside={E_inside_at_R:.3e}, outside={E_outside_at_R:.3e}. "
            if not math.isclose(B_inside_at_R, B_outside_at_R, rel_tol=1e-2, abs_tol=1e-2):
                ok = False
                msg += f"B_phi mismatch at R: inside={B_inside_at_R:.3e}, outside={B_outside_at_R:.3e}. "
            if ok:
                return (True, "E_r and B_phi continuous at r=R.")
            else:
                return (False, msg)
        except Exception as e:
            return (False, f"test_invariant_boundary_conditions crashed: {e}")

    def test_invariant_energy_density_inside(self):
        if any(x is None for x in [self.u_inside, self.E_r_inside, self.B_phi_inside,
                                    self.epsilon0, self.mu0]):
            return (False, "SKIP: missing data for energy density inside")
        r = self.r_in
        try:
            E = float(sp.sympify(self.E_r_inside).subs(sp.Symbol('r'), r))
            B = float(sp.sympify(self.B_phi_inside).subs(sp.Symbol('r'), r))
            u_computed = 0.5 * self.epsilon0 * E**2 + 0.5 / self.mu0 * B**2
            u_given = float(sp.sympify(self.u_inside).subs(sp.Symbol('r'), r))
            if math.isclose(u_given, u_computed, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"u_inside OK: {u_given:.3e} vs computed {u_computed:.3e}")
            else:
                return (False, f"u_inside mismatch: {u_given:.3e} vs computed {u_computed:.3e}")
        except Exception as e:
            return (False, f"test_invariant_energy_density_inside crashed: {e}")

    def test_invariant_energy_density_outside(self):
        if any(x is None for x in [self.u_outside, self.E_r_outside, self.B_phi_outside,
                                    self.epsilon0, self.mu0]):
            return (False, "SKIP: missing data for energy density outside")
        r = self.r_out
        try:
            E = float(sp.sympify(self.E_r_outside).subs(sp.Symbol('r'), r))
            B = float(sp.sympify(self.B_phi_outside).subs(sp.Symbol('r'), r))
            u_computed = 0.5 * self.epsilon0 * E**2 + 0.5 / self.mu0 * B**2
            u_given = float(sp.sympify(self.u_outside).subs(sp.Symbol('r'), r))
            if math.isclose(u_given, u_computed, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"u_outside OK: {u_given:.3e} vs computed {u_computed:.3e}")
            else:
                return (False, f"u_outside mismatch: {u_given:.3e} vs computed {u_computed:.3e}")
        except Exception as e:
            return (False, f"test_invariant_energy_density_outside crashed: {e}")

    def test_invariant_momentum_density_inside(self):
        if any(x is None for x in [self.g_z_inside, self.E_r_inside, self.B_phi_inside, self.epsilon0]):
            return (False, "SKIP: missing data for momentum density inside")
        r = self.r_in
        try:
            E = float(sp.sympify(self.E_r_inside).subs(sp.Symbol('r'), r))
            B = float(sp.sympify(self.B_phi_inside).subs(sp.Symbol('r'), r))
            g_computed = self.epsilon0 * E * B
            g_given = float(sp.sympify(self.g_z_inside).subs(sp.Symbol('r'), r))
            if math.isclose(g_given, g_computed, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"g_z_inside OK: {g_given:.3e} vs computed {g_computed:.3e}")
            else:
                return (False, f"g_z_inside mismatch: {g_given:.3e} vs computed {g_computed:.3e}")
        except Exception as e:
            return (False, f"test_invariant_momentum_density_inside crashed: {e}")

    def test_invariant_momentum_density_outside(self):
        if any(x is None for x in [self.g_z_outside, self.E_r_outside, self.B_phi_outside, self.epsilon0]):
            return (False, "SKIP: missing data for momentum density outside")
        r = self.r_out
        try:
            E = float(sp.sympify(self.E_r_outside).subs(sp.Symbol('r'), r))
            B = float(sp.sympify(self.B_phi_outside).subs(sp.Symbol('r'), r))
            g_computed = self.epsilon0 * E * B
            g_given = float(sp.sympify(self.g_z_outside).subs(sp.Symbol('r'), r))
            if math.isclose(g_given, g_computed, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"g_z_outside OK: {g_given:.3e} vs computed {g_computed:.3e}")
            else:
                return (False, f"g_z_outside mismatch: {g_given:.3e} vs computed {g_computed:.3e}")
        except Exception as e:
            return (False, f"test_invariant_momentum_density_outside crashed: {e}")

    def test_invariant_E_B_ratio_inside(self):
        if any(x is None for x in [self.E_r_inside, self.B_phi_inside, self.v, self.c]):
            return (False, "SKIP: missing data for E‑B ratio inside")
        r = self.r_in
        try:
            E = float(sp.sympify(self.E_r_inside).subs(sp.Symbol('r'), r))
            B = float(sp.sympify(self.B_phi_inside).subs(sp.Symbol('r'), r))
            expected_B = (self.v / self.c**2) * E
            if math.isclose(B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E‑B ratio inside OK: B={B:.3e}, v/c^2*E={expected_B:.3e}")
            else:
                return (False, f"E‑B ratio inside mismatch: B={B:.3e}, expected {expected_B:.3e}")
        except Exception as e:
            return (False, f"test_invariant_E_B_ratio_inside crashed: {e}")

    def test_invariant_E_B_ratio_outside(self):
        if any(x is None for x in [self.E_r_outside, self.B_phi_outside, self.v, self.c]):
            return (False, "SKIP: missing data for E‑B ratio outside")
        r = self.r_out
        try:
            E = float(sp.sympify(self.E_r_outside).subs(sp.Symbol('r'), r))
            B = float(sp.sympify(self.B_phi_outside).subs(sp.Symbol('r'), r))
            expected_B = (self.v / self.c**2) * E
            if math.isclose(B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E‑B ratio outside OK: B={B:.3e}, v/c^2*E={expected_B:.3e}")
            else:
                return (False, f"E‑B ratio outside mismatch: B={B:.3e}, expected {expected_B:.3e}")
        except Exception as e:
            return (False, f"test_invariant_E_B_ratio_outside crashed: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_electric_field_r_dependence(self):
        """Check that E_r ∝ r inside and ∝ 1/r outside."""
        if any(x is None for x in [self.E_r_inside, self.E_r_outside, self.R]):
            return (False, "SKIP: missing data for modeling E radial")
        try:
            sym_r = sp.Symbol('r')
            # inside: E ∝ r  => E(r)/r should be constant
            r1 = self.r_in * 0.5
            r2 = self.r_in
            E1 = float(sp.sympify(self.E_r_inside).subs(sym_r, r1))
            E2 = float(sp.sympify(self.E_r_inside).subs(sym_r, r2))
            ratio1 = E1 / r1 if r1 != 0 else None
            ratio2 = E2 / r2 if r2 != 0 else None
            if ratio1 is None or ratio2 is None:
                return (False, "Zero radius encountered in inside check")
            if not math.isclose(ratio1, ratio2, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"E_r inside not linear: ratios {ratio1:.3e} vs {ratio2:.3e}")

            # outside: E ∝ 1/r => E(r)*r should be constant
            r3 = self.r_out
            r4 = self.r_out * 1.5
            E3 = float(sp.sympify(self.E_r_outside).subs(sym_r, r3))
            E4 = float(sp.sympify(self.E_r_outside).subs(sym_r, r4))
            prod3 = E3 * r3
            prod4 = E4 * r4
            if not math.isclose(prod3, prod4, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"E_r outside not 1/r: products {prod3:.3e} vs {prod4:.3e}")
            return (True, "E_r radial dependence correct: linear inside, 1/r outside.")
        except Exception as e:
            return (False, f"test_modeling_electric_field_r_dependence crashed: {e}")

    def test_modeling_energy_density_r_dependence(self):
        """Check that u ∝ r^2 inside and ∝ 1/r^2 outside."""
        if any(x is None for x in [self.u_inside, self.u_outside, self.R]):
            return (False, "SKIP: missing data for modeling energy density")
        try:
            sym_r = sp.Symbol('r')
            # inside: u ∝ r^2 => u(r)/r^2 constant
            r1 = self.r_in * 0.5
            r2 = self.r_in
            u1 = float(sp.sympify(self.u_inside).subs(sym_r, r1))
            u2 = float(sp.sympify(self.u_inside).subs(sym_r, r2))
            norm1 = u1 / (r1**2) if r1 != 0 else None
            norm2 = u2 / (r2**2) if r2 != 0 else None
            if norm1 is None or norm2 is None:
                return (False, "Zero radius encountered in inside check")
            if not math.isclose(norm1, norm2, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"u_inside not ∝ r^2: norms {norm1:.3e} vs {norm2:.3e}")

            # outside: u ∝ 1/r^2 => u(r)*r^2 constant
            r3 = self.r_out
            r4 = self.r_out * 1.5
            u3 = float(sp.sympify(self.u_outside).subs(sym_r, r3))
            u4 = float(sp.sympify(self.u_outside).subs(sym_r, r4))
            prod3 = u3 * (r3**2)
            prod4 = u4 * (r4**2)
            if not math.isclose(prod3, prod4, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"u_outside not ∝ 1/r^2: products {prod3:.3e} vs {prod4:.3e}")
            return (True, "Energy density radial dependence correct: ∝ r^2 inside, ∝ 1/r^2 outside.")
        except Exception as e:
            return (False, f"test_modeling_energy_density_r_dependence crashed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_E_inside_linear(self):
        """Perturb r inside: E should increase with r."""
        if any(x is None for x in [self.E_r_inside]):
            return (False, "SKIP: missing data for E inside monotonicity")
        try:
            r_small = self.r_in * 0.5
            r_large = self.r_in
            E_small = float(sp.sympify(self.E_r_inside).subs(sp.Symbol('r'), r_small))
            E_large = float(sp.sympify(self.E_r_inside).subs(sp.Symbol('r'), r_large))
            if E_large > E_small and E_small > 0:
                return (True, f"E_r inside increases with r: {E_small:.3e} to {E_large:.3e}")
            else:
                return (False, f"Non‑increasing: E({r_small})={E_small:.3e}, E({r_large})={E_large:.3e}")
        except Exception as e:
            return (False, f"test_monotonicity_E_inside_linear crashed: {e}")

    def test_monotonicity_E_outside_decays(self):
        """Perturb r outside: E should decrease with r."""
        if any(x is None for x in [self.E_r_outside]):
            return (False, "SKIP: missing data for E outside monotonicity")
        try:
            r_small = self.r_out
            r_large = self.r_out * 1.5
            E_small = float(sp.sympify(self.E_r_outside).subs(sp.Symbol('r'), r_small))
            E_large = float(sp.sympify(self.E_r_outside).subs(sp.Symbol('r'), r_large))
            if E_large < E_small and E_small > 0:
                return (True, f"E_r outside decreases with r: {E_small:.3e} to {E_large:.3e}")
            else:
                return (False, f"Non‑decreasing: E({r_small})={E_small:.3e}, E({r_large})={E_large:.3e}")
        except Exception as e:
            return (False, f"test_monotonicity_E_outside_decays crashed: {e}")

    def test_monotonicity_B_phi_inside_linear(self):
        """Perturb r inside: B should increase with r."""
        if any(x is None for x in [self.B_phi_inside]):
            return (False, "SKIP: missing data for B inside monotonicity")
        try:
            r_small = self.r_in * 0.5
            r_large = self.r_in
            B_small = float(sp.sympify(self.B_phi_inside).subs(sp.Symbol('r'), r_small))
            B_large = float(sp.sympify(self.B_phi_inside).subs(sp.Symbol('r'), r_large))
            if B_large > B_small and B_small > 0:
                return (True, f"B_phi inside increases with r: {B_small:.3e} to {B_large:.3e}")
            else:
                return (False, f"Non‑increasing: B({r_small})={B_small:.3e}, B({r_large})={B_large:.3e}")
        except Exception as e:
            return (False, f"test_monotonicity_B_phi_inside_linear crashed: {e}")

    def test_monotonicity_B_phi_outside_decays(self):
        """Perturb r outside: B should decrease with r."""
        if any(x is None for x in [self.B_phi_outside]):
            return (False, "SKIP: missing data for B outside monotonicity")
        try:
            r_small = self.r_out
            r_large = self.r_out * 1.5
            B_small = float(sp.sympify(self.B_phi_outside).subs(sp.Symbol('r'), r_small))
            B_large = float(sp.sympify(self.B_phi_outside).subs(sp.Symbol('r'), r_large))
            if B_large < B_small and B_small > 0:
                return (True, f"B_phi outside decreases with r: {B_small:.3e} to {B_large:.3e}")
            else:
                return (False, f"Non‑decreasing: B({r_small})={B_small:.3e}, B({r_large})={B_large:.3e}")
        except Exception as e:
            return (False, f"test_monotonicity_B_phi_outside_decays crashed: {e}")

    def test_monotonicity_B_phi_inside_with_v(self):
        """Perturb v inside: B should increase linearly with v."""
        if any(x is None for x in [self.B_phi_inside, self.v]):
            return (False, "SKIP: missing data for B‑v monotonicity inside")
        try:
            r = self.r_in
            v_base = self.v
            v_new = v_base * 1.1
            # Explicitly recompute B by substituting into expression (if expression contains v)
            B_expr = sp.sympify(self.B_phi_inside)
            B_base = float(B_expr.subs({sp.Symbol('v'): v_base, sp.Symbol('r'): r}))
            B_perturbed = float(B_expr.subs({sp.Symbol('v'): v_new, sp.Symbol('r'): r}))
            expected_B = B_base * (v_new / v_base)
            if math.isclose(B_perturbed, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"B_phi ~ v: base={B_base:.3e}, perturbed={B_perturbed:.3e}, expected={expected_B:.3e}")
            else:
                return (False, f"B_phi not proportional to v: base={B_base:.3e}, perturbed={B_perturbed:.3e}, expected={expected_B:.3e}")
        except Exception as e:
            return (False, f"test_monotonicity_B_phi_inside_with_v crashed: {e}")

    def test_monotonicity_g_z_inside_with_v(self):
        """Perturb v inside: g_z should increase quadratically with v."""
        if any(x is None for x in [self.g_z_inside, self.v]):
            return (False, "SKIP: missing data for g‑v monotonicity inside")
        try:
            r = self.r_in
            v_base = self.v
            v_new = v_base * 1.1
            g_expr = sp.sympify(self.g_z_inside)
            g_base = float(g_expr.subs({sp.Symbol('v'): v_base, sp.Symbol('r'): r}))
            g_perturbed = float(g_expr.subs({sp.Symbol('v'): v_new, sp.Symbol('r'): r}))
            expected_g = g_base * (v_new**2 / v_base**2)
            if math.isclose(g_perturbed, expected_g, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"g_z ~ v^2: base={g_base:.3e}, perturbed={g_perturbed:.3e}, expected={expected_g:.3e}")
            else:
                return (False, f"g_z not proportional to v^2: base={g_base:.3e}, perturbed={g_perturbed:.3e}, expected={expected_g:.3e}")
        except Exception as e:
            return (False, f"test_monotonicity_g_z_inside_with_v crashed: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_nonzero(self):
        keys = ['E_r_inside', 'E_r_outside', 'B_phi_inside', 'B_phi_outside',
                'u_inside', 'u_outside', 'g_z_inside', 'g_z_outside']
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"Missing {key}")
            try:
                r = self.r_in if 'inside' in key else self.r_out
                numeric = float(sp.sympify(val).subs(sp.Symbol('r'), r))
                if numeric <= 0 or math.isnan(numeric) or math.isinf(numeric):
                    return (False, f"{key} is non‑positive or non‑finite: {numeric}")
            except Exception as e:
                return (False, f"Evaluating {key} crashed: {e}")
        return (True, "All results are finite and positive (for r>0, v>0).")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
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