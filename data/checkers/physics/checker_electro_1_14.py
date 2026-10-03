import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Auditor for electric field and potential of a uniformly charged sphere.
    Checks interface, Gauss's law invariants, modeling expressions, continuity,
    derivative relations, monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Givens (may also appear in results; prefer givens)
        self.R = self.givens.get('R', self.results.get('R', None))
        self.rho = self.givens.get('rho', self.results.get('rho', None))
        self.epsilon0 = self.givens.get('epsilon0', self.results.get('epsilon0', None))
        # pi is usually a constant, but check if provided
        self.pi = self.givens.get('pi', self.results.get('pi', math.pi))

        # Expected result keys
        self.E_outside = self.results.get('E_outside', None)
        self.E_inside = self.results.get('E_inside', None)
        self.phi_outside = self.results.get('phi_outside', None)
        self.phi_inside = self.results.get('phi_inside', None)

    # ------------------------------------------------------------------
    # Helper: safely convert to sympy expression
    # ------------------------------------------------------------------
    def _sympify(self, expr):
        if expr is None:
            return None
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception:
                return None
        if isinstance(expr, sp.Basic):
            return expr
        # Might be a number or other type; try sympify
        try:
            return sp.sympify(expr)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Symbolic expected forms
    # ------------------------------------------------------------------
    def _expected_E_outside(self):
        R, r, rho, eps0 = sp.symbols('R r rho epsilon0')
        Q_total = (4/3) * sp.pi * R**3 * rho
        return Q_total / (4 * sp.pi * eps0 * r**2)

    def _expected_E_inside(self):
        r, rho, eps0 = sp.symbols('r rho epsilon0')
        return (rho * r) / (3 * eps0)

    def _expected_phi_outside(self):
        R, r, rho, eps0 = sp.symbols('R r rho epsilon0')
        Q_total = (4/3) * sp.pi * R**3 * rho
        return Q_total / (4 * sp.pi * eps0 * r)

    def _expected_phi_inside(self):
        R, r, rho, eps0 = sp.symbols('R r rho epsilon0')
        return (rho / (6 * eps0)) * (3 * R**2 - r**2)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four result keys exist."""
        missing = []
        for key in ['E_outside', 'E_inside', 'phi_outside', 'phi_inside']:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens(self):
        """Check that essential givens (R, rho, epsilon0) are provided."""
        missing = []
        if self.R is None:
            missing.append('R')
        if self.rho is None:
            missing.append('rho')
        if self.epsilon0 is None:
            missing.append('epsilon0')
        if missing:
            return (False, f"Missing required givens: {missing}")
        return (True, "All essential givens present")

    # ------------------------------------------------------------------
    # Invariant tests (Gauss law)
    # ------------------------------------------------------------------
    def test_invariant_gauss_outside(self):
        """For r>R, check E_outside * 4π r^2 = Q_total/ε0."""
        E = self._sympify(self.E_outside)
        if E is None:
            return (False, "SKIP: E_outside not available or not sympifiable")

        # Build left and right sides symbolically
        R, r, rho, eps0 = sp.symbols('R r rho epsilon0')
        pi = sp.pi
        Q_total = (4/3) * pi * R**3 * rho
        left = E * 4 * pi * r**2
        right = Q_total / eps0
        diff = sp.simplify(left - right)
        if diff == 0:
            return (True, "Gauss law outside satisfied symbolically")

        # Numerical fallback if constants are given
        if self.R is not None and self.rho is not None and self.epsilon0 is not None:
            try:
                subs = {R: self.R, rho: self.rho, eps0: self.epsilon0}
                # pick a test r > R
                test_r = self.R + 0.1
                left_val = left.subs({r: test_r, **subs}).evalf()
                right_val = right.subs(subs).evalf()
                if math.isclose(float(left_val), float(right_val), rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "Gauss law outside satisfied numerically")
                else:
                    return (False, f"Numerical mismatch: left={left_val}, right={right_val}")
            except Exception as e:
                return (False, f"Numerical fallback error: {e}")
        else:
            return (False, f"Symbolic check diff = {diff}. Insufficient givens for numeric check.")

    def test_invariant_gauss_inside(self):
        """For r<=R, check E_inside * 4π r^2 = Q_enc/ε0."""
        E = self._sympify(self.E_inside)
        if E is None:
            return (False, "SKIP: E_inside not available or not sympifiable")

        R, r, rho, eps0 = sp.symbols('R r rho epsilon0')
        pi = sp.pi
        Q_enc = (4/3) * pi * r**3 * rho
        left = E * 4 * pi * r**2
        right = Q_enc / eps0
        diff = sp.simplify(left - right)
        if diff == 0:
            return (True, "Gauss law inside satisfied symbolically")

        if self.R is not None and self.rho is not None and self.epsilon0 is not None:
            try:
                subs = {R: self.R, rho: self.rho, eps0: self.epsilon0}
                test_r = self.R / 2  # sample inside
                left_val = left.subs({r: test_r, **subs}).evalf()
                right_val = right.subs({r: test_r, **subs}).evalf()
                if math.isclose(float(left_val), float(right_val), rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "Gauss law inside satisfied numerically")
                else:
                    return (False, f"Numerical mismatch: left={left_val}, right={right_val}")
            except Exception as e:
                return (False, f"Numerical fallback error: {e}")
        else:
            return (False, f"Symbolic check diff = {diff}. Insufficient givens for numeric check.")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_trace_e_field_expression_outside(self):
        """Check that E_outside matches expected form Q_total/(4πε0 r^2)."""
        E = self._sympify(self.E_outside)
        if E is None:
            return (False, "SKIP: E_outside not sympifiable")
        expected = self._expected_E_outside()
        diff = sp.simplify(E - expected)
        if diff == 0:
            return (True, "E_outside expression matches expected form")
        return (False, f"E_outside differs: simplified difference = {diff}")

    def test_trace_e_field_expression_inside(self):
        """Check that E_inside matches expected form (ρ r)/(3ε0)."""
        E = self._sympify(self.E_inside)
        if E is None:
            return (False, "SKIP: E_inside not sympifiable")
        expected = self._expected_E_inside()
        diff = sp.simplify(E - expected)
        if diff == 0:
            return (True, "E_inside expression matches expected form")
        return (False, f"E_inside differs: simplified difference = {diff}")

    def test_trace_phi_continuity_at_rR(self):
        """Check that phi_outside(R) == phi_inside(R)."""
        phi_out = self._sympify(self.phi_outside)
        phi_in = self._sympify(self.phi_inside)
        if phi_out is None or phi_in is None:
            return (False, "SKIP: one of phi expressions missing")

        R = sp.symbols('R')
        diff = sp.simplify(phi_out.subs({sp.Symbol('r'): R}) - phi_in.subs({sp.Symbol('r'): R}))
        if diff == 0:
            return (True, "Potential is continuous at r=R")
        return (False, f"Potential not continuous at r=R: difference = {diff}")

    def test_trace_phi_derivative_outside(self):
        """Check that -d(phi_outside)/dr ≈ E_outside for a sample r > R."""
        phi_out = self._sympify(self.phi_outside)
        E_out = self._sympify(self.E_outside)
        if phi_out is None or E_out is None:
            return (False, "SKIP: phi_outside or E_outside missing")
        r = sp.symbols('r', positive=True)
        dphi_dr = sp.diff(phi_out, r)
        diff = sp.simplify(-dphi_dr - E_out)
        if diff == 0:
            return (True, "E = -dφ/dr outside satisfied symbolically")

        # Numeric fallback
        if self.R is not None and self.rho is not None and self.epsilon0 is not None:
            try:
                subs = {sp.Symbol('R'): self.R, sp.Symbol('rho'): self.rho, sp.Symbol('epsilon0'): self.epsilon0}
                test_r = self.R + 0.5
                dphi_val = float(dphi_dr.subs({r: test_r, **subs}).evalf())
                E_val = float(E_out.subs({r: test_r, **subs}).evalf())
                if math.isclose(-dphi_val, E_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "E = -dφ/dr outside satisfied numerically")
                else:
                    return (False, f"Numerical mismatch: -dφ/dr={-dphi_val}, E={E_val}")
            except Exception as e:
                return (False, f"Numeric fallback error: {e}")
        else:
            return (False, f"Symbolic difference = {diff} (non‑zero). Insufficient givens for numeric.")

    def test_trace_phi_derivative_inside(self):
        """Check that -d(phi_inside)/dr ≈ E_inside for a sample r < R."""
        phi_in = self._sympify(self.phi_inside)
        E_in = self._sympify(self.E_inside)
        if phi_in is None or E_in is None:
            return (False, "SKIP: phi_inside or E_inside missing")
        r = sp.symbols('r', positive=True)
        dphi_dr = sp.diff(phi_in, r)
        diff = sp.simplify(-dphi_dr - E_in)
        if diff == 0:
            return (True, "E = -dφ/dr inside satisfied symbolically")

        if self.R is not None and self.rho is not None and self.epsilon0 is not None:
            try:
                subs = {sp.Symbol('R'): self.R, sp.Symbol('rho'): self.rho, sp.Symbol('epsilon0'): self.epsilon0}
                test_r = self.R / 2
                dphi_val = float(dphi_dr.subs({r: test_r, **subs}).evalf())
                E_val = float(E_in.subs({r: test_r, **subs}).evalf())
                if math.isclose(-dphi_val, E_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "E = -dφ/dr inside satisfied numerically")
                else:
                    return (False, f"Numerical mismatch: -dφ/dr={-dphi_val}, E={E_val}")
            except Exception as e:
                return (False, f"Numeric fallback error: {e}")
        else:
            return (False, f"Symbolic difference = {diff} (non‑zero). Insufficient givens for numeric.")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_e_outside_increases_with_rho(self):
        """Perturb charge density rho (+20%) and verify E_outside increases."""
        E = self._sympify(self.E_outside)
        if E is None:
            return (False, "SKIP: E_outside not sympifiable")
        if self.R is None or self.rho is None or self.epsilon0 is None:
            return (False, "SKIP: insufficient givens (R, rho, epsilon0) for numeric perturbation")
        try:
            # Use a fixed test radius > R
            test_r = self.R + 0.5
            R_sym, rho_sym, eps0_sym = sp.symbols('R rho epsilon0')
            subs_baseline = {R_sym: self.R, rho_sym: self.rho, eps0_sym: self.epsilon0}
            # Baseline value
            baseline = float(E.subs({sp.Symbol('r'): test_r, **subs_baseline}).evalf())
            # Perturbed: increase rho by 20%
            rho_perturbed = self.rho * 1.2
            subs_perturbed = {R_sym: self.R, rho_sym: rho_perturbed, eps0_sym: self.epsilon0}
            perturbed = float(E.subs({sp.Symbol('r'): test_r, **subs_perturbed}).evalf())
            if perturbed > baseline:
                return (True, f"E_outside increases with rho: baseline={baseline}, perturbed (rho+20%)={perturbed}")
            else:
                return (False, f"E_outside did not increase: baseline={baseline}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Numeric perturbation error: {e}")

    def test_monotonicity_phi_outside_decreases_with_epsilon0(self):
        """Perturb permittivity epsilon0 (+10%) and verify phi_outside decreases."""
        phi = self._sympify(self.phi_outside)
        if phi is None:
            return (False, "SKIP: phi_outside not sympifiable")
        if self.R is None or self.rho is None or self.epsilon0 is None:
            return (False, "SKIP: insufficient givens for numeric perturbation")
        try:
            test_r = self.R + 1.0
            R_sym, rho_sym, eps0_sym = sp.symbols('R rho epsilon0')
            subs_baseline = {R_sym: self.R, rho_sym: self.rho, eps0_sym: self.epsilon0}
            baseline = float(phi.subs({sp.Symbol('r'): test_r, **subs_baseline}).evalf())
            # Increase epsilon0 by 10%
            eps_perturbed = self.epsilon0 * 1.1
            subs_perturbed = {R_sym: self.R, rho_sym: self.rho, eps0_sym: eps_perturbed}
            perturbed = float(phi.subs({sp.Symbol('r'): test_r, **subs_perturbed}).evalf())
            if perturbed < baseline:
                return (True, f"phi_outside decreases with epsilon0: baseline={baseline}, perturbed (eps0+10%)={perturbed}")
            else:
                return (False, f"phi_outside did not decrease: baseline={baseline}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Numeric perturbation error: {e}")

    def test_monotonicity_phi_inside_increases_with_rho(self):
        """Perturb charge density rho (+20%) and verify phi_inside increases."""
        phi = self._sympify(self.phi_inside)
        if phi is None:
            return (False, "SKIP: phi_inside not sympifiable")
        if self.R is None or self.rho is None or self.epsilon0 is None:
            return (False, "SKIP: insufficient givens for numeric perturbation")
        try:
            test_r = self.R / 2  # inside
            R_sym, rho_sym, eps0_sym = sp.symbols('R rho epsilon0')
            subs_baseline = {R_sym: self.R, rho_sym: self.rho, eps0_sym: self.epsilon0}
            baseline = float(phi.subs({sp.Symbol('r'): test_r, **subs_baseline}).evalf())
            # Increase rho by 20%
            rho_pert = self.rho * 1.2
            subs_perturbed = {R_sym: self.R, rho_sym: rho_pert, eps0_sym: self.epsilon0}
            perturbed = float(phi.subs({sp.Symbol('r'): test_r, **subs_perturbed}).evalf())
            if perturbed > baseline:
                return (True, f"phi_inside increases with rho: baseline={baseline}, perturbed (rho+20%)={perturbed}")
            else:
                return (False, f"phi_inside did not increase: baseline={baseline}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Numeric perturbation error: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_phi_finite(self):
        """Check that phi values are finite (not NaN or inf) when numeric."""
        # This test is weak: just ensure we can evaluate numerically without errors
        for name, val in [('phi_outside', self.phi_outside), ('phi_inside', self.phi_inside)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            sym_val = self._sympify(val)
            if sym_val is None:
                continue
            # If numeric, check finite
            if sym_val.is_Number:
                if math.isfinite(float(sym_val)):
                    continue
                else:
                    return (False, f"{name} is not finite: {sym_val}")
            # Otherwise symbolic – assume finite as long as no division by zero issues (skip)
        return (True, "All phi forms are finite (or symbolic)")

    # ------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)