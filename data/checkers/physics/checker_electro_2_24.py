import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Required result quantities
        self.B_r_coeff = self.results.get('B_r_coeff', None)   # expected a(z) = -B0 nu z
        self.j_theta = self.results.get('j_theta', None)       # expected - (B0 nu / mu0) r

        # Physical constants (may be provided or symbolic)
        self.B0 = self.givens.get('B0', None)
        self.nu = self.givens.get('nu', None)
        if self.nu is None:
            self.nu = self.givens.get('ν', None)
        self.mu0 = self.givens.get('mu0', None)
        if self.mu0 is None:
            self.mu0 = self.givens.get('mu_0', None)

        # Symbolic variables
        self.r = sp.Symbol('r', real=True, positive=True)
        self.z = sp.Symbol('z', real=True)
        self.B0_s = sp.Symbol('B0', positive=True)
        self.nu_s = sp.Symbol('nu')
        self.mu0_s = sp.Symbol('mu0', positive=True)

        # Attempt to parse results as sympy expressions
        self.a_sym = self._to_sym(self.B_r_coeff)   # a(z)
        self.j_sym = self._to_sym(self.j_theta)     # j_theta(r)

    def _to_sym(self, expr):
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        try:
            return sp.sympify(expr)
        except (sp.SympifyError, TypeError):
            return None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_has_B_r_coeff(self):
        if self.B_r_coeff is None:
            return False, "Missing B_r_coeff in results"
        return True, "B_r_coeff present"

    def test_interface_has_j_theta(self):
        if self.j_theta is None:
            return False, "Missing j_theta in results"
        return True, "j_theta present"

    def test_interface_B_r_coeff_depends_on_z(self):
        if self.a_sym is None:
            return False, "B_r_coeff cannot be parsed symbolically"
        if self.z not in self.a_sym.free_symbols:
            return False, "B_r_coeff does not depend on z"
        return True, "B_r_coeff depends on z"

    def test_interface_j_theta_depends_on_r(self):
        if self.j_sym is None:
            return False, "j_theta cannot be parsed symbolically"
        if self.r not in self.j_sym.free_symbols:
            return False, "j_theta does not depend on r"
        return True, "j_theta depends on r"

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_divergence_free(self):
        """Check ∇·B=0 near axis: 2*a(z) + ∂B_z0/∂z = 0."""
        if self.a_sym is None:
            return False, "B_r_coeff not available for divergence test"
        # Expected on-axis B_z
        B_z0_expr = self.B0_s * (1 + self.nu_s * self.z**2)
        dB_z0_dz = sp.diff(B_z0_expr, self.z)
        expected_a = -dB_z0_dz / 2   # = -B0*nu*z

        # Substitute known constants if available, otherwise use symbolic
        if self.B0 is not None and self.nu is not None:
            subs = {self.B0_s: self.B0, self.nu_s: self.nu}
            expected_a_num = expected_a.subs(subs)
            provided_a = self.a_sym.subs(subs) if self.B0_s in self.a_sym.free_symbols else self.a_sym
        else:
            expected_a_num = expected_a
            provided_a = self.a_sym

        # Symbolic equality
        diff = sp.simplify(expected_a_num - provided_a)
        if diff == 0:
            return True, "B_r_coeff satisfies divergence condition symbolically"

        # Numerical fallback (isclose)
        try:
            test_z = 0.5
            B0_val = self.B0 if self.B0 is not None else 1.0
            nu_val = self.nu if self.nu is not None else 1.0
            subs_num = {self.z: test_z, self.B0_s: B0_val, self.nu_s: nu_val}
            expected_val = float(expected_a_num.subs(subs_num))
            provided_val = float(provided_a.subs(subs_num))
            if math.isclose(expected_val, provided_val, rel_tol=1e-2, abs_tol=1e-2):
                return True, "B_r_coeff matches divergence near-axis condition numerically"
            else:
                return False, f"B_r_coeff {provided_val} vs expected {expected_val} at z={test_z}"
        except Exception:
            return False, "Cannot evaluate divergence check numerically"

    def test_invariant_curl_j_theta(self):
        """Check j_theta = (1/mu0)(∂B_r/∂z - ∂B_z/∂r) for all r."""
        if self.a_sym is None or self.j_sym is None:
            return False, "B_r_coeff or j_theta missing for curl test"
        # B_r = a(z) * r
        B_r_expr = self.a_sym * self.r
        dB_r_dz = sp.diff(B_r_expr, self.z)
        # For the field to be valid for all r, B_z must be independent of r (from divergence)
        # Then ∂B_z/∂r = 0.
        expected_j = dB_r_dz / self.mu0_s
        diff = sp.simplify(self.j_sym - expected_j)
        if diff == 0:
            return True, "j_theta satisfies curl relation symbolically"

        # Numerical fallback
        try:
            test_r = 0.1
            test_z = 0.5
            B0_val = self.B0 if self.B0 is not None else 1.0
            nu_val = self.nu if self.nu is not None else 1.0
            mu0_val = self.mu0 if self.mu0 is not None else 4e-7 * math.pi
            subs_num = {self.r: test_r, self.z: test_z,
                        self.B0_s: B0_val, self.nu_s: nu_val, self.mu0_s: mu0_val}
            expected_val = float(expected_j.subs(subs_num))
            provided_val = float(self.j_sym.subs(subs_num))
            if math.isclose(expected_val, provided_val, rel_tol=1e-2, abs_tol=1e-2):
                return True, "j_theta matches curl relation numerically"
            else:
                return False, f"j_theta {provided_val} vs expected {expected_val} at r={test_r}, z={test_z}"
        except Exception:
            return False, "Cannot evaluate curl check numerically"

    # ----------------------------------------------------------------------
    # Modeling tests (quality of explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_B_r_linear_in_r(self):
        """B_r near axis should be linear in r: B_r = f(z)*r. a(z) must not depend on r."""
        if self.a_sym is None:
            return False, "B_r_coeff not available"
        if self.r in self.a_sym.free_symbols:
            return False, "B_r_coeff depends on r, violating near-axis linear form"
        return True, "B_r_coeff independent of r (linear in r form)"

    def test_modeling_j_theta_linear_in_r(self):
        """j_theta should be linear in r (proportional to r) and independent of z."""
        if self.j_sym is None:
            return False, "j_theta not available"
        # Check linear in r
        if not self._is_linear_in_r(self.j_sym):
            return False, "j_theta is not linear in r"
        # Check independence of z
        if self.z in self.j_sym.free_symbols:
            return False, "j_theta depends on z unexpectedly (should be independent of z)"
        return True, "j_theta is linear in r and independent of z"

    def _is_linear_in_r(self, expr):
        if not expr.has(self.r):
            return False
        if expr.is_polynomial(self.r) and sp.degree(expr, self.r) == 1:
            coeff = sp.expand(expr / self.r)
            return self.r not in coeff.free_symbols
        return False

    # ----------------------------------------------------------------------
    # Monotonicity tests (trend checks via input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_B_r_increases_with_r(self):
        """For fixed z>0, |B_r| should increase with r (since B_r = a(z)*r)."""
        if self.a_sym is None:
            return False, "B_r_coeff not available"
        try:
            B0_val = self.B0 if self.B0 is not None else 1.0
            nu_val = self.nu if self.nu is not None else 1.0
            z_test = 0.5
            subs = {self.B0_s: B0_val, self.nu_s: nu_val, self.z: z_test}
            a_val = float(self.a_sym.subs(subs))
            # Compare |B_r| at two r values (perturb r)
            r_baseline = 0.1
            r_perturbed = 0.2
            B_r_base = abs(a_val * r_baseline)
            B_r_pert = abs(a_val * r_perturbed)
            if B_r_pert > B_r_base:
                return True, f"|B_r| increases with r: baseline {B_r_base}, perturbed {B_r_pert}"
            else:
                return False, f"|B_r| did not increase: baseline {B_r_base}, perturbed {B_r_pert}"
        except Exception:
            return False, "Cannot perform monotonicity check on B_r with r"

    def test_monotonicity_B_r_decreases_with_z(self):
        """For fixed r, B_r should become more negative as z increases (since a(z) = -B0 nu z)."""
        if self.a_sym is None:
            return False, "B_r_coeff not available"
        try:
            B0_val = self.B0 if self.B0 is not None else 1.0
            nu_val = self.nu if self.nu is not None else 1.0
            r_test = 0.1
            # Compare B_r at two z values (perturb z)
            z_baseline = 0.2
            z_perturbed = 0.5
            subs = {self.B0_s: B0_val, self.nu_s: nu_val, self.r: r_test}
            a_base = float(self.a_sym.subs({**subs, self.z: z_baseline}))
            a_pert = float(self.a_sym.subs({**subs, self.z: z_perturbed}))
            B_r_base = a_base * r_test
            B_r_pert = a_pert * r_test
            if B_r_pert < B_r_base:
                return True, f"B_r becomes more negative as z increases: baseline {B_r_base}, perturbed {B_r_pert}"
            else:
                return False, f"B_r did not decrease: baseline {B_r_base}, perturbed {B_r_pert}"
        except Exception:
            return False, "Cannot perform monotonicity check on B_r with z"

    def test_monotonicity_j_theta_increases_magnitude_with_r(self):
        """|j_theta| should increase linearly with r."""
        if self.j_sym is None:
            return False, "j_theta not available"
        try:
            B0_val = self.B0 if self.B0 is not None else 1.0
            nu_val = self.nu if self.nu is not None else 1.0
            mu0_val = self.mu0 if self.mu0 is not None else 1e-6  # placeholder
            z_test = 0.5
            # Compare at two r values
            r_baseline = 0.05
            r_perturbed = 0.15
            subs = {self.B0_s: B0_val, self.nu_s: nu_val, self.mu0_s: mu0_val, self.z: z_test}
            j_base = abs(float(self.j_sym.subs({**subs, self.r: r_baseline})))
            j_pert = abs(float(self.j_sym.subs({**subs, self.r: r_perturbed})))
            if j_pert > j_base:
                return True, f"|j_theta| increases with r: baseline {j_base}, perturbed {j_pert}"
            else:
                return False, f"|j_theta| did not increase: baseline {j_base}, perturbed {j_pert}"
        except Exception:
            return False, "Cannot perform monotonicity check on j_theta"

    # ----------------------------------------------------------------------
    # Answer sanity (weak, non-exact, non-deriving)
    # ----------------------------------------------------------------------
    def test_answer_sanity_signs(self):
        """
        Weak sign check: For positive B0, nu, and r>0, both B_r_coeff and j_theta should be negative.
        We do not re-derive the expression; just check sign under typical non-zero conditions.
        """
        if self.a_sym is None and self.j_sym is None:
            return False, "Neither B_r_coeff nor j_theta available"
        # Check B_r_coeff sign
        if self.a_sym is not None:
            try:
                B0_val = self.B0 if self.B0 is not None else 1.0
                nu_val = self.nu if self.nu is not None else 1.0
                z_test = 0.5
                subs = {self.B0_s: B0_val, self.nu_s: nu_val, self.z: z_test}
                a_val = float(self.a_sym.subs(subs))
                if a_val >= 0:
                    return False, f"B_r_coeff should be negative but got {a_val} for typical parameters"
            except Exception:
                pass  # skip if cannot evaluate

        # Check j_theta sign
        if self.j_sym is not None:
            try:
                B0_val = self.B0 if self.B0 is not None else 1.0
                nu_val = self.nu if self.nu is not None else 1.0
                mu0_val = self.mu0 if self.mu0 is not None else 1e-6
                r_test = 0.1
                z_test = 0.5
                subs = {self.B0_s: B0_val, self.nu_s: nu_val, self.mu0_s: mu0_val,
                        self.r: r_test, self.z: z_test}
                j_val = float(self.j_sym.subs(subs))
                if j_val >= 0:
                    return False, f"j_theta should be negative but got {j_val} for typical parameters"
            except Exception:
                pass

        return True, "Signs of B_r_coeff and j_theta are negative (as expected) if evaluable"

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
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