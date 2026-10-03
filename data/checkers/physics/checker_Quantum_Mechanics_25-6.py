import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits the energy spectrum calculations for the 3D isotropic harmonic oscillator
    in rectilinear and spherical bases.
    
    Expected keys in solution_instance.results:
        - 'E_rect': energy expression in terms of nx, ny, nz (sympy expression or string)
        - 'E_sph': energy expression in terms of k, ell, m (sympy expression or string)
    
    Additional optional attributes: hbar, omega, nx, ny, nz, ell, m, n_r, k
    """

    def __init__(self, solution_instance):
        # Extract results with None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.E_rect = self.results.get('E_rect', None)
        self.E_sph = self.results.get('E_sph', None)

        # Optional givens (use None defaults)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.nx = getattr(solution_instance, 'nx', None)
        self.ny = getattr(solution_instance, 'ny', None)
        self.nz = getattr(solution_instance, 'nz', None)
        self.ell = getattr(solution_instance, 'ell', None)
        self.m = getattr(solution_instance, 'm', None)
        self.n_r = getattr(solution_instance, 'n_r', None)
        self.k = getattr(solution_instance, 'k', None)

        # Prepare symbolic expressions if needed
        self._prepare_symbolic_expressions()

    def _prepare_symbolic_expressions(self):
        """Convert string expressions to sympy if possible."""
        # Define common symbols
        self.hbar_sym = sp.Symbol('hbar', positive=True)
        self.omega_sym = sp.Symbol('omega', positive=True)
        self.nx_sym = sp.Symbol('nx', integer=True, nonnegative=True)
        self.ny_sym = sp.Symbol('ny', integer=True, nonnegative=True)
        self.nz_sym = sp.Symbol('nz', integer=True, nonnegative=True)
        self.ell_sym = sp.Symbol('ell', integer=True, nonnegative=True)
        self.m_sym = sp.Symbol('m', integer=True)
        self.k_sym = sp.Symbol('k', integer=True, nonnegative=True)
        self.nr_sym = sp.Symbol('n_r', integer=True, nonnegative=True)

        # Convert E_rect
        if isinstance(self.E_rect, str):
            try:
                self.E_rect_expr = sp.sympify(self.E_rect, locals={
                    'hbar': self.hbar_sym, 'omega': self.omega_sym,
                    'nx': self.nx_sym, 'ny': self.ny_sym, 'nz': self.nz_sym
                })
            except:
                self.E_rect_expr = None
        else:
            self.E_rect_expr = self.E_rect  # assume sympy expression or numeric

        # Convert E_sph
        if isinstance(self.E_sph, str):
            try:
                self.E_sph_expr = sp.sympify(self.E_sph, locals={
                    'hbar': self.hbar_sym, 'omega': self.omega_sym,
                    'k': self.k_sym, 'ell': self.ell_sym, 'm': self.m_sym
                })
            except:
                self.E_sph_expr = None
        else:
            self.E_sph_expr = self.E_sph

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_E_rect_present(self):
        if self.E_rect is None:
            return (False, "E_rect result is None or missing from results")
        return (True, "E_rect is present")

    def test_interface_E_sph_present(self):
        if self.E_sph is None:
            return (False, "E_sph result is None or missing from results")
        return (True, "E_sph is present")

    def test_interface_E_rect_type(self):
        if self.E_rect is None:
            return (False, "SKIP: E_rect not available")
        if not isinstance(self.E_rect_expr, (sp.Expr, int, float)):
            return (False, f"E_rect has unsupported type: {type(self.E_rect)}")
        return (True, "E_rect has valid type (sympy expression or number)")

    def test_interface_E_sph_type(self):
        if self.E_sph is None:
            return (False, "SKIP: E_sph not available")
        if not isinstance(self.E_sph_expr, (sp.Expr, int, float)):
            return (False, f"E_sph has unsupported type: {type(self.E_sph)}")
        return (True, "E_sph has valid type (sympy expression or number)")

    # ------------------------------------------------------------------
    # Invariant / theorem-based tests
    # ------------------------------------------------------------------
    def test_invariant_energy_positive(self):
        """Check that both energy expressions are positive for ground state."""
        if self.E_rect_expr is None and self.E_sph_expr is None:
            return (False, "SKIP: no energy expressions available")
        subs = {self.hbar_sym: 1, self.omega_sym: 1, self.nx_sym: 0, self.ny_sym: 0, self.nz_sym: 0,
                self.k_sym: 0, self.ell_sym: 0, self.m_sym: 0}
        try:
            if self.E_rect_expr is not None:
                val = self.E_rect_expr.subs(subs).evalf()
                if val <= 0:
                    return (False, f"E_rect evaluates to non-positive: {val}")
            if self.E_sph_expr is not None:
                val = self.E_sph_expr.subs(subs).evalf()
                if val <= 0:
                    return (False, f"E_sph evaluates to non-positive: {val}")
            return (True, "Energies are positive for ground state")
        except Exception as e:
            return (False, f"Positivity check error: {e}")

    def test_invariant_rotational_invariance_of_E_sph(self):
        """E_sph must not depend on magnetic quantum number m."""
        if self.E_sph_expr is None:
            return (False, "SKIP: E_sph not available")
        if hasattr(self.E_sph_expr, 'free_symbols'):
            if self.m_sym in self.E_sph_expr.free_symbols:
                return (False, "E_sph contains m, violating rotational invariance")
            return (True, "E_sph is independent of m (rotational symmetry)")
        return (False, "Cannot inspect free symbols; use sympy expression")

    def test_invariant_rect_energy_form(self):
        """Check structural equivalence: E_rect = hbar*omega*(nx+ny+nz+3/2)."""
        if self.E_rect_expr is None:
            return (False, "SKIP: E_rect not available")
        expected = self.hbar_sym * self.omega_sym * (self.nx_sym + self.ny_sym + self.nz_sym + sp.Rational(3,2))
        try:
            diff = sp.simplify(self.E_rect_expr - expected)
            if diff == 0:
                return (True, "E_rect exactly matches hbar*omega*(nx+ny+nz+3/2)")
            else:
                return (False, f"Structure difference: {diff}")
        except Exception as e:
            return (False, f"Sympy comparison failed: {e}")

    def test_invariant_sph_energy_form(self):
        """Check structural equivalence: E_sph = hbar*omega*(k+1/2)."""
        if self.E_sph_expr is None:
            return (False, "SKIP: E_sph not available")
        expected = self.hbar_sym * self.omega_sym * (self.k_sym + sp.Rational(1,2))
        try:
            diff = sp.simplify(self.E_sph_expr - expected)
            if diff == 0:
                return (True, "E_sph exactly matches hbar*omega*(k+1/2)")
            else:
                return (False, f"Structure difference: {diff}")
        except Exception as e:
            return (False, f"Sympy comparison failed: {e}")

    def test_invariant_equivalence_of_bases(self):
        """E_rect and E_sph should be equivalent under k = nx+ny+nz+1."""
        if self.E_rect_expr is None or self.E_sph_expr is None:
            return (False, "SKIP: at least one expression missing")
        subs = {self.k_sym: self.nx_sym + self.ny_sym + self.nz_sym + 1}
        try:
            transformed = self.E_sph_expr.subs(subs)
            diff = sp.simplify(self.E_rect_expr - transformed)
            if diff == 0:
                return (True, "E_rect and E_sph match under k = nx+ny+nz+1")
            else:
                return (False, f"Equivalence check difference: {diff}")
        except Exception as e:
            return (False, f"Equivalence check failed: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_separation_of_variables(self):
        """E_rect must be sum of three 1D HO energies hbar*omega*(ni+1/2)."""
        if self.E_rect_expr is None:
            return (False, "SKIP: E_rect not available")
        part = lambda n: self.hbar_sym * self.omega_sym * (n + sp.Rational(1,2))
        sum_1d = part(self.nx_sym) + part(self.ny_sym) + part(self.nz_sym)
        try:
            diff = sp.simplify(self.E_rect_expr - sum_1d)
            if diff == 0:
                return (True, "E_rect is sum of three 1D HO energies")
            else:
                return (False, f"Difference from sum of 1D energies: {diff}")
        except Exception as e:
            return (False, f"Separation check error: {e}")

    def test_modeling_radial_quantum_number_relation(self):
        """Verify E_sph expressed in terms of n_r and ℓ yields E = hbar*omega*(2n_r+ℓ+3/2)."""
        if self.E_sph_expr is None:
            return (False, "SKIP: E_sph not available")
        radial_expr = self.hbar_sym * self.omega_sym * (2*self.nr_sym + self.ell_sym + sp.Rational(3,2))
        subs = {self.k_sym: 2*self.nr_sym + self.ell_sym + 1}
        try:
            transformed = self.E_sph_expr.subs(subs)
            diff = sp.simplify(transformed - radial_expr)
            if diff == 0:
                return (True, "E_sph matches radial formula under k=2n_r+ℓ+1")
            else:
                return (False, f"Difference from radial form: {diff}")
        except Exception as e:
            return (False, f"Radial relation check error: {e}")

    # All remaining modeling tests are skipped because they would require re-deriving numeric constants,
    # which is not the intended pattern.

    def test_modeling_series_termination_condition(self):
        # This test would involve j_max and ρ0; not directly checkable from results alone.
        return (False, "SKIP: Series termination condition not directly checkable with available results")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit numeric perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_rect_energy_increases_with_nx(self):
        """Perturb nx by +1 and verify E_rect increases by exactly hbar*omega."""
        if self.E_rect_expr is None:
            return (False, "SKIP: E_rect not available")
        # Use numeric evaluation with default hbar=1, omega=1
        subs_base = {self.hbar_sym: 1, self.omega_sym: 1, self.nx_sym: 0, self.ny_sym: 0, self.nz_sym: 0}
        subs_perturbed = dict(subs_base)
        subs_perturbed[self.nx_sym] = 1  # perturb nx by +1
        try:
            E_base = self.E_rect_expr.subs(subs_base).evalf()
            E_pert = self.E_rect_expr.subs(subs_perturbed).evalf()
            if E_pert <= E_base:
                return (False, f"E_rect did not increase: base={E_base}, perturbed={E_pert}")
            # Optionally check exact increment
            if not math.isclose(E_pert - E_base, 1.0, rel_tol=1e-9):
                return (False, f"Increment not hbar*omega (1): increment={E_pert-E_base}")
            return (True, f"E_rect increases by 1 (hbar*omega) when nx↑1: {E_base} -> {E_pert}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    def test_monotonicity_rect_energy_increases_with_ny(self):
        """Perturb ny by +1."""
        if self.E_rect_expr is None:
            return (False, "SKIP: E_rect not available")
        subs_base = {self.hbar_sym: 1, self.omega_sym: 1, self.nx_sym: 0, self.ny_sym: 0, self.nz_sym: 0}
        subs_perturbed = dict(subs_base)
        subs_perturbed[self.ny_sym] = 1
        try:
            E_base = self.E_rect_expr.subs(subs_base).evalf()
            E_pert = self.E_rect_expr.subs(subs_perturbed).evalf()
            if E_pert <= E_base:
                return (False, f"E_rect did not increase: base={E_base}, perturbed={E_pert}")
            if not math.isclose(E_pert - E_base, 1.0, rel_tol=1e-9):
                return (False, f"Increment not 1: {E_pert-E_base}")
            return (True, f"E_rect increases by 1 (hbar*omega) when ny↑1: {E_base} -> {E_pert}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    def test_monotonicity_rect_energy_increases_with_nz(self):
        """Perturb nz by +1."""
        if self.E_rect_expr is None:
            return (False, "SKIP: E_rect not available")
        subs_base = {self.hbar_sym: 1, self.omega_sym: 1, self.nx_sym: 0, self.ny_sym: 0, self.nz_sym: 0}
        subs_perturbed = dict(subs_base)
        subs_perturbed[self.nz_sym] = 1
        try:
            E_base = self.E_rect_expr.subs(subs_base).evalf()
            E_pert = self.E_rect_expr.subs(subs_perturbed).evalf()
            if E_pert <= E_base:
                return (False, f"E_rect did not increase: base={E_base}, perturbed={E_pert}")
            if not math.isclose(E_pert - E_base, 1.0, rel_tol=1e-9):
                return (False, f"Increment not 1: {E_pert-E_base}")
            return (True, f"E_rect increases by 1 (hbar*omega) when nz↑1: {E_base} -> {E_pert}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    def test_monotonicity_sph_energy_increases_with_k(self):
        """Perturb k by +1 and verify E_sph increases by hbar*omega."""
        if self.E_sph_expr is None:
            return (False, "SKIP: E_sph not available")
        # Use numeric evaluation: hbar=1, omega=1, baseline with k=0, ell=0, m=0
        subs_base = {self.hbar_sym: 1, self.omega_sym: 1, self.k_sym: 0, self.ell_sym: 0, self.m_sym: 0}
        subs_perturbed = dict(subs_base)
        subs_perturbed[self.k_sym] = 1  # perturb k by +1
        try:
            E_base = self.E_sph_expr.subs(subs_base).evalf()
            E_pert = self.E_sph_expr.subs(subs_perturbed).evalf()
            if E_pert <= E_base:
                return (False, f"E_sph did not increase: base={E_base}, perturbed={E_pert}")
            if not math.isclose(E_pert - E_base, 1.0, rel_tol=1e-9):
                return (False, f"Increment not 1: {E_pert-E_base}")
            return (True, f"E_sph increases by 1 (hbar*omega) when k↑1: {E_base} -> {E_pert}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_not_nan(self):
        """Check that energy expressions evaluate to finite numbers for typical quantum numbers."""
        if self.E_rect_expr is None and self.E_sph_expr is None:
            return (False, "SKIP: no expressions")
        args_rect = {self.hbar_sym: 1, self.omega_sym: 1, self.nx_sym: 2, self.ny_sym: 3, self.nz_sym: 1}
        args_sph = {self.hbar_sym: 1, self.omega_sym: 1, self.k_sym: 6, self.ell_sym: 2, self.m_sym: 0}
        try:
            if self.E_rect_expr is not None:
                val = self.E_rect_expr.subs(args_rect).evalf()
                if math.isnan(val) or math.isinf(val):
                    return (False, f"E_rect evaluates to NaN/Inf: {val}")
            if self.E_sph_expr is not None:
                val = self.E_sph_expr.subs(args_sph).evalf()
                if math.isnan(val) or math.isinf(val):
                    return (False, f"E_sph evaluates to NaN/Inf: {val}")
            return (True, "Energies are finite (no NaN/Inf)")
        except Exception as e:
            return (False, f"Evaluation error: {e}")

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)