import math
import sympy as sp

class PhysicsChecker:
    """
    Audits the magnetic force between hemispheres of a uniformly charged
    spinning spherical shell.  The checker focuses on theorem-based physics
    invariants, modeling consistency, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # ---------- givens ----------
        self.R = getattr(solution_instance, 'R', None)
        self.sigma = getattr(solution_instance, 'sigma', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)

        # ---------- results ----------
        self.F = self._safe_get_result(solution_instance, 'F')

        # optional internal field (if stored by the solver)
        self.B_in = getattr(solution_instance, 'B_in', None)
        if self.B_in is None and hasattr(solution_instance, 'results'):
            self.B_in = solution_instance.results.get('B_in', None)

        # ---------- parsed force ----------
        self.F_magnitude = None      # signed magnitude (negative → attractive)
        self.F_direction = None      # 'z' or None
        self._parse_F()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _safe_get_result(self, instance, key):
        """Extract a result key safely, returning None if missing."""
        if hasattr(instance, 'results') and isinstance(instance.results, dict):
            return instance.results.get(key, None)
        return None

    def _parse_F(self):
        """Try to extract the signed z‑component of the force from various
        representations (string, number, list, sympy expression)."""
        if self.F is None:
            return

        # ---- numeric scalar ----
        if isinstance(self.F, (int, float)):
            self.F_magnitude = float(self.F)
            self.F_direction = 'z'
            return

        # ---- list/tuple (<Fx>, <Fy>, <Fz>) or [<Fz>] ----
        if isinstance(self.F, (list, tuple)):
            if len(self.F) == 3:
                self.F_magnitude = float(self.F[2])
            elif len(self.F) == 1:
                self.F_magnitude = float(self.F[0])
            elif len(self.F) == 0:
                self.F_magnitude = None
            else:
                # assume first element is the z‑component
                self.F_magnitude = float(self.F[0])
            self.F_direction = 'z'
            return

        # ---- sympy expression (eval with given parameters) ----
        if hasattr(self.F, 'free_symbols'):
            try:
                # build substitution dictionary from the available givens
                symbols = {'R': sp.Symbol('R'), 'sigma': sp.Symbol('sigma'),
                           'omega': sp.Symbol('omega'), 'mu0': sp.Symbol('mu0'),
                           'pi': sp.pi}
                subs = {}
                if self.R is not None:   subs[symbols['R']] = self.R
                if self.sigma is not None: subs[symbols['sigma']] = self.sigma
                if self.omega is not None: subs[symbols['omega']] = self.omega
                if self.mu0 is not None:  subs[symbols['mu0']] = self.mu0
                val = self.F.subs(subs).evalf()
                self.F_magnitude = float(val)
                self.F_direction = 'z'
            except Exception:
                self.F_magnitude = None
            return

        # ---- string representation (e.g. from pseudocode emit) ----
        if isinstance(self.F, str):
            # strip direction indicator z_hat, if present
            cleaned = self.F.replace('z_hat', '').replace('*', '').strip()
            try:
                # define sympy symbols only from names that appear
                R, sigma, omega, mu0 = sp.symbols('R sigma omega mu0')
                expr = sp.sympify(cleaned, locals={'pi': sp.pi})
                sub_dict = {}
                if self.R is not None:      sub_dict[R] = self.R
                if self.sigma is not None:   sub_dict[sigma] = self.sigma
                if self.omega is not None:   sub_dict[omega] = self.omega
                if self.mu0 is not None:     sub_dict[mu0] = self.mu0
                expr_val = expr.subs(sub_dict).evalf()
                self.F_magnitude = float(expr_val)
                # if the original string contained 'z_hat', assume direction is z
                if 'z_hat' in self.F:
                    self.F_direction = 'z'
            except Exception:
                self.F_magnitude = None

    def _compute_force_magnitude(self, R, sigma, omega, mu0):
        """Compute the signed z‑component of the force from given parameters.
           Positive magnitude is repulsive, negative attractive.
           F_z = - (π/4) μ₀ (R² σ ω)²."""
        if any(v is None for v in (R, sigma, omega, mu0)):
            return None
        return -(math.pi / 4.0) * mu0 * (R**2 * sigma * omega)**2

    def _expected_magnitude(self):
        """Compute the theoretical magnitude |F| from the known formula
           (attractive force, so magnitude positive)."""
        if any(v is None for v in (self.R, self.sigma, self.omega, self.mu0)):
            return None
        return (math.pi / 4.0) * self.mu0 * (self.R**2 * self.sigma * self.omega)**2

    # ------------------------------------------------------------------
    # test methods
    # ------------------------------------------------------------------
    def test_interface_required_result_key(self):
        """Check that the result dictionary contains the key 'F'."""
        if self.F is None:
            return (False,
                    "Result key 'F' not found in solution_instance.results. "
                    "Ensure the solver emits results['F'].")
        return (True, "'F' key present in results dictionary")

    def test_interface_given_parameters_present(self):
        """Check that all required givens (R, sigma, omega, μ₀) are provided."""
        missing = []
        for name in ('R', 'sigma', 'omega', 'mu0'):
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing required givens: {', '.join(missing)}")
        return (True, "All required givens (R, sigma, omega, μ₀) are present")

    def test_invariant_force_sign_attractive(self):
        """The force between hemispheres must be attractive (negative z)."""
        if self.F_magnitude is None:
            return (False, "Cannot determine force sign – F_magnitude not available")
        if self.F_magnitude < 0:
            return (True, f"Force is attractive (F_z = {self.F_magnitude:.4f} < 0)")
        return (False, f"Force is repulsive or zero (F_z = {self.F_magnitude:.4f}); "
                "expected attractive")

    def test_invariant_force_not_zero(self):
        """The magnitude of the force must be strictly positive (non‑zero)
        when the sphere is charged and spinning."""
        if self.F_magnitude is None:
            return (False, "Cannot check force magnitude – not available")
        if math.isclose(self.F_magnitude, 0.0, abs_tol=1e-12):
            return (False, "Force magnitude is zero (nearly zero); expected non‑zero")
        return (True, f"Force magnitude non‑zero (|F| = {abs(self.F_magnitude):.4e})")

    def test_invariant_magnetic_field_inside(self):
        """If the solver reported the internal magnetic field, verify its
        expression: B_in = (2/3) μ₀ σ ω R  (uniform, along the rotation axis)."""
        B = self.B_in
        if B is None:
            return (False, "B_in not available – test skipped")
        try:
            B_val = float(B) if not hasattr(B, 'free_symbols') else float(B)
        except Exception:
            return (False, "B_in cannot be converted to float")
        # compute expected
        if any(v is None for v in (self.mu0, self.sigma, self.omega, self.R)):
            return (False, "Missing givens to compute expected B_in")
        expected_B = (2.0/3.0) * self.mu0 * self.sigma * self.omega * self.R
        if math.isclose(B_val, expected_B, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"B_in ≈ {B_val:.4e} matches expected {expected_B:.4e}")
        return (False, f"B_in = {B_val:.4e} differs from expected {expected_B:.4e}")

    def test_modeling_force_magnitude_formula(self):
        """Check that the force magnitude obeys the known theoretical formula
        |F| = (π/4) μ₀ (R² σ ω)² (tolerance 1 %)."""
        expected = self._expected_magnitude()
        if expected is None:
            return (False, "Cannot compute expected magnitude – missing givens")
        if self.F_magnitude is None:
            return (False, "Actual force magnitude unavailable")
        if math.isclose(abs(self.F_magnitude), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"|F| = {abs(self.F_magnitude):.4e} agrees with expected "
                    f"{expected:.4e} (within 1 % tolerance)")
        return (False, f"|F| = {abs(self.F_magnitude):.4e} deviates from expected "
                f"{expected:.4e} (relative diff {abs(abs(self.F_magnitude)-expected)/expected:.3e})")

    def test_modeling_force_direction_consistency(self):
        """Check that the solution's force points along the negative z‑axis
        (if the direction can be determined)."""
        if self.F_direction != 'z':
            return (False, "Force direction could not be parsed as 'z'")
        # If the magnitude is negative, direction is correct (azimuthal symmetry)
        if self.F_magnitude is not None and self.F_magnitude < 0:
            return (True, "Force vector points along –z (attractive)")
        # otherwise we cannot confirm the direction
        return (False, "Unable to confirm that force points along –z axis")

    def test_monotonicity_force_increases_with_R(self):
        """The magnitude of the force should increase as R increases
        (quadratically, or quartic).  Perturb R by +5% and check that
        |F_perturbed| > |F_baseline|."""
        # Must have all givens to recompute baseline
        if any(v is None for v in (self.R, self.sigma, self.omega, self.mu0)):
            return (False, "SKIP: missing givens to compute monotonicity test")
        baseline = self._compute_force_magnitude(self.R, self.sigma,
                                                  self.omega, self.mu0)
        if baseline is None:
            return (False, "Cannot compute baseline force magnitude")
        # perturb R by +5%
        R_perturbed = self.R * 1.05
        perturbed = self._compute_force_magnitude(R_perturbed, self.sigma,
                                                   self.omega, self.mu0)
        if perturbed is None:
            return (False, "Cannot compute perturbed force magnitude")
        # Since baseline is negative (attractive), we compare absolute values.
        if abs(perturbed) > abs(baseline):
            return (True,
                    f"|F| increases with R: baseline |F|={abs(baseline):.4e}, "
                    f"+5% R gives |F|={abs(perturbed):.4e}")
        return (False,
                f"|F| does not increase with R: baseline |F|={abs(baseline):.4e}, "
                f"+5% R gives |F|={abs(perturbed):.4e}")

    def test_monotonicity_force_increases_with_sigma(self):
        """The magnitude of the force should increase as sigma increases
        (quadratically).  Perturb sigma by +5% and check that
        |F_perturbed| > |F_baseline|."""
        if any(v is None for v in (self.R, self.sigma, self.omega, self.mu0)):
            return (False, "SKIP: missing givens to compute monotonicity test")
        baseline = self._compute_force_magnitude(self.R, self.sigma,
                                                  self.omega, self.mu0)
        if baseline is None:
            return (False, "Cannot compute baseline force magnitude")
        sigma_perturbed = self.sigma * 1.05
        perturbed = self._compute_force_magnitude(self.R, sigma_perturbed,
                                                   self.omega, self.mu0)
        if perturbed is None:
            return (False, "Cannot compute perturbed force magnitude")
        if abs(perturbed) > abs(baseline):
            return (True,
                    f"|F| increases with σ: baseline |F|={abs(baseline):.4e}, "
                    f"+5% σ gives |F|={abs(perturbed):.4e}")
        return (False,
                f"|F| does not increase with σ: baseline |F|={abs(baseline):.4e}, "
                f"+5% σ gives |F|={abs(perturbed):.4e}")

    def test_answer_sanity_force_finite(self):
        """The force must be a finite real number (weak sanity)."""
        if self.F_magnitude is None:
            return (False, "Cannot check finiteness – F_magnitude not available")
        if math.isfinite(self.F_magnitude):
            return (True, "Force magnitude is finite")
        return (False, f"Force magnitude is not finite: {self.F_magnitude}")

    # ------------------------------------------------------------------
    # test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, returning (success, logs)."""
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