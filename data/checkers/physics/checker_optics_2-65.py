import math
import sympy
import traceback

class PhysicsChecker:
    """
    Audits a solution for the problem:
    'Estimate the distance at which car headlights (1.3 m apart) can just be resolved
     by the human eye (pupil diameter 4 mm, wavelength 5500 Å).'
    """
    def __init__(self, solution_instance):
        # === Extract input givens safely ===
        # Attempt to read from solution_instance.givens or directly as attributes
        self.d = None
        self.D = None
        self.lam = None
        self.L = None

        # Try to read from a 'givens' dict or attribute
        if hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            if isinstance(givens, dict):
                self.d = givens.get('d', None)
                self.D = givens.get('D', None)
                self.lam = givens.get('lambda', None)
            else:
                # fallback: treat givens as object
                self.d = getattr(givens, 'd', None)
                self.D = getattr(givens, 'D', None)
                self.lam = getattr(givens, 'lambda', None)
        else:
            # direct attributes on solution_instance
            self.d = getattr(solution_instance, 'd', None)
            self.D = getattr(solution_instance, 'D', None)
            self.lam = getattr(solution_instance, 'lambda', None)

        # Also try common alternatives for wavelength
        if self.lam is None:
            self.lam = getattr(solution_instance, 'lambda_', None)
        if self.lam is None:
            self.lam = getattr(solution_instance, 'wavelength', None)

        # === Extract results safely ===
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.L = solution_instance.results.get('L', None)
        else:
            self.L = getattr(solution_instance, 'L', None)

        # Store the original instance for potential further queries
        self.solution_instance = solution_instance

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_key_L_in_results(self):
        """Check that the distance L is present in results."""
        if self.L is None:
            return (False, "SKIP: results missing key 'L'")
        return (True, "Key 'L' present in results")

    def test_interface_givens_present(self):
        """Check that all required givens are present."""
        missing = []
        if self.d is None:
            missing.append('d')
        if self.D is None:
            missing.append('D')
        if self.lam is None:
            missing.append('lambda')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ----------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------
    def test_invariant_rayleigh_criterion(self):
        """
        Verify that L satisfies the Rayleigh resolution equation:
        d / L = 1.22 * lambda / D
        """
        if any(v is None for v in [self.d, self.D, self.lam, self.L]):
            return (False, "SKIP: missing data for Rayleigh invariant")
        # Compute angular resolution from Rayleigh
        theta_R = 1.22 * self.lam / self.D
        # Compute angular separation from headlight geometry
        theta_sep = self.d / self.L
        # Must be approximately equal (just resolution)
        if math.isclose(theta_sep, theta_R, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Rayleigh criterion satisfied: theta_sep={theta_sep:.4e}, theta_R={theta_R:.4e}")
        else:
            return (False, f"Rayleigh not satisfied: theta_sep={theta_sep:.4e}, theta_R={theta_R:.4e}")

    def test_invariant_all_quantities_positive(self):
        """Check that all physical quantities are positive."""
        for name, val in [('d', self.d), ('D', self.D), ('lam', self.lam), ('L', self.L)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if val <= 0:
                return (False, f"{name} = {val} is not positive")
        return (True, "All quantities positive")

    def test_invariant_small_angle_valid(self):
        """
        The small angle approximation tan(theta) ≈ theta is valid when theta << 1 rad.
        Check that the angular separation d/L is small (e.g., < 0.1 rad).
        """
        if any(v is None for v in [self.d, self.L]):
            return (False, "SKIP: missing d or L for small angle check")
        theta = self.d / self.L
        if theta < 0.1:
            return (True, f"Angle theta = {theta:.4e} rad is small (<0.1)")
        else:
            return (False, f"Angle {theta:.4e} rad is not small; small-angle approximation may be invalid")

    # ----------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------
    def test_modeling_derived_L_forward_consistent(self):
        """
        Check that L matches the direct formula L = d * D / (1.22 * lam).
        This is a modeling consistency test: re-derive L from givens.
        """
        if any(v is None for v in [self.d, self.D, self.lam, self.L]):
            return (False, "SKIP: missing data for forward consistency")
        L_calc = self.d * self.D / (1.22 * self.lam)
        if math.isclose(self.L, L_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L computed from formula: L_calc={L_calc:.2f} m, provided L={self.L:.2f} m")
        else:
            return (False, f"L mismatch: computed {L_calc:.2f} m, provided {self.L:.2f} m")

    def test_modeling_explanation_uses_rayleigh(self):
        """
        Check that the explanation steps reference the Rayleigh criterion.
        This is a weak test: we just ensure the explanation text contains certain keywords.
        """
        explanation = getattr(self.solution_instance, 'explanation_steps', '')
        if not explanation:
            return (False, "SKIP: no explanation_steps attribute")
        keywords = ['Rayleigh', '1.22', 'resolution']
        found = [k for k in keywords if k.lower() in explanation.lower()]
        if found:
            return (True, f"Explanation contains Rayleigh keywords: {found}")
        else:
            return (False, "Explanation does not mention Rayleigh criterion explicitly")

    # ----------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------
    def test_monotonicity_L_increases_with_d(self):
        """
        Perturb d by +10% and check that L increases.
        """
        if any(v is None for v in [self.d, self.D, self.lam, self.L]):
            return (False, "SKIP: missing data for monotonicity test")
        d_perturbed = self.d * 1.10
        L_new = d_perturbed * self.D / (1.22 * self.lam)
        if L_new > self.L:
            return (True, f"L increases with d: L_orig={self.L:.2f}, L_perturbed={L_new:.2f}")
        else:
            return (False, f"L does not increase with d: L_orig={self.L:.2f}, L_perturbed={L_new:.2f}")

    def test_monotonicity_L_decreases_with_lambda(self):
        """
        Perturb lambda by +10% and check that L decreases.
        """
        if any(v is None for v in [self.d, self.D, self.lam, self.L]):
            return (False, "SKIP: missing data for monotonicity test")
        lam_perturbed = self.lam * 1.10
        L_new = self.d * self.D / (1.22 * lam_perturbed)
        if L_new < self.L:
            return (True, f"L decreases with lambda: L_orig={self.L:.2f}, L_perturbed={L_new:.2f}")
        else:
            return (False, f"L does not decrease with lambda: L_orig={self.L:.2f}, L_perturbed={L_new:.2f}")

    def test_monotonicity_L_increases_with_D(self):
        """
        Perturb D by +10% and check that L increases.
        """
        if any(v is None for v in [self.d, self.D, self.lam, self.L]):
            return (False, "SKIP: missing data for monotonicity test")
        D_perturbed = self.D * 1.10
        L_new = self.d * D_perturbed / (1.22 * self.lam)
        if L_new > self.L:
            return (True, f"L increases with D: L_orig={self.L:.2f}, L_perturbed={L_new:.2f}")
        else:
            return (False, f"L does not increase with D: L_orig={self.L:.2f}, L_perturbed={L_new:.2f}")

    # ----------------------------------------------------------------
    # Answer sanity (weak, at most one)
    # ----------------------------------------------------------------
    def test_answer_sanity_L_range(self):
        """
        Weak sanity: L should be between 1 km and 100 km.
        (Expected ~7.75 km, but we only do a broad range check.)
        """
        if self.L is None:
            return (False, "SKIP: L not available")
        if self.L <= 0:
            return (False, f"L = {self.L} m is not positive")
        if 1000 <= self.L <= 100000:
            return (True, f"L = {self.L:.0f} m is within plausible range [1 km, 100 km]")
        else:
            return (False, f"L = {self.L:.0f} m is outside plausible range")

    # ----------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------
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