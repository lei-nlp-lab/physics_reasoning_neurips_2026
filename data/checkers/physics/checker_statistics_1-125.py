import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the charged soap bubble problem.
    The expected relation is Δp = φ²/(4π r²) (in CGS units, consistent with the provided pseudocode).
    """

    def __init__(self, solution_instance):
        # Safely extract inputs and results, defaulting to None
        self.r = getattr(solution_instance, 'r', None)
        if self.r is None:
            # Also check a 'givens' dictionary if present
            givens = getattr(solution_instance, 'givens', None)
            if givens and isinstance(givens, dict):
                self.r = givens.get('r', None)

        self.phi = getattr(solution_instance, 'phi', None)
        if self.phi is None:
            givens = getattr(solution_instance, 'givens', None)
            if givens and isinstance(givens, dict):
                self.phi = givens.get('phi', None)

        # Result from pseudocode [emit]
        self.results = getattr(solution_instance, 'results', {})
        self.delta_p = self.results.get('delta_p', None)

        # Store full solution instance for potential other checks
        self.solution_instance = solution_instance

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_result_key(self):
        """Check that 'delta_p' key exists in results."""
        if not isinstance(self.results, dict):
            return False, "results is not a dict"
        if 'delta_p' not in self.results:
            return False, "Key 'delta_p' missing in results"
        return True, "delta_p key present"

    def test_interface_required_inputs(self):
        """Check that r and phi are provided."""
        if self.r is None:
            return False, "Input 'r' missing (None)"
        if self.phi is None:
            return False, "Input 'phi' missing (None)"
        return True, "r and phi are available"

    # ----------------------------------------------------------------------
    # Positivity / finite checks (theorem-based)
    # ----------------------------------------------------------------------
    def test_invariant_r_positive(self):
        """r must be positive (physical radius)."""
        if self.r is None:
            return False, "SKIP: r not available"
        if self.r <= 0:
            return False, f"r = {self.r} is not positive"
        return True, "r > 0"

    def test_invariant_phi_positive(self):
        """Potential φ must be positive (non-zero net charge)."""
        if self.phi is None:
            return False, "SKIP: φ not available"
        if self.phi <= 0:
            return False, f"φ = {self.phi} is not positive"
        return True, "φ > 0"

    def test_invariant_delta_p_positive(self):
        """Excess pressure Δp must be positive (outward electrostatic pressure)."""
        if self.delta_p is None:
            return False, "SKIP: Δp not available"
        if self.delta_p <= 0:
            return False, f"Δp = {self.delta_p} is not positive"
        return True, "Δp > 0"

    # ----------------------------------------------------------------------
    # Theorem-based invariant: equilibrium relation Δp = φ²/(4π r²)
    # ----------------------------------------------------------------------
    def test_invariant_electrostatic_equilibrium(self):
        """
        From mechanical equilibrium and the relation φ = 4π r σ,
        the electrostatic pressure is p_e = φ²/(4π r²).
        Hence we must have Δp ≈ p_e.
        """
        if None in (self.r, self.phi, self.delta_p):
            return False, "SKIP: missing r, φ, or Δp"
        if self.r <= 0 or self.phi <= 0:
            return False, "SKIP: non-positive r or φ"
        # Theoretical value from CGS (consistent with pseudocode)
        p_e_expected = (self.phi ** 2) / (4 * math.pi * self.r ** 2)
        if not math.isclose(self.delta_p, p_e_expected, rel_tol=1e-2, abs_tol=1e-15):
            return False, (
                f"Δp = {self.delta_p} does not match φ²/(4π r²) = {p_e_expected}; "
                f"relative diff = {abs(self.delta_p - p_e_expected)/p_e_expected:.4f}"
            )
        return True, "Δp ≈ φ²/(4π r²) within 1% tolerance"

    # ----------------------------------------------------------------------
    # Modeling/trace quality tests (at least two)
    # ----------------------------------------------------------------------
    def test_modeling_surface_charge_density_consistency(self):
        """
        From the given φ and r, compute σ = φ/(4π r) (CGS) and then
        electrostatic pressure p_e = 4π σ². This should equal Δp.
        This verifies the intermediate modeling step in the pseudocode.
        """
        if None in (self.r, self.phi, self.delta_p):
            return False, "SKIP: missing r, φ, or Δp"
        if self.r <= 0 or self.phi <= 0:
            return False, "SKIP: non-positive r or φ"
        sigma = self.phi / (4 * math.pi * self.r)   # φ = 4π r σ
        p_e = 4 * math.pi * sigma ** 2               # CGS electrostatic pressure
        if not math.isclose(self.delta_p, p_e, rel_tol=1e-2, abs_tol=1e-15):
            return False, (
                f"Intermediate check fails: p_e = 4πσ² = {p_e} but Δp = {self.delta_p}; "
                f"σ = {sigma}"
            )
        return True, "Intermediate σ and p_e are consistent with Δp"

    def test_modeling_sigma_from_delta_p_consistency(self):
        """
        Compute σ from the output Δp via σ = sqrt(Δp/(4π)) (since p_e = Δp).
        Then recompute φ from σ and r: φ_reconstructed = 4π r σ.
        The reconstructed φ should match the given φ.
        This checks internal consistency of the solution.
        """
        if None in (self.r, self.phi, self.delta_p):
            return False, "SKIP: missing r, φ, or Δp"
        if self.r <= 0 or self.phi <= 0 or self.delta_p <= 0:
            return False, "SKIP: non-positive input or result"
        # Compute σ from Δp (CGS: p_e = 4πσ²)
        sigma_from_delta = math.sqrt(self.delta_p / (4 * math.pi))
        # Reconstruct φ
        phi_reconstructed = 4 * math.pi * self.r * sigma_from_delta
        if not math.isclose(phi_reconstructed, self.phi, rel_tol=1e-2, abs_tol=1e-15):
            return False, (
                f"Reconstructed φ = {phi_reconstructed} does not match given φ = {self.phi}; "
                f"relative diff = {abs(phi_reconstructed - self.phi)/self.phi:.4f}"
            )
        return True, "Reconstructed φ from σ and r matches provided φ"

    # ----------------------------------------------------------------------
    # Monotonicity test
    # ----------------------------------------------------------------------
    def test_monotonicity_delta_p_increases_with_phi(self):
        """
        From Δp ∝ φ², increasing φ should increase Δp (with r fixed).
        Perturb φ by +5% and verify that the new Δp (computed from the relation) 
        is larger than the baseline Δp.
        """
        if None in (self.r, self.phi, self.delta_p):
            return False, "SKIP: missing r, φ, or Δp"
        if self.r <= 0 or self.phi <= 0:
            return False, "SKIP: non-positive r or φ"

        baseline = self.delta_p

        # Perturb φ upward by 5%
        phi_perturbed = self.phi * 1.05
        # Compute expected Δp from the known relation (re-derived in the checker)
        delta_p_perturbed = (phi_perturbed ** 2) / (4 * math.pi * self.r ** 2)

        # The perturbed value should be larger
        if delta_p_perturbed <= baseline:
            return False, (
                f"Monotonicity fails: with φ increased from {self.phi} to {phi_perturbed}, "
                f"Δp went from {baseline} to {delta_p_perturbed}, expected increase."
            )
        return True, (
            f"Δp increases with φ: baseline {baseline:.6e}, "
            f"perturbed φ = {phi_perturbed:.6e}, new Δp = {delta_p_perturbed:.6e}"
        )

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, secondary)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_not_nan(self):
        """Check that Δp (and r, φ) are finite and not NaN."""
        if self.delta_p is None:
            return False, "SKIP: Δp not available"
        if math.isnan(self.delta_p) or math.isinf(self.delta_p):
            return False, f"Δp is invalid: {self.delta_p}"
        # Also check that we can compute the relation without error
        if self.r is not None and self.phi is not None:
            if self.r <= 0 or self.phi <= 0:
                return False, "Non-positive r or φ"
            try:
                _ = (self.phi ** 2) / (4 * math.pi * self.r ** 2)
            except Exception as e:
                return False, f"Cannot evaluate reference expression: {e}"
        return True, "Δp is finite and the expression is computable"

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
                # Include traceback for debugging
                tb = traceback.format_exc()
                passed, msg = False, f"CRASHED: {e}\n{tb}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs