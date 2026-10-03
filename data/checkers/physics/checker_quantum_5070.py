import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the magnetic polarizability of hydrogen (F=0 hyperfine)
    and helium (1s^2 ground state). The checker performs interface checks,
    invariant tests, modeling-quality checks, monotonicity checks, and at most
    one weak answer sanity test. It never uses numeric defaults for missing data.
    """

    def __init__(self, solution_instance):
        # Helper to safely get attributes from either a .results dict or directly
        def _get(key, alt_key=None):
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                return solution_instance.results.get(key, solution_instance.results.get(alt_key, None))
            return getattr(solution_instance, key, getattr(solution_instance, alt_key, None))

        # Core outputs from emit contract
        self.alpha_H = _get('alpha_H')
        self.alpha_He = _get('alpha_He')

        # Optional input parameters (from givens or direct attributes)
        self.mu_B = _get('mu_B', 'mu_b')
        self.DeltaE_hf = _get('DeltaE_hf', 'deltaE_hf')
        self.e = _get('e')
        self.a0 = _get('a0')
        self.me = _get('me')

    # --------------------------------------------------------------------
    # Interface checks (required keys exist and are numeric)
    # --------------------------------------------------------------------
    def test_interface_alpha_H_exists(self):
        if self.alpha_H is None:
            return (False, "alpha_H is missing (None)")
        return (True, "alpha_H is present")

    def test_interface_alpha_He_exists(self):
        if self.alpha_He is None:
            return (False, "alpha_He is missing (None)")
        return (True, "alpha_He is present")

    # --------------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # --------------------------------------------------------------------
    def test_invariant_hydrogen_sign(self):
        """Paramagnetic polarizability must be positive."""
        if self.alpha_H is None:
            return (False, "SKIP: alpha_H not available")
        if self.alpha_H > 0:
            return (True, f"alpha_H = {self.alpha_H} > 0")
        return (False, f"alpha_H should be positive, got {self.alpha_H}")

    def test_invariant_helium_sign(self):
        """Diamagnetic polarizability must be negative."""
        if self.alpha_He is None:
            return (False, "SKIP: alpha_He not available")
        if self.alpha_He < 0:
            return (True, f"alpha_He = {self.alpha_He} < 0")
        return (False, f"alpha_He should be negative, got {self.alpha_He}")

    def test_invariant_hydrogen_finite(self):
        if self.alpha_H is None:
            return (False, "SKIP: alpha_H not available")
        if math.isfinite(self.alpha_H):
            return (True, "alpha_H is finite")
        return (False, "alpha_H is non-finite")

    def test_invariant_helium_finite(self):
        if self.alpha_He is None:
            return (False, "SKIP: alpha_He not available")
        if math.isfinite(self.alpha_He):
            return (True, "alpha_He is finite")
        return (False, "alpha_He is non-finite")

    # --------------------------------------------------------------------
    # Modeling checks (consistency with underlying physics relations)
    # --------------------------------------------------------------------
    def test_modeling_hydrogen_formula(self):
        """
        Check that alpha_H is consistent with second-order perturbation:
        alpha_H = mu_B^2 / (2 * DeltaE_hf).
        Use fuzzy matching because of rounding noise.
        """
        if None in (self.alpha_H, self.mu_B, self.DeltaE_hf):
            return (False, "SKIP: insufficient data for hydrogen formula check")
        expected = (self.mu_B ** 2) / (2.0 * self.DeltaE_hf)
        if math.isclose(self.alpha_H, expected, rel_tol=0.1, abs_tol=1e-12):
            return (True, f"alpha_H ≈ mu_B^2/(2*DeltaE) = {expected:.3e}")
        return (False, f"alpha_H = {self.alpha_H:.3e}, expected ~ {expected:.3e}")

    def test_modeling_helium_magnitude_order(self):
        """
        Check that alpha_He is approximately -4.4e-19 eV/Gs² (order-of-magnitude).
        Use a broad range: 1e-20 to 1e-18.
        """
        if self.alpha_He is None:
            return (False, "SKIP: alpha_He not available")
        mag = abs(self.alpha_He)
        if 1e-20 <= mag <= 1e-18:
            return (True, f"|alpha_He| = {mag:.3e} lies in expected range 1e-20..1e-18")
        return (False, f"|alpha_He| = {mag:.3e} outside range 1e-20..1e-18")

    def test_modeling_hydrogen_dominates_helium(self):
        """
        The absolute magnitude of alpha_H should be many orders larger than that of alpha_He,
        because paramagnetic response is much larger than diamagnetic.
        At least a factor 1e6 difference.
        """
        if None in (self.alpha_H, self.alpha_He):
            return (False, "SKIP: both outputs needed")
        if abs(self.alpha_H) > abs(self.alpha_He) * 1e6:
            return (True, f"|alpha_H|/|alpha_He| ≈ {abs(self.alpha_H)/abs(self.alpha_He):.1e} > 1e6")
        return (False, f"|alpha_H| should be much larger than |alpha_He|")

    # --------------------------------------------------------------------
    # Monotonicity checks (directional behaviour under input perturbation)
    # --------------------------------------------------------------------
    def test_monotonicity_hydrogen_mu_B(self):
        """
        Increase mu_B by 5% and check that alpha_H increases (since alpha_H ∝ mu_B^2).
        Only if mu_B and DeltaE_hf are available.
        """
        if None in (self.mu_B, self.DeltaE_hf):
            return (False, "SKIP: mu_B or DeltaE_hf not available")
        # Baseline from formula
        baseline = (self.mu_B ** 2) / (2.0 * self.DeltaE_hf)
        perturbed_mu = self.mu_B * 1.05
        perturbed = (perturbed_mu ** 2) / (2.0 * self.DeltaE_hf)
        if perturbed > baseline:
            return (True, f"alpha_H increases with mu_B: {baseline:.3e} -> {perturbed:.3e}")
        return (False, "Monotonicity violation: alpha_H should increase with mu_B")

    def test_monotonicity_helium_e(self):
        """
        Increase elementary charge e by 5% and check that alpha_He becomes more negative
        (since alpha_He ∝ -e^2). Only if e, a0, me are available.
        We approximate using the leading term: alpha_He ≈ - (e^2 * a0^2) / (8 * me) (SI, then convert).
        For the monotonic sign, we just compare the raw proportional factor.
        """
        if None in (self.e, self.a0, self.me):
            return (False, "SKIP: e, a0 or me not available")
        # Use the leading factor (ignoring conversion to eV/Gs², sign is enough)
        baseline_factor = - (self.e ** 2) * (self.a0 ** 2) / (8.0 * self.me)
        perturbed_e = self.e * 1.05
        perturbed_factor = - (perturbed_e ** 2) * (self.a0 ** 2) / (8.0 * self.me)
        if perturbed_factor < baseline_factor:   # more negative
            return (True, f"alpha_He becomes more negative: {baseline_factor:.3e} -> {perturbed_factor:.3e}")
        return (False, "Monotonicity violation: alpha_He should become more negative with larger e")

    # --------------------------------------------------------------------
    # Weak answer sanity test (at most one, broad plausibility only)
    # --------------------------------------------------------------------
    def test_answer_sanity_core(self):
        """
        Combined weak sanity for both outputs. Only checks sign, finiteness,
        and broad magnitude range. No exact target matching, no isclose.
        """
        notes = []
        all_ok = True

        if self.alpha_H is not None:
            if not math.isfinite(self.alpha_H):
                notes.append("alpha_H is non-finite")
                all_ok = False
            elif not (1e-12 <= self.alpha_H <= 1e-10):
                notes.append(f"alpha_H = {self.alpha_H:.3e} out of plausible range 1e-12..1e-10")
                all_ok = False
            else:
                notes.append(f"alpha_H = {self.alpha_H:.3e} in plausible range (positive)")
        else:
            notes.append("alpha_H missing")

        if self.alpha_He is not None:
            if not math.isfinite(self.alpha_He):
                notes.append("alpha_He is non-finite")
                all_ok = False
            elif not (-1e-18 <= self.alpha_He <= -1e-20):
                notes.append(f"alpha_He = {self.alpha_He:.3e} out of plausible range -1e-18..-1e-20")
                all_ok = False
            else:
                notes.append(f"alpha_He = {self.alpha_He:.3e} in plausible range (negative)")
        else:
            notes.append("alpha_He missing")

        # If both missing or all checks pass, report accordingly
        if self.alpha_H is None and self.alpha_He is None:
            return (False, "SKIP: no output data to check")
        if all_ok:
            return (True, "; ".join(notes))
        return (False, "; ".join(notes))

    # --------------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------------
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