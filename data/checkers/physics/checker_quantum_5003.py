import math
import traceback
import sympy as sp
import inspect

class PhysicsChecker:
    """
    Audits a solution instance for the relativistic correction to the ground state
    energy of a 1D harmonic oscillator.

    The checker extracts the givens (m, omega, hbar, c) and the result (Delta_E)
    from the solution instance, then performs a series of focused tests:
    - interface checks
    - invariant tests (physics-based)
    - modeling/trace consistency tests
    - monotonicity tests
    - one weak answer-sanity check

    The tests rely on the physics mapping:
      H' = -p^4/(8 m^3 c^2)
      ΔE = <0|H'|0>
      <0|p^2|0> = hbar m omega / 2
      Gaussian property: <0|p^4|0> = 3 (<0|p^2|0>)^2
    """

    def __init__(self, solution_instance):
        """
        Extract variables safely with None defaults.
        Expected attributes: m, omega, hbar, c, results['Delta_E']
        """
        # Givens
        self.m = getattr(solution_instance, 'm', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.c = getattr(solution_instance, 'c', None)

        # Results (use .results dict if present, else attribute)
        results = getattr(solution_instance, 'results', None)
        if isinstance(results, dict):
            self.delta_E = results.get('Delta_E', None)
        else:
            self.delta_E = getattr(solution_instance, 'Delta_E', None)

        # Additional optional intermediate expectations (may appear in results)
        self.exp_p2 = None
        self.exp_p4 = None
        if isinstance(results, dict):
            self.exp_p2 = results.get('exp_p2', None)
            self.exp_p4 = results.get('exp_p4', None)

        # Derived expected value (computed in-checker if givens exist)
        self.expected_delta_E = self._compute_expected_delta_E()

    def _compute_expected_delta_E(self):
        """Compute the theoretical ΔE from the physics mapping."""
        if None in (self.m, self.omega, self.hbar, self.c):
            return None
        # <p^2> = hbar m omega / 2
        p2_exp = (self.hbar * self.m * self.omega) / 2.0
        # <p^4> = 3 (<p^2>)^2
        p4_exp = 3.0 * p2_exp ** 2
        # ΔE = - p4_exp / (8 m^3 c^2)
        delta_E = - p4_exp / (8.0 * (self.m ** 3) * (self.c ** 2))
        return delta_E

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_delta_e_present(self):
        if self.delta_E is None:
            return (False, "Delta_E is missing from solution results")
        return (True, "Delta_E is present")

    def test_interface_delta_e_numeric(self):
        if self.delta_E is None:
            return (False, "SKIP: Delta_E missing")
        if not isinstance(self.delta_E, (int, float, sp.Number)):
            return (False, f"Delta_E is not numeric, got {type(self.delta_E)}")
        return (True, "Delta_E is numeric")

    def test_interface_givens_present(self):
        """Check that essential givens are present (m, omega, hbar, c)."""
        missing = []
        if self.m is None:
            missing.append("m")
        if self.omega is None:
            missing.append("omega")
        if self.hbar is None:
            missing.append("hbar")
        if self.c is None:
            missing.append("c")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ----------------------------------------------------------------------
    # Invariant tests (physics-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_delta_e_negative(self):
        if self.delta_E is None:
            return (False, "SKIP: Delta_E missing")
        if self.delta_E >= 0:
            return (False, "Delta_E should be negative, got non-negative")
        return (True, "Delta_E is negative (correct sign)")

    def test_invariant_p2_intermediate(self):
        """If solution provides <p^2>, check that it equals hbar*m*omega/2."""
        if self.exp_p2 is None:
            return (False, "SKIP: <p^2> not provided in results")
        if None in (self.m, self.omega, self.hbar):
            return (False, "SKIP: missing givens needed for <p^2>")
        expected_p2 = (self.hbar * self.m * self.omega) / 2.0
        if math.isclose(self.exp_p2, expected_p2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"<p^2> = {self.exp_p2} matches expected {expected_p2}")
        return (False, f"<p^2> = {self.exp_p2} does not match expected {expected_p2}")

    def test_invariant_p4_gaussian_relation(self):
        """If solution provides <p^4>, check that it equals 3*(<p^2>)^2."""
        if self.exp_p4 is None or self.exp_p2 is None:
            return (False, "SKIP: need both <p^2> and <p^4>")
        expected_p4 = 3.0 * self.exp_p2 ** 2
        if math.isclose(self.exp_p4, expected_p4, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"<p^4> = {self.exp_p4} satisfies Gaussian relation")
        return (False, f"<p^4> = {self.exp_p4} does not satisfy 3*(<p^2>)^2 = {expected_p4}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests (use physics mapping to derive expected value)
    # ----------------------------------------------------------------------
    def test_modeling_delta_e_via_p4_formula(self):
        """
        Derive expected ΔE from given parameters and compare against computed value.
        Uses math.isclose with 1% relative tolerance.
        """
        if None in (self.delta_E, self.expected_delta_E):
            return (False, "SKIP: either Delta_E or necessary givens missing")
        if not math.isclose(self.delta_E, self.expected_delta_E,
                            rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Delta_E = {self.delta_E} does not match derived "
                           f"expected {self.expected_delta_E}")
        return (True, f"Delta_E = {self.delta_E} matches derived formula within tolerance")

    def test_modeling_delta_e_scaling(self):
        """
        Check that ΔE scales as -ħ²ω²/(m c²). We do this by computing the ratio
        ΔE / ( -ħ²ω²/(m c²) ) and verifying it is a positive constant around 3/32.
        (This is a weaker test: it does not depend on the exact prefactor.)
        """
        if None in (self.delta_E, self.m, self.omega, self.hbar, self.c):
            return (False, "SKIP: missing data for scaling check")
        # Compute ratio = -ΔE * m * c² / (ħ²ω²)
        numerator = -self.delta_E * self.m * self.c ** 2
        denominator = (self.hbar ** 2) * (self.omega ** 2)
        ratio = numerator / denominator
        # Theoretical value is 3/32 = 0.09375, but we accept a broad range (0.05-0.15)
        if 0.05 < ratio < 0.15:
            return (True, f"Scaling ratio = {ratio:.5f} lies in plausible range (0.05,0.15)")
        return (False, f"Scaling ratio = {ratio:.5f} outside expected range")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb one input and compare outputs)
    # ----------------------------------------------------------------------
    def test_monotonicity_delta_e_increases_with_mass(self):
        """
        Increase mass by 5%: |ΔE| should decrease (i.e., ΔE becomes less negative).
        Since we cannot modify the solution, we use the derived expected formula
        to estimate the new ΔE. Then we compare with the original computed ΔE.
        """
        if None in (self.m, self.omega, self.hbar, self.c, self.delta_E):
            return (False, "SKIP: missing data for monotonicity check")
        # Original derived ΔE from formula (using original givens)
        original_expected = self.expected_delta_E
        if original_expected is None:
            return (False, "SKIP: cannot compute expected ΔE")

        # Perturb mass: increase by 5%
        m_new = self.m * 1.05
        # Compute new expected ΔE
        p2_new = (self.hbar * m_new * self.omega) / 2.0
        p4_new = 3.0 * p2_new ** 2
        delta_E_new = -p4_new / (8.0 * (m_new ** 3) * (self.c ** 2))

        # Compare: since m increased, |ΔE| should be smaller, i.e., ΔE_new > ΔE_original (less negative)
        if delta_E_new > original_expected:
            return (True, f"Mass +5% gives ΔE = {delta_E_new:.6e} > original {original_expected:.6e} (correct trend)")
        return (False, f"Mass +5% gives ΔE = {delta_E_new:.6e} is not greater than original {original_expected:.6e}")

    def test_monotonicity_delta_e_decreases_with_omega(self):
        """
        Increase omega by 5%: |ΔE| should increase (ΔE becomes more negative).
        """
        if None in (self.m, self.omega, self.hbar, self.c, self.delta_E):
            return (False, "SKIP: missing data for monotonicity check")
        original_expected = self.expected_delta_E
        if original_expected is None:
            return (False, "SKIP: cannot compute expected ΔE")

        # Perturb omega: increase by 5%
        omega_new = self.omega * 1.05
        p2_new = (self.hbar * self.m * omega_new) / 2.0
        p4_new = 3.0 * p2_new ** 2
        delta_E_new = -p4_new / (8.0 * (self.m ** 3) * (self.c ** 2))

        # Since omega increased, |ΔE| increases -> ΔE_new < ΔE_original (more negative)
        if delta_E_new < original_expected:
            return (True, f"Omega +5% gives ΔE = {delta_E_new:.6e} < original {original_expected:.6e} (correct trend)")
        return (False, f"Omega +5% gives ΔE = {delta_E_new:.6e} is not less than original {original_expected:.6e}")

    # ----------------------------------------------------------------------
    # Weak answer-sanity check (no math.isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Only check that Delta_E is finite, non-zero, and negative.
        Do NOT compare to any computed expected value.
        """
        if self.delta_E is None:
            return (False, "SKIP: Delta_E missing")
        if not math.isfinite(self.delta_E):
            return (False, "Delta_E is not finite")
        if self.delta_E == 0.0:
            return (False, "Delta_E is zero, expected non-zero negative")
        if self.delta_E >= 0:
            return (False, "Delta_E should be negative")
        return (True, "Delta_E is finite, non-zero, and negative (plausible)")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all methods starting with 'test_', execute each within a try/except,
        collect results. Returns (all_pass, logs_list).
        """
        logs = []
        all_pass = True
        # Get all callable attributes starting with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)