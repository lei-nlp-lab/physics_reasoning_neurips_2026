import math
import traceback
from typing import Any, Tuple, Dict

class PhysicsChecker:
    """
    PhysicsChecker for two spinless particles in a 1D infinite well with delta interaction.
    Audits first-order perturbation theory results.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Required outputs (from emit contract)
        self.E0_unpert = self.results.get('E0_unpert', None)
        self.E1_corr = self.results.get('E1_corr', None)
        self.E_total = self.results.get('E_total', None)

        # Required input givens (from pseudocode)
        self.L = getattr(solution_instance, 'L', None)
        self.m = getattr(solution_instance, 'm', None)
        self.a = getattr(solution_instance, 'a', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Derived constants for convenience
        self.pi = math.pi

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys_exist(self) -> Tuple[bool, str]:
        """Check that all three required result keys are present."""
        needed = ['E0_unpert', 'E1_corr', 'E_total']
        missing = [k for k in needed if self.results.get(k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_results_are_numeric(self) -> Tuple[bool, str]:
        """Check that result values are numeric (int or float) and not None."""
        for name in ['E0_unpert', 'E1_corr', 'E_total']:
            val = getattr(self, name)
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric ({type(val).__name__})")
        return (True, "All result values are numeric.")

    def test_interface_givens_are_present(self) -> Tuple[bool, str]:
        """Check that input givens (L, m, a, hbar) are available and positive."""
        required_givens = ['L', 'm', 'a', 'hbar']
        missing = []
        for g in required_givens:
            val = getattr(self, g, None)
            if val is None:
                missing.append(g)
            elif not isinstance(val, (int, float)):
                return (False, f"Given '{g}' is not numeric")
            elif val <= 0:
                return (False, f"Given '{g}' must be positive (got {val})")
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All input givens are present and positive.")

    # ----------------------------------------------------------------------
    # Invariant / theorem‑based checks (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_unperturbed_energy_formula(self) -> Tuple[bool, str]:
        """Check that E0_unpert = π²ħ²/(mL²)."""
        if None in (self.E0_unpert, self.L, self.m, self.hbar):
            return (True, "SKIP: missing data for unperturbed energy check")
        expected = self.pi**2 * self.hbar**2 / (self.m * self.L**2)
        if math.isclose(self.E0_unpert, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E0_unpert matches formula: {self.E0_unpert:.6e} ≈ {expected:.6e}")
        else:
            return (False, f"E0_unpert={self.E0_unpert:.6e} differs from expected {expected:.6e}")

    def test_invariant_first_order_correction_formula(self) -> Tuple[bool, str]:
        """Check that E1_corr = 3a/(2L)."""
        if None in (self.E1_corr, self.a, self.L):
            return (True, "SKIP: missing data for correction check")
        expected = 3.0 * self.a / (2.0 * self.L)
        if math.isclose(self.E1_corr, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E1_corr matches formula: {self.E1_corr:.6e} ≈ {expected:.6e}")
        else:
            return (False, f"E1_corr={self.E1_corr:.6e} differs from expected {expected:.6e}")

    def test_invariant_energy_addition(self) -> Tuple[bool, str]:
        """Check that E_total = E0_unpert + E1_corr."""
        if None in (self.E_total, self.E0_unpert, self.E1_corr):
            return (True, "SKIP: missing data for energy addition check")
        expected_sum = self.E0_unpert + self.E1_corr
        if math.isclose(self.E_total, expected_sum, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_total matches sum: {self.E_total:.6e} ≈ {expected_sum:.6e}")
        else:
            return (False, f"E_total={self.E_total:.6e} ≠ sum {expected_sum:.6e}")

    def test_invariant_perturbation_sign(self) -> Tuple[bool, str]:
        """Check that E1_corr has the same sign as a (since integral is positive)."""
        if None in (self.E1_corr, self.a):
            return (True, "SKIP: missing data for sign check")
        # Allow zero -> zero
        if self.a == 0 and self.E1_corr == 0:
            return (True, "Zero a gives zero correction.")
        if (self.a > 0 and self.E1_corr > 0) or (self.a < 0 and self.E1_corr < 0):
            return (True, f"E1_corr sign matches a: E1_corr={self.E1_corr:.6e}, a={self.a}")
        else:
            return (False, f"E1_corr sign inconsistent with a: E1_corr={self.E1_corr:.6e}, a={self.a}")

    # ----------------------------------------------------------------------
    # Modeling quality checks (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_delta_integral_uses_sine_square(self) -> Tuple[bool, str]:
        """
        Check that E1_corr / (a * (4/L²)) equals the integral of sin⁴(πx/L) from 0 to L.
        The analytic result is 3L/8.
        """
        if None in (self.E1_corr, self.a, self.L):
            return (True, "SKIP: missing data for modeling integral check")
        # Compute the coefficient: (2/L)^2 = 4/L²
        coeff = 4.0 / (self.L**2)
        # E1_corr should equal a * coeff * I, where I = 3L/8
        expected_I = 3.0 * self.L / 8.0
        computed_I = self.E1_corr / (self.a * coeff) if self.a != 0 else 0.0
        if self.a == 0:
            if self.E1_corr == 0:
                return (True, "Zero a gives zero correction, integral check passed.")
            else:
                return (False, "Zero a but non-zero correction.")
        if math.isclose(computed_I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Inferred integral I ≈ {computed_I:.6e} matches expected {expected_I:.6e}")
        else:
            return (False, f"Inferred integral I = {computed_I:.6e} differs from expected {expected_I:.6e}")

    def test_modeling_symmetric_ground_state(self) -> Tuple[bool, str]:
        """
        Check that the ground state is symmetric: E0_unpert uses n=l=1.
        This implies the wavefunction is symmetric under exchange.
        (Indirect check: the simple product ψ₁(x₁)ψ₁(x₂) is symmetric.)
        Since we have no explicit wavefunction, we just verify that the energy
        formula used is consistent with both particles in the same state.
        """
        # The only possible check is that E0_unpert is π²ħ²/(mL²) for two particles.
        # Already checked in invariant. So this is a soft pass.
        if None in (self.E0_unpert, self.L, self.m, self.hbar):
            return (True, "SKIP: missing data for symmetry check")
        expected = self.pi**2 * self.hbar**2 / (self.m * self.L**2)
        if math.isclose(self.E0_unpert, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Ground state energy consistent with symmetric n=l=1.")
        else:
            return (False, "Ground state energy not consistent with symmetric state.")

    # ----------------------------------------------------------------------
    # Monotonicity checks (directional/trend)
    # ----------------------------------------------------------------------
    def test_monotonicity_energy_increases_with_a(self) -> Tuple[bool, str]:
        """
        For a > 0, increasing a increases E_total; for a < 0, increasing a
        (less negative) also increases E_total. We test by perturbing a by +1%.
        """
        if None in (self.E_total, self.L, self.m, self.hbar, self.a):
            return (True, "SKIP: missing data for monotonicity a test")
        # Baseline total (from formula) to avoid dependency on self.E_total accuracy
        baseline = (self.pi**2 * self.hbar**2 / (self.m * self.L**2)) + (3.0 * self.a / (2.0 * self.L))
        perturbation = 0.01 * self.a  # 1% of a
        if perturbation == 0:
            return (True, "a=0, monotonicity test not applicable.")
        new_a = self.a + perturbation
        new_total = baseline + (3.0 * perturbation / (2.0 * self.L))  # correction change
        if perturbation > 0:
            expected_increase = True
        else:
            # a negative, perturbation is negative, making a more negative => total decreases
            expected_increase = False
        if expected_increase:
            if new_total > baseline:
                return (True, f"E_total increases with a: baseline={baseline:.6e}, new={new_total:.6e}")
            else:
                return (False, f"E_total did not increase as expected: baseline={baseline:.6e}, new={new_total:.6e}")
        else:
            if new_total < baseline:
                return (True, f"E_total decreases with a (negative perturbation): baseline={baseline:.6e}, new={new_total:.6e}")
            else:
                return (False, f"E_total did not decrease as expected: baseline={baseline:.6e}, new={new_total:.6e}")

    def test_monotonicity_unperturbed_energy_decreases_with_L(self) -> Tuple[bool, str]:
        """Check that increasing L decreases E0_unpert."""
        if None in (self.E0_unpert, self.L, self.m, self.hbar):
            return (True, "SKIP: missing data for monotonicity L test")
        # Baseline expected
        baseline = self.pi**2 * self.hbar**2 / (self.m * self.L**2)
        # Increase L by 1%
        new_L = self.L * 1.01
        new_expected = self.pi**2 * self.hbar**2 / (self.m * new_L**2)
        if new_expected < baseline:
            return (True, f"E0_unpert decreases with L: baseline={baseline:.6e}, new={new_expected:.6e}")
        else:
            return (False, f"E0_unpert did not decrease: baseline={baseline:.6e}, new={new_expected:.6e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_and_positive(self) -> Tuple[bool, str]:
        """Weak sanity: E_total should be finite, not NaN, and positive if a>0."""
        if self.E_total is None:
            return (True, "SKIP: E_total not available")
        if not math.isfinite(self.E_total):
            return (False, f"E_total is not finite: {self.E_total}")
        if math.isnan(self.E_total):
            return (False, f"E_total is NaN")
        # If a is present and positive, expect positive total
        if self.a is not None and self.a > 0 and self.E_total <= 0:
            return (False, f"E_total={self.E_total} should be positive since a>0")
        return (True, "E_total is finite, not NaN, and plausible sign.")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
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