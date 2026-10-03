import math
import traceback

class PhysicsChecker:
    """
    Auditor for a solution estimating the temperature below which hydrogen
    atom-atom scattering is essentially S-wave.
    """

    def __init__(self, solution_instance):
        # Safely extract the threshold temperature result
        # Expected key: 'T0'
        results = getattr(solution_instance, 'results', {})
        self.T0 = results.get('T0', None)

        # Fundamental constants (from problem statement)
        self.h = 6.626e-34       # J·s
        self.m_H = 1.67e-27      # kg
        self.k_B = 1.38e-23      # J/K
        self.R = 4e-10           # m

    # ---------- Interface checks ----------
    def test_interface_required_keys(self):
        """Check that the required result key 'T0' exists and is numeric."""
        if self.T0 is None:
            return (False, "Missing required key 'T0' in results.")
        if not isinstance(self.T0, (int, float)):
            return (False, f"T0 must be numeric, got {type(self.T0).__name__}")
        return (True, "T0 present and numeric.")

    # ---------- Invariant (theorem-based) checks ----------
    def test_invariant_inequality_from_kR_eq1(self):
        """
        Verify that T0 satisfies the equality derived from kR = 1:
        T0 = h^2 / (8 pi^3 m_H k_B R^2)
        Uses math.isclose with 2% relative tolerance.
        """
        if self.T0 is None:
            return (False, "SKIP: T0 not available")
        numerator = self.h ** 2
        denominator = 8 * (math.pi ** 3) * self.m_H * self.k_B * (self.R ** 2)
        expected_T0 = numerator / denominator
        if not math.isclose(self.T0, expected_T0, rel_tol=0.02, abs_tol=1e-5):
            return (False, f"T0 = {self.T0:.2e}, expected ~ {expected_T0:.2e} "
                           f"(from kR=1 condition)")
        return (True, f"T0 agrees with kR=1 condition within 2%.")

    def test_invariant_wavenumber_condition(self):
        """
        Check that using T0, the wave number k satisfies k*R ≈ 1.
        k = 2π / λ_th, λ_th = h / sqrt(2π m_H k_B T0)
        """
        if self.T0 is None:
            return (False, "SKIP: T0 not available")
        # Thermal de Broglie wavelength at T0
        lambda_th = self.h / math.sqrt(2 * math.pi * self.m_H * self.k_B * self.T0)
        k = 2 * math.pi / lambda_th
        product = k * self.R
        if not math.isclose(product, 1.0, rel_tol=0.02, abs_tol=1e-3):
            return (False, f"k*R = {product:.3f}, expected ~ 1.0 (within 2%)")
        return (True, f"k*R = {product:.3f}, consistent with S-wave threshold.")

    # ---------- Modeling / trace checks ----------
    def test_modeling_derived_T0_consistency(self):
        """
        Re-derive T0 from the inequality using the exact formula and
        check consistency with the provided T0 (within 1% tolerance).
        This tests that the derivation from physics principles is correct.
        """
        if self.T0 is None:
            return (False, "SKIP: T0 not available")
        # Re-derive T0 from the given constants using the formula
        numerator = self.h ** 2
        denominator = 8 * (math.pi ** 3) * self.m_H * self.k_B * (self.R ** 2)
        derived_T0 = numerator / denominator
        if not math.isclose(self.T0, derived_T0, rel_tol=0.01, abs_tol=1e-5):
            return (False, f"Derived T0 = {derived_T0:.2e}, "
                           f"provided T0 = {self.T0:.2e} differ by more than 1%")
        return (True, "Derived T0 matches provided T0 within 1%.")

    def test_modeling_thermal_wavelength_positive(self):
        """
        Verify that the thermal de Broglie wavelength computed at T0
        is positive and finite, and that it satisfies the condition
        lambda_th = h / sqrt(2π m_H k_B T0).
        """
        if self.T0 is None:
            return (False, "SKIP: T0 not available")
        try:
            lambda_th = self.h / math.sqrt(2 * math.pi * self.m_H * self.k_B * self.T0)
        except (ValueError, ZeroDivisionError, OverflowError):
            return (False, "Failed to compute thermal de Broglie wavelength.")
        if not (math.isfinite(lambda_th) and lambda_th > 0):
            return (False, f"Thermal wavelength is not positive/finite: {lambda_th}")
        # Check that the computed wavelength is consistent with the formula by
        # verifying that k = 2π/λ_th gives k*R ≈ 1.
        k = 2 * math.pi / lambda_th
        if not math.isclose(k * self.R, 1.0, rel_tol=0.05):
            return (False, f"k*R = {k*self.R:.3f}, not close to 1 (within 5%)")
        return (True, f"Thermal wavelength {lambda_th:.2e} m leads to k*R ≈ {k*self.R:.3f}")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_T0_inverse_R2(self):
        """
        Perturb R by +5% and check that T0 scales as 1/R^2.
        Because T0 = h^2 / (8 pi^3 m_H k_B R^2), a +5% increase in R should
        reduce T0 by a factor (1.05)^(-2) ≈ 0.907.
        """
        if self.T0 is None:
            return (False, "SKIP: T0 not available")
        # baseline values (re-derive to avoid copying formula)
        numerator = self.h ** 2
        denominator = 8 * (math.pi ** 3) * self.m_H * self.k_B * (self.R ** 2)
        base_T0 = numerator / denominator

        # perturb R upwards by 5%
        R_perturbed = self.R * 1.05
        new_denominator = 8 * (math.pi ** 3) * self.m_H * self.k_B * (R_perturbed ** 2)
        perturbed_T0 = numerator / new_denominator
        expected_ratio = (1.05) ** (-2)
        actual_ratio = perturbed_T0 / base_T0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.01):
            return (False, f"T0 should scale as 1/R^2: ratio = {actual_ratio:.4f}, "
                           f"expected {expected_ratio:.4f}")
        return (True, f"T0 scales correctly with R (ratio {actual_ratio:.4f}).")

    # ---------- Final answer sanity (weak, at most one) ----------
    def test_answer_sanity_magnitude(self):
        """
        Weak check: T0 should be on the order of several K (e.g., 0.1–10 K).
        This is a broad sanity, not a precise match.
        """
        if self.T0 is None:
            return (False, "SKIP: T0 not available")
        if self.T0 < 0.1 or self.T0 > 10:
            return (False, f"T0 = {self.T0:.3f} K is outside plausible range (0.1–10 K)")
        return (True, f"T0 = {self.T0:.3f} K lies in plausible order-of-magnitude range.")

    # ---------- Test runner ----------
    def run_all_tests(self):
        """Discover and run all test_ methods, collecting results."""
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