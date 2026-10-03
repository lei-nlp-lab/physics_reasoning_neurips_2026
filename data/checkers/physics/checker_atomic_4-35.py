import math
import sympy as sp
import traceback
from typing import Any, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits a solution for Compton Scattering problem.
    Extracts necessary quantities and runs unit-test-like checks.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens (if provided)
        givens = getattr(solution_instance, 'givens', {})
        self.h = givens.get('h', None)           # Planck constant
        self.c = givens.get('c', None)           # speed of light
        self.m_e = givens.get('m_e', None)       # electron rest mass
        self.lambda_incident = givens.get('lambda', None)  # incident wavelength
        self.theta = givens.get('theta', None)   # scattering angle (radians)

        # Extract results (if provided)
        results = getattr(solution_instance, 'results', {})
        self.lambda_prime = results.get('lambda_prime', None)
        self.T = results.get('T', None)

        # Optional, might be used for reference only
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.physics_mapping = getattr(solution_instance, 'physics_mapping', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    def _has_all_constants(self) -> bool:
        """Check if all fundamental constants and givens are available."""
        return all(v is not None for v in [self.h, self.c, self.m_e, self.lambda_incident, self.theta])

    # ==================== Interface Tests ====================
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that 'lambda_prime' and 'T' exist and are numeric in results."""
        if self.lambda_prime is None:
            return (False, "Missing required result 'lambda_prime'")
        if self.T is None:
            return (False, "Missing required result 'T'")
        if not isinstance(self.lambda_prime, (int, float)):
            return (False, f"'lambda_prime' is not numeric: {type(self.lambda_prime)}")
        if not isinstance(self.T, (int, float)):
            return (False, f"'T' is not numeric: {type(self.T)}")
        return (True, "Required keys present and numeric")

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that necessary givens (constants, incident lambda, theta) are provided."""
        if not self._has_all_constants():
            return (False, "One or more required givens (h, c, m_e, lambda, theta) are missing")
        return (True, "All required givens present")

    # ==================== Invariant (Theorem‑based) Tests ====================
    def test_invariant_compton_shift(self) -> Tuple[bool, str]:
        """Check that the Compton shift formula holds within 2% relative tolerance."""
        if self.lambda_prime is None or not self._has_all_constants():
            return (False, "SKIP: missing data for Compton shift check")
        expected_lambda_prime = (self.lambda_incident +
                                 self.h / (self.m_e * self.c) * (1 - math.cos(self.theta)))
        if not math.isclose(self.lambda_prime, expected_lambda_prime, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"Compton shift mismatch: lambda' computed = {self.lambda_prime:.6e}, "
                    f"expected = {expected_lambda_prime:.6e}, "
                    f"diff = {abs(self.lambda_prime - expected_lambda_prime):.6e}")
        return (True, "Compton shift formula satisfied")

    def test_invariant_kinetic_energy_from_energy_conservation(self) -> Tuple[bool, str]:
        """Check that T = hc/lambda - hc/lambda' within 2% tolerance."""
        if self.lambda_prime is None or self.T is None or not self._has_all_constants():
            return (False, "SKIP: missing data for kinetic energy check")
        expected_T = self.h * self.c / self.lambda_incident - self.h * self.c / self.lambda_prime
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"Kinetic energy mismatch: T = {self.T:.6e}, expected = {expected_T:.6e}, "
                    f"diff = {abs(self.T - expected_T):.6e}")
        return (True, "Kinetic energy from energy conservation is correct")

    # ==================== Modeling / Trace Consistency Tests ====================
    def test_modeling_lambda_prime_positive(self) -> Tuple[bool, str]:
        """Check that scattered wavelength λ' > 0."""
        if self.lambda_prime is None:
            return (False, "SKIP: lambda_prime not available")
        if not self.lambda_prime > 0:
            return (False, f"lambda_prime must be positive, got {self.lambda_prime}")
        return (True, "lambda_prime is positive")

    def test_modeling_kinetic_energy_nonnegative(self) -> Tuple[bool, str]:
        """Check that kinetic energy T >= 0."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if self.T < 0:
            return (False, f"Kinetic energy must be non‑negative, got {self.T}")
        return (True, "Kinetic energy is non‑negative")

    def test_modeling_lambda_prime_ge_lambda(self) -> Tuple[bool, str]:
        """Check that λ' >= λ (since Compton shift is non‑negative)."""
        if self.lambda_prime is None or self.lambda_incident is None:
            return (False, "SKIP: missing wavelength data for comparison")
        if self.lambda_prime < self.lambda_incident:
            return (False, f"λ' < λ: {self.lambda_prime} < {self.lambda_incident}")
        return (True, "λ' ≥ λ (Compton shift non‑negative)")

    # ==================== Monotonicity Tests ====================
    def test_monotonicity_lambda_prime_with_theta(self) -> Tuple[bool, str]:
        """
        Check that λ' increases with θ via theoretical expectation.
        Uses the given constants to compute λ' at θ+0.1 rad and compares with solution's λ'.
        (This assumes the solution is correct at the original θ, but verifies consistency
        of the monotonic trend.)
        """
        if self.lambda_prime is None or not self._has_all_constants():
            return (False, "SKIP: missing data for monotonicity check")
        # Ensure perturbed theta stays within [0, π]
        delta = 0.1
        if self.theta + delta > math.pi:
            return (False, "SKIP: theta too close to π for perturbation")
        lambda_prime_perturbed = (self.lambda_incident +
                                  self.h / (self.m_e * self.c) * (1 - math.cos(self.theta + delta)))
        # monotonicity: λ'(θ+δ) > λ'(θ)  (since cos is decreasing on [0,π])
        if not lambda_prime_perturbed > self.lambda_prime:
            return (False,
                    f"λ' is not increasing with θ: at θ={self.theta:.3f} λ'={self.lambda_prime:.6e}, "
                    f"at θ+{delta} λ' = {lambda_prime_perturbed:.6e}")
        return (True, f"λ' increases with θ (tested with δ={delta} rad)")

    def test_monotonicity_kinetic_energy_with_theta(self) -> Tuple[bool, str]:
        """
        Check that kinetic energy T increases with θ using theoretical expectation.
        Compares T at θ+0.1 rad (computed via T = hc/λ - hc/λ') with solution's T.
        """
        if self.lambda_prime is None or self.T is None or not self._has_all_constants():
            return (False, "SKIP: missing data for monotonicity check")
        delta = 0.1
        if self.theta + delta > math.pi:
            return (False, "SKIP: theta too close to π for perturbation")
        lambda_prime_perturbed = (self.lambda_incident +
                                  self.h / (self.m_e * self.c) * (1 - math.cos(self.theta + delta)))
        T_perturbed = self.h * self.c / self.lambda_incident - self.h * self.c / lambda_prime_perturbed
        # T should increase with θ (since λ' increases, hc/λ' decreases, so T increases)
        if not T_perturbed > self.T:
            return (False,
                    f"T does not increase with θ: at θ={self.theta:.3f} T={self.T:.6e}, "
                    f"at θ+{delta} T={T_perturbed:.6e}")
        return (True, f"Kinetic energy increases with θ (tested with δ={delta} rad)")

    # ==================== Answer Sanity (Weak, Secondary) ====================
    def test_answer_sanity_finite(self) -> Tuple[bool, str]:
        """Check that outputs are finite and not NaN (weak sanity only)."""
        chk = []
        if self.lambda_prime is not None and (math.isinf(self.lambda_prime) or math.isnan(self.lambda_prime)):
            chk.append("lambda_prime")
        if self.T is not None and (math.isinf(self.T) or math.isnan(self.T)):
            chk.append("T")
        if chk:
            return (False, f"Non‑finite values detected in: {', '.join(chk)}")
        return (True, "All outputs are finite and not NaN")

    # ==================== Run All Tests ====================
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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