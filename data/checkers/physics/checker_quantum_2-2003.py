import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits solution for the problem:
    A NaCl crystal with negative ion vacancies (electrons) treated as particles
    in a 3D infinite square well. Estimate longest wavelength absorbed.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.lambda_val: Optional[float] = results.get('lambda', None)

        # Extract givens (assumed to be attributes or in givens dict)
        self.m_e: Optional[float] = None
        self.h: Optional[float] = None
        self.c: Optional[float] = None
        self.a: Optional[float] = None

        # Try attribute access first
        for attr in ['m_e', 'h', 'c', 'a']:
            setattr(self, attr, getattr(solution_instance, attr, None))

        # Fallback: try givens dict if attributes not set
        givens = getattr(solution_instance, 'givens', {})
        if self.m_e is None:
            self.m_e = givens.get('m_e', None)
        if self.h is None:
            self.h = givens.get('h', None)
        if self.c is None:
            self.c = givens.get('c', None)
        if self.a is None:
            self.a = givens.get('a', None)

        # Standard constants (if needed for fallback, but we use self.*)
        self._const_m_e = 9.11e-31
        self._const_h = 6.626e-34
        self._const_c = 3.00e8
        self._const_a = 0.564e-9  # in meters

    # ----------------------------------------------------------------
    # Interface tests: check that required keys exist and have valid types
    # ----------------------------------------------------------------
    def test_interface_lambda_key_exists(self) -> Tuple[bool, str]:
        if self.lambda_val is None:
            return (False, "Key 'lambda' not found in results.")
        return (True, "Key 'lambda' exists in results.")

    def test_interface_lambda_is_positive(self) -> Tuple[bool, str]:
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        if not isinstance(self.lambda_val, (int, float)):
            return (False, f"lambda is not numeric: {type(self.lambda_val)}")
        if self.lambda_val <= 0:
            return (False, f"lambda is not positive: {self.lambda_val}")
        return (True, f"lambda = {self.lambda_val} > 0")

    def test_interface_lambda_is_finite(self) -> Tuple[bool, str]:
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        if not math.isfinite(self.lambda_val):
            return (False, f"lambda is not finite: {self.lambda_val}")
        return (True, "lambda is finite")

    # ----------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity checks
    # ----------------------------------------------------------------
    def test_invariant_transition_energy_formula(self) -> Tuple[bool, str]:
        """
        Check that lambda = 8*m_e*a²*c/(3h) is consistent with the result.
        Uses math.isclose with relaxed tolerance because of rounding noise.
        """
        # If any required quantity is missing, use constants as fallback
        m_e = self.m_e if self.m_e is not None else self._const_m_e
        h = self.h if self.h is not None else self._const_h
        c = self.c if self.c is not None else self._const_c
        a = self.a if self.a is not None else self._const_a

        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")

        try:
            expected_lambda_m = (8 * m_e * a * a * c) / (3 * h)
            expected_lambda_A = expected_lambda_m * 1e10  # convert to Angstrom
        except Exception as e:
            return (False, f"Computation failed: {e}")

        # Use relaxed tolerance as per invariance rule
        if math.isclose(self.lambda_val, expected_lambda_A, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"lambda matches formula: {self.lambda_val:.1f} vs {expected_lambda_A:.1f} Å")
        else:
            return (False, f"lambda {self.lambda_val:.1f} deviates from expected {expected_lambda_A:.1f} Å")

    def test_invariant_positiveness_of_derived_deltaE(self) -> Tuple[bool, str]:
        """Check that the implied Delta_E is positive (energy gap > 0)."""
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        if self.lambda_val <= 0:
            return (False, "Lambda is not positive, Delta_E cannot be positive.")
        # Compute Delta_E = h*c / lambda (lambda in meters)
        h = self.h if self.h is not None else self._const_h
        c = self.c if self.c is not None else self._const_c
        lambda_m = self.lambda_val * 1e-10
        delta_E = h * c / lambda_m
        if delta_E > 0:
            return (True, f"Implied Delta_E = {delta_E:.2e} J > 0")
        else:
            return (False, "Implied Delta_E is not positive.")

    # ----------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------
    def test_modeling_transition_is_ground_to_first_excited(self) -> Tuple[bool, str]:
        """Check that the energy gap corresponds to (1,1,2) or (2,1,1) transition from ground (1,1,1)."""
        # The gap should be (3h^2)/(8 m_e a^2). If result matches, it's using correct quantum numbers.
        m_e = self.m_e if self.m_e is not None else self._const_m_e
        h = self.h if self.h is not None else self._const_h
        a = self.a if self.a is not None else self._const_a
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        try:
            delta_E_expected = (3 * h * h) / (8 * m_e * a * a)
            lambda_expected_m = h * self._const_c / delta_E_expected
            lambda_expected_A = lambda_expected_m * 1e10
        except Exception as e:
            return (False, f"Computation error: {e}")
        if math.isclose(self.lambda_val, lambda_expected_A, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Energy gap consistent with ground to first excited state transition.")
        else:
            return (False, f"Lambda {self.lambda_val:.1f} not consistent with expected transition {lambda_expected_A:.1f} Å")

    def test_modeling_length_scale_is_lattice_constant(self) -> Tuple[bool, str]:
        """Check that the well dimension a is on the order of the lattice constant (~0.56 nm)."""
        a = self.a if self.a is not None else self._const_a
        if a is None:
            return (False, "SKIP: lattice constant a not available.")
        if a < 0.1e-9 or a > 10e-9:
            return (False, f"a = {a:.2e} m, outside expected range (0.1-10 nm).")
        # Specific check: a ~ 0.564 nm
        if 0.5e-9 <= a <= 0.6e-9:
            return (True, f"a = {a:.3e} m, consistent with NaCl lattice constant.")
        else:
            return (False, f"a = {a:.3e} m, but expected ~0.564 nm.")

    # ----------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------
    def test_monotonicity_lambda_increases_with_a(self) -> Tuple[bool, str]:
        """Perturb lattice constant a by +10% and check lambda increases (since lambda ∝ a²)."""
        m_e = self.m_e if self.m_e is not None else self._const_m_e
        h = self.h if self.h is not None else self._const_h
        c = self.c if self.c is not None else self._const_c
        a = self.a if self.a is not None else self._const_a
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")

        try:
            # Compute expected lambda at current a and at perturbed a
            lambda_original = (8 * m_e * a * a * c) / (3 * h) * 1e10
            a_perturbed = a * 1.1
            lambda_perturbed = (8 * m_e * a_perturbed * a_perturbed * c) / (3 * h) * 1e10
        except Exception as e:
            return (False, f"Computation error: {e}")

        if lambda_perturbed > lambda_original:
            return (True, f"Lambda increases with a: original {lambda_original:.1f}, perturbed {lambda_perturbed:.1f} Å")
        else:
            return (False, "Lambda did not increase with a as expected (lambda ∝ a²)")

    def test_monotonicity_lambda_decreases_with_m_e(self) -> Tuple[bool, str]:
        """Perturb electron mass by +10% and check lambda decreases (lambda ∝ m_e)."""
        m_e = self.m_e if self.m_e is not None else self._const_m_e
        h = self.h if self.h is not None else self._const_h
        c = self.c if self.c is not None else self._const_c
        a = self.a if self.a is not None else self._const_a
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")

        try:
            lambda_original = (8 * m_e * a * a * c) / (3 * h) * 1e10
            m_perturbed = m_e * 1.1
            lambda_perturbed = (8 * m_perturbed * a * a * c) / (3 * h) * 1e10
        except Exception as e:
            return (False, f"Computation error: {e}")

        if lambda_perturbed < lambda_original:
            return (True, f"Lambda decreases with m_e: original {lambda_original:.1f}, perturbed {lambda_perturbed:.1f} Å")
        else:
            return (False, "Lambda did not decrease with m_e as expected (lambda ∝ m_e)")

    # ----------------------------------------------------------------
    # Weak answer sanity (only one such method allowed)
    # ----------------------------------------------------------------
    def test_answer_sanity_broad_magnitude(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that lambda is within a broad physical range.
        For NaCl, expected λ ~ 110 Å, so range 10-500 Å is generous.
        """
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        if 10.0 <= self.lambda_val <= 500.0:
            return (True, f"lambda = {self.lambda_val:.1f} Å within reasonable range.")
        else:
            return (False, f"lambda = {self.lambda_val:.1f} Å, expected near 110 Å (range 10-500).")

    # ----------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Collect all public methods starting with 'test_'
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)