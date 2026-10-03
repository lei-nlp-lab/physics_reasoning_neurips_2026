import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits the Born approximation solution for delta-potential scattering.
    Checks interface, physical invariants, modeling formulas, monotonicity, and weak sanity.
    """

    def __init__(self, solution_instance: Any):
        # Inputs (givens) – safe extraction
        self.m: Optional[float] = None
        self.g: Optional[float] = None
        self.hbar: Optional[float] = None
        if hasattr(solution_instance, 'm'):
            self.m = solution_instance.m
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.m = solution_instance.results.get('m', None)

        if hasattr(solution_instance, 'g'):
            self.g = solution_instance.g
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.g = solution_instance.results.get('g', None)

        if hasattr(solution_instance, 'hbar'):
            self.hbar = solution_instance.hbar
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.hbar = solution_instance.results.get('hbar', None)

        # Results – safe extraction from solution_instance.results
        self.f: Optional[float] = None
        self.dsigma_dOmega: Optional[float] = None
        self.sigma_total: Optional[float] = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.f = solution_instance.results.get('f', None)
            self.dsigma_dOmega = solution_instance.results.get('dsigma_dOmega', None)
            self.sigma_total = solution_instance.results.get('sigma_total', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods. Never crashes."""
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

    # ============= Interface tests =============
    def test_interface_results_exist(self) -> Tuple[bool, str]:
        """Check that required result keys are present and not None."""
        if self.f is None:
            return (False, "Required result 'f' is missing or None")
        if self.dsigma_dOmega is None:
            return (False, "Required result 'dsigma_dOmega' is missing or None")
        if self.sigma_total is None:
            return (False, "Required result 'sigma_total' is missing or None")
        return (True, "All required results present")

    def test_interface_results_types(self) -> Tuple[bool, str]:
        """Check that results are numeric (int/float) and not NaN."""
        for name, val in [('f', self.f), ('dsigma_dOmega', self.dsigma_dOmega),
                         ('sigma_total', self.sigma_total)]:
            if val is None:
                continue  # already reported in previous test
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
            if math.isnan(val):
                return (False, f"{name} is NaN")
        return (True, "All results are numeric and finite")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that at least m, g, hbar are available for formula tests."""
        if self.m is None:
            return (False, "Given 'm' not available")
        if self.g is None:
            return (False, "Given 'g' not available")
        if self.hbar is None:
            return (False, "Given 'hbar' not available")
        return (True, "All required givens available")

    # ============= Invariant tests (theorem-based, lenient isclose) =============
    def test_invariant_dsigma_is_abs_f_squared(self) -> Tuple[bool, str]:
        """Check that dσ/dΩ = |f|²."""
        if self.f is None or self.dsigma_dOmega is None:
            return (False, "SKIP: f or dsigma_dOmega missing")
        f_sq = abs(self.f) ** 2
        if not math.isclose(self.dsigma_dOmega, f_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dσ/dΩ ({self.dsigma_dOmega}) != |f|² ({f_sq})")
        return (True, "dσ/dΩ matches |f|²")

    def test_invariant_sigma_is_4pi_dsigma(self) -> Tuple[bool, str]:
        """Check that σ_total = 4π dσ/dΩ (isotropic scattering)."""
        if self.dsigma_dOmega is None or self.sigma_total is None:
            return (False, "SKIP: dsigma_dOmega or sigma_total missing")
        expected_sigma = 4.0 * math.pi * self.dsigma_dOmega
        if not math.isclose(self.sigma_total, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"σ_total ({self.sigma_total}) != 4π dσ/dΩ ({expected_sigma})")
        return (True, "σ_total = 4π dσ/dΩ holds")

    # ============= Modeling / formula tests (derive expected values) =============
    def test_modeling_f_formula(self) -> Tuple[bool, str]:
        """Check f = - (m g) / (2π ħ²)."""
        if self.m is None or self.g is None or self.hbar is None:
            return (False, "SKIP: missing givens for f formula")
        if self.f is None:
            return (False, "SKIP: f not available")
        expected_f = - (self.m * self.g) / (2.0 * math.pi * self.hbar ** 2)
        if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f ({self.f}) differs from expected ({expected_f})")
        return (True, "f matches Born amplitude formula")

    def test_modeling_dsigma_formula(self) -> Tuple[bool, str]:
        """Check dσ/dΩ = m² g² / (4π² ħ⁴)."""
        if self.m is None or self.g is None or self.hbar is None:
            return (False, "SKIP: missing givens for dsigma formula")
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega not available")
        expected = (self.m ** 2 * self.g ** 2) / (4.0 * math.pi ** 2 * self.hbar ** 4)
        if not math.isclose(self.dsigma_dOmega, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dσ/dΩ ({self.dsigma_dOmega}) differs from expected ({expected})")
        return (True, "dσ/dΩ matches formula")

    def test_modeling_sigma_formula(self) -> Tuple[bool, str]:
        """Check σ_total = m² g² / (π ħ⁴)."""
        if self.m is None or self.g is None or self.hbar is None:
            return (False, "SKIP: missing givens for sigma formula")
        if self.sigma_total is None:
            return (False, "SKIP: sigma_total not available")
        expected = (self.m ** 2 * self.g ** 2) / (math.pi * self.hbar ** 4)
        if not math.isclose(self.sigma_total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"σ_total ({self.sigma_total}) differs from expected ({expected})")
        return (True, "σ_total matches formula")

    # ============= Trace consistency (pseudocode contract) =============
    def test_trace_f_to_dsigma_consistency(self) -> Tuple[bool, str]:
        """Trace: given f, dσ/dΩ should be exactly |f|² (numerical consistency)."""
        if self.f is None or self.dsigma_dOmega is None:
            return (False, "SKIP: missing f or dsigma_dOmega")
        f_sq = abs(self.f) ** 2
        if not math.isclose(self.dsigma_dOmega, f_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Trace mismatch: dσ/dΩ ({self.dsigma_dOmega}) != |f|² ({f_sq})")
        return (True, "Trace: f → dσ/dΩ consistent")

    def test_trace_dsigma_to_sigma_consistency(self) -> Tuple[bool, str]:
        """Trace: given dσ/dΩ, σ_total should be 4π dσ/dΩ."""
        if self.dsigma_dOmega is None or self.sigma_total is None:
            return (False, "SKIP: missing dsigma_dOmega or sigma_total")
        expected = 4.0 * math.pi * self.dsigma_dOmega
        if not math.isclose(self.sigma_total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Trace mismatch: σ_total ({self.sigma_total}) != 4π dσ/dΩ ({expected})")
        return (True, "Trace: dσ/dΩ → σ_total consistent")

    # ============= Monotonicity tests (perturb an input, compare baseline vs perturbed) =============
    def test_monotonicity_mass_increases_sigma(self) -> Tuple[bool, str]:
        """
        Monotonicity: increasing mass m should increase total cross section.
        Use the formula σ ∝ m² to compare baseline (original m) with perturbed m (increase by 5%).
        """
        if self.m is None or self.g is None or self.hbar is None:
            return (False, "SKIP: missing givens for monotonicity test")
        # Compute baseline total cross section from formula
        sigma_baseline = (self.m ** 2 * self.g ** 2) / (math.pi * self.hbar ** 4)
        # Perturb mass upward by 5%
        m_perturbed = self.m * 1.05
        sigma_perturbed = (m_perturbed ** 2 * self.g ** 2) / (math.pi * self.hbar ** 4)
        # Check monotonic increase
        if sigma_perturbed <= sigma_baseline:
            return (False, f"Monotonicity failed: increasing mass did not increase sigma "
                           f"(baseline {sigma_baseline:.6e}, perturbed {sigma_perturbed:.6e})")
        return (True, "Increasing mass correctly increases total cross section (∝ m²)")

    # Optionally, one could add a monotonicity test for g or hbar, but one is sufficient.

    # ============= Weak answer sanity (only one such test, no isclose, no derivation) =============
    def test_answer_sanity_sign_and_nonnegativity(self) -> Tuple[bool, str]:
        """Weak sanity: cross sections non‑negative, f finite."""
        if self.dsigma_dOmega is not None:
            if self.dsigma_dOmega < 0:
                return (False, "dσ/dΩ is negative (must be ≥ 0)")
        if self.sigma_total is not None:
            if self.sigma_total < 0:
                return (False, "σ_total is negative (must be ≥ 0)")
        if self.f is not None:
            if not math.isfinite(self.f):
                return (False, "f is not finite")
        return (True, "Cross sections non‑negative, f finite")