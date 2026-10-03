import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Sanity and physics consistency checker for the Čerenkov radiation problem.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary with None defaults
        results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None) or {}
        givens: Optional[Dict[str, Any]] = getattr(solution_instance, 'givens', None) or {}
        explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None) or []

        # Givens
        self.n = givens.get('n', None)
        self.m0_e = givens.get('m0_e', None)   # electron rest mass (MeV/c^2)
        self.p = givens.get('p', None)         # momentum (GeV/c)
        self.delta_theta = givens.get('delta_theta', None)  # rad
        self.m0_near = givens.get('m0_near', None)          # GeV/c^2

        # Results (from [emit] contract)
        self.cos_theta = results.get('cos_theta', None)        # equation string or number
        self.beta_th = results.get('beta_th', None)            # threshold beta
        self.gamma_th = results.get('gamma_th', None)          # threshold gamma
        self.KE_min = results.get('KE_min', None)              # MeV
        self.beta0 = results.get('beta0', None)                # beta for p, m0_near
        self.theta0 = results.get('theta0', None)              # rad
        self.dtheta_dm0 = results.get('dtheta_dm0', None)      # rad/(GeV/c^2)
        self.delta_m0 = results.get('delta_m0', None)          # GeV/c^2
        self.frac_err = results.get('frac_err', None)          # dimensionless

        # Explanation steps for guidance
        self.explanation_steps = explanation_steps

    # ----------------------------------------------------------------------
    # Interface checks (keys exist and have valid types)
    # ----------------------------------------------------------------------
    def test_interface_cos_theta(self) -> Tuple[bool, str]:
        """Check that cos_theta result exists."""
        if self.cos_theta is None:
            return (False, "cos_theta missing")
        return (True, "cos_theta present")

    def test_interface_beta_th(self) -> Tuple[bool, str]:
        """Check that beta_th result exists."""
        if self.beta_th is None:
            return (False, "beta_th missing")
        return (True, "beta_th present")

    def test_interface_gamma_th(self) -> Tuple[bool, str]:
        """Check that gamma_th result exists."""
        if self.gamma_th is None:
            return (False, "gamma_th missing")
        return (True, "gamma_th present")

    def test_interface_KE_min(self) -> Tuple[bool, str]:
        """Check that KE_min result exists."""
        if self.KE_min is None:
            return (False, "KE_min missing")
        return (True, "KE_min present")

    def test_interface_beta0(self) -> Tuple[bool, str]:
        """Check that beta0 result exists."""
        if self.beta0 is None:
            return (False, "beta0 missing")
        return (True, "beta0 present")

    def test_interface_theta0(self) -> Tuple[bool, str]:
        """Check that theta0 result exists."""
        if self.theta0 is None:
            return (False, "theta0 missing")
        return (True, "theta0 present")

    def test_interface_dtheta_dm0(self) -> Tuple[bool, str]:
        """Check that dtheta_dm0 result exists."""
        if self.dtheta_dm0 is None:
            return (False, "dtheta_dm0 missing")
        return (True, "dtheta_dm0 present")

    def test_interface_delta_m0(self) -> Tuple[bool, str]:
        """Check that delta_m0 result exists."""
        if self.delta_m0 is None:
            return (False, "delta_m0 missing")
        return (True, "delta_m0 present")

    def test_interface_frac_err(self) -> Tuple[bool, str]:
        """Check that frac_err result exists."""
        if self.frac_err is None:
            return (False, "frac_err missing")
        return (True, "frac_err present")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, using math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_threshold_beta(self) -> Tuple[bool, str]:
        """Threshold beta must equal 1/n."""
        if self.beta_th is None or self.n is None:
            return (False, "SKIP: missing beta_th or n")
        expected = 1.0 / self.n
        if math.isclose(self.beta_th, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"beta_th = {self.beta_th} close to 1/n = {expected}")
        return (False, f"beta_th = {self.beta_th} not close to 1/n = {expected}")

    def test_invariant_threshold_gamma(self) -> Tuple[bool, str]:
        """Threshold gamma must equal n / sqrt(n^2 - 1)."""
        if self.gamma_th is None or self.n is None:
            return (False, "SKIP: missing gamma_th or n")
        expected = self.n / math.sqrt(self.n**2 - 1)
        if math.isclose(self.gamma_th, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"gamma_th = {self.gamma_th} close to n/sqrt(n^2-1) = {expected}")
        return (False, f"gamma_th = {self.gamma_th} not close to n/sqrt(n^2-1) = {expected}")

    def test_invariant_ke_min_from_gamma(self) -> Tuple[bool, str]:
        """KE_min must equal (gamma_th - 1) * m0_e * c^2 (with c=1 in natural units)."""
        if self.KE_min is None or self.gamma_th is None or self.m0_e is None:
            return (False, "SKIP: missing KE_min, gamma_th, or m0_e")
        # Using natural units where c=1, so m0_e already in MeV/c^2, energy in MeV.
        expected = (self.gamma_th - 1) * self.m0_e
        if math.isclose(self.KE_min, expected, rel_tol=1e-2, abs_tol=1e-1):
            return (True, f"KE_min = {self.KE_min} close to (gamma_th-1)*m0_e = {expected}")
        return (False, f"KE_min = {self.KE_min} not close to (gamma_th-1)*m0_e = {expected}")

    def test_invariant_cerenkov_condition(self) -> Tuple[bool, str]:
        """
        For the given beta0 and n, cos(theta0) must be close to 1/(n*beta0).
        Uses numerical values if available.
        """
        if self.beta0 is None or self.n is None or self.theta0 is None:
            return (False, "SKIP: missing beta0, n, or theta0")
        cos_computed = math.cos(self.theta0)
        expected = 1.0 / (self.n * self.beta0)
        if math.isclose(cos_computed, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"cos(theta0) = {cos_computed} close to 1/(n*beta0) = {expected}")
        return (False, f"cos(theta0) = {cos_computed} not close to 1/(n*beta0) = {expected}")

    # ----------------------------------------------------------------------
    # Modeling checks (based on explanation steps and derivative relations)
    # ----------------------------------------------------------------------
    def test_modeling_derivative_relation(self) -> Tuple[bool, str]:
        """
        Check that dtheta_dm0 is close to the analytic derivative:
        dθ/dm0 = (m0 * β0) / (n * p^2 * sin(θ0))   (with c=1)
        This test uses the provided dtheta_dm0 and checks consistency.
        """
        if (self.dtheta_dm0 is None or self.beta0 is None or self.n is None or
            self.p is None or self.theta0 is None or self.m0_near is None):
            return (False, "SKIP: missing one or more required quantities")
        # Use natural units: p in GeV/c, m0_near in GeV/c^2, c=1
        expected_der = (self.m0_near * self.beta0) / (self.n * self.p**2 * math.sin(self.theta0))
        if math.isclose(self.dtheta_dm0, expected_der, rel_tol=1e-2, abs_tol=1e-4):
            return (True, f"dtheta_dm0 = {self.dtheta_dm0} close to expected = {expected_der}")
        return (False, f"dtheta_dm0 = {self.dtheta_dm0} not close to expected = {expected_der}")

    def test_modeling_error_propagation(self) -> Tuple[bool, str]:
        """
        Check that delta_m0 is consistent with delta_theta / |dtheta_dm0|.
        This is a direct consequence of error propagation.
        """
        if (self.delta_m0 is None or self.delta_theta is None or
            self.dtheta_dm0 is None):
            return (False, "SKIP: missing delta_m0, delta_theta, or dtheta_dm0")
        expected = self.delta_theta / abs(self.dtheta_dm0)
        if math.isclose(self.delta_m0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_m0 = {self.delta_m0} close to delta_theta/|dtheta_dm0| = {expected}")
        return (False, f"delta_m0 = {self.delta_m0} not close to delta_theta/|dtheta_dm0| = {expected}")

    # ----------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input and comparison)
    # ----------------------------------------------------------------------
    def test_monotonicity_gamma_increases_with_beta(self) -> Tuple[bool, str]:
        """
        Perturb beta_th by a small positive amount and check that the
        corresponding gamma (computed from the relativistic definition)
        increases. This verifies the monotonic relation γ = 1/√(1-β²).
        Baseline: beta_th, gamma_th (or compute from beta_th).
        Perturbation: β' = β_th + ε (with ε small enough to keep β'<1).
        """
        if self.beta_th is None:
            return (False, "SKIP: beta_th missing")

        # Ensure beta_th is less than 1 to allow a small perturbation.
        if self.beta_th >= 1.0:
            return (False, f"beta_th = {self.beta_th} >= 1, cannot perturb")

        # Use a small relative epsilon (0.1% of (1-beta_th) but at most 1e-6).
        delta = min(1e-6, (1.0 - self.beta_th) * 0.1)
        epsilon = delta

        # Compute baseline gamma from beta_th (or use gamma_th if available)
        gamma_baseline = 1.0 / math.sqrt(1.0 - self.beta_th**2)

        # Perturb beta upward
        beta_pert = self.beta_th + epsilon
        if beta_pert >= 1.0:
            beta_pert = 0.999999999 * self.beta_th + (1.0 - self.beta_th) * 0.9  # safer fallback
            epsilon = beta_pert - self.beta_th
        gamma_pert = 1.0 / math.sqrt(1.0 - beta_pert**2)

        # Monotonicity: gamma_pert should be larger than gamma_baseline.
        if gamma_pert > gamma_baseline:
            return (True, f"β increased by {epsilon:.2e}, γ increased from {gamma_baseline:.6f} to {gamma_pert:.6f}")
        else:
            return (False, f"γ did not increase: baseline {gamma_baseline}, perturbed {gamma_pert} (β changed by {epsilon})")

    def test_monotonicity_theta_increases_with_beta(self) -> Tuple[bool, str]:
        """
        For the Čerenkov condition cosθ = 1/(nβ), as β increases, θ increases.
        Use beta0 and theta0 to verify that a small increase in beta gives
        a larger theta (computed from the condition).
        """
        if self.beta0 is None or self.n is None or self.theta0 is None:
            return (False, "SKIP: missing beta0, n, or theta0")

        # Baseline theta from result
        theta_baseline = self.theta0

        # Perturb beta0 upward by a small relative amount
        delta_beta = min(1e-8, (1.0 - self.beta0) * 0.1)
        beta_pert = self.beta0 + delta_beta
        if beta_pert >= 1.0:
            beta_pert = 0.999999999 * self.beta0 + (1.0 - self.beta0) * 0.9
            delta_beta = beta_pert - self.beta0

        # Compute perturbed theta from Čerenkov condition
        cos_theta_pert = 1.0 / (self.n * beta_pert)
        if cos_theta_pert > 1.0:
            return (False, f"Perturbed cosθ = {cos_theta_pert} > 1, perturbation too large")
        theta_pert = math.acos(cos_theta_pert)

        # Because cosθ decreases with β, θ should increase.
        if theta_pert > theta_baseline:
            return (True, f"β increased by {delta_beta:.2e}, θ increased from {theta_baseline:.6f} to {theta_pert:.6f}")
        else:
            return (False, f"θ did not increase: baseline {theta_baseline}, perturbed {theta_pert}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (sign, finite, plausible magnitude)
    # ----------------------------------------------------------------------
    def test_answer_sanity_frac_err(self) -> Tuple[bool, str]:
        """
        Check that the fractional error is positive and of order 0.1 (the target is 0.13).
        This is a weak sanity – no exact matching.
        """
        if self.frac_err is None:
            return (False, "SKIP: frac_err missing")
        if self.frac_err <= 0:
            return (False, f"frac_err = {self.frac_err} must be positive")
        if math.isnan(self.frac_err) or math.isinf(self.frac_err):
            return (False, f"frac_err = {self.frac_err} is NaN or Inf")
        # Plausible range: between 0.01 and 1.0
        if 0.01 <= self.frac_err <= 1.0:
            return (True, f"frac_err = {self.frac_err} within plausible range [0.01, 1.0]")
        return (False, f"frac_err = {self.frac_err} outside plausible range")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
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