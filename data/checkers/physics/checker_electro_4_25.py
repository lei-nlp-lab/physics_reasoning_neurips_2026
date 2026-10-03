import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

import sympy as sp

class PhysicsChecker:
    """
    Checker for attenuation distance of a plane wave in a good conductor.
    Tests focus on theorem invariants, modeling quality, and monotonicity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed data safely, using None for missing values.
        Never assign numeric defaults like 0.0, 9.81, etc.
        """
        # Try to get results dictionary; fallback to empty dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}

        # Main result: attenuation distance delta
        self.delta = results.get('delta', None)
        # Additional keys that might be present for deeper checking
        self.alpha = results.get('alpha', None)
        self.k = results.get('k', None)  # complex wave number
        self.sigma = results.get('sigma', None)
        self.mu = results.get('mu', None)
        self.omega = results.get('omega', None)

        # Explanation steps for modeling context (not for hardcoding numbers)
        self.explanation = getattr(solution_instance, 'explanation_steps', None)
        # Pseudocode for trace reference, not for final authority
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface tests: required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_delta_exists(self) -> Tuple[bool, str]:
        """Check that attenuation distance delta is provided."""
        if self.delta is None:
            return (False, "Missing 'delta' in results")
        return (True, "delta is present")

    def test_interface_delta_type(self) -> Tuple[bool, str]:
        """Check that delta is a number (int, float, or sympy numeric)."""
        if self.delta is None:
            return (False, "SKIP: delta not available")
        if isinstance(self.delta, (int, float, sp.core.numbers.Number)):
            return (True, "delta is a number")
        return (False, f"delta has type {type(self.delta).__name__}, expected numeric")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_alpha_delta_relation(self) -> Tuple[bool, str]:
        """Check that delta = 1 / alpha, where alpha = sqrt(omega*mu*sigma/2)."""
        # Requires alpha computed or derivable from given parameters
        # Try to compute alpha from sigma, mu, omega if given
        sigma = self.sigma
        mu = self.mu
        omega = self.omega
        delta = self.delta

        if any(x is None for x in (sigma, mu, omega, delta)):
            # Fall back to using alpha from results if available
            if self.alpha is not None and delta is not None:
                # Check delta = 1/alpha
                if isinstance(delta, (int, float)) and isinstance(self.alpha, (int, float)):
                    if math.isclose(delta, 1.0 / self.alpha, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"delta ≈ 1/alpha (alpha={self.alpha})")
                    else:
                        return (False, f"delta={delta} does not match 1/alpha={1/self.alpha}")
                else:
                    return (False, "SKIP: non-numeric alpha or delta")
            else:
                return (False, "SKIP: insufficient data to check alpha-delta relation")
        # All numeric: compute theoretical alpha
        try:
            sigma_f = float(sigma)
            mu_f = float(mu)
            omega_f = float(omega)
            delta_f = float(delta)
        except (TypeError, ValueError):
            return (False, "SKIP: cannot convert givens to float")

        alpha_theory = math.sqrt(omega_f * mu_f * sigma_f / 2.0)
        delta_from_alpha = 1.0 / alpha_theory if alpha_theory != 0 else float('inf')

        if math.isclose(delta_f, delta_from_alpha, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta ≈ sqrt(2/(ωμσ)) = {delta_from_alpha}")
        else:
            return (False, f"delta={delta_f} does not match theoretical {delta_from_alpha}")

    def test_invariant_dispersion_relation(self) -> Tuple[bool, str]:
        """Check that k^2 = i*omega*mu*sigma."""
        # Need k, omega, mu, sigma
        k = self.k
        omega = self.omega
        mu = self.mu
        sigma = self.sigma
        if any(x is None for x in (k, omega, mu, sigma)):
            return (False, "SKIP: missing k or givens for dispersion check")

        try:
            k_f = complex(k)
            omega_f = float(omega)
            mu_f = float(mu)
            sigma_f = float(sigma)
        except (TypeError, ValueError):
            return (False, "SKIP: cannot convert to numeric")

        lhs = k_f ** 2
        rhs = 1j * omega_f * mu_f * sigma_f
        if math.isclose(lhs.real, rhs.real, rel_tol=1e-2, abs_tol=1e-2) and \
           math.isclose(lhs.imag, rhs.imag, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "k^2 ≈ iωμσ")
        else:
            return (False, f"k^2={lhs} does not match iωμσ={rhs}")

    # ------------------------------------------------------------------
    # Modeling / trace tests: quality checks based on explanation steps
    # ------------------------------------------------------------------
    def test_modeling_delta_from_alpha(self) -> Tuple[bool, str]:
        """Check that delta is derived as 1/alpha from the attenuation constant."""
        # If both delta and alpha are present, check relation
        if self.delta is None or self.alpha is None:
            return (False, "SKIP: delta or alpha missing")
        try:
            d = float(self.delta)
            a = float(self.alpha)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric values")

        if math.isclose(d * a, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "delta = 1/alpha holds")
        else:
            return (False, f"delta*alpha = {d*a}, expected ≈1")

    def test_modeling_positive_alpha(self) -> Tuple[bool, str]:
        """Alpha (attenuation constant) must be positive."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        try:
            a = float(self.alpha)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric alpha")
        if a > 0:
            return (True, f"alpha={a} is positive")
        else:
            return (False, f"alpha must be positive, got {a}")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb physical parameters and check directional effect
    # ------------------------------------------------------------------
    def test_monotonicity_delta_decreases_with_sigma(self) -> Tuple[bool, str]:
        """
        Check that if sigma increases, delta decreases.
        Use small positive perturbation (e.g., +10%).
        """
        delta = self.delta
        sigma = self.sigma
        if delta is None or sigma is None:
            return (False, "SKIP: delta or sigma missing")

        try:
            sigma0 = float(sigma)
            delta0 = float(delta)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric values")

        # Perturb sigma by +10%
        sigma1 = sigma0 * 1.10
        # Recompute delta using theoretical formula (since checker cannot call external solver)
        mu = self.mu
        omega = self.omega
        if mu is None or omega is None:
            return (False, "SKIP: mu or omega missing for recomputation")

        try:
            mu_val = float(mu)
            omega_val = float(omega)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric mu/omega")

        # Theoretical delta with perturbed sigma (assuming mu, omega unchanged)
        # delta = sqrt(2/(omega*mu*sigma))
        delta1 = math.sqrt(2.0 / (omega_val * mu_val * sigma1))

        if delta1 < delta0:
            return (True, f"Increasing sigma by 10% reduces delta from {delta0} to {delta1}")
        else:
            return (False, f"Expected delta to decrease but got delta0={delta0}, delta1={delta1}")

    def test_monotonicity_delta_decreases_with_omega(self) -> Tuple[bool, str]:
        """Check that if omega increases, delta decreases."""
        delta = self.delta
        omega = self.omega
        if delta is None or omega is None:
            return (False, "SKIP: delta or omega missing")

        try:
            omega0 = float(omega)
            delta0 = float(delta)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric values")

        omega1 = omega0 * 1.10
        sigma = self.sigma
        mu = self.mu
        if sigma is None or mu is None:
            return (False, "SKIP: sigma or mu missing for recomputation")
        try:
            sigma_val = float(sigma)
            mu_val = float(mu)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric sigma/mu")

        delta1 = math.sqrt(2.0 / (omega1 * mu_val * sigma_val))
        if delta1 < delta0:
            return (True, f"Increasing omega by 10% reduces delta from {delta0} to {delta1}")
        else:
            return (False, f"Expected delta to decrease but got delta0={delta0}, delta1={delta1}")

    def test_monotonicity_delta_decreases_with_mu(self) -> Tuple[bool, str]:
        """Check that if mu increases, delta decreases."""
        delta = self.delta
        mu = self.mu
        if delta is None or mu is None:
            return (False, "SKIP: delta or mu missing")

        try:
            mu0 = float(mu)
            delta0 = float(delta)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric values")

        mu1 = mu0 * 1.10
        sigma = self.sigma
        omega = self.omega
        if sigma is None or omega is None:
            return (False, "SKIP: sigma or omega missing for recomputation")
        try:
            sigma_val = float(sigma)
            omega_val = float(omega)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric sigma/omega")

        delta1 = math.sqrt(2.0 / (omega_val * mu1 * sigma_val))
        if delta1 < delta0:
            return (True, f"Increasing mu by 10% reduces delta from {delta0} to {delta1}")
        else:
            return (False, f"Expected delta to decrease but got delta0={delta0}, delta1={delta1}")

    # ------------------------------------------------------------------
    # Answer sanity check: weak and non-dominating
    # ------------------------------------------------------------------
    def test_answer_sanity_delta_positive_and_finite(self) -> Tuple[bool, str]:
        """Check that delta is positive, finite, and not NaN."""
        if self.delta is None:
            return (False, "SKIP: delta not available")
        try:
            d = float(self.delta)
        except (TypeError, ValueError):
            return (False, "delta is non-numeric")
        if math.isnan(d) or math.isinf(d):
            return (False, f"delta is {d}: nan or inf")
        if d > 0:
            return (True, f"delta={d} is positive and finite")
        else:
            return (False, f"delta={d} is not positive")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Collect all test methods, run each, and return (overall_pass, logs).
        """
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)