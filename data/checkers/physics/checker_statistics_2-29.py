import math
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

class PhysicsChecker:
    """
    Audit solution for grand-canonical hydrogen states near a metal surface.
    Checks interface, physical invariants, modeling consistency, and monotonicity.
    """

    def __init__(self, solution_instance: Any):
        # Extract result dictionary safely
        results: Dict[str, Any] = getattr(solution_instance, 'results', None) or {}
        if not isinstance(results, dict):
            results = {}
        self.w0 = results.get('w0', None)
        self.w1 = results.get('w1', None)
        self.w2 = results.get('w2', None)
        self.p0 = results.get('p0', None)
        self.p1 = results.get('p1', None)
        self.p2 = results.get('p2', None)
        self.mu_solution = results.get('mu_solution', None)
        self.Phi_solution = results.get('Phi_solution', None)

        # Extract givens (work function, temperature, etc.) from solution instance
        # Try attributes or nested givens dict
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}
        if isinstance(givens, dict):
            self.work_function = givens.get('work_function', getattr(solution_instance, 'work_function', None))
            self.T = givens.get('T', getattr(solution_instance, 'T', None))
        else:
            self.work_function = getattr(solution_instance, 'work_function', None)
            self.T = getattr(solution_instance, 'T', None)

        # Boltzmann constant in eV/K
        self.k_B = 8.617333262e-5

        # Derived beta
        self.beta = None
        if self.T is not None and self.k_B is not None:
            try:
                self.beta = 1.0 / (self.k_B * self.T)
            except (TypeError, ZeroDivisionError):
                self.beta = None

        # Expected energies (eV)
        self.E0 = 0.0
        self.E1 = -13.6
        self.E2 = -14.2
        self.g0 = 1
        self.g1 = 2
        self.g2 = 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _weight_ratio_Hplus_H0(self, mu: float) -> Optional[float]:
        """Compute theoretical ratio w1/w0 = g1/g0 * exp(beta*(mu*(1-0) - (E1-E0)))"""
        if self.beta is None:
            return None
        return self.g1 / self.g0 * math.exp(self.beta * (mu - (self.E1 - self.E0)))

    def _weight_ratio_Hminus_H0(self, mu: float) -> Optional[float]:
        """Compute theoretical ratio w2/w0 = g2/g0 * exp(beta*(2*mu - E2))"""
        if self.beta is None:
            return None
        return self.g2 / self.g0 * math.exp(self.beta * (2 * mu - self.E2))

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_has_weights(self) -> Tuple[bool, str]:
        """Check that all three unnormalised weights are present."""
        if None in (self.w0, self.w1, self.w2):
            return (False, "Missing one or more weights (w0, w1, w2)")
        return (True, "All weights present")

    def test_interface_has_probabilities(self) -> Tuple[bool, str]:
        """Check that all three probabilities are present."""
        if None in (self.p0, self.p1, self.p2):
            return (False, "Missing one or more probabilities (p0, p1, p2)")
        return (True, "All probabilities present")

    def test_interface_has_mu_solution(self) -> Tuple[bool, str]:
        """Check that mu_solution key exists."""
        if self.mu_solution is None:
            return (False, "Missing mu_solution")
        return (True, "mu_solution present")

    def test_interface_has_Phi_solution(self) -> Tuple[bool, str]:
        """Check that Phi_solution key exists."""
        if self.Phi_solution is None:
            return (False, "Missing Phi_solution")
        return (True, "Phi_solution present")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_probability_sum(self) -> Tuple[bool, str]:
        """Probabilities must sum to 1 (within tolerance)."""
        if None in (self.p0, self.p1, self.p2):
            return (False, "SKIP: probabilities not all available")
        s = self.p0 + self.p1 + self.p2
        if math.isclose(s, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Sum = {s:.6f}  (close to 1)")
        else:
            return (False, f"Sum = {s:.6f}  (not close to 1)")

    def test_invariant_weight_ratio_Hplus_H0(self) -> Tuple[bool, str]:
        """
        Check that w1/w0 equals 2*exp(beta*(mu+13.6))  (with mu = -work_function).
        Uses provided work_function if available, otherwise compares ratio directly from weights.
        """
        if None in (self.w0, self.w1):
            return (False, "SKIP: w0 or w1 missing")
        if self.w0 == 0:
            return (False, "SKIP: w0 is zero, ratio undefined")
        ratio_observed = self.w1 / self.w0

        # If we have work_function and beta, compute expected ratio
        mu = None
        if self.work_function is not None and self.beta is not None:
            mu = -self.work_function
        elif self.mu_solution is not None and self.beta is not None:
            # Fallback: use Q2 mu as approximation? Not ideal, but better than nothing
            mu = self.mu_solution
        else:
            return (False, "SKIP: cannot compute expected ratio (no work_function or beta)")

        expected = self._weight_ratio_Hplus_H0(mu)
        if expected is None or expected == 0:
            return (False, "SKIP: could not compute expected ratio")

        # Check relative difference
        rel_diff = abs(ratio_observed - expected) / expected
        if math.isclose(ratio_observed, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"w1/w0 = {ratio_observed:.2e}, expected {expected:.2e}")
        else:
            return (False, f"w1/w0 = {ratio_observed:.2e}, expected {expected:.2e}, rel diff {rel_diff:.4f}")

    def test_invariant_weight_ratio_Hminus_H0(self) -> Tuple[bool, str]:
        """w2/w0 should equal 1*exp(beta*(2μ + 14.2))."""
        if None in (self.w0, self.w2):
            return (False, "SKIP: w0 or w2 missing")
        if self.w0 == 0:
            return (False, "SKIP: w0 is zero")
        ratio_observed = self.w2 / self.w0

        mu = None
        if self.work_function is not None and self.beta is not None:
            mu = -self.work_function
        elif self.mu_solution is not None and self.beta is not None:
            mu = self.mu_solution
        else:
            return (False, "SKIP: cannot compute expected ratio (no work_function or beta)")

        expected = self._weight_ratio_Hminus_H0(mu)
        if expected is None:
            return (False, "SKIP: could not compute expected ratio")
        if math.isclose(ratio_observed, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"w2/w0 = {ratio_observed:.2e}, expected {expected:.2e}")
        else:
            return (False, f"w2/w0 = {ratio_observed:.2e}, expected {expected:.2e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_q1_relative_magnitudes(self) -> Tuple[bool, str]:
        """For Φ=4 eV (Q1), p1 should be ~1, p0 and p2 ~0."""
        if None in (self.p0, self.p1, self.p2):
            return (False, "SKIP: probabilities missing")
        # check approximate dominance of H0
        if self.p1 > 0.5 and self.p0 < 0.01 and self.p2 < 0.01:
            return (True, f"p1={self.p1:.4f}, p0={self.p0:.4f}, p2={self.p2:.4f}  (consistent with dominance)")
        else:
            return (False, f"Expected p1≈1, p0≈0, p2≈0 but got p1={self.p1:.4f}, p0={self.p0:.4f}, p2={self.p2:.4f}")

    def test_modeling_q2_mu_from_equal_P(self) -> Tuple[bool, str]:
        """Check that mu_solution satisfies p1 = p2 condition:
           ln(g1) + β(μ + 13.6) = ln(g2) + β(2μ + 14.2)  → μ = kT ln(1/2) - 0.6.
        """
        if self.mu_solution is None or self.beta is None:
            return (False, "SKIP: mu_solution or beta missing")
        # Derive expected mu from equality
        expected_mu = (math.log(1.0 / 2.0) / self.beta) - 0.6
        if math.isclose(self.mu_solution, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"mu_solution = {self.mu_solution:.4f}, expected ≈ {expected_mu:.4f}")
        else:
            return (False, f"mu_solution = {self.mu_solution:.4f}, expected ≈ {expected_mu:.4f}")

    def test_modeling_q2_Phi_from_mu(self) -> Tuple[bool, str]:
        """Phi_solution should equal -mu_solution approximately."""
        if self.mu_solution is None or self.Phi_solution is None:
            return (False, "SKIP: mu_solution or Phi_solution missing")
        expected_Phi = -self.mu_solution
        if math.isclose(self.Phi_solution, expected_Phi, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Phi_solution = {self.Phi_solution:.4f}, -mu = {expected_Phi:.4f}")
        else:
            return (False, f"Phi_solution = {self.Phi_solution:.4f}, -mu = {expected_Phi:.4f}")

    # ------------------------------------------------------------------
    # Answer sanity (weak)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs_and_range(self) -> Tuple[bool, str]:
        """Very basic sanity: all probabilities non-negative, sum ~1, mu negative, Phi positive."""
        checks = []
        if None in (self.p0, self.p1, self.p2):
            checks.append("probabilities missing – skip")
        else:
            if self.p0 < -1e-6 or self.p1 < -1e-6 or self.p2 < -1e-6:
                return (False, "Negative probability detected")
            if abs(self.p0 + self.p1 + self.p2 - 1.0) > 0.1:
                return (False, f"Probabilities sum = {self.p0+self.p1+self.p2:.4f} (not ≈1)")
            checks.append("probabilities OK")
        if self.mu_solution is not None:
            if self.mu_solution >= 0:
                return (False, f"mu_solution = {self.mu_solution} should be negative")
            checks.append("mu negative")
        if self.Phi_solution is not None:
            if self.Phi_solution <= 0:
                return (False, f"Phi_solution = {self.Phi_solution} should be positive")
            checks.append("Phi positive")
        if not checks:
            return (False, "No data to check")
        return (True, "; ".join(checks))

    # ------------------------------------------------------------------
    # Monotonicity (if work_function provided)
    # ------------------------------------------------------------------
    def test_monotonicity_work_function_increases_p0(self) -> Tuple[bool, str]:
        """
        When work function increases (more binding), μ becomes more negative,
        so the probability of H+ (empty) should increase.
        Check this by computing p0 for two work functions (current and +1%) using provided data.
        """
        if self.work_function is None or self.beta is None:
            return (False, "SKIP: work_function or beta missing")
        if None in (self.g0, self.g1, self.g2, self.E0, self.E1, self.E2):
            return (False, "SKIP: constants missing")
        # Baseline
        mu0 = -self.work_function
        Xi0 = (self.g0 * math.exp(self.beta * (0 * mu0 - self.E0)) +
                self.g1 * math.exp(self.beta * (1 * mu0 - self.E1)) +
                self.g2 * math.exp(self.beta * (2 * mu0 - self.E2)))
        p0_baseline = self.g0 * math.exp(self.beta * (0 * mu0 - self.E0)) / Xi0
        # Perturb work function upward by 1%
        wf_pert = self.work_function * 1.01
        mu_pert = -wf_pert
        Xi_pert = (self.g0 * math.exp(self.beta * (0 * mu_pert - self.E0)) +
                   self.g1 * math.exp(self.beta * (1 * mu_pert - self.E1)) +
                   self.g2 * math.exp(self.beta * (2 * mu_pert - self.E2)))
        p0_pert = self.g0 * math.exp(self.beta * (0 * mu_pert - self.E0)) / Xi_pert
        if p0_pert > p0_baseline:
            return (True, f"p0 increased from {p0_baseline:.6e} to {p0_pert:.6e} when work function increased")
        else:
            return (False, f"p0 decreased from {p0_baseline:.6e} to {p0_pert:.6e} (expected increase)")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, return overall success and log."""
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
        return all_pass, logs