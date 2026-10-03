import sympy as sp
import math
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits the solution to the heavy neutrino decay problem.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract givens dictionary safely
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Result quantities (None if missing)
        self.E_gamma = self.results.get('E_gamma', None)
        self.E_gamma_approx = self.results.get('E_gamma_approx', None)
        self.detection_outlook = self.results.get('detection_outlook', None)

        # Given parameters
        self.m_H = self.givens.get('m_H', None)
        self.m_L = self.givens.get('m_L', None)

        # Fallback to direct attributes on solution_instance
        if self.m_H is None and hasattr(solution_instance, 'm_H'):
            self.m_H = solution_instance.m_H
        if self.m_L is None and hasattr(solution_instance, 'm_L'):
            self.m_L = solution_instance.m_L

        # Optional explanation steps (for reference only)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods and execute them safely."""
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
        return all_pass, logs

    # ==============================================================
    # INTERFACE TESTS – required keys and types
    # ==============================================================
    def test_interface_required_keys_exists(self) -> Tuple[bool, str]:
        if self.E_gamma is None:
            return (False, "SKIP: E_gamma is missing")
        if self.E_gamma_approx is None:
            return (False, "SKIP: E_gamma_approx is missing")
        if self.detection_outlook is None:
            return (False, "SKIP: detection_outlook is missing")
        return (True, "All required result keys present")

    def test_interface_photon_energy_type(self) -> Tuple[bool, str]:
        if self.E_gamma is None:
            return (False, "SKIP: E_gamma not available")
        if isinstance(self.E_gamma, (int, float)):
            return (True, "E_gamma is numeric")
        return (False, f"E_gamma type is {type(self.E_gamma)}")

    def test_interface_photon_energy_approx_type(self) -> Tuple[bool, str]:
        if self.E_gamma_approx is None:
            return (False, "SKIP: E_gamma_approx not available")
        if isinstance(self.E_gamma_approx, (int, float)):
            return (True, "E_gamma_approx is numeric")
        return (False, f"E_gamma_approx type is {type(self.E_gamma_approx)}")

    def test_interface_detection_outlook_string(self) -> Tuple[bool, str]:
        if self.detection_outlook is None:
            return (False, "SKIP: detection_outlook not available")
        if isinstance(self.detection_outlook, str) and len(self.detection_outlook) > 0:
            return (True, "detection_outlook is a non-empty string")
        return (False, f"detection_outlook is not a non-empty string: {repr(self.detection_outlook)}")

    # ==============================================================
    # INVARIANT TESTS – theorem-based physical sanity (use math.isclose)
    # ==============================================================
    def test_invariant_energy_conservation_formula(self) -> Tuple[bool, str]:
        """
        Check that submitted E_gamma matches the exact formula
        re-derived from m_H and m_L.
        """
        if self.E_gamma is None or self.m_H is None or self.m_L is None:
            return (False, "SKIP: missing E_gamma, m_H, or m_L")
        if self.m_H == 0:
            return (False, "SKIP: m_H is zero")
        expected = (self.m_H**2 - self.m_L**2) / (2.0 * self.m_H)
        if math.isclose(self.E_gamma, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_gamma = {self.E_gamma} agrees with formula ({expected})")
        return (False, f"E_gamma = {self.E_gamma} deviates from expected {expected}")

    def test_invariant_photon_energy_positive(self) -> Tuple[bool, str]:
        if self.E_gamma is None:
            return (False, "SKIP: E_gamma not available")
        if self.E_gamma > 0:
            return (True, "E_gamma > 0")
        return (False, f"E_gamma = {self.E_gamma} should be positive")

    def test_invariant_energy_less_than_mH(self) -> Tuple[bool, str]:
        if self.E_gamma is None or self.m_H is None:
            return (False, "SKIP: missing E_gamma or m_H")
        if self.E_gamma < self.m_H:
            return (True, f"E_gamma = {self.E_gamma} < m_H = {self.m_H}")
        return (False, f"E_gamma = {self.E_gamma} should be less than m_H = {self.m_H}")

    # ==============================================================
    # MODELING / TRACE TESTS – check physics consistency of the solution
    # ==============================================================
    def test_modeling_conservation_laws_satisfied(self) -> Tuple[bool, str]:
        """
        Using the submitted E_gamma, verify that the momentum/energy
        conservation and the relativistic dispersion relation hold.
        """
        if self.E_gamma is None or self.m_H is None or self.m_L is None:
            return (False, "SKIP: missing E_gamma, m_H, or m_L")
        # Re-derive the relations
        # E_L = m_H - E_gamma
        E_L = self.m_H - self.E_gamma
        # p_gamma = E_gamma (c=1)
        p_gamma = self.E_gamma
        # p_L = p_gamma (momentum conservation)
        p_L = p_gamma
        # Check E_L^2 = p_L^2 + m_L^2
        lhs = E_L**2
        rhs = p_L**2 + self.m_L**2
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Conservation and dispersion satisfied")
        return (False, f"E_L^2 = {lhs} != p_L^2 + m_L^2 = {rhs}")

    def test_modeling_detection_outlook_meaningful(self) -> Tuple[bool, str]:
        """
        The detection outlook string should be indicative of the physics.
        We only verify it has reasonable length and is not trivial.
        """
        if self.detection_outlook is None:
            return (False, "SKIP: detection_outlook not available")
        if len(self.detection_outlook) >= 3:
            return (True, "detection_outlook has sufficient length")
        return (False, "detection_outlook too short to be meaningful")

    # ==============================================================
    # MONOTONICITY TESTS – perturb an input and compare outputs
    # ==============================================================
    def test_monotonicity_photon_energy_decreases_with_mL(self) -> Tuple[bool, str]:
        """
        From the exact formula, perturb m_L upward by 1% and verify
        that the predicted photon energy decreases.
        """
        if self.m_H is None or self.m_L is None:
            return (False, "SKIP: missing m_H or m_L")
        if self.m_H == 0:
            return (False, "SKIP: m_H is zero")
        # Baseline output (using formula)
        baseline = (self.m_H**2 - self.m_L**2) / (2.0 * self.m_H)
        # Perturb m_L by 1%
        delta = 0.01 * abs(self.m_L) if self.m_L != 0 else 0.001  # avoid zero delta
        m_L_pert = self.m_L + delta
        perturbed = (self.m_H**2 - m_L_pert**2) / (2.0 * self.m_H)
        if perturbed < baseline:
            return (True, f"Perturbing m_L by +{delta} reduces E_gamma from {baseline} to {perturbed}")
        return (False, f"E_gamma did not decrease: baseline {baseline}, perturbed {perturbed}")

    def test_monotonicity_photon_energy_approx_trend_with_mL(self) -> Tuple[bool, str]:
        """
        In the limit m_L→0, the approximate energy should approach m_H/2.
        We perturb m_L from 0 to a small positive value and check that
        E_gamma (from formula) decreases, consistent with the approximation.
        """
        if self.m_H is None or self.m_H <= 0:
            return (False, "SKIP: m_H missing or non-positive")
        # Baseline: m_L = 0
        baseline = self.m_H / 2.0
        # Perturb: take a small m_L (1% of m_H)
        m_L_small = 0.01 * self.m_H
        perturbed = (self.m_H**2 - m_L_small**2) / (2.0 * self.m_H)
        if perturbed < baseline:
            return (True, f"With m_L={m_L_small}, E_gamma ≈ {perturbed} < m_H/2 = {baseline}")
        return (False, f"Expected decrease but got baseline {baseline}, perturbed {perturbed}")

    # ==============================================================
    # ANSWER SANITY TEST (weak, at most one)
    # ==============================================================
    def test_answer_sanity_photon_energy_finite(self) -> Tuple[bool, str]:
        if self.E_gamma is None:
            return (False, "SKIP: E_gamma not available")
        if math.isfinite(self.E_gamma):
            return (True, "E_gamma is finite")
        return (False, f"E_gamma is not finite: {self.E_gamma}")

# End of PhysicsChecker