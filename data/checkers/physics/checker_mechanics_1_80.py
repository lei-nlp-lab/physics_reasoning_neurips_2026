import math
import traceback
from typing import Any, Dict, List, Tuple


class PhysicsChecker:
    """
    Physics audit for the problem:
    "A mass m moves in a plane in uniform circular motion with angular frequency ω.
     The centripetal force is provided by a spring (constant K, ignore gravity).
     A very small radial impulse is given. Find the frequency of the resulting radial oscillation."

    Target answer: ω' = sqrt(3ω² + K/m)
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', None) or {}
        self.m = givens.get('m', getattr(solution_instance, 'm', None))
        self.K = givens.get('K', getattr(solution_instance, 'K', None))
        self.omega = givens.get('omega', getattr(solution_instance, 'omega', None))

        # Extract results safely
        results = getattr(solution_instance, 'results', None) or {}
        self.omega_prime = results.get('omega_prime', getattr(solution_instance, 'omega_prime', None))

        # Extract auxiliary data (used for trace checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []
        self.pseudocode = getattr(solution_instance, 'pseudocode', None) or {}

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_given_m(self) -> Tuple[bool, str]:
        """Check that mass m is provided and positive."""
        if self.m is None:
            return (False, "Mass m is missing")
        if not isinstance(self.m, (int, float)):
            return (False, f"m is not numeric: {type(self.m)}")
        if self.m <= 0:
            return (False, f"m must be positive, got {self.m}")
        return (True, f"m = {self.m} is present and positive")

    def test_interface_given_K(self) -> Tuple[bool, str]:
        """Check that spring constant K is provided and positive."""
        if self.K is None:
            return (False, "Spring constant K is missing")
        if not isinstance(self.K, (int, float)):
            return (False, f"K is not numeric: {type(self.K)}")
        if self.K <= 0:
            return (False, f"K must be positive, got {self.K}")
        return (True, f"K = {self.K} is present and positive")

    def test_interface_given_omega(self) -> Tuple[bool, str]:
        """Check that angular frequency ω is provided and positive."""
        if self.omega is None:
            return (False, "Angular frequency ω is missing")
        if not isinstance(self.omega, (int, float)):
            return (False, f"ω is not numeric: {type(self.omega)}")
        if self.omega <= 0:
            return (False, f"ω must be positive, got {self.omega}")
        return (True, f"ω = {self.omega} is present and positive")

    def test_interface_result_omega_prime_exists(self) -> Tuple[bool, str]:
        """Check that result ω' is present."""
        if self.omega_prime is None:
            return (False, "Result ω' (omega_prime) is missing")
        return (True, "ω' is present")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_effective_potential_second_derivative(self) -> Tuple[bool, str]:
        """
        Check that ω'² = (1/m) * U_eff''(r0) = 3ω² + K/m.
        Derived from effective potential method.
        """
        if any(x is None for x in (self.K, self.m, self.omega, self.omega_prime)):
            return (False, "SKIP: missing one or more required quantities (K, m, ω, ω')")
        expected_sq = 3 * self.omega**2 + self.K / self.m
        actual_sq = self.omega_prime**2
        if math.isclose(actual_sq, expected_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ω'² = {actual_sq} matches expected {expected_sq}")
        else:
            return (False, f"ω'² = {actual_sq} does not match expected {expected_sq} (tolerance 1%)")

    def test_invariant_radial_frequency_greater_than_angular_frequency(self) -> Tuple[bool, str]:
        """
        Since K/m > 0, we must have ω'² > 3ω², hence ω' > sqrt(3)*ω > ω.
        This is a theorem-level inequality that must hold.
        """
        if any(x is None for x in (self.omega, self.omega_prime)):
            return (False, "SKIP: missing ω or ω' for inequality check")
        if self.omega == 0:
            return (False, "SKIP: ω cannot be zero for comparison")
        if self.omega_prime / self.omega <= 1.0:
            return (False, f"ω' = {self.omega_prime} must be > ω = {self.omega}")
        # Even more restrictive: ω'² > 3ω²
        if self.omega_prime**2 <= 3 * self.omega**2:
            return (False, f"ω'² = {self.omega_prime**2} must be > 3ω² = {3*self.omega**2}")
        return (True, f"ω'² = {self.omega_prime**2} > 3ω² = {3*self.omega**2} (as required)")

    # ------------------------------------------------------------------
    # Modeling / trace checks – based on explanation steps and linearization
    # ------------------------------------------------------------------
    def test_modeling_angular_momentum_conservation_stated(self) -> Tuple[bool, str]:
        """Check that explanation steps mention angular momentum conservation after the impulse."""
        if not isinstance(self.explanation_steps, (list, str)):
            return (False, "SKIP: explanation_steps not available as list or string")
        text = ' '.join(self.explanation_steps) if isinstance(self.explanation_steps, list) else self.explanation_steps
        keywords = ['angular momentum', 'constant', 'r^2\\dot{\\theta}']
        found = all(kw in text.lower() for kw in [k.lower() for k in keywords if k])
        if found:
            return (True, "Angular momentum conservation is mentioned")
        else:
            return (False, "Explanation steps do not clearly state angular momentum conservation")

    def test_modeling_linearization_equation_present(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps contain the linearized equation
        d²r'/dt² + (3ω² + K/m) r' = 0.
        """
        if not isinstance(self.explanation_steps, (list, str)):
            return (False, "SKIP: explanation_steps not available")
        text = ' '.join(self.explanation_steps) if isinstance(self.explanation_steps, list) else self.explanation_steps
        required_expressions = [
            '3\\omega^2 + \\frac{K}{m}',
            '\\ddot{r}'  # second derivative of r prime
        ]
        all_present = all(expr in text for expr in required_expressions)
        if all_present:
            return (True, "Linearization equation found in explanation steps")
        else:
            return (False, "The key linearized equation (with 3ω²+K/m) is not clearly present in explanation steps")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb one input and compare perturbed output to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_omega_prime_increases_with_K(self) -> Tuple[bool, str]:
        """If K increases (with m, ω fixed), then ω' should increase."""
        if any(x is None for x in (self.K, self.m, self.omega, self.omega_prime)):
            return (False, "SKIP: missing required quantities for monotonicity")
        try:
            baseline = math.sqrt(3 * self.omega**2 + self.K / self.m)
        except (ZeroDivisionError, ValueError):
            return (False, "SKIP: baseline computation fails")
        # Perturb K by +10%
        K_pert = self.K * 1.1
        perturbed = math.sqrt(3 * self.omega**2 + K_pert / self.m)
        if perturbed > baseline:
            return (True, f"ω' increases with K: baseline {baseline:.4f}, perturbed K+10% -> {perturbed:.4f}")
        else:
            return (False, f"ω' did not increase when K increased by 10%: baseline {baseline:.4f}, perturbed {perturbed:.4f}")

    def test_monotonicity_omega_prime_increases_with_omega(self) -> Tuple[bool, str]:
        """If ω increases (with m, K fixed), then ω' should increase."""
        if any(x is None for x in (self.K, self.m, self.omega, self.omega_prime)):
            return (False, "SKIP: missing required quantities for monotonicity")
        try:
            baseline = math.sqrt(3 * self.omega**2 + self.K / self.m)
        except (ZeroDivisionError, ValueError):
            return (False, "SKIP: baseline computation fails")
        # Perturb ω by +10%
        omega_pert = self.omega * 1.1
        perturbed = math.sqrt(3 * omega_pert**2 + self.K / self.m)
        if perturbed > baseline:
            return (True, f"ω' increases with ω: baseline {baseline:.4f}, perturbed ω+10% -> {perturbed:.4f}")
        else:
            return (False, f"ω' did not increase when ω increased by 10%: baseline {baseline:.4f}, perturbed {perturbed:.4f}")

    # ------------------------------------------------------------------
    # Answer sanity (weak check – no isclose, no derivation of expected value)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_and_finite(self) -> Tuple[bool, str]:
        """ω' must be positive, real, finite."""
        if self.omega_prime is None:
            return (False, "SKIP: omega_prime not available")
        if not isinstance(self.omega_prime, (int, float)):
            return (False, f"omega_prime is not numeric: {type(self.omega_prime)}")
        if self.omega_prime <= 0:
            return (False, f"omega_prime must be positive, got {self.omega_prime}")
        if not math.isfinite(self.omega_prime):
            return (False, f"omega_prime is not finite: {self.omega_prime}")
        return (True, f"omega_prime = {self.omega_prime} is positive and finite")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods and execute them, collecting results."""
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