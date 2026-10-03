import math
import sys
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for the 'Idealized ping pong ball in ground state' problem.
    Checks interface completeness, invariant physics, modeling consistency,
    monotonicity trends, and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all relevant quantities from the solution instance.
        Use None as default for missing values.
        """
        # Extract results dictionary safely
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Expected keys from the [emit] contract
        self.alpha: Optional[float] = results.get('alpha', None)
        self.K: Optional[float] = results.get('K', None)
        self.epsilon: Optional[float] = results.get('epsilon', None)

        # Problem constants (as per target answer)
        self.m = 1.0                # grams
        self.g = 980.0              # cm/s²
        self.h = 6.626e-27          # erg·s (Planck constant)
        self.hbar = self.h / (2.0 * math.pi)  # erg·s (reduced Planck)

    # ----------------------------------------------------------------------
    # Interface tests – required keys exist and have valid type
    # ----------------------------------------------------------------------
    def test_interface_has_alpha(self) -> Tuple[bool, str]:
        """Check that 'alpha' is present and is a number."""
        if self.alpha is None:
            return (False, "SKIP: 'alpha' not provided")
        if not isinstance(self.alpha, (int, float)):
            return (False, f"'alpha' is not numeric: {type(self.alpha)}")
        return (True, f"alpha = {self.alpha} is present and numeric")

    def test_interface_has_K(self) -> Tuple[bool, str]:
        """Check that 'K' is present and is a number."""
        if self.K is None:
            return (False, "SKIP: 'K' not provided")
        if not isinstance(self.K, (int, float)):
            return (False, f"'K' is not numeric: {type(self.K)}")
        return (True, f"K = {self.K} is present and numeric")

    def test_interface_has_epsilon(self) -> Tuple[bool, str]:
        """Check that 'epsilon' is present and is a number."""
        if self.epsilon is None:
            return (False, "SKIP: 'epsilon' not provided")
        if not isinstance(self.epsilon, (int, float)):
            return (False, f"'epsilon' is not numeric: {type(self.epsilon)}")
        return (True, f"epsilon = {self.epsilon} is present and numeric")

    # ----------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_alpha_value(self) -> Tuple[bool, str]:
        """Dimensional analysis gives α = −1/3."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        expected_alpha = -1.0 / 3.0
        if not math.isclose(self.alpha, expected_alpha, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"alpha = {self.alpha}, expected ~{expected_alpha:.4f}")
        return (True, f"alpha = {self.alpha} agrees with dimensional analysis")

    def test_invariant_K_positive(self) -> Tuple[bool, str]:
        """Variational result K must be positive."""
        if self.K is None:
            return (False, "SKIP: K not available")
        if not (self.K > 0):
            return (False, f"K = {self.K} is not positive")
        return (True, f"K = {self.K} > 0")

    def test_invariant_epsilon_positive(self) -> Tuple[bool, str]:
        """Ground state energy must be positive."""
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        if not (self.epsilon > 0):
            return (False, f"epsilon = {self.epsilon} is not positive")
        return (True, f"epsilon = {self.epsilon} > 0")

    # ----------------------------------------------------------------------
    # Modeling/trace tests – check expected relationships
    # ----------------------------------------------------------------------
    def test_modeling_K_expression(self) -> Tuple[bool, str]:
        """K should be 3*(3/(4π))^(1/3) from the variational calculation."""
        if self.K is None:
            return (False, "SKIP: K not available")
        expected_K = 3.0 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        if not math.isclose(self.K, expected_K, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"K = {self.K}, expected ~{expected_K:.6f}")
        return (True, f"K = {self.K} matches variational expression")

    def test_modeling_energy_consistency(self) -> Tuple[bool, str]:
        """
        Check that epsilon = K * m * g * (hbar²/(m² g))^(1/3)  [since α = −1/3].
        Uses Σ = hbar (the reduced Planck constant) because the variational
        calculation naturally involves hbar.
        """
        if None in (self.alpha, self.K, self.epsilon):
            return (False, "SKIP: missing one of alpha, K, epsilon")
        # Re‑derive expected epsilon using the formula from part (a) with hbar
        #   ε = K m g (m² g / hbar²)^α , α = −1/3
        #   = K m g (hbar²/(m² g))^(1/3)
        m = self.m
        g = self.g
        hbar = self.hbar
        expected_eps = self.K * m * g * ((hbar**2) / (m**2 * g)) ** (1.0 / 3.0)
        if not math.isclose(self.epsilon, expected_eps, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"epsilon = {self.epsilon:.3e}, expected from K,α,constants "
                    f"= {expected_eps:.3e} (using hbar)")
        return (True,
                f"epsilon = {self.epsilon:.3e} consistent with K,α,constants")

    # ----------------------------------------------------------------------
    # Monotonicity tests – trend checks by perturbing inputs
    # ----------------------------------------------------------------------
    def _compute_epsilon(self, K: float, m: float, g: float, hbar: float) -> float:
        """Helper: recompute epsilon from K, m, g, hbar using α = −1/3."""
        return K * m * g * ((hbar**2) / (m**2 * g)) ** (1.0 / 3.0)

    def test_monotonicity_epsilon_increases_with_mass(self) -> Tuple[bool, str]:
        """
        For fixed g, hbar, K, epsilon ∝ m^(1/3).  Increasing m should increase epsilon.
        """
        if self.K is None:
            return (False, "SKIP: K not available")
        K = self.K
        g = self.g
        hbar = self.hbar
        m_base = self.m
        m_perturbed = m_base * 1.1   # +10%

        eps_base = self._compute_epsilon(K, m_base, g, hbar)
        eps_pert = self._compute_epsilon(K, m_perturbed, g, hbar)

        if not (eps_pert > eps_base):
            return (False,
                    f"epsilon(m={m_perturbed}) = {eps_pert:.3e} is not > "
                    f"epsilon(m={m_base}) = {eps_base:.3e}")
        return (True,
                f"epsilon increases with mass: {eps_base:.3e} → {eps_pert:.3e}")

    def test_monotonicity_epsilon_increases_with_gravity(self) -> Tuple[bool, str]:
        """
        For fixed m, hbar, K, epsilon ∝ g^(4/3).  Increasing g should increase epsilon.
        """
        if self.K is None:
            return (False, "SKIP: K not available")
        K = self.K
        m = self.m
        hbar = self.hbar
        g_base = self.g
        g_perturbed = g_base * 1.05   # +5%

        eps_base = self._compute_epsilon(K, m, g_base, hbar)
        eps_pert = self._compute_epsilon(K, m, g_perturbed, hbar)

        if not (eps_pert > eps_base):
            return (False,
                    f"epsilon(g={g_perturbed}) = {eps_pert:.3e} is not > "
                    f"epsilon(g={g_base}) = {eps_base:.3e}")
        return (True,
                f"epsilon increases with gravity: {eps_base:.3e} → {eps_pert:.3e}")

    def test_monotonicity_epsilon_decreases_with_Planck(self) -> Tuple[bool, str]:
        """
        For fixed m, g, K, epsilon ∝ hbar^(-2/3).  Increasing hbar (or h) should
        decrease epsilon.  We perturb hbar by +1%.
        """
        if self.K is None:
            return (False, "SKIP: K not available")
        K = self.K
        m = self.m
        g = self.g
        hbar_base = self.hbar
        hbar_perturbed = hbar_base * 1.01   # +1%

        eps_base = self._compute_epsilon(K, m, g, hbar_base)
        eps_pert = self._compute_epsilon(K, m, g, hbar_perturbed)

        if not (eps_pert < eps_base):
            return (False,
                    f"epsilon(hbar={hbar_perturbed:.3e}) = {eps_pert:.3e} is not < "
                    f"epsilon(hbar={hbar_base:.3e}) = {eps_base:.3e}")
        return (True,
                f"epsilon decreases with Planck constant: {eps_base:.3e} → {eps_pert:.3e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one, no isclose, no derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_epsilon_magnitude(self) -> Tuple[bool, str]:
        """
        Very weak sanity: epsilon should be positive and not astronomically large.
        Expected order is ~10⁻¹⁶ erg, so < 10⁻⁶ is a safe bound.
        """
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        if not (0 < self.epsilon < 1e-6):
            return (False, f"epsilon = {self.epsilon:.3e} is outside plausible range "
                           f"(should be positive and ≪ 1e-6 erg)")
        return (True, f"epsilon = {self.epsilon:.3e} has plausible magnitude")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run every test_* method, collect results."""
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