import math
import traceback
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Audit checker for blackbody radiation with heat shields.
    Tests follow ROBUSTNESS rules: safe defaults, fuzzy matching for invariants,
    weak answer sanity, and monotonicity via perturbed recomputation.
    """

    def __init__(self, solution_instance: Any):
        # ------------------------------------------------------------
        # Extract results safely (use None for missing keys)
        # ------------------------------------------------------------
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        self.T_shield = self.results.get('T_shield', None)
        self.P_net = self.results.get('P_net', None)
        self.P_N = self.results.get('P_N', None)

        # ------------------------------------------------------------
        # Extract givens (R, T, sigma, N) from multiple possible sources
        # ------------------------------------------------------------
        self.R = getattr(solution_instance, 'R', None)
        self.T = getattr(solution_instance, 'T', None)
        self.sigma = getattr(solution_instance, 'sigma', None)
        self.N = getattr(solution_instance, 'N', None)

        # Fallback: look for a 'givens' attribute (dict)
        if self.R is None or self.T is None or self.sigma is None or self.N is None:
            givens: Dict = getattr(solution_instance, 'givens', None)
            if givens is not None:
                if self.R is None:
                    self.R = givens.get('R', None)
                if self.T is None:
                    self.T = givens.get('T', None)
                if self.sigma is None:
                    self.sigma = givens.get('sigma', None)
                if self.N is None:
                    self.N = givens.get('N', None)

        # ------------------------------------------------------------
        # Precompute bare sphere power P0 if possible (used in many tests)
        # ------------------------------------------------------------
        self.P0: float = None
        if (self.R is not None and self.T is not None and self.sigma is not None):
            self.P0 = 4.0 * math.pi * self.R ** 2 * self.sigma * self.T ** 4

    # =================================================================
    # 1. Interface tests – ensure required keys and types
    # =================================================================
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all three required result keys exist and are not None."""
        missing = []
        if self.T_shield is None:
            missing.append('T_shield')
        if self.P_net is None:
            missing.append('P_net')
        if self.P_N is None:
            missing.append('P_N')
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check that result values are numeric (int or float)."""
        for name, val in [('T_shield', self.T_shield),
                          ('P_net', self.P_net),
                          ('P_N', self.P_N)]:
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"{name} has type {type(val).__name__}, expected numeric")
        return (True, "All result values are numeric")

    # =================================================================
    # 2. Invariant tests – theorem-based physical sanity (fuzzy matching)
    # =================================================================
    def test_invariant_T_shield_relation(self) -> Tuple[bool, str]:
        """Check that T_shield^4 ≈ T^4 / 2 (Stefan-Boltzmann equilibrium)."""
        if self.T_shield is None or self.T is None:
            return (False, "SKIP: T_shield or T not available")
        T4 = self.T ** 4
        Ts4 = self.T_shield ** 4
        if math.isclose(2 * Ts4, T4, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T_shield^4 * 2 ≈ T^4: {Ts4*2:.4e} ≈ {T4:.4e}")
        else:
            return (False, f"T_shield^4 * 2 = {Ts4*2:.4e}, expected ≈ {T4:.4e}")

    def test_invariant_P_net_relation(self) -> Tuple[bool, str]:
        """Check that P_net ≈ P0 / 2 (with P0 computed from givens)."""
        if self.P_net is None or self.P0 is None:
            return (False, "SKIP: P_net or P0 (needs R, T, sigma) not available")
        expected = self.P0 / 2.0
        if math.isclose(self.P_net, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_net ≈ P0/2: {self.P_net:.4e} ≈ {expected:.4e}")
        else:
            return (False, f"P_net = {self.P_net:.4e}, expected P0/2 = {expected:.4e}")

    def test_invariant_P_N_relation(self) -> Tuple[bool, str]:
        """Check that P_N ≈ P0 / (N+1)."""
        if self.P_N is None or self.P0 is None or self.N is None:
            return (False, "SKIP: P_N, P0, or N not available")
        expected = self.P0 / (self.N + 1)
        if math.isclose(self.P_N, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_N ≈ P0/(N+1): {self.P_N:.4e} ≈ {expected:.4e}")
        else:
            return (False, f"P_N = {self.P_N:.4e}, expected P0/(N+1) = {expected:.4e}")

    # =================================================================
    # 3. Modeling / trace tests – quality of the physical equations
    # =================================================================
    def test_modeling_shield_equilibrium(self) -> Tuple[bool, str]:
        """
        Check the radiative equilibrium equation for one shield:
        4πR^2σ(T^4 - T_shield^4)  ≈  4πR^2σ T_shield^4
        Equivalent to 2*T_shield^4 ≈ T^4, but we write it explicitly.
        """
        if self.T_shield is None or self.T is None or self.R is None or self.sigma is None:
            return (False, "SKIP: missing needed givens (T_shield, T, R, sigma)")
        area = 4.0 * math.pi * self.R ** 2
        absorbed = area * self.sigma * (self.T ** 4 - self.T_shield ** 4)
        emitted = area * self.sigma * self.T_shield ** 4
        if math.isclose(absorbed, emitted, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Absorbed ≈ emitted: {absorbed:.4e} ≈ {emitted:.4e}")
        else:
            return (False, f"Absorbed = {absorbed:.4e}, emitted = {emitted:.4e}, not equal")

    def test_modeling_N_shields_series_consistency(self) -> Tuple[bool, str]:
        """
        For N shields, the net power to surroundings is P0/(N+1),
        which follows from the series equilibrium of temperatures.
        We check that P_N = P0/(N+1) using the theoretical formula.
        """
        if self.P_N is None or self.P0 is None or self.N is None:
            return (False, "SKIP: P_N, P0, or N not available")
        if self.N < 1:
            return (False, f"N = {self.N} is not a positive integer")
        expected = self.P0 / (self.N + 1)
        if math.isclose(self.P_N, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_N ≈ P0/(N+1) for N={self.N}: {self.P_N:.4e} ≈ {expected:.4e}")
        else:
            return (False, f"P_N = {self.P_N:.4e}, expected P0/(N+1) = {expected:.4e}")

    # =================================================================
    # 4. Monotonicity tests – perturb an input and compare output
    # =================================================================
    def test_monotonicity_P_net_increases_with_T(self) -> Tuple[bool, str]:
        """
        Perturb T by +1% and recompute P_net using the formula P_net = P0/2.
        The perturbed value should be strictly larger.
        """
        if self.P_net is None or self.T is None or self.R is None or self.sigma is None:
            return (False, "SKIP: missing data for monotonicity (P_net, T, R, sigma)")
        current_P0 = 4.0 * math.pi * self.R ** 2 * self.sigma * self.T ** 4
        current_Pnet = current_P0 / 2.0

        new_T = self.T * 1.01
        new_P0 = 4.0 * math.pi * self.R ** 2 * self.sigma * new_T ** 4
        new_Pnet = new_P0 / 2.0

        if new_Pnet > current_Pnet:
            return (True, f"P_net increases with T: {new_Pnet:.4e} > {current_Pnet:.4e}")
        else:
            return (False, f"P_net did not increase: {new_Pnet:.4e} <= {current_Pnet:.4e}")

    def test_monotonicity_P_N_increases_with_T(self) -> Tuple[bool, str]:
        """
        Perturb T by +1% and recompute P_N using the formula P_N = P0/(N+1).
        The perturbed value should be strictly larger.
        """
        if self.P_N is None or self.T is None or self.R is None or self.sigma is None or self.N is None:
            return (False, "SKIP: missing data for monotonicity (P_N, T, R, sigma, N)")
        current_P0 = 4.0 * math.pi * self.R ** 2 * self.sigma * self.T ** 4
        current_PN = current_P0 / (self.N + 1)

        new_T = self.T * 1.01
        new_P0 = 4.0 * math.pi * self.R ** 2 * self.sigma * new_T ** 4
        new_PN = new_P0 / (self.N + 1)

        if new_PN > current_PN:
            return (True, f"P_N increases with T: {new_PN:.4e} > {current_PN:.4e}")
        else:
            return (False, f"P_N did not increase: {new_PN:.4e} <= {current_PN:.4e}")

    # =================================================================
    # 5. Answer sanity – one weak method (no re‑derivation, no isclose)
    # =================================================================
    def test_answer_sanity_T_shield_and_P_net_basic(self) -> Tuple[bool, str]:
        """
        Weak checks: T_shield positive and if T known, T_shield < T;
        P_net positive and finite.
        These are broad sanity checks, not re‑derived values.
        """
        # Check T_shield
        if self.T_shield is not None:
            if not (self.T_shield > 0):
                return (False, f"T_shield is not positive: {self.T_shield}")
            if self.T is not None and not (self.T_shield < self.T):
                return (False, f"T_shield ({self.T_shield}) is not < T ({self.T})")
        else:
            return (False, "SKIP: T_shield not available")

        # Check P_net
        if self.P_net is not None:
            if not (self.P_net > 0 and math.isfinite(self.P_net)):
                return (False, f"P_net is not a positive finite number: {self.P_net}")
        else:
            return (False, "SKIP: P_net not available")

        return (True, "T_shield and P_net pass basic sanity checks")

    # =================================================================
    # Main runner – discover and execute all test_ methods
    # =================================================================
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)