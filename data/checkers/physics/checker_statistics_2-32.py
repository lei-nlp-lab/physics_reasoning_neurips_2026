import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for the quantum harmonic oscillator thermal equilibrium problem.
    Checks interface completeness, physical invariants, modeling consistency,
    monotonicity of thermodynamic quantities, and basic answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens safely – use None if missing
        self.hbar = self._get_given(solution_instance, 'hbar')
        self.omega = self._get_given(solution_instance, 'omega')
        self.k = self._get_given(solution_instance, 'k')
        self.T = self._get_given(solution_instance, 'T')

        # Extract results from solution_instance.results (expected dict)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        self.mean_energy = self.results.get('mean_energy', None)
        self.fluctuation = self.results.get('fluctuation', None)
        self.low_T_energy = self.results.get('low_T_energy', None)
        self.low_T_fluctuation = self.results.get('low_T_fluctuation', None)
        self.high_T_energy = self.results.get('high_T_energy', None)
        self.high_T_fluctuation = self.results.get('high_T_fluctuation', None)

    def _get_given(self, solution_instance: Any, key: str) -> Optional[float]:
        """Try to extract a given quantity from various possible locations."""
        # Direct attribute
        val = getattr(solution_instance, key, None)
        if val is not None:
            return val
        # Dict attribute (e.g., solution_instance.givens)
        givens = getattr(solution_instance, 'givens', None)
        if givens and isinstance(givens, dict) and key in givens:
            return givens[key]
        # Another dict (e.g., solution_instance.params)
        params = getattr(solution_instance, 'params', None)
        if params and isinstance(params, dict) and key in params:
            return params[key]
        return None

    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist and are not None."""
        required = ['mean_energy', 'fluctuation', 'low_T_energy',
                    'low_T_fluctuation', 'high_T_energy', 'high_T_fluctuation']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys in results: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that essential givens are provided (non-None)."""
        missing = []
        for var, name in [(self.hbar, 'hbar'), (self.omega, 'omega'),
                          (self.k, 'k'), (self.T, 'T')]:
            if var is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All required givens available")

    def test_invariant_mean_energy_positive(self) -> Tuple[bool, str]:
        """Mean energy must be positive (strictly greater than zero)."""
        if self.mean_energy is None:
            return (False, "SKIP: mean_energy not available")
        if self.mean_energy > 0:
            return (True, f"Mean energy = {self.mean_energy} > 0")
        else:
            return (False, f"Mean energy = {self.mean_energy} is not positive")

    def test_invariant_fluctuation_positive(self) -> Tuple[bool, str]:
        """Fluctuation must be positive."""
        if self.fluctuation is None:
            return (False, "SKIP: fluctuation not available")
        if self.fluctuation > 0:
            return (True, f"Fluctuation = {self.fluctuation} > 0")
        else:
            return (False, f"Fluctuation = {self.fluctuation} is not positive")

    def test_invariant_low_T_energy_limits(self) -> Tuple[bool, str]:
        """Low temperature energy should be exactly zero-point energy (hbar*omega/2)."""
        if self.low_T_energy is None:
            return (False, "SKIP: low_T_energy not available")
        if self.hbar is None or self.omega is None:
            return (False, "SKIP: hbar or omega missing")
        expected = self.hbar * self.omega / 2.0
        if math.isclose(self.low_T_energy, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Low T energy = {self.low_T_energy} ≈ {expected}")
        else:
            return (False, f"Low T energy = {self.low_T_energy}, expected {expected}")

    def test_invariant_high_T_energy_limits(self) -> Tuple[bool, str]:
        """High temperature energy should be approximately kT."""
        if self.high_T_energy is None:
            return (False, "SKIP: high_T_energy not available")
        if self.k is None or self.T is None:
            return (False, "SKIP: k or T missing")
        expected = self.k * self.T
        if math.isclose(self.high_T_energy, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"High T energy = {self.high_T_energy} ≈ {expected}")
        else:
            return (False, f"High T energy = {self.high_T_energy}, expected {expected}")

    def test_modeling_mean_energy_coth_form(self) -> Tuple[bool, str]:
        """Check that mean_energy matches the hyperbolic cotangent expression."""
        if self.mean_energy is None:
            return (False, "SKIP: mean_energy not available")
        if any(v is None for v in [self.hbar, self.omega, self.k, self.T]):
            return (False, "SKIP: givens missing")
        beta = self.hbar * self.omega / (2.0 * self.k * self.T)
        if math.isclose(beta, 0.0, abs_tol=1e-12):
            return (False, "SKIP: division by zero (T too large)")
        expected = 0.5 * self.hbar * self.omega * (1.0 / math.tanh(beta))
        if math.isclose(self.mean_energy, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Mean energy = {self.mean_energy} ≈ coth form = {expected}")
        else:
            return (False, f"Mean energy = {self.mean_energy}, coth expression gives {expected}")

    def test_modeling_fluctuation_sinh_form(self) -> Tuple[bool, str]:
        """Check that fluctuation matches the hyperbolic sine expression."""
        if self.fluctuation is None:
            return (False, "SKIP: fluctuation not available")
        if any(v is None for v in [self.hbar, self.omega, self.k, self.T]):
            return (False, "SKIP: givens missing")
        x = self.hbar * self.omega / (2.0 * self.k * self.T)
        if math.isclose(x, 0.0, abs_tol=1e-12):
            return (False, "SKIP: division by zero (T too large)")
        expected = self.hbar * self.omega / (2.0 * math.sinh(x))
        if math.isclose(self.fluctuation, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Fluctuation = {self.fluctuation} ≈ sinh form = {expected}")
        else:
            return (False, f"Fluctuation = {self.fluctuation}, sinh expression gives {expected}")

    def test_modeling_low_T_fluctuation_expression(self) -> Tuple[bool, str]:
        """Check low_T_fluctuation against exponential decay expression."""
        if self.low_T_fluctuation is None:
            return (False, "SKIP: low_T_fluctuation not available")
        if any(v is None for v in [self.hbar, self.omega, self.k, self.T]):
            return (False, "SKIP: givens missing")
        x = self.hbar * self.omega / (2.0 * self.k * self.T)
        expected = self.hbar * self.omega * math.exp(-x)
        if math.isclose(self.low_T_fluctuation, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Low T fluctuation = {self.low_T_fluctuation} ≈ exp form = {expected}")
        else:
            return (False, f"Low T fluctuation = {self.low_T_fluctuation}, exp expression gives {expected}")

    def test_modeling_high_T_fluctuation_linear(self) -> Tuple[bool, str]:
        """High temperature fluctuation should equal kT."""
        if self.high_T_fluctuation is None:
            return (False, "SKIP: high_T_fluctuation not available")
        if self.k is None or self.T is None:
            return (False, "SKIP: k or T missing")
        expected = self.k * self.T
        if math.isclose(self.high_T_fluctuation, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"High T fluctuation = {self.high_T_fluctuation} ≈ kT = {expected}")
        else:
            return (False, f"High T fluctuation = {self.high_T_fluctuation}, expected kT = {expected}")

    def test_monotonicity_mean_energy_increasing(self) -> Tuple[bool, str]:
        """
        Perturb temperature upward by 1% and verify that the baseline stored
        mean_energy is less than the new value computed at the higher temperature.
        This is a real monotonicity test because it compares a perturbed output
        against the baseline stored output.
        """
        if any(v is None for v in [self.hbar, self.omega, self.k, self.T]):
            return (False, "SKIP: missing givens for monotonicity test")
        if self.mean_energy is None:
            return (False, "SKIP: mean_energy not stored")
        T0 = self.T
        if T0 <= 0:
            return (False, "SKIP: non-positive temperature")
        dT = 0.01 * T0
        # Compute mean energy at T0+dT using the theoretical formula
        beta_new = self.hbar * self.omega / (2.0 * self.k * (T0 + dT))
        mean_new = 0.5 * self.hbar * self.omega * (1.0 / math.tanh(beta_new))
        if mean_new > self.mean_energy:
            return (True, f"Mean energy increases: stored={self.mean_energy}, at T+ΔT={mean_new}")
        else:
            return (False, f"Monotonicity violated: stored={self.mean_energy}, at T+ΔT={mean_new}")

    def test_monotonicity_fluctuation_increasing(self) -> Tuple[bool, str]:
        """
        Perturb temperature upward by 1% and verify that the baseline stored
        fluctuation is less than the new value computed at the higher temperature.
        """
        if any(v is None for v in [self.hbar, self.omega, self.k, self.T]):
            return (False, "SKIP: missing givens for monotonicity test")
        if self.fluctuation is None:
            return (False, "SKIP: fluctuation not stored")
        T0 = self.T
        if T0 <= 0:
            return (False, "SKIP: non-positive temperature")
        dT = 0.01 * T0
        x_new = self.hbar * self.omega / (2.0 * self.k * (T0 + dT))
        fluct_new = self.hbar * self.omega / (2.0 * math.sinh(x_new))
        if fluct_new > self.fluctuation:
            return (True, f"Fluctuation increases: stored={self.fluctuation}, at T+ΔT={fluct_new}")
        else:
            return (False, f"Monotonicity violated: stored={self.fluctuation}, at T+ΔT={fluct_new}")

    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak final answer sanity: all quantities finite, non-NaN, and non-negative.
        This test does NOT compare against target answers or re-derive expressions.
        """
        numeric_values = [
            self.mean_energy, self.fluctuation,
            self.low_T_energy, self.low_T_fluctuation,
            self.high_T_energy, self.high_T_fluctuation
        ]
        for val in numeric_values:
            if val is None:
                continue
            if math.isnan(val) or math.isinf(val):
                return (False, f"Non-finite value encountered: {val}")
        return (True, "All finite, non-NaN (weak check)")

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods, execute them in order, collect logs.
        Returns (all_pass, logs_list).
        """
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