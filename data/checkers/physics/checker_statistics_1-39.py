import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Verifies physical consistency and modeling quality for the heat pump / heater problem.
    
    Extracts inputs (T0, alpha, W) and results (T_e, T_prime_e) from a solution instance.
    Runs multiple small tests: interface, invariants, modeling, monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract givens – use None if missing
        self.T0 = getattr(solution_instance, 'T0', None)
        self.alpha = getattr(solution_instance, 'alpha', None)
        self.W = getattr(solution_instance, 'W', None)

        # Safely extract results – prefer dict access, fallback to attributes
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.T_e = solution_instance.results.get('T_e', None)
            self.T_prime_e = solution_instance.results.get('T_prime_e', None)
        else:
            self.T_e = getattr(solution_instance, 'T_e', None)
            self.T_prime_e = getattr(solution_instance, 'T_prime_e', None)

        # Optional explanation_steps for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------------------- Interface tests ----------------------
    def test_interface_required_results(self) -> Tuple[bool, str]:
        """Check that T_e and T_prime_e exist and are numeric."""
        if self.T_e is None or self.T_prime_e is None:
            return (False, "Missing required result(s): T_e or T_prime_e")
        if not isinstance(self.T_e, (int, float)):
            return (False, f"T_e is not numeric: {type(self.T_e).__name__}")
        if not isinstance(self.T_prime_e, (int, float)):
            return (False, f"T_prime_e is not numeric: {type(self.T_prime_e).__name__}")
        return (True, "Required results present and numeric")

    def test_interface_givens_positive(self) -> Tuple[bool, str]:
        """Ensure all given parameters are positive."""
        for name, val in [('T0', self.T0), ('alpha', self.alpha), ('W', self.W)]:
            if val is None:
                return (False, f"Missing given: {name}")
            if not isinstance(val, (int, float)):
                return (False, f"Given {name} is not numeric: {type(val).__name__}")
            if val <= 0:
                return (False, f"Given {name} must be positive, got {val}")
        return (True, "All givens positive")

    # ---------------------- Invariant tests (theorem-based, use math.isclose) ----------------------
    def test_invariant_energy_balance_heatpump(self) -> Tuple[bool, str]:
        """Check: W * T_e / (T_e - T0) ≈ α (T_e - T0)."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_e is None:
            return (False, "SKIP: T_e not available")
        if self.T_e <= self.T0:
            return (False, f"T_e ({self.T_e}) must be > T0 ({self.T0}) for meaningful COP")
        delta = self.T_e - self.T0
        if delta == 0:
            return (False, "Zero delta, cannot compute heat pump balance")
        left = self.W * self.T_e / delta
        right = self.alpha * delta
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Heat pump energy balance failed: left={left}, right={right}")
        return (True, "Heat pump energy balance satisfied")

    def test_invariant_energy_balance_heater(self) -> Tuple[bool, str]:
        """Check: W ≈ α (T_prime_e - T0)."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_prime_e is None:
            return (False, "SKIP: T_prime_e not available")
        if self.T_prime_e <= self.T0:
            return (False, f"T_prime_e ({self.T_prime_e}) must be > T0 ({self.T0})")
        delta = self.T_prime_e - self.T0
        right = self.alpha * delta
        if not math.isclose(self.W, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Heater energy balance failed: W={self.W}, αΔT={right}")
        return (True, "Heater energy balance satisfied")

    def test_invariant_temperature_order(self) -> Tuple[bool, str]:
        """Check T_e > T_prime_e > T0."""
        if self.T_e is None or self.T_prime_e is None or self.T0 is None:
            return (False, "SKIP: missing temperatures")
        if self.T_e <= self.T_prime_e:
            return (False, f"T_e ({self.T_e}) should be > T_prime_e ({self.T_prime_e})")
        if self.T_prime_e <= self.T0:
            return (False, f"T_prime_e ({self.T_prime_e}) should be > T0 ({self.T0})")
        return (True, "Temperature order correct: T_e > T_prime_e > T0")

    # ---------------------- Modeling / trace tests (check explicit formulas re-derived) ----------------------
    def test_modeling_quadratic_relation(self) -> Tuple[bool, str]:
        """Verify that T_e satisfies the quadratic: α (T_e - T0)^2 = W T_e."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_e is None:
            return (False, "SKIP: T_e not available")
        delta = self.T_e - self.T0
        left = self.alpha * delta ** 2
        right = self.W * self.T_e
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Quadratic relation not satisfied: αΔ²={left}, W T_e={right}")
        return (True, "Quadratic relation satisfied")

    def test_modeling_heatpump_formula(self) -> Tuple[bool, str]:
        """Check that T_e matches the explicit formula T0 + (W/(2α))[1 + sqrt(1 + 4αT0/W)] (re-derived)."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_e is None:
            return (False, "SKIP: T_e not available")
        # Re-derive expected T_e from formula
        expected = self.T0 + (self.W / (2 * self.alpha)) * (1 + math.sqrt(1 + 4 * self.alpha * self.T0 / self.W))
        if not math.isclose(self.T_e, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_e {self.T_e} does not match re-derived formula {expected}")
        return (True, "T_e matches explicit formula")

    def test_modeling_heater_formula(self) -> Tuple[bool, str]:
        """Check that T_prime_e matches the linear formula T0 + W/α (re-derived)."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_prime_e is None:
            return (False, "SKIP: T_prime_e not available")
        expected = self.T0 + self.W / self.alpha
        if not math.isclose(self.T_prime_e, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_prime_e {self.T_prime_e} does not match re-derived formula {expected}")
        return (True, "T_prime_e matches linear formula")

    # ---------------------- Monotonicity tests (explicit perturbation) ----------------------
    def test_monotonicity_T_e_with_W(self) -> Tuple[bool, str]:
        """Increasing W should increase T_e."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_e is None:
            return (False, "SKIP: T_e not available")
        W2 = self.W * 1.1
        # Solve quadratic α (T - T0)^2 = W2 * T  => α δ^2 - W2 δ - W2 T0 = 0
        a = self.alpha
        b = -W2
        c = -W2 * self.T0
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return (False, "Perturbed discriminant negative")
        # Positive delta root: (-b + sqrt(d))/(2a)
        delta_new = (-b + math.sqrt(discriminant)) / (2*a)
        T_e_new = self.T0 + delta_new
        if T_e_new <= self.T_e:
            return (False, f"Monotonicity failed: T_e did not increase (old={self.T_e}, new={T_e_new})")
        return (True, f"Monotonicity with W: T_e increased (old={self.T_e}, new={T_e_new})")

    def test_monotonicity_T_prime_e_with_W(self) -> Tuple[bool, str]:
        """Increasing W should increase T_prime_e linearly."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_prime_e is None:
            return (False, "SKIP: T_prime_e not available")
        W2 = self.W * 1.1
        T_prime_new = self.T0 + W2 / self.alpha
        if T_prime_new <= self.T_prime_e:
            return (False, f"Monotonicity failed: T_prime_e did not increase (old={self.T_prime_e}, new={T_prime_new})")
        return (True, f"Monotonicity with W: T_prime_e increased (old={self.T_prime_e}, new={T_prime_new})")

    def test_monotonicity_T_e_with_alpha(self) -> Tuple[bool, str]:
        """Increasing α should decrease T_e (since heat loss increases)."""
        if self.T0 is None or self.alpha is None or self.W is None:
            return (False, "SKIP: missing givens")
        if self.T_e is None:
            return (False, "SKIP: T_e not available")
        alpha2 = self.alpha * 1.1
        # Solve quadratic with new alpha
        a = alpha2
        b = -self.W
        c = -self.W * self.T0
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return (False, "Perturbed discriminant negative")
        delta_new = (-b + math.sqrt(discriminant)) / (2*a)
        T_e_new = self.T0 + delta_new
        if T_e_new >= self.T_e:   # Expect lower T_e with larger α
            return (False, f"Monotonicity failed: T_e did not decrease with α increase (old={self.T_e}, new={T_e_new})")
        return (True, f"Monotonicity with α: T_e decreased as expected (old={self.T_e}, new={T_e_new})")

    # ---------------------- Weak answer sanity (no isclose, no re-derivation) ----------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Verify that temperatures are finite, positive, and within a broad range."""
        for name, val in [('T_e', self.T_e), ('T_prime_e', self.T_prime_e)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")
            if val <= 0:
                return (False, f"{name} must be positive, got {val}")
            if val > 1e9:
                return (False, f"{name} is unreasonably large: {val}")
        return (True, "Temperature values are finite, positive, and within reasonable range")

    # ---------------------- Main runner ----------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, run each in try/except, and aggregate results."""
        logs: List[str] = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED with {type(e).__name__}: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)