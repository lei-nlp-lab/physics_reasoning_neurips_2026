import math
import sympy as sp
from typing import Any, List, Tuple, Union

class PhysicsChecker:
    """
    PhysicsChecker for U-235 half-life order-of-magnitude problem.
    Audits solution_instance for correct half-life value and order magnitude.
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction: attribute, results dict, or None
        self.half_life_U235 = self._safe_extract(solution_instance, 'half_life_U235')
        self.order = self._safe_extract(solution_instance, 'order')

    @staticmethod
    def _safe_extract(inst: Any, key: str) -> Any:
        """Extract value from attribute or results dict, defaulting to None."""
        val = getattr(inst, key, None)
        if val is not None:
            return val
        if hasattr(inst, 'results') and isinstance(inst.results, dict):
            val = inst.results.get(key, None)
            if val is not None:
                return val
        if hasattr(inst, 'result') and isinstance(inst.result, dict):
            val = inst.result.get(key, None)
            if val is not None:
                return val
        return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that both expected result keys exist and are not None."""
        if self.half_life_U235 is None:
            return (False, "half_life_U235 is missing or None")
        if self.order is None:
            return (False, "order is missing or None")
        return (True, "Both result keys present")

    def test_interface_types_half_life(self) -> Tuple[bool, str]:
        """half_life_U235 should be numeric (int or float)."""
        if self.half_life_U235 is None:
            return (False, "SKIP: half_life_U235 not available")
        if not isinstance(self.half_life_U235, (int, float)):
            return (False, f"half_life_U235 has type {type(self.half_life_U235).__name__}, expected numeric")
        return (True, "half_life_U235 is numeric")

    def test_interface_types_order(self) -> Tuple[bool, str]:
        """order should be numeric or string representing a power of 10 (e.g., '10^9')."""
        if self.order is None:
            return (False, "SKIP: order not available")
        if isinstance(self.order, (int, float)):
            return (True, "order is numeric")
        if isinstance(self.order, str) and '10^' in self.order:
            return (True, "order is string power-of-10 representation")
        return (False, f"order has type {type(self.order).__name__} and doesn't look like a power-of-10")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_positive_half_life(self) -> Tuple[bool, str]:
        """Half-life must be positive and finite."""
        if self.half_life_U235 is None:
            return (False, "SKIP: half_life_U235 not available")
        hl = self.half_life_U235
        if not (isinstance(hl, (int, float)) and math.isfinite(hl)):
            return (False, "half_life_U235 is not a finite number")
        if hl <= 0:
            return (False, f"half_life_U235 is {hl}, must be positive")
        return (True, "half_life_U235 is positive and finite")

    def test_invariant_order_positive(self) -> Tuple[bool, str]:
        """Order must be positive and finite (if numeric)."""
        if self.order is None:
            return (False, "SKIP: order not available")
        ord_val = self.order
        if isinstance(ord_val, (int, float)):
            if not (math.isfinite(ord_val) and ord_val > 0):
                return (False, f"order is {ord_val}, must be positive")
            return (True, "order is positive numeric")
        if isinstance(ord_val, str):
            try:
                exponent = int(ord_val.split('^')[1])
                if exponent <= 0:
                    return (False, f"order string has non-positive exponent {exponent}")
                return (True, "order string represents positive exponent")
            except:
                return (False, "order string could not be parsed")
        return (True, "order is acceptable")

    def test_invariant_known_half_life_range(self) -> Tuple[bool, str]:
        """Half-life of U-235 is approximately 7.04e8 years (within 10% tolerance)."""
        if self.half_life_U235 is None:
            return (False, "SKIP: half_life_U235 not available")
        hl = self.half_life_U235
        known = 7.04e8
        if not math.isclose(hl, known, rel_tol=0.1):
            return (False, f"half_life_U235 = {hl}, expected approximately {known}")
        return (True, "half_life_U235 within 10% of known value")

    def test_invariant_order_consistent_with_half_life(self) -> Tuple[bool, str]:
        """Order must be the nearest power of 10 to the half-life value."""
        if self.half_life_U235 is None or self.order is None:
            return (False, "SKIP: half_life or order missing")
        hl = self.half_life_U235
        ord_val = self._order_to_numeric()
        if ord_val is None:
            return (False, "order could not be converted to numeric")
        if hl <= 0:
            return (False, "half_life_U235 is non-positive")
        log10_hl = math.log10(hl)
        nearest_power = 10 ** round(log10_hl)
        if not math.isclose(ord_val, nearest_power, rel_tol=0.1):
            return (False, f"order {ord_val} does not match nearest power of 10 {nearest_power} from half-life {hl}")
        return (True, "order is consistent with half-life value")

    def test_invariant_order_is_reasonable(self) -> Tuple[bool, str]:
        """Order should be one of the plausible half-life orders (10^8, 10^9, 10^10)."""
        if self.order is None:
            return (False, "SKIP: order not available")
        ord_val = self._order_to_numeric()
        if ord_val is None:
            return (False, "order could not be interpreted as number")
        plausible = [1e8, 1e9, 1e10]
        if not any(math.isclose(ord_val, p, rel_tol=1e-3) for p in plausible):
            return (False, f"order {ord_val} is not a plausible half-life order (expected ~1e8-1e10)")
        return (True, "order is within plausible range")

    # ------------------------------------------------------------------
    # Modeling/Trace consistency tests (at least two)
    # ------------------------------------------------------------------
    def test_modeling_accurate_half_life(self) -> Tuple[bool, str]:
        """Half-life must be close to the known value (1% tolerance) to indicate accurate modeling."""
        if self.half_life_U235 is None:
            return (False, "SKIP: half_life_U235 not available")
        hl = self.half_life_U235
        known = 7.04e8
        if not math.isclose(hl, known, rel_tol=0.01):
            return (False, f"half_life_U235 = {hl} deviates more than 1% from known value {known}")
        return (True, "half_life_U235 is accurate (within 1% of known value)")

    def test_modeling_exponent_consistency(self) -> Tuple[bool, str]:
        """The exponent of the order (log10 of numeric order) must equal round(log10(half_life))."""
        if self.half_life_U235 is None or self.order is None:
            return (False, "SKIP: half_life or order missing")
        hl = self.half_life_U235
        ord_val = self._order_to_numeric()
        if ord_val is None:
            return (False, "order could not be converted to numeric")
        if hl <= 0:
            return (False, "half_life_U235 is non-positive")
        # Expected exponent: integer nearest to log10(hl)
        expected_exponent = round(math.log10(hl))
        # Actual exponent: log10 of order (which is a power of 10, so exponent is integral)
        actual_exponent = round(math.log10(ord_val))
        if actual_exponent != expected_exponent:
            return (False, f"Order exponent {actual_exponent} does not match expected {expected_exponent} from half-life")
        return (True, f"Order exponent {actual_exponent} matches half-life log10 rounding")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_increasing_half_life_does_not_decrease_order(self) -> Tuple[bool, str]:
        """If half-life increases, the order of magnitude (nearest power of 10) should not decrease."""
        if self.half_life_U235 is None or self.order is None:
            return (False, "SKIP: half_life or order missing")
        hl = self.half_life_U235
        order_numeric = self._order_to_numeric()
        if order_numeric is None:
            return (False, "order could not be converted to numeric")
        # Perturb half-life upward by 10%
        hl_pert = hl * 1.1
        if hl_pert <= 0:
            return (False, "Perturbed half-life became non-positive")
        log10_pert = math.log10(hl_pert)
        expected_order = 10 ** round(log10_pert)
        # Order should not decrease (monotonic non-decreasing)
        if expected_order < order_numeric:
            return (False, f"Perturbation (+10%) gives order {expected_order} < original {order_numeric}, violates monotonicity")
        return (True, f"Monotonicity holds: increased half-life does not decrease order")

    def test_monotonicity_decreasing_half_life_does_not_increase_order(self) -> Tuple[bool, str]:
        """If half-life decreases, the order should not increase."""
        if self.half_life_U235 is None or self.order is None:
            return (False, "SKIP: half_life or order missing")
        hl = self.half_life_U235
        order_numeric = self._order_to_numeric()
        if order_numeric is None:
            return (False, "order could not be converted to numeric")
        # Perturb half-life downward by 10%
        hl_pert = hl * 0.9
        if hl_pert <= 0:
            return (False, "Perturbed half-life became non-positive")
        log10_pert = math.log10(hl_pert)
        expected_order = 10 ** round(log10_pert)
        # Order should not increase
        if expected_order > order_numeric:
            return (False, f"Perturbation (-10%) gives order {expected_order} > original {order_numeric}, violates monotonicity")
        return (True, f"Monotonicity holds: decreased half-life does not increase order")

    # ------------------------------------------------------------------
    # Answer sanity (weak)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """Basic sanity: both values exist, finite, and non-NaN (no re-derivation)."""
        if self.half_life_U235 is None or self.order is None:
            return (False, "Required values missing")
        hl = self.half_life_U235
        ord_val = self.order
        if isinstance(hl, (int, float)):
            if not math.isfinite(hl):
                return (False, "half_life_U235 is not finite")
        if isinstance(ord_val, (int, float)):
            if not math.isfinite(ord_val):
                return (False, "order is not finite")
        return (True, "Basic answer sanity passed (values present and finite)")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _order_to_numeric(self) -> Union[float, None]:
        """Convert order to a numeric value if possible. Returns None otherwise."""
        if self.order is None:
            return None
        if isinstance(self.order, (int, float)):
            return float(self.order)
        if isinstance(self.order, str):
            try:
                return float(self.order)
            except ValueError:
                if '10^' in self.order:
                    try:
                        exponent = self.order.split('10^')[1]
                        return 10 ** int(exponent)
                    except:
                        pass
        return None

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, run them, and collect logs."""
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
                passed = False
                msg = f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)