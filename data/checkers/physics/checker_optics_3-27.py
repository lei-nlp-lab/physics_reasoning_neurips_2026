import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits the estimated luminous efficiencies for incandescent bulb and LED.
    Checks key existence, physical bounds, modeling consistency with given factors,
    monotonicity of the model, and weak final-answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract results and givens safely. All defaults are None.
        """
        # Results from [emit] contract
        res = getattr(solution_instance, 'results', None)
        self.eta_inc = res.get('eta_inc', None) if res else None
        self.eta_LED = res.get('eta_LED', None) if res else None

        # Optional givens from pseudocode (may be stored in solution_instance)
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = getattr(solution_instance, 'params', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Extract individual givens if they exist
        self.P_rad_frac_inc = self.givens.get('P_rad_frac_inc', None)
        self.f_vis_inc = self.givens.get('f_vis_inc', None)
        self.eta_int_LED = self.givens.get('eta_int_LED', None)
        self.eta_ext_LED = self.givens.get('eta_ext_LED', None)
        self.eta_ph_LED = self.givens.get('eta_ph_LED', None)

    # ---------------------------------------------------------------
    # Interface checks – required keys exist and are numeric
    # ---------------------------------------------------------------
    def test_interface_eta_inc_exists(self) -> Tuple[bool, str]:
        if self.eta_inc is None:
            return False, "Missing result key 'eta_inc'"
        if not isinstance(self.eta_inc, (int, float)):
            return False, f"eta_inc must be numeric, got {type(self.eta_inc)}"
        return True, "eta_inc present and numeric"

    def test_interface_eta_LED_exists(self) -> Tuple[bool, str]:
        if self.eta_LED is None:
            return False, "Missing result key 'eta_LED'"
        if not isinstance(self.eta_LED, (int, float)):
            return False, f"eta_LED must be numeric, got {type(self.eta_LED)}"
        return True, "eta_LED present and numeric"

    # ---------------------------------------------------------------
    # Invariant tests – physical bounds (0 ≤ η ≤ 1)
    # ---------------------------------------------------------------
    def test_invariant_eta_bounds(self) -> Tuple[bool, str]:
        if self.eta_inc is None and self.eta_LED is None:
            return (False, "SKIP: no efficiency data available")
        messages = []
        ok = True
        if self.eta_inc is not None:
            if not (0 <= self.eta_inc <= 1):
                ok = False
                messages.append(f"eta_inc={self.eta_inc} not in [0,1]")
            else:
                messages.append("eta_inc in [0,1]")
        if self.eta_LED is not None:
            if not (0 <= self.eta_LED <= 1):
                ok = False
                messages.append(f"eta_LED={self.eta_LED} not in [0,1]")
            else:
                messages.append("eta_LED in [0,1]")
        return ok, "; ".join(messages)

    def test_invariant_inc_less_than_LED(self) -> Tuple[bool, str]:
        """Typical incandescent efficiency (~2%) is lower than LED (~25%)"""
        if self.eta_inc is None or self.eta_LED is None:
            return (False, "SKIP: both efficiencies needed")
        if self.eta_inc >= self.eta_LED:
            return False, f"eta_inc={self.eta_inc} is not less than eta_LED={self.eta_LED}"
        return True, f"eta_inc < eta_LED holds"

    # ---------------------------------------------------------------
    # Modeling checks – product consistency from pseudocode
    # ---------------------------------------------------------------
    def test_modeling_eta_inc_product(self) -> Tuple[bool, str]:
        """If givens for incandescent are present, check eta_inc ≈ P_rad_frac * f_vis"""
        if self.P_rad_frac_inc is None or self.f_vis_inc is None:
            return (False, "SKIP: needed givens (P_rad_frac_inc, f_vis_inc) not provided")
        if self.eta_inc is None:
            return (False, "SKIP: eta_inc not available")
        expected = self.P_rad_frac_inc * self.f_vis_inc
        if not math.isclose(self.eta_inc, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"eta_inc={self.eta_inc} differs from product {expected:.4f}"
        return True, f"eta_inc ≈ product of given factors"

    def test_modeling_eta_LED_product(self) -> Tuple[bool, str]:
        """If givens for LED are present, check eta_LED ≈ product of three factors"""
        if any(g is None for g in [self.eta_int_LED, self.eta_ext_LED, self.eta_ph_LED]):
            return (False, "SKIP: needed LED givens not provided")
        if self.eta_LED is None:
            return (False, "SKIP: eta_LED not available")
        expected = self.eta_int_LED * self.eta_ext_LED * self.eta_ph_LED
        if not math.isclose(self.eta_LED, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"eta_LED={self.eta_LED} differs from product {expected:.4f}"
        return True, f"eta_LED ≈ product of given factors"

    # ---------------------------------------------------------------
    # Monotonicity tests – perturb an input and check direction
    # ---------------------------------------------------------------
    def test_monotonicity_inc_with_rad_frac(self) -> Tuple[bool, str]:
        """Increasing P_rad_frac_inc should increase calculated efficiency."""
        if self.P_rad_frac_inc is None or self.f_vis_inc is None:
            return (False, "SKIP: needed givens for incandescent not present")
        base = self.P_rad_frac_inc * self.f_vis_inc
        perturbed = (1.1 * self.P_rad_frac_inc) * self.f_vis_inc
        if perturbed <= base:
            return False, f"Perturbed efficiency ({perturbed:.4f}) not greater than baseline ({base:.4f})"
        return True, f"Monotonic: increasing P_rad_frac_inc by 10% increases efficiency from {base:.4f} to {perturbed:.4f}"

    def test_monotonicity_LED_with_int_eff(self) -> Tuple[bool, str]:
        """Increasing eta_int_LED should increase calculated efficiency."""
        if any(g is None for g in [self.eta_int_LED, self.eta_ext_LED, self.eta_ph_LED]):
            return (False, "SKIP: needed LED givens not present")
        base = self.eta_int_LED * self.eta_ext_LED * self.eta_ph_LED
        perturbed = (1.1 * self.eta_int_LED) * self.eta_ext_LED * self.eta_ph_LED
        if perturbed <= base:
            return False, f"Perturbed efficiency ({perturbed:.4f}) not greater than baseline ({base:.4f})"
        return True, f"Monotonic: increasing eta_int_LED by 10% increases efficiency from {base:.4f} to {perturbed:.4f}"

    def test_monotonicity_LED_with_ext_eff(self) -> Tuple[bool, str]:
        """Increasing eta_ext_LED should increase calculated efficiency."""
        if any(g is None for g in [self.eta_int_LED, self.eta_ext_LED, self.eta_ph_LED]):
            return (False, "SKIP: needed LED givens not present")
        base = self.eta_int_LED * self.eta_ext_LED * self.eta_ph_LED
        perturbed = self.eta_int_LED * (1.1 * self.eta_ext_LED) * self.eta_ph_LED
        if perturbed <= base:
            return False, f"Perturbed efficiency ({perturbed:.4f}) not greater than baseline ({base:.4f})"
        return True, f"Monotonic: increasing eta_ext_LED by 10% increases efficiency from {base:.4f} to {perturbed:.4f}"

    # ---------------------------------------------------------------
    # Single weak final-answer sanity (order-of-magnitude only)
    # ---------------------------------------------------------------
    def test_answer_sanity_efficiencies_reasonable(self) -> Tuple[bool, str]:
        """Check both efficiencies lie in broad physically expected ranges."""
        # Use only comparisons, no derivation of expected values, no isclose
        if self.eta_inc is None and self.eta_LED is None:
            return (False, "SKIP: no efficiency data available")
        messages = []
        ok = True
        if self.eta_inc is not None:
            if not (0.01 <= self.eta_inc <= 0.05):
                ok = False
                messages.append(f"eta_inc={self.eta_inc} outside plausible range [0.01,0.05]")
            else:
                messages.append("eta_inc in plausible range")
        if self.eta_LED is not None:
            if not (0.1 <= self.eta_LED <= 0.4):
                ok = False
                messages.append(f"eta_LED={self.eta_LED} outside plausible range [0.1,0.4]")
            else:
                messages.append("eta_LED in plausible range")
        return ok, "; ".join(messages)

    # ---------------------------------------------------------------
    # Main test runner
    # ---------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_passed = True
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
                all_passed = False
        return (all_passed, logs)