import math
import traceback
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Physics checker for the problem:
    For a diatomic ideal gas near room temperature, what fraction of the heat supplied
    is available for external work if the gas is expanded at constant pressure?
    At constant temperature?

    Expected result keys: 'f1' (constant pressure fraction) and 'f2' (isothermal fraction).
    """

    def __init__(self, solution_instance: Any):
        # Safely extract results dictionary
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = getattr(solution_instance, 'result', {})
        if results is None:
            results = {}

        self.f1 = results.get('f1', None)
        self.f2 = results.get('f2', None)

        # Optional auxiliary data for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # -------------------------------------------------------------------------
    # Interface tests: check that required keys exist and are numeric
    # -------------------------------------------------------------------------

    def test_interface_results_dict_exists(self) -> Tuple[bool, str]:
        """Check that the solution_instance has a results dict (not None)."""
        if self.f1 is None and self.f2 is None:
            return (False, "Both f1 and f2 are None. Results dict missing or empty.")
        return (True, "Results dict present and contains at least one key.")

    def test_interface_f1_present(self) -> Tuple[bool, str]:
        """Check that f1 key exists and is not None."""
        if self.f1 is None:
            return (False, "f1 (constant pressure fraction) is missing or None.")
        return (True, "f1 present.")

    def test_interface_f2_present(self) -> Tuple[bool, str]:
        """Check that f2 key exists and is not None."""
        if self.f2 is None:
            return (False, "f2 (isothermal fraction) is missing or None.")
        return (True, "f2 present.")

    def test_interface_f1_numeric(self) -> Tuple[bool, str]:
        """Check that f1 is a numeric type (int or float)."""
        if self.f1 is None:
            return (False, "f1 is None, cannot check type.")
        if not isinstance(self.f1, (int, float)):
            return (False, f"f1 has invalid type {type(self.f1).__name__}. Expected int or float.")
        return (True, "f1 is numeric.")

    def test_interface_f2_numeric(self) -> Tuple[bool, str]:
        """Check that f2 is a numeric type (int or float)."""
        if self.f2 is None:
            return (False, "f2 is None, cannot check type.")
        if not isinstance(self.f2, (int, float)):
            return (False, f"f2 has invalid type {type(self.f2).__name__}. Expected int or float.")
        return (True, "f2 is numeric.")

    # -------------------------------------------------------------------------
    # Invariant (theorem‑based) tests
    # -------------------------------------------------------------------------

    def test_invariant_f1_positive(self) -> Tuple[bool, str]:
        """Work fraction for constant pressure expansion must be positive."""
        if self.f1 is None:
            return (False, "SKIP: f1 not available.")
        if self.f1 <= 0:
            return (False, f"f1 = {self.f1} is not positive (must be >0).")
        return (True, "f1 > 0.")

    def test_invariant_f1_less_than_one(self) -> Tuple[bool, str]:
        """For a constant pressure expansion, work fraction must be < 1 (since ΔU > 0)."""
        if self.f1 is None:
            return (False, "SKIP: f1 not available.")
        if self.f1 >= 1:
            return (False, f"f1 = {self.f1} is not less than 1.")
        return (True, "f1 < 1.")

    def test_invariant_f2_positive(self) -> Tuple[bool, str]:
        """Work fraction for isothermal expansion must be positive."""
        if self.f2 is None:
            return (False, "SKIP: f2 not available.")
        if self.f2 <= 0:
            return (False, f"f2 = {self.f2} is not positive (must be >0).")
        return (True, "f2 > 0.")

    def test_invariant_f2_equals_one(self) -> Tuple[bool, str]:
        """
        For an isothermal process of an ideal gas, ΔU = 0, so Q = W, hence f = W/Q = 1.
        This is a direct consequence of the first law.
        """
        if self.f2 is None:
            return (False, "SKIP: f2 not available.")
        if not math.isclose(self.f2, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return (False, f"f2 = {self.f2} should be exactly 1 (isothermal, ΔU=0).")
        return (True, "f2 equals 1 (within numerical tolerance).")

    def test_invariant_f1_less_than_f2(self) -> Tuple[bool, str]:
        """
        The constant‑pressure fraction must be less than the isothermal fraction (which is 1).
        This is a simple ordering constraint.
        """
        if self.f1 is None or self.f2 is None:
            return (False, "SKIP: f1 or f2 not available.")
        if self.f1 >= self.f2:
            return (False, f"Violation: f1 ({self.f1}) >= f2 ({self.f2}). "
                           f"For constant pressure, fraction should be less than 1.")
        return (True, "f1 < f2 (expected ordering).")

    # -------------------------------------------------------------------------
    # Modeling quality checks (based on explanation_steps and pseudocode)
    # -------------------------------------------------------------------------

    def test_modeling_explanation_contains_constant_pressure(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps mention constant pressure (or isobaric) process.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: 'explanation_steps' attribute not found.")
        steps_text = str(self.explanation_steps).lower()
        if "constant pressure" not in steps_text and "isobaric" not in steps_text:
            return (False, "Explanation steps do not mention 'constant pressure' or 'isobaric'.")
        return (True, "Explanation steps mention constant pressure / isobaric.")

    def test_modeling_explanation_contains_isothermal(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps mention isothermal (or constant temperature) process.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: 'explanation_steps' attribute not found.")
        steps_text = str(self.explanation_steps).lower()
        if "isothermal" not in steps_text and "constant temperature" not in steps_text:
            return (False, "Explanation steps do not mention 'isothermal' or 'constant temperature'.")
        return (True, "Explanation steps mention isothermal / constant temperature.")

    def test_modeling_pseudocode_emits_f1_and_f2(self) -> Tuple[bool, str]:
        """
        Optionally check that the pseudocode emits the expected result keys.
        This is a weak consistency check on the reasoning trace.
        """
        if self.pseudocode is None:
            return (False, "SKIP: 'pseudocode' attribute not found.")
        pseudocode_text = str(self.pseudocode)
        if "results['f1']" not in pseudocode_text or "results['f2']" not in pseudocode_text:
            return (False, "Pseudocode does not emit both f1 and f2 (maybe they are emitted under different names).")
        return (True, "Pseudocode contains emit statements for f1 and f2.")

    # -------------------------------------------------------------------------
    # Monotonicity test
    # -------------------------------------------------------------------------

    def test_monotonicity_f1_decreases_with_increasing_Cp(self) -> Tuple[bool, str]:
        """
        For a constant pressure process, f = W/Q = R/Cp.
        If Cp increases (more degrees of freedom), f should decrease.
        We simulate a 10% increase in Cp and check that the resulting fraction (f1_new)
        is less than the original f1.
        Note: This uses the relationship f1 = R/Cp, derived from first law and ideal gas.
        """
        if self.f1 is None:
            return (False, "SKIP: f1 not available.")
        if self.f1 <= 0:
            return (False, "f1 must be positive to compute monotonicity.")

        # From f1 = R/Cp, we have Cp = R / f1.
        # Increase Cp by 10%: Cp_new = 1.1 * Cp = 1.1 * (R / f1)
        # Then new fraction f1_new = R / Cp_new = R / (1.1 * R / f1) = f1 / 1.1
        f1_new = self.f1 / 1.1

        # Expect f1_new < f1 (since Cp increased)
        if not (f1_new < self.f1):
            return (False, f"Monotonicity violation: f1_new = {f1_new} is not less than original f1 = {self.f1}. "
                           f"Expected decrease when Cp is increased.")
        return (True, f"f1 decreased from {self.f1} to {f1_new} when Cp was increased by 10%.")

    # -------------------------------------------------------------------------
    # Main test runner
    # -------------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them,
        and collect results. Returns (all_passed, log_messages).
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)