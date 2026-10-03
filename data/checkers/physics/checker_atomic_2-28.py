import math
import traceback
from typing import List, Tuple, Any, Optional

class PhysicsChecker:
    """
    Audits solutions for the binding energy proportionality problem.
    Checks interface contracts, physical invariants, modeling quality,
    and monotonicity trends.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely; default to None for missing keys.
        self.results: Optional[dict] = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        self.B_vs_A: Optional[str] = self.results.get('B_vs_A', None)
        self.a_v: Optional[float] = self.results.get('a_v', None)
        self.a_v_units: Optional[str] = self.results.get('a_v_units', None)
        self.reason: Optional[str] = self.results.get('reason', None)

    # ------- Interface Test Methods -------

    def test_interface_B_vs_A_exists(self) -> Tuple[bool, str]:
        """Check that 'B_vs_A' key is present in results."""
        if 'B_vs_A' not in self.results:
            return (False, "Missing key 'B_vs_A' in results")
        return (True, "Key 'B_vs_A' is present")

    def test_interface_a_v_exists(self) -> Tuple[bool, str]:
        """Check that 'a_v' key is present."""
        if 'a_v' not in self.results:
            return (False, "Missing key 'a_v' in results")
        return (True, "Key 'a_v' is present")

    def test_interface_a_v_units_exists(self) -> Tuple[bool, str]:
        """Check that 'a_v_units' key is present."""
        if 'a_v_units' not in self.results:
            return (False, "Missing key 'a_v_units' in results")
        return (True, "Key 'a_v_units' is present")

    def test_interface_reason_exists(self) -> Tuple[bool, str]:
        """Check that 'reason' key is present."""
        if 'reason' not in self.results:
            return (False, "Missing key 'reason' in results")
        return (True, "Key 'reason' is present")

    def test_interface_a_v_is_numeric(self) -> Tuple[bool, str]:
        """Check that a_v has a numeric type (int or float)."""
        if self.a_v is None:
            return (False, "a_v is None, cannot check type")
        if not isinstance(self.a_v, (int, float)):
            return (False, f"a_v is not numeric: {type(self.a_v).__name__}")
        return (True, "a_v is numeric")

    # ------- Invariant Test Methods -------

    def test_invariant_a_v_range(self) -> Tuple[bool, str]:
        """Weak sanity: a_v must be positive and roughly 10-20 MeV."""
        if self.a_v is None:
            return (False, "a_v is None, cannot check range")
        if not isinstance(self.a_v, (int, float)):
            return (False, "a_v is not numeric")
        if self.a_v <= 0:
            return (False, f"a_v = {self.a_v}; must be positive")
        if self.a_v < 10 or self.a_v > 20:
            return (False, f"a_v = {self.a_v} is outside expected range [10, 20] MeV")
        return (True, f"a_v = {self.a_v} MeV is in plausible range")

    def test_invariant_a_v_units_MeV(self) -> Tuple[bool, str]:
        """Units should be MeV (expect 'MeV' string)."""
        if self.a_v_units is None:
            return (False, "a_v_units is None")
        if not isinstance(self.a_v_units, str):
            return (False, "a_v_units is not a string")
        if self.a_v_units != 'MeV':
            return (False, f"Units are '{self.a_v_units}', expected 'MeV'")
        return (True, "Units correctly set to 'MeV'")

    def test_invariant_B_vs_A_linear(self) -> Tuple[bool, str]:
        """
        Check that B_vs_A indicates proportionality to A (not A^2).
        Expected: string containing 'prop' and 'A', but not 'A^2'.
        """
        if self.B_vs_A is None:
            return (False, "B_vs_A is None")
        if not isinstance(self.B_vs_A, str):
            return (False, "B_vs_A is not a string")
        if 'prop' not in self.B_vs_A and 'proportional' not in self.B_vs_A:
            return (False, f"B_vs_A = '{self.B_vs_A}' does not indicate proportionality")
        if 'A^2' in self.B_vs_A or 'A²' in self.B_vs_A:
            return (False, f"B_vs_A = '{self.B_vs_A}' suggests quadratic dependence")
        if 'A' not in self.B_vs_A:
            return (False, f"B_vs_A = '{self.B_vs_A}' missing reference to mass number A")
        return (True, "B_vs_A correctly indicates linear proportionality to A")

    # ------- Modeling / Trace Test Methods -------

    def test_modeling_reason_short_range(self) -> Tuple[bool, str]:
        """Check that reason mentions 'short-range' force."""
        if self.reason is None:
            return (False, "reason is None")
        if not isinstance(self.reason, str):
            return (False, "reason is not a string")
        if 'short-range' not in self.reason:
            return (False, f"reason '{self.reason}' does not contain 'short-range'")
        return (True, "reason mentions short-range force")

    def test_modeling_reason_saturation(self) -> Tuple[bool, str]:
        """Check that reason mentions 'saturat' (saturation, saturating)."""
        if self.reason is None:
            return (False, "reason is None")
        if not isinstance(self.reason, str):
            return (False, "reason is not a string")
        if 'saturat' not in self.reason:
            return (False, f"reason '{self.reason}' does not mention saturation")
        return (True, "reason mentions saturation of nuclear force")

    # ------- Monotonicity Test Method (Perturbation) -------

    def test_monotonicity_binding_energy_with_A(self) -> Tuple[bool, str]:
        """
        Check monotonicity: For a given a_v, binding energy B = a_v * A should
        increase with A. Use two hypothetical A values (A1=50, A2=100) and
        compute B1 and B2. Verify that B2 > B1.
        This test perturbs the input A (mass number) and checks the trend.
        """
        if self.a_v is None:
            return (False, "SKIP: a_v is None, cannot compute binding energy values")
        if not isinstance(self.a_v, (int, float)):
            return (False, "SKIP: a_v is not numeric")

        # Choose two distinct mass numbers
        A1 = 50
        A2 = 100  # A2 > A1

        # Compute volume-term binding energies (hypothetical)
        B1 = self.a_v * A1
        B2 = self.a_v * A2

        if B2 <= B1:
            return (False, f"Monotonicity violated: B(A=100)={B2} <= B(A=50)={B1}. "
                           "Binding energy should increase with A for constant a_v.")
        # Also check that the ratio is exactly A2/A1 (linear scaling)
        # Use math.isclose to allow floating point tolerance
        if not math.isclose(B2 / B1, A2 / A1, rel_tol=1e-9):
            return (False, f"Scaling mismatch: B2/B1 = {B2/B1} != A2/A1 = {A2/A1}")
        return (True, "Binding energy increases linearly with A (monotonicity holds)")

    # ------- Weak Answer Sanity -------

    def test_answer_sanity_a_v_positive(self) -> Tuple[bool, str]:
        """
        Weak sanity: a_v must be positive and finite.
        This test is secondary; it does NOT re-derive the answer.
        """
        if self.a_v is None:
            return (False, "a_v is None, cannot check positivity")
        if not isinstance(self.a_v, (int, float)):
            return (False, "a_v is not numeric")
        if math.isnan(self.a_v) or math.isinf(self.a_v):
            return (False, "a_v is NaN or infinite")
        if self.a_v <= 0:
            return (False, f"a_v = {self.a_v} is not positive")
        return (True, "a_v is positive and finite")

    # ------- Main Runner -------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods and execute them.
        Returns (all_pass, logs) where all_pass is True if every test passed.
        """
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith('test_'):
                continue
            method = getattr(self, name, None)
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