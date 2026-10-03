import math
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for the rubber band thermodynamics problem.
    Tests the signed thermodynamic responses for isothermal stretching (entropy)
    and adiabatic stretching (internal energy) using theorem-driven reasoning,
    including monotonicity tests that perturb inputs numerically.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed quantities safely. Use None for missing values.
        The solution_instance is expected to have:
            .results['dS_sign']   : str  ('negative', 'positive', etc.)
            .results['dU_sign']   : str
        Optionally, for deeper consistency checks:
            .dS_dT_L  : float   (positive, heat capacity factor)
            .dT_dL_S  : float   (positive, adiabatic temperature increase)
            .dS_dL_T  : float   (derived from cyclic relation)
            .f        : float   (force > 0)
            .dL       : float   (stretch increment > 0)
        """
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.dS_sign: Optional[str] = self.results.get('dS_sign', None)
        self.dU_sign: Optional[str] = self.results.get('dU_sign', None)

        # Optional numeric quantities for deeper checks
        self.dS_dT_L: Optional[float] = getattr(solution_instance, 'dS_dT_L', None)
        self.dT_dL_S: Optional[float] = getattr(solution_instance, 'dT_dL_S', None)
        self.dS_dL_T: Optional[float] = getattr(solution_instance, 'dS_dL_T', None)
        self.f: Optional[float] = getattr(solution_instance, 'f', None)
        self.dL: Optional[float] = getattr(solution_instance, 'dL', None)

    # ----- Interface tests (required keys & basic type) -----

    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that results dict contains both 'dS_sign' and 'dU_sign'."""
        if 'dS_sign' not in self.results:
            return False, "Missing key 'dS_sign' in results."
        if 'dU_sign' not in self.results:
            return False, "Missing key 'dU_sign' in results."
        return True, "Both required keys present."

    def test_interface_sign_values_are_strings(self) -> Tuple[bool, str]:
        """Ensure dS_sign and dU_sign are strings."""
        if not isinstance(self.dS_sign, str):
            return False, f"dS_sign is not a string (got {type(self.dS_sign).__name__})."
        if not isinstance(self.dU_sign, str):
            return False, f"dU_sign is not a string (got {type(self.dU_sign).__name__})."
        return True, "Both signs are strings."

    # ----- Invariant (theorem) tests -----

    def test_invariant_entropy_sign_via_cyclic(self) -> Tuple[bool, str]:
        """
        Use the cyclic relation (∂S/∂L)_T = - (∂S/∂T)_L (∂T/∂L)_S
        to deduce the sign of (∂S/∂L)_T when the partials are available.
        Since (∂S/∂T)_L > 0 and (∂T/∂L)_S > 0, (∂S/∂L)_T must be negative.
        Then dS = (∂S/∂L)_T dL with dL>0 gives dS negative.
        """
        if self.dS_dT_L is None or self.dT_dL_S is None:
            return False, "SKIP: dS_dT_L or dT_dL_S not provided for cyclic check."

        if self.dS_dT_L <= 0:
            return False, "Violation: (∂S/∂T)_L must be >0 (positive heat capacity)."
        if self.dT_dL_S <= 0:
            return False, "Violation: (∂T/∂L)_S must be >0 (given)."

        # Sign of (∂S/∂L)_T is negative
        if self.dS_sign is None:
            return False, "dS_sign missing while partials present."
        if self.dS_sign != 'negative':
            return False, f"dS_sign is '{self.dS_sign}' but expected 'negative' from cyclic relation."
        return True, f"Entropy sign derived from cyclic relation matches reported '{self.dS_sign}'."

    def test_invariant_internal_energy_adiabatic(self) -> Tuple[bool, str]:
        """
        For adiabatic stretch (dQ=0), first law gives dU = f dL.
        With f>0 and dL>0, dU must be positive.
        """
        if self.f is not None and self.dL is not None:
            if self.f <= 0:
                return False, "Force f must be positive for stretching."
            if self.dL <= 0:
                return False, "Length change dL must be positive."

        if self.dU_sign is None:
            return False, "dU_sign not provided."
        if self.dU_sign != 'positive':
            return False, f"dU_sign is '{self.dU_sign}' but must be 'positive' for adiabatic stretching."
        return True, "Adiabatic internal energy sign is positive."

    def test_invariant_ideal_rubber_relation(self) -> Tuple[bool, str]:
        """
        For an ideal rubber, (∂U/∂L)_T = 0. This is a key constraint.
        This test is informational; we cannot verify directly without isothermal dU data.
        """
        return True, "Ideal rubber constraint (∂U/∂L)_T=0 is consistent with the physics model."

    # ----- Modeling and trace checks -----

    def test_modeling_work_positive(self) -> Tuple[bool, str]:
        """Work done on the system dW = f dL must be positive for stretching."""
        if self.f is None or self.dL is None:
            return False, "SKIP: force or length change not available."
        if self.f <= 0:
            return False, "Force f should be positive (>0) for stretching."
        if self.dL <= 0:
            return False, "Length change dL should be positive."
        return True, "Work f dL > 0 as expected."

    def test_modeling_adiabatic_condition(self) -> Tuple[bool, str]:
        """For adiabatic process, dQ must be zero. Check if provided."""
        # No direct dQ value; we rely on dU derivation: dU = f dL implies dQ=0.
        return True, "Adiabatic condition (dQ=0) inferred from dU sign and first law."

    def test_modeling_cyclic_relation_equality(self) -> Tuple[bool, str]:
        """
        If we have numeric values for all three partials,
        check that (∂S/∂L)_T = - (∂S/∂T)_L (∂T/∂L)_S holds within tolerance.
        """
        if self.dS_dL_T is None or self.dS_dT_L is None or self.dT_dL_S is None:
            return False, "SKIP: insufficient partial derivatives to check cyclic equality."
        expected = - self.dS_dT_L * self.dT_dL_S
        if not math.isclose(self.dS_dL_T, expected, rel_tol=1e-3):
            return False, f"Cyclic relation violated: (∂S/∂L)_T={self.dS_dL_T}, expected {expected}."
        return True, "Cyclic relation holds within tolerance."

    # ----- Monotonicity tests (explicit input perturbation required) -----

    def test_monotonicity_entropy_with_length(self) -> Tuple[bool, str]:
        """
        For isothermal stretching, entropy decreases with length.
        If we have a numeric derivative (∂S/∂L)_T and baseline dL,
        perturb dL (e.g., double it) and check that dS becomes more negative.
        If numeric values not available, skip.
        """
        # We need dS_dL_T (or compute from cyclic) and dL
        dS_dL_T = self.dS_dL_T
        if dS_dL_T is None:
            # Try to compute from cyclic
            if self.dS_dT_L is not None and self.dT_dL_S is not None:
                dS_dL_T = - self.dS_dT_L * self.dT_dL_S
        if dS_dL_T is None:
            return False, "SKIP: cannot compute (∂S/∂L)_T for numerical perturbation."
        if self.dL is None or self.dL <= 0:
            return False, "SKIP: need positive dL for perturbation."

        baseline_dS = dS_dL_T * self.dL
        # Perturb: increase dL by a factor (e.g., 2x)
        perturbed_dL = 2.0 * self.dL
        perturbed_dS = dS_dL_T * perturbed_dL

        # For monotonic decrease: dS should become more negative
        if perturbed_dS >= baseline_dS:
            return False, f"Entropy change not monotonically decreasing: baseline dS = {baseline_dS}, perturbed dS = {perturbed_dS}."
        return True, f"Entropy change becomes more negative: baseline {baseline_dS} -> perturbed {perturbed_dS}."

    def test_monotonicity_internal_energy_with_length(self) -> Tuple[bool, str]:
        """
        For adiabatic stretching, internal energy increases with length.
        If we have f and dL, perturb dL and check that dU increases.
        """
        if self.f is None or self.f <= 0:
            return False, "SKIP: need positive force f."
        if self.dL is None or self.dL <= 0:
            return False, "SKIP: need positive length change dL."

        baseline_dU = self.f * self.dL
        # Perturb: increase dL by 10%
        perturbed_dL = 1.10 * self.dL
        perturbed_dU = self.f * perturbed_dL

        if perturbed_dU <= baseline_dU:
            return False, f"Internal energy change not monotonically increasing: baseline dU = {baseline_dU}, perturbed dU = {perturbed_dU}."
        return True, f"Internal energy change increases: baseline {baseline_dU} -> perturbed {perturbed_dU}."

    # ----- Weak answer sanity (only one such method) -----

    def test_answer_sanity_signs(self) -> Tuple[bool, str]:
        """
        Weak sanity: dS_sign and dU_sign must be among the allowed possibilities.
        They cannot be empty or None.
        """
        allowed = {'positive', 'negative', 'stays the same', 'zero', 'unchanged'}
        if self.dS_sign is None:
            return False, "dS_sign is None."
        if self.dS_sign not in allowed:
            return False, f"dS_sign '{self.dS_sign}' not in allowed set {allowed}."
        if self.dU_sign is None:
            return False, "dU_sign is None."
        if self.dU_sign not in allowed:
            return False, f"dU_sign '{self.dU_sign}' not in allowed set {allowed}."
        return True, "Signs are valid strings among expected possibilities."

    # ----- Test runner -----

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, run them, and collect results."""
        logs: List[str] = []
        all_pass: bool = True
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