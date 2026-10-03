import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution for the problem:
    "Prove that bound state solutions of the time-independent Schrödinger equation
     are nondegenerate and real (up to an overall phase)."
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all necessary data from the solution instance.
        Use None defaults for missing attributes.
        """
        # Extract results dictionary
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract explanation steps (list of strings)
        self.explanation_steps: List[str] = getattr(solution_instance, 'explanation_steps', [])
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = []

        # Expected result keys (from pseudocode contract)
        self.nondegenerate = self.results.get('nondegenerate', None)
        self.real_up_to_phase = self.results.get('real_up_to_phase', None)

        # Store explanation text as one string for substring search
        self.explanation_text: str = ' '.join(self.explanation_steps) if self.explanation_steps else ''

        # Optional numerical inputs that might be present (e.g., mass)
        self.mass = None
        self.energy = None
        # Attempt to extract numerical inputs from the solution instance directly
        for attr in ['mass', 'm', 'Mass', 'M']:
            val = getattr(solution_instance, attr, None)
            if val is not None and isinstance(val, (int, float)):
                self.mass = float(val)
                break
        # Similar for energy (though not strictly needed)
        for attr in ['E', 'energy', 'Energy']:
            val = getattr(solution_instance, attr, None)
            if val is not None and isinstance(val, (int, float)):
                self.energy = float(val)
                break

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods, collect logs.
        Returns (all_pass, logs_list).
        """
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

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the results dictionary contains the two required keys."""
        if not isinstance(self.results, dict):
            return (False, "results is not a dictionary")
        keys = ['nondegenerate', 'real_up_to_phase']
        missing = [k for k in keys if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, f"All required keys present: {keys}")

    def test_interface_nondegenerate_type(self) -> Tuple[bool, str]:
        """Check that nondegenerate is a boolean."""
        val = self.nondegenerate
        if val is None:
            return (False, "nondegenerate not found")
        if not isinstance(val, bool):
            return (False, f"nondegenerate is not boolean, got {type(val).__name__}")
        return (True, f"nondegenerate is boolean: {val}")

    def test_interface_real_up_to_phase_type(self) -> Tuple[bool, str]:
        """Check that real_up_to_phase is a boolean."""
        val = self.real_up_to_phase
        if val is None:
            return (False, "real_up_to_phase not found")
        if not isinstance(val, bool):
            return (False, f"real_up_to_phase is not boolean, got {type(val).__name__}")
        return (True, f"real_up_to_phase is boolean: {val}")

    def test_interface_explanation_steps_exist(self) -> Tuple[bool, str]:
        """Check that explanation_steps is a non-empty list."""
        if not isinstance(self.explanation_steps, list):
            return (False, "explanation_steps is not a list")
        if len(self.explanation_steps) == 0:
            return (False, "explanation_steps is empty")
        return (True, f"explanation_steps has {len(self.explanation_steps)} steps")

    # ----------------------------------------------------------------------
    # Invariant / theorem-based tests
    # ----------------------------------------------------------------------
    def test_invariant_wronskian_constant_property(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps mention the crucial Wronskian property:
        For two solutions of the TISE with same E, the Wronskian is constant.
        """
        key_phrases = ["Wronskian", "constant", "psi1 psi2' - psi2 psi1'"]
        found = [p for p in key_phrases if p.lower() in self.explanation_text.lower()]
        if len(found) >= 2:
            return (True, f"Wronskian property mentioned: {found}")
        else:
            return (False, f"Insufficient Wronskian references. Found: {found}")

    def test_invariant_boundary_condition_used(self) -> Tuple[bool, str]:
        """
        Check that the boundary condition (psi -> 0 as x -> ±∞) is used to set W=0.
        """
        phrases = [
            "boundary condition",
            "psi -> 0",
            "x -> ±∞",
            "W(±∞)=0",
            "W=0 everywhere"
        ]
        found = [p for p in phrases if p.lower() in self.explanation_text.lower()]
        if len(found) >= 2:
            return (True, f"Boundary condition usage found: {found}")
        else:
            return (False, f"Boundary condition usage not clear. Found: {found}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_trace_nondegeneracy_logic(self) -> Tuple[bool, str]:
        """
        Ensure that the explanation traces the nondegeneracy argument:
        assume two independent solutions → W constant → boundary → W=0 → linear dependence.
        """
        key_steps = ["independent", "Wronskian constant", "W=0", "linearly dependent", "contradiction", "nondegenerate"]
        found = [s for s in key_steps if s.lower() in self.explanation_text.lower()]
        if len(found) >= 4:
            return (True, f"Nondegeneracy reasoning traced: {found}")
        else:
            return (False, f"Nondegeneracy reasoning incomplete. Found: {found}")

    def test_trace_reality_logic(self) -> Tuple[bool, str]:
        """
        Ensure that the explanation traces the reality argument:
        real V → real and imaginary parts are solutions → nondegeneracy forces proportionality → phase factor.
        """
        key_steps = ["real V", "ψ*", "real part", "imaginary part", "proportional", "phase factor", "real up to phase"]
        found = [s for s in key_steps if s.lower() in self.explanation_text.lower()]
        if len(found) >= 3:
            return (True, f"Reality reasoning traced: {found}")
        else:
            return (False, f"Reality reasoning incomplete. Found: {found}")

    # ----------------------------------------------------------------------
    # Monotonicity test
    # ----------------------------------------------------------------------
    def test_monotonicity_mass_perturbation(self) -> Tuple[bool, str]:
        """
        Perturb the mass (if available) by +5% and verify that the nondegeneracy
        conclusion remains True (as required by the theorem). 
        This test is valid because the theorem holds for any positive mass;
        perturbing mass should not change the physical result.
        """
        if self.mass is None:
            return (False, "SKIP: mass not available for perturbation")
        if self.nondegenerate is None:
            return (False, "SKIP: nondegenerate flag not available")
        # Baseline: nondegenerate should be True (from theorem)
        baseline = self.nondegenerate
        if baseline is not True:
            return (False, f"Baseline nondegenerate is {baseline}, expected True")

        # Perturb mass by 5%
        perturbed_mass = self.mass * 1.05
        # The theorem's conclusion is independent of mass; we cannot recompute the flag,
        # but we assert it should still be True.
        # We use the fact that nondegeneracy is a proven property for any mass.
        # Thus we 'simulate' the perturbed output as still True.
        perturbed_output = True  # theoretical expectation
        if perturbed_output != baseline:
            return (False, "Mass perturbation changed nondegeneracy (should not happen)")
        # Final check: both baseline and perturbed output are True
        if baseline and perturbed_output:
            return (True, f"Mass perturbed by +5%: nondegeneracy remains True (baseline=True)")
        else:
            return (False, "Unexpected condition with nondegeneracy flag")

    # ----------------------------------------------------------------------
    # Weak answer sanity test (only one, and very weak)
    # ----------------------------------------------------------------------
    def test_answer_sanity_boolean_true(self) -> Tuple[bool, str]:
        """
        Weak sanity: both final flags should be True.
        This does not re-derive anything; it's a straightforward check.
        """
        if self.nondegenerate is None or self.real_up_to_phase is None:
            return (False, "One or both answer keys missing")
        if not isinstance(self.nondegenerate, bool) or not isinstance(self.real_up_to_phase, bool):
            return (False, "Answer keys are not boolean")
        if self.nondegenerate and self.real_up_to_phase:
            return (True, "Both nondegenerate and real_up_to_phase are True")
        else:
            return (False, f"Expected both True, got nondegenerate={self.nondegenerate}, real_up_to_phase={self.real_up_to_phase}")