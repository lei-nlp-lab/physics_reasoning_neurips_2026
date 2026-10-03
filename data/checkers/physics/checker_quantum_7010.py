import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution for the two-fermion infinite square well problem.
    Verifies state classification, spin assignments, and first-order perturbation corrections.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results dictionary safely
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if self.results is None:
            # Fallback: try to access attributes directly if no results dict
            self.lowest_state = getattr(solution_instance, 'lowest_state', None)
            self.second_lowest_state = getattr(solution_instance, 'second_lowest_state', None)
            self.third_lowest_state = getattr(solution_instance, 'third_lowest_state', None)
            self.E2_correction = getattr(solution_instance, 'E2_correction', None)
            self.E3_correction = getattr(solution_instance, 'E3_correction', None)
        else:
            self.lowest_state = self.results.get('lowest_state', None)
            self.second_lowest_state = self.results.get('second_lowest_state', None)
            self.third_lowest_state = self.results.get('third_lowest_state', None)
            self.E2_correction = self.results.get('E2_correction', None)
            self.E3_correction = self.results.get('E3_correction', None)

        # Optional: extraction of givens needed for monotonicity test
        self.m = getattr(solution_instance, 'm', None)
        self.L = getattr(solution_instance, 'L', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discovers all test_* methods, executes them, and returns (all_pass, logs)."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface checks: expected keys exist and have plausible types
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all five expected result keys are present and non-None."""
        keys = ['lowest_state', 'second_lowest_state', 'third_lowest_state',
                'E2_correction', 'E3_correction']
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"Missing or None: {key}")
        return (True, "All required keys present and non-None")

    def test_interface_state_types(self) -> Tuple[bool, str]:
        """State descriptions must be strings."""
        for key in ['lowest_state', 'second_lowest_state', 'third_lowest_state']:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"SKIP: {key} not available")
            if not isinstance(val, str):
                return (False, f"{key} is not a string (type: {type(val).__name__})")
        return (True, "All state descriptions are strings")

    def test_interface_correction_types(self) -> Tuple[bool, str]:
        """Correction expressions must be strings (integral forms)."""
        for key in ['E2_correction', 'E3_correction']:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"SKIP: {key} not available")
            if not isinstance(val, str):
                return (False, f"{key} is not a string (type: {type(val).__name__})")
        return (True, "Both correction expressions are strings")

    # ------------------------------------------------------------------
    # Invariant checks: physical laws (Pauli principle, state classification)
    # ------------------------------------------------------------------
    def test_invariant_pauli_principle(self) -> Tuple[bool, str]:
        """Check that second-lowest state has antisymmetric spatial part and triplet spin,
           and third-lowest has symmetric spatial and singlet spin. (Pauli antisymmetry.)"""
        second = self.second_lowest_state
        third = self.third_lowest_state
        if second is None or third is None:
            return (False, "SKIP: second or third state not available")
        s_lower = second.lower()
        t_lower = third.lower()
        if 'antisymmetric' not in s_lower or 'triplet' not in s_lower:
            return (False, "Second state description lacks 'antisymmetric' and/or 'triplet'")
        if 'symmetric' not in t_lower or 'singlet' not in t_lower:
            return (False, "Third state description lacks 'symmetric' and/or 'singlet'")
        return (True, "Pauli principle satisfied by state descriptions (antisymmetry combination)")

    def test_invariant_ground_state_singlet(self) -> Tuple[bool, str]:
        """Ground state must be symmetric spatial * singlet spin."""
        gs = self.lowest_state
        if gs is None:
            return (False, "SKIP: lowest_state not available")
        gs_lower = gs.lower()
        if 'singlet' not in gs_lower:
            return (False, "Ground state should be spin singlet")
        if 'n=1' not in gs_lower and 'both' not in gs_lower:
            return (False, "Ground state description should indicate both particles in n=1")
        return (True, "Ground state correctly described as symmetric spatial, singlet spin")

    # ------------------------------------------------------------------
    # Modeling checks: correctness of perturbation correction expressions
    # ------------------------------------------------------------------
    def test_modeling_E2_antisymmetric_integral(self) -> Tuple[bool, str]:
        """Check that E2_correction is an integral over |psi_12^A|^2 V."""
        expr = self.E2_correction
        if expr is None:
            return (False, "SKIP: E2_correction not available")
        e_lower = expr.lower()
        required_parts = ['integral', 'psi', 'v(x1-x2)']
        if 'antisymmetric' not in e_lower and 'psi_a' not in e_lower and 'psi^a' not in e_lower:
            if '^a' not in e_lower and '_a' not in e_lower:
                return (False, "E2_correction does not mention antisymmetric wavefunction")
        for part in required_parts:
            if part not in e_lower:
                return (False, f"E2_correction missing '{part}'")
        return (True, "E2_correction appears as integral over antisymmetric spatial state")

    def test_modeling_E3_symmetric_integral(self) -> Tuple[bool, str]:
        """Check that E3_correction is an integral over |psi_12^S|^2 V."""
        expr = self.E3_correction
        if expr is None:
            return (False, "SKIP: E3_correction not available")
        e_lower = expr.lower()
        required_parts = ['integral', 'psi', 'v(x1-x2)']
        if 'symmetric' not in e_lower and 'psi_s' not in e_lower and 'psi^s' not in e_lower:
            if '^s' not in e_lower and '_s' not in e_lower:
                return (False, "E3_correction does not mention symmetric wavefunction")
        for part in required_parts:
            if part not in e_lower:
                return (False, f"E3_correction missing '{part}'")
        return (True, "E3_correction appears as integral over symmetric spatial state")

    # ------------------------------------------------------------------
    # Monotonicity check: perturb L and verify unperturbed energy scaling
    # ------------------------------------------------------------------
    def test_monotonicity_unperturbed_energy_scales_with_L(self) -> Tuple[bool, str]:
        """
        Perturb the well length L by +10% and verify that the unperturbed
        energy of the (1,2) manifold (E = 5 * hbar^2 * π^2 / (2*m*L^2))
        changes by the factor (1/1.1^2). This checks the monotonic
        inverse-square relation.
        """
        if self.L is None or self.m is None or self.hbar is None:
            return (False, "SKIP: required givens (m, L, hbar) not provided")
        # Baseline energy of (1,2) manifold (without perturbation)
        factor = 5.0 * (math.pi ** 2) / (2.0 * self.m * (self.L ** 2))
        E_baseline = self.hbar ** 2 * factor

        # Perturb L to 1.1*L
        L_perturbed = 1.1 * self.L
        E_perturbed = (self.hbar ** 2) * 5.0 * (math.pi ** 2) / (2.0 * self.m * (L_perturbed ** 2))

        # Expected ratio from inverse-square scaling: E ∝ 1/L^2
        scale_factor = 1.0 / (1.1 ** 2)  # ≈ 0.826446
        expected_E_perturbed = E_baseline * scale_factor

        if not math.isclose(E_perturbed, expected_E_perturbed, rel_tol=1e-9):
            return (False, f"Unperturbed energy scaling failed: E_baseline={E_baseline}, "
                           f"E_perturbed={E_perturbed}, expected={expected_E_perturbed}")
        return (True, "Unperturbed energy of (1,2) manifold scales correctly as ~1/L^2")

    # ------------------------------------------------------------------
    # Weak answer sanity (single method, does not compute expected values)
    # ------------------------------------------------------------------
    def test_answer_sanity_general(self) -> Tuple[bool, str]:
        """Weak check: state descriptions are distinct, mention n=1 and n=2, and corrections are symbolic integrals."""
        states = [self.lowest_state, self.second_lowest_state, self.third_lowest_state]
        if any(s is None for s in states):
            return (False, "SKIP: one or more state descriptions missing")
        if len(set(states)) < 3:
            return (False, "State descriptions are not all distinct")
        all_text = " ".join(states).lower()
        if "n=1" not in all_text or "n=2" not in all_text:
            return (False, "State descriptions do not mention principal quantum numbers n=1 and n=2")
        for key in ['E2_correction', 'E3_correction']:
            expr = getattr(self, key, None)
            if expr is None:
                return (False, f"SKIP: {key} missing")
            stripped = expr.strip()
            if stripped.replace('.', '').replace('-', '').isdigit():
                return (False, f"{key} appears to be a plain number, not an integral expression")
            if 'int' not in stripped.lower():
                return (False, f"{key} does not contain 'integral' or an integral symbol")
        return (True, "States distinct with n=1,n=2; corrections are symbolic integrals (not numeric)")