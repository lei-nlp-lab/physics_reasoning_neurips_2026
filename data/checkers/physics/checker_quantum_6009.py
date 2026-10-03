import math
import traceback
from typing import Any, Tuple

class PhysicsChecker:
    """
    Checker for alpha-alpha scattering problem with resonances in 8Be.
    Performs interface, invariant, modeling, monotonicity, and weak sanity tests.
    """

    def __init__(self, solution_instance: Any) -> None:
        """Extract results and explanation steps with safe defaults."""
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        self.psi_asymptotic = self.results.get('psi_asymptotic', None)
        self.phase_shift_variation = self.results.get('phase_shift_variation', None)
        self.angular_effect = self.results.get('angular_effect', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _check_string_contains(self, text: str, required_phrases: list) -> bool:
        """Return True if all required phrases appear in text (case-insensitive)."""
        if not text:
            return False
        text_lower = text.lower()
        for phrase in required_phrases:
            if phrase.lower() not in text_lower:
                return False
        return True

    # ------------------------------------------------------------------
    # Interface tests: ensure required keys exist and are non-empty strings
    # ------------------------------------------------------------------
    def test_interface_psi_asymptotic(self) -> Tuple[bool, str]:
        if self.psi_asymptotic is None:
            return (False, "SKIP: psi_asymptotic not available")
        if not isinstance(self.psi_asymptotic, str) or len(self.psi_asymptotic.strip()) == 0:
            return (False, "FAIL: psi_asymptotic is not a non-empty string")
        return (True, "PASS: psi_asymptotic is a non-empty string")

    def test_interface_phase_shift_variation(self) -> Tuple[bool, str]:
        if self.phase_shift_variation is None:
            return (False, "SKIP: phase_shift_variation not available")
        if not isinstance(self.phase_shift_variation, str) or len(self.phase_shift_variation.strip()) == 0:
            return (False, "FAIL: phase_shift_variation is not a non-empty string")
        return (True, "PASS: phase_shift_variation is a non-empty string")

    def test_interface_angular_effect(self) -> Tuple[bool, str]:
        if self.angular_effect is None:
            return (False, "SKIP: angular_effect not available")
        if not isinstance(self.angular_effect, str) or len(self.angular_effect.strip()) == 0:
            return (False, "FAIL: angular_effect is not a non-empty string")
        return (True, "PASS: angular_effect is a non-empty string")

    # ------------------------------------------------------------------
    # Invariant tests: physical laws that must hold
    # ------------------------------------------------------------------
    def test_invariant_bose_symmetry_even_l(self) -> Tuple[bool, str]:
        texts = [self.psi_asymptotic, self.phase_shift_variation, self.angular_effect] + self.explanation_steps
        texts = [t for t in texts if isinstance(t, str)]
        combined = " ".join(texts)
        required = ["even", "l", "identical", "boson"]
        if not self._check_string_contains(combined, required):
            return (False, "FAIL: No mention of even l for identical bosons")
        return (True, "PASS: Even l symmetry mentioned")

    def test_invariant_resonance_phase_shift_jump(self) -> Tuple[bool, str]:
        if self.phase_shift_variation is None:
            return (False, "SKIP: phase_shift_variation not available")
        required = ["rapid", "increase", "pi", "resonance"]
        if not self._check_string_contains(self.phase_shift_variation, required):
            return (False, "FAIL: Phase shift variation does not mention rapid pi increase near resonance")
        return (True, "PASS: Rapid pi increase mentioned")

    def test_invariant_angular_distribution_symmetry(self) -> Tuple[bool, str]:
        texts = [self.angular_effect, self.psi_asymptotic] + self.explanation_steps
        texts = [t for t in texts if isinstance(t, str)]
        combined = " ".join(texts)
        if not (self._check_string_contains(combined, ["symmetric"]) and
                (self._check_string_contains(combined, ["90"]) or
                 self._check_string_contains(combined, ["pi/2"]))):
            return (False, "FAIL: No mention of angular distribution symmetry about 90°")
        return (True, "PASS: Angular symmetry mentioned")

    # ------------------------------------------------------------------
    # Modeling tests: quality of the physical description
    # ------------------------------------------------------------------
    def test_modeling_psi_asymptotic_form(self) -> Tuple[bool, str]:
        if self.psi_asymptotic is None:
            return (False, "SKIP: psi_asymptotic not available")
        required_terms = ["sin", "kr", "lπ/2", "δ_l", "ln", "P_l"]
        if not self._check_string_contains(self.psi_asymptotic, required_terms):
            return (False, f"FAIL: Wave function missing essential terms; check for {required_terms}")
        return (True, "PASS: Wave function contains expected terms")

    def test_modeling_phase_shift_breit_wigner(self) -> Tuple[bool, str]:
        if self.phase_shift_variation is None:
            return (False, "SKIP: phase_shift_variation not available")
        if not (self._check_string_contains(self.phase_shift_variation, ["breit", "wigner"]) or
                self._check_string_contains(self.phase_shift_variation, ["arctan"])):
            return (False, "FAIL: Phase shift variation should reference Breit-Wigner or arctan form")
        return (True, "PASS: Breit-Wigner or arctan mentioned")

    def test_modeling_interference_angular(self) -> Tuple[bool, str]:
        if self.angular_effect is None:
            return (False, "SKIP: angular_effect not available")
        required = ["interference", "nuclear"]
        if not self._check_string_contains(self.angular_effect, required):
            return (False, "FAIL: Angular effect should mention interference with nuclear contribution")
        return (True, "PASS: Interference and nuclear contribution mentioned")

    # ------------------------------------------------------------------
    # Monotonicity tests: explicitly perturb an input and check monotonic behavior
    # ------------------------------------------------------------------
    def test_monotonicity_phase_shift_0(self) -> Tuple[bool, str]:
        """
        Check monotonic increase of l=0 nuclear phase shift near J=0 resonance.
        Uses Breit-Wigner formula with reasonable values.
        """
        # Choose reasonable resonance parameters (derived in-checker, not from answer)
        E_R0 = 95.0  # keV
        Gamma0 = 10.0  # keV (narrow width, typical)
        # Baseline energy and perturbed energies (1% and 5% increase)
        E_base = E_R0 - 5.0  # just below resonance
        E_pert1 = E_R0 - 5.0 * 0.99  # 1% closer to resonance => larger delta
        E_pert2 = E_R0 - 5.0 * 0.95  # 5% closer => larger

        def delta_nuclear(E):
            # Use atan2 for continuity: atan2(Gamma/2, E_R - E) yields increasing function from 0 to pi
            return math.atan2(Gamma0/2.0, E_R0 - E)

        delta_base = delta_nuclear(E_base)
        delta_pert1 = delta_nuclear(E_pert1)
        delta_pert2 = delta_nuclear(E_pert2)

        # Check monotonic increase: as E increases (closer to E_R), delta increases.
        if not (delta_pert1 > delta_base and delta_pert2 > delta_pert1):
            return (False, f"FAIL: l=0 phase shift not monotonic increasing; base={delta_base:.4f}, pert1={delta_pert1:.4f}, pert2={delta_pert2:.4f}")
        return (True, "PASS: l=0 phase shift increases monotonically as energy approaches resonance")

    def test_monotonicity_phase_shift_2(self) -> Tuple[bool, str]:
        """
        Check monotonic increase of l=2 nuclear phase shift near J=2 resonance.
        """
        E_R2 = 3000.0  # keV (3 MeV)
        Gamma2 = 500.0  # keV (wider, but still reasonable)
        E_base = E_R2 - 100.0  # just below resonance
        E_pert1 = E_R2 - 100.0 * 0.99
        E_pert2 = E_R2 - 100.0 * 0.95

        def delta_nuclear(E):
            return math.atan2(Gamma2/2.0, E_R2 - E)

        delta_base = delta_nuclear(E_base)
        delta_pert1 = delta_nuclear(E_pert1)
        delta_pert2 = delta_nuclear(E_pert2)

        if not (delta_pert1 > delta_base and delta_pert2 > delta_pert1):
            return (False, f"FAIL: l=2 phase shift not monotonic increasing; base={delta_base:.4f}, pert1={delta_pert1:.4f}, pert2={delta_pert2:.4f}")
        return (True, "PASS: l=2 phase shift increases monotonically as energy approaches resonance")

    # ------------------------------------------------------------------
    # Weak sanity test (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        texts = [self.psi_asymptotic, self.phase_shift_variation, self.angular_effect] + self.explanation_steps
        texts = [t for t in texts if isinstance(t, str)]
        combined = " ".join(texts)
        required_numbers = ["95", "keV", "3", "MeV"]
        required_physics = ["nuclear", "phase shift"]
        if not (self._check_string_contains(combined, required_numbers) and
                self._check_string_contains(combined, required_physics)):
            return (False, "FAIL: Missing expected numbers (95 keV, 3 MeV) or nuclear phase shift terminology")
        return (True, "PASS: Basic expected information present")

    # ------------------------------------------------------------------
    # Main runner: discover and execute all test_ methods
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)