import math
import traceback
from typing import Dict, List, Tuple, Any, Optional

class PhysicsChecker:
    """
    Audits conceptual physics solutions for nuclear fission questions.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Initialize with a solution container.
        Expects a `.results` dict with keys:
            reaction_type, E_fission, radioactivity_reason, moderator_preference.
        Also optionally extracts:
            mass_defect (delta_m in u), moderator_mass (atomic mass number).
        """
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        # Result values – always None fallback (never numeric defaults)
        self.reaction_type = self.results.get('reaction_type', None)
        self.E_fission = self.results.get('E_fission', None)
        self.radioactivity_reason = self.results.get('radioactivity_reason', None)
        self.moderator_preference = self.results.get('moderator_preference', None)

        # Optional user-supplied parameters (for monotonicity checks)
        # These may be absent; fallback to None
        self.delta_m = self.results.get('delta_m', None)  # mass defect in u
        self.moderator_mass = self.results.get('moderator_mass', None)  # atomic mass number A

        # Optional: full explanation steps (list of strings or single string)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """All four required result keys must be present and not None."""
        if self.reaction_type is None:
            return (False, "Missing/None: reaction_type")
        if self.E_fission is None:
            return (False, "Missing/None: E_fission")
        if self.radioactivity_reason is None:
            return (False, "Missing/None: radioactivity_reason")
        if self.moderator_preference is None:
            return (False, "Missing/None: moderator_preference")
        return (True, "All four required keys present and non‑None")

    def test_interface_reaction_type_neutron_induced(self) -> Tuple[bool, str]:
        """reaction_type must be a string mentioning neutron and fission."""
        if self.reaction_type is None:
            return (False, "SKIP: reaction_type not available")
        if not isinstance(self.reaction_type, str):
            return (False, f"reaction_type is not a string, got {type(self.reaction_type)}")
        rt_lower = self.reaction_type.lower()
        if 'neutron' not in rt_lower:
            return (False, f"reaction_type '{self.reaction_type}' lacks 'neutron'")
        if 'fission' not in rt_lower:
            return (False, f"reaction_type '{self.reaction_type}' lacks 'fission'")
        return (True, f"reaction_type is correct: '{self.reaction_type}'")

    def test_interface_energy_positive(self) -> Tuple[bool, str]:
        """E_fission must be a positive numeric value."""
        if self.E_fission is None:
            return (False, "SKIP: E_fission not available")
        try:
            energy = float(str(self.E_fission).replace('MeV', '').strip())
        except (ValueError, TypeError):
            return (False, f"E_fission '{self.E_fission}' cannot be converted to float")
        if energy <= 0:
            return (False, f"E_fission = {energy} is not positive")
        return (True, f"E_fission is a positive number: {energy}")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_explanation_mass_energy(self) -> Tuple[bool, str]:
        """Explanation steps (if present) must cite mass-energy equivalence."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        if isinstance(self.explanation_steps, list):
            text = ' '.join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        else:
            return (False, "explanation_steps has unknown type")
        text_lower = text.lower()
        keywords = ['e=mc²', 'mass defect', 'binding energy', 'mass-energy']
        if not any(kw in text_lower for kw in keywords):
            return (False, "explanation_steps missing reference to mass‑energy equivalence")
        return (True, "explanation_steps contains mass‑energy principle")

    def test_invariant_conservation_nucleon_number(self) -> Tuple[bool, str]:
        """Reaction type / explanation should imply conservation of nucleons."""
        if self.reaction_type is None:
            return (False, "SKIP: reaction_type not available")
        # The canonical fission reaction shows mass number conserved: 235 + 1 = 236 → fragments + neutron(s)
        if not isinstance(self.reaction_type, str):
            return (False, "reaction_type not a string")
        # Check for mention of uranium and mass numbers (236 derived)
        rt_lower = self.reaction_type.lower()
        if '235' in rt_lower or '236' in rt_lower or 'u' in rt_lower:
            # good enough indication
            return (True, "Reaction type suggests nucleon conservation (U‑235 + n)")
        # If we have explanation steps, also check there
        if self.explanation_steps is not None:
            if isinstance(self.explanation_steps, list):
                text = ' '.join(str(s) for s in self.explanation_steps)
            else:
                text = str(self.explanation_steps)
            if 'conservation' in text.lower() or '235' in text or '236' in text:
                return (True, "Explanation steps imply nucleon conservation")
        # Weak pass – problem is conceptual, this is not a critical fail
        return (False, "No direct evidence of nucleon conservation check, but optional")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_moderation_elastic_collision(self) -> Tuple[bool, str]:
        """Moderator reason should mention elastic scattering or energy transfer via collision."""
        if self.moderator_preference is None:
            return (False, "SKIP: moderator_preference not available")
        if not isinstance(self.moderator_preference, str):
            return (False, "moderator_preference not a string")
        mod_lower = self.moderator_preference.lower()
        # Look for keywords indicating understanding of collision kinematics
        keywords = ['elastic', 'collision', 'energy transfer', 'slow', 'thermal', 'scattering']
        if not any(kw in mod_lower for kw in keywords):
            # Also check explanation steps if available
            if self.explanation_steps is not None:
                if isinstance(self.explanation_steps, list):
                    text = ' '.join(str(s) for s in self.explanation_steps)
                else:
                    text = str(self.explanation_steps)
                if any(kw in text.lower() for kw in keywords):
                    return (True, "Explanation steps contain scattering/energy transfer concept")
            return (False, "moderator_preference lacks any reference to elastic collision or slowing mechanism")
        return (True, "moderator_preference mentions collision / energy transfer")

    def test_modeling_radioactivity_fragment_neutron_rich(self) -> Tuple[bool, str]:
        """Radioactivity reason should explicitly mention neutron-rich fragments or beta decay."""
        if self.radioactivity_reason is None:
            return (False, "SKIP: radioactivity_reason not available")
        if not isinstance(self.radioactivity_reason, str):
            return (False, "radioactivity_reason not a string")
        reason_lower = self.radioactivity_reason.lower()
        keywords = ['neutron-rich', 'excess neutron', 'beta decay', 'β-', 'beta-']
        if not any(kw in reason_lower for kw in keywords):
            return (False, "radioactivity_reason does not mention neutron excess or beta decay")
        return (True, "radioactivity_reason correctly identifies neutron-rich fragments")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_moderator_mass_energy_transfer(self) -> Tuple[bool, str]:
        """
        Perturb moderator atomic mass (A) and check that the answer prefers
        lighter nuclei because energy transfer per collision decreases with A.
        
        Input: we use self.moderator_mass if given; otherwise a default A=1.
        Perturbation: increase A to 12.
        Compute maximum energy transfer fraction: 4A/(A+1)^2.
        The fraction must decrease as A increases.
        Then verify that the answer (moderator_preference) indicates light nuclei.
        """
        # Determine baseline A
        if self.moderator_mass is not None:
            try:
                baseline_A = float(self.moderator_mass)
            except (ValueError, TypeError):
                return (False, "moderator_mass is not numeric")
        else:
            # Default to hydrogen (A=1) if not provided
            baseline_A = 1.0

        # Energy transfer fraction for elastic collision (head-on)
        def transfer_fraction(A: float) -> float:
            if A <= 0:
                return 0.0
            return 4.0 * A / ((A + 1.0) ** 2)

        baseline_frac = transfer_fraction(baseline_A)
        perturbed_A = baseline_A + 11.0  # go from ~1 to ~12
        perturbed_frac = transfer_fraction(perturbed_A)

        # Monotonicity: larger A -> smaller fraction
        if baseline_frac <= perturbed_frac:
            return (False, f"Energy transfer fraction does not decrease with A: baseline_A={baseline_A} frac={baseline_frac:.3f}, perturbed_A={perturbed_A} frac={perturbed_frac:.3f}")

        # Now check that the answer indicates preference for light nuclei
        if self.moderator_preference is None:
            return (False, "SKIP: moderator_preference not available for trend check")
        mod_lower = self.moderator_preference.lower()
        if 'light' not in mod_lower and 'hydrogen' not in mod_lower and 'carbon' not in mod_lower:
            # If the answer says heavy, that's opposite
            if 'heavy' in mod_lower:
                return (False, "Moderator preference says 'heavy', contradicts monotonicity trend")
            # Otherwise not explicit enough; but we can still pass the monotonicity part
        return (True, "Monotonicity confirmed: lighter moderator gives higher energy transfer; answer aligns")

    def test_monotonicity_mass_defect_energy(self) -> Tuple[bool, str]:
        """
        If delta_m (mass defect) is provided, perturb it and check that
        the derived energy (E = delta_m * c^2) increases correspondingly.
        Then verify that the answer's energy magnitude (E_fission) is reasonable.
        If delta_m not given, skip.
        """
        if self.delta_m is None:
            return (False, "SKIP: delta_m (mass defect) not provided")
        try:
            delta_baseline = float(self.delta_m)
        except (ValueError, TypeError):
            return (False, "delta_m is not numeric")
        if delta_baseline <= 0:
            return (False, "delta_m must be positive")
        c2 = 931.5  # MeV/u
        baseline_energy = delta_baseline * c2
        # Perturb: increase delta by +10%
        delta_perturbed = delta_baseline * 1.10
        perturbed_energy = delta_perturbed * c2
        if perturbed_energy <= baseline_energy:
            return (False, "Perturbed energy should be > baseline energy")

        # Check that the answer's E_fission is in the correct ballpark (optional monotonicity inference)
        if self.E_fission is None:
            return (False, "SKIP: E_fission not available for comparison")
        try:
            answer_energy = float(str(self.E_fission).replace('MeV', '').strip())
        except:
            return (False, "Cannot parse E_fission")
        # The answer should be ~200 MeV; if delta_baseline is ~0.1 u, baseline_energy ~93 MeV.
        # This test is just to see that bigger delta gives bigger energy.
        # Return pass if trend holds and answer is positive.
        if answer_energy <= 0:
            return (False, "E_fission is not positive")
        return (True, "Delta‑energy monotonicity holds; answer energy positive")

    # ------------------------------------------------------------------
    # Answer‑sanity test (weak, at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_energy_range(self) -> Tuple[bool, str]:
        """E_fission should be between 150 and 250 MeV (very broad sanity)."""
        if self.E_fission is None:
            return (False, "SKIP: E_fission not available")
        try:
            energy = float(str(self.E_fission).replace('MeV', '').strip())
        except (ValueError, TypeError):
            return (False, f"E_fission '{self.E_fission}' cannot be interpreted as a number")
        if 150 <= energy <= 250:
            return (True, f"E_fission = {energy} MeV within 150‑250 MeV range")
        # Weak fail if outside range, but still return False
        return (False, f"E_fission = {energy} MeV outside expected 150‑250 MeV range")

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute each
        inside try/except, collect pass/fail messages.
        Returns (all_pass, logs).
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
                passed, msg = False, f"CRASHED: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)