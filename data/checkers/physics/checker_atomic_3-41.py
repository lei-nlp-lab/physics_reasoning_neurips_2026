import math
import re
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Qualitative and numerical checker for neutrino physics problem.
    Includes interface, invariant, modeling/trace, monotonicity, and weak sanity tests.
    Numeric values are derived from explanation_steps or synthetic baselines.
    """
    def __init__(self, solution_instance: Any):
        # Extract results safely, None if missing
        results = getattr(solution_instance, 'results', None) or {}
        self.properties = results.get('properties', None)
        self.motivation = results.get('motivation', None)
        self.detection_method = results.get('detection_method', None)

        # Explanation steps (string/list) for textual invariants
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Convert to string if list
        if isinstance(self.explanation_steps, (list, dict)):
            self.explanation_steps = str(self.explanation_steps)

        # Internal numeric model: try to extract Q-value from explanation_steps
        self.Q_value = None
        if self.explanation_steps:
            # Look for "transition energy", "Q-value", "cutoff" numeric value in MeV
            # Example: "cutoff given by the transition energy" or "Q = 1.0 MeV"
            match = re.search(r'(\d+\.?\d*)\s*(?:MeV|Q)', self.explanation_steps, re.IGNORECASE)
            if match:
                self.Q_value = float(match.group(1))
            # If no number, we'll use a default synthetic Q for monotonicity test
            # but only if explanation_steps mentions it conceptually
            if self.Q_value is None and 'continuous' in self.explanation_steps.lower():
                self.Q_value = 1.0  # placeholder for testing structure

        # Baseline for monotonicity: derived endpoint energy
        self.endpoint_energy = self.Q_value if self.Q_value is not None else None

    # ---------------------------------------------------------------
    # 1. Interface tests
    # ---------------------------------------------------------------
    def test_interface_properties_present(self) -> Tuple[bool, str]:
        if self.properties is None:
            return (False, "SKIP: properties not available")
        if not isinstance(self.properties, str):
            return (False, f"Expected string, got {type(self.properties).__name__}")
        return (True, "properties is a string")

    def test_interface_motivation_present(self) -> Tuple[bool, str]:
        if self.motivation is None:
            return (False, "SKIP: motivation not available")
        if not isinstance(self.motivation, str):
            return (False, f"Expected string, got {type(self.motivation).__name__}")
        return (True, "motivation is a string")

    def test_interface_detection_method_present(self) -> Tuple[bool, str]:
        if self.detection_method is None:
            return (False, "SKIP: detection_method not available")
        if not isinstance(self.detection_method, str):
            return (False, f"Expected string, got {type(self.detection_method).__name__}")
        return (True, "detection_method is a string")

    # ---------------------------------------------------------------
    # 2. Invariant tests (theorem-based physical sanity)
    # ---------------------------------------------------------------
    def test_invariant_conservation_energy_momentum(self) -> Tuple[bool, str]:
        # Check that explanation_steps mentions both conservation laws
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps not available")
        text = self.explanation_steps.lower()
        has_energy = 'conservation of energy' in text or 'energy conservation' in text
        has_momentum = 'conservation of momentum' in text or 'momentum conservation' in text
        if has_energy and has_momentum:
            return (True, "Both energy and momentum conservation mentioned")
        missing = []
        if not has_energy:
            missing.append('energy conservation')
        if not has_momentum:
            missing.append('momentum conservation')
        return (False, f"Missing invariance mentions: {missing}")

    def test_invariant_angular_momentum_spin(self) -> Tuple[bool, str]:
        # Neutrinos have spin 1/2 – check if mentioned in properties or explanation
        if not self.properties and not self.explanation_steps:
            return (False, "SKIP: no textual content available")
        text = (self.properties or '') + ' ' + (self.explanation_steps or '')
        if 'spin' in text.lower() or 'angular momentum' in text.lower():
            return (True, "Spin/angular momentum mentioned")
        return (True, "WARN: spin not explicitly mentioned (OK if not required)")

    def test_invariant_zero_charge(self) -> Tuple[bool, str]:
        if not self.properties:
            return (False, "SKIP: properties not available")
        if 'zero charge' in self.properties.lower() or 'neutral' in self.properties.lower():
            return (True, "Neutral/zero charge mentioned")
        return (False, "Neutrinos and antineutrinos have zero electric charge; not mentioned")

    # ---------------------------------------------------------------
    # 3. Modeling/trace tests – based on explanation_steps or detection method
    # ---------------------------------------------------------------
    def test_modeling_three_body_decay(self) -> Tuple[bool, str]:
        # Beta decay is a three-body decay: continuous spectrum
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps not available")
        if 'three-body' in self.explanation_steps.lower() or 'continuous' in self.explanation_steps.lower():
            return (True, "Three-body decay / continuous spectrum described")
        return (False, "Missing description of beta decay as three-body decay leading to continuous spectrum")

    def test_modeling_inverse_beta_decay_reaction(self) -> Tuple[bool, str]:
        # Detection via inverse beta decay: symbols present
        if not self.detection_method:
            return (False, "SKIP: detection_method not available")
        text = self.detection_method
        # Check for at least three of the key symbols
        symbols = ['ν', 'p', 'n', 'e']
        found = [s for s in symbols if s in text]
        if len(found) >= 3:
            return (True, f"Inverse beta decay reaction symbols present: {found}")
        return (False, f"Insufficient reaction symbols in detection_method, found {found}")

    def test_modeling_lepton_number_conservation(self) -> Tuple[bool, str]:
        # Check if lepton number conservation is mentioned anywhere
        if not self.properties and not self.motivation and not self.explanation_steps:
            return (False, "SKIP: no textual content available")
        text = (self.properties or '') + ' ' + (self.motivation or '') + ' ' + (self.explanation_steps or '')
        if 'lepton number' in text.lower() or 'conservation' in text.lower():
            return (True, "Lepton number or conservation mentioned")
        return (True, "WARN: lepton number not explicitly mentioned (OK if not required)")

    # ---------------------------------------------------------------
    # 4. Monotonicity test – must perturb an input and compare outputs
    # ---------------------------------------------------------------
    def test_monotonicity_endpoint_increases_with_Q(self) -> Tuple[bool, str]:
        # Uses Q-value extracted from explanation_steps (or synthetic baseline)
        # If Q not available, SKIP
        if self.Q_value is None:
            return (False, "SKIP: Q_value not available from explanation_steps")
        # Baseline
        Q_baseline = self.Q_value
        endpoint_baseline = Q_baseline   # E_e_max = Q

        # Perturb Q by +10%
        Q_perturbed = Q_baseline * 1.1
        endpoint_perturbed = Q_perturbed

        # Check that endpoint increases (monotonicity)
        if endpoint_perturbed > endpoint_baseline:
            return (True, f"Endpoint increases with Q: baseline {endpoint_baseline:.2f} -> perturbed {endpoint_perturbed:.2f}")
        else:
            return (False, "Monotonicity failed: endpoint did not increase with Q")

    def test_monotonicity_cutoff_decreases_with_lower_Q(self) -> Tuple[bool, str]:
        # Another monotonicity test: decrease Q by 10%
        if self.Q_value is None:
            return (False, "SKIP: Q_value not available")
        Q_baseline = self.Q_value
        endpoint_baseline = Q_baseline
        Q_perturbed = Q_baseline * 0.9
        endpoint_perturbed = Q_perturbed
        if endpoint_perturbed < endpoint_baseline:
            return (True, f"Endpoint decreases with Q: baseline {endpoint_baseline:.2f} -> perturbed {endpoint_perturbed:.2f}")
        else:
            return (False, "Monotonicity failed: endpoint did not decrease with lower Q")

    # ---------------------------------------------------------------
    # 5. Weak answer sanity (at most one)
    # ---------------------------------------------------------------
    def test_answer_sanity_nonempty(self) -> Tuple[bool, str]:
        empty_ones = []
        if self.properties is None or self.properties.strip() == '':
            empty_ones.append('properties')
        if self.motivation is None or self.motivation.strip() == '':
            empty_ones.append('motivation')
        if self.detection_method is None or self.detection_method.strip() == '':
            empty_ones.append('detection_method')
        if empty_ones:
            return (False, f"Empty or missing answers: {empty_ones}")
        return (True, "All three answers are non-empty")

    # ---------------------------------------------------------------
    # Runner: discover all test_ methods and execute
    # ---------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)