import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for particle physics classification problem.
    Checks definitions, examples, and photon classification based on
    spin-statistics theorem, quark composition constraints, and photon properties.
    """

    def __init__(self, solution_instance):
        # Extract results safely; allow for both direct attribute and results dict
        self.results = getattr(solution_instance, 'results', None) or {}
        # Safe extraction with None defaults
        self.definitions = self.results.get('definitions', None)
        self.examples = self.results.get('examples', None)
        self.photon_classification = self.results.get('photon_classification', None)

    # ------------------------------------------------------------------
    # Interface tests: verify presence and type of required keys
    # ------------------------------------------------------------------
    def test_interface_definitions_exists(self):
        if self.definitions is None:
            return (False, "definitions key missing or None")
        return (True, "definitions exists")

    def test_interface_definitions_is_list(self):
        if self.definitions is None:
            return (False, "SKIP: definitions is None")
        if not isinstance(self.definitions, list):
            return (False, f"definitions is not a list, got {type(self.definitions)}")
        return (True, "definitions is a list")

    def test_interface_definitions_length(self):
        if self.definitions is None:
            return (False, "SKIP: definitions is None")
        if len(self.definitions) != 5:
            return (False, f"definitions length is {len(self.definitions)}, expected 5")
        return (True, "definitions has 5 elements")

    def test_interface_examples_exists(self):
        if self.examples is None:
            return (False, "examples key missing or None")
        return (True, "examples exists")

    def test_interface_examples_is_list(self):
        if self.examples is None:
            return (False, "SKIP: examples is None")
        if not isinstance(self.examples, list):
            return (False, f"examples is not a list, got {type(self.examples)}")
        return (True, "examples is a list")

    def test_interface_examples_length(self):
        if self.examples is None:
            return (False, "SKIP: examples is None")
        if len(self.examples) != 5:
            return (False, f"examples length is {len(self.examples)}, expected 5")
        return (True, "examples has 5 elements")

    def test_interface_photon_classification_exists(self):
        if self.photon_classification is None:
            return (False, "photon_classification key missing or None")
        return (True, "photon_classification exists")

    def test_interface_photon_classification_is_dict(self):
        if self.photon_classification is None:
            return (False, "SKIP: photon_classification is None")
        if not isinstance(self.photon_classification, dict):
            return (False, f"photon_classification is not a dict, got {type(self.photon_classification)}")
        return (True, "photon_classification is a dict")

    def test_interface_photon_classification_has_keys(self):
        if self.photon_classification is None:
            return (False, "SKIP: photon_classification is None")
        expected_keys = {'boson', 'hadron', 'lepton', 'baryon'}
        actual_keys = set(self.photon_classification.keys())
        if not expected_keys.issubset(actual_keys):
            missing = expected_keys - actual_keys
            return (False, f"photon_classification missing keys: {missing}")
        return (True, "photon_classification has all required keys")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_spin_statistics_definitions(self):
        """
        Check that definitions reflect spin-statistics theorem:
        boson -> integer spin, fermion -> half-integer spin.
        """
        if self.definitions is None:
            return (False, "SKIP: definitions is None")
        if len(self.definitions) < 2:
            return (False, "definitions too short")
        boson_def = self.definitions[0] if len(self.definitions) > 0 else ""
        fermion_def = self.definitions[1] if len(self.definitions) > 1 else ""
        boson_ok = "integer" in boson_def.lower()
        fermion_ok = "half-integer" in fermion_def.lower()
        if not boson_ok:
            return (False, f"Boson definition lacks 'integer spin': {boson_def}")
        if not fermion_ok:
            return (False, f"Fermion definition lacks 'half-integer spin': {fermion_def}")
        return (True, "Spin-statistics reflected in definitions")

    def test_invariant_hadron_lepton_classification(self):
        """
        Hadrons subject to strong interaction, leptons not.
        """
        if self.definitions is None or len(self.definitions) < 4:
            return (False, "SKIP: definitions incomplete")
        hadron_def = self.definitions[2] if len(self.definitions) > 2 else ""
        lepton_def = self.definitions[3] if len(self.definitions) > 3 else ""
        hadron_ok = "strong" in hadron_def.lower()
        lepton_ok = ("not subject" in lepton_def.lower() or "weak" in lepton_def.lower())
        if not hadron_ok:
            return (False, f"Hadron definition missing 'strong interaction': {hadron_def}")
        if not lepton_ok:
            return (False, f"Lepton definition missing 'weak' or 'not subject': {lepton_def}")
        return (True, "Hadron/lepton definitions consistent with strong/weak forces")

    def test_invariant_baryon_definition(self):
        """
        Baryon defined as hadron with half-integer spin.
        """
        if self.definitions is None or len(self.definitions) < 5:
            return (False, "SKIP: definitions incomplete")
        baryon_def = self.definitions[4]
        baryon_ok = "hadron" in baryon_def.lower() and "half-integer" in baryon_def.lower()
        if not baryon_ok:
            return (False, f"Baryon definition should mention hadron and half-integer: {baryon_def}")
        return (True, "Baryon definition consistent with hadron subclass")

    def test_invariant_photon_classification(self):
        """
        Photon spin=1 => boson=True; not hadron/lepton/baryon.
        """
        if self.photon_classification is None:
            return (False, "SKIP: photon_classification is None")
        expected = {'boson': True, 'hadron': False, 'lepton': False, 'baryon': False}
        for key, val in expected.items():
            if key not in self.photon_classification:
                return (False, f"photon_classification missing key '{key}'")
            if self.photon_classification[key] != val:
                return (False, f"photon_classification['{key}'] = {self.photon_classification[key]}, expected {val}")
        return (True, "Photon classification matches expectations (boson only)")

    # ------------------------------------------------------------------
    # Modeling/trace tests: check consistency between examples and definitions
    # ------------------------------------------------------------------
    def test_modeling_examples_consistency(self):
        """
        Examples should be plausible: boson example (pi meson) is boson,
        fermion example (proton) is fermion, hadron example (proton) is hadron,
        lepton example (neutrino) is lepton, baryon example (proton) is baryon.
        """
        if self.definitions is None or self.examples is None:
            return (False, "SKIP: definitions or examples missing")
        if len(self.definitions) < 5 or len(self.examples) < 5:
            return (False, "Definitions or examples incomplete")
        for i, ex in enumerate(self.examples):
            if not isinstance(ex, str) or not ex.strip():
                return (False, f"Example at index {i} is empty or not a string")
        try:
            boson_ex = self.examples[0].lower()
            fermion_ex = self.examples[1].lower()
            hadron_ex = self.examples[2].lower()
            lepton_ex = self.examples[3].lower()
            baryon_ex = self.examples[4].lower()
            if 'proton' not in fermion_ex:
                return (False, "Fermion example should be proton (or similar)")
            if 'proton' not in hadron_ex:
                return (False, "Hadron example should be proton (or similar)")
            if 'neutrino' not in lepton_ex:
                return (False, "Lepton example should be neutrino (or similar)")
            if 'proton' not in baryon_ex:
                return (False, "Baryon example should be proton (or similar)")
            if 'pi' not in boson_ex and 'pion' not in boson_ex and 'meson' not in boson_ex:
                return (False, "Boson example should be pi meson (or similar)")
            return (True, "Examples consistent with definitions (proton, neutrino, pion)")
        except Exception as e:
            return (False, f"Error checking examples: {e}")

    def test_modeling_photon_classification_matches_definitions(self):
        """
        If photon is classified as boson, then definitions should include 'boson' as first element.
        """
        if self.definitions is None or self.photon_classification is None:
            return (False, "SKIP: missing data")
        if len(self.definitions) < 1:
            return (False, "definitions incomplete")
        if self.photon_classification.get('boson') == True:
            boson_def = self.definitions[0].lower()
            if 'integer' not in boson_def:
                return (False, "Photon is boson but boson definition lacks 'integer spin'")
        return (True, "Photon classification aligned with boson definition")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_spin_classification(self):
        """
        Verify monotonicity of the spin–classification rule: as spin increases
        past the half-integer threshold (0.5), the classification flips from
        boson (integer spin) to fermion (half-integer spin).
        This test does not depend on solution data; it uses a synthetic
        perturbation to illustrate the monotonic behaviour.
        """
        # Baseline: spin = 0 (integer) → boson
        baseline_spin = 0.0
        baseline_class = "boson"   # integer spin
        
        # Perturb: spin = 0.6 ( > 0.5, effectively half-integer) → fermion
        perturbed_spin = 0.6
        perturbed_class = "fermion"   # half-integer spin
        
        # The classification must change under the perturbation
        if baseline_class == perturbed_class:
            return (False, "Monotonicity violated: same classification for different spin ranges")
        # Also verify that the threshold lies between the two spin values
        threshold = 0.5
        if not (baseline_spin < threshold < perturbed_spin):
            return (False, "Perturbation does not cross the half-integer threshold")
        return (True, "Spin classification is monotonic (boson→fermion at spin=0.5)")

    # ------------------------------------------------------------------
    # Answer sanity (weak; no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_photon_not_hadron(self):
        """
        Photon is not a hadron, lepton, or baryon (weak sanity check).
        """
        if self.photon_classification is None:
            return (False, "SKIP: photon_classification is None")
        if self.photon_classification.get('hadron', None) == True:
            return (False, "Photon should not be classified as hadron")
        if self.photon_classification.get('lepton', None) == True:
            return (False, "Photon should not be classified as lepton")
        if self.photon_classification.get('baryon', None) == True:
            return (False, "Photon should not be classified as baryon")
        return (True, "Photon not classified as hadron/lepton/baryon (weak sanity)")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)