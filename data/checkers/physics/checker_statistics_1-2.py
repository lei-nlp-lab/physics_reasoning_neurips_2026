import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dict and other attributes
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        # Expected result keys; default None if missing
        self.range_magnetic = self.results.get('range_magnetic', None)
        self.advantage_magnetic = self.results.get('advantage_magnetic', None)
        self.range_optical = self.results.get('range_optical', None)
        self.advantage_optical = self.results.get('advantage_optical', None)
        self.range_vapor = self.results.get('range_vapor', None)
        self.advantage_vapor = self.results.get('advantage_vapor', None)
        # For potential extra context
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------- Interface Checks ----------
    def test_interface_all_keys_present(self):
        missing = []
        for key in ['range_magnetic', 'advantage_magnetic', 'range_optical', 'advantage_optical', 'range_vapor', 'advantage_vapor']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All 6 required result keys present")

    def test_interface_values_are_strings(self):
        if self.range_magnetic is None or self.advantage_magnetic is None or self.range_optical is None or self.advantage_optical is None or self.range_vapor is None or self.advantage_vapor is None:
            return (False, "One or more values are None; cannot check type")
        all_str = all(isinstance(v, str) for v in [self.range_magnetic, self.advantage_magnetic, self.range_optical, self.advantage_optical, self.range_vapor, self.advantage_vapor])
        if not all_str:
            return (False, "Not all values are strings")
        return (True, "All values are strings")

    def test_interface_no_empty_values(self):
        values = [self.range_magnetic, self.advantage_magnetic, self.range_optical, self.advantage_optical, self.range_vapor, self.advantage_vapor]
        if any(v is None for v in values):
            return (False, "Some values are None, cannot check emptiness")
        empty = [k for k, v in zip(['range_magnetic', 'advantage_magnetic', 'range_optical', 'advantage_optical', 'range_vapor', 'advantage_vapor'], values) if v.strip() == '']
        if empty:
            return (False, f"Empty strings for keys: {empty}")
        return (True, "All values are non-empty strings")

    # ---------- Invariant / Principle Checks (qualitative) ----------
    def test_invariant_magnetic_covers_1K(self):
        """Magnetic thermometer must measure down to 1 K (target answer: 'below 1K')."""
        if self.range_magnetic is None:
            return (False, "SKIP: range_magnetic not available")
        text = self.range_magnetic.lower()
        if '1k' not in text.replace(' ', ''):
            return (False, "range_magnetic does not mention '1K' or '1 K'")
        return (True, "range_magnetic includes mention of 1 K")

    def test_invariant_optical_advantage_non_contact(self):
        """Optical pyrometer advantage should mention non-contact / no contact."""
        if self.advantage_optical is None:
            return (False, "SKIP: advantage_optical not available")
        text = self.advantage_optical.lower()
        keywords = ['contact', 'non-contact', 'noncontact', 'remote', 'no contact']
        if not any(kw in text for kw in keywords):
            return (False, "advantage_optical does not mention non-contact or similar")
        return (True, "advantage_optical mentions non-contact measurement")

    def test_invariant_vapor_range_above_14K(self):
        """Vapor pressure thermometer range must be above 14 K (target: >14K)."""
        if self.range_vapor is None:
            return (False, "SKIP: range_vapor not available")
        text = self.range_vapor.lower().replace(' ', '')
        if '14k' not in text and '14' not in text:
            return (False, "range_vapor does not mention '14 K'")
        return (True, "range_vapor mentions 14 K (or 14)")

    def test_invariant_at_least_one_down_to_1K(self):
        """At least one instrument must measure down to 1 K."""
        ranges = [self.range_magnetic, self.range_optical, self.range_vapor]
        if any(r is None for r in ranges):
            return (False, "One or more range values None; cannot verify")
        found = False
        for r in ranges:
            if r is not None and ('1k' in r.lower().replace(' ', '') or '1 k' in r.lower()):
                found = True
                break
        if not found:
            return (False, "No range string mentions 1K or 1 K")
        return (True, "At least one instrument covers down to 1 K")

    # ---------- Modeling / Trace Checks (qualitative) ----------
    def test_modeling_magnetic_principle_keywords(self):
        """Check that magnetic range/advantage mention typical physical concepts (susceptibility, sensitivity)."""
        if self.range_magnetic is None or self.advantage_magnetic is None:
            return (False, "SKIP: magnetic values not available")
        text = (self.range_magnetic + ' ' + self.advantage_magnetic).lower()
        concepts = ['susceptibility', 'magnetic', 'sensitivity', 'curie', 'paramagnet']
        found = [c for c in concepts if c in text]
        if len(found) < 1:
            return (False, "No magnetic physics keyword found in range/advantage")
        return (True, f"Found magnetic physics keywords: {found}")

    def test_modeling_optical_principle_keywords(self):
        """Optical pyrometer should mention radiation, temperature, etc."""
        if self.range_optical is None or self.advantage_optical is None:
            return (False, "SKIP: optical values not available")
        text = (self.range_optical + ' ' + self.advantage_optical).lower()
        concepts = ['radiation', 'thermal', 'infrared', 'light', 'emission', 'blackbody', 'high temperature']
        found = [c for c in concepts if c in text]
        if len(found) < 1:
            return (False, "No optical physics keyword found")
        return (True, f"Found optical physics keywords: {found}")

    def test_modeling_vapor_principle_keywords(self):
        """Vapor pressure thermometer should mention vapor, pressure, cryogenic."""
        if self.range_vapor is None or self.advantage_vapor is None:
            return (False, "SKIP: vapor values not available")
        text = (self.range_vapor + ' ' + self.advantage_vapor).lower()
        concepts = ['vapor', 'pressure', 'boiling', 'cryogenic', 'low temperature', 'clausius', 'clapeyron']
        found = [c for c in concepts if c in text]
        if len(found) < 1:
            return (False, "No vapor pressure physics keyword found")
        return (True, f"Found vapor pressure physics keywords: {found}")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_magnetic_susceptibility_decreases_with_T(self):
        """
        Use Curie's law χ = C/T to verify monotonic behavior.
        Baseline: χ at T1 = 0.5 K. Perturbation: T2 = T1 + 5% (0.525 K).
        Expect χ(T2) < χ(T1).
        This test does not depend on solution output; it validates the underlying physics.
        """
        # Use C=1 for simplicity (ratio independent of C)
        C = 1.0
        T1 = 0.5   # K, baseline
        T2 = T1 * 1.05  # +5% perturbation
        chi1 = C / T1
        chi2 = C / T2
        # Baseline output: chi1, perturbed output: chi2
        if chi2 >= chi1:
            return (False, f"Susceptibility did NOT decrease: χ(T1)={chi1}, χ(T2)={chi2}")
        return (True, f"Susceptibility decreases: χ(T1)={chi1:.4f} -> χ(T2)={chi2:.4f} (decrease of {((chi1-chi2)/chi1*100):.2f}%)")

    # ---------- Answer Sanity (weak, limited) ----------
    def test_answer_sanity_no_contradictory_ranges(self):
        """Rough sanity: optical range should mention high temperatures."""
        if self.range_optical is None:
            return (False, "SKIP: range_optical not available")
        text = self.range_optical.lower()
        if 'high' not in text and 'above' not in text and '1000' not in text:
            return (False, "optical range does not contain 'high' or 'above' typical high-temp indication")
        return (True, "Optical range mentions high temperature qualitatively")

    # ---------- Master Runner ----------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)