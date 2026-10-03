import sympy
import math
import traceback
import re

class PhysicsChecker:
    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.results = results
        self.reaction1 = results.get('reaction1')
        self.reaction2 = results.get('reaction2')
        self.reaction3 = results.get('reaction3')
        self.largest_cross_section = results.get('largest_cross_section')
        self.neutron_use = results.get('neutron_use')

    def _extract_energy(self, s):
        """Return first energy in MeV from a reaction string, or None."""
        if s is None:
            return None
        nums = re.findall(r'(\d+\.?\d*)\s*MeV', s)
        if not nums:
            return None
        try:
            return float(nums[0])
        except ValueError:
            return None

    # ---------- Interface Checks ----------
    def test_interface_required_keys(self):
        if self.results is None:
            return (False, "SKIP: results is None")
        expected_keys = ['reaction1', 'reaction2', 'reaction3',
                         'largest_cross_section', 'neutron_use']
        missing = [k for k in expected_keys if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_reaction_strings_nonempty(self):
        for key in ['reaction1', 'reaction2', 'reaction3']:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"SKIP: {key} is None")
            if not isinstance(val, str) or len(val.strip()) == 0:
                return (False, f"{key} is not a non‑empty string: {val}")
        return (True, "All reaction strings are non‑empty")

    def test_interface_largest_cross_section_string(self):
        val = self.largest_cross_section
        if val is None:
            return (False, "SKIP: largest_cross_section is None")
        if not isinstance(val, str) or len(val.strip()) == 0:
            return (False, f"largest_cross_section is invalid: {val}")
        return (True, "largest_cross_section is a non‑empty string")

    def test_interface_neutron_use_string(self):
        val = self.neutron_use
        if val is None:
            return (False, "SKIP: neutron_use is None")
        if not isinstance(val, str) or len(val.strip()) == 0:
            return (False, f"neutron_use is invalid: {val}")
        return (True, "neutron_use is a non‑empty string")

    # ---------- Invariant Checks (theorem‑based) ----------
    def test_invariant_reaction1_products_correct(self):
        r = self.reaction1
        if r is None:
            return (False, "SKIP: reaction1 is None")
        has_d = 'D' in r
        has_he3 = 'He3' in r or '3He' in r or 'He-3' in r
        has_n = 'n' in r
        if has_d and has_he3 and has_n:
            return (True, "Reaction1 contains D, He3, n")
        return (False, f"Reaction1 does not match expected products: {r}")

    def test_invariant_reaction2_products_correct(self):
        r = self.reaction2
        if r is None:
            return (False, "SKIP: reaction2 is None")
        has_d = 'D' in r
        has_t = 'T' in r or 'tritium' in r
        has_p = 'p' in r or 'proton' in r
        if has_d and has_t and has_p:
            return (True, "Reaction2 contains D, T, p")
        return (False, f"Reaction2 does not match expected products: {r}")

    def test_invariant_reaction3_products_correct(self):
        r = self.reaction3
        if r is None:
            return (False, "SKIP: reaction3 is None")
        has_d = 'D' in r
        has_t = 'T' in r or 'tritium' in r
        has_he4 = 'He4' in r or '4He' in r or 'He-4' in r
        has_n = 'n' in r
        if has_d and has_t and has_he4 and has_n:
            return (True, "Reaction3 contains D, T, He4, n")
        return (False, f"Reaction3 does not match expected products: {r}")

    def test_invariant_largest_cross_section_is_DT(self):
        val = self.largest_cross_section
        if val is None:
            return (False, "SKIP: largest_cross_section is None")
        pattern = r'D\s*[+\-]\s*T|DT|deuterium\s*[- ]?tritium'
        if re.search(pattern, val, re.IGNORECASE):
            return (True, "Largest cross section correctly identifies D+T")
        return (False, f"largest_cross_section does not mention D+T: {val}")

    def test_invariant_neutron_use_includes_breeding(self):
        val = self.neutron_use
        if val is None:
            return (False, "SKIP: neutron_use is None")
        keywords = ['tritium', 'breeding', 'fission', 'Li', 'lithium']
        found = [kw for kw in keywords if kw.lower() in val.lower()]
        if found:
            return (True, f"neutron_use contains keywords: {found}")
        return (False, f"neutron_use lacks expected keywords: {val}")

    # ---------- Modeling/Trace Checks ----------
    def test_modeling_energy_values_positive(self):
        for key in ['reaction1', 'reaction2', 'reaction3']:
            r = getattr(self, key, None)
            if r is None:
                return (False, f"SKIP: {key} is None")
            energy = self._extract_energy(r)
            if energy is None:
                return (False, f"{key}: could not extract energy")
            if energy <= 0:
                return (False, f"{key}: energy {energy} is not positive")
        return (True, "All reactions have positive energy values")

    def test_modeling_dt_energy_is_largest(self):
        e1 = self._extract_energy(self.reaction1)
        e2 = self._extract_energy(self.reaction2)
        e3 = self._extract_energy(self.reaction3)
        if e1 is None or e2 is None or e3 is None:
            return (False, "SKIP: cannot extract energies from one or more reactions")
        if e3 > e1 and e3 > e2:
            return (True, f"D+T energy {e3} MeV is largest among the three")
        return (False, f"D+T energy {e3} is not largest: e1={e1}, e2={e2}, e3={e3}")

    def test_modeling_energies_plausible_range(self):
        for key in ['reaction1', 'reaction2', 'reaction3']:
            r = getattr(self, key, None)
            if r is None:
                return (False, f"SKIP: {key} is None")
            energy = self._extract_energy(r)
            if energy is None:
                return (False, f"{key}: no energy found")
            if not (1.0 <= energy <= 20.0):
                return (False, f"{key}: energy {energy} out of plausible 1‑20 MeV")
        return (True, "All energies are within reasonable range")

    # ---------- Monotonicity Checks (each perturbs an input explicitly) ----------
    def test_monotonicity_energy_ordering_under_perturbation(self):
        """Perturb reaction1 energy upward and verify D+T still largest."""
        e1 = self._extract_energy(self.reaction1)
        e3 = self._extract_energy(self.reaction3)
        if e1 is None or e3 is None:
            return (False, "SKIP: cannot extract energies for monotonicity test")
        perturbed_e1 = e1 * 1.1  # 10% increase
        if math.isclose(e3, perturbed_e1, rel_tol=0.01, abs_tol=0.5):
            return (False, "Perturbation makes energies too close; monotonic order may not be preserved")
        if e3 > perturbed_e1:
            return (True, f"After perturbing reaction1 to {perturbed_e1:.2f} MeV, D+T energy {e3} still larger")
        else:
            return (False, f"After perturbing reaction1 to {perturbed_e1:.2f} MeV, D+T energy {e3} is not larger")

    def test_monotonicity_largest_remains_largest_under_decrease(self):
        """Perturb D+T energy downward and check it still exceeds the others."""
        e1 = self._extract_energy(self.reaction1)
        e2 = self._extract_energy(self.reaction2)
        e3 = self._extract_energy(self.reaction3)
        if e1 is None or e2 is None or e3 is None:
            return (False, "SKIP: cannot extract energies for monotonicity test")
        perturbed_e3 = e3 * 0.9  # 10% decrease
        if perturbed_e3 > e1 and perturbed_e3 > e2:
            return (True, f"After decreasing D+T energy to {perturbed_e3:.2f} MeV, it remains largest: e1={e1}, e2={e2}")
        else:
            return (False, f"After decreasing D+T energy to {perturbed_e3:.2f} MeV, it no longer exceeds both: e1={e1}, e2={e2}")

    # ---------- Answer Sanity (weak, secondary) ----------
    def test_answer_sanity_weak_basic_content(self):
        """Weak sanity: check that answers contain some expected substrings."""
        failures = []
        if self.reaction1 and not ('D' in self.reaction1 and 'He3' in self.reaction1):
            failures.append("reaction1 missing D or He3")
        if self.reaction2 and not ('D' in self.reaction2 and 'T' in self.reaction2):
            failures.append("reaction2 missing D or T")
        if self.reaction3 and not ('D' in self.reaction3 and 'T' in self.reaction3 and 'He4' in self.reaction3):
            failures.append("reaction3 missing D,T,He4")
        if self.largest_cross_section and 'D+T' not in self.largest_cross_section:
            failures.append("largest_cross_section does not contain D+T")
        if self.neutron_use and ('tritium' not in self.neutron_use.lower() and 'fission' not in self.neutron_use.lower()):
            failures.append("neutron_use does not mention tritium or fission")
        if failures:
            return (False, "; ".join(failures))
        return (True, "All answers contain expected basic content")

    # ---------- Runner ----------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)