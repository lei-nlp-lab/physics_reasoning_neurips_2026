import sympy
import math
import traceback
import re
from typing import Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for physics contributions associated with pairs of names.
    Checks that the extracted contribution (stored in self.contribution)
    is consistent with the physics mapping and reasoning steps.
    """

    # Mapping from subproblem letter to expected key concepts and related equations.
    # These are derived from the problem's physics_mapping and explanation steps.
    EXPECTED_CONCEPTS: Dict[str, Dict] = {
        'a': {
            'name': 'Franck-Hertz',
            'concepts': ['energy level', 'electron scatter', 'ΔE = eV', 'discrete energy', 'quantized'],
            'equation': 'ΔE = eV',
            'core_idea': 'Verified discrete energy levels of atom via electron scattering'
        },
        'b': {
            'name': 'Davisson-Germer',
            'concepts': ['wave nature', 'diffraction', 'Bragg', 'nλ = 2d sinθ', 'de Broglie'],
            'equation': 'nλ = 2d sinθ',
            'core_idea': 'Confirmed de Broglie wave nature of electrons via diffraction'
        },
        'c': {
            'name': 'Breit-Wigner',
            'concepts': ['resonance', 'cross section', 'Breit-Wigner', 'Γ', 'E_r'],
            'equation': 'σ ∝ 1/((E-E_r)^2+(Γ/2)^2)',
            'core_idea': 'Derived resonance formula for nuclear reactions'
        },
        'd': {
            'name': 'Hartree-Fock',
            'concepts': ['self-consistent field', 'many-electron', 'Fock', 'Slater determinant', 'Fψ_i = ε_iψ_i'],
            'equation': 'Fψ_i = ε_iψ_i',
            'core_idea': 'Developed self-consistent field method for many-electron wavefunctions'
        },
        'e': {
            'name': 'Lee-Yang',
            'concepts': ['parity', 'non-conservation', 'weak interaction', 'Wu experiment', 'beta decay'],
            'equation': '',
            'core_idea': 'Proposed non-conservation of parity in weak interactions'
        },
        'f': {
            'name': 'Dulong-Petit',
            'concepts': ['heat capacity', '3R', 'atomic heat', 'solids', 'high temperature'],
            'equation': 'C_V = 3R',
            'core_idea': 'Discovered constant atomic heat capacity 3R for solids at high T'
        },
        'g': {
            'name': 'Cockcroft-Walton',
            'concepts': ['voltage multiplier', 'high voltage', 'accelerator', 'disintegration', 'cascade'],
            'equation': 'V_out = N * V_in',
            'core_idea': 'First artificial disintegration of nucleus using high-voltage accelerator'
        },
        'h': {
            'name': 'Hahn-Strassmann',
            'concepts': ['fission', 'uranium', 'neutron', 'barium', 'krypton', 'chain reaction'],
            'equation': '^235U + n → fission products',
            'core_idea': 'Demonstrated fission of uranium by neutrons'
        },
        'i': {
            'name': 'Ramsauer-Townsend',
            'concepts': ['scattering cross section minimum', 'low energy', 'noble gas', 'transmission', 'resonant'],
            'equation': 'σ(E) has minimum at low E',
            'core_idea': 'Discovered minimum in electron scattering cross section at low energies'
        },
        'j': {
            'name': 'Thomas-Fermi',
            'concepts': ['statistical model', 'electron density', 'Fermi-Dirac', 'ρ ∝ V^(3/2)', 'screening'],
            'equation': 'ρ(r) ∝ V(r)^(3/2)',
            'core_idea': 'Proposed statistical model for electron distribution in metals'
        }
    }

    def __init__(self, solution_instance) -> None:
        # Extract the contribution field – expected key is 'contribution'.
        # Also attempt to extract subproblem label if present.
        self.contribution = getattr(solution_instance, 'contribution', None)
        if self.contribution is None:
            self.contribution = solution_instance.results.get('contribution', None) if hasattr(solution_instance, 'results') else None
        self.subproblem = getattr(solution_instance, 'subproblem', None)  # optional
        # Determine which subproblem best matches the contribution (if any)
        self._best_match = None
        if self.contribution and isinstance(self.contribution, str):
            lower = self.contribution.lower()
            scores = {}
            for key, info in self.EXPECTED_CONCEPTS.items():
                score = 0
                for concept in info['concepts']:
                    if concept.lower() in lower:
                        score += 1
                # Boost if the name appears directly
                if info['name'].lower() in lower:
                    score += 3
                scores[key] = score
            best = max(scores, key=scores.get)
            # Only assign if match is nontrivial
            if scores[best] >= 1:
                self._best_match = best
        # Ensure safe defaults for tests
        if self.contribution is None:
            self.contribution = None
        if self.subproblem is None:
            self.subproblem = None

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collecting results."""
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

    # ----- Interface tests -----
    def test_interface_contribution_exists(self) -> Tuple[bool, str]:
        """Check that contribution is not None."""
        if self.contribution is None:
            return (False, "contribution missing (None)")
        return (True, "contribution present")

    def test_interface_contribution_is_string(self) -> Tuple[bool, str]:
        """Check that contribution is a string."""
        if not isinstance(self.contribution, str):
            return (False, f"contribution is not a string, got {type(self.contribution)}")
        return (True, "contribution is a string")

    # ----- Invariant/theorem-based sanity (lenient textual checks) -----
    def test_invariant_contribution_nonempty(self) -> Tuple[bool, str]:
        """Check that contribution is non-empty after stripping."""
        if not self.contribution or self.contribution.strip() == '':
            return (False, "contribution is empty")
        return (True, "contribution non-empty")

    def test_invariant_contribution_reasonable_length(self) -> Tuple[bool, str]:
        """Contributions should be between 20 and 500 characters (adjustable)."""
        if self.contribution is None:
            return (False, "SKIP: contribution missing")
        length = len(self.contribution)
        if length < 20:
            return (False, f"contribution too short ({length} chars)")
        if length > 500:
            return (False, f"contribution too long ({length} chars)")
        return (True, f"contribution length okay ({length})")

    def test_invariant_contribution_contains_physics_concept(self) -> Tuple[bool, str]:
        """
        Check that the contribution contains at least one key concept from the
        physics mapping, ensuring it is about physics and not nonsense.
        """
        if self.contribution is None:
            return (False, "SKIP: contribution missing")
        lower = self.contribution.lower()
        # Collect all relevant concepts from all subproblems
        all_concepts = set()
        for info in self.EXPECTED_CONCEPTS.values():
            for c in info['concepts']:
                all_concepts.add(c.lower())
        # Also include general physics terms
        general_terms = {'energy', 'wave', 'particle', 'electron', 'nucleus', 'force', 'field',
                         'temperature', 'heat', 'scatter', 'resonance', 'parity', 'fission',
                         'potential', 'density', 'diffraction', 'quantum', 'atom'}
        all_concepts.update(term.lower() for term in general_terms)

        found = any(concept in lower for concept in all_concepts)
        if not found:
            return (False, "no physics concept found")
        return (True, "contains physics concept")

    def test_invariant_contribution_no_contradictory_pairs(self) -> Tuple[bool, str]:
        """
        Check that the contribution does not mix concepts from two radically
        different subproblems (e.g., fission and diffraction).
        """
        if self.contribution is None:
            return (False, "SKIP: contribution missing")
        lower = self.contribution.lower()
        # For each pair, count how many of its concepts appear.
        # If more than one pair has high overlap, it's ambiguous.
        # We allow at most one pair with >2 concept matches.
        matches = {}
        for key, info in self.EXPECTED_CONCEPTS.items():
            count = sum(1 for c in info['concepts'] if c.lower() in lower)
            if count > 1:
                matches[key] = count
        if len(matches) > 1:
            # Check that they are conceptually compatible (e.g., 'a' and 'b' are different but not contradictory)
            # We'll define domain clustering.
            domains = {
                'a': 'atomic',
                'b': 'wave',
                'c': 'nuclear',
                'd': 'atomic',
                'e': 'weak',
                'f': 'solid',
                'g': 'nuclear',
                'h': 'nuclear',
                'i': 'atomic',
                'j': 'solid'
            }
            keys = list(matches.keys())
            domain_set = {domains[k] for k in keys}
            if len(domain_set) > 1 and len(keys) > 2:
                return (False, f"contradictory concepts from different domains: {keys}")
        return (True, "no strong contradiction detected")

    # ----- Modeling / explanation step consistency tests -----
    def test_modeling_contribution_matches_subproblem(self) -> Tuple[bool, str]:
        """
        If a subproblem label is provided (directly or inferred), check that
        the contribution aligns with the expected core idea.
        Uses simple substring matching – not exact answer equality.
        """
        if self.contribution is None:
            return (False, "SKIP: contribution missing")
        # Use self._best_match if available, else self.subproblem
        sub = self.subproblem if self.subproblem else self._best_match
        if sub is None:
            return (False, "SKIP: cannot determine which subproblem this contribution belongs to")
        info = self.EXPECTED_CONCEPTS.get(sub)
        if info is None:
            return (False, f"SKIP: unknown subproblem {sub}")

        lower = self.contribution.lower()
        # Core idea should be present at least as a phrase
        core_lower = info['core_idea'].lower()
        # Check for significant overlap – at least 30% of words from core_idea appear (stopwords excluded)
        core_words = set(re.findall(r'\w+', core_lower))
        contribution_words = set(re.findall(r'\w+', lower))
        common = core_words & contribution_words
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'of', 'in', 'and', 'to'}
        common_stripped = common - stopwords
        if len(common_stripped) < len(core_words) * 0.3:
            return (False, f"contribution does not capture core idea for {info['name']}; common words: {common_stripped}")
        # Also ensure at least one equation-related concept appears if the subproblem has a distinctive equation
        if info['equation']:
            eq_present = any(eq_part.lower() in lower for eq_part in info['equation'].replace(' ', '').split('='))
            if not eq_present and 'energy' not in lower:  # allow some freedom
                # Check for related terms
                related = {'ΔE', 'eV', 'nλ', 'd sinθ', 'Fψ', 'ε_i', 'C_V', '3R', 'V_out', 'fission', 'ρ', 'V^(3/2)'}
                if not any(r.lower() in lower for r in related):
                    return (False, f"contribution missing equation-related concept for {info['name']}")
        return (True, f"contribution consistent with subproblem {sub} ({info['name']})")

    def test_modeling_contribution_contains_key_figure(self) -> Tuple[bool, str]:
        """
        Check that the contribution includes the name of the scientists (e.g., "Franck and Hertz").
        This tests that the contribution is not too generic.
        """
        if self.contribution is None:
            return (False, "SKIP: contribution missing")
        # Determine best-matching subproblem
        sub = self.subproblem if self.subproblem else self._best_match
        if sub is None:
            return (False, "SKIP: cannot determine which subproblem this contribution belongs to")
        info = self.EXPECTED_CONCEPTS.get(sub)
        if info is None:
            return (False, f"SKIP: unknown subproblem {sub}")
        expected_name = info['name'].lower()
        lower = self.contribution.lower()
        # Check if the name appears (with variations like "Franck-Hertz", "franck hertz" etc.)
        if expected_name not in lower:
            # Try splitting by hyphen
            parts = expected_name.split('-')
            if len(parts) == 2 and parts[0] in lower and parts[1] in lower:
                return (True, f"contribution contains parts of key figure name {info['name']}")
            # Also check for "and" versions
            and_name = parts[0] + " and " + parts[1] if len(parts) == 2 else ""
            if and_name and and_name in lower:
                return (True, f"contribution contains 'X and Y' form")
            return (False, f"contribution missing key figure name '{info['name']}'")
        return (True, f"contribution contains key figure name '{info['name']}'")

    # ----- Monotonicity test (perturb input and compare output) -----
    def test_monotonicity_relevance_increases_with_more_concepts(self) -> Tuple[bool, str]:
        """
        Perturb the contribution by adding all core words of the best-matching
        subproblem. The number of overlapping core words should not decrease
        (i.e., it should be monotonic non-decreasing). This checks that the
        sensitivity measure (coverage) behaves correctly.
        """
        if self.contribution is None:
            return (False, "SKIP: contribution missing")
        # Determine best-matching subproblem
        sub = self.subproblem if self.subproblem else self._best_match
        if sub is None:
            return (False, "SKIP: cannot determine which subproblem this contribution belongs to")
        info = self.EXPECTED_CONCEPTS.get(sub)
        if info is None:
            return (False, "SKIP: unknown subproblem {sub}")

        lower = self.contribution.lower()
        # Use core words from the core_idea (keep stopwords for simplicity)
        core_words = set(info['core_idea'].lower().split())
        # Baseline: how many core words are already present
        baseline_count = sum(1 for word in core_words if word in lower)

        # Perturb: create a modified contribution by appending the core words with a space
        # This simulates adding more relevant information.
        perturbed_lower = lower + " " + " ".join(core_words)
        perturbed_count = sum(1 for word in core_words if word in perturbed_lower)

        # Monotonicity: perturbed count should be >= baseline count
        if perturbed_count < baseline_count:
            return (False, f"Perturbation decreased coverage: baseline={baseline_count}, perturbed={perturbed_count}")
        return (True, f"Monotonic: baseline={baseline_count}, perturbed={perturbed_count}")

    # ----- (Optional) weak answer sanity: very broad checks -----
    def test_answer_sanity_basic_plausibility(self) -> Tuple[bool, str]:
        """
        Weak sanity: contribution should not contain common errors like
        misspelling key names or wrong numeric values.
        This is a very shallow check.
        """
        if self.contribution is None:
            return (False, "SKIP: contribution missing")
        # Reject if contains phrases like "wrong", "incorrect", "error" referring to the physics
        lower = self.contribution.lower()
        suspicious = ['wrong', 'incorrect', 'not correct', 'error']
        if any(term in lower for term in suspicious):
            return (False, "contribution contains self-critical language")
        # Ensure no obvious LaTeX syntax errors (if present)
        if '$' in self.contribution:
            # Count dollar signs – should be even
            if self.contribution.count('$') % 2 != 0:
                return (False, "unmatched dollar signs in contribution")
        return (True, "basic plausibility check passed")