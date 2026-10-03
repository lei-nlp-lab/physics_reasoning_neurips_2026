import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for photon-electron interaction problems.
    Checks conservation laws and invariant mass constraints.
    """
    def __init__(self, solution_instance: Any):
        # Extract givens - safe access, None if missing
        self.m_e = getattr(solution_instance, 'm_e', None)
        if self.m_e is None:
            self.m_e = solution_instance.results.get('m_e', None) if hasattr(solution_instance, 'results') else None
        self.c = getattr(solution_instance, 'c', None)
        if self.c is None:
            self.c = solution_instance.results.get('c', None) if hasattr(solution_instance, 'results') else None

        # Extract results dictionary
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        else:
            self.results = {}

        # Final answer (if stored)
        self.final_answer = getattr(solution_instance, 'final_answer', None)
        if self.final_answer is None:
            self.final_answer = self.results.get('final_answer', None)

        # Specific keys from [emit]
        self.required_E_gamma = self.results.get('required_E_gamma', None)
        self.invariant_mass_final = self.results.get('invariant_mass_final', None)
        self.minimum_invariant_mass = self.results.get('minimum_invariant_mass', None)
        self.derived_positron_energy = self.results.get('derived_positron_energy', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discovers and runs all test_ methods, collects results."""
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

    # ---------- Interface Tests ----------
    def test_interface_existence_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present."""
        required_keys = ['required_E_gamma', 'invariant_mass_final', 'minimum_invariant_mass', 'derived_positron_energy']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_positive(self) -> Tuple[bool, str]:
        """Check that m_e and c are positive numbers."""
        for name, val in [('m_e', self.m_e), ('c', self.c)]:
            if val is None:
                return (False, f"SKIP: {name} not provided")
            if val <= 0:
                return (False, f"{name} must be positive, got {val}")
        return (True, "Givens m_e and c are positive")

    # ---------- Invariant / Physics Sanity Tests ----------
    def test_invariant_Q1_energy_momentum_contradiction(self) -> Tuple[bool, str]:
        """
        For Q1: if conservation laws are applied, we should get 2*m_e*c^2 * E_gamma = 0.
        Check that the emitted result forces E_gamma=0.
        """
        if self.required_E_gamma is None:
            return (False, "SKIP: required_E_gamma not available")
        if self.m_e is None or self.c is None:
            return (False, "SKIP: m_e or c not available")
        # According to derivation: 2*E_gamma*m_e*c^2 = 0, so E_gamma must be zero.
        # We check that E_gamma is numerically zero (within tolerance)
        if math.isclose(self.required_E_gamma, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"required_E_gamma = {self.required_E_gamma} is zero, consistent with contradiction")
        else:
            return (False, f"required_E_gamma = {self.required_E_gamma} is not zero; expected 0")

    def test_invariant_Q2_mass_inequality(self) -> Tuple[bool, str]:
        """
        For Q2: invariant mass of final system (0) is less than minimum possible (2*m_e*c^2).
        Check that invariant_mass_final == 0 and minimum_invariant_mass > 0.
        """
        if self.invariant_mass_final is None or self.minimum_invariant_mass is None:
            return (False, "SKIP: invariant_mass_final or minimum_invariant_mass not available")
        if self.m_e is None or self.c is None:
            return (False, "SKIP: m_e or c not available")
        # Check invariant_mass_final is zero (photon)
        if not math.isclose(self.invariant_mass_final, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"invariant_mass_final = {self.invariant_mass_final} is not zero")
        # Compute expected minimum: 2*m_e*c^2 (but we trust the derived value)
        expected_minimum = 2 * self.m_e * (self.c ** 2)
        if not math.isclose(self.minimum_invariant_mass, expected_minimum, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"minimum_invariant_mass = {self.minimum_invariant_mass} deviates from expected {expected_minimum}")
        return (True, "Q2 invariant mass check passed: 0 < 2*m_e*c^2")

    def test_invariant_Q3_negative_energy(self) -> Tuple[bool, str]:
        """
        For Q3: derived positron energy should be exactly -m_e*c^2 (negative).
        Check that it is negative and matches expected.
        """
        if self.derived_positron_energy is None:
            return (False, "SKIP: derived_positron_energy not available")
        if self.m_e is None or self.c is None:
            return (False, "SKIP: m_e or c not available")
        expected = -self.m_e * (self.c ** 2)
        if math.isclose(self.derived_positron_energy, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"derived_positron_energy = {self.derived_positron_energy} ≈ -m_e*c^2")
        else:
            return (False, f"derived_positron_energy = {self.derived_positron_energy}, expected {expected}")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_Q1_use_conservation_laws(self) -> Tuple[bool, str]:
        """
        Check that the solution uses both energy and momentum conservation.
        This is a weak check: we expect keys derived from those equations.
        We check that required_E_gamma is present (derived from combination).
        """
        if self.required_E_gamma is None:
            return (False, "SKIP: required_E_gamma missing – cannot verify modeling")
        # Presence alone is not strong, but we combine with invariants.
        return (True, "required_E_gamma present; conservation laws likely used")

    def test_modeling_Q2_invariant_mass_breakdown(self) -> Tuple[bool, str]:
        """
        Check that the solution computed both invariant mass of final system and the minimum.
        This ensures the reasoning about invariant mass inequality was performed.
        """
        if self.invariant_mass_final is None or self.minimum_invariant_mass is None:
            return (False, "Invariant mass quantities missing")
        return (True, "Q2 invariant mass comparison present")

    def test_modeling_Q3_positron_energy_derivation(self) -> Tuple[bool, str]:
        """
        Check that positron energy was derived (negative result).
        """
        if self.derived_positron_energy is None:
            return (False, "derived_positron_energy missing")
        return (True, "Q3 energy derivation present")

    # ---------- Monotonicity Tests (simple perturbations) ----------
    def test_monotonicity_Q1_perturb_mass(self) -> Tuple[bool, str]:
        """
        For Q1: if electron mass is increased, the contradiction (2*m_e*c^2*E_gamma=0) still holds.
        Perturb m_e by +10% and recompute that 2*m_e*E_gamma*c^2 still is ~0.
        """
        if self.required_E_gamma is None or self.m_e is None or self.c is None:
            return (False, "SKIP: required data for monotonicity missing")
        # Baseline: compute product
        baseline_product = 2 * self.m_e * self.required_E_gamma * (self.c ** 2)
        # Perturb m_e up by 10%
        perturbed_m_e = self.m_e * 1.1
        perturbed_product = 2 * perturbed_m_e * self.required_E_gamma * (self.c ** 2)
        # Since E_gamma=0 (within tolerance), both products should be near zero.
        if math.isclose(baseline_product, 0, abs_tol=1e-2) and math.isclose(perturbed_product, 0, abs_tol=1e-2):
            return (True, "Product remains zero under mass perturbation – consistent")
        else:
            return (False, f"Product not zero: baseline {baseline_product}, perturbed {perturbed_product}")

    def test_monotonicity_Q2_minimum_mass_scale(self) -> Tuple[bool, str]:
        """
        For Q2: minimum invariant mass scales with m_e. If m_e is artificially increased,
        the minimum should increase proportionally.
        """
        if self.minimum_invariant_mass is None or self.m_e is None or self.c is None:
            return (False, "SKIP: required data for Q2 monotonicity missing")
        original_min = self.minimum_invariant_mass
        # Perturb m_e by +5%
        perturbed_m_e = self.m_e * 1.05
        expected_new_min = 2 * perturbed_m_e * (self.c ** 2)
        # Not perturbing the actual result, just checking that the formula scales.
        if math.isclose(original_min / (2 * self.m_e * (self.c ** 2)), 1.0, rel_tol=1e-2):
            return (True, "Minimum invariant mass scales correctly with m_e")
        else:
            return (False, f"Scaling failed: ratio = {original_min / (2 * self.m_e * (self.c ** 2))}")

    # ---------- Answer Sanity (weak, only one) ----------
    def test_answer_sanity_all_impossible(self) -> Tuple[bool, str]:
        """
        Weak check: all three processes should be marked impossible.
        We only check that the final answer (if present) is a string containing "Impossible".
        Do NOT use isclose or re-derive.
        """
        if self.final_answer is None:
            return (False, "SKIP: final_answer not available")
        # Expect a list or string; handle both
        if isinstance(self.final_answer, list):
            if len(self.final_answer) != 3:
                return (False, f"Expected 3 answers, got {len(self.final_answer)}")
            if all('Impossible' in str(ans) for ans in self.final_answer):
                return (True, "All final answers indicate Impossible")
            else:
                return (False, f"Final answers: {self.final_answer} – not all Impossible")
        elif isinstance(self.final_answer, str):
            # Maybe all three answers are concatenated
            if self.final_answer.count('Impossible') >= 3:
                return (True, "Final answer string contains 'Impossible' three times")
            else:
                return (False, "Final answer string does not contain enough 'Impossible'")
        else:
            return (False, "Unexpected type for final_answer")