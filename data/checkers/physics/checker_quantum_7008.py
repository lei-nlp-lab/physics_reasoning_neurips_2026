import math
import traceback
from typing import Any, Dict, Tuple, List, Optional


class PhysicsChecker:
    """
    Robust checker for the 1D infinite square well with electrons.
    Tests focus on physical invariants, modeling consistency, and monotonicity.
    Does NOT hardcode the final answer values; instead verifies relations.
    """

    def __init__(self, solution_instance: Any):
        # --- Safe extraction of givens and results ---
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.a = givens.get('a', None)                # well width (Å)
        self.T = givens.get('T', None)                # temperature (K)
        self.N3 = givens.get('N3', None)              # number of electrons in first case
        self.N4 = givens.get('N4', None)              # number of electrons in second case
        self.E_avg_3 = givens.get('E_avg_3', None)    # average energy for 3 electrons (eV)

        # Results (emitted by pseudocode)
        results = getattr(solution_instance, 'results', {}) or {}
        self.E1 = results.get('E1', None)              # ground state energy (eV)
        self.E_avg_4 = results.get('E_avg_4', None)    # average energy for 4 electrons (eV)

        # Additional results that might be present (but not required)
        self.E2 = results.get('E2', None)              # first excited state (eV)
        self.filling = results.get('filling', None)    # list of occupancy per level

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that required result keys exist and are numeric."""
        if self.E1 is None:
            return (False, "Missing required result key 'E1'")
        if self.E_avg_4 is None:
            return (False, "Missing required result key 'E_avg_4'")
        if not isinstance(self.E1, (int, float)):
            return (False, "'E1' is not numeric")
        if not isinstance(self.E_avg_4, (int, float)):
            return (False, "'E_avg_4' is not numeric")
        if math.isnan(self.E1) or math.isinf(self.E1):
            return (False, "'E1' is NaN or inf")
        if math.isnan(self.E_avg_4) or math.isinf(self.E_avg_4):
            return (False, "'E_avg_4' is NaN or inf")
        return (True, "All required keys present and numeric")

    def test_interface_given_E_avg_3(self) -> Tuple[bool, str]:
        """If E_avg_3 is provided, check it's numeric and positive."""
        if self.E_avg_3 is None:
            return (False, "SKIP: E_avg_3 not available")
        if not isinstance(self.E_avg_3, (int, float)):
            return (False, "'E_avg_3' is not numeric")
        if self.E_avg_3 <= 0:
            return (False, "E_avg_3 must be positive")
        return (True, "E_avg_3 is numeric and positive")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_energy_scaling(self) -> Tuple[bool, str]:
        """If E2 is present, check E2 = 4*E1 (from particle-in-a-box)."""
        if self.E2 is None:
            return (True, "SKIP: E2 not provided")
        if not math.isclose(self.E2, 4 * self.E1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E2={self.E2} should be 4*E1={4*self.E1}")
        return (True, "E2 = 4*E1")

    def test_invariant_energy_positive(self) -> Tuple[bool, str]:
        """All energies must be positive."""
        if self.E1 is not None and self.E1 <= 0:
            return (False, f"E1={self.E1} is not positive")
        if self.E_avg_4 is not None and self.E_avg_4 <= 0:
            return (False, f"E_avg_4={self.E_avg_4} is not positive")
        if self.E2 is not None and self.E2 <= 0:
            return (False, f"E2={self.E2} is not positive")
        return (True, "All provided energies are positive")

    # ------------------------------------------------------------
    # Modeling/trace tests (check explanation steps)
    # ------------------------------------------------------------
    def test_modeling_average_4_from_E1(self) -> Tuple[bool, str]:
        """
        For 4 electrons at T=0: fill n=1 (2 e-) and n=2 (2 e-).
        Average = (2*E1 + 2*4*E1)/4 = 2.5*E1.
        """
        if self.E1 is None or self.E_avg_4 is None:
            return (False, "SKIP: E1 or E_avg_4 missing")
        expected = 2.5 * self.E1
        if not math.isclose(self.E_avg_4, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_avg_4={self.E_avg_4} should be 2.5*E1={expected}")
        return (True, f"E_avg_4 = 2.5*E1 = {expected}")

    def test_modeling_average_3_from_E1(self) -> Tuple[bool, str]:
        """
        If E_avg_3 is given, check that E1 = E_avg_3/2.
        For 3 electrons: average = 2*E1 (filling 2 in n=1, 1 in n=2).
        """
        if self.E_avg_3 is None or self.E1 is None:
            return (False, "SKIP: need both E_avg_3 and E1")
        expected_E1 = self.E_avg_3 / 2.0
        if not math.isclose(self.E1, expected_E1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E1={self.E1} should equal E_avg_3/2={expected_E1}")
        return (True, f"E1 = E_avg_3/2 = {expected_E1}")

    def test_modeling_consistent_filling(self) -> Tuple[bool, str]:
        """
        If filling data is provided, check that at most 2 per spatial state.
        Only run if filling is a list.
        """
        if self.filling is None:
            return (True, "SKIP: filling not provided")
        if not isinstance(self.filling, list):
            return (False, "filling must be a list")
        for i, count in enumerate(self.filling, start=1):
            if count > 2:
                return (False, f"Level n={i} has {count} electrons, max 2")
        return (True, "Filling respects Pauli principle")

    # ------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input)
    # ------------------------------------------------------------
    def test_monotonicity_average_increase_with_Eavg3(self) -> Tuple[bool, str]:
        """
        Perturb E_avg_3 upward by 1% and verify that E_avg_4 (computed from relation)
        increases.  The relation E_avg_4 = 1.25 * E_avg_3 is derived from
        E1 = E_avg_3/2 and E_avg_4 = 2.5*E1.
        If E_avg_3 is missing or non‑numeric, skip.
        """
        if self.E_avg_3 is None:
            return (True, "SKIP: E_avg_3 not available for perturbation")
        if not isinstance(self.E_avg_3, (int, float)):
            return (False, "'E_avg_3' is not numeric")
        epsilon = 0.01  # 1% increase
        baseline_Eavg4 = 1.25 * self.E_avg_3
        perturbed_Eavg3 = self.E_avg_3 * (1 + epsilon)
        perturbed_Eavg4 = 1.25 * perturbed_Eavg3
        if perturbed_Eavg4 <= baseline_Eavg4:
            return (False, f"Perturbed E_avg_4 ({perturbed_Eavg4:.6f}) "
                           f"should be larger than baseline ({baseline_Eavg4:.6f})")
        return (True, f"1% increase in E_avg_3 yields ~{epsilon*100:.1f}% larger E_avg_4")

    def test_monotonicity_well_width(self) -> Tuple[bool, str]:
        """
        If a is available, perturb it upward by 1% and check that the
        resulting E_avg_4 (derived from E1 ∝ 1/a² and E_avg_4 = 2.5 E1)
        decreases.  Uses the baseline self.E1 to compute the scaling.
        If a or E1 missing, skip.
        """
        if self.a is None or self.E1 is None:
            return (True, "SKIP: well width a or E1 not available for perturbation")
        if not isinstance(self.a, (int, float)):
            return (False, "'a' is not numeric")
        epsilon = 0.01  # 1% increase
        baseline_Eavg4 = 2.5 * self.E1
        # E1 ∝ 1/a^2, so new E1' = E1 * (a / a')^2
        perturbed_a = self.a * (1 + epsilon)
        scale_factor = (self.a / perturbed_a) ** 2
        perturbed_E1 = self.E1 * scale_factor
        perturbed_Eavg4 = 2.5 * perturbed_E1
        if perturbed_Eavg4 >= baseline_Eavg4:
            return (False, f"Perturbed E_avg_4 ({perturbed_Eavg4:.6f}) "
                           f"should be smaller than baseline ({baseline_Eavg4:.6f})")
        return (True, f"1% increase in well width decreases E_avg_4 as expected")

    # ------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose, no derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """
        Verify that E_avg_4 is positive and not unreasonably large.
        Accept any value between 0 and 1e6 eV (physically plausible).
        """
        if self.E_avg_4 is None:
            return (False, "SKIP: E_avg_4 missing")
        if not (0 < self.E_avg_4 < 1e6):
            return (False, f"E_avg_4={self.E_avg_4} out of plausible range (0, 1e6 eV)")
        if self.E1 is not None:
            # If E1 is known, check that E_avg_4 is at least E1 (cannot be lower than ground state)
            # and at most 9*E1 (if all 4 electrons were in n=3, but that's the highest plausible)
            lower = self.E1
            upper = 9 * self.E1
            if not (lower <= self.E_avg_4 <= upper):
                return (False, f"E_avg_4={self.E_avg_4} not between E1 ({lower}) and 9*E1 ({upper})")
        return (True, "E_avg_4 passes broad sanity")

    # ------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, collect logs."""
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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)