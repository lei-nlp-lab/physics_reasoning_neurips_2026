import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for quantum mechanics / ionization energy problem.
    Checks interface completeness, physical invariants, modeling quality,
    and monotonicity constraints.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely (no numeric defaults)
        self.results: Optional[Dict] = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Required result keys – default to None for missing values
        self.trend_explanation: Optional[str] = self.results.get('trend_explanation', None)
        self.E_II_He: Optional[float] = self.results.get('E_II_He', None)
        self.l_dependence: Optional[str] = self.results.get('l_dependence', None)

        # Known physical constants (not extracted from solution)
        self.Ry = 13.6  # eV (Rydberg constant)

    # ===================== Interface tests =====================
    def test_interface_trend_explanation_type(self) -> Tuple[bool, str]:
        """Check that trend_explanation exists and is a string."""
        if self.trend_explanation is None:
            return (False, "trend_explanation missing from results")
        if not isinstance(self.trend_explanation, str):
            return (False, f"trend_explanation is not string, got {type(self.trend_explanation)}")
        return (True, "trend_explanation is a valid string")

    def test_interface_E_II_He_type(self) -> Tuple[bool, str]:
        """Check that E_II_He exists and is numeric."""
        if self.E_II_He is None:
            return (False, "E_II_He missing from results")
        if not isinstance(self.E_II_He, (int, float)):
            return (False, f"E_II_He is not numeric, got {type(self.E_II_He)}")
        if math.isnan(self.E_II_He) or math.isinf(self.E_II_He):
            return (False, f"E_II_He is NaN or infinite")
        return (True, "E_II_He is a valid float")

    def test_interface_l_dependence_type(self) -> Tuple[bool, str]:
        """Check that l_dependence exists and is a string."""
        if self.l_dependence is None:
            return (False, "l_dependence missing from results")
        if not isinstance(self.l_dependence, str):
            return (False, f"l_dependence is not string, got {type(self.l_dependence)}")
        return (True, "l_dependence is a valid string")

    # ===================== Theorem-based invariant tests =====================
    def test_invariant_second_ionization_energy_he_plus(self) -> Tuple[bool, str]:
        """
        Verify that the second ionization energy of He matches the hydrogen-like formula:
        E_II = Z^2 * Ry = 4 * 13.6 = 54.4 eV.
        This derives directly from Coulomb physics for a hydrogen-like ion.
        """
        if self.E_II_He is None:
            return (False, "SKIP: E_II_He not available")
        expected = self.Ry * 4.0  # Z^2 with Z=2
        if math.isclose(self.E_II_He, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_II_He = {self.E_II_He} eV matches expected {expected} eV (within 1% tolerance)")
        else:
            return (False, f"E_II_He = {self.E_II_He} eV but expected {expected} eV (deviation exceeds 1%)")

    def test_invariant_second_ionization_energy_positive(self) -> Tuple[bool, str]:
        """Check that ionization energy is positive (removing an electron requires energy)."""
        if self.E_II_He is None:
            return (False, "SKIP: E_II_He not available")
        if self.E_II_He > 0:
            return (True, f"E_II_He = {self.E_II_He} eV > 0")
        else:
            return (False, f"E_II_He = {self.E_II_He} eV should be positive")

    # ===================== Modeling / trace quality tests =====================
    def test_modeling_trend_explanation_contains_shielding(self) -> Tuple[bool, str]:
        """The explanation for the ionization energy trend should mention shell structure or shielding."""
        if self.trend_explanation is None:
            return (False, "SKIP: trend_explanation not available")
        explanation = self.trend_explanation.lower()
        keywords = ["shielding", "screening", "shell", "effective nuclear charge"]
        found = [kw for kw in keywords if kw in explanation]
        if found:
            return (True, f"trend_explanation contains shielding/shell keywords: {found}")
        else:
            return (False, "trend_explanation lacks any shielding/shell/screening keywords")

    def test_modeling_l_dependence_contains_penetration(self) -> Tuple[bool, str]:
        """The explanation for l-dependence should mention orbital penetration or quantum defect."""
        if self.l_dependence is None:
            return (False, "SKIP: l_dependence not available")
        explanation = self.l_dependence.lower()
        keywords = ["penetration", "orbital penetration", "quantum defect", "effective nuclear charge"]
        found = [kw for kw in keywords if kw in explanation]
        if found:
            return (True, f"l_dependence contains penetration/defect keywords: {found}")
        else:
            return (False, "l_dependence lacks any mention of penetration or quantum defect")

    # ===================== Monotonicity tests =====================
    def test_monotonicity_ionization_energy_increases_with_Z(self) -> Tuple[bool, str]:
        """
        Explicit perturbative check: for a hydrogen-like atom, increasing the nuclear charge Z
        increases the ionization energy (since E ∝ Z^2). We compute the baseline at Z=2 (He+)
        and perturb Z by +5% to Z=2.1, then check that the perturbed value is larger.
        """
        # Use the known physical formula (re-derived inside checker)
        baseline_Z = 2.0
        perturbed_Z = 2.1  # 5% increase

        baseline_energy = self.Ry * (baseline_Z ** 2)
        perturbed_energy = self.Ry * (perturbed_Z ** 2)

        if perturbed_energy > baseline_energy:
            return (True,
                    f"Ionization energy increases with Z: baseline (Z=2) = {baseline_energy:.2f} eV, "
                    f"perturbed (Z=2.1) = {perturbed_energy:.2f} eV")
        else:
            return (False,
                    f"Monotonicity violated: perturbed energy {perturbed_energy:.2f} is not greater "
                    f"than baseline {baseline_energy:.2f}")

    # ===================== Weak final answer sanity (at most one) =====================
    def test_answer_sanity_nonempty_explanations(self) -> Tuple[bool, str]:
        """Both explanation strings should be non-empty (trivial sanity)."""
        if self.trend_explanation is None and self.l_dependence is None:
            return (False, "SKIP: both explanations missing")
        # Check non-empty if present
        if self.trend_explanation is not None and len(self.trend_explanation.strip()) == 0:
            return (False, "trend_explanation is empty")
        if self.l_dependence is not None and len(self.l_dependence.strip()) == 0:
            return (False, "l_dependence is empty")
        return (True, "explanations (when present) are non-empty")

    # ===================== Main runner =====================
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, execute them safely, collect logs."""
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