import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Conceptual checker for a problem about the origin of energy bands in crystals.
    Designed to be robust, providing many small tests with safe attribute access.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results dictionary safely
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract givens (though not used much here)
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Explanation steps (list of strings)
        self.explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None)
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = None

        # Pseudocode fields (optional)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)
        self.solving_pseudocode: Optional[dict] = getattr(solution_instance, 'solving_pseudocode', None)

        # Specific result key we care about
        self.band_origin: Optional[str] = self.results.get('band_origin', None)

    # =========================================================================
    # Interface tests: check that required keys exist and have valid types
    # =========================================================================

    def test_interface_band_origin_exists(self) -> Tuple[bool, str]:
        if self.band_origin is None:
            return (False, "results['band_origin'] is missing")
        return (True, "band_origin key present")

    def test_interface_band_origin_is_string(self) -> Tuple[bool, str]:
        if self.band_origin is None:
            return (False, "cannot check type: band_origin is None")
        if not isinstance(self.band_origin, str):
            return (False, f"band_origin is not a str, got {type(self.band_origin)}")
        return (True, "band_origin is a string")

    def test_interface_band_origin_nonempty(self) -> Tuple[bool, str]:
        if self.band_origin is None:
            return (False, "cannot check non-empty: band_origin is None")
        if len(self.band_origin.strip()) == 0:
            return (False, "band_origin is an empty string")
        return (True, "band_origin is non-empty")

    def test_interface_explanation_steps_list(self) -> Tuple[bool, str]:
        if self.explanation_steps is None:
            return (False, "explanation_steps is missing or not a list")
        if len(self.explanation_steps) == 0:
            return (False, "explanation_steps is empty")
        return (True, "explanation_steps present and non-empty")

    # =========================================================================
    # Invariant tests: theorem-based physical sanity checks
    # =========================================================================

    def test_invariant_bloch_theorem_mentioned(self) -> Tuple[bool, str]:
        """Check that Bloch's theorem is mentioned in the output or steps."""
        text = ""
        if self.band_origin:
            text += self.band_origin + " "
        if self.explanation_steps:
            text += " ".join(self.explanation_steps)
        if "bloch" in text.lower():
            return (True, "Bloch theorem concept present")
        return (False, "No mention of Bloch theorem")

    def test_invariant_periodic_potential_mentioned(self) -> Tuple[bool, str]:
        """Periodic potential V(r+R)=V(r) is central."""
        text = ""
        if self.band_origin:
            text += self.band_origin + " "
        if self.explanation_steps:
            text += " ".join(self.explanation_steps)
        keywords = ["periodic", "periodicity", "potential", "lattice"]
        found = [kw for kw in keywords if kw in text.lower()]
        if len(found) >= 2:
            return (True, f"Periodic potential concept present: {found}")
        return (False, f"Only found {len(found)} of required periodic keywords: {found}")

    def test_invariant_single_electron_approximation(self) -> Tuple[bool, str]:
        text = ""
        if self.explanation_steps:
            text = " ".join(self.explanation_steps)
        if "single-electron" in text.lower() or "single electron" in text.lower():
            return (True, "Single-electron approximation mentioned")
        return (False, "No mention of single-electron approximation")

    def test_invariant_energy_gaps_mentioned(self) -> Tuple[bool, str]:
        text = ""
        if self.band_origin:
            text += self.band_origin + " "
        if self.explanation_steps:
            text += " ".join(self.explanation_steps)
        for word in ["gap", "gap separation", "forbidden"]:
            if word in text.lower():
                return (True, "Energy gaps concept present")
        return (False, "No mention of energy gaps or forbidden regions")

    def test_invariant_brillouin_zone_boundary(self) -> Tuple[bool, str]:
        text = ""
        if self.explanation_steps:
            text = " ".join(self.explanation_steps)
        if "brillouin" in text.lower() or "zone boundary" in text.lower() or "bragg" in text.lower():
            return (True, "Brillouin zone boundary or Bragg condition mentioned")
        return (False, "No mention of Brillouin zone or Bragg condition")

    # =========================================================================
    # Modeling/trace tests: quality checks based on explanation_steps
    # =========================================================================

    def test_modeling_nearly_free_electron_model(self) -> Tuple[bool, str]:
        text = ""
        if self.explanation_steps:
            text = " ".join(self.explanation_steps)
        if "nearly free electron" in text.lower() or "perturbation" in text.lower():
            return (True, "Nearly free electron model or perturbation theory mentioned")
        return (False, "No mention of nearly free electron model or perturbation")

    def test_modeling_bloch_wavefunction_form(self) -> Tuple[bool, str]:
        text = ""
        if self.explanation_steps:
            text = " ".join(self.explanation_steps)
        if "exp" in text and "u" in text.lower():
            # Bloch form: exp(ikx) u(x)
            return (True, "Bloch wavefunction exponent and periodic factor mentioned")
        return (False, "No explicit mention of Bloch wavefunction form (exp * u)")

    def test_modeling_boundary_conditions(self) -> Tuple[bool, str]:
        text = ""
        if self.explanation_steps:
            text = " ".join(self.explanation_steps)
        if "boundary condition" in text.lower() or "born-von karman" in text.lower():
            return (True, "Boundary conditions mentioned")
        return (False, "No mention of boundary conditions")

    def test_modeling_explanation_coverage(self) -> Tuple[bool, str]:
        """Check that at least 4 key concepts appear in explanation_steps."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = " ".join(self.explanation_steps).lower()
        concepts = ["bloch", "periodic", "gap", "brillouin", "nearly free",
                     "perturbation", "wavefunction", "boundary"]
        found = sum(1 for c in concepts if c in text)
        required = 4  # lenient
        if found >= required:
            return (True, f"Found {found} key concepts (required {required})")
        return (False, f"Only found {found} key concepts, expected at least {required}")

    # =========================================================================
    # Monotonicity tests: explicit perturbation of a physical input
    # =========================================================================

    def test_monotonicity_band_gap_with_potential_strength(self) -> Tuple[bool, str]:
        """
        From the nearly free electron model, the band gap E_gap = 2|V_G|.
        If we increase the periodic potential Fourier component V_G, the band gap
        must increase. We test this with baseline V_G=1.0 and perturbed V_G'=1.1.
        """
        # Baseline
        V_G_baseline = 1.0
        gap_baseline = 2 * abs(V_G_baseline)   # = 2.0

        # Perturbed (increase by 10%)
        V_G_perturbed = 1.1
        gap_perturbed = 2 * abs(V_G_perturbed) # = 2.2

        # Check that gap increased (monotonicity)
        if gap_perturbed <= gap_baseline:
            return (False, f"Gap did not increase: baseline {gap_baseline}, perturbed {gap_perturbed}")

        # Also check the ratio is approximately 1.1 (from theory)
        ratio = gap_perturbed / gap_baseline
        expected_ratio = V_G_perturbed / V_G_baseline
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-2):
            return (False, f"Ratio mismatch: {ratio} vs expected {expected_ratio}")
        return (True, f"Gap increased from {gap_baseline} to {gap_perturbed} (ratio {ratio:.3f})")

    # =========================================================================
    # Answer sanity (weak, one method max)
    # =========================================================================

    def test_answer_sanity_band_origin_content(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that band_origin string contains keywords
        'periodicity' and 'boundary conditions' (or close synonyms).
        """
        if self.band_origin is None:
            return (False, "band_origin missing, cannot check")
        text = self.band_origin.lower()
        # Check for periodicity/periodic and boundary/lattice
        has_periodicity = ("period" in text or "periodic" in text)
        has_boundary = ("boundary" in text or "boundaries" in text or "lattice" in text)
        if has_periodicity and has_boundary:
            return (True, "band_origin contains periodicity and boundary concepts")
        elif has_periodicity:
            return (False, "band_origin mentions periodicity but not boundary conditions")
        elif has_boundary:
            return (False, "band_origin mentions boundary but not periodicity")
        else:
            return (False, "band_origin mentions neither periodicity nor boundary conditions")

    # =========================================================================
    # Runner: collect and execute all test_* methods
    # =========================================================================

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)