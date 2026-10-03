import math
import traceback
from typing import Any, Dict, Optional, Tuple, List

class PhysicsChecker:
    """
    Audits solutions for the excitation energy of the T=6 isospin state in ^90Zr.
    Uses binding energies and Coulomb displacement energy formula.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract required result quantities from the results dictionary (pseudocode emit)
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.R: Optional[float] = results.get('R', None)
        self.dE_C: Optional[float] = results.get('dE_C', None)
        self.E_ex: Optional[float] = results.get('E_ex', None)

        # Extract givens (provided in the problem statement)
        givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        self.B_Zr: Optional[float] = givens.get('B_Zr', None)
        self.B_Y: Optional[float] = givens.get('B_Y', None)
        self.e2: Optional[float] = givens.get('e2', None)      # e^2 = 1.44 MeV·fm
        self.r0: Optional[float] = givens.get('r0', None)      # nuclear radius constant
        self.A: Optional[float] = givens.get('A', None)        # mass number
        self.Z: Optional[float] = givens.get('Z', None)        # atomic number of Zr

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test methods, execute them, and return aggregated results."""
        logs: List[str] = []
        all_pass: bool = True
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
        return all_pass, logs

    # ---------- Interface checks ----------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        """Ensure the three required result keys exist and are numeric."""
        missing = []
        for key, val in [("R", self.R), ("dE_C", self.dE_C), ("E_ex", self.E_ex)]:
            if val is None:
                missing.append(key)
            elif not isinstance(val, (int, float)):
                missing.append(f"{key} (non-numeric: {type(val).__name__})")
        if missing:
            return False, f"Missing or invalid result keys: {', '.join(missing)}"
        return True, "All required result keys present and numeric."

    # ---------- Invariant checks (use math.isclose) ----------
    def test_invariant_radius_formula(self) -> Tuple[bool, str]:
        """Check that the nuclear radius R = r0 * A^(1/3) is consistent."""
        if self.R is None:
            return False, "SKIP: R not available"
        if self.r0 is None or self.A is None:
            return False, "SKIP: r0 or A not available"
        expected_R = self.r0 * (self.A ** (1.0 / 3.0))
        if math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"R = {self.R} matches expected {expected_R:.4f} fm"
        else:
            return False, f"R = {self.R} differs from expected {expected_R:.4f}"

    def test_invariant_coulomb_displacement(self) -> Tuple[bool, str]:
        """Check dE_C = (3/5)*(e^2 / R)*(2Z - 1)."""
        if self.dE_C is None:
            return False, "SKIP: dE_C not available"
        if any(x is None for x in (self.e2, self.R, self.Z)):
            return False, "SKIP: e2, R, or Z not available for Coulomb displacement"
        expected = (3.0 / 5.0) * (self.e2 / self.R) * (2.0 * self.Z - 1.0)
        if math.isclose(self.dE_C, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"dE_C = {self.dE_C} matches expected {expected:.4f} MeV"
        else:
            return False, f"dE_C = {self.dE_C} differs from expected {expected:.4f}"

    def test_invariant_excitation_energy(self) -> Tuple[bool, str]:
        """Check E_ex = B_Zr - B_Y + dE_C."""
        if self.E_ex is None:
            return False, "SKIP: E_ex not available"
        if any(x is None for x in (self.B_Zr, self.B_Y, self.dE_C)):
            return False, "SKIP: B_Zr, B_Y, or dE_C not available"
        expected = self.B_Zr - self.B_Y + self.dE_C
        if math.isclose(self.E_ex, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_ex = {self.E_ex} matches expected {expected:.4f} MeV"
        else:
            return False, f"E_ex = {self.E_ex} differs from expected {expected:.4f}"

    # ---------- Modeling / trace checks ----------
    def test_modeling_all_positive(self) -> Tuple[bool, str]:
        """All three result quantities should be positive."""
        checks = []
        if self.R is not None and self.R <= 0:
            checks.append("R <= 0")
        if self.dE_C is not None and self.dE_C <= 0:
            checks.append("dE_C <= 0")
        if self.E_ex is not None and self.E_ex <= 0:
            checks.append("E_ex <= 0")
        if checks:
            return False, f"Non-positive values: {', '.join(checks)}"
        return True, "All available result quantities are positive."

    def test_modeling_natural_units(self) -> Tuple[bool, str]:
        """Check that R is in fm (order ~5-10 fm) and energies in MeV (~1-20 MeV)."""
        messages = []
        if self.R is not None and (self.R < 4 or self.R > 10):
            messages.append(f"R={self.R} outside typical range (4–10 fm)")
        if self.dE_C is not None and (self.dE_C < 5 or self.dE_C > 15):
            messages.append(f"dE_C={self.dE_C} outside typical range (5–15 MeV)")
        if self.E_ex is not None and (self.E_ex < 5 or self.E_ex > 20):
            messages.append(f"E_ex={self.E_ex} outside typical range (5–20 MeV)")
        if messages:
            return False, "; ".join(messages)
        return True, "Quantities within physically reasonable ranges."

    # ---------- Monotonicity checks ----------
    def test_monotonicity_dE_C_with_Z(self) -> Tuple[bool, str]:
        """
        Increase Z by 1 and verify that dE_C increases (qualitative monotonicity).
        """
        if any(x is None for x in (self.Z, self.e2, self.R)):
            return False, "SKIP: Z, e2, or R not available"
        # Baseline dE_C from the solution
        base_dE_C = (3.0/5.0) * (self.e2 / self.R) * (2.0 * self.Z - 1.0)
        # Perturbed Z (Z+1, still using the same radius)
        perturbed_Z = self.Z + 1
        perturbed_dE_C = (3.0/5.0) * (self.e2 / self.R) * (2.0 * perturbed_Z - 1.0)
        if perturbed_dE_C > base_dE_C:
            return True, f"dE_C increases when Z increases ({base_dE_C:.4f} -> {perturbed_dE_C:.4f})"
        else:
            return False, f"dE_C does not increase with Z: base {base_dE_C:.4f}, perturbed {perturbed_dE_C:.4f}"

    def test_monotonicity_E_ex_with_B_diff(self) -> Tuple[bool, str]:
        """
        Increase the binding energy difference B_Zr - B_Y by 10% and check that E_ex increases.
        """
        if any(x is None for x in (self.B_Zr, self.B_Y, self.dE_C, self.E_ex)):
            return False, "SKIP: required quantities not available"
        base_diff = self.B_Zr - self.B_Y
        perturbed_diff = base_diff * 1.1
        perturbed_E_ex = perturbed_diff + self.dE_C
        if perturbed_E_ex > self.E_ex:
            return True, f"E_ex increases with larger B_diff ({self.E_ex:.4f} -> {perturbed_E_ex:.4f})"
        else:
            return False, f"E_ex does not increase with B_diff: base {self.E_ex:.4f}, perturbed {perturbed_E_ex:.4f}"

    # ---------- Answer sanity (weak, single method) ----------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: excitation energy should be between 10 and 20 MeV and be finite/not NaN.
        """
        if self.E_ex is None:
            return False, "SKIP: E_ex not available"
        if math.isnan(self.E_ex) or math.isinf(self.E_ex):
            return False, f"E_ex is NaN or infinite: {self.E_ex}"
        if 10.0 <= self.E_ex <= 20.0:
            return True, f"E_ex = {self.E_ex} within 10–20 MeV range and finite."
        else:
            return False, f"E_ex = {self.E_ex} outside reasonable range 10–20 MeV."