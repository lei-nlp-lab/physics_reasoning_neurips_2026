import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Physics checker for Coulomb barrier penetration problem.
    Checks interface, invariants, modeling consistency, and monotonicity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed data from solution_instance using safe None defaults.
        """
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', None) or {}
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract givens dictionary if present
        self.givens = getattr(solution_instance, 'givens', None) or {}
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Extract explanation_steps (list of dicts or similar)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = []

        # --- Main quantity from [emit] contract ---
        self.U_barrier = self.results.get('U_barrier', None)

        # --- Potential givens (if provided) ---
        self.Z = self.givens.get('Z', None)          # atomic number
        self.R_nucleus = self.givens.get('R_nucleus', None)  # nuclear radius in fm
        # Also try generic 'R'
        if self.R_nucleus is None:
            self.R_nucleus = self.givens.get('R', None)

        # Ensure numeric types are float or None
        for attr in ['U_barrier', 'Z', 'R_nucleus']:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, (int, float)):
                setattr(self, attr, float(val) if val else None)

        # Physics constant re-derived from e^2/(4πϵ0) = 1.44 MeV·fm
        self._CONST = 1.44

    # ------------------------------------------------------------------
    # Interface Checks
    # ------------------------------------------------------------------

    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that 'U_barrier' exists in results and is numeric."""
        if self.U_barrier is None:
            return (False, "Required key 'U_barrier' missing or None")
        if not isinstance(self.U_barrier, (int, float)):
            return (False, f"U_barrier is not numeric: {type(self.U_barrier)}")
        if math.isnan(self.U_barrier) or math.isinf(self.U_barrier):
            return (False, "U_barrier is NaN or Inf")
        return (True, "U_barrier present and numeric")

    # ------------------------------------------------------------------
    # Invariant / Theorem-based Sanity Checks
    # ------------------------------------------------------------------

    def test_invariant_positive_energy(self) -> Tuple[bool, str]:
        """Energy must be positive."""
        if self.U_barrier is None:
            return (False, "SKIP: U_barrier not available")
        if self.U_barrier <= 0:
            return (False, f"U_barrier = {self.U_barrier} is not positive")
        return (True, f"U_barrier = {self.U_barrier} > 0")

    def test_invariant_energy_order(self) -> Tuple[bool, str]:
        """
        For a light nucleus, Coulomb barrier should be of order 1 MeV.
        Check that U_barrier is within a factor of 10 of 1 MeV.
        """
        if self.U_barrier is None:
            return (False, "SKIP: U_barrier not available")
        low = 0.1
        high = 10.0
        if low <= self.U_barrier <= high:
            return (True, f"U_barrier = {self.U_barrier} MeV is within [{low}, {high}] MeV")
        else:
            return (False, f"U_barrier = {self.U_barrier} MeV outside expected range ({low}–{high} MeV)")

    # ------------------------------------------------------------------
    # Modeling Quality Checks (using explanation steps / formula)
    # ------------------------------------------------------------------

    def test_modeling_formula_consistency(self) -> Tuple[bool, str]:
        """
        If Z and R_nucleus are given, verify that U_barrier ≈ 1.44 * Z / R_nucleus (in MeV).
        Use math.isclose with relaxed tolerance.
        """
        if self.U_barrier is None:
            return (False, "SKIP: U_barrier not available")
        if self.Z is None or self.R_nucleus is None:
            return (False, "SKIP: Z or R_nucleus not provided in givens")
        if self.R_nucleus <= 0:
            return (False, f"Invalid R_nucleus = {self.R_nucleus} (must be >0)")
        expected = self._CONST * self.Z / self.R_nucleus
        # Allow 5% relative tolerance since R is approximate
        if math.isclose(self.U_barrier, expected, rel_tol=0.05, abs_tol=0.05):
            return (True, f"U_barrier = {self.U_barrier} ≈ {expected} (1.44 * {self.Z}/{self.R_nucleus})")
        else:
            return (False, f"U_barrier = {self.U_barrier} deviates from expected {expected}")

    def test_modeling_nuclear_radius_approx(self) -> Tuple[bool, str]:
        """
        If R_nucleus not given, check that the result implicitly used a reasonable radius.
        For Z=1, we expect R_implied = 1.44 / U_barrier to be in typical range (0.5–5 fm).
        """
        if self.U_barrier is None:
            return (False, "SKIP: U_barrier not available")
        # We can only check if Z=1 (light hydrogen)
        if self.Z is None or self.Z != 1:
            return (False, "SKIP: Z is not given or not equal to 1")
        if self.U_barrier <= 0:
            return (False, "U_barrier <=0, cannot compute implied radius")
        R_implied = self._CONST / self.U_barrier
        if 0.5 <= R_implied <= 5.0:
            return (True, f"Implied radius R ≈ {R_implied:.2f} fm is within typical range (0.5–5 fm)")
        else:
            return (False, f"Implied radius R ≈ {R_implied:.2f} fm outside typical range (0.5–5 fm)")

    # ------------------------------------------------------------------
    # Monotonicity Checks (if givens available)
    # ------------------------------------------------------------------

    def test_monotonicity_with_Z(self) -> Tuple[bool, str]:
        """
        If Z is given, perturb Z upward and verify that U_barrier increases.
        Uses the formula U = 1.44 * Z / R_nucleus.
        """
        if self.U_barrier is None or self.R_nucleus is None or self.Z is None:
            return (False, "SKIP: U_barrier, Z, or R_nucleus not available")
        if self.R_nucleus <= 0 or self.Z <= 0:
            return (False, "Non-positive Z or R_nucleus")
        # Baseline expected using Z
        expected_base = self._CONST * self.Z / self.R_nucleus
        # Perturb Z by +10%
        Z_pert = self.Z * 1.1
        expected_pert = self._CONST * Z_pert / self.R_nucleus
        # U_barrier should increase
        if expected_pert <= expected_base:
            return (False, f"Monotonicity fails: Z increase should increase barrier (base={expected_base}, perturb={expected_pert})")
        # Check consistency of reported U_barrier with baseline
        if not math.isclose(self.U_barrier, expected_base, rel_tol=0.05, abs_tol=0.05):
            return (False, f"U_barrier = {self.U_barrier} not consistent with baseline {expected_base}, cannot test monotonicity")
        return (True, f"Z monotonicity holds: Z increase → barrier increase (base={expected_base}, perturb={expected_pert})")

    def test_monotonicity_with_R(self) -> Tuple[bool, str]:
        """
        If R_nucleus given, perturb it downward (by 10%) and verify that barrier increases.
        """
        if self.U_barrier is None or self.R_nucleus is None or self.Z is None:
            return (False, "SKIP: U_barrier, Z, or R_nucleus not available")
        if self.R_nucleus <= 0 or self.Z <= 0:
            return (False, "Non-positive Z or R_nucleus")
        expected_base = self._CONST * self.Z / self.R_nucleus
        # Perturb R downward by 10%
        R_pert = self.R_nucleus * 0.9
        expected_pert = self._CONST * self.Z / R_pert
        if expected_pert <= expected_base:
            return (False, f"Monotonicity fails: R decrease should increase barrier (base={expected_base}, perturb={expected_pert})")
        if not math.isclose(self.U_barrier, expected_base, rel_tol=0.05, abs_tol=0.05):
            return (False, f"U_barrier = {self.U_barrier} not consistent with baseline {expected_base}, cannot test monotonicity")
        return (True, f"R monotonicity holds: R decrease → barrier increase (base={expected_base}, perturb={expected_pert})")

    # ------------------------------------------------------------------
    # Weak Answer Sanity (only one allowed)
    # ------------------------------------------------------------------

    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Very weak sanity: U_barrier is finite, not NaN, and positive.
        No hardcoded target value comparisons.
        """
        if self.U_barrier is None:
            return (False, "SKIP: U_barrier not available")
        if math.isnan(self.U_barrier) or math.isinf(self.U_barrier):
            return (False, "U_barrier is NaN or Inf")
        if self.U_barrier <= 0:
            return (False, "U_barrier is not positive")
        return (True, f"U_barrier = {self.U_barrier} MeV is finite and positive")

    # ------------------------------------------------------------------
    # Main Runner
    # ------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them safely,
        and return (all_pass, logs).
        """
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith('test_'):
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