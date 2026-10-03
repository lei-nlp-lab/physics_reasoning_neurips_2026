import math
import traceback
from typing import Any, Dict, List, Tuple, Optional, Callable

class PhysicsChecker:
    """
    Audits a solution for the adiabatic compression of air vs helium.
    The checker focuses on physical invariants, modeling quality, monotonicity,
    and a weak final-answer sanity check.
    """

    # Physical constants
    GAMMA_AIR = 1.4
    GAMMA_HE = 5.0 / 3.0  # ≈1.6667
    EXPONENT_AIR = (GAMMA_AIR - 1.0) / GAMMA_AIR  # ≈0.2857
    EXPONENT_HE = (GAMMA_HE - 1.0) / GAMMA_HE    # 0.4

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract required data from the solution instance.
        Use safe defaults (None) for missing values.
        """
        # Extract from .results (expected emit keys)
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.T2_air = results.get('T2_air', None)
        self.T2_He = results.get('T2_He', None)
        self.exponent_air = results.get('exponent_air', None)
        self.exponent_He = results.get('exponent_He', None)

        # Extract optional givens (T1, pressure ratio) for deeper checks
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}

        self.T1 = givens.get('T1', None)
        # Try to get pressure ratio: either directly or from P1 and P2
        self.P2P1 = givens.get('P2/P1', None) or givens.get('R', None)
        if self.P2P1 is None:
            P1 = givens.get('P1', None)
            P2 = givens.get('P2', None)
            if P1 is not None and P2 is not None and P1 > 0:
                self.P2P1 = P2 / P1

        # Store the full solution instance for potential extra info
        self.solution = solution_instance

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist and are numeric."""
        missing = []
        for key in ['T2_air', 'T2_He', 'exponent_air', 'exponent_He']:
            val = getattr(self, key, None)
            if val is None:
                missing.append(key)
            elif not isinstance(val, (int, float)):
                missing.append(f"{key} (non-numeric)")
        if missing:
            return False, f"Missing or non-numeric keys: {', '.join(missing)}"
        return True, "All required result keys present and numeric."

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose with lenient tolerances)
    # ------------------------------------------------------------------
    def test_invariant_exponent_air(self) -> Tuple[bool, str]:
        """Check that exponent_air ≈ 0.2857 (from gamma_air=1.4)."""
        if self.exponent_air is None:
            return False, "SKIP: exponent_air not provided"
        expected = self.EXPONENT_AIR
        if math.isclose(self.exponent_air, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"exponent_air ≈ {expected:.4f}"
        return False, f"exponent_air = {self.exponent_air:.4f}, expected ≈ {expected:.4f}"

    def test_invariant_exponent_he(self) -> Tuple[bool, str]:
        """Check that exponent_He ≈ 0.4 (from gamma_He=5/3)."""
        if self.exponent_He is None:
            return False, "SKIP: exponent_He not provided"
        expected = self.EXPONENT_HE
        if math.isclose(self.exponent_He, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"exponent_He ≈ {expected:.4f}"
        return False, f"exponent_He = {self.exponent_He:.4f}, expected ≈ {expected:.4f}"

    def test_invariant_T2_he_greater_than_T2_air(self) -> Tuple[bool, str]:
        """
        Core physics: for same pressure ratio >1 and exponent_He > exponent_air,
        T2_He must be > T2_air.
        """
        if self.T2_air is None or self.T2_He is None:
            return False, "SKIP: T2_air or T2_He not provided"
        if not (isinstance(self.T2_air, (int, float)) and isinstance(self.T2_He, (int, float))):
            return False, "T2 values must be numeric"
        if self.T2_He > self.T2_air:
            return True, "T2_He > T2_air (physically correct)"
        return False, f"T2_He ({self.T2_He}) is not > T2_air ({self.T2_air})"

    # ------------------------------------------------------------------
    # Modeling / trace tests (use isclose when appropriate)
    # ------------------------------------------------------------------
    def test_modeling_exponent_consistency(self) -> Tuple[bool, str]:
        """
        Check that exponent = (gamma-1)/gamma for the corresponding gamma.
        This uses the known gammas from physics.
        """
        # We already have separate invariant tests, this one checks both together.
        if self.exponent_air is None or self.exponent_He is None:
            return False, "SKIP: exponents not provided"
        ok_air = math.isclose(self.exponent_air, self.EXPONENT_AIR, rel_tol=1e-2, abs_tol=1e-2)
        ok_he = math.isclose(self.exponent_He, self.EXPONENT_HE, rel_tol=1e-2, abs_tol=1e-2)
        if ok_air and ok_he:
            return True, "Both exponents consistent with known gamma values"
        issues = []
        if not ok_air:
            issues.append(f"air exponent {self.exponent_air} != {self.EXPONENT_AIR:.4f}")
        if not ok_he:
            issues.append(f"He exponent {self.exponent_He} != {self.EXPONENT_HE:.4f}")
        return False, "; ".join(issues)

    def test_modeling_T2_from_adiabatic_formula(self) -> Tuple[bool, str]:
        """
        If T1 and pressure ratio are available, check that T2 values
        are consistent with T2 = T1 * (P2/P1)^exponent.
        """
        if self.T1 is None or self.P2P1 is None:
            return False, "SKIP: T1 or pressure ratio not available"
        if self.T2_air is None or self.exponent_air is None:
            return False, "SKIP: T2_air or exponent_air missing"
        if self.T2_He is None or self.exponent_He is None:
            return False, "SKIP: T2_He or exponent_He missing"
        # Compute expected values
        expected_air = self.T1 * (self.P2P1 ** self.exponent_air)
        expected_He = self.T1 * (self.P2P1 ** self.exponent_He)
        ok_air = math.isclose(self.T2_air, expected_air, rel_tol=1e-2, abs_tol=1e-2)
        ok_He = math.isclose(self.T2_He, expected_He, rel_tol=1e-2, abs_tol=1e-2)
        if ok_air and ok_He:
            return True, "Both T2 values consistent with adiabatic formula"
        issues = []
        if not ok_air:
            issues.append(f"T2_air = {self.T2_air}, expected ≈ {expected_air:.2f}")
        if not ok_He:
            issues.append(f"T2_He = {self.T2_He}, expected ≈ {expected_He:.2f}")
        return False, "; ".join(issues)

    def test_modeling_exponent_ordering(self) -> Tuple[bool, str]:
        """
        Check that exponent_He > exponent_air (0.4 > 0.286).
        This explains the larger temperature rise for He.
        """
        if self.exponent_air is None or self.exponent_He is None:
            return False, "SKIP: exponents not provided"
        if self.exponent_He > self.exponent_air:
            return True, "exponent_He > exponent_air (physically correct)"
        return False, f"exponent_He ({self.exponent_He}) is not > exponent_air ({self.exponent_air})"

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_T2_increases_with_pressure_ratio(self) -> Tuple[bool, str]:
        """
        Perturb the pressure ratio upward and check that both T2 values increase.
        Requires T1 and P2P1 from givens.
        """
        if self.T1 is None or self.P2P1 is None:
            return False, "SKIP: T1 or pressure ratio not available"
        if self.T2_air is None or self.exponent_air is None:
            return False, "SKIP: T2_air or exponent_air missing"
        if self.T2_He is None or self.exponent_He is None:
            return False, "SKIP: T2_He or exponent_He missing"

        # Baseline values are already stored as T2_air, T2_He
        # Now consider a perturbed ratio +10%
        perturbed_ratio = self.P2P1 * 1.10
        new_T2_air = self.T1 * (perturbed_ratio ** self.exponent_air)
        new_T2_He = self.T1 * (perturbed_ratio ** self.exponent_He)

        # Check monotonic increase (strict)
        ok_air = new_T2_air > self.T2_air
        ok_he = new_T2_He > self.T2_He
        if ok_air and ok_he:
            return True, "Both T2 increase when pressure ratio increases (monotonic)"
        issues = []
        if not ok_air:
            issues.append(f"T2_air baseline {self.T2_air:.2f}, perturbed {new_T2_air:.2f} (not larger)")
        if not ok_he:
            issues.append(f"T2_He baseline {self.T2_He:.2f}, perturbed {new_T2_He:.2f} (not larger)")
        return False, "; ".join(issues)

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one such test, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_temperature_positive(self) -> Tuple[bool, str]:
        """Weak sanity: both T2 values should be positive (and finite)."""
        checks = []
        for name, val in [("T2_air", self.T2_air), ("T2_He", self.T2_He)]:
            if val is None:
                checks.append(f"{name} missing")
            elif not isinstance(val, (int, float)):
                checks.append(f"{name} not numeric")
            elif val <= 0 or math.isinf(val) or math.isnan(val):
                checks.append(f"{name} not positive/finite")
        if checks:
            return False, "; ".join(checks)
        return True, "Both T2 values are positive and finite."

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all methods starting with 'test_', execute them,
        and collect results. Each test must return (passed_bool, msg_str).
        """
        logs: List[str] = []
        all_pass = True
        # Get all callable attributes that start with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                # Include traceback for debugging
                tb = traceback.format_exc()
                msg = f"CRASHED: {e}\n{tb}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs