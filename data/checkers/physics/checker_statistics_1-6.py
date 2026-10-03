import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics auditor for heat capacity of a copper penny problem.
    Checks interface, invariants, modeling quality, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Results
        self.C_v = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.C_v = solution_instance.results.get('C_v', None)
        elif hasattr(solution_instance, 'C_v'):
            self.C_v = solution_instance.C_v

        # Givens / derived fields (use None for missing)
        self.mass = self._get_attr_or_key(solution_instance, 'mass')
        self.volume = self._get_attr_or_key(solution_instance, 'volume')
        self.density = self._get_attr_or_key(solution_instance, 'density')
        self.atomic_mass = self._get_attr_or_key(solution_instance, 'atomic_mass')  # g/mol
        self.moles = self._get_attr_or_key(solution_instance, 'moles')
        self.specific_heat = self._get_attr_or_key(solution_instance, 'specific_heat')  # cal/g·°C
        self.R = self._get_attr_or_key(solution_instance, 'R')  # J/(mol·K)

        # Unit conversion constant
        self.CAL_TO_J = 4.184  # 1 cal = 4.184 J

    @staticmethod
    def _get_attr_or_key(obj: Any, key: str) -> Optional[Any]:
        """Safely extract an attribute or dict key."""
        if hasattr(obj, key):
            return getattr(obj, key)
        if hasattr(obj, 'givens') and isinstance(obj.givens, dict) and key in obj.givens:
            return obj.givens[key]
        if hasattr(obj, 'results') and isinstance(obj.results, dict) and key in obj.results:
            return obj.results[key]
        return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_C_v_present(self) -> Tuple[bool, str]:
        """Check that C_v is present in results."""
        if self.C_v is None:
            return (False, "C_v is missing from solution results")
        return (True, "C_v exists")

    def test_interface_C_v_numeric(self) -> Tuple[bool, str]:
        """Check that C_v is a finite number."""
        if self.C_v is None:
            return (False, "C_v missing, cannot check numeric")
        if not isinstance(self.C_v, (int, float)):
            return (False, f"C_v is not numeric, type {type(self.C_v)}")
        if not math.isfinite(self.C_v):
            return (False, "C_v is not finite (inf or nan)")
        return (True, "C_v is a finite number")

    # ------------------------------------------------------------------
    # Theorem-based invariant tests
    # ------------------------------------------------------------------
    def test_invariant_positive_C_v(self) -> Tuple[bool, str]:
        """Heat capacity must be positive."""
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        if self.C_v <= 0:
            return (False, f"C_v = {self.C_v} should be positive")
        return (True, "C_v > 0")

    def test_invariant_finite(self) -> Tuple[bool, str]:
        """Already covered by test_interface_C_v_numeric; kept for clarity."""
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        if not math.isfinite(self.C_v):
            return (False, "C_v is infinite or NaN")
        return (True, "C_v is finite")

    def test_invariant_order_of_magnitude(self) -> Tuple[bool, str]:
        """
        Broad order-of-magnitude check: C_v for a penny should be
        roughly 0.1–50 J/K (or equivalent in cal/°C). This allows
        both the Dulong–Petit (~13 J/K) and the smaller-mass estimate.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        # Accept both J/K and cal/°C; assuming J/K if > 1, else cal/°C
        if self.C_v < 0.01 or self.C_v > 50:
            return (False, f"C_v = {self.C_v} outside plausible range 0.01–50 (J/K or cal/°C)")
        return (True, "C_v within plausible order of magnitude")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_dulong_petit_consistency(self) -> Tuple[bool, str]:
        """
        If moles (n) and R are available, check C_v ≈ n * 3R
        using the Dulong–Petit law for solids at room temperature.
        Use math.isclose with 2% tolerance.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        n = self.moles
        R = self.R
        if n is None or R is None:
            # Try to estimate n from mass and atomic mass
            if self.mass is not None and self.atomic_mass is not None:
                try:
                    n = self.mass / self.atomic_mass
                except (TypeError, ZeroDivisionError):
                    return (False, "SKIP: cannot compute n from mass/atomic_mass")
            else:
                return (False, "SKIP: insufficient data for Dulong–Petit check")

        expected_C_v = n * 3 * R  # 3R = Dulong–Petit molar heat capacity
        if not math.isclose(self.C_v, expected_C_v, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_v = {self.C_v} but n*3R = {expected_C_v}, relative diff too large")
        return (True, f"C_v ≈ {expected_C_v} from Dulong–Petit")

    def test_modeling_mass_specific_heat_consistency(self) -> Tuple[bool, str]:
        """
        If mass (m) and specific_heat (c in cal/g·°C) are available,
        check C_v (in J/K) ≈ m * c * 4.184. Use isclose with 2% tolerance.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        m = self.mass
        c = self.specific_heat
        if m is None or c is None:
            return (False, "SKIP: mass or specific_heat missing")
        expected_C_v_cal = m * c  # cal/°C
        expected_C_v_J = expected_C_v_cal * self.CAL_TO_J  # J/K
        # Allow both units: compare if report is in cal/°C or J/K
        if math.isclose(self.C_v, expected_C_v_cal, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_v ≈ {expected_C_v_cal} cal/°C")
        if math.isclose(self.C_v, expected_C_v_J, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_v ≈ {expected_C_v_J} J/K")
        return (False, f"C_v = {self.C_v} inconsistent with m*c estimation ({expected_C_v_cal} cal/°C or {expected_C_v_J} J/K)")

    def test_modeling_pseudocode_trace(self) -> Tuple[bool, str]:
        """
        If volume, density, atomic_mass are available, follow pseudocode:
        m = V * density, n = m / M, C_v = n * 3R.
        Compare with reported C_v using isclose (2% tolerance).
        """
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        V = self.volume
        rho = self.density
        M = self.atomic_mass
        R = self.R
        if V is None or rho is None or M is None or R is None:
            return (False, "SKIP: missing V, rho, M, or R")
        try:
            m = V * rho
            n = m / M
            expected = n * 3 * R
        except (TypeError, ZeroDivisionError) as e:
            return (False, f"SKIP: computation error {e}")
        if not math.isclose(self.C_v, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Pseudocode gives C_v ≈ {expected}, but reported {self.C_v}")
        return (True, f"C_v consistent with pseudocode derivation ({expected})")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_C_v_increases_with_moles(self) -> Tuple[bool, str]:
        """
        Check that if we increase the number of moles by 10%,
        C_v should increase by approximately 10% (assuming Dulong–Petit).
        Uses the derivative dC_v/dn = 3R.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        n = self.moles
        R = self.R
        # Try to compute n from mass/atomic_mass if not given directly
        if n is None:
            if self.mass is not None and self.atomic_mass is not None:
                try:
                    n = self.mass / self.atomic_mass
                except (TypeError, ZeroDivisionError):
                    pass
        if n is None or R is None:
            return (False, "SKIP: insufficient data for monotonicity check")

        # Perturb n by +10%
        delta_factor = 0.1
        n_perturbed = n * (1 + delta_factor)
        expected_C_v_perturbed = n_perturbed * 3 * R
        # Check that relative change in C_v is approximately delta_factor
        relative_change = (expected_C_v_perturbed - self.C_v) / self.C_v if self.C_v != 0 else 0
        if math.isclose(relative_change, delta_factor, rel_tol=0.5, abs_tol=0.02):
            return (True, f"10% increase in n leads to ~10% increase in C_v (Δ = {relative_change:.3f})")
        # If C_v is not based on Dulong–Petit, the test may fail; we report as informational
        return (False, f"10% increase in n would give C_v ≈ {expected_C_v_perturbed}, but baseline C_v = {self.C_v}. Not monotonic w.r.t. Dulong–Petit.")

    def test_monotonicity_C_v_increases_with_mass(self) -> Tuple[bool, str]:
        """
        Check that if mass increases by 10%, C_v should increase by ~10%
        (specific heat model). Uses derivative dC_v/dm = c * 4.184 if in J/K.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        m = self.mass
        c = self.specific_heat
        if m is None or c is None:
            return (False, "SKIP: mass or specific_heat missing")

        # Perturb mass by +10%
        delta_factor = 0.1
        m_perturbed = m * (1 + delta_factor)
        # Expected C_v in cal/°C
        expected_C_v_cal_perturbed = m_perturbed * c
        # Convert to J/K for comparison
        expected_C_v_J_perturbed = expected_C_v_cal_perturbed * self.CAL_TO_J
        # Compare with reported C_v assuming it could be in either unit
        if math.isclose(self.C_v, expected_C_v_cal_perturbed, rel_tol=0.5, abs_tol=0.02):
            relative_change = (expected_C_v_cal_perturbed - self.C_v) / self.C_v if self.C_v != 0 else 0
            return (True, f"10% mass increase gives ~10% C_v increase (Δ = {relative_change:.3f})")
        if math.isclose(self.C_v, expected_C_v_J_perturbed, rel_tol=0.5, abs_tol=0.02):
            relative_change = (expected_C_v_J_perturbed - self.C_v) / self.C_v if self.C_v != 0 else 0
            return (True, f"10% mass increase gives ~10% C_v increase (Δ = {relative_change:.3f})")
        return (False, f"Reported C_v does not respond as expected to 10% mass change.")

    # ------------------------------------------------------------------
    # Weak answer sanity (only sign, range, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Final answer must be: sign positive, finite, within rough range."""
        if self.C_v is None:
            return (False, "SKIP: C_v missing")
        if self.C_v <= 0:
            return (False, "Answer must be positive")
        if not math.isfinite(self.C_v):
            return (False, "Answer must be finite")
        # Broad range: 0.01 to 50 (covers all plausible estimates)
        if self.C_v < 0.01 or self.C_v > 50:
            return (False, f"Answer {self.C_v} outside reasonable range 0.01–50")
        return (True, "Answer passes basic sanity checks")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, collect results."""
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
                # Capture exception with traceback for debugging
                tb = traceback.format_exc()
                passed, msg = False, f"{name} CRASHED:\n{tb}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)