import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """Auditor for a solution to: 'By what arguments and using what measurable quantities
    can one determine the mass of Earth, mass of Moon, distance Earth-Sun?' """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely – they are expected to be in a dict-like container
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        # If it's a function, call it; otherwise assume dict
        if callable(results):
            results = results()
        elif not isinstance(results, dict):
            results = {}

        self.M_e: Optional[float] = results.get('M_e', None)
        self.M_m: Optional[float] = results.get('M_m', None)
        self.a_e: Optional[float] = results.get('a_e', None)

        # Also try to extract the measurable quantities that the solution might have stored
        self.g: Optional[float] = results.get('g', None)
        self.R_e: Optional[float] = results.get('R_e', None)
        self.G: Optional[float] = results.get('G', None)
        self.a_em: Optional[float] = results.get('a_em', None)          # Earth-Moon semi-major axis
        self.T_m: Optional[float] = results.get('T_m', None)            # Moon orbital period (s)
        self.c: Optional[float] = results.get('c', None)                # speed of light
        self.dt: Optional[float] = results.get('dt', None)              # round-trip time
        self.T_e: Optional[float] = results.get('T_e', None)            # Earth orbital period (s)
        self.T_v: Optional[float] = results.get('T_v', None)            # Venus orbital period (s)

        # Explanation_steps – we expect a list of strings or a single string
        self.explanation: Optional[str] = None
        raw_expl = getattr(solution_instance, 'explanation_steps', None)
        if raw_expl is not None:
            if isinstance(raw_expl, list):
                self.explanation = " ".join(raw_expl)
            elif isinstance(raw_expl, str):
                self.explanation = raw_expl
            else:
                self.explanation = str(raw_expl)

    # -------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy comparison for invariant/modeling tests (allowed by rule #2)."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # -------------------------------------------------------------------
    # interface tests – required keys must exist
    # -------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Ensure M_e, M_m, a_e are present (not None)."""
        missing = []
        if self.M_e is None:
            missing.append("M_e")
        if self.M_m is None:
            missing.append("M_m")
        if self.a_e is None:
            missing.append("a_e")
        if missing:
            return (False, f"Missing required result key(s): {', '.join(missing)}")
        return (True, "All required keys (M_e, M_m, a_e) present.")

    def test_interface_measurable_quantities(self) -> Tuple[bool, str]:
        """Check that g, R_e, G, a_em, T_m, c, dt, T_e, T_v are present in results."""
        missing = []
        for name, val in [('g', self.g), ('R_e', self.R_e), ('G', self.G),
                          ('a_em', self.a_em), ('T_m', self.T_m),
                          ('c', self.c), ('dt', self.dt),
                          ('T_e', self.T_e), ('T_v', self.T_v)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing measurable quantity key(s): {', '.join(missing)}")
        return (True, "All measurable quantities present.")

    # -------------------------------------------------------------------
    # invariant tests – theorem-based physical sanity checks using fuzzy matching
    # -------------------------------------------------------------------
    def test_invariant_earth_mass(self) -> Tuple[bool, str]:
        """Verify g = G * M_e / R_e^2  (Newton's law of gravitation)."""
        if any(v is None for v in (self.g, self.R_e, self.G, self.M_e)):
            return (False, "SKIP: one or more required inputs (g, R_e, G, M_e) missing.")
        expected_M_e = self.g * self.R_e**2 / self.G
        if self._isclose(self.M_e, expected_M_e, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M_e ≈ {expected_M_e:.3e} matches g*R_e^2/G.")
        return (False, f"M_e = {self.M_e:.3e} but expected ≈ {expected_M_e:.3e} (g*R_e^2/G).")

    def test_invariant_moon_mass(self) -> Tuple[bool, str]:
        """Check Kepler's third law for Earth-Moon: T_m^2 = 4π^2 a_em^3/(G (M_e+M_m))."""
        if any(v is None for v in (self.M_e, self.M_m, self.a_em, self.T_m, self.G)):
            return (False, "SKIP: required inputs (M_e, M_m, a_em, T_m, G) missing.")
        expected_total = 4 * math.pi**2 * self.a_em**3 / (self.G * self.T_m**2)
        actual_total = self.M_e + self.M_m
        if self._isclose(actual_total, expected_total, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M_e+M_m ≈ {expected_total:.3e} matches 4π²a³/(G T²).")
        return (False, f"M_e+M_m = {actual_total:.3e} but expected ≈ {expected_total:.3e} from Kepler.")

    def test_invariant_earth_sun_distance(self) -> Tuple[bool, str]:
        """Check relation: a_e = (c*dt/2) / (1 - (T_v/T_e)^(2/3))."""
        if any(v is None for v in (self.a_e, self.c, self.dt, self.T_e, self.T_v)):
            return (False, "SKIP: required inputs (a_e, c, dt, T_e, T_v) missing.")
        try:
            denominator = 1 - (self.T_v / self.T_e)**(2/3)
        except ZeroDivisionError:
            return (False, "Denominator zero (T_v = T_e?), cannot compute.")
        expected_a_e = (self.c * self.dt / 2) / denominator
        if self._isclose(self.a_e, expected_a_e, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"a_e ≈ {expected_a_e:.3e} matches radar+Kepler formula.")
        return (False, f"a_e = {self.a_e:.3e} but expected ≈ {expected_a_e:.3e} from combined law.")

    # -------------------------------------------------------------------
    # modeling / trace checks – examine explanation_steps for key principles
    # -------------------------------------------------------------------
    def test_modeling_explanation_principles(self) -> Tuple[bool, str]:
        """Ensure explanation_steps mentions Newton's law, Kepler's third law, radar ranging, inferior conjunction."""
        if self.explanation is None:
            return (False, "SKIP: no explanation_steps provided.")
        txt = self.explanation.lower()
        required = ['newton', 'gravitation', 'kepler', 'radar', 'inferior conjunction']
        missing = [word for word in required if word not in txt]
        if missing:
            return (False, f"Missing key principle(s) in explanation: {', '.join(missing)}")
        return (True, "Explanation covers Newton, Kepler, radar, inferior conjunction.")

    def test_modeling_measurable_quantities(self) -> Tuple[bool, str]:
        """Check that explanation mentions the key measurable quantities (g, R_e, G, a_em, T_m, c, dt, T_e, T_v)."""
        if self.explanation is None:
            return (False, "SKIP: no explanation_steps provided.")
        txt = self.explanation.lower()
        # we require at least g, R_e, G, and some period and distance references
        required = ['g', 'r_e', 'g(=6.67', 'a_em', 't_m', 'c', 't_e', 't_v']
        found = [item for item in required if item in txt]
        if len(found) < 4:  # be lenient, we only expect at least 4 distinct measurables
            return (False, f"Too few measurable quantities mentioned in explanation.")
        return (True, "Explanation mentions several measurable quantities.")

    # -------------------------------------------------------------------
    # monotonicity tests – perturb one input and recompute direction
    # -------------------------------------------------------------------
    def test_monotonicity_earth_mass_g(self) -> Tuple[bool, str]:
        """Perturb g upward: M_e should increase (for fixed R_e, G)."""
        if any(v is None for v in (self.g, self.R_e, self.G, self.M_e)):
            return (False, "SKIP: required inputs (g, R_e, G, M_e) missing.")
        baseline_M_e = self.M_e
        perturbed_g = self.g * 1.1   # +10%
        new_M_e = perturbed_g * self.R_e**2 / self.G
        if new_M_e > baseline_M_e:
            return (True, f"Increasing g by 10% increases M_e (from {baseline_M_e:.3e} to {new_M_e:.3e}).")
        return (False, f"Increasing g by 10% should increase M_e, but got new M_e = {new_M_e:.3e} ≤ baseline {baseline_M_e:.3e}.")

    def test_monotonicity_moon_mass_period(self) -> Tuple[bool, str]:
        """Perturb T_m downward: M_m should increase (since M_e fixed, total mass ~ 1/T^2)."""
        if any(v is None for v in (self.M_e, self.M_m, self.a_em, self.T_m, self.G)):
            return (False, "SKIP: required inputs (M_e, M_m, a_em, T_m, G) missing.")
        # Baseline total mass
        baseline_total = self.M_e + self.M_m
        # Decrease period by 10%
        perturbed_T_m = self.T_m * 0.9
        new_total = 4 * math.pi**2 * self.a_em**3 / (self.G * perturbed_T_m**2)
        new_M_m = new_total - self.M_e
        if new_M_m > self.M_m:
            return (True, f"Decreasing T_m by 10% increases M_m (from {self.M_m:.3e} to {new_M_m:.3e}).")
        return (False, f"Decreasing T_m should increase M_m, but got new M_m = {new_M_m:.3e} ≤ baseline {self.M_m:.3e}.")

    # -------------------------------------------------------------------
    # weak answer sanity – only broad sign / finite checks (rule #8)
    # -------------------------------------------------------------------
    def test_answer_sanity_nonnegative(self) -> Tuple[bool, str]:
        """All emitted results must be nonnegative (masses and distances are positive)."""
        errors = []
        if self.M_e is not None and self.M_e < 0:
            errors.append(f"M_e = {self.M_e} is negative")
        if self.M_m is not None and self.M_m < 0:
            errors.append(f"M_m = {self.M_m} is negative")
        if self.a_e is not None and self.a_e < 0:
            errors.append(f"a_e = {self.a_e} is negative")
        if errors:
            return (False, "; ".join(errors))
        return (True, "All emitted results are nonnegative (physically reasonable).")

    # -------------------------------------------------------------------
    # Test runner – discovers all test_ methods, runs them with try/except
    # -------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        # Get all bound methods starting with "test_"
        test_methods = []
        for name in dir(self):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if callable(method):
                test_methods.append((name, method))
        # Sort for deterministic order
        test_methods.sort(key=lambda x: x[0])
        for name, method in test_methods:
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)