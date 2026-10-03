import math
import traceback
from typing import Dict, Tuple, List, Any, Optional

class PhysicsChecker:
    """Auditor for relativistic inelastic collision problem."""
    
    def __init__(self, solution_instance: Any):
        # Safe extraction with None defaults – never numeric fallbacks.
        self.m0 = getattr(solution_instance, 'm0', None)
        self.c = getattr(solution_instance, 'c', None)
        self.results = getattr(solution_instance, 'results', {})
        self.M = self.results.get('M', None)
        self.v = self.results.get('v', None)

        # Derived totals (only if required inputs present)
        self._E_total = None
        self._p_total = None
        if self.m0 is not None and self.c is not None:
            # From problem: E_A = 3 m0 c^2, E_B = 2 m0 c^2
            self._E_total = 5.0 * self.m0 * self.c**2
            # p_A = sqrt((3 m0 c^2)^2/c^2 - (m0 c)^2) = sqrt(9 m0^2 c^2 - m0^2 c^2) = sqrt(8) m0 c
            self._p_total = math.sqrt(8.0) * self.m0 * self.c

    def _gamma(self, v: float, c: float) -> Optional[float]:
        if v is None or c is None or v >= c:
            return None
        return 1.0 / math.sqrt(1.0 - (v / c)**2)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_results_keys(self) -> Tuple[bool, str]:
        """Check that required keys exist in results."""
        if 'M' not in self.results:
            return (False, "Missing key 'M' in results")
        if 'v' not in self.results:
            return (False, "Missing key 'v' in results")
        return (True, "Both 'M' and 'v' present")

    def test_interface_numeric_types(self) -> Tuple[bool, str]:
        """Check that M and v are numeric (or None)."""
        for name, val in [('M', self.M), ('v', self.v)]:
            if val is None:
                return (False, f"{name} is None")
            try:
                _ = float(val)
            except (TypeError, ValueError):
                return (False, f"{name} is not numeric, got {type(val).__name__}")
        return (True, "M and v are numeric")

    # ------------------------------------------------------------------
    # Invariant (theorem) checks – use math.isclose with rel_tol=1e-2
    # ------------------------------------------------------------------
    def test_invariant_invariant_mass(self) -> Tuple[bool, str]:
        """Check that M matches invariant mass calculated from total E and p."""
        if self.m0 is None or self.c is None:
            return (False, "SKIP: m0 or c missing")
        if self.M is None:
            return (False, "SKIP: M missing")
        E = self._E_total
        p = self._p_total
        M_inv = math.sqrt(E**2 / self.c**4 - p**2 / self.c**2)
        if math.isclose(M_inv, self.M, rel_tol=1e-2):
            return (True, f"Invariant mass matches (computed {M_inv:.6f}, got {self.M:.6f})")
        else:
            return (False, f"Invariant mass mismatch (computed {M_inv:.6f}, got {self.M:.6f})")

    def test_invariant_energy_conservation(self) -> Tuple[bool, str]:
        """Check that composite energy equals total incident energy."""
        if self.m0 is None or self.c is None or self.M is None or self.v is None:
            return (False, "SKIP: missing data for energy check")
        gamma = self._gamma(self.v, self.c)
        if gamma is None:
            return (False, "SKIP: invalid v/c for gamma")
        E_comp = gamma * self.M * self.c**2
        if math.isclose(E_comp, self._E_total, rel_tol=1e-2):
            return (True, f"Energy conserved: composite E={E_comp:.6e} ≈ {self._E_total:.6e}")
        else:
            return (False, f"Energy NOT conserved: composite {E_comp:.6e} vs {self._E_total:.6e}")

    def test_invariant_momentum_conservation(self) -> Tuple[bool, str]:
        """Check that composite momentum equals incident momentum."""
        if self.m0 is None or self.c is None or self.M is None or self.v is None:
            return (False, "SKIP: missing data for momentum check")
        gamma = self._gamma(self.v, self.c)
        if gamma is None:
            return (False, "SKIP: invalid v/c for gamma")
        p_comp = gamma * self.M * self.v
        if math.isclose(p_comp, self._p_total, rel_tol=1e-2):
            return (True, f"Momentum conserved: composite p={p_comp:.6e} ≈ {self._p_total:.6e}")
        else:
            return (False, f"Momentum NOT conserved: composite {p_comp:.6e} vs {self._p_total:.6e}")

    # ------------------------------------------------------------------
    # Modeling / trace checks – verify one relation per test
    # ------------------------------------------------------------------
    def test_modeling_velocity_from_momentum_and_energy(self) -> Tuple[bool, str]:
        """Check v = p c^2 / E using total values."""
        if self._E_total is None or self._p_total is None or self.v is None:
            return (False, "SKIP: missing total E/p or v")
        v_derived = self._p_total * self.c**2 / self._E_total
        # Use math.isclose because it's a derivation
        if math.isclose(v_derived, self.v, rel_tol=1e-2):
            return (True, f"v from p/E matches (derived {v_derived:.6e}, got {self.v:.6e})")
        else:
            return (False, f"v mismatch (derived {v_derived:.6e}, got {self.v:.6e})")

    def test_modeling_gamma_composite_less_than_gamma_incident(self) -> Tuple[bool, str]:
        """Check that composite gamma is less than incident gamma (=3)."""
        if self.v is None or self.c is None:
            return (False, "SKIP: v/c missing")
        gamma_comp = self._gamma(self.v, self.c)
        if gamma_comp is None:
            return (False, "SKIP: invalid gamma")
        # Incident particle has gamma_A = 3 (from kinetic energy 2 m0 c^2)
        gamma_A = 3.0
        if gamma_comp < gamma_A:
            return (True, f"Composite gamma {gamma_comp:.4f} < incident gamma {gamma_A}")
        else:
            return (False, f"Composite gamma {gamma_comp:.4f} should be strictly less than {gamma_A}")

    # ------------------------------------------------------------------
    # Monotonicity test – perturb an input and compare baseline
    # ------------------------------------------------------------------
    def test_monotonicity_mass_increases_with_kinetic_energy(self) -> Tuple[bool, str]:
        """Perturb K_A by +10% and check that composite mass increases."""
        if self.m0 is None or self.c is None:
            return (False, "SKIP: m0 or c missing")
        # Baseline: K_A = 2 m0 c^2
        # Compute baseline mass (should match self.M if correct)
        baseline_M = math.sqrt(17.0) * self.m0  # sqrt(17)*m0

        # Perturbed: increase K_A by 10% -> K_A' = 2.2 m0 c^2
        # New E_A' = m0 c^2 + 2.2 m0 c^2 = 3.2 m0 c^2
        # gamma_A' = 3.2
        # p_A' = sqrt((3.2)^2 - 1) m0 c = sqrt(10.24 - 1) m0 c = sqrt(9.24) m0 c ≈ 3.04138 m0 c
        # E_total' = 3.2 m0 c^2 + 2 m0 c^2 = 5.2 m0 c^2
        # p_total' = p_A' ≈ 3.04138 m0 c
        # M' = sqrt( (5.2)^2 - (3.04138)^2 ) m0 = sqrt(27.04 - 9.250) m0 = sqrt(17.79) m0 ≈ 4.218 m0
        # Compare to baseline_M = 4.123... m0 => M' > baseline_M
        E_total_new = 5.2 * self.m0 * self.c**2
        p_total_new = math.sqrt(10.24 - 1.0) * self.m0 * self.c   # sqrt(9.24) * m0 c
        M_new = math.sqrt(E_total_new**2 / self.c**4 - p_total_new**2 / self.c**2)

        # Check that M_new > baseline_M
        if M_new > baseline_M:
            return (True, f"Perturbed mass M'={M_new:.6f} > baseline M={baseline_M:.6f}")
        else:
            return (False, f"Monotonicity violation: M'={M_new:.6f} <= baseline M={baseline_M:.6f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_v_less_than_c(self) -> Tuple[bool, str]:
        """Weak sanity: velocity must be less than light speed."""
        if self.v is None or self.c is None:
            return (False, "SKIP: missing v or c")
        if self.v < self.c:
            return (True, f"v = {self.v:.4e} < c = {self.c:.4e}")
        else:
            return (False, f"v = {self.v:.4e} should be less than c")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)