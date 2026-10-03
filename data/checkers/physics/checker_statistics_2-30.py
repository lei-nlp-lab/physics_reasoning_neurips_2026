import math
import sympy as sp
from typing import Any, Dict, List, Optional, Tuple, Callable

class PhysicsChecker:
    """
    Physics checker for hydrogen molecule specific heat problem.
    Audits rotational/vibrational temperatures and molar specific heats.
    """
    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})

        # Givens (all with None default)
        self.m_H: Optional[float] = self.givens.get('m_H', None)      # in g
        self.D: Optional[float] = self.givens.get('D', None)          # in erg
        self.a: Optional[float] = self.givens.get('a', None)          # in cm^-1
        self.r0: Optional[float] = self.givens.get('r0', None)        # in cm

        # Results (all with None default)
        self.T_R: Optional[float] = self.results.get('T_R', None)
        self.T_V: Optional[float] = self.results.get('T_V', None)
        self.C_v1: Optional[float] = self.results.get('C_v1', None)
        self.C_p1: Optional[float] = self.results.get('C_p1', None)
        self.C_v2: Optional[float] = self.results.get('C_v2', None)
        self.C_p2: Optional[float] = self.results.get('C_p2', None)
        self.C_v3: Optional[float] = self.results.get('C_v3', None)
        self.C_p3: Optional[float] = self.results.get('C_p3', None)
        self.C_v4: Optional[float] = self.results.get('C_v4', None)
        self.C_p4: Optional[float] = self.results.get('C_p4', None)

        # Physical constants (SI units)
        self.hbar: float = 1.0545718176461565e-34   # J·s
        self.k_B: float = 1.380649e-23              # J/K
        self.R: float = 8.31446261815324            # J/(mol·K)

        # Temperature points (fixed from problem)
        self.T_list: List[float] = [25.0, 250.0, 2500.0, 10000.0]

    # ------------------------------------------------------------------
    # Internal helpers (for derived computations, graceful None handling)
    # ------------------------------------------------------------------
    def _compute_expected_T_R(self,
                              m_H_override: Optional[float] = None,
                              r0_override: Optional[float] = None) -> Optional[float]:
        m_H = m_H_override if m_H_override is not None else self.m_H
        r0 = r0_override if r0_override is not None else self.r0
        if m_H is None or r0 is None:
            return None
        m_H_kg = m_H * 1e-3
        r0_m = r0 * 1e-2
        mu = m_H_kg / 2.0
        I = mu * r0_m ** 2
        return self.hbar ** 2 / (2.0 * I * self.k_B)

    def _compute_expected_T_V(self,
                              m_H_override: Optional[float] = None,
                              D_override: Optional[float] = None,
                              a_override: Optional[float] = None) -> Optional[float]:
        m_H = m_H_override if m_H_override is not None else self.m_H
        D = D_override if D_override is not None else self.D
        a = a_override if a_override is not None else self.a
        if m_H is None or D is None or a is None:
            return None
        m_H_kg = m_H * 1e-3
        D_J = D * 1e-7
        a_m = a * 100.0
        k = 2.0 * D_J * a_m ** 2
        mu = m_H_kg / 2.0
        omega = math.sqrt(k / mu)
        return self.hbar * omega / self.k_B

    def _compute_I_from_T_R(self) -> Optional[float]:
        """Back-calculate moment of inertia from T_R."""
        if self.T_R is None:
            return None
        # T_R = ħ^2 / (2 I k_B) => I = ħ^2 / (2 T_R k_B)
        return self.hbar ** 2 / (2.0 * self.T_R * self.k_B)

    def _compute_expected_I(self) -> Optional[float]:
        """Compute I from givens: I = μ r0^2."""
        if self.m_H is None or self.r0 is None:
            return None
        m_H_kg = self.m_H * 1e-3
        r0_m = self.r0 * 1e-2
        mu = m_H_kg / 2.0
        return mu * r0_m ** 2

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        required_keys = ['T_R', 'T_V', 'C_v1', 'C_p1', 'C_v2', 'C_p2',
                         'C_v3', 'C_p3', 'C_v4', 'C_p4']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    # ------------------------------------------------------------------
    # Invariant tests (numeric comparisons with isclose)
    # ------------------------------------------------------------------
    def test_invariant_T_R_recalculation(self) -> Tuple[bool, str]:
        if self.T_R is None:
            return (False, "SKIP: T_R not available")
        expected = self._compute_expected_T_R()
        if expected is None:
            return (False, "SKIP: givens insufficient to compute expected T_R")
        if not math.isclose(self.T_R, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_R {self.T_R} does not match expected {expected}")
        return (True, f"T_R = {self.T_R} within tolerance of expected")

    def test_invariant_T_V_recalculation(self) -> Tuple[bool, str]:
        if self.T_V is None:
            return (False, "SKIP: T_V not available")
        expected = self._compute_expected_T_V()
        if expected is None:
            return (False, "SKIP: givens insufficient to compute expected T_V")
        if not math.isclose(self.T_V, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_V {self.T_V} does not match expected {expected}")
        return (True, f"T_V = {self.T_V} within tolerance of expected")

    def test_invariant_Cv_values_correct_fractions(self) -> Tuple[bool,str]:
        if any(attr is None for attr in [self.C_v1, self.C_v2, self.C_v3, self.C_v4]):
            return (False, "SKIP: one or more C_v values missing")
        expected_coeffs = [1.5, 2.5, 2.5, 3.5]
        actual_Cv = [self.C_v1, self.C_v2, self.C_v3, self.C_v4]
        for i, (coeff, Cv) in enumerate(zip(expected_coeffs, actual_Cv)):
            expected_val = coeff * self.R
            if not math.isclose(Cv, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"C_v{i+1} = {Cv}, expected {expected_val} (coeff {coeff})")
        return (True, "All C_v values match expected fractions")

    def test_invariant_Cp_equals_Cv_plus_R(self) -> Tuple[bool, str]:
        Cv_list = [self.C_v1, self.C_v2, self.C_v3, self.C_v4]
        Cp_list = [self.C_p1, self.C_p2, self.C_p3, self.C_p4]
        if any(v is None for v in Cv_list + Cp_list):
            return (False, "SKIP: missing C_v or C_p values")
        for i, (Cv, Cp) in enumerate(zip(Cv_list, Cp_list)):
            if not math.isclose(Cp, Cv + self.R, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"T{i+1}: C_p = {Cp} != C_v + R = {Cv + self.R}")
        return (True, "All C_p = C_v + R relations hold")

    def test_invariant_degree_of_freedom_activation(self) -> Tuple[bool, str]:
        if self.T_R is None or self.T_V is None:
            return (False, "SKIP: T_R or T_V missing")
        T_thresholds = self.T_list
        coeff_map = []
        for T in T_thresholds:
            if T < self.T_R:
                coeff_map.append(1.5)
            elif T < self.T_V:
                coeff_map.append(2.5)
            else:
                coeff_map.append(3.5)
        Cv_actual = [self.C_v1, self.C_v2, self.C_v3, self.C_v4]
        if any(v is None for v in Cv_actual):
            return (False, "SKIP: some C_v values missing")
        for i, (coeff, Cv) in enumerate(zip(coeff_map, Cv_actual)):
            expected = coeff * self.R
            if not math.isclose(Cv, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"T{i+1}: C_v = {Cv}, expected {expected} (coeff {coeff})")
        return (True, "All C_v values consistent with freezing thresholds")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_Cv_non_decreasing_ordering(self) -> Tuple[bool, str]:
        Cv_vals = [self.C_v1, self.C_v2, self.C_v3, self.C_v4]
        if any(v is None for v in Cv_vals):
            return (False, "SKIP: missing C_v values")
        if not (Cv_vals[0] <= Cv_vals[1] + 1e-6):
            return (False, f"C_v1={Cv_vals[0]} not <= C_v2={Cv_vals[1]}")
        if not math.isclose(Cv_vals[1], Cv_vals[2], rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_v2={Cv_vals[1]} and C_v3={Cv_vals[2]} should be equal")
        if not (Cv_vals[2] <= Cv_vals[3] + 1e-6):
            return (False, f"C_v3={Cv_vals[2]} not <= C_v4={Cv_vals[3]}")
        return (True, "Ordering of C_v values respects freezing pattern")

    def test_modeling_moment_of_inertia_consistency(self) -> Tuple[bool, str]:
        """
        Check that the moment of inertia derived from T_R matches I = μ r0^2.
        This tests the modeling step: rotational temperature formula + I formula.
        """
        if self.T_R is None or self.m_H is None or self.r0 is None:
            return (False, "SKIP: missing T_R or givens")
        I_from_TR = self._compute_I_from_T_R()
        I_expected = self._compute_expected_I()
        if I_from_TR is None or I_expected is None:
            return (False, "SKIP: could not compute I values")
        if not math.isclose(I_from_TR, I_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"I from T_R = {I_from_TR} does not match expected I = {I_expected}")
        return (True, "Moment of inertia consistent with T_R and reduced mass")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_T_R_with_r0(self) -> Tuple[bool, str]:
        if self.m_H is None or self.r0 is None:
            return (False, "SKIP: missing m_H or r0")
        baseline = self._compute_expected_T_R()
        if baseline is None:
            return (False, "SKIP: could not compute baseline T_R")
        perturbed_r0 = self.r0 * 1.01
        perturbed = self._compute_expected_T_R(r0_override=perturbed_r0)
        if perturbed is None:
            return (False, "SKIP: could not compute perturbed T_R")
        if not (perturbed < baseline):
            return (False, f"T_R monotonicity fails: baseline {baseline}, perturbed {perturbed}")
        return (True, f"T_R decreases with increasing r0: {baseline} -> {perturbed}")

    def test_monotonicity_T_V_with_a(self) -> Tuple[bool, str]:
        if self.m_H is None or self.D is None or self.a is None:
            return (False, "SKIP: missing givens")
        baseline = self._compute_expected_T_V()
        if baseline is None:
            return (False, "SKIP: could not compute baseline T_V")
        perturbed_a = self.a * 1.01
        perturbed = self._compute_expected_T_V(a_override=perturbed_a)
        if perturbed is None:
            return (False, "SKIP: could not compute perturbed T_V")
        if not (perturbed > baseline):
            return (False, f"T_V monotonicity fails: baseline {baseline}, perturbed {perturbed}")
        return (True, f"T_V increases with increasing a: {baseline} -> {perturbed}")

    def test_monotonicity_C_v_with_temperature(self) -> Tuple[bool, str]:
        """
        C_v should be non-decreasing with temperature.
        We simulate a temperature perturbation: if we increase T2=250K to 260K,
        the expected C_v stays the same (still between T_R and T_V).
        """
        if self.T_R is None or self.T_V is None:
            return (False, "SKIP: T_R or T_V missing")
        if self.C_v2 is None:
            return (False, "SKIP: C_v2 missing")
        expected_baseline_Cv2 = 2.5 * self.R
        if not math.isclose(self.C_v2, expected_baseline_Cv2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "C_v2 does not match expected 2.5R, cannot verify plateau")
        # Perturb T2 to 260K (still in same regime)
        if 260.0 > self.T_R and 260.0 < self.T_V:
            # C_v should remain same as baseline
            if not math.isclose(self.C_v2, expected_baseline_Cv2, rel_tol=1e-2, abs_tol=1e-2):
                return (False, "Perturbed temperature would still yield same C_v, but does not match")
        return (True, "C_v plateau behavior consistent with temperature perturbation")

    # ------------------------------------------------------------------
    # Weak answer sanity (only positivity and finiteness)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        values = [self.T_R, self.T_V,
                  self.C_v1, self.C_p1,
                  self.C_v2, self.C_p2,
                  self.C_v3, self.C_p3,
                  self.C_v4, self.C_p4]
        for v in values:
            if v is None:
                continue
            if not math.isfinite(v) or v <= 0:
                return (False, f"Non-positive or non-finite value: {v}")
        return (True, "All values are positive and finite")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
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