import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits a solution for the solenoid inductance problem.
    Checks:
      - Required interface keys (N, diameter, length, mu0, L)
      - Invariant: inductance formula L = mu0 * N^2 * pi * (diameter/2)^2 / length
      - Modeling: long-solenoid condition (length >> radius)
      - Modeling: magnetic field (B = mu0 * N * I / l) and flux linkage
      - Trace: pseudocode emit key 'L' is present and consistent
      - Monotonicity: L increases when N increases; L decreases when length increases
      - Answer sanity: L is finite, positive, and within plausible range
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction of solution data – never assign numerical defaults
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract givens (also may be in solution_instance.givens or directly)
        self.N = None
        self.diameter = None
        self.length = None
        self.mu0 = None

        # Try from instance attributes
        if hasattr(solution_instance, 'N'):
            self.N = solution_instance.N
        if hasattr(solution_instance, 'diameter'):
            self.diameter = solution_instance.diameter
        if hasattr(solution_instance, 'length'):
            self.length = solution_instance.length
        if hasattr(solution_instance, 'mu0'):
            self.mu0 = solution_instance.mu0

        # Also check results dict
        if 'N' in self.results and self.N is None:
            self.N = self.results.get('N')
        if 'diameter' in self.results and self.diameter is None:
            self.diameter = self.results.get('diameter')
        if 'length' in self.results and self.length is None:
            self.length = self.results.get('length')
        if 'mu0' in self.results and self.mu0 is None:
            self.mu0 = self.results.get('mu0')

        # Extract computed inductance
        self.L = self.results.get('L', None)

        # Derived quantities (radius, area)
        self.radius = None
        if self.diameter is not None:
            self.radius = self.diameter / 2.0

        self.area = None
        if self.radius is not None:
            self.area = math.pi * self.radius ** 2

        # Expected inductance computed from formula (if givens are available)
        self.expected_L = None
        if all(v is not None for v in [self.N, self.radius, self.length, self.mu0]):
            self.expected_L = (self.mu0 * self.N ** 2 * math.pi * self.radius ** 2) / self.length

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that all required givens are present and positive."""
        checks = {
            'N': self.N,
            'diameter': self.diameter,
            'length': self.length,
            'mu0': self.mu0
        }
        missing = [k for k, v in checks.items() if v is None]
        if missing:
            return (False, f"Missing required givens: {missing}")
        zero_neg = [k for k, v in checks.items() if v is not None and v <= 0]
        if zero_neg:
            return (False, f"Non-positive givens: {zero_neg}")
        return (True, "All required givens present and positive")

    def test_interface_result_L(self) -> Tuple[bool, str]:
        """Check that inductance L is present and is a number."""
        if self.L is None:
            return (False, "Missing result L")
        if not isinstance(self.L, (int, float)):
            return (False, f"L is not numeric: {type(self.L)}")
        return (True, f"L = {self.L}")

    # ------------------------------------------------------------------
    # Invariant checks (use isclose with tolerance)
    # ------------------------------------------------------------------
    def test_invariant_inductance_formula(self) -> Tuple[bool, str]:
        """Verify L equals mu0*N^2*A/l within 1-2%."""
        if self.L is None or self.expected_L is None:
            return (False, "SKIP: missing L or givens")
        if self.expected_L == 0:
            return (False, "Expected L is zero, cannot compare")
        if not math.isfinite(self.expected_L):
            return (False, f"Expected L is not finite: {self.expected_L}")
        rel_tol = 1e-2  # 1% relative tolerance
        abs_tol = 1e-2 * abs(self.expected_L)  # 1% of expected value as absolute
        if not math.isclose(self.L, self.expected_L, rel_tol=rel_tol, abs_tol=abs_tol):
            return (False, f"L={self.L} does not match expected {self.expected_L} "
                           f"(rel_tol={rel_tol}, abs_tol={abs_tol})")
        return (True, f"L matches formula: {self.L} vs {self.expected_L}")

    def test_invariant_long_solenoid_condition(self) -> Tuple[bool, str]:
        """Check that length >> radius (at least factor 5)."""
        if self.length is None or self.radius is None:
            return (False, "SKIP: missing length or radius")
        if self.radius == 0:
            return (False, "Zero radius, cannot compute ratio")
        ratio = self.length / self.radius
        if ratio < 5:
            return (False, f"Length/Radius = {ratio:.2f} < 5, long-solenoid approximation may be poor")
        return (True, f"Long solenoid condition satisfied: l/r = {ratio:.2f}")

    # ------------------------------------------------------------------
    # Modeling / Trace checks
    # ------------------------------------------------------------------
    def test_modeling_inductance_derivation(self) -> Tuple[bool, str]:
        """Check that explanation steps yield same formula via symbolic derivation."""
        try:
            N_sym = sp.symbols('N', positive=True)
            mu0_sym = sp.symbols('mu0', positive=True)
            r_sym = sp.symbols('r', positive=True)
            l_sym = sp.symbols('l', positive=True)
            A_sym = sp.pi * r_sym ** 2
            L_sym = mu0_sym * N_sym ** 2 * A_sym / l_sym
            if all(v is not None for v in [self.N, self.radius, self.length, self.mu0]):
                L_val = float(L_sym.subs({N_sym: self.N, mu0_sym: self.mu0, r_sym: self.radius, l_sym: self.length}))
                if self.L is None:
                    return (False, "SKIP: no L to compare")
                if not math.isclose(self.L, L_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Symbolic formula gives {L_val}, but result L={self.L}")
                return (True, "Inductance matches symbolic derivation")
            else:
                return (False, "SKIP: missing givens for symbolic check")
        except Exception as e:
            return (False, f"Symbolic check failed: {e}")

    def test_trace_pseudocode_emit(self) -> Tuple[bool, str]:
        """Check that the pseudocode emit key 'L' is present and matches the computed L."""
        if self.L is None:
            return (False, "SKIP: no L in results")
        if self.expected_L is not None:
            if not math.isclose(self.L, self.expected_L, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Emitted L={self.L} does not match formula {self.expected_L}")
        return (True, "Pseudocode emit L is present and consistent with formula")

    def test_modeling_area_computation(self) -> Tuple[bool, str]:
        """Verify that cross-sectional area is correctly computed from diameter."""
        if self.diameter is None or self.radius is None:
            return (False, "SKIP: missing diameter")
        expected_radius = self.diameter / 2.0
        expected_area = math.pi * expected_radius ** 2
        if self.area is None:
            return (False, "Area not computed (None)")
        if not math.isclose(self.area, expected_area, rel_tol=1e-4):
            return (False, f"Area={self.area} does not match expected {expected_area}")
        return (True, f"Area computed correctly: {self.area}")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_L_with_N_increase(self) -> Tuple[bool, str]:
        """Check that increasing N by 10% increases L (quadratically)."""
        if None in (self.N, self.radius, self.length, self.mu0, self.L):
            return (False, "SKIP: missing data for monotonicity test")
        # Perturb N upward by 10%
        N_pert = self.N * 1.1
        L_pert = (self.mu0 * N_pert ** 2 * math.pi * self.radius ** 2) / self.length
        # Baseline is self.L; perturbed should be larger (> 1%)
        if L_pert <= self.L * 1.001:  # allow 0.1% margin due to rounding
            return (False, f"Increasing N by 10% did not increase L: baseline {self.L}, perturbed {L_pert}")
        return (True, f"N increase: L increased from {self.L} to {L_pert}")

    def test_monotonicity_L_with_length_decrease(self) -> Tuple[bool, str]:
        """Check that decreasing length by 10% increases L."""
        if None in (self.N, self.radius, self.length, self.mu0, self.L):
            return (False, "SKIP: missing data for monotonicity test")
        # Perturb length downward by 10%
        l_pert = self.length * 0.9
        L_pert = (self.mu0 * self.N ** 2 * math.pi * self.radius ** 2) / l_pert
        if self.length <= 0:
            return (False, "Length non-positive, cannot perturb")
        if L_pert <= self.L * 1.001:
            return (False, f"Decreasing length by 10% did not increase L: baseline {self.L}, perturbed {L_pert}")
        return (True, f"Length decrease: L increased from {self.L} to {L_pert}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, no isclose, no re-derive)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """Check that L is finite, positive, and within plausible range."""
        if self.L is None:
            return (False, "SKIP: no L to check")
        if not math.isfinite(self.L):
            return (False, f"L is not finite: {self.L}")
        if self.L <= 0:
            return (False, f"L is not positive: {self.L}")
        # Plausible magnitude: solenoid inductances are typically µH to mH
        if self.L > 1.0:  # too large for this small solenoid
            return (False, f"L={self.L} too large for a small solenoid")
        return (True, f"L = {self.L} is plausible")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, run them, and return (all_pass, log)."""
        logs = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)