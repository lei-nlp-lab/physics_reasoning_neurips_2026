import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for compound microscope problem:
    - objective focal length f_o = 0.5 cm
    - eyepiece focal length f_e = 2 cm
    - lens separation L = 22 cm
    - final image at infinity
    - near point D = 15 cm (as given in problem statement)
    Required results: u (object distance from objective), M (magnifying power)
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely (using .get to avoid missing key errors)
        givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})

        self.f_o: Optional[float] = givens.get('f_o', None)
        self.f_e: Optional[float] = givens.get('f_e', None)
        self.L: Optional[float] = givens.get('L', None)  # distance between lenses
        # Try multiple keys for near point
        self.D: Optional[float] = givens.get('D', 
                                   givens.get('near_point', 
                                   givens.get('near_point_distance', None)))

        # Results from solving pseudocode
        self.u: Optional[float] = results.get('u', None)
        self.M: Optional[float] = results.get('M', None)

        # Explanation steps – useful for deciding what to check but we will not copy values
        self.explanation_steps: List[str] = getattr(solution_instance, 'explanation_steps', [])

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods."""
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

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        """Required result keys 'u' and 'M' must exist and be numeric."""
        if self.u is None:
            return False, "Missing result 'u'"
        if self.M is None:
            return False, "Missing result 'M'"
        if not isinstance(self.u, (int, float)):
            return False, f"'u' is not numeric: {type(self.u)}"
        if not isinstance(self.M, (int, float)):
            return False, f"'M' is not numeric: {type(self.M)}"
        return True, f"'u'={self.u:.4f}, 'M'={self.M:.2f}"

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that essential givens are provided (but not required for all tests)."""
        missing = []
        if self.f_o is None:
            missing.append("f_o")
        if self.f_e is None:
            missing.append("f_e")
        if self.L is None:
            missing.append("L")
        if self.D is None:
            missing.append("D (near point)")
        if missing:
            return True, f"Optional givens missing: {', '.join(missing)}. Some tests may be skipped."
        return True, "All givens present."

    # ------------------------------------------------------------------
    # Invariant / theorem‑based checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_v_o_equals_L_minus_f_e(self) -> Tuple[bool, str]:
        """Intermediate image formed at eyepiece front focal point: v_o = L - f_e."""
        if self.L is None or self.f_e is None:
            return False, "SKIP: L or f_e not available"
        # We do not have v_o directly; we can compute expected v_o and compare via lens eq later.
        # But we can check if the explanation steps contain consistent numbers.
        # Alternatively, we verify that from the lens equation everything is consistent.
        # However, this method is meant to verify the condition. We'll compute expected v_o.
        v_o_expected = self.L - self.f_e
        # Now we need to check if u and f_o satisfy lens equation with v_o_expected
        if self.u is None or self.f_o is None:
            return True, f"v_o expected = {v_o_expected:.2f} cm (cannot cross-check u/f_o)"
        # Compute v_o from lens equation: 1/v_o = 1/f_o - 1/u
        if self.u == 0:
            return False, "u is zero, cannot compute v_o"
        v_o_from_lens = 1.0 / (1.0 / self.f_o - 1.0 / self.u)
        if not math.isclose(v_o_from_lens, v_o_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"v_o from lens eq ({v_o_from_lens:.2f}) != expected {v_o_expected:.2f}"
        return True, f"v_o = {v_o_expected:.2f} cm, consistent with lens equation"

    def test_invariant_lens_equation_objective(self) -> Tuple[bool, str]:
        """Check 1/u + 1/v_o = 1/f_o with v_o = L - f_e."""
        if self.u is None or self.f_o is None or self.L is None or self.f_e is None:
            return False, "SKIP: missing u, f_o, L, or f_e"
        if self.u == 0:
            return False, "u is zero"
        v_o_expected = self.L - self.f_e
        lhs = 1.0 / self.u + 1.0 / v_o_expected
        rhs = 1.0 / self.f_o
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"1/u + 1/v_o = {lhs:.6f} != 1/f_o = {rhs:.6f}"
        return True, f"Lens equation satisfied: 1/{self.u:.3f} + 1/{v_o_expected:.1f} = {lhs:.4f} ≈ 1/{self.f_o:.1f}"

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_magnifying_power_formula(self) -> Tuple[bool, str]:
        """Verify M = (v_o / u) * (D / f_e)."""
        if self.u is None or self.M is None or self.f_e is None or self.D is None:
            return False, "SKIP: missing u, M, f_e, or D"
        if self.L is None:
            return False, "SKIP: missing L"
        v_o = self.L - self.f_e
        M_computed = (v_o / self.u) * (self.D / self.f_e)
        if not math.isclose(self.M, M_computed, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"M = {self.M:.2f} but expected (v_o/u)*(D/f_e) = {M_computed:.2f}"
        return True, f"M = {self.M:.2f} matches formula with D={self.D} cm"

    def test_modeling_v_o_derived_correctly(self) -> Tuple[bool, str]:
        """Cross‑check v_o from lens equation with the expected value L - f_e."""
        if self.u is None or self.f_o is None or self.L is None or self.f_e is None:
            return False, "SKIP: missing required quantities"
        v_o_expected = self.L - self.f_e
        if self.u == 0:
            return False, "u is zero"
        v_o_actual = 1.0 / (1.0 / self.f_o - 1.0 / self.u)
        if not math.isclose(v_o_actual, v_o_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"v_o derived from u={self.u:.3f} gives {v_o_actual:.2f}, expected {v_o_expected:.2f}"
        return True, f"v_o = {v_o_actual:.2f} cm ≈ {v_o_expected:.2f} cm"

    # ------------------------------------------------------------------
    # Monotonicity checks (perturb input, observe output trend)
    # ------------------------------------------------------------------
    def test_monotonicity_u_with_f_o(self) -> Tuple[bool, str]:
        """If f_o increases, u should increase (for fixed v_o)."""
        if self.u is None or self.f_o is None or self.L is None or self.f_e is None:
            return False, "SKIP: missing u, f_o, L, f_e"
        v_o = self.L - self.f_e
        # Perturb f_o upward by 10%
        f_o_pert = self.f_o * 1.1
        # Compute new u (still using same v_o)
        # u_new = 1/(1/f_o_pert - 1/v_o)
        if math.isclose(1/f_o_pert, 1/v_o, rel_tol=1e-6):
            return False, "Denominator zero, cannot compute u_new"
        u_new = 1.0 / (1.0 / f_o_pert - 1.0 / v_o)
        if u_new <= self.u:
            return False, f"Increasing f_o from {self.f_o:.2f} to {f_o_pert:.2f} should increase u, but u went from {self.u:.3f} to {u_new:.3f}"
        return True, f"Monotonicity OK: f_o +10% → u from {self.u:.3f} to {u_new:.3f}"

    def test_monotonicity_M_with_D(self) -> Tuple[bool, str]:
        """If near point D increases, magnifying power M should increase proportionally."""
        if self.M is None or self.D is None or self.u is None or self.f_e is None:
            return False, "SKIP: missing M, D, u, f_e"
        if self.L is None:
            return False, "SKIP: missing L"
        v_o = self.L - self.f_e
        # Compute expected M with D increased by 10%
        D_pert = self.D * 1.1
        M_pert = (v_o / self.u) * (D_pert / self.f_e)
        if M_pert <= self.M:
            return False, f"Increasing D from {self.D:.1f} to {D_pert:.1f} should increase M, but M went from {self.M:.2f} to {M_pert:.2f}"
        return True, f"Monotonicity OK: D +10% → M from {self.M:.2f} to {M_pert:.2f}"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_magnitude(self) -> Tuple[bool, str]:
        """Very weak check: u positive and reasonable, M finite and large."""
        if self.u is None or self.M is None:
            return False, "SKIP: missing u or M"
        msgs = []
        if self.u <= 0:
            msgs.append("u not positive")
        if self.u > 10:  # object distance for a microscope is typically < 1 cm
            msgs.append(f"u={self.u:.2f} is unusually large")
        if math.isinf(self.M) or math.isnan(self.M):
            msgs.append("M is infinite or NaN")
        if self.M < 100:
            msgs.append(f"M={self.M:.1f} is below typical microscope range (100+)")
        if self.M > 1000:
            msgs.append(f"M={self.M:.1f} seems too large for simple microscope")
        if msgs:
            return False, "; ".join(msgs)
        return True, f"u={self.u:.3f}, M={self.M:.1f} pass sanity bounds"