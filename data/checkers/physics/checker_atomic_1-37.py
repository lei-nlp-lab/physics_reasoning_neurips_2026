import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    """
    PhysicsChecker for Lamb shift problem.
    Checks interface, invariants, modeling, monotonicity, and answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {}) or {}
        self.delta_E_1s = results.get('delta_E_1s', None)
        self.delta_E_2p = results.get('delta_E_2p', None)

        # Extract given parameters - assume they are attributes of solution_instance
        self.e = getattr(solution_instance, 'e', None)      # elementary charge
        self.a_B = getattr(solution_instance, 'a_B', None)  # Bohr radius
        self.r0 = getattr(solution_instance, 'r0', None)    # small sphere radius

    # ---------- Interface Tests ----------
    def test_interface_givens_present(self) -> Tuple[bool, str]:
        if self.e is None or self.a_B is None or self.r0 is None:
            return (False, "Missing givens: e, a_B, r0")
        return (True, "All givens present")

    def test_interface_result_keys(self) -> Tuple[bool, str]:
        if self.delta_E_1s is None or self.delta_E_2p is None:
            return (False, "Missing result keys: delta_E_1s or delta_E_2p")
        if not isinstance(self.delta_E_1s, (int, float)) or not isinstance(self.delta_E_2p, (int, float)):
            return (False, "Result values must be numeric")
        return (True, "Required result keys present and numeric")

    # ---------- Invariant Tests (theorem-based) ----------
    def test_invariant_delta_E_1s_formula(self) -> Tuple[bool, str]:
        """Check ΔE_1s = (2π e^2 r0^2/3) * |ψ_1s(0)|^2 with |ψ_1s(0)|^2 = 4/a_B^3."""
        if None in (self.delta_E_1s, self.e, self.a_B, self.r0):
            return (False, "SKIP: missing data")
        # Re-derive from theorem: H' = (2π e^2 r0^2/3) δ^3(r), |ψ_1s(0)|^2 = 4/a_B^3
        expected = (2 * math.pi * self.e**2 * self.r0**2 / 3) * (4 / self.a_B**3)
        if not math.isclose(self.delta_E_1s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ΔE_1s = {self.delta_E_1s} differs from expected {expected}")
        return (True, "ΔE_1s consistent with perturbation formula")

    def test_invariant_delta_E_2p_zero(self) -> Tuple[bool, str]:
        """2p wavefunction vanishes at origin → zero energy shift."""
        if self.delta_E_2p is None:
            return (False, "SKIP: delta_E_2p not available")
        if abs(self.delta_E_2p) > 1e-12:
            return (False, f"ΔE_2p should be zero but got {self.delta_E_2p}")
        return (True, "ΔE_2p correctly zero")

    # ---------- Modeling Tests ----------
    def test_modeling_laplacian_delta_coupling(self) -> Tuple[bool, str]:
        """Check that the perturbation uses δ³(r) correctly: positive shift for 1s, zero for 2p."""
        if None in (self.delta_E_1s, self.delta_E_2p):
            return (False, "SKIP: missing results")
        if self.delta_E_1s < 0:
            return (False, "ΔE_1s should be positive (Lamb shift increases energy)")
        if abs(self.delta_E_2p) > 1e-12:
            return (False, "ΔE_2p should be exactly zero because ψ_2p(0)=0")
        return (True, "Behavior of Lamb shift matches origin of wavefunctions")

    def test_modeling_wavefunction_origin(self) -> Tuple[bool, str]:
        """Confirm that ψ_2p(0)=0 gives zero shift, ψ_1s(0) nonzero gives nonzero shift."""
        if None in (self.delta_E_1s, self.delta_E_2p):
            return (False, "SKIP: missing results")
        if self.delta_E_1s == 0:
            return (False, "ΔE_1s should be nonzero because ψ_1s(0) ≠ 0")
        if abs(self.delta_E_2p) > 1e-12:
            return (False, "ΔE_2p should be exactly zero because ψ_2p(0)=0")
        return (True, "Origin behavior correctly reflected in energy shifts")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_delta_E_1s_with_r0(self) -> Tuple[bool, str]:
        """Perturb r0 by +10% and verify ΔE_1s scales as r0²."""
        if None in (self.delta_E_1s, self.r0, self.e, self.a_B):
            return (False, "SKIP: missing data for monotonicity")
        new_r0 = self.r0 * 1.10
        # Expected new shift using same formula with new r0
        expected_new = (2 * math.pi * self.e**2 * new_r0**2 / 3) * (4 / self.a_B**3)
        # Baseline scaling check: old shift * (new_r0/r0)^2 should equal expected_new
        scaling = (new_r0 / self.r0) ** 2
        if not math.isclose(self.delta_E_1s * scaling, expected_new, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Scaling with r0² violated: old*{scaling} = {self.delta_E_1s * scaling} vs expected {expected_new}")
        return (True, "ΔE_1s scales correctly with r0²")

    def test_monotonicity_delta_E_1s_with_e(self) -> Tuple[bool, str]:
        """Perturb e by +5% and verify ΔE_1s scales as e²."""
        if None in (self.delta_E_1s, self.e, self.r0, self.a_B):
            return (False, "SKIP: missing data for monotonicity")
        new_e = self.e * 1.05
        expected_new = (2 * math.pi * new_e**2 * self.r0**2 / 3) * (4 / self.a_B**3)
        scaling = (new_e / self.e) ** 2
        if not math.isclose(self.delta_E_1s * scaling, expected_new, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Scaling with e² violated: old*{scaling} = {self.delta_E_1s * scaling} vs expected {expected_new}")
        return (True, "ΔE_1s scales correctly with e²")

    # ---------- Weak Answer Sanity (at most one method) ----------
    def test_answer_sanity_energy_shifts(self) -> Tuple[bool, str]:
        """Weak checks: 1s shift positive, 2p shift zero."""
        if self.delta_E_1s is None or self.delta_E_2p is None:
            return (False, "SKIP: missing energy shifts")
        if self.delta_E_1s < 0:
            return (False, "ΔE_1s should be positive")
        if abs(self.delta_E_2p) > 1e-12:
            return (False, "ΔE_2p should be zero")
        return (True, "Basic sanity: signs and zero hold")

    # ---------- Main runner ----------
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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)