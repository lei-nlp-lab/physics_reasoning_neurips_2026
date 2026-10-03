import sympy as sp
import math
from typing import Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance: Any) -> None:
        self.results = getattr(solution_instance, 'results', {})
        # Extract required keys with None defaults
        self.C_xy = self.results.get('C_xy', None)
        self.comm_j0 = self.results.get('comm_j0', None)
        self.comm_jhalf = self.results.get('comm_jhalf', None)
        self.eigenbasis = self.results.get('eigenbasis', None)

    # ----------------------------------------------------------------
    # Helper: safe numeric check (only for scalar zero comparisons)
    # ----------------------------------------------------------------
    def _is_zero(self, val: Any) -> bool:
        """Return True if val is exactly 0 (int, float, sympy zero)."""
        if val is None:
            return False
        if isinstance(val, (int, float)):
            return val == 0
        if sp.isinstance(val, sp.Expr):
            return sp.simplify(val) == 0
        return False

    # ----------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------
    def test_interface_C_xy_exists(self) -> Tuple[bool, str]:
        if self.C_xy is None:
            return (False, "C_xy not found in results")
        return (True, "C_xy present")

    def test_interface_comm_j0_exists(self) -> Tuple[bool, str]:
        if self.comm_j0 is None:
            return (False, "comm_j0 not found in results")
        return (True, "comm_j0 present")

    def test_interface_comm_jhalf_exists(self) -> Tuple[bool, str]:
        if self.comm_jhalf is None:
            return (False, "comm_jhalf not found in results")
        return (True, "comm_jhalf present")

    def test_interface_eigenbasis_exists(self) -> Tuple[bool, str]:
        if self.eigenbasis is None:
            return (False, "eigenbasis not found in results")
        return (True, "eigenbasis present")

    def test_interface_C_xy_type(self) -> Tuple[bool, str]:
        if self.C_xy is None:
            return (False, "SKIP: C_xy missing")
        if not isinstance(self.C_xy, (sp.Expr, str)):
            return (False, f"C_xy type is {type(self.C_xy)}, expected sympy expression or string")
        return (True, "C_xy type OK")

    def test_interface_comm_j0_type(self) -> Tuple[bool, str]:
        if self.comm_j0 is None:
            return (False, "SKIP: comm_j0 missing")
        if not self._is_zero(self.comm_j0):
            return (False, f"comm_j0 is {self.comm_j0}, expected zero")
        return (True, "comm_j0 type and value OK (zero)")

    def test_interface_comm_jhalf_type(self) -> Tuple[bool, str]:
        if self.comm_jhalf is None:
            return (False, "SKIP: comm_jhalf missing")
        if not self._is_zero(self.comm_jhalf):
            return (False, f"comm_jhalf is {self.comm_jhalf}, expected zero")
        return (True, "comm_jhalf type and value OK (zero)")

    def test_interface_eigenbasis_type(self) -> Tuple[bool, str]:
        if self.eigenbasis is None:
            return (False, "SKIP: eigenbasis missing")
        if not isinstance(self.eigenbasis, list):
            return (False, f"eigenbasis type is {type(self.eigenbasis)}, expected list")
        if len(self.eigenbasis) != 3:
            return (False, f"eigenbasis length is {len(self.eigenbasis)}, expected 3")
        return (True, "eigenbasis type and length OK")

    # ----------------------------------------------------------------
    # INVARIANT TESTS (theorem-based)
    # ----------------------------------------------------------------
    def test_invariant_comm_j0_zero(self) -> Tuple[bool, str]:
        if self.comm_j0 is None:
            return (False, "SKIP: comm_j0 missing")
        if not self._is_zero(self.comm_j0):
            return (False, "comm_j0 should be zero, but it is not")
        return (True, "comm_j0 is zero as required")

    def test_invariant_comm_jhalf_zero(self) -> Tuple[bool, str]:
        if self.comm_jhalf is None:
            return (False, "SKIP: comm_jhalf missing")
        if not self._is_zero(self.comm_jhalf):
            return (False, "comm_jhalf should be zero, but it is not")
        return (True, "comm_jhalf is zero as required")

    def test_invariant_comm_j1_vanishes(self) -> Tuple[bool, str]:
        """
        For j=1, explicitly compute [J_x^2, J_y^2] in the standard 3D representation
        and check it is the zero matrix.
        """
        if self.eigenbasis is None:
            return (False, "SKIP: eigenbasis missing, cannot construct j=1 matrices")
        hbar = sp.Symbol('hbar', real=True, positive=True)
        # Standard j=1 matrices
        Jz = sp.diag(hbar, 0, -hbar)
        Jx = (hbar/sp.sqrt(2)) * sp.Matrix([[0,1,0],[1,0,1],[0,1,0]])
        Jy = (hbar/sp.sqrt(2)) * sp.Matrix([[0,-sp.I,0],[sp.I,0,-sp.I],[0,sp.I,0]])
        Jx2 = Jx * Jx
        Jy2 = Jy * Jy
        comm = Jx2 * Jy2 - Jy2 * Jx2
        if sp.simplify(comm) == sp.zeros(3,3):
            return (True, "[J_x^2, J_y^2] = 0 for j=1 representation")
        # Fallback numerical check
        comm_num = sp.N(comm.subs(hbar, 1))
        if all(abs(comm_num[i,j]) < 1e-10 for i in range(3) for j in range(3)):
            return (True, "[J_x^2, J_y^2] ≈ 0 numerically")
        return (False, f"Commutator is not zero; max entry = {max(abs(comm_num[i,j]) for i,j in sp.iterable)}")

    # ----------------------------------------------------------------
    # MODELING / TRACE TESTS
    # ----------------------------------------------------------------
    def test_modeling_eigenbasis_length(self) -> Tuple[bool, str]:
        if self.eigenbasis is None:
            return (False, "SKIP: eigenbasis missing")
        if len(self.eigenbasis) != 3:
            return (False, f"Expected 3 basis states, got {len(self.eigenbasis)}")
        return (True, "Eigenbasis has 3 states")

    def test_modeling_eigenbasis_content(self) -> Tuple[bool, str]:
        """
        Check that the eigenbasis contains the expected forms:
        |1,0> and two sqrt(2) combos of |1,±1>.
        """
        if self.eigenbasis is None:
            return (False, "SKIP: eigenbasis missing")
        basis = self.eigenbasis
        try:
            strs = [str(entry) for entry in basis]
        except Exception:
            return (False, "Cannot convert eigenbasis entries to string")
        has_10 = any("1,0" in s or "|1,0" in s for s in strs)
        combos = [s for s in strs if ("1,1" in s or "1,-1" in s) and "sqrt" in s]
        if len(combos) == 2:
            return (True, "Eigenbasis contains |1,0> and two sqrt(2) combos of |1,±1>")
        elif has_10 and len(combos) >= 1:
            return (True, "Approximate: contains |1,0> and at least one combo")
        else:
            return (False, f"Eigenbasis content not as expected: {strs}")

    # ----------------------------------------------------------------
    # MONOTONICITY TEST
    # ----------------------------------------------------------------
    def test_monotonicity_comm_sensitive_to_matrix_element(self) -> Tuple[bool, str]:
        """
        Perturb an off-diagonal matrix element of J_x in the j=1 representation
        and verify that [J_x^2, J_y^2] becomes non-zero.
        This shows that the vanishing commutator is sensitive to the exact matrix structure.
        """
        # We need the eigenbasis to exist to proceed (indicates j=1 case is relevant)
        if self.eigenbasis is None:
            return (False, "SKIP: eigenbasis missing, cannot construct j=1 matrices")
        
        epsilon = sp.Symbol('epsilon', real=True)
        hbar = 1  # Use numeric for clarity; the perturbation is on the matrix entry
        
        # Baseline: exact matrices
        Jx_exact = (1/sp.sqrt(2)) * sp.Matrix([[0,1,0],[1,0,1],[0,1,0]])
        Jy = (1/sp.sqrt(2)) * sp.Matrix([[0,-sp.I,0],[sp.I,0,-sp.I],[0,sp.I,0]])
        Jx2_exact = Jx_exact * Jx_exact
        Jy2 = Jy * Jy
        comm_exact = Jx2_exact * Jy2 - Jy2 * Jx2_exact
        # Baseline commutator should be zero
        if not (sp.simplify(comm_exact) == sp.zeros(3,3)):
            return (False, "Baseline commutator is not zero; test cannot proceed")
        
        # Perturb Jx by adding epsilon to entry (0,1) (the first off-diagonal)
        Jx_pert = Jx_exact.copy()
        Jx_pert[0,1] += epsilon
        Jx2_pert = Jx_pert * Jx_pert
        comm_pert = Jx2_pert * Jy2 - Jy2 * Jx2_pert
        
        # The perturbed commutator should be non-zero
        # Compute its Frobenius norm squared at epsilon = 0.1 (non-zero)
        epsilon_val = 0.1
        comm_pert_num = sp.N(comm_pert.subs(epsilon, epsilon_val))
        norm_sq = sum(comm_pert_num[i,j]**2 for i in range(3) for j in range(3))
        
        # Expect a non-zero norm (we just check > 1e-6 to avoid numerical noise)
        if norm_sq < 1e-6:
            return (False, f"Perturbed commutator norm is {norm_sq}, expected > 1e-6")
        return (True, f"Perturbation changed commutator from zero to non-zero (norm^2 = {norm_sq})")

    # ----------------------------------------------------------------
    # ANSWER SANITY (weak)
    # ----------------------------------------------------------------
    def test_answer_sanity_general(self) -> Tuple[bool, str]:
        """
        Very weak sanity: ensure that at least one of the main results is present.
        """
        if self.C_xy is None and self.comm_j0 is None and self.comm_jhalf is None and self.eigenbasis is None:
            return (False, "All results are None; no answer to sanity check")
        return (True, "At least one result present")

    # ----------------------------------------------------------------
    # RUN ALL TESTS
    # ----------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)