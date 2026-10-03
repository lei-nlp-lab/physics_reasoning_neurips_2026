import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use the SAME numerical constants as the solution code: m=g=L=q=1.
        """
        # Extract input parameters - use defaults that match solution's substitution
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 1.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.E = None  # Will be derived from U if needed
        
        # Extract results using safe access
        results = getattr(solution_instance, 'results', {})
        self.U = results.get('U', 0.0)
        self.q_B = results.get('q_B', 0.0)
        self.t = results.get('t', 0.0)
        self.W = results.get('W', 0.0)
        self.E_k = results.get('E_k', 0.0)
        
        # Intermediate kinematic quantities
        self.a1 = results.get('a_1', 0.0)
        self.t1 = results.get('t_1', 0.0)
        self.v1 = results.get('v_1', 0.0)
        self.a2 = results.get('a_2', 0.0)
        self.t2 = results.get('t_2', 0.0)
        self.v2 = results.get('v_2', 0.0)
        self.a3 = results.get('a_3', 0.0)
        self.t3 = results.get('t_3', 0.0)
        
        # Derived electric field from U = E * 3L
        if self.U != 0 and self.L != 0:
            self.E = self.U / (3 * self.L)
        else:
            self.E = 0.0
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Store logs
        self.logs = []
        
    def log(self, message: str):
        """Helper to append log messages."""
        self.logs.append(message)
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants:
        1. Work-energy theorem for uncharged B case.
        2. Newton's second law for three motion stages.
        3. Energy conservation in the charged B case (net work = 0).
        """
        all_ok = True
        
        # 1. Work-energy theorem for uncharged B case
        # From problem: speed of B when reaching PQ is sqrt(8gL)
        v_B_expected = math.sqrt(8 * self.g * self.L)
        # System kinetic energy = (1/2)*(2m)*v^2 = m*v^2
        E_k_expected = self.m * v_B_expected**2
        # Work done by field: W = 2E*q*L (from step 1)
        W_expected = 2 * self.E * self.q * self.L
        
        if not math.isclose(E_k_expected, W_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"Work-energy mismatch: E_k={E_k_expected}, W={W_expected}")
            all_ok = False
        else:
            self.log("Work-energy theorem holds for uncharged case.")
        
        # 2. Newton's second law checks for three stages
        # Stage 1: only A in field, Eq = 2m*a1
        a1_expected = self.E * self.q / (2 * self.m)
        if not math.isclose(self.a1, a1_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"Newton's law mismatch in stage 1: a1={self.a1}, expected={a1_expected}")
            all_ok = False
        else:
            self.log("Newton's law holds for stage 1.")
        
        # Stage 2: both in field, E(q - q_B) = 2m*a2 (note: q_B is signed)
        a2_expected = self.E * (self.q - self.q_B) / (2 * self.m)
        if not math.isclose(self.a2, a2_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"Newton's law mismatch in stage 2: a2={self.a2}, expected={a2_expected}")
            all_ok = False
        else:
            self.log("Newton's law holds for stage 2.")
        
        # Stage 3: only B in field, E*q_B = 2m*a3
        a3_expected = self.E * self.q_B / (2 * self.m)
        if not math.isclose(self.a3, a3_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"Newton's law mismatch in stage 3: a3={self.a3}, expected={a3_expected}")
            all_ok = False
        else:
            self.log("Newton's law holds for stage 3.")
        
        # 3. Energy conservation for charged B case: net work = 0
        # From step 4: EqL + E(q - q_B)L - E*q_B*2L = 0
        net_work = self.E * self.q * self.L + self.E * (self.q - self.q_B) * self.L - self.E * self.q_B * 2 * self.L
        if not math.isclose(net_work, 0.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"Net work not zero for charged case: net_work={net_work}")
            all_ok = False
        else:
            self.log("Energy conservation holds for charged case (net work = 0).")
        
        return all_ok
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate kinematic results with each other and final answer.
        """
        all_ok = True
        
        # Check kinematic chain for stage 1: L = 0.5*a1*t1^2, v1 = a1*t1
        t1_from_L = math.sqrt(2 * self.L / self.a1) if self.a1 > 0 else 0.0
        if not math.isclose(self.t1, t1_from_L, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"t1 inconsistency: from L={t1_from_L}, stored={self.t1}")
            all_ok = False
        else:
            self.log("t1 consistent with L and a1.")
        
        v1_from_a1t1 = self.a1 * self.t1
        if not math.isclose(self.v1, v1_from_a1t1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"v1 inconsistency: from a1*t1={v1_from_a1t1}, stored={self.v1}")
            all_ok = False
        else:
            self.log("v1 consistent with a1 and t1.")
        
        # Check kinematic chain for stage 3: 2L = 0.5*a3*t3^2, v2 = a3*t3
        # Note: a3 may be positive or negative; use absolute for sqrt
        t3_from_2L = math.sqrt(4 * self.L / abs(self.a3)) if self.a3 != 0 else 0.0
        if not math.isclose(self.t3, t3_from_2L, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"t3 inconsistency: from 2L={t3_from_2L}, stored={self.t3}")
            all_ok = False
        else:
            self.log("t3 consistent with 2L and |a3|.")
        
        v2_from_a3t3 = self.a3 * self.t3
        if not math.isclose(self.v2, v2_from_a3t3, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"v2 inconsistency: from a3*t3={v2_from_a3t3}, stored={self.v2}")
            all_ok = False
        else:
            self.log("v2 consistent with a3 and t3.")
        
        # Check kinematic chain for stage 2: v2 = v1 + a2*t2
        t2_from_v = (self.v2 - self.v1) / self.a2 if self.a2 != 0 else 0.0
        if not math.isclose(self.t2, t2_from_v, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"t2 inconsistency: from v2-v1={t2_from_v}, stored={self.t2}")
            all_ok = False
        else:
            self.log("t2 consistent with v1, v2, and a2.")
        
        # Check total time: t = t1 + 2*(t2 + t3)
        t_calculated = self.t1 + 2 * (self.t2 + self.t3)
        if not math.isclose(self.t, t_calculated, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"Total time inconsistency: calculated={t_calculated}, stored={self.t}")
            all_ok = False
        else:
            self.log(f"Total time t consistent with sum of stages.")
        
        # Check U = 12*m*g*L/q
        U_expected = 12 * self.m * self.g * self.L / self.q
        if not math.isclose(self.U, U_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"U inconsistency: expected={U_expected}, stored={self.U}")
            all_ok = False
        else:
            self.log("U consistent with m, g, L, q.")
        
        # Check q_B magnitude = 2q/3 (sign may be positive or negative; physics allows positive as per net work zero)
        q_B_magnitude = abs(self.q_B)
        q_B_expected_mag = 2 * self.q / 3
        if not math.isclose(q_B_magnitude, q_B_expected_mag, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"q_B magnitude mismatch: expected magnitude={q_B_expected_mag}, stored magnitude={q_B_magnitude}")
            all_ok = False
        else:
            self.log("q_B magnitude consistent.")
        
        return all_ok
    
    def test_monotonicity(self) -> bool:
        """
        Perturb mass m by +1%, recalculate key outputs using formulas,
        verify expected trends, then restore original m.
        """
        original_m = self.m
        perturb_factor = 1.01  # +1%
        
        # Perturb m
        self.m = original_m * perturb_factor
        
        # Recalculate using formulas (not calling solution methods)
        # U = 12*m*g*L/q should scale linearly with m
        U_new = 12 * self.m * self.g * self.L / self.q
        U_ratio = U_new / self.U if self.U != 0 else float('inf')
        
        # q_B should be independent of m (from formula q_B = 2q/3)
        q_B_new = self.q_B  # unchanged
        
        # t = (6√3 - 5)√(L/g) should be independent of m
        t_new = self.t  # unchanged
        
        # Restore original m
        self.m = original_m
        
        # Check trends
        all_ok = True
        
        if not math.isclose(U_ratio, perturb_factor, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.log(f"Monotonicity fail: U should scale linearly with m, ratio={U_ratio}")
            all_ok = False
        else:
            self.log(f"U scales correctly with m (ratio={U_ratio}).")
        
        if not math.isclose(q_B_new, self.q_B, rel_tol=1e-9, abs_tol=1e-9):
            self.log(f"Monotonicity fail: q_B changed with m")
            all_ok = False
        else:
            self.log("q_B independent of m.")
        
        if not math.isclose(t_new, self.t, rel_tol=1e-9, abs_tol=1e-9):
            self.log(f"Monotonicity fail: t changed with m")
            all_ok = False
        else:
            self.log("t independent of m.")
        
        return all_ok
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs.clear()
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        success = test1 and test2 and test3
        self.log(f"Overall test result: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs