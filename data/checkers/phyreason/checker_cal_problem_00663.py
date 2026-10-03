import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Get constants from results or use defaults
        self.e = self.results.get('e', 1.602e-19)
        self.m = self.results.get('m', 9.109e-31)
        self.E = self.results.get('E', 1.0)
        self.B = self.results.get('B', 1.0)
        self.L = self.results.get('L', 1.0)
        
        # Get target outputs from results
        self.U = self.results.get('U', 0.0)
        self.W = self.results.get('W', 0.0)
        
        # Intermediate quantities (computed from known formulas)
        # Protect against division by zero
        self.v = self.E / self.B if self.B != 0 else 0.0  # from force balance eE = evB
        self.a = self.e * self.E / self.m if self.m != 0 else 0.0  # acceleration from electric field
        self.t = self.L / self.v if self.v != 0 else 0.0  # time of flight
        self.y = 0.5 * self.a * self.t**2 if self.t != 0 else 0.0  # vertical displacement
        
        self.logs = []

    def _log(self, msg: str):
        self.logs.append(msg)

    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self) -> bool:
        success = True
        
        # 1. Force balance: eE = e*v*B => v = E/B
        # Check if e is non-zero to avoid multiplication issues
        if self.e == 0:
            self._log("SKIP: Force balance test (e = 0)")
        else:
            lhs_force = self.e * self.E
            rhs_force = self.e * self.v * self.B
            if not self._is_close(lhs_force, rhs_force):
                self._log(f"FAIL: Force balance invariant: eE = {lhs_force} ≠ evB = {rhs_force}")
                success = False
            else:
                self._log(f"PASS: Force balance invariant holds (v = {self.v:.3e} m/s)")
        
        # 2. Energy conservation: eU = 1/2 * m * v^2
        if self.e == 0 or self.m == 0:
            self._log("SKIP: Energy invariant test (e = 0 or m = 0)")
        else:
            energy_lhs = self.e * self.U
            energy_rhs = 0.5 * self.m * self.v**2
            if not self._is_close(energy_lhs, energy_rhs):
                self._log(f"FAIL: Energy invariant: eU = {energy_lhs:.3e} ≠ 1/2 mv^2 = {energy_rhs:.3e}")
                success = False
            else:
                self._log(f"PASS: Energy invariant holds (U = {self.U:.3e} V)")
        
        # 3. Check derived formula for U: U = m * E^2 / (2 * e * B^2)
        if self.e == 0 or self.B == 0:
            self._log("SKIP: U formula test (e = 0 or B = 0)")
        else:
            U_expected = self.m * self.E**2 / (2 * self.e * self.B**2)
            if not self._is_close(self.U, U_expected):
                self._log(f"FAIL: U mismatch: computed {self.U:.3e} ≠ expected {U_expected:.3e}")
                success = False
            else:
                self._log(f"PASS: U matches theoretical formula")
        
        # 4. Check derived formula for W: W = e^2 * L^2 * B^2 / (2 * m)
        if self.m == 0:
            self._log("SKIP: W formula test (m = 0)")
        else:
            W_expected = self.e**2 * self.L**2 * self.B**2 / (2 * self.m)
            if not self._is_close(self.W, W_expected):
                self._log(f"FAIL: W mismatch: computed {self.W:.3e} ≠ expected {W_expected:.3e}")
                success = False
            else:
                self._log(f"PASS: W matches theoretical formula")
        
        return success

    def test_trace_consistency(self) -> bool:
        success = True
        
        # Trace 1: From force balance and energy, verify v and U are consistent
        if self.e == 0 or self.m == 0 or self.U <= 0:
            self._log("SKIP: v consistency test (e = 0 or m = 0 or U <= 0)")
        else:
            v_from_U = math.sqrt(2 * self.e * self.U / self.m)
            if not self._is_close(self.v, v_from_U):
                self._log(f"FAIL: v inconsistent: from force balance {self.v:.3e} ≠ from U {v_from_U:.3e}")
                success = False
            else:
                self._log(f"PASS: v consistency check passed")
        
        # Trace 2: From kinematics, verify y and W = eE * y
        if self.e == 0 or self.E == 0:
            self._log("SKIP: W from y test (e = 0 or E = 0)")
        else:
            W_from_y = self.e * self.E * self.y
            if not self._is_close(self.W, W_from_y):
                self._log(f"FAIL: W from y mismatch: computed {self.W:.3e} ≠ eEy = {W_from_y:.3e}")
                success = False
            else:
                self._log(f"PASS: W consistent with vertical displacement y = {self.y:.3e} m")
        
        return success

    def test_monotonicity(self) -> bool:
        success = True
        
        # Skip if any critical value is zero
        if self.e == 0 or self.m == 0 or self.B == 0:
            self._log("SKIP: Monotonicity tests (e = 0 or m = 0 or B = 0)")
            return True
        
        # Perturb B by +10% and check trends
        original_B = self.B
        perturbed_B = original_B * 1.10
        
        # Expected changes in U and W (both proportional to B^2)
        U_original = self.m * self.E**2 / (2 * self.e * original_B**2)
        U_perturbed = self.m * self.E**2 / (2 * self.e * perturbed_B**2)
        W_original = self.e**2 * self.L**2 * original_B**2 / (2 * self.m)
        W_perturbed = self.e**2 * self.L**2 * perturbed_B**2 / (2 * self.m)
        
        # U should decrease with increasing B (since U ∝ 1/B^2)
        if U_perturbed >= U_original:
            self._log(f"FAIL: Monotonicity for U: U increases when B increases (U ∝ 1/B^2)")
            success = False
        else:
            self._log(f"PASS: U decreases as B increases (U_original={U_original:.3e}, U_perturbed={U_perturbed:.3e})")
        
        # W should increase with increasing B (since W ∝ B^2)
        if W_perturbed <= W_original:
            self._log(f"FAIL: Monotonicity for W: W does not increase when B increases (W ∝ B^2)")
            success = False
        else:
            self._log(f"PASS: W increases as B increases (W_original={W_original:.3e}, W_perturbed={W_perturbed:.3e})")
        
        # Restore original B
        self.B = original_B
        
        # Also test with perturbed L (W ∝ L^2, U independent of L)
        original_L = self.L
        perturbed_L = original_L * 1.10
        W_original_L = self.e**2 * original_L**2 * self.B**2 / (2 * self.m)
        W_perturbed_L = self.e**2 * perturbed_L**2 * self.B**2 / (2 * self.m)
        if W_perturbed_L <= W_original_L:
            self._log(f"FAIL: Monotonicity for W with L: W does not increase when L increases (W ∝ L^2)")
            success = False
        else:
            self._log(f"PASS: W increases as L increases (W_original={W_original_L:.3e}, W_perturbed={W_perturbed_L:.3e})")
        
        # Restore L
        self.L = original_L
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        self.logs = []
        self._log("=== PhysicsChecker started ===")
        self._log(f"Input values: e={self.e:.3e}, m={self.m:.3e}, E={self.E:.3e}, B={self.B:.3e}, L={self.L:.3e}")
        self._log(f"Output values: U={self.U:.3e}, W={self.W:.3e}")
        
        success_invariants = self.test_invariants()
        success_trace = self.test_trace_consistency()
        success_monotonicity = self.test_monotonicity()
        
        overall_success = success_invariants and success_trace and success_monotonicity
        self._log(f"=== PhysicsChecker finished: {overall_success} ===")
        
        return overall_success, self.logs