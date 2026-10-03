import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        # Input parameters with defaults
        self.m = float(getattr(solution_instance, 'm', self.results.get('m', 1.0)))
        self.g = float(getattr(solution_instance, 'g', self.results.get('g', 9.8)))
        self.R = float(getattr(solution_instance, 'R', self.results.get('R', 1.0)))
        self.q = float(getattr(solution_instance, 'q', self.results.get('q', 1.0)))
        
        # Extract results with safe access
        self.F_N = float(getattr(solution_instance, 'F_N', self.results.get('F_N', 0.0)))
        self.x = float(getattr(solution_instance, 'x', self.results.get('x', 0.0)))
        self.E_k = float(getattr(solution_instance, 'E_k', self.results.get('E_k', 0.0)))
        self.v_0 = float(getattr(solution_instance, 'v_0', self.results.get('v_0', 0.0)))
        self.twoR_minus_y = float(getattr(solution_instance, '2R-y', self.results.get('2R-y', 0.0)))
        self.v_B = float(getattr(solution_instance, 'v_B', self.results.get('v_B', 0.0)))
        self.half_g_t2 = float(getattr(solution_instance, '1/2*g*t2', self.results.get('1/2*g*t2', 0.0)))
        self.E = float(getattr(solution_instance, 'E', self.results.get('E', 0.0)))
        
        # Target answers (numerical)
        self.target_F_N = 5 * self.m * self.g
        self.target_E = (self.m * self.g) / (5 * self.q)
        self.target_E_k = (68/45) * self.m * self.g * self.R
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []

    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def _log(self, msg: str, success: bool):
        """Add log message with status."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {msg}")
        return success

    def test_invariants(self) -> bool:
        """Verify energy conservation, Newton's laws, and work-energy theorem."""
        success = True
        
        # 1. Conservation of Mechanical Energy from A to O (Step 1)
        # Initial: PE = m*g*2R, KE = 0
        # Final: KE = 0.5*m*v_0^2, PE = 0 (at O)
        energy_A = self.m * self.g * 2 * self.R
        energy_O = 0.5 * self.m * self.v_0**2
        if not self._log(f"Energy conservation A->O: {energy_A:.3f} ≈ {energy_O:.3f}", 
                        self._is_close(energy_A, energy_O)):
            success = False
        
        # 2. Newton's Second Law at O (Step 2)
        # F_N - m*g = m*v_0^2/R
        lhs = self.F_N - self.m * self.g
        rhs = self.m * self.v_0**2 / self.R
        if not self._log(f"Newton's 2nd Law at O: {lhs:.3f} ≈ {rhs:.3f}", 
                        self._is_close(lhs, rhs)):
            success = False
        
        # 3. Newton's Second Law at B (Step 4)
        # m*g - q*E = m*v_B^2/R
        lhs2 = self.m * self.g - self.q * self.E
        rhs2 = self.m * self.v_B**2 / self.R
        if not self._log(f"Newton's 2nd Law at B: {lhs2:.3f} ≈ {rhs2:.3f}", 
                        self._is_close(lhs2, rhs2)):
            success = False
        
        # 4. Work-Energy Theorem from A to B (Step 5)
        # Electric field does work over vertical displacement 2R in second quadrant
        work = self.q * self.E * 2 * self.R
        delta_KE = 0.5 * self.m * self.v_B**2  # started from rest at A
        if not self._log(f"Work-Energy A->B: {work:.3f} ≈ {delta_KE:.3f}", 
                        self._is_close(work, delta_KE)):
            success = False
        
        # 5. Work-Energy Theorem from B to impact point (Step 9)
        # Gravity does work: m*g*(2R - y) = E_k - 0.5*m*v_B^2
        gravity_work = self.m * self.g * self.twoR_minus_y
        delta_KE2 = self.E_k - 0.5 * self.m * self.v_B**2
        if not self._log(f"Work-Energy B->impact: {gravity_work:.3f} ≈ {delta_KE2:.3f}", 
                        self._is_close(gravity_work, delta_KE2)):
            success = False
        
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check v_0 from energy conservation (Step 1)
        v0_expected = math.sqrt(4 * self.g * self.R)
        if not self._log(f"v_0 consistency: {self.v_0:.3f} ≈ {v0_expected:.3f}", 
                        self._is_close(self.v_0, v0_expected)):
            success = False
        
        # 2. Check F_N from v_0 (Step 2-3)
        F_N_expected = self.m * self.g + self.m * self.v_0**2 / self.R
        if not self._log(f"F_N from v_0: {self.F_N:.3f} ≈ {F_N_expected:.3f}", 
                        self._is_close(self.F_N, F_N_expected)):
            success = False
        
        # 3. Check E and v_B consistency (Step 4-6)
        # From simultaneous equations: E = m*g/(5*q), v_B^2 = 4*g*R/5
        v_B_sq_expected = 4 * self.g * self.R / 5
        if not self._log(f"v_B^2 consistency: {self.v_B**2:.3f} ≈ {v_B_sq_expected:.3f}", 
                        self._is_close(self.v_B**2, v_B_sq_expected)):
            success = False
        
        # 4. Check projectile motion relation (Step 7-8)
        # On parabola: y = x^2/(2R) => 2R - y = 2R - x^2/(2R)
        # From kinematics: 2R - y = 0.5*g*t^2 and x = v_B*t
        # So: 2R - (v_B^2 * t^2)/(2R) = 0.5*g*t^2
        # Solve for 0.5*g*t^2:
        # Let T = t^2, then: 2R - (v_B^2 * T)/(2R) = 0.5*g*T
        # => Multiply by 2: 4R - (v_B^2 * T)/R = g*T
        # => T = 4R / (g + v_B^2/R)
        T = 4 * self.R / (self.g + self.v_B**2 / self.R)
        half_gt2_expected = 0.5 * self.g * T
        if not self._log(f"Projectile relation: {self.half_g_t2:.3f} ≈ {half_gt2_expected:.3f}", 
                        self._is_close(self.half_g_t2, half_gt2_expected)):
            success = False
        
        # 5. Check E_k from energy (Step 9-10)
        E_k_expected = 0.5 * self.m * self.v_B**2 + self.m * self.g * self.twoR_minus_y
        if not self._log(f"E_k consistency: {self.E_k:.3f} ≈ {E_k_expected:.3f}", 
                        self._is_close(self.E_k, E_k_expected)):
            success = False
        
        return success

    def test_monotonicity(self) -> bool:
        """Perturb mass m and check trend consistency."""
        success = True
        original_m = self.m
        
        # Perturb mass by +10%
        self.m = original_m * 1.10
        
        # Recalculate expected values using formulas (not from solution)
        new_target_F_N = 5 * self.m * self.g
        new_target_E = (self.m * self.g) / (5 * self.q)
        new_target_E_k = (68/45) * self.m * self.g * self.R
        
        # Check monotonicity: all should increase with m
        if not (new_target_F_N > self.target_F_N and 
                new_target_E > self.target_E and 
                new_target_E_k > self.target_E_k):
            self._log("Monotonicity: Expected all outputs to increase with mass", False)
            success = False
        else:
            self._log("Monotonicity: All outputs increase with mass as expected", True)
        
        # Restore original mass
        self.m = original_m
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        success = True
        
        # Test 1: Physics invariants
        if not self.test_invariants():
            success = False
        
        # Test 2: Trace consistency
        if not self.test_trace_consistency():
            success = False
        
        # Test 3: Monotonicity
        if not self.test_monotonicity():
            success = False
        
        # Test 4: Final answer comparison
        if not self._log(f"F_N vs target: {self.F_N:.3f} ≈ {self.target_F_N:.3f}", 
                        self._is_close(self.F_N, self.target_F_N)):
            success = False
        if not self._log(f"E vs target: {self.E:.3f} ≈ {self.target_E:.3f}", 
                        self._is_close(self.E, self.target_E)):
            success = False
        if not self._log(f"E_k vs target: {self.E_k:.3f} ≈ {self.target_E_k:.3f}", 
                        self._is_close(self.E_k, self.target_E_k)):
            success = False
        
        return success, self.logs

# Optional: Example usage
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.m = 0.1
            self.g = 9.8
            self.R = 0.5
            self.q = 1e-6
            self.results = {
                'F_N': 5 * 0.1 * 9.8,
                'v_0': math.sqrt(4 * 9.8 * 0.5),
                'v_B': 2 * math.sqrt(9.8 * 0.5 / 5),
                'E': (0.1 * 9.8) / (5 * 1e-6),
                '2R-y': 10 * 0.5 / 9,
                '1/2*g*t2': 9.8 * 10 * 0.5 / (2 * 9),
                'E_k': (68/45) * 0.1 * 9.8 * 0.5,
                'x': 0.0  # Not critical for checks
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    print(f"Overall success: {success}")
    for log in logs:
        print(log)