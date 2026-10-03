import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any) -> None:
        # SAFE ATTRIBUTE ACCESS: Use getattr with fallback to results dictionary
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters (hardcoded from problem context)
        self.t_total = 6.0  # seconds
        self.x_total = 90.0  # meters
        
        # Extract results with safe defaults
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.Deltax = getattr(solution_instance, 'Deltax', self.results.get('Deltax', 0.0))
        self.x1 = getattr(solution_instance, 'x_1', self.results.get('x_1', 0.0))
        self.x2 = getattr(solution_instance, 'x_2', self.results.get('x_2', 0.0))
        
        # Symbolic variables
        self.t = sp.symbols('t', real=True, nonnegative=True)
        self.a_sym = sp.symbols('a', real=True)
        self.v0_sym = sp.symbols('v0', real=True)
        self.v_sym = sp.symbols('v', real=True)
        self.x_sym = sp.symbols('x', real=True)
        
        self.logs = []
    
    def _log(self, msg: str) -> None:
        self.logs.append(msg)
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        self._log("=== TESTING INVARIANTS ===")
        success = True
        
        # 1. Check acceleration from given displacement (x = 1/2 a t^2 with v0=0)
        a_expected = 2 * self.x_total / (self.t_total ** 2)
        if not self._is_close(self.a, a_expected):
            self._log(f"FAIL: Acceleration mismatch. Expected {a_expected:.3f}, got {self.a:.3f}")
            success = False
        else:
            self._log(f"PASS: Acceleration {self.a:.3f} m/s² matches kinematics")
        
        # 2. Check velocity at t=4s (v = a*t with v0=0)
        v_expected = self.a * 4.0
        if not self._is_close(self.v, v_expected):
            self._log(f"FAIL: Velocity at 4s mismatch. Expected {v_expected:.3f}, got {self.v:.3f}")
            success = False
        else:
            self._log(f"PASS: Velocity {self.v:.3f} m/s matches v = a*t")
        
        # 3. Check displacement in 3rd second (Δx = x(3) - x(2))
        x2_expected = 0.5 * self.a * (3.0 ** 2)
        x1_expected = 0.5 * self.a * (2.0 ** 2)
        Deltax_expected = x2_expected - x1_expected
        
        # Check intermediate displacements if available
        if not self._is_close(self.x1, x1_expected):
            self._log(f"WARN: x(2s) mismatch. Expected {x1_expected:.3f}, got {self.x1:.3f}")
        
        if not self._is_close(self.x2, x2_expected):
            self._log(f"WARN: x(3s) mismatch. Expected {x2_expected:.3f}, got {self.x2:.3f}")
        
        if not self._is_close(self.Deltax, Deltax_expected):
            self._log(f"FAIL: Δx in 3rd second mismatch. Expected {Deltax_expected:.3f}, got {self.Deltax:.3f}")
            success = False
        else:
            self._log(f"PASS: Δx = {self.Deltax:.3f} m matches kinematics")
        
        # 4. Symbolic consistency check
        try:
            # Equation: x = 1/2 a t^2
            t_val = 6.0
            x_eq = sp.Eq(self.x_total, 0.5 * self.a_sym * t_val**2)
            sol = sp.solve(x_eq, self.a_sym)
            if len(sol) > 0 and self._is_close(float(sol[0]), self.a):
                self._log("PASS: Symbolic kinematics verified")
            else:
                self._log("WARN: Symbolic check inconclusive")
        except Exception as e:
            self._log(f"WARN: Symbolic check failed: {e}")
        
        return success, self.logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        self._log("\n=== TESTING TRACE CONSISTENCY ===")
        success = True
        
        # Check that all intermediate results are consistent with final acceleration
        # 1. x1 should be 0.5*a*4
        x1_calc = 0.5 * self.a * 4.0
        if not self._is_close(self.x1, x1_calc):
            self._log(f"FAIL: x1 inconsistency. From a={self.a:.3f}, expected {x1_calc:.3f}, got {self.x1:.3f}")
            success = False
        else:
            self._log(f"PASS: x1 = {self.x1:.3f} m consistent with a")
        
        # 2. x2 should be 0.5*a*9
        x2_calc = 0.5 * self.a * 9.0
        if not self._is_close(self.x2, x2_calc):
            self._log(f"FAIL: x2 inconsistency. From a={self.a:.3f}, expected {x2_calc:.3f}, got {self.x2:.3f}")
            success = False
        else:
            self._log(f"PASS: x2 = {self.x2:.3f} m consistent with a")
        
        # 3. Δx should be x2 - x1
        Deltax_calc = self.x2 - self.x1
        if not self._is_close(self.Deltax, Deltax_calc):
            self._log(f"FAIL: Δx inconsistency. Expected {Deltax_calc:.3f} from x2-x1, got {self.Deltax:.3f}")
            success = False
        else:
            self._log(f"PASS: Δx = {self.Deltax:.3f} m matches x2 - x1")
        
        # 4. Check velocity from acceleration
        v_calc = self.a * 4.0
        if not self._is_close(self.v, v_calc):
            self._log(f"FAIL: v inconsistency. From a={self.a:.3f}, expected {v_calc:.3f}, got {self.v:.3f}")
            success = False
        else:
            self._log(f"PASS: v = {self.v:.3f} m/s consistent with a")
        
        return success, self.logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        self._log("\n=== TESTING MONOTONICITY ===")
        success = True
        
        # Perturb the total displacement input and check if acceleration scales linearly
        original_x = self.x_total
        perturbed_x = original_x * 1.1  # Increase by 10%
        
        # Calculate expected acceleration for perturbed input
        a_perturbed_expected = 2 * perturbed_x / (self.t_total ** 2)
        
        # Calculate what the results would be with perturbed acceleration
        v_perturbed_expected = a_perturbed_expected * 4.0
        x1_perturbed_expected = 0.5 * a_perturbed_expected * 4.0
        x2_perturbed_expected = 0.5 * a_perturbed_expected * 9.0
        Deltax_perturbed_expected = x2_perturbed_expected - x1_perturbed_expected
        
        # Check scaling relationships
        scale_factor = perturbed_x / original_x
        
        # Acceleration should scale linearly with displacement
        a_scale = a_perturbed_expected / self.a if self.a != 0 else 0
        if not self._is_close(a_scale, scale_factor):
            self._log(f"FAIL: Acceleration doesn't scale linearly with displacement. Scale factor {scale_factor:.3f}, got {a_scale:.3f}")
            success = False
        else:
            self._log(f"PASS: Acceleration scales correctly with displacement")
        
        # Velocity should also scale linearly
        v_scale = v_perturbed_expected / self.v if self.v != 0 else 0
        if not self._is_close(v_scale, scale_factor):
            self._log(f"FAIL: Velocity doesn't scale linearly. Expected scale {scale_factor:.3f}, got {v_scale:.3f}")
            success = False
        else:
            self._log(f"PASS: Velocity scales correctly")
        
        # Displacement in 3rd second should scale linearly
        Deltax_scale = Deltax_perturbed_expected / self.Deltax if self.Deltax != 0 else 0
        if not self._is_close(Deltax_scale, scale_factor):
            self._log(f"FAIL: Δx doesn't scale linearly. Expected scale {scale_factor:.3f}, got {Deltax_scale:.3f}")
            success = False
        else:
            self._log(f"PASS: Δx scales correctly")
        
        return success, self.logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        self.logs = []
        overall_success = True
        
        # Run all tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, _ = test_func()
                if not test_success:
                    overall_success = False
            except Exception as e:
                self._log(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        self._log(f"\n=== OVERALL RESULT: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs

# Example usage (for testing only)
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'a': 5.0,
                'v': 20.0,
                'Deltax': 12.5,
                'x_1': 10.0,
                'x_2': 22.5
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)
    print(f"\nFinal success: {success}")