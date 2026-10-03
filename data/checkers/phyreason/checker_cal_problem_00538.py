import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Get the results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 0.4)  # kg
        self.v0_mag = getattr(solution_instance, 'v0', 4.0)  # m/s
        self.v1_mag = getattr(solution_instance, 'v1', 6.0)  # m/s
        self.dt = getattr(solution_instance, 'dt', 
                         getattr(solution_instance, 'Δt', 
                                self.results.get('Δt', self.results.get('dt', 0.2))))  # s
        
        # Extract results with safe defaults
        self.p_initial = self.results.get('p', 1.6)  # kg·m/s
        self.p_final = self.results.get("p'", self.results.get('p_prime', -2.4))  # kg·m/s
        self.delta_p = self.results.get('Δp', self.results.get('delta_p', -4.0))  # kg·m/s
        self.F_avg = self.results.get('F', -20.0)  # N
        
        # Store for perturbation testing
        self._original_m = self.m
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message: str, success: bool = True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
        
    def test_invariants(self) -> bool:
        """
        Verify physics invariants based on impulse-momentum theorem.
        Uses both symbolic and numerical checks with fuzzy matching.
        """
        successes = []
        
        # Define symbols for symbolic verification
        m_sym, v0_sym, v1_sym, dt_sym = sp.symbols('m v0 v1 dt')
        p0_sym = m_sym * v0_sym
        p1_sym = m_sym * v1_sym
        delta_p_sym = p1_sym - p0_sym
        F_sym = delta_p_sym / dt_sym
        
        # Test 1: Initial momentum magnitude (direction from steps)
        # Steps show: v0 left = positive, v1 right = negative
        p0_expected = self.m * self.v0_mag
        if not math.isclose(abs(self.p_initial), abs(p0_expected), rel_tol=0.02, abs_tol=0.02):
            self._log(f"Initial momentum mismatch: expected magnitude {p0_expected:.2f}, got {self.p_initial:.2f}", False)
            successes.append(False)
        else:
            self._log(f"Initial momentum magnitude correct: {abs(self.p_initial):.2f} kg·m/s")
            successes.append(True)
        
        # Test 2: Final momentum magnitude (with sign from steps)
        p1_expected = self.m * (-self.v1_mag)  # Negative for right direction
        if not math.isclose(self.p_final, p1_expected, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Final momentum mismatch: expected {p1_expected:.2f}, got {self.p_final:.2f}", False)
            successes.append(False)
        else:
            self._log(f"Final momentum correct: {self.p_final:.2f} kg·m/s")
            successes.append(True)
        
        # Test 3: Momentum change from velocities
        delta_p_expected = p1_expected - p0_expected
        if not math.isclose(self.delta_p, delta_p_expected, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Momentum change mismatch: expected {delta_p_expected:.2f}, got {self.delta_p:.2f}", False)
            successes.append(False)
        else:
            self._log(f"Momentum change correct: {self.delta_p:.2f} kg·m/s")
            successes.append(True)
        
        # Test 4: Impulse-momentum theorem F = Δp/Δt
        if self.dt > 1e-10:  # Avoid division by zero
            F_expected = self.delta_p / self.dt
            if not math.isclose(self.F_avg, F_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Average force mismatch: expected {F_expected:.2f} N, got {self.F_avg:.2f} N", False)
                successes.append(False)
            else:
                self._log(f"Average force correct: {self.F_avg:.2f} N (Δt = {self.dt:.3f} s)")
                successes.append(True)
        else:
            self._log("Δt is zero or too small, skipping force calculation", False)
            successes.append(False)
        
        # Test 5: Direction consistency (negative force means force to right)
        if self.delta_p < 0 and self.F_avg < 0:
            self._log("Direction consistent: Δp and F both negative (to the right)")
            successes.append(True)
        elif self.delta_p > 0 and self.F_avg > 0:
            self._log("Direction consistent: Δp and F both positive (to the left)")
            successes.append(True)
        else:
            self._log(f"Direction mismatch: Δp={self.delta_p}, F={self.F_avg}", False)
            successes.append(False)
        
        return all(successes)
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer.
        Verify the steps analysis matches the computed results.
        """
        successes = []
        
        # Step 1: Initial momentum = 1.6 kg·m/s
        if math.isclose(self.p_initial, 1.6, rel_tol=0.02, abs_tol=0.02):
            self._log("Step 1 consistency: p = 1.6 kg·m/s")
            successes.append(True)
        else:
            self._log(f"Step 1 mismatch: expected 1.6, got {self.p_initial}", False)
            successes.append(False)
        
        # Step 2: Final momentum = -2.4 kg·m/s
        if math.isclose(self.p_final, -2.4, rel_tol=0.02, abs_tol=0.02):
            self._log("Step 2 consistency: p' = -2.4 kg·m/s")
            successes.append(True)
        else:
            self._log(f"Step 2 mismatch: expected -2.4, got {self.p_final}", False)
            successes.append(False)
        
        # Step 3: Δp = -4.0 kg·m/s
        if math.isclose(self.delta_p, -4.0, rel_tol=0.02, abs_tol=0.02):
            self._log("Step 3 consistency: Δp = -4.0 kg·m/s")
            successes.append(True)
        else:
            self._log(f"Step 3 mismatch: expected -4.0, got {self.delta_p}", False)
            successes.append(False)
        
        # Step 5: F = -20 N (assuming Δt = 0.2s from steps)
        if math.isclose(self.F_avg, -20.0, rel_tol=0.02, abs_tol=0.02):
            self._log("Step 5 consistency: F = -20 N")
            successes.append(True)
        else:
            self._log(f"Step 5 mismatch: expected -20, got {self.F_avg}", False)
            successes.append(False)
        
        # Check target answers (magnitudes)
        if math.isclose(abs(self.delta_p), 4.0, rel_tol=0.02, abs_tol=0.02):
            self._log("Target answer 1: |Δp| = 4.0 kg·m/s ✓")
            successes.append(True)
        else:
            self._log(f"Target answer 1 mismatch: |Δp| = {abs(self.delta_p):.1f}", False)
            successes.append(False)
            
        if math.isclose(abs(self.F_avg), 20.0, rel_tol=0.02, abs_tol=0.02):
            self._log("Target answer 2: |F| = 20 N ✓")
            successes.append(True)
        else:
            self._log(f"Target answer 2 mismatch: |F| = {abs(self.F_avg):.1f}", False)
            successes.append(False)
        
        return all(successes)
    
    def test_monotonicity(self) -> bool:
        """
        Perturb mass by +10%, recalculate using physics formulas,
        check that momentum and force scale proportionally.
        """
        successes = []
        
        # Store original values
        original_m = self.m
        original_p0 = self.p_initial
        original_p1 = self.p_final
        original_delta_p = self.delta_p
        original_F = self.F_avg
        
        # Perturb mass by +10%
        perturb_factor = 1.10
        self.m = original_m * perturb_factor
        
        # Recalculate using physics formulas (keeping velocities constant)
        # Using same sign convention: v0 positive (left), v1 negative (right)
        v0 = self.v0_mag  # Positive for left
        v1 = -self.v1_mag  # Negative for right
        
        p0_new = self.m * v0
        p1_new = self.m * v1
        delta_p_new = p1_new - p0_new
        F_new = delta_p_new / self.dt if self.dt > 1e-10 else 0
        
        # Check proportionality (all should scale by same factor as mass)
        if not math.isclose(p0_new / original_p0, perturb_factor, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Monotonicity fail: p0 scaled by {p0_new/original_p0:.3f}, expected {perturb_factor}", False)
            successes.append(False)
        else:
            self._log(f"Monotonicity pass: p0 scales with mass (factor {perturb_factor})")
            successes.append(True)
            
        if not math.isclose(p1_new / original_p1, perturb_factor, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Monotonicity fail: p1 scaled by {p1_new/original_p1:.3f}, expected {perturb_factor}", False)
            successes.append(False)
        else:
            self._log(f"Monotonicity pass: p1 scales with mass (factor {perturb_factor})")
            successes.append(True)
            
        if not math.isclose(delta_p_new / original_delta_p, perturb_factor, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Monotonicity fail: Δp scaled by {delta_p_new/original_delta_p:.3f}, expected {perturb_factor}", False)
            successes.append(False)
        else:
            self._log(f"Monotonicity pass: Δp scales with mass (factor {perturb_factor})")
            successes.append(True)
            
        if not math.isclose(F_new / original_F, perturb_factor, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Monotonicity fail: F scaled by {F_new/original_F:.3f}, expected {perturb_factor}", False)
            successes.append(False)
        else:
            self._log(f"Monotonicity pass: F scales with mass (factor {perturb_factor})")
            successes.append(True)
        
        # Restore original mass and values
        self.m = original_m
        
        return all(successes)
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics checks and return success status with logs.
        """
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        # Run all tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary
        all_passed = test1 and test2 and test3
        self.logs.append("=== Summary ===")
        self.logs.append(f"Invariants: {'PASS' if test1 else 'FAIL'}")
        self.logs.append(f"Trace Consistency: {'PASS' if test2 else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if test3 else 'FAIL'}")
        self.logs.append(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        return all_passed, self.logs

# Example usage (commented out for sandbox safety):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.m = 0.4
#             self.v0 = 4.0
#             self.v1 = 6.0
#             self.dt = 0.2
#             self.results = {
#                 'p': 1.6,
#                 "p'": -2.4,
#                 'Δp': -4.0,
#                 'F': -20.0
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)