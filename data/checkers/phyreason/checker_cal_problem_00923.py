import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Safe access to attributes via getattr with fallback to results dict
        self.I = self._safe_get(solution_instance, 'I')
        self.k = self._safe_get(solution_instance, 'k')
        self.Deltax = self._safe_get(solution_instance, 'Deltax')  # This is Δx' from step 3
        
        # Extract intermediate values if available (for consistency checks)
        self.I1 = self._safe_get(solution_instance, 'I1')  # Current in step 2
        self.I2 = self._safe_get(solution_instance, 'I2')  # Current in step 3
        self.delta_x_original = self._safe_get(solution_instance, 'delta_x')  # Δx from step 2
        
        # Physical constants from problem statement
        self.m = 0.010  # kg (10g)
        self.g = 9.8    # m/s²
        self.B = 0.49   # T
        self.L = 0.50   # m (50cm)
        
        # Target answers from problem
        self.target_I = 0.4      # A
        self.target_k = 24.5     # N/m
        self.target_Deltax = 0.003  # m
        
        self.logs = []
    
    def _safe_get(self, obj, key: str) -> Optional[float]:
        """Safely get attribute from solution instance or its results dict."""
        val = getattr(obj, key, None)
        if val is None and hasattr(obj, 'results'):
            val = obj.results.get(key, None)
        return float(val) if val is not None else None
    
    def _log(self, msg: str, success: bool = True):
        """Add log message with success indicator."""
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
    
    def test_invariants(self) -> bool:
        """Verify physical invariants: force balance and Ampere's law."""
        success = True
        
        # 1. Check step 1: I = mg/(BL)
        if self.I is not None:
            expected_I = (self.m * self.g) / (self.B * self.L)
            if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 1 invariant failed: I = {self.I:.3f}, expected mg/(BL) = {expected_I:.3f}", False)
                success = False
            else:
                self._log(f"Step 1 invariant passed: I = mg/(BL) = {expected_I:.3f} A")
        
        # 2. Check step 2: k = (mg - B*I1*L)/(2*Δx) if we have I1 and Δx
        if self.k is not None and self.I1 is not None and self.delta_x_original is not None:
            numerator = self.m * self.g - self.B * self.I1 * self.L
            denominator = 2 * self.delta_x_original
            if denominator != 0:
                expected_k = numerator / denominator
                if not math.isclose(self.k, expected_k, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Step 2 invariant failed: k = {self.k:.3f}, expected (mg - B*I1*L)/(2Δx) = {expected_k:.3f}", False)
                    success = False
                else:
                    self._log(f"Step 2 invariant passed: k = (mg - B*I1*L)/(2Δx)")
            else:
                self._log("Warning: denominator zero in step 2 check", False)
        
        # 3. Check step 3: Δx' = (mg + B*I2*L)/(2k) if we have I2 and k
        if self.Deltax is not None and self.I2 is not None and self.k is not None and self.k != 0:
            expected_Deltax = (self.m * self.g + self.B * self.I2 * self.L) / (2 * self.k)
            if not math.isclose(self.Deltax, expected_Deltax, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 3 invariant failed: Δx' = {self.Deltax:.5f}, expected (mg + B*I2*L)/(2k) = {expected_Deltax:.5f}", False)
                success = False
            else:
                self._log(f"Step 3 invariant passed: Δx' = (mg + B*I2*L)/(2k)")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check against target answers
        if self.I is not None:
            if not math.isclose(self.I, self.target_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 1 result mismatch: I = {self.I:.3f}, target = {self.target_I}", False)
                success = False
            else:
                self._log(f"Step 1 matches target: {self.I:.1f} A")
        
        if self.k is not None:
            if not math.isclose(self.k, self.target_k, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 2 result mismatch: k = {self.k:.3f}, target = {self.target_k}", False)
                success = False
            else:
                self._log(f"Step 2 matches target: {self.k:.1f} N/m")
        
        if self.Deltax is not None:
            if not math.isclose(self.Deltax, self.target_Deltax, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 3 result mismatch: Δx' = {self.Deltax:.5f}, target = {self.target_Deltax}", False)
                success = False
            else:
                self._log(f"Step 3 matches target: {self.Deltax:.3f} m")
        
        # Check that all results are positive (physical constraint)
        if self.I is not None and self.I < 0:
            self._log(f"Current I should be positive, got {self.I}", False)
            success = False
        
        if self.k is not None and self.k < 0:
            self._log(f"Spring constant k should be positive, got {self.k}", False)
            success = False
        
        if self.Deltax is not None and self.Deltax < 0:
            self._log(f"Elongation Δx' should be positive, got {self.Deltax}", False)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb mass and check physical trends."""
        success = True
        
        # Test 1: Increase mass should increase I in step 1
        original_m = self.m
        perturbed_m = original_m * 1.1  # 10% increase
        
        # Step 1: I = mg/(BL)
        original_I = (original_m * self.g) / (self.B * self.L)
        perturbed_I = (perturbed_m * self.g) / (self.B * self.L)
        
        if perturbed_I <= original_I:
            self._log(f"Monotonicity failed: I should increase with mass", False)
            success = False
        else:
            self._log(f"Monotonicity passed: I increases with mass (ΔI = {perturbed_I - original_I:.4f} A)")
        
        # Test 2: For fixed k and I2, increase mass should increase Δx'
        if self.k is not None and self.I2 is not None:
            original_Deltax = (original_m * self.g + self.B * self.I2 * self.L) / (2 * self.k)
            perturbed_Deltax = (perturbed_m * self.g + self.B * self.I2 * self.L) / (2 * self.k)
            
            if perturbed_Deltax <= original_Deltax:
                self._log(f"Monotonicity failed: Δx' should increase with mass", False)
                success = False
            else:
                self._log(f"Monotonicity passed: Δx' increases with mass (ΔΔx' = {perturbed_Deltax - original_Deltax:.6f} m)")
        
        # Restore original mass
        self.m = original_m
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success flag with logs."""
        self.logs = []
        
        self._log("=" * 50)
        self._log("PhysicsChecker Audit Report")
        self._log("=" * 50)
        
        # Run tests
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity),
        ]
        
        all_success = True
        for test_name, test_func in tests:
            self._log(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    all_success = False
            except Exception as e:
                self._log(f"Test failed with error: {str(e)}", False)
                all_success = False
        
        self._log("\n" + "=" * 50)
        self._log(f"Overall Result: {'PASS' if all_success else 'FAIL'}")
        self._log("=" * 50)
        
        return all_success, self.logs

# Example usage:
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'I': 0.4,
                'k': 24.5,
                'Deltax': 0.003,
                'I1': 0.2,  # Assuming I1 for step 2
                'I2': 0.6,  # Assuming I2 for step 3
                'delta_x': 0.001  # Assuming Δx for step 2
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)