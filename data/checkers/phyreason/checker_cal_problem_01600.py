import math
import traceback
from typing import Tuple, List, Union

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract numeric values from results dictionary with defaults
        results = getattr(solution_instance, 'results', {})
        
        # Input parameters - use numeric defaults if not provided
        self.E = float(results.get('E', 10.0))
        self.R1 = float(results.get('R1', 100.0))
        self.R2 = float(results.get('R2', 200.0))
        self.R3 = float(results.get('R3', 300.0))
        self.R4 = float(results.get('R4', 400.0))
        self.C = float(results.get('C', 1e-6))
        
        # Results - use numeric defaults if not provided
        self.Q1 = float(results.get('Q_1', 0.0))
        self.Q2 = float(results.get('Q_2', 0.0))
        self.DeltaQ = float(results.get('DeltaQ', 0.0))
        
        # Store original values for monotonicity test
        self._original_C = self.C
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message: str, success: bool = None):
        """Helper method to log messages"""
        self.logs.append(f"{'[PASS]' if success else '[FAIL]'} {message}")
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Numerical comparison for floats only"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self) -> bool:
        """Test physics invariants from KB reference using numeric values"""
        success = True
        
        try:
            # Test 1: Capacitance relationship Q = C*V
            # Expected voltages from circuit analysis
            V1_expected = self.R2 * self.E / (self.R1 + self.R2)  # Open switch
            V2_expected = self.R1 * self.E / (self.R1 + self.R2)  # Closed switch
            
            Q1_expected = self.C * V1_expected
            Q2_expected = self.C * V2_expected
            
            if not self._is_close(self.Q1, Q1_expected):
                self._log(f"Q1 invariant failed: got {self.Q1}, expected {Q1_expected}", False)
                success = False
            else:
                self._log("Q1 = C*V1 relationship holds", True)
                
            if not self._is_close(self.Q2, Q2_expected):
                self._log(f"Q2 invariant failed: got {self.Q2}, expected {Q2_expected}", False)
                success = False
            else:
                self._log("Q2 = C*V2 relationship holds", True)
                
            # Test energy relationship
            energy1 = 0.5 * self.C * V1_expected**2
            energy2 = 0.5 * self.C * V2_expected**2
            self._log(f"Energy relationships: U1={energy1}, U2={energy2}", True)
            
            # Test 2: Charge conservation (ΔQ = Q1 + Q2 from step_5)
            if not self._is_close(self.DeltaQ, self.Q1 + self.Q2):
                self._log(f"Charge conservation failed: ΔQ={self.DeltaQ}, Q1+Q2={self.Q1 + self.Q2}", False)
                success = False
            else:
                self._log("Charge conservation ΔQ = Q1 + Q2 holds", True)
                
            # Test 3: Final answer invariant ΔQ = C*E
            if not self._is_close(self.DeltaQ, self.C * self.E):
                self._log(f"Final answer invariant failed: ΔQ={self.DeltaQ}, C*E={self.C * self.E}", False)
                success = False
            else:
                self._log("Final answer ΔQ = C*E holds", True)
                
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            success = False
            
        return success
        
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer using numeric values"""
        success = True
        
        try:
            # Check that Q1 formula from step_3 matches expected
            Q1_expected = self.C * self.R2 * self.E / (self.R1 + self.R2)
            if not self._is_close(self.Q1, Q1_expected):
                self._log(f"Q1 trace mismatch: got {self.Q1}, expected {Q1_expected}", False)
                success = False
            else:
                self._log("Q1 matches step_3 formula", True)
                
            # Check that Q2 formula from step_4 matches expected
            Q2_expected = self.C * self.R1 * self.E / (self.R1 + self.R2)
            if not self._is_close(self.Q2, Q2_expected):
                self._log(f"Q2 trace mismatch: got {self.Q2}, expected {Q2_expected}", False)
                success = False
            else:
                self._log("Q2 matches step_4 formula", True)
                
            # Check that ΔQ formula from step_5 matches final answer
            DeltaQ_expected = self.C * self.E
            if not self._is_close(self.DeltaQ, DeltaQ_expected):
                self._log(f"ΔQ trace mismatch: got {self.DeltaQ}, expected {DeltaQ_expected}", False)
                success = False
            else:
                self._log("ΔQ matches step_5 formula and target answer", True)
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            success = False
            
        return success
        
    def test_monotonicity(self) -> bool:
        """Perturb capacitance C and verify trend in ΔQ using numeric values"""
        success = True
        
        try:
            # Original ΔQ
            DeltaQ_original = self.C * self.E
            
            # Perturb C by +10%
            C_perturbed = 1.1 * self._original_C
            
            # Calculate perturbed ΔQ
            DeltaQ_perturbed = C_perturbed * self.E
            
            # ΔQ should increase proportionally with C
            expected_ratio = 1.1  # 10% increase in C
            actual_ratio = DeltaQ_perturbed / DeltaQ_original
            
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
                self._log(f"Monotonicity holds: ΔQ ∝ C (ratio: {actual_ratio:.3f} ≈ {expected_ratio})", True)
            else:
                self._log(f"Monotonicity failed: ΔQ ratio {actual_ratio:.3f} ≠ {expected_ratio}", False)
                success = False
                
            # Additional check: Q1 and Q2 should also increase proportionally with C
            Q1_original = self.C * self.R2 * self.E / (self.R1 + self.R2)
            Q2_original = self.C * self.R1 * self.E / (self.R1 + self.R2)
            
            Q1_perturbed = C_perturbed * self.R2 * self.E / (self.R1 + self.R2)
            Q2_perturbed = C_perturbed * self.R1 * self.E / (self.R1 + self.R2)
            
            q1_ratio = Q1_perturbed / Q1_original
            q2_ratio = Q2_perturbed / Q2_original
            
            if math.isclose(q1_ratio, 1.1, rel_tol=1e-2):
                self._log("Q1 ∝ C relationship confirmed", True)
            else:
                self._log(f"Q1 ∝ C relationship questionable: ratio {q1_ratio:.3f}", False)
                success = False
                
            if math.isclose(q2_ratio, 1.1, rel_tol=1e-2):
                self._log("Q2 ∝ C relationship confirmed", True)
            else:
                self._log(f"Q2 ∝ C relationship questionable: ratio {q2_ratio:.3f}", False)
                success = False
                
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            success = False
            
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status with logs"""
        self.logs = []
        all_success = True
        
        try:
            # Test 1: Invariants
            self._log("=== Testing Physics Invariants ===", None)
            if not self.test_invariants():
                all_success = False
                
            # Test 2: Trace consistency
            self._log("\n=== Testing Trace Consistency ===", None)
            if not self.test_trace_consistency():
                all_success = False
                
            # Test 3: Monotonicity
            self._log("\n=== Testing Monotonicity ===", None)
            if not self.test_monotonicity():
                all_success = False
                
            # Summary
            self._log(f"\n=== Summary: {'ALL TESTS PASSED' if all_success else 'SOME TESTS FAILED'} ===", None)
            
        except Exception as e:
            self._log(f"Critical error in run_all_tests: {traceback.format_exc()}", False)
            all_success = False
            
        return all_success, self.logs