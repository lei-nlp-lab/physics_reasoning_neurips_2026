import sympy as sp
import math
import traceback
from typing import Dict, Any, List, Tuple


class PhysicsChecker:
    """Physics auditor for two-block friction problem."""
    
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution."""
        # Problem constants (with safe defaults)
        self.m = getattr(solution_instance, 'm', 4.0)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Results dictionary with safe access
        results = getattr(solution_instance, 'results', {})
        
        # Extract all result values safely
        self.F_max = results.get('F_max', 0.0)
        self.f_2 = results.get('f_2', 0.0)
        self.N_A = results.get('N/A', 0.0)  # Friction force on A by ground
        self.F_2 = results.get('F_2', 0.0)
        self.N_AB = results.get('N_AB', 0.0)  # Elastic force A→B
        self.f = results.get('f', 0.0)  # Kinetic friction per block
        
        # Store logs
        self.logs = []
        
        # Known applied forces from problem
        self.F1_applied = 14.0  # Force on B in first scenario
        self.F2_applied = 16.0  # Force on A in second scenario
        
    def _log(self, msg: str, success: bool = None):
        """Add message to logs."""
        if success is not None:
            status = "✓" if success else "✗"
            self.logs.append(f"{status} {msg}")
        else:
            self.logs.append(f"  {msg}")
    
    def _isclose(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison for physics quantities."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics laws."""
        success = True
        
        # 1. Maximum static friction formula
        expected_F_max = 2 * self.mu * self.m * self.g  # μ * (2mg)
        if not self._isclose(self.F_max, expected_F_max):
            self._log(f"F_max mismatch: {self.F_max} vs {expected_F_max}", False)
            success = False
        else:
            self._log(f"F_max correct: μ×2mg = {expected_F_max}", True)
        
        # 2. First scenario: Newton's 1st Law (F1=14N < F_max, system at rest)
        #    f_2 (friction on B from ground) must balance applied force
        if not self._isclose(self.f_2, self.F1_applied):
            self._log(f"First scenario: f_2 should equal F1={self.F1_applied}, got {self.f_2}", False)
            success = False
        else:
            self._log(f"f_2 correctly balances F1={self.F1_applied}", True)
        
        # 3. First scenario: Block A experiences no friction (at rest, no tendency to move)
        if not self._isclose(self.N_A, 0.0):
            self._log(f"First scenario: N_A should be 0, got {self.N_A}", False)
            success = False
        else:
            self._log("N_A correctly zero (no friction on A)", True)
        
        # 4. Second scenario: Kinetic friction per block
        expected_f = self.mu * self.m * self.g  # μmg
        if not self._isclose(self.f, expected_f):
            self._log(f"Kinetic friction per block mismatch: {self.f} vs μmg={expected_f}", False)
            success = False
        else:
            self._log(f"Kinetic friction correct: μmg = {expected_f}", True)
        
        # 5. Second scenario: F_2 balances total friction (both blocks moving at constant velocity)
        expected_F_2 = 2 * self.mu * self.m * self.g  # 2μmg
        if not self._isclose(self.F_2, expected_F_2):
            self._log(f"F_2 mismatch: {self.F_2} vs 2μmg={expected_F_2}", False)
            success = False
        else:
            self._log(f"F_2 correct: 2μmg = {expected_F_2}", True)
        
        # 6. Second scenario: Elastic force N_AB equals friction on B
        if not self._isclose(self.N_AB, self.mu * self.m * self.g):
            self._log(f"N_AB mismatch: {self.N_AB} vs μmg={expected_f}", False)
            success = False
        else:
            self._log(f"N_AB correct: equals friction on B = {expected_f}", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check intermediate steps align with final answers."""
        success = True
        
        # Cross-check relationships from STEPS ANALYSIS
        # Step 2: f_2 = F1_applied (when at rest)
        if not self._isclose(self.f_2, self.F1_applied):
            self._log(f"Step 2 inconsistency: f_2={self.f_2} ≠ F1={self.F1_applied}", False)
            success = False
        
        # Step 6: F_2 = 2μmg
        expected_F2 = 2 * self.mu * self.m * self.g
        if not self._isclose(self.F_2, expected_F2):
            self._log(f"Step 6 inconsistency: F_2={self.F_2} ≠ 2μmg={expected_F2}", False)
            success = False
        
        # Step 8: N_AB = μmg
        expected_N_AB = self.mu * self.m * self.g
        if not self._isclose(self.N_AB, expected_N_AB):
            self._log(f"Step 8 inconsistency: N_AB={self.N_AB} ≠ μmg={expected_N_AB}", False)
            success = False
        
        # Verify both target answers
        answer1_correct = (self._isclose(self.f_2, 14.0) and self._isclose(self.N_A, 0.0))
        answer2_correct = (self._isclose(self.F_2, 16.0) and self._isclose(self.N_AB, 8.0))
        
        if not answer1_correct:
            self._log(f"First answer mismatch: expected '14N, 0', got '{self.f_2}N, {self.N_A}'", False)
            success = False
        else:
            self._log("First answer correct: 14N, 0", True)
        
        if not answer2_correct:
            self._log(f"Second answer mismatch: expected '16N, 8N', got '{self.F_2}N, {self.N_AB}N'", False)
            success = False
        else:
            self._log("Second answer correct: 16N, 8N", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test that forces scale correctly with mass changes."""
        success = True
        
        # Save original values
        original_m = self.m
        original_f_2 = self.f_2
        original_F_2 = self.F_2
        original_N_AB = self.N_AB
        
        # Perturb mass by +10%
        perturbed_m = original_m * 1.10
        
        # Recalculate expected forces using physics formulas
        # First scenario: f_2 should remain 14N (system still at rest if F_max > 14)
        perturbed_F_max = 2 * self.mu * perturbed_m * self.g
        if perturbed_F_max > 14.0:  # System still at rest
            expected_f_2 = 14.0  # Unchanged
        else:
            expected_f_2 = self.mu * 2 * perturbed_m * self.g  # Kinetic friction if moving
        
        # Second scenario forces scale linearly with mass
        expected_F_2 = 2 * self.mu * perturbed_m * self.g
        expected_N_AB = self.mu * perturbed_m * self.g
        
        # Check scaling factors
        mass_ratio = perturbed_m / original_m
        
        # First scenario check (only if system remains at rest)
        if perturbed_F_max > 14.0:
            if not self._isclose(expected_f_2, original_f_2):
                self._log(f"Monotonicity fail: f_2 should not change when system at rest", False)
                success = False
        
        # Second scenario forces should scale with mass
        if not self._isclose(expected_F_2 / original_F_2, mass_ratio):
            self._log(f"Monotonicity fail: F_2 scaling {expected_F_2/original_F_2:.3f} ≠ mass ratio {mass_ratio:.3f}", False)
            success = False
        
        if not self._isclose(expected_N_AB / original_N_AB, mass_ratio):
            self._log(f"Monotonicity fail: N_AB scaling {expected_N_AB/original_N_AB:.3f} ≠ mass ratio {mass_ratio:.3f}", False)
            success = False
        
        if success:
            self._log("Monotonicity test passed: forces scale correctly with mass", True)
        
        # Restore original values
        self.m = original_m
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all audit tests and return success status."""
        self.logs = ["=== Physics Audit Report ==="]
        
        test_results = []
        try:
            self._log("1. Testing physics invariants...")
            test_results.append(self.test_invariants())
            
            self._log("\n2. Testing trace consistency...")
            test_results.append(self.test_trace_consistency())
            
            self._log("\n3. Testing monotonicity...")
            test_results.append(self.test_monotonicity())
            
            overall_success = all(test_results)
            
            if overall_success:
                self._log("\n✓ All tests passed!", True)
            else:
                self._log("\n✗ Some tests failed", False)
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"Audit crashed: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, False)
            return False, self.logs


# Example usage (commented out for production)
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.m = 4.0
            self.mu = 0.2
            self.g = 10.0
            self.results = {
                'F_max': 16.0,
                'f_2': 14.0,
                'N/A': 0.0,
                'F_2': 16.0,
                'N_AB': 8.0,
                'f': 8.0
            }
    
    # Run audit
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)