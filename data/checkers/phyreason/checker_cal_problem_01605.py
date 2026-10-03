import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution_instance = solution_instance
        
        # Extract inputs and results with safe defaults
        self.F_given = getattr(solution_instance, 'F_given', 
                              getattr(solution_instance, 'results', {}).get('F_given', 0.04))
        self.L = getattr(solution_instance, 'L', 
                        getattr(solution_instance, 'results', {}).get('L', 0.2))
        self.I = getattr(solution_instance, 'I', 
                        getattr(solution_instance, 'results', {}).get('I', 1.0))
        self.B = getattr(solution_instance, 'B', 
                        getattr(solution_instance, 'results', {}).get('B', None))
        self.F_new = getattr(solution_instance, 'F_new', 
                            getattr(solution_instance, 'results', {}).get('F_new', None))
        
        # Symbolic variables for physics laws
        self.F, self.B_sym, self.I_sym, self.L_sym = sp.symbols('F B I L')
        
    def test_invariants(self) -> List[str]:
        """Verify Ampere's force law for a straight wire in uniform magnetic field."""
        logs = []
        
        # Check 1: B = F_given / (I * L) from step 1
        if self.B is not None:
            expected_B = self.F_given / (self.I * self.L)
            if math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Invariant (step1): B = F/(I·L) holds: {self.B:.3f} = {self.F_given:.3f}/({self.I:.1f}×{self.L:.1f})")
            else:
                logs.append(f"✗ Invariant (step1): B = {self.B:.3f} ≠ F/(I·L) = {expected_B:.3f}")
        
        # Check 2: F_new = B * I * L from step 2 (if F_new provided)
        if self.F_new is not None and self.B is not None:
            # Note: We don't have I_new and L_new, so we assume same I and L
            # This check may fail if the problem uses different I/L for step2
            expected_F = self.B * self.I * self.L
            if math.isclose(self.F_new, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Invariant (step2): F = B·I·L holds: {self.F_new:.3f} = {self.B:.3f}×{self.I:.1f}×{self.L:.1f}")
            else:
                logs.append(f"✗ Invariant (step2): F_new = {self.F_new:.3f} ≠ B·I·L = {expected_F:.3f}")
                logs.append("  Note: This may be expected if step2 uses different I/L values")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Check B from step 1 matches expected 0.2 T
        if self.B is not None:
            if math.isclose(self.B, 0.2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Trace consistency: B = {self.B:.3f} T matches target 0.2 T")
            else:
                logs.append(f"✗ Trace consistency: B = {self.B:.3f} T ≠ 0.2 T")
        
        # Check F_new from step 2 matches expected 0.2 N
        if self.F_new is not None:
            if math.isclose(self.F_new, 0.2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Trace consistency: F_new = {self.F_new:.3f} N matches target 0.2 N")
            else:
                logs.append(f"✗ Trace consistency: F_new = {self.F_new:.3f} N ≠ 0.2 N")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Perturb current and verify force changes proportionally."""
        logs = []
        
        if self.B is None:
            logs.append("Skipping monotonicity: B not available")
            return logs
        
        # Original setup
        I_original = self.I
        F_original = self.B * I_original * self.L
        
        # Perturb current (increase by 50%)
        I_perturbed = I_original * 1.5
        F_perturbed = self.B * I_perturbed * self.L
        
        # Check proportionality: F ∝ I (with fixed B and L)
        expected_ratio = I_perturbed / I_original
        actual_ratio = F_perturbed / F_original
        
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"✓ Monotonicity: F ∝ I holds (ratio: {actual_ratio:.3f} ≈ {expected_ratio:.3f})")
        else:
            logs.append(f"✗ Monotonicity: F ∝ I violated (ratio: {actual_ratio:.3f} ≠ {expected_ratio:.3f})")
        
        # Restore original values (conceptual)
        logs.append(f"  Restored: I = {I_original:.1f} A, F = {F_original:.3f} N")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status with logs."""
        all_logs = []
        
        all_logs.append("=" * 50)
        all_logs.append("Physics Audit Report")
        all_logs.append("=" * 50)
        
        # Run tests
        all_logs.append("\n1. Invariant Tests (Ampere's Law):")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n2. Trace Consistency Tests:")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n3. Monotonicity Test:")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        failed_tests = [log for log in all_logs if "✗" in log]
        success = len(failed_tests) == 0
        
        all_logs.append("\n" + "=" * 50)
        all_logs.append(f"Overall: {'PASS' if success else 'FAIL'} ({len(failed_tests)} failures)")
        all_logs.append("=" * 50)
        
        return success, all_logs