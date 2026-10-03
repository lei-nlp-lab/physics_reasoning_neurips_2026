import math
import sympy as sp
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        # Get results with safe fallback
        self.results = getattr(solution_instance, 'results', {})
        self.I_0 = self._safe_get('I_0', 0.0)
        self.I_1 = self._safe_get('I_1', 0.0)
        self.I = self._safe_get('I', 0.0)  # Step 3 current
        self.I_2 = self._safe_get('I_2', 0.0)
        self.U = self._safe_get('U', 0.0)
        
        # Constants from problem (base units: Ω, A, V)
        self.R1 = 0.5
        self.R2 = 2.0
        self.R3 = 100.0
        self.Rg = 500.0
        self.Ig = 0.001  # 1 mA in Amperes
        
        # Tolerance for fuzzy matching (2% relative, 0.02 absolute)
        self.rel_tol = 0.02
        self.abs_tol = 0.02
        
        # Original values for monotonicity test backup
        self.original_R1 = self.R1
        self.original_results = {
            'I_0': self.I_0,
            'I_1': self.I_1,
            'I': self.I,
            'I_2': self.I_2,
            'U': self.U
        }
        
        # Test logs
        self.logs = []
        
    def _safe_get(self, key: str, default: float) -> float:
        """Safely get value from results or return default."""
        return self.results.get(key, default)
    
    def _is_close(self, a: float, b: float, rel_tol: float = None, abs_tol: float = None) -> bool:
        """Fuzzy numerical matching with physics-appropriate tolerances."""
        if rel_tol is None:
            rel_tol = self.rel_tol
        if abs_tol is None:
            abs_tol = self.abs_tol
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Verify conservation laws and circuit relationships from KB."""
        success = True
        
        # Test 1: Step 1 equation - I₀·R₁ = Ig·(Rg + R₂)
        left1 = self.I_0 * self.R1
        right1 = self.Ig * (self.Rg + self.R2)
        if not self._is_close(left1, right1):
            self.logs.append(f"FAIL Step 1 invariant: I₀·R₁={left1:.4f}, Ig·(Rg+R₂)={right1:.4f}")
            success = False
        else:
            self.logs.append(f"PASS Step 1 invariant: I₀·R₁ ≈ Ig·(Rg+R₂)")
        
        # Test 2: Step 2 equation - I₁ = I₀ + Ig
        sum_I0_Ig = self.I_0 + self.Ig
        if not self._is_close(self.I_1, sum_I0_Ig):
            self.logs.append(f"FAIL Step 2 invariant: I₁={self.I_1:.4f}, I₀+Ig={sum_I0_Ig:.4f}")
            success = False
        else:
            self.logs.append(f"PASS Step 2 invariant: I₁ ≈ I₀+Ig")
        
        # Test 3: Step 3 equation - Ig·Rg = I·(R₁ + R₂)
        left3 = self.Ig * self.Rg
        right3 = self.I * (self.R1 + self.R2)
        if not self._is_close(left3, right3):
            self.logs.append(f"FAIL Step 3 invariant: Ig·Rg={left3:.4f}, I·(R₁+R₂)={right3:.4f}")
            success = False
        else:
            self.logs.append(f"PASS Step 3 invariant: Ig·Rg ≈ I·(R₁+R₂)")
        
        # Test 4: Step 4 equation - I₂ = I + Ig
        sum_I_Ig = self.I + self.Ig
        if not self._is_close(self.I_2, sum_I_Ig):
            self.logs.append(f"FAIL Step 4 invariant: I₂={self.I_2:.4f}, I+Ig={sum_I_Ig:.4f}")
            success = False
        else:
            self.logs.append(f"PASS Step 4 invariant: I₂ ≈ I+Ig")
        
        # Test 5: Step 5 equation - U = I₂·R₃ + Ig·Rg
        left5 = self.I_2 * self.R3 + self.Ig * self.Rg
        if not self._is_close(self.U, left5):
            self.logs.append(f"FAIL Step 5 invariant: U={self.U:.4f}, I₂·R₃+Ig·Rg={left5:.4f}")
            success = False
        else:
            self.logs.append(f"PASS Step 5 invariant: U ≈ I₂·R₃+Ig·Rg")
        
        # Test 6: Voltage equality in parallel branches (terminal a-b)
        # Voltage across R1 should equal voltage across (Rg + R2)
        V_R1 = self.I_0 * self.R1
        V_Rg_R2 = self.Ig * (self.Rg + self.R2)
        if not self._is_close(V_R1, V_Rg_R2):
            self.logs.append(f"FAIL Parallel voltage equality (a-b): V_R1={V_R1:.4f}, V_Rg+R2={V_Rg_R2:.4f}")
            success = False
        else:
            self.logs.append(f"PASS Parallel voltage equality (a-b)")
        
        # Test 7: Voltage equality in parallel branches (terminal a-c)
        # Voltage across Rg should equal voltage across (R1 + R2)
        V_Rg = self.Ig * self.Rg
        V_R1_R2 = self.I * (self.R1 + self.R2)
        if not self._is_close(V_Rg, V_R1_R2):
            self.logs.append(f"FAIL Parallel voltage equality (a-c): V_Rg={V_Rg:.4f}, V_R1+R2={V_R1_R2:.4f}")
            success = False
        else:
            self.logs.append(f"PASS Parallel voltage equality (a-c)")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check that I_1 matches target (1005 mA = 1.005 A)
        if not self._is_close(self.I_1, 1.005, rel_tol=1e-3, abs_tol=1e-3):
            self.logs.append(f"FAIL I₁ consistency: {self.I_1:.6f} A ≠ 1.005 A")
            success = False
        else:
            self.logs.append(f"PASS I₁ consistent with target 1.005 A")
        
        # Check that I_2 matches target (205 mA = 0.205 A)
        if not self._is_close(self.I_2, 0.205, rel_tol=1e-3, abs_tol=1e-3):
            self.logs.append(f"FAIL I₂ consistency: {self.I_2:.6f} A ≠ 0.205 A")
            success = False
        else:
            self.logs.append(f"PASS I₂ consistent with target 0.205 A")
        
        # Check that U matches target (21 V)
        if not self._is_close(self.U, 21.0, rel_tol=1e-3, abs_tol=1e-3):
            self.logs.append(f"FAIL U consistency: {self.U:.4f} V ≠ 21.0 V")
            success = False
        else:
            self.logs.append(f"PASS U consistent with target 21.0 V")
        
        # Check internal consistency: I from I₂ and Ig
        calc_I = self.I_2 - self.Ig
        if not self._is_close(self.I, calc_I, rel_tol=1e-3, abs_tol=1e-3):
            self.logs.append(f"FAIL Internal consistency: I={self.I:.6f}, I₂-Ig={calc_I:.6f}")
            success = False
        else:
            self.logs.append(f"PASS Internal consistency: I ≈ I₂-Ig")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb input resistance and check expected trends."""
        success = True
        
        # Backup original state
        original_R1 = self.R1
        perturbed_R1 = original_R1 * 1.1  # Increase by 10%
        
        # Apply perturbation
        self.R1 = perturbed_R1
        
        # Recalculate using the formulas from steps
        # Step 1: I₀ = Ig·(Rg + R₂) / R₁
        new_I_0 = self.Ig * (self.Rg + self.R2) / self.R1
        
        # Step 2: I₁ = I₀ + Ig
        new_I_1 = new_I_0 + self.Ig
        
        # Step 3: I = Ig·Rg / (R₁ + R₂)
        new_I = self.Ig * self.Rg / (self.R1 + self.R2)
        
        # Step 4: I₂ = I + Ig
        new_I_2 = new_I + self.Ig
        
        # Step 5: U = I₂·R₃ + Ig·Rg
        new_U = new_I_2 * self.R3 + self.Ig * self.Rg
        
        # Expected trends based on physics:
        # 1. Increasing R₁ should decrease I₀ (Ohm's Law: I = V/R, V fixed)
        if new_I_0 >= self.original_results['I_0']:
            self.logs.append(f"FAIL Monotonicity I₀: Increased R₁ but I₀ didn't decrease")
            success = False
        else:
            self.logs.append(f"PASS Monotonicity I₀: R₁↑ → I₀↓")
        
        # 2. I₁ should follow same trend as I₀ (since I₁ = I₀ + Ig, Ig constant)
        if new_I_1 >= self.original_results['I_1']:
            self.logs.append(f"FAIL Monotonicity I₁: Increased R₁ but I₁ didn't decrease")
            success = False
        else:
            self.logs.append(f"PASS Monotonicity I₁: R₁↑ → I₁↓")
        
        # 3. Increasing R₁ should decrease I (from Step 3 formula)
        if new_I >= self.original_results['I']:
            self.logs.append(f"FAIL Monotonicity I: Increased R₁ but I didn't decrease")
            success = False
        else:
            self.logs.append(f"PASS Monotonicity I: R₁↑ → I↓")
        
        # 4. I₂ should follow same trend as I
        if new_I_2 >= self.original_results['I_2']:
            self.logs.append(f"FAIL Monotonicity I₂: Increased R₁ but I₂ didn't decrease")
            success = False
        else:
            self.logs.append(f"PASS Monotonicity I₂: R₁↑ → I₂↓")
        
        # 5. U should follow same trend as I₂ (since U = I₂·R₃ + constant)
        if new_U >= self.original_results['U']:
            self.logs.append(f"FAIL Monotonicity U: Increased R₁ but U didn't decrease")
            success = False
        else:
            self.logs.append(f"PASS Monotonicity U: R₁↑ → U↓")
        
        # Restore original state
        self.R1 = original_R1
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status with logs."""
        self.logs = ["=== Physics Checker Results ==="]
        
        # Run all tests
        inv_success = self.test_invariants()
        trace_success = self.test_trace_consistency()
        mono_success = self.test_monotonicity()
        
        # Summary
        all_success = inv_success and trace_success and mono_success
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Invariants: {'PASS' if inv_success else 'FAIL'}")
        self.logs.append(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}")
        self.logs.append(f"Overall: {'PASS' if all_success else 'FAIL'}")
        
        return all_success, self.logs

# Example usage (commented out for safety in sandbox):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'I_0': 1.004,  # A
#                 'I_1': 1.005,  # A
#                 'I': 0.204,    # A
#                 'I_2': 0.205,  # A
#                 'U': 21.0      # V
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)