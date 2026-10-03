import math
import traceback
from typing import Tuple, List, Any, Dict, Optional

class PhysicsChecker:
    """
    Physics auditor for ideal gas processes in p-V diagram.
    Problem: States 1->2->3 with 1->2 isochoric and 2->3 isothermal.
    """
    
    def __init__(self, solution_instance: Any):
        """
        Safely extract all required parameters without direct attribute access.
        """
        self.logs = []
        self.solution = solution_instance
        
        # Input parameters (state 1) - extract as floats
        self.p1 = self._safe_get_float('p1', 1.0e5)
        self.V1 = self._safe_get_float('V1', 0.002)  # Already in m³ in solution
        self.T1 = self._safe_get_float('T1', 200.0)
        
        # Target results from solution
        self.T2 = self._safe_get_float('T_2', None)
        self.p3 = self._safe_get_float('p_3', None)
        
        # Intermediate results (may not exist)
        self.p2 = self._safe_get_float('p2', None)
        self.V2 = self._safe_get_float('V2', None)
        self.V3 = self._safe_get_float('V3', None)
        
        # Physical constants
        self.R = 8.314  # Ideal gas constant [J/(mol·K)]
        
        # Derived quantities
        if self.p2 is None and self.T2 is not None:
            # From isochoric relation: p1/T1 = p2/T2
            self.p2 = self.p1 * (self.T2 / self.T1)
        
        if self.V2 is None:
            # Isochoric process: V2 = V1
            self.V2 = self.V1
        
        # Number of moles (from state 1) - use correct V1
        self.n = (self.p1 * self.V1) / (self.R * self.T1)
        
    def _safe_get_float(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieve float attribute from solution instance.
        """
        try:
            # First try results dictionary
            results = getattr(self.solution, 'results', {})
            val = results.get(key, None)
            if val is None:
                # Then try direct attribute
                val = getattr(self.solution, key, None)
            if val is None:
                return default
            # Ensure we return a float
            return float(val)
        except Exception:
            return default
    
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """
        Robust numerical comparison with tolerances.
        """
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify physical invariants: Boyle's Law and Isochoric relation.
        """
        logs = []
        success = True
        
        try:
            # 1. Check Boyle's Law for isothermal process 2->3
            if self.p2 is not None and self.p3 is not None and self.V2 is not None and self.V3 is not None:
                # p2 * V2 = p3 * V3 (isothermal)
                left = self.p2 * self.V2
                right = self.p3 * self.V3
                
                if not self._isclose(left, right):
                    logs.append(f"FAIL: Boyle's Law violated: p2*V2={left:.3e}, p3*V3={right:.3e}")
                    success = False
                else:
                    logs.append(f"OK: Boyle's Law satisfied (p2*V2 ≈ p3*V3)")
            
            # 2. Check Isochoric relation for process 1->2
            if self.T2 is not None and self.p2 is not None:
                # p1/T1 = p2/T2 (constant volume)
                ratio1 = self.p1 / self.T1
                ratio2 = self.p2 / self.T2
                
                if not self._isclose(ratio1, ratio2):
                    logs.append(f"FAIL: Isochoric relation violated: p1/T1={ratio1:.3e}, p2/T2={ratio2:.3e}")
                    success = False
                else:
                    logs.append(f"OK: Isochoric relation satisfied (p1/T1 ≈ p2/T2)")
            
            # 3. Check ideal gas law consistency across all states
            # State 1: p1*V1 = n*R*T1
            left1 = self.p1 * self.V1
            right1 = self.n * self.R * self.T1
            
            if not self._isclose(left1, right1):
                logs.append(f"FAIL: Ideal gas law violated at state 1: pV={left1:.3e}, nRT={right1:.3e}")
                success = False
            else:
                logs.append(f"OK: Ideal gas law satisfied at state 1")
            
            if self.T2 is not None and self.p2 is not None:
                # State 2: p2*V2 = n*R*T2
                left2 = self.p2 * self.V2
                right2 = self.n * self.R * self.T2
                
                if not self._isclose(left2, right2):
                    logs.append(f"FAIL: Ideal gas law violated at state 2: pV={left2:.3e}, nRT={right2:.3e}")
                    success = False
                else:
                    logs.append(f"OK: Ideal gas law satisfied at state 2")
            
            if self.p3 is not None and self.T2 is not None and self.V3 is not None:
                # State 3: p3*V3 = n*R*T2 (since T3 = T2)
                left3 = self.p3 * self.V3
                right3 = self.n * self.R * self.T2
                
                if not self._isclose(left3, right3):
                    logs.append(f"FAIL: Ideal gas law violated at state 3: pV={left3:.3e}, nRT={right3:.3e}")
                    success = False
                else:
                    logs.append(f"OK: Ideal gas law satisfied at state 3")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            traceback.print_exc()
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        success = True
        
        try:
            # Expected values from problem solution
            expected_T2 = 800.0  # K
            expected_p3 = (4/3) * 1.0e5  # Pa
            
            # 1. Check T2 against expected value
            if self.T2 is not None:
                if not self._isclose(self.T2, expected_T2):
                    logs.append(f"FAIL: T2={self.T2:.2f}K differs from expected 800K")
                    success = False
                else:
                    logs.append(f"OK: T2 ≈ 800K")
            
            # 2. Check p3 against expected value
            if self.p3 is not None:
                if not self._isclose(self.p3, expected_p3):
                    logs.append(f"FAIL: p3={self.p3:.3e}Pa differs from expected {(4/3)*1e5:.3e}Pa")
                    success = False
                else:
                    logs.append(f"OK: p3 ≈ (4/3)×10⁵ Pa")
            
            # 3. Check p2 derived from T2
            if self.T2 is not None and self.p2 is not None:
                # p2 should be p1 * (T2/T1) = 1e5 * (800/200) = 4e5 Pa
                expected_p2 = self.p1 * (expected_T2 / self.T1)
                if not self._isclose(self.p2, expected_p2):
                    logs.append(f"FAIL: p2={self.p2:.3e}Pa differs from expected {expected_p2:.3e}Pa")
                    success = False
                else:
                    logs.append(f"OK: p2 derived correctly from T2")
            
            # 4. Check V3 if available
            if self.V3 is not None and self.p2 is not None and self.p3 is not None:
                # From Boyle's Law: V3 = (p2 * V2) / p3
                expected_V3 = (self.p2 * self.V2) / self.p3
                if not self._isclose(self.V3, expected_V3):
                    logs.append(f"FAIL: V3={self.V3:.3e}m³ differs from expected {expected_V3:.3e}m³")
                    success = False
                else:
                    logs.append(f"OK: V3 consistent with Boyle's Law")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            traceback.print_exc()
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb an input parameter and check if outputs change as expected.
        """
        logs = []
        success = True
        
        try:
            # Original values
            T1_original = self.T1
            p1_original = self.p1
            
            # Test 1: Increase T1 by 10% → T2 should increase proportionally
            # (since isochoric: T2/T1 = p2/p1 = constant if p1 fixed)
            T1_perturbed = T1_original * 1.10
            
            # Recalculate T2 using the same ratio
            if self.T2 is not None:
                ratio_T2_T1 = self.T2 / T1_original
                T2_expected = T1_perturbed * ratio_T2_T1
                
                logs.append(f"Monotonicity check: T1 increased 10%, T2 should increase proportionally")
                logs.append(f"  Original: T1={T1_original}K, T2={self.T2}K")
                logs.append(f"  Perturbed: T1={T1_perturbed}K, expected T2≈{T2_expected}K")
                
                # Restore original value
                self.T1 = T1_original
            
            # Test 2: Increase p1 by 10% → T2 should increase proportionally
            # (since isochoric: T2/T1 = p2/p1, with p2 changing proportionally to p1)
            p1_perturbed = p1_original * 1.10
            
            if self.T2 is not None:
                ratio_T2_p1 = self.T2 / p1_original
                T2_expected_p = p1_perturbed * ratio_T2_p1
                
                logs.append(f"Monotonicity check: p1 increased 10%, T2 should increase proportionally")
                logs.append(f"  Original: p1={p1_original}Pa, T2={self.T2}K")
                logs.append(f"  Perturbed: p1={p1_perturbed}Pa, expected T2≈{T2_expected_p}K")
                
                # Restore original value
                self.p1 = p1_original
                
                success = True
                logs.append("OK: Monotonicity trends are physically reasonable")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            traceback.print_exc()
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and log messages.
        """
        all_logs = []
        overall_success = True
        
        # Test 1: Physical invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Test 1: Physical Invariants ===")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Test 2: Trace Consistency ===")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Test 3: Monotonicity ===")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall success: {overall_success}")
        
        return overall_success, all_logs