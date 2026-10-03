import math
import sympy as sp
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with fallbacks
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'inputs', {}).get('L', 1.0))
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'inputs', {}).get('E', 1.0))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 9.8))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        
        # Safe extraction of results with fallbacks
        results = getattr(solution_instance, 'results', {})
        self.r = results.get('r', 0.0)
        self.t_2 = results.get('t_2', 0.0)
        self.T = results.get('T', 0.0)
        self.q = results.get('q', 0.0)
        self.q_over_m = results.get('q/m', 0.0)
        self.t = results.get('t', 0.0)
        self.v_avg = results.get('v_avg', 0.0)
        self.t_3 = results.get('t_3', 0.0)
        self.t_all = results.get('t_all', 0.0)
        self.s = results.get('s', 0.0)
        self.t_1 = results.get('t_1', 0.0)
        
        self.logs = []
        
    def _log(self, message: str, success: bool = True) -> None:
        self.logs.append(f"{'✓' if success else '✗'} {message}")
    
    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def _check_formula(self, computed: float, expected_expr: str, desc: str) -> bool:
        """Check if computed value matches expected symbolic formula."""
        try:
            # Evaluate expected expression using extracted parameters
            expected = eval(expected_expr, {
                'L': self.L, 'E': self.E, 'B': self.B, 'g': self.g,
                'm': self.m, 'pi': math.pi, 'sqrt': math.sqrt
            })
            if self._is_close(computed, expected):
                self._log(f"{desc}: {computed:.4g} ≈ {expected_expr} = {expected:.4g}")
                return True
            else:
                self._log(f"{desc}: {computed:.4g} ≠ {expected_expr} = {expected:.4g}", False)
                return False
        except Exception as e:
            self._log(f"{desc}: Formula check failed: {str(e)}", False)
            return False
    
    def test_invariants(self) -> bool:
        """Verify physical invariants from KB theorems."""
        success = True
        
        # 1. Force balance: qE = mg (from step_1)
        if self.q != 0 and self.m != 0:
            force_balance = self._is_close(self.q * self.E, self.m * self.g)
            if not force_balance:
                self._log(f"Force balance violated: qE = {self.q*self.E:.4g} ≠ mg = {self.m*self.g:.4g}", False)
                success = False
            else:
                self._log(f"Force balance: qE = mg = {self.m*self.g:.4g}")
            
            # Specific charge q/m = g/E
            expected_q_over_m = self.g / self.E
            if not self._is_close(self.q_over_m, expected_q_over_m):
                self._log(f"Specific charge mismatch: {self.q_over_m:.4g} ≠ g/E = {expected_q_over_m:.4g}", False)
                success = False
            else:
                self._log(f"Specific charge: q/m = g/E = {expected_q_over_m:.4g}")
        
        # 2. Magnetic centripetal force invariant (using period formula)
        if self.T > 0 and self.B != 0 and self.q_over_m != 0:
            # From KB: T = 2πm/(|q|B) = 2π/(|q/m|·B)
            expected_T = 2 * math.pi / (abs(self.q_over_m) * self.B)
            if not self._is_close(self.T, expected_T):
                self._log(f"Period mismatch: T = {self.T:.4g} ≠ 2π/(|q/m|B) = {expected_T:.4g}", False)
                success = False
            else:
                self._log(f"Period: T = 2π/(|q/m|B) = {expected_T:.4g}")
        
        # 3. Geometric invariants
        if not self._is_close(self.r, self.L):
            self._log(f"Radius mismatch: r = {self.r:.4g} ≠ L = {self.L:.4g}", False)
            success = False
        else:
            self._log(f"Radius: r = L = {self.L:.4g}")
        
        expected_s = math.sqrt(2) * self.L
        if not self._is_close(self.s, expected_s):
            self._log(f"Displacement mismatch: s = {self.s:.4g} ≠ √2·L = {expected_s:.4g}", False)
            success = False
        else:
            self._log(f"Displacement: s = √2·L = {expected_s:.4g}")
        
        # 4. Quarter-circle time relation
        if self.T > 0 and self.t > 0:
            if not self._is_close(self.t, self.T / 4):
                self._log(f"Time mismatch: t = {self.t:.4g} ≠ T/4 = {self.T/4:.4g}", False)
                success = False
            else:
                self._log(f"Quarter-circle time: t = T/4 = {self.T/4:.4g}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check average velocity formula from step_6
        if self.v_avg > 0 and self.t > 0 and self.s > 0:
            # Numerical: v_avg = s/t
            v_avg_from_s_t = self.s / self.t
            if not self._is_close(self.v_avg, v_avg_from_s_t):
                self._log(f"Average velocity mismatch: {self.v_avg:.4g} ≠ s/t = {v_avg_from_s_t:.4g}", False)
                success = False
            else:
                self._log(f"Average velocity: v_avg = s/t = {v_avg_from_s_t:.4g}")
            
            # Symbolic: v_avg = (2√2·B·g·L)/(π·E)
            expected_v_avg = (2 * math.sqrt(2) * self.B * self.g * self.L) / (math.pi * self.E)
            if not self._is_close(self.v_avg, expected_v_avg):
                self._log(f"Average velocity formula mismatch: {self.v_avg:.4g} ≠ (2√2·B·g·L)/(π·E) = {expected_v_avg:.4g}", False)
                success = False
            else:
                self._log(f"Average velocity formula: (2√2·B·g·L)/(π·E) = {expected_v_avg:.4g}")
        
        # 2. Check period formula consistency
        if self.T > 0:
            # From step_4: T = 2πE/(Bg)
            expected_T_formula = 2 * math.pi * self.E / (self.B * self.g)
            if not self._is_close(self.T, expected_T_formula):
                self._log(f"Period formula mismatch: T = {self.T:.4g} ≠ 2πE/(Bg) = {expected_T_formula:.4g}", False)
                success = False
            else:
                self._log(f"Period formula: T = 2πE/(Bg) = {expected_T_formula:.4g}")
        
        # 3. Check time segment consistency if available
        if all(x > 0 for x in [self.t_1, self.t_2, self.t_3, self.t_all]):
            t_sum = self.t_1 + self.t_2 + self.t_3
            if not self._is_close(self.t_all, t_sum):
                self._log(f"Total time mismatch: t_all = {self.t_all:.4g} ≠ sum(t_i) = {t_sum:.4g}", False)
                success = False
            else:
                self._log(f"Total time consistency: t_all = sum(t_i) = {t_sum:.4g}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb input parameters and check expected trends."""
        success = True
        
        # Store original values
        orig_L = self.L
        orig_v_avg = self.v_avg
        
        # Test 1: Increase L by 10%, v_avg should increase proportionally
        try:
            self.L *= 1.1
            # Recalculate v_avg using formula: (2√2·B·g·L)/(π·E)
            new_v_avg = (2 * math.sqrt(2) * self.B * self.g * self.L) / (math.pi * self.E)
            
            if new_v_avg > orig_v_avg:
                self._log(f"Monotonicity L↑ → v_avg↑: {orig_v_avg:.4g} → {new_v_avg:.4g}")
            else:
                self._log(f"Monotonicity violation: L↑ but v_avg↓", False)
                success = False
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
        finally:
            self.L = orig_L  # Restore
        
        # Test 2: Increase B by 10%, v_avg should increase proportionally
        orig_B = self.B
        try:
            self.B *= 1.1
            new_v_avg = (2 * math.sqrt(2) * self.B * self.g * self.L) / (math.pi * self.E)
            
            if new_v_avg > orig_v_avg:
                self._log(f"Monotonicity B↑ → v_avg↑: {orig_v_avg:.4g} → {new_v_avg:.4g}")
            else:
                self._log(f"Monotonicity violation: B↑ but v_avg↓", False)
                success = False
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
        finally:
            self.B = orig_B
        
        # Test 3: Increase E by 10%, v_avg should decrease
        orig_E = self.E
        try:
            self.E *= 1.1
            new_v_avg = (2 * math.sqrt(2) * self.B * self.g * self.L) / (math.pi * self.E)
            
            if new_v_avg < orig_v_avg:
                self._log(f"Monotonicity E↑ → v_avg↓: {orig_v_avg:.4g} → {new_v_avg:.4g}")
            else:
                self._log(f"Monotonicity violation: E↑ but v_avg↑", False)
                success = False
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
        finally:
            self.E = orig_E
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics checks."""
        self.logs = []
        self.logs.append("=" * 60)
        self.logs.append("Physics Audit Report")
        self.logs.append("=" * 60)
        
        overall_success = True
        
        try:
            # Test 1: Invariants
            self.logs.append("\n1. Testing Physical Invariants:")
            if not self.test_invariants():
                overall_success = False
            
            # Test 2: Trace Consistency
            self.logs.append("\n2. Testing Trace Consistency:")
            if not self.test_trace_consistency():
                overall_success = False
            
            # Test 3: Monotonicity
            self.logs.append("\n3. Testing Monotonicity:")
            if not self.test_monotonicity():
                overall_success = False
            
            # Summary
            self.logs.append("\n" + "=" * 60)
            if overall_success:
                self.logs.append("✓ All physics checks passed!")
            else:
                self.logs.append("✗ Some physics checks failed.")
            self.logs.append("=" * 60)
            
        except Exception as e:
            self.logs.append(f"\n✗ Checker crashed: {str(e)}")
            self.logs.append(traceback.format_exc())
            overall_success = False
        
        return overall_success, self.logs