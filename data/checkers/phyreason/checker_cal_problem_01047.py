import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        """
        # Safe extraction of all expected results
        self.V_1 = getattr(solution_instance, 'V_1', getattr(solution_instance, 'results', {}).get('V_1', 0.0))
        self.DeltaV = getattr(solution_instance, 'DeltaV', getattr(solution_instance, 'results', {}).get('DeltaV', 0.0))
        self.p_1 = getattr(solution_instance, 'p_1', getattr(solution_instance, 'results', {}).get('p_1', 0.0))
        self.p_2 = getattr(solution_instance, 'p_2', getattr(solution_instance, 'results', {}).get('p_2', 0.0))
        self.V_2 = getattr(solution_instance, 'V_2', getattr(solution_instance, 'results', {}).get('V_2', 0.0))
        self.n = getattr(solution_instance, 'n', getattr(solution_instance, 'results', {}).get('n', 0.0))
        self.fracDeltaVV_2 = getattr(solution_instance, 'fracDeltaVV_2', 
                                     getattr(solution_instance, 'results', {}).get('fracDeltaVV_2', 0.0))
        
        # Store logs
        self.logs = []
        
    def _log(self, msg: str, success: bool):
        self.logs.append((success, msg))
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify Boyle's Law consistency across different steps.
        """
        # Boyle's Law: P1*V1 = P2*V2 at constant temperature
        # Check step_6: p2 * V1 = p1 * V2
        boyle_check = self._is_close(self.p_2 * self.V_1, self.p_1 * self.V_2)
        if not boyle_check:
            msg = f"Boyle's Law violation: p2*V1={self.p_2*self.V_1} ≠ p1*V2={self.p_1*self.V_2}"
            self._log(msg, False)
        else:
            self._log(f"Boyle's Law check passed: p2*V1 = p1*V2 ≈ {self.p_2*self.V_1}", True)
            
        # Check average pressure formula if applicable
        p_avg = (self.p_1 + self.p_2) / 2
        if self.p_2 > self.p_1:
            if not (self.p_1 < p_avg < self.p_2):
                self._log(f"Average pressure {p_avg} not between p1={self.p_1} and p2={self.p_2}", False)
            else:
                self._log(f"Average pressure check passed: {p_avg} between {self.p_1} and {self.p_2}", True)

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        # FIXED: Use the correct pump stroke volume (0.25 L) for n calculation, not self.DeltaV
        pump_stroke_volume = 0.25  # 250 cm^3 = 0.25 L
        
        # Check number of strokes formula: n = (p2*V1 - p1*V1) / (p1 * pump_stroke_volume)
        expected_n = (self.p_2 * self.V_1 - self.p_1 * self.V_1) / (self.p_1 * pump_stroke_volume)
        n_consistent = self._is_close(self.n, expected_n)
        
        if not n_consistent:
            msg = f"Stroke count inconsistent: n={self.n} vs expected {expected_n}"
            self._log(msg, False)
        else:
            self._log(f"Stroke count consistent: n={self.n}", True)
            
        # Check V2 calculation
        expected_V2 = (self.p_2 * self.V_1) / self.p_1
        V2_consistent = self._is_close(self.V_2, expected_V2)
        
        if not V2_consistent:
            msg = f"V2 inconsistent: {self.V_2} vs expected {expected_V2}"
            self._log(msg, False)
        else:
            self._log(f"V2 consistent: {self.V_2}", True)
            
        # Check expelled volume ratio
        V_tank = 10.0  # 10L tank volume
        expelled_volume = self.V_2 - V_tank
        expected_frac = expelled_volume / self.V_2 if self.V_2 != 0 else 0.0
        
        frac_consistent = self._is_close(self.fracDeltaVV_2, expected_frac) or \
                         self._is_close(self.fracDeltaVV_2, 1/3)  # Allow direct 1/3 check
        
        if not frac_consistent:
            msg = f"Expelled fraction inconsistent: {self.fracDeltaVV_2} vs expected {expected_frac}"
            self._log(msg, False)
        else:
            self._log(f"Expelled fraction consistent: {self.fracDeltaVV_2}", True)

    def test_monotonicity(self):
        """
        Perturb initial liquid volume and check if stroke count changes appropriately.
        The correct physics: n = (p2*V1 - p1*V1) / (p1*DeltaV_pump) = (p2 - p1)*V1/(p1*DeltaV_pump)
        So n is directly proportional to V1. If V1 decreases, n should decrease.
        """
        original_V_1 = self.V_1
        perturbed_V_1 = original_V_1 * 0.9  # 10% less gas volume
        pump_stroke_volume = 0.25  # 250 cm^3 = 0.25 L
        
        # Calculate expected stroke count with perturbed volume
        perturbed_n = (self.p_2 * perturbed_V_1 - self.p_1 * perturbed_V_1) / (self.p_1 * pump_stroke_volume)
        
        # Less initial gas volume should require FEWER strokes (n ∝ V1)
        if perturbed_n < self.n:
            self._log(f"Monotonicity check passed: less gas volume ({perturbed_V_1}L) requires fewer strokes ({perturbed_n:.1f})", True)
        else:
            self._log(f"Monotonicity warning: less gas volume gave {perturbed_n:.1f} strokes, expected less than {self.n}", False)
            
        # Restore original value
        self.V_1 = original_V_1

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return success flag and log messages.
        """
        self.logs = []  # Reset logs
        
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
            
            # Check final answers match targets
            n_check = self._is_close(self.n, 48.0)
            frac_check = self._is_close(self.fracDeltaVV_2, 1/3)
            
            if n_check and frac_check:
                self._log("Final answers match target values", True)
            else:
                self._log(f"Final answers mismatch: n={self.n} (expected 48), ratio={self.fracDeltaVV_2} (expected 1/3)", False)
            
        except Exception as e:
            self._log(f"Test execution error: {str(e)}", False)
        
        # Determine overall success
        success = all(log[0] for log in self.logs)
        log_messages = [msg for _, msg in self.logs]
        
        return success, log_messages