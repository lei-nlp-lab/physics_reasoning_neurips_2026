import math
import traceback
from typing import List, Tuple, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Initialize with empty results
        self.results = {}
        
        # ONLY get results from the results dictionary
        if hasattr(solution_instance, 'results'):
            self.results = getattr(solution_instance, 'results', {}).copy()
        
        # Extract values directly from results dict
        self.B = self.results.get('B')
        self.m = self.results.get('m')
        self.q = self.results.get('q')
        self.d = self.results.get('d')
        self.v_pos = self.results.get('v')
        self.R_pos = self.results.get('R')
        
        # For negative charge (not required but might be present)
        self.v_neg = self.results.get("v''")
        self.R_neg = self.results.get("R''")
        
        # Store test logs
        self.logs = []
    
    def _log(self, msg: str, success: bool) -> None:
        """Helper to log test results."""
        self.logs.append((success, msg))
    
    def _is_close(self, a: float, b: float) -> bool:
        """Robust numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def _get_numeric_value(self, val):
        """Convert value to float if possible."""
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants."""
        success = True
        
        # Get numeric values
        B = self._get_numeric_value(self.B)
        m = self._get_numeric_value(self.m)
        q = self._get_numeric_value(self.q)
        d = self._get_numeric_value(self.d)
        
        # Check for division by zero
        if m == 0 or B == 0 or q == 0:
            self._log(f"Cannot test invariants: zero value in m={m}, B={B}, q={q}", False)
            return False
        
        # Skip if missing essential inputs
        if None in [B, m, q, d]:
            self._log(f"Missing essential inputs for invariant tests: B={B}, m={m}, q={q}, d={d}", False)
            return False
        
        # Test 1: Radius formula for positive charge
        if self.R_pos is not None and self.v_pos is not None:
            R_num = self._get_numeric_value(self.R_pos)
            v_num = self._get_numeric_value(self.v_pos)
            
            if R_num is not None and v_num is not None:
                # r = m*v/(|q|*B)
                expected_R = abs(m * v_num / (abs(q) * B))
                if not self._is_close(R_num, expected_R):
                    self._log(f"Positive charge radius mismatch: R={R_num}, expected={expected_R}", False)
                    success = False
                else:
                    self._log("Positive charge radius formula verified", True)
        
        # Test 2: Radius formula for negative charge (if available)
        if self.R_neg is not None and self.v_neg is not None:
            R_num = self._get_numeric_value(self.R_neg)
            v_num = self._get_numeric_value(self.v_neg)
            
            if R_num is not None and v_num is not None:
                expected_R = abs(m * v_num / (abs(q) * B))
                if not self._is_close(R_num, expected_R):
                    self._log(f"Negative charge radius mismatch: R'={R_num}, expected={expected_R}", False)
                    success = False
                else:
                    self._log("Negative charge radius formula verified", True)
        
        # Test 3: Period invariance
        if B is not None and m is not None and q is not None:
            T = 2 * math.pi * m / (abs(q) * B)
            self._log(f"Period T = {T:.3e}s (should be same for both charges)", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate and final results."""
        success = True
        
        # Get numeric values
        B = self._get_numeric_value(self.B)
        m = self._get_numeric_value(self.m)
        q = self._get_numeric_value(self.q)
        d = self._get_numeric_value(self.d)
        
        # Check for division by zero
        if m == 0:
            self._log(f"Cannot test trace consistency: m={m} (division by zero)", False)
            return False
        
        if None in [B, m, q, d]:
            self._log(f"Missing inputs for trace consistency: B={B}, m={m}, q={q}, d={d}", False)
            return False
        
        sqrt2 = math.sqrt(2)
        
        # Check positive charge derivation chain
        if self.v_pos is not None:
            v_pos_num = self._get_numeric_value(self.v_pos)
            
            # v = (2+√2)*B*q*d/m
            expected_v_pos = (2 + sqrt2) * B * abs(q) * d / m
            
            if not self._is_close(v_pos_num, expected_v_pos):
                self._log(f"Positive charge velocity mismatch: v={v_pos_num:.3e}, expected={expected_v_pos:.3e}", False)
                success = False
            else:
                self._log("Positive charge derivation consistent", True)
                
            # Cross-check with radius
            if self.R_pos is not None:
                R_num = self._get_numeric_value(self.R_pos)
                # d = R - R/√2 → R = d/(1 - 1/√2)
                expected_R = d / (1 - 1/sqrt2)
                if not self._is_close(R_num, expected_R):
                    self._log(f"Positive charge geometry mismatch: R={R_num:.3e}, expected={expected_R:.3e}", False)
                    success = False
        
        # Check negative charge derivation chain (if available)
        if self.v_neg is not None:
            v_neg_num = self._get_numeric_value(self.v_neg)
            
            # v' = (2-√2)*B*q*d/m
            expected_v_neg = (2 - sqrt2) * B * abs(q) * d / m
            
            if not self._is_close(v_neg_num, expected_v_neg):
                self._log(f"Negative charge velocity mismatch: v'={v_neg_num:.3e}, expected={expected_v_neg:.3e}", False)
                success = False
            else:
                self._log("Negative charge derivation consistent", True)
                
            if self.R_neg is not None:
                R_num = self._get_numeric_value(self.R_neg)
                # d = R' + R'/√2 → R' = d/(1 + 1/√2)
                expected_R = d / (1 + 1/sqrt2)
                if not self._is_close(R_num, expected_R):
                    self._log(f"Negative charge geometry mismatch: R'={R_num:.3e}, expected={expected_R:.3e}", False)
                    success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test how results change with perturbed inputs."""
        success = True
        
        # Get numeric values
        B = self._get_numeric_value(self.B)
        m = self._get_numeric_value(self.m)
        q = self._get_numeric_value(self.q)
        d = self._get_numeric_value(self.d)
        
        # Check for division by zero
        if m == 0 or B == 0 or q == 0:
            self._log(f"Cannot test monotonicity: zero value in m={m}, B={B}, q={q}", False)
            return False
        
        if None in [B, m, q, d]:
            self._log(f"Missing inputs for monotonicity test: B={B}, m={m}, q={q}, d={d}", False)
            return False
        
        sqrt2 = math.sqrt(2)
        
        # Test 1: Increase B by 10%, velocity should increase proportionally
        B_perturbed = B * 1.10
        original_v_pos = (2 + sqrt2) * B * abs(q) * d / m
        perturbed_v_pos = (2 + sqrt2) * B_perturbed * abs(q) * d / m
        
        if not math.isclose(perturbed_v_pos/original_v_pos, 1.10, rel_tol=1e-2):
            self._log(f"Positive v not proportional to B: {perturbed_v_pos/original_v_pos:.3f} ≠ 1.10", False)
            success = False
        else:
            self._log("Positive charge velocity ∝ B (monotonic increase)", True)
        
        # Test 2: Increase m by 10%, velocity should decrease
        m_perturbed = m * 1.10
        perturbed_v_pos_m = (2 + sqrt2) * B * abs(q) * d / m_perturbed
        
        if perturbed_v_pos_m >= original_v_pos:
            self._log(f"Positive v did not decrease with increased mass", False)
            success = False
        else:
            self._log("Positive charge velocity ∝ 1/m (monotonic decrease)", True)
        
        # Test 3: Check negative charge monotonicity (if available)
        if self.v_neg is not None:
            original_v_neg = (2 - sqrt2) * B * abs(q) * d / m
            perturbed_v_neg = (2 - sqrt2) * B_perturbed * abs(q) * d / m
            
            if not math.isclose(perturbed_v_neg/original_v_neg, 1.10, rel_tol=1e-2):
                self._log(f"Negative v not proportional to B: {perturbed_v_neg/original_v_neg:.3f} ≠ 1.10", False)
                success = False
            else:
                self._log("Negative charge velocity ∝ B (monotonic increase)", True)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        
        # Run tests
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            # Overall success
            success = inv_ok and trace_ok and mono_ok
        except ZeroDivisionError as e:
            self._log(f"ZeroDivisionError in tests: {e}", False)
            success = False
        except Exception as e:
            self._log(f"Error in tests: {e}", False)
            success = False
        
        # Format logs
        formatted_logs = []
        for test_success, msg in self.logs:
            status = "✓" if test_success else "✗"
            formatted_logs.append(f"{status} {msg}")
        
        return success, formatted_logs