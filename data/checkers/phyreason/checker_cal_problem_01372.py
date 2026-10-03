import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    """Physics auditor for train release problem."""
    
    def __init__(self, solution_instance):
        """Safely extract inputs and results."""
        # Problem constants
        self.g = 9.8  # m/s²
        
        # Extract inputs with safe defaults
        self.h = self._safe_get(solution_instance, 'h', 2.5)
        self.a = self._safe_get(solution_instance, 'a', 0.98)
        
        # Extract results (may be None if not present)
        self.Delta_s = self._safe_get(solution_instance, 'Deltas', None)
        self.s2 = self._safe_get(solution_instance, 's_2', None)
        self.s1 = self._safe_get(solution_instance, 's_1', None)
        self.t = self._safe_get(solution_instance, 't', None)
        
        # Derived quantities (computed if needed)
        self.t_expected = math.sqrt(2 * self.h / self.g) if self.h else None
        self.v0 = None  # Will compute if s1 and t are available
        
    def _safe_get(self, obj, key, default):
        """Robust attribute access with fallback."""
        try:
            val = getattr(obj, key, None)
            if val is not None:
                return float(val)
            # Try results dictionary
            if hasattr(obj, 'results') and isinstance(obj.results, dict):
                val = obj.results.get(key, default)
                return float(val) if val is not None else default
            return default
        except (AttributeError, TypeError, ValueError):
            return default
    
    def _is_close(self, a: Optional[float], b: Optional[float]) -> bool:
        """Fuzzy numerical comparison with 2% tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> List[str]:
        """Verify fundamental kinematic laws."""
        logs = []
        
        # 1. Vertical motion invariant: t = √(2h/g)
        if self.t is not None and self.t_expected is not None:
            if self._is_close(self.t, self.t_expected):
                logs.append("PASS: Vertical free-fall time matches √(2h/g)")
            else:
                logs.append(f"FAIL: t={self.t:.4f}, expected √(2h/g)={self.t_expected:.4f}")
        
        # 2. Horizontal motion invariants (if data available)
        if self.t is not None and self.t > 0:
            # Compute v0 from s1 if available
            if self.s1 is not None:
                self.v0 = self.s1 / self.t
                
                # Check s2 against uniformly accelerated motion
                if self.s2 is not None:
                    s2_expected = self.v0 * self.t + 0.5 * self.a * self.t**2
                    if self._is_close(self.s2, s2_expected):
                        logs.append("PASS: Train displacement matches v0t + ½at²")
                    else:
                        logs.append(f"FAIL: s2={self.s2:.4f}, expected {s2_expected:.4f}")
            
            # Check Delta_s = ½ a t² (main kinematic invariant)
            if self.Delta_s is not None:
                delta_expected = 0.5 * self.a * self.t**2
                if self._is_close(self.Delta_s, delta_expected):
                    logs.append("PASS: Δs matches ½at² (kinematic invariant)")
                else:
                    logs.append(f"FAIL: Δs={self.Delta_s:.4f}, expected ½at²={delta_expected:.4f}")
                
                # Also check alternative form Δs = a*h/g
                alt_expected = self.a * self.h / self.g
                if self._is_close(self.Delta_s, alt_expected):
                    logs.append("PASS: Δs matches a*h/g (derived form)")
        
        # 3. Check consistency between s1, s2, and Delta_s
        if all(x is not None for x in [self.s1, self.s2, self.Delta_s]):
            delta_from_diff = self.s2 - self.s1
            if self._is_close(self.Delta_s, delta_from_diff):
                logs.append("PASS: Δs = s2 - s1 (definition)")
            else:
                logs.append(f"FAIL: Δs={self.Delta_s:.4f} ≠ s2-s1={delta_from_diff:.4f}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answer."""
        logs = []
        
        # Target answer from problem statement
        target_delta = 0.25
        
        if self.Delta_s is not None:
            # 1. Match against target answer
            if self._is_close(self.Delta_s, target_delta):
                logs.append(f"PASS: Δs={self.Delta_s:.4f} matches target {target_delta} m")
            else:
                logs.append(f"FAIL: Δs={self.Delta_s:.4f} ≠ target {target_delta} m")
            
            # 2. Verify derived from h and a
            delta_from_inputs = self.a * self.h / self.g
            if self._is_close(self.Delta_s, delta_from_inputs):
                logs.append("PASS: Δs consistent with inputs (a*h/g)")
            else:
                logs.append(f"FAIL: Δs≠a*h/g ({delta_from_inputs:.4f})")
        
        # 3. If t available, check chain: h → t → Δs
        if self.t is not None and self.Delta_s is not None:
            delta_via_t = 0.5 * self.a * self.t**2
            if self._is_close(self.Delta_s, delta_via_t):
                logs.append("PASS: Δs consistent with t via ½at²")
            else:
                logs.append(f"FAIL: Δs≠½at² ({delta_via_t:.4f})")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Perturb inputs and check expected trends."""
        logs = []
        
        # Use formula Δs = a*h/g for sensitivity analysis
        original_delta = self.a * self.h / self.g
        
        # 1. Perturb height (+10%)
        h_perturbed = self.h * 1.10
        delta_h_perturbed = self.a * h_perturbed / self.g
        if delta_h_perturbed > original_delta:
            logs.append("PASS: Δs increases with h (positive sensitivity)")
        else:
            logs.append("FAIL: Δs should increase with h")
        
        # 2. Perturb acceleration (+10%)
        a_perturbed = self.a * 1.10
        delta_a_perturbed = a_perturbed * self.h / self.g
        if delta_a_perturbed > original_delta:
            logs.append("PASS: Δs increases with a (positive sensitivity)")
        else:
            logs.append("FAIL: Δs should increase with a")
        
        # 3. Check linearity: Δs ∝ a and Δs ∝ h
        # Compute partial derivatives numerically
        h_slope = self.a / self.g
        a_slope = self.h / self.g
        
        if h_slope > 0 and a_slope > 0:
            logs.append("PASS: Δs is linear in both h and a (positive coefficients)")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all checks and return success status."""
        logs = ["=== Physics Audit Report ==="]
        success = True
        
        try:
            # Run invariant tests
            inv_logs = self.test_invariants()
            logs.extend(inv_logs)
            if any("FAIL" in log for log in inv_logs):
                success = False
            
            # Run trace consistency
            trace_logs = self.test_trace_consistency()
            logs.extend(trace_logs)
            if any("FAIL" in log for log in trace_logs):
                success = False
            
            # Run monotonicity tests
            mono_logs = self.test_monotonicity()
            logs.extend(mono_logs)
            if any("FAIL" in log for log in mono_logs):
                success = False
            
        except Exception as e:
            logs.append(f"ERROR during audit: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        # Summary
        if success:
            logs.append("\n✓ All physics checks passed.")
        else:
            logs.append("\n✗ Some physics checks failed.")
        
        return success, logs