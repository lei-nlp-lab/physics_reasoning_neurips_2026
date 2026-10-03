import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Safely extract all inputs and results using robust attribute access."""
        self.solution = solution_instance
        
        # Extract input parameters with safe fallbacks - use numeric defaults
        self.a = self._safe_get_numeric('a', 1.0)
        self.k = self._safe_get_numeric('k', 1.0)  # charge-to-mass ratio
        self.B = self._safe_get_numeric('B', 1.0)  # magnetic field strength
        
        # Extract results with safe fallbacks
        self.r = self._safe_get_numeric('r', None)
        self.v = self._safe_get_numeric('v', None)
        self.T = self._safe_get_numeric('T', None)
        self.t = self._safe_get_numeric('t', None)
        
        # Store logs for test results
        self.logs: List[str] = []
        
    def _safe_get_numeric(self, key: str, default: Any = None) -> Any:
        """Safely get numeric attribute from solution instance or results dict."""
        # First try direct attribute
        value = getattr(self.solution, key, None)
        if value is not None:
            return float(value) if not isinstance(value, (int, float)) else value
        
        # Then try results dictionary
        results = getattr(self.solution, 'results', {})
        value = results.get(key, None)
        
        # If still None, return default
        if value is None:
            return default
            
        # Convert to float if possible
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    
    def _log(self, message: str, success: bool = None):
        """Add message to logs with optional success indicator."""
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(f"  {message}")
    
    def _is_close(self, a, b, tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerance."""
        try:
            a_num = float(a)
            b_num = float(b)
            return math.isclose(a_num, b_num, rel_tol=tol, abs_tol=tol)
        except (ValueError, TypeError):
            return False
    
    def test_invariants(self) -> bool:
        """Verify fundamental physical laws from KB."""
        success = True
        
        try:
            # 1. Geometry invariant: r = 2a (from step_1)
            expected_r = 2 * self.a
            if not self._is_close(self.r, expected_r):
                self._log(f"Geometry invariant failed: r={self.r}, 2a={expected_r}", False)
                success = False
            else:
                self._log("Geometry invariant passed: r = 2a", True)
            
            # 2. Lorentz force as centripetal force: v = k*B*r
            expected_v = self.k * self.B * self.r
            if not self._is_close(self.v, expected_v):
                self._log(f"Lorentz force invariant failed: v={self.v}, k*B*r={expected_v}", False)
                success = False
            else:
                self._log("Lorentz force invariant passed: v = k*B*r", True)
            
            # 3. Period formula: T = 2π/(k*B) (k is positive)
            expected_T = 2 * math.pi / (self.k * self.B)
            if not self._is_close(self.T, expected_T):
                self._log(f"Period invariant failed: T={self.T}, expected {expected_T:.4f}", False)
                success = False
            else:
                self._log("Period invariant passed: T = 2π/(kB)", True)
            
            # 4. Time from arc length: t = T/6
            expected_t_from_T = self.T / 6
            if not self._is_close(self.t, expected_t_from_T):
                self._log(f"Time from period failed: t={self.t}, T/6={expected_t_from_T:.4f}", False)
                success = False
            else:
                self._log("Time from period passed: t = T/6", True)
            
            # 5. Combined time formula: t = π/(3*k*B)
            expected_t_direct = math.pi / (3 * self.k * self.B)
            if not self._is_close(self.t, expected_t_direct):
                self._log(f"Direct time formula failed: t={self.t}, π/(3kB)={expected_t_direct:.4f}", False)
                success = False
            else:
                self._log("Direct time formula passed: t = π/(3kB)", True)
                
        except Exception as e:
            self._log(f"Invariant test error: {str(e)}", False)
            success = False
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        try:
            # Target answers from problem statement
            target_r = 2 * self.a
            target_v = 2 * self.k * self.B * self.a
            target_t = math.pi / (3 * self.k * self.B)
            
            # Check radius
            if not self._is_close(self.r, target_r):
                self._log(f"Trace consistency (radius): r={self.r}, target={target_r:.4f}", False)
                success = False
            else:
                self._log("Trace consistency (radius) passed", True)
            
            # Check velocity
            if not self._is_close(self.v, target_v):
                self._log(f"Trace consistency (velocity): v={self.v}, target={target_v:.4f}", False)
                success = False
            else:
                self._log("Trace consistency (velocity) passed", True)
            
            # Check time
            if not self._is_close(self.t, target_t):
                self._log(f"Trace consistency (time): t={self.t}, target={target_t:.4f}", False)
                success = False
            else:
                self._log("Trace consistency (time) passed", True)
                
        except Exception as e:
            self._log(f"Trace consistency error: {str(e)}", False)
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """Test that results change correctly when inputs are perturbed."""
        success = True
        
        try:
            # Use current values as baseline
            a_base = self.a
            k_base = self.k
            B_base = self.B
            
            # Calculate baseline results using physics formulas
            r_base = 2 * a_base
            v_base = k_base * B_base * r_base
            t_base = math.pi / (3 * k_base * B_base)
            
            # Test perturbation of each parameter
            for param_name, perturb_factor in [('B', 1.1), ('k', 1.1), ('a', 1.1)]:
                # Create perturbed values
                a_val = a_base * (1.1 if param_name == 'a' else 1.0)
                k_val = k_base * (1.1 if param_name == 'k' else 1.0)
                B_val = B_base * (1.1 if param_name == 'B' else 1.0)
                
                # Calculate perturbed results
                r_pert = 2 * a_val
                v_pert = k_val * B_val * r_pert
                t_pert = math.pi / (3 * k_val * B_val)
                
                # Check trends
                if param_name == 'B':
                    # B increases → r unchanged, v increases, t decreases
                    if not self._is_close(r_pert, r_base):
                        self._log(f"Monotonicity (B↑): r should be unchanged", False)
                        success = False
                    if v_pert <= v_base:
                        self._log(f"Monotonicity (B↑): v should increase", False)
                        success = False
                    if t_pert >= t_base:
                        self._log(f"Monotonicity (B↑): t should decrease", False)
                        success = False
                    else:
                        self._log(f"Monotonicity (B↑) trends correct", True)
                        
                elif param_name == 'k':
                    # k increases → r unchanged, v increases, t decreases
                    if not self._is_close(r_pert, r_base):
                        self._log(f"Monotonicity (k↑): r should be unchanged", False)
                        success = False
                    if v_pert <= v_base:
                        self._log(f"Monotonicity (k↑): v should increase", False)
                        success = False
                    if t_pert >= t_base:
                        self._log(f"Monotonicity (k↑): t should decrease", False)
                        success = False
                    else:
                        self._log(f"Monotonicity (k↑) trends correct", True)
                        
                elif param_name == 'a':
                    # a increases → r increases, v increases, t unchanged
                    if r_pert <= r_base:
                        self._log(f"Monotonicity (a↑): r should increase", False)
                        success = False
                    if v_pert <= v_base:
                        self._log(f"Monotonicity (a↑): v should increase", False)
                        success = False
                    if not self._is_close(t_pert, t_base):
                        self._log(f"Monotonicity (a↑): t should be unchanged", False)
                        success = False
                    else:
                        self._log(f"Monotonicity (a↑) trends correct", True)
                
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", False)
            success = False
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        self.logs = []  # Reset logs
        overall_success = True
        
        self._log("=== Physics Checker Results ===")
        
        # Run each test
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        # Summary
        self._log(f"\n=== Summary ===")
        self._log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs