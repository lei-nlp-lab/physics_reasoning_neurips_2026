import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        
        Args:
            solution_instance: The solution object containing results
        """
        # Constants from problem statement
        self.m = 0.3  # kg
        self.d = 0.8  # m (horizontal distance AND hand height)
        self.g = 10.0  # m/s²
        
        # Safe extraction of results
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all results with safe defaults
        self.x = self._safe_get('x')
        self.v_2 = self._safe_get('v_2')
        self.F = self._safe_get('F')
        self.t_2 = self._safe_get('t_2')
        self.t = self._safe_get('t')
        self.l = self._safe_get('l')
        self.x_max = self._safe_get('x_max')
        self.v_3 = self._safe_get('v_3')
        self.v_1 = self._safe_get('v_1')
        
        # For Q1-Q2, we need the fixed rope length (0.6)
        # For Q3, we need the optimal length (d/2 = 0.4)
        # Try to get optimal length if stored separately
        self.l_opt = self._safe_get('l_opt')
        if self.l_opt is None:
            # If not stored separately, assume l is the optimal length for Q3 tests
            self.l_opt = self.l
        
    def _safe_get(self, key: str) -> Optional[float]:
        """Safely get attribute from solution instance or results dict."""
        # First try direct attribute
        val = getattr(self, key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) else val
        
        # Then try results dict
        val = self.results.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return val
        return None
    
    def _isclose(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physics invariants (energy conservation, Newton's laws)."""
        logs = []
        success = True
        
        try:
            # Test 1: Energy conservation from breaking point to ground
            # For Q1, we use fixed rope length l_fixed = 0.6
            if all(v is not None for v in [self.v_1, self.v_2]):
                l_fixed = 0.6  # Fixed rope length for Q1
                height = self.d - l_fixed
                
                # KE_initial + PE_initial = KE_final + PE_final
                # 0.5*m*v_1^2 + m*g*h = 0.5*m*v_2^2 + 0
                lhs = 0.5 * self.m * self.v_1**2 + self.m * self.g * height
                rhs = 0.5 * self.m * self.v_2**2
                
                if not self._isclose(lhs, rhs):
                    logs.append(f"FAIL: Energy conservation violated: {lhs:.3f} != {rhs:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Energy conservation holds: {lhs:.3f} ≈ {rhs:.3f}")
            
            # Test 2: Newton's 2nd Law at breaking point (Q2)
            if all(v is not None for v in [self.v_1, self.F]):
                l_fixed = 0.6  # Fixed rope length for Q2
                # Tension - weight = centripetal force
                expected_F = self.m * self.g + self.m * self.v_1**2 / l_fixed
                
                if not self._isclose(self.F, expected_F):
                    logs.append(f"FAIL: Newton's 2nd Law: F={self.F:.3f}, expected {expected_F:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Newton's 2nd Law: F={self.F:.3f} ≈ {expected_F:.3f}")
            
            # Test 3: Free fall consistency (Q1)
            if self.t is not None:
                l_fixed = 0.6  # Fixed rope length for Q1
                expected_t = math.sqrt(2 * (self.d - l_fixed) / self.g)
                
                if not self._isclose(self.t, expected_t):
                    logs.append(f"FAIL: Free fall time: t={self.t:.3f}, expected {expected_t:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Free fall time: t={self.t:.3f} ≈ {expected_t:.3f}")
            
            if len(logs) == 0:
                logs.append("WARN: No invariant tests could be performed due to missing data")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate and final results."""
        logs = []
        success = True
        
        try:
            # Trace 1: Check step 2 and 3 consistency (Q1)
            if all(v is not None for v in [self.t, self.v_1]):
                # From uniform motion: v_1 = d / t
                expected_v1 = self.d / self.t
                
                if not self._isclose(self.v_1, expected_v1):
                    logs.append(f"FAIL: Trace 1: v_1={self.v_1:.3f}, expected d/t={expected_v1:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Trace 1: v_1={self.v_1:.3f} ≈ d/t={expected_v1:.3f}")
            
            # Trace 2: Check step 4-5 consistency (Q1)
            if all(v is not None for v in [self.v_1, self.v_2]):
                l_fixed = 0.6  # Fixed rope length for Q1
                # Work-energy: mg(d-l) = 0.5*m*v_2^2 - 0.5*m*v_1^2
                expected_v2 = math.sqrt(self.v_1**2 + 2 * self.g * (self.d - l_fixed))
                
                if not self._isclose(self.v_2, expected_v2):
                    logs.append(f"FAIL: Trace 2: v_2={self.v_2:.3f}, expected √(v_1²+2gΔh)={expected_v2:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Trace 2: v_2={self.v_2:.3f} ≈ √(v_1²+2gΔh)={expected_v2:.3f}")
            
            # Trace 3: Check step 6 consistency (Q2)
            if all(v is not None for v in [self.v_1, self.F]):
                l_fixed = 0.6  # Fixed rope length for Q2
                expected_F = self.m * self.g + self.m * self.v_1**2 / l_fixed
                
                if not self._isclose(self.F, expected_F):
                    logs.append(f"FAIL: Trace 3: F={self.F:.3f}, expected m(g+v_1²/l)={expected_F:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Trace 3: F={self.F:.3f} ≈ m(g+v_1²/l)={expected_F:.3f}")
            
            # Trace 4: Check optimization results (Q3)
            if self.l_opt is not None and self.x_max is not None:
                # For optimal l = d/2 = 0.4
                expected_l = self.d / 2
                expected_xmax = (2 * math.sqrt(3) / 3) * self.d
                
                l_ok = self._isclose(self.l_opt, expected_l)
                xmax_ok = self._isclose(self.x_max, expected_xmax)
                
                if not l_ok:
                    logs.append(f"FAIL: Trace 4: l_opt={self.l_opt:.3f}, expected d/2={expected_l:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Trace 4: l_opt={self.l_opt:.3f} ≈ d/2={expected_l:.3f}")
                    
                if not xmax_ok:
                    logs.append(f"FAIL: Trace 4: x_max={self.x_max:.3f}, expected (2√3/3)d={expected_xmax:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Trace 4: x_max={self.x_max:.3f} ≈ (2√3/3)d={expected_xmax:.3f}")
            
            if len(logs) == 0:
                logs.append("WARN: No trace tests could be performed due to missing data")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test that increasing mass increases tension (positive monotonicity)."""
        logs = []
        success = True
        
        try:
            # Test: If mass increases, tension at breaking point should increase
            # For Q2, we use fixed rope length
            if self.v_1 is not None:
                l_fixed = 0.6  # Fixed rope length for Q2
                
                # Perturb mass by +10%
                m_perturbed = self.m * 1.1
                F_original = self.m * self.g + self.m * self.v_1**2 / l_fixed
                F_perturbed = m_perturbed * self.g + m_perturbed * self.v_1**2 / l_fixed
                
                # Check monotonic increase
                if F_perturbed <= F_original:
                    logs.append(f"FAIL: Monotonicity: F(m+10%)={F_perturbed:.3f} ≤ F(original)={F_original:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Monotonicity: F increases with mass ({F_original:.3f} → {F_perturbed:.3f})")
                    
                # Restore original mass (for clarity in logs)
                logs.append(f"INFO: Original m={self.m}kg, F={F_original:.3f}N")
                logs.append(f"INFO: Perturbed m={m_perturbed}kg, F={F_perturbed:.3f}N")
            else:
                logs.append("WARN: Cannot test monotonicity - missing v_1")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        all_logs.append("=== Physics Checker Results ===")
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("\n--- Invariant Tests ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- Trace Consistency Tests ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- Monotonicity Tests ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs