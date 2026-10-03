import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: never access solution_instance.attr directly.
        Use getattr with fallback to results dictionary.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants from problem statement
        self.m = 10.0  # kg
        self.g = 10.0  # m/s²
        self.R = 2.5   # m
        self.r = 1.5   # m
        self.h_pool = 5.0  # m (height of water above horizontal track)
        
        # Extract results safely with fallbacks
        self.F = self._safe_get('F', 0.0)
        self.v_E2 = self._safe_get('v_E2', 0.0)
        self.v_E = self._safe_get('v_E', 0.0)  # Now expecting numeric, not string
        self.v_P = self._safe_get('v_P', 0.0)
        self.x_2 = self._safe_get('x_2', 0.0)
        self.h = self._safe_get('h', 0.0)  # Now expecting numeric, not string
        self.h_1 = self._safe_get('h_1', 0.0)
        self.x_1 = self._safe_get('x_1', 0.0)
        self.v_E1 = self._safe_get('v_E1', 0.0)
        
        # For range checks, also look for h_min/h_max if available
        self.h_min = self._safe_get('h_min', 0.0)
        self.h_max = self._safe_get('h_max', 0.0)
        
    def _safe_get(self, key: str, default: Any) -> Any:
        """Safely get value from results with multiple fallbacks."""
        # First try direct key in results
        val = self.results.get(key, default)
        # If it's a string that might be numeric, try to convert
        if isinstance(val, str):
            try:
                # Handle fractions like "1000/3"
                if '/' in val:
                    parts = val.split('/')
                    return float(parts[0]) / float(parts[1])
                # Regular number
                return float(val)
            except (ValueError, TypeError):
                return val
        return val
    
    def _isclose(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self) -> List[str]:
        """Test fundamental physics invariants."""
        logs = []
        
        # 1. Velocity at top of track A (v_P = sqrt(gR))
        v_P_expected = math.sqrt(self.g * self.R)
        if not self._isclose(self.v_P, v_P_expected):
            logs.append(f"FAIL: v_P = {self.v_P}, expected sqrt(gR) = {v_P_expected:.2f}")
        else:
            logs.append(f"PASS: v_P = sqrt(gR) = {v_P_expected:.2f} m/s")
        
        # 2. Energy conservation: from A to P (minimum height condition)
        # mgh_1 = mg(2R) + 0.5*m*v_P^2
        h_1_expected = 2*self.R + self.v_P**2/(2*self.g)
        if not self._isclose(self.h_1, h_1_expected):
            logs.append(f"FAIL: h_1 = {self.h_1}, expected 2R + v_P²/(2g) = {h_1_expected:.2f}")
        else:
            logs.append(f"PASS: h_1 = 2R + v_P²/(2g) = {h_1_expected:.2f} m")
        
        # 3. Force at top of track B (F = 5mg(R-r)/r)
        F_expected = 5 * self.m * self.g * (self.R - self.r) / self.r
        if not self._isclose(self.F, F_expected):
            logs.append(f"FAIL: F = {self.F}, expected 5mg(R-r)/r = {F_expected:.2f}")
        else:
            logs.append(f"PASS: F = 5mg(R-r)/r = {F_expected:.2f} N")
        
        # 4. Projectile motion time (t = sqrt(2h_pool/g))
        t = math.sqrt(2 * self.h_pool / self.g)
        
        # 5. Horizontal velocities from projectile ranges
        v_E1_expected = self.x_1 / t
        if not self._isclose(self.v_E1, v_E1_expected):
            logs.append(f"FAIL: v_E1 = {self.v_E1}, expected x_1/t = {v_E1_expected:.2f}")
        else:
            logs.append(f"PASS: v_E1 = x_1/t = {v_E1_expected:.2f} m/s")
        
        v_E2_expected = self.x_2 / t
        if not self._isclose(self.v_E2, v_E2_expected):
            logs.append(f"FAIL: v_E2 = {self.v_E2}, expected x_2/t = {v_E2_expected:.2f}")
        else:
            logs.append(f"PASS: v_E2 = x_2/t = {v_E2_expected:.2f} m/s")
        
        # 6. Energy conservation from A to E: v_E = sqrt(2gh)
        # The actual h range should be determined by v_E1 and v_E2
        h_min_from_vE1 = self.v_E1**2 / (2*self.g)
        h_max_from_vE2 = self.v_E2**2 / (2*self.g)
        
        # But also must satisfy h >= 6.25 (from track A clearance)
        h_min_constrained = max(h_min_from_vE1, 6.25)
        
        # Check if we have h_min and h_max values
        if hasattr(self, 'h_min') and hasattr(self, 'h_max'):
            if not (self._isclose(self.h_min, h_min_constrained) and 
                    self._isclose(self.h_max, h_max_from_vE2)):
                logs.append(f"FAIL: h range = ({self.h_min}, {self.h_max}), "
                           f"expected ({h_min_constrained:.2f}, {h_max_from_vE2:.2f})")
            else:
                logs.append(f"PASS: h range consistent with energy conservation")
        else:
            # Fallback: check the single h value
            if not self._isclose(self.h, h_min_constrained):
                logs.append(f"WARN: h = {self.h}, expected min h = {h_min_constrained:.2f}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Check consistency between intermediate and final results."""
        logs = []
        
        # Check that v_E1 and v_E2 are consistent with expected values
        # v_E should be between v_E1 and v_E2
        if not (self._isclose(self.v_E, self.v_E1) or 
                (self.v_E1 <= self.v_E <= self.v_E2)):
            logs.append(f"WARN: v_E = {self.v_E} not in range [{self.v_E1}, {self.v_E2}]")
        else:
            logs.append(f"PASS: v_E consistent with range")
        
        # Check that h_1 equals minimum h from range
        if hasattr(self, 'h_min'):
            if not self._isclose(self.h_1, self.h_min):
                logs.append(f"FAIL: h_1 = {self.h_1} doesn't match h_min = {self.h_min}")
            else:
                logs.append(f"PASS: h_1 equals minimum h in range")
        else:
            if not self._isclose(self.h_1, self.h):
                logs.append(f"WARN: h_1 = {self.h_1} doesn't match h = {self.h}")
        
        # Check that F = 333.33 N (1000/3)
        if not self._isclose(self.F, 1000/3):
            logs.append(f"FAIL: F = {self.F} doesn't match expected 1000/3 ≈ 333.33")
        else:
            logs.append(f"PASS: F = 1000/3 N ≈ 333.33 N")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """
        Perturb mass and check that force F scales linearly.
        Restore original values after test.
        """
        logs = []
        
        # Save original mass
        original_m = self.m
        
        # Test 1% increase
        self.m = original_m * 1.01
        F_new = 5 * self.m * self.g * (self.R - self.r) / self.r
        F_original = 5 * original_m * self.g * (self.R - self.r) / self.r
        
        ratio_expected = 1.01  # 1% increase in mass
        ratio_actual = F_new / F_original
        
        if not self._isclose(ratio_actual, ratio_expected):
            logs.append(f"FAIL: F doesn't scale linearly with mass. "
                       f"Ratio = {ratio_actual:.4f}, expected {ratio_expected:.4f}")
        else:
            logs.append(f"PASS: F scales linearly with mass (ratio = {ratio_actual:.4f})")
        
        # Restore original mass
        self.m = original_m
        
        # Test monotonicity using numeric derivatives
        # F = 5mg(R-r)/r
        # dF/dm = 5g(R-r)/r > 0
        dF_dm = 5 * self.g * (self.R - self.r) / self.r
        if dF_dm <= 0:
            logs.append(f"FAIL: dF/dm should be positive, got {dF_dm}")
        
        # dF/dR = 5mg/r > 0
        dF_dR = 5 * self.m * self.g / self.r
        if dF_dR <= 0:
            logs.append(f"FAIL: dF/dR should be positive, got {dF_dR}")
        
        # dF/dr = -5mg(R-r)/r² < 0
        dF_dr = -5 * self.m * self.g * (self.R - self.r) / (self.r**2)
        if dF_dr >= 0:
            logs.append(f"FAIL: dF/dr should be negative, got {dF_dr}")
        
        if dF_dm > 0 and dF_dR > 0 and dF_dr < 0:
            logs.append(f"PASS: Numeric monotonicity checks passed")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run invariant tests
            logs = self.test_invariants()
            all_logs.append("=== Invariant Tests ===")
            all_logs.extend(logs)
            if any("FAIL" in log for log in logs):
                success = False
            
            # Run trace consistency tests
            logs = self.test_trace_consistency()
            all_logs.append("\n=== Trace Consistency Tests ===")
            all_logs.extend(logs)
            if any("FAIL" in log for log in logs):
                success = False
            
            # Run monotonicity tests
            logs = self.test_monotonicity()
            all_logs.append("\n=== Monotonicity Tests ===")
            all_logs.extend(logs)
            if any("FAIL" in log for log in logs):
                success = False
            
        except Exception as e:
            all_logs.append(f"\nERROR during testing: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs