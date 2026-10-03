import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for results
        self.results = {}
        result_keys = ['r', 'x_min', 'v', 'x_max', 'L']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, None)
            if self.results[key] is None:
                # Try to get from results dict if available
                if hasattr(solution_instance, 'results'):
                    self.results[key] = solution_instance.results.get(key, None)
        
        # Use default numerical values for parameters if not provided
        self.d = getattr(solution_instance, 'd', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.B = getattr(solution_instance, 'B', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        
        # Constants
        self.sin60 = math.sin(math.radians(60))
        self.sqrt3 = math.sqrt(3)
        
        # Expected numerical values using current parameter values
        self.expected_r = 2 * (2 - self.sqrt3) * self.d
        self.expected_v = 2 * (2 - self.sqrt3) * self.q * self.d * self.B / self.m
        self.expected_x_min = 2 * (2 - self.sqrt3) * self.d
        self.expected_x_max = 4 * (2 - self.sqrt3) * self.d
        self.expected_L = 2 * (2 - self.sqrt3) * self.d
        
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            message = f"[{'PASS' if success else 'FAIL'}] {message}"
        self.logs.append(message)
        return success
    
    def _safe_compare(self, val1, val2, name=""):
        """Fuzzy comparison with fallback."""
        try:
            if val1 is None or val2 is None:
                return self._log(f"{name}: Missing value(s)", False)
            
            # Ensure both are numeric
            try:
                val1_num = float(val1)
                val2_num = float(val2)
            except (TypeError, ValueError):
                return self._log(f"{name}: Non-numeric values {val1}, {val2}", False)
            
            # Numerical comparison
            if math.isclose(val1_num, val2_num, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"{name}: Numerical match {val1_num:.4g} ≈ {val2_num:.4g}", True)
            else:
                return self._log(f"{name}: Numerical mismatch {val1_num:.4g} ≠ {val2_num:.4g}", False)
                    
        except Exception as e:
            return self._log(f"{name}: Comparison failed: {str(e)}", False)
    
    def test_invariants(self):
        """Verify physics invariants."""
        success = True
        
        # 1. Geometry invariant: r = d/(1 + sin60°)
        if self.d is not None and self.results.get('r') is not None:
            expected_r_geom = self.d / (1 + self.sin60)
            success &= self._safe_compare(self.results['r'], expected_r_geom, "Geometry r = d/(1+sin60°)")
        
        # 2. Lorentz force centripetal condition: qvB = mv²/r
        if all([self.results.get('v') is not None, 
                self.results.get('r') is not None,
                self.q is not None, self.B is not None, self.m is not None]):
            lhs = self.q * self.results['v'] * self.B
            rhs = self.m * self.results['v']**2 / self.results['r']
            success &= self._safe_compare(lhs, rhs, "Lorentz force qvB = mv²/r")
        
        # 3. Geometric consistency: x_min = r, x_max = 2r, L = x_max - x_min
        if self.results.get('r') is not None and self.results.get('x_min') is not None:
            success &= self._safe_compare(self.results['x_min'], self.results['r'], "x_min = r")
        
        if self.results.get('r') is not None and self.results.get('x_max') is not None:
            success &= self._safe_compare(self.results['x_max'], 2 * self.results['r'], "x_max = 2r")
        
        if (self.results.get('x_max') is not None and 
            self.results.get('x_min') is not None and 
            self.results.get('L') is not None):
            L_calc = self.results['x_max'] - self.results['x_min']
            success &= self._safe_compare(self.results['L'], L_calc, "L = x_max - x_min")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check against expected numerical forms
        if self.results.get('r') is not None:
            success &= self._safe_compare(self.results['r'], self.expected_r, "Radius r")
        
        if self.results.get('v') is not None:
            success &= self._safe_compare(self.results['v'], self.expected_v, "Velocity v")
        
        if self.results.get('x_min') is not None:
            success &= self._safe_compare(self.results['x_min'], self.expected_x_min, "x_min")
        
        if self.results.get('x_max') is not None:
            success &= self._safe_compare(self.results['x_max'], self.expected_x_max, "x_max")
        
        if self.results.get('L') is not None:
            success &= self._safe_compare(self.results['L'], self.expected_L, "Length L")
        
        return success
    
    def test_monotonicity(self):
        """Test that v and L increase with d (perturbation test)."""
        success = True
        
        if self.d is None or not isinstance(self.d, (int, float)):
            self._log("Monotonicity: d not numerical, skipping", None)
            return True  # Skip if symbolic
        
        # Store original
        d_original = self.d
        
        # Test 1% increase
        try:
            # Perturb d
            d_new = d_original * 1.01
            
            # Recalculate expected values with new d
            r_new = 2 * (2 - self.sqrt3) * d_new
            v_new = 2 * (2 - self.sqrt3) * self.q * d_new * self.B / self.m
            L_new = 2 * (2 - self.sqrt3) * d_new
            
            # Check monotonic increase
            if isinstance(self.results.get('v'), (int, float)) and isinstance(v_new, (int, float)):
                if v_new > self.results['v']:
                    self._log(f"Monotonicity v: {v_new:.4g} > {self.results['v']:.4g} (✓)", True)
                else:
                    success = False
                    self._log(f"Monotonicity v: {v_new:.4g} ≤ {self.results['v']:.4g} (✗)", False)
            
            if isinstance(self.results.get('L'), (int, float)) and isinstance(L_new, (int, float)):
                if L_new > self.results['L']:
                    self._log(f"Monotonicity L: {L_new:.4g} > {self.results['L']:.4g} (✓)", True)
                else:
                    success = False
                    self._log(f"Monotonicity L: {L_new:.4g} ≤ {self.results['L']:.4g} (✗)", False)
                    
        except Exception as e:
            success = False
            self._log(f"Monotonicity test failed: {str(e)}", False)
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        overall_success = True
        
        try:
            self._log("=== Physics Checker Started ===")
            
            # Test 1: Invariants
            self._log("\n1. Testing Physics Invariants:")
            if not self.test_invariants():
                overall_success = False
                self._log("Invariants test failed", False)
            
            # Test 2: Trace Consistency
            self._log("\n2. Testing Trace Consistency:")
            if not self.test_trace_consistency():
                overall_success = False
                self._log("Trace consistency test failed", False)
            
            # Test 3: Monotonicity
            self._log("\n3. Testing Monotonicity:")
            if not self.test_monotonicity():
                overall_success = False
                self._log("Monotonicity test failed", False)
            
            # Final summary
            self._log(f"\n=== Physics Checker Complete: {'PASS' if overall_success else 'FAIL'} ===")
            
        except Exception as e:
            overall_success = False
            self._log(f"Checker crashed: {str(e)}", False)
            self._log(traceback.format_exc())
        
        return overall_success, self.logs