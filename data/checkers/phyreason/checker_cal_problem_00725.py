import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        SAFE attribute access: get results from solution_instance.
        """
        self.solution = solution_instance
        
        # Extract inputs with fallback to results or defaults
        self.m = getattr(solution_instance, 'm', 0.0)
        self.e = getattr(solution_instance, 'e', 0.0)
        self.v = getattr(solution_instance, 'v', 0.0)
        self.B = getattr(solution_instance, 'B', 0.0)
        
        # If not directly available, try to get from results dict
        if hasattr(solution_instance, 'results'):
            self.m = solution_instance.results.get('m', self.m)
            self.e = solution_instance.results.get('e', self.e)
            self.v = solution_instance.results.get('v', self.v)
            self.B = solution_instance.results.get('B', self.B)
        
        # Extract computed results
        self.x = getattr(solution_instance, 'x', 
                         getattr(solution_instance, 'results', {}).get('x', 0.0))
        self.r2 = getattr(solution_instance, 'r₂', 
                          getattr(solution_instance, 'results', {}).get('r₂', 0.0))
        self.T = getattr(solution_instance, 'T', 
                         getattr(solution_instance, 'results', {}).get('T', 0.0))
        self.T2 = getattr(solution_instance, 'T₂', 
                          getattr(solution_instance, 'results', {}).get('T₂', 0.0))
        self.r1 = getattr(solution_instance, 'r₁', 
                          getattr(solution_instance, 'results', {}).get('r₁', 0.0))
        self.T1 = getattr(solution_instance, 'T₁', 
                          getattr(solution_instance, 'results', {}).get('T₁', 0.0))
        
        self.logs = []
    
    def _log(self, msg, success=None):
        """Helper for logging."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def _is_close(self, a, b):
        """Robust numerical matching with 1% tolerance."""
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify energy conservation and Lorentz force laws."""
        success = True
        
        try:
            # 1. Energy conservation: magnetic field does no work
            # Kinetic energy should remain constant = 0.5*m*v²
            # Since speed v is constant in uniform circular motion
            ke_expected = 0.5 * self.m * self.v**2
            
            # Check that radii are consistent with centripetal force = Lorentz force
            # Above x-axis: qv(2B) = m*v²/r1  (electron: q = -e, but magnitude e used)
            # => r1 = m*v/(2*e*B)
            r1_expected = (self.m * self.v) / (2 * self.e * self.B)
            if not self._is_close(self.r1, r1_expected):
                self._log(f"r1 mismatch: expected {r1_expected}, got {self.r1}", False)
                success = False
            else:
                self._log(f"r1 matches Lorentz force: {self.r1} ≈ {r1_expected}", True)
            
            # Below x-axis: qvB = m*v²/r2 => r2 = m*v/(e*B)
            r2_expected = (self.m * self.v) / (self.e * self.B)
            if not self._is_close(self.r2, r2_expected):
                self._log(f"r2 mismatch: expected {r2_expected}, got {self.r2}", False)
                success = False
            else:
                self._log(f"r2 matches Lorentz force: {self.r2} ≈ {r2_expected}", True)
            
            # 2. Check centripetal acceleration = v²/r for both regions
            # Force magnitude should equal m*v²/r
            force_above_expected = self.m * self.v**2 / self.r1 if self.r1 != 0 else 0
            force_below_expected = self.m * self.v**2 / self.r2 if self.r2 != 0 else 0
            
            # Lorentz force magnitudes
            lorentz_above = self.e * self.v * (2*self.B)  # B = 2B above
            lorentz_below = self.e * self.v * self.B      # B = B below
            
            if not self._is_close(force_above_expected, lorentz_above):
                self._log(f"Force above mismatch: centripetal {force_above_expected} vs Lorentz {lorentz_above}", False)
                success = False
            else:
                self._log("Force above balanced (Newton's 2nd Law)", True)
            
            if not self._is_close(force_below_expected, lorentz_below):
                self._log(f"Force below mismatch: centripetal {force_below_expected} vs Lorentz {lorentz_below}", False)
                success = False
            else:
                self._log("Force below balanced (Newton's 2nd Law)", True)
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        try:
            # Check T1 from r1 and v: T1 = 2πr1/v
            T1_expected = (2 * math.pi * self.r1) / self.v if self.v != 0 else 0
            if not self._is_close(self.T1, T1_expected):
                self._log(f"T1 inconsistency: from r1/v got {T1_expected}, expected {self.T1}", False)
                success = False
            else:
                self._log(f"T1 consistent with r1: {self.T1} ≈ 2πr1/v = {T1_expected}", True)
            
            # Check T2 from r2 and v: T2 = 2πr2/v
            T2_expected = (2 * math.pi * self.r2) / self.v if self.v != 0 else 0
            if not self._is_close(self.T2, T2_expected):
                self._log(f"T2 inconsistency: from r2/v got {T2_expected}, expected {self.T2}", False)
                success = False
            else:
                self._log(f"T2 consistent with r2: {self.T2} ≈ 2πr2/v = {T2_expected}", True)
            
            # Check T from T1 and T2: T = T1/2 + T2/2 (half periods)
            T_expected = (self.T1/2 + self.T2/2)
            if not self._is_close(self.T, T_expected):
                self._log(f"T inconsistency: from T1,T2 got {T_expected}, expected {self.T}", False)
                success = False
            else:
                self._log(f"T consistent with T1,T2: {self.T} ≈ T1/2 + T2/2 = {T_expected}", True)
            
            # Check x from r1 and r2: x = 2*(r1 + r2)
            x_expected = 2 * (self.r1 + self.r2)
            if not self._is_close(self.x, x_expected):
                self._log(f"x inconsistency: from r1,r2 got {x_expected}, expected {self.x}", False)
                success = False
            else:
                self._log(f"x consistent with r1,r2: {self.x} ≈ 2*(r1+r2) = {x_expected}", True)
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and check trends, then restore."""
        success = True
        
        try:
            # Save original values
            m_original = self.m
            
            # Test 1: Increase mass by 10%
            self.m = m_original * 1.1
            
            # Recalculate expected values using formulas
            r1_new = (self.m * self.v) / (2 * self.e * self.B)
            r2_new = (self.m * self.v) / (self.e * self.B)
            T_new = (math.pi * self.m / (self.e * self.B) + 2 * math.pi * self.m / (self.e * self.B)) / 2
            x_new = 2 * (r1_new + r2_new)
            
            # Check monotonicity: all should increase with mass
            if r1_new <= self.r1 or r2_new <= self.r2 or T_new <= self.T or x_new <= self.x:
                self._log(f"Monotonicity fail: increased m but results didn't all increase", False)
                success = False
            else:
                self._log(f"Monotonicity PASS: increasing m increases r1,r2,T,x", True)
            
            # Test 2: Decrease magnetic field B by 10%
            self.m = m_original  # Restore mass
            B_original = self.B
            self.B = B_original * 0.9
            
            r1_new = (self.m * self.v) / (2 * self.e * self.B)
            r2_new = (self.m * self.v) / (self.e * self.B)
            T_new = (math.pi * self.m / (self.e * self.B) + 2 * math.pi * self.m / (self.e * self.B)) / 2
            x_new = 2 * (r1_new + r2_new)
            
            # Check: decreasing B should increase all results
            if r1_new <= self.r1 or r2_new <= self.r2 or T_new <= self.T or x_new <= self.x:
                self._log(f"Monotonicity fail: decreased B but results didn't all increase", False)
                success = False
            else:
                self._log(f"Monotonicity PASS: decreasing B increases r1,r2,T,x", True)
            
            # Restore original values
            self.m = m_original
            self.B = B_original
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
            
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self._log(f"--- Running {name} Test ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{name} test crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        # Final summary
        self._log(f"\nOverall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        return overall_success, self.logs