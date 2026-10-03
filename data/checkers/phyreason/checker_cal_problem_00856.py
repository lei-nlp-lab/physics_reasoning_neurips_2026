import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Try to extract input parameters from various sources
        # First try: get from solution_instance attributes
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.B = getattr(solution_instance, 'B', None)
        self.g = getattr(solution_instance, 'g', None)
        self.h = getattr(solution_instance, 'h', None)
        
        # If not found, try to get from results dictionary
        if any(x is None for x in [self.m, self.q, self.B, self.g, self.h]):
            self.results = getattr(solution_instance, 'results', {})
            # h is in results
            if self.h is None:
                self.h = self.results.get('h')
            # For other parameters, use default values from problem context
            # The problem uses m=1, g=9.8, q=1, B=1 for numerical evaluation
            if self.m is None:
                self.m = 1.0
            if self.g is None:
                self.g = 9.8
            if self.q is None:
                self.q = 1.0
            if self.B is None:
                self.B = 1.0
        
        # Calculate derived quantities
        self.M = 4 * self.m
        self.v0 = self.m * self.g / (self.q * self.B)
        
        # Extract all expected quantities from results
        self.H = self.results.get('H')
        self.x = self.results.get('x')
        self.v_2 = self.results.get('v_2')
        self.KE_before = self.results.get('KE_before')
        self.P_before = self.results.get('P_before')
        self.v_1 = self.results.get('v_1')
        self.v_linear = self.results.get('v_linear')
        self.F_L = self.results.get('F_L')
        self.v = self.results.get('v')
        self.R = self.results.get('R')
        self.h_res = self.results.get('h')
        self.P_after = self.results.get('P_after')
        self.F_c = self.results.get('F_c')
        self.KE_after = self.results.get('KE_after')
        self.v_circular = self.results.get('v_circular')
        
        # Store logs for reporting
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        
    def _check_numeric(self, computed, expected, name, tol=1e-2):
        """Fuzzy numeric comparison with tolerance"""
        if computed is None or expected is None:
            self._log(f"{name}: Missing data", False)
            return False
        
        # Handle division by zero in expected
        if isinstance(expected, (int, float)) and expected == 0 and computed == 0:
            self._log(f"{name}: Both zero", True)
            return True
            
        if math.isclose(computed, expected, rel_tol=tol, abs_tol=tol):
            self._log(f"{name}: {computed:.4f} ≈ {expected:.4f}", True)
            return True
        else:
            self._log(f"{name}: {computed:.4f} ≠ {expected:.4f}", False)
            return False
    
    def test_invariants(self):
        """Verify fundamental physics invariants"""
        success = True
        
        if None in [self.m, self.q, self.B, self.g, self.v0, self.v_1, self.v_2]:
            self._log(f"Missing parameters: m={self.m}, q={self.q}, B={self.B}, g={self.g}, v0={self.v0}, v_1={self.v_1}, v_2={self.v_2}", False)
            return False
        
        # 1. Conservation of momentum (elastic collision)
        p_before_calc = self.M * self.v0
        p_after_calc = self.M * self.v_1 + self.m * self.v_2
        if not self._check_numeric(p_before_calc, p_after_calc, "Momentum Conservation"):
            success = False
        
        # 2. Conservation of kinetic energy (elastic collision)
        KE_before_calc = 0.5 * self.M * self.v0**2
        KE_after_calc = 0.5 * self.M * self.v_1**2 + 0.5 * self.m * self.v_2**2
        if not self._check_numeric(KE_before_calc, KE_after_calc, "Kinetic Energy Conservation"):
            success = False
        
        # 3. Velocity decomposition for ball Q
        if self.v_linear is not None and self.v_circular is not None:
            v2_decomposed = self.v_linear + self.v_circular
            if not self._check_numeric(self.v_2, v2_decomposed, "Velocity Decomposition (v₂ = v_linear + v_circular)"):
                success = False
        
        # 4. Lorentz force = Centripetal force
        if all(x is not None for x in [self.q, self.B, self.v_circular, self.F_L, self.m, self.R, self.F_c]):
            F_L_calc = self.q * self.v_circular * self.B
            # Avoid division by zero for F_c calculation
            if self.R != 0:
                F_c_calc = self.m * self.v_circular**2 / self.R
            else:
                F_c_calc = float('inf')
            
            if not self._check_numeric(F_L_calc, self.F_L, "Lorentz Force Calculation"):
                success = False
            if self.R != 0 and not self._check_numeric(F_c_calc, self.F_c, "Centripetal Force Calculation"):
                success = False
            if self.R != 0 and not self._check_numeric(F_L_calc, F_c_calc, "Lorentz = Centripetal"):
                success = False
        
        # 5. Radius from circular motion
        if all(x is not None for x in [self.m, self.v_circular, self.q, self.B, self.R]):
            R_calc = self.m * self.v_circular / (self.q * self.B)
            if not self._check_numeric(self.R, R_calc, "Radius from Circular Motion"):
                success = False
        
        # 6. Maximum height geometry
        if self.H is not None and self.R is not None and self.h is not None:
            H_calc = self.h + 2 * self.R
            if not self._check_numeric(self.H, H_calc, "Maximum Height (H = h + 2R)"):
                success = False
        
        # 7. Velocity at maximum height
        if all(x is not None for x in [self.v, self.v_linear, self.v_circular]):
            v_max_height_calc = self.v_linear - self.v_circular
            if not self._check_numeric(self.v, v_max_height_calc, "Velocity at Max Height"):
                success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        success = True
        
        if None in [self.m, self.q, self.B, self.g, self.h, self.v0]:
            self._log("Missing parameters for trace consistency", False)
            return False
        
        # Expected numeric expressions from steps analysis
        # Step 3: v₁ = 3mg/(5qB)
        v1_expected = 3 * self.m * self.g / (5 * self.q * self.B)
        if self.v_1 is not None and not self._check_numeric(self.v_1, v1_expected, "v₁ from Step 3"):
            success = False
        
        # Step 4: v₂ = 8mg/(5qB)
        v2_expected = 8 * self.m * self.g / (5 * self.q * self.B)
        if self.v_2 is not None and not self._check_numeric(self.v_2, v2_expected, "v₂ from Step 4"):
            success = False
        
        # Step 7: x = (3m/(5qB))√(2gh)
        x_expected = (3 * self.m / (5 * self.q * self.B)) * math.sqrt(2 * self.g * self.h)
        if self.x is not None and not self._check_numeric(self.x, x_expected, "x from Step 7"):
            success = False
        
        # Step 10: R = 3m²g/(5q²B²)
        R_expected = 3 * self.m**2 * self.g / (5 * self.q**2 * self.B**2)
        if self.R is not None and not self._check_numeric(self.R, R_expected, "R from Step 10"):
            success = False
        
        # Step 11: H = h + 6m²g/(5q²B²)
        H_expected = self.h + 6 * self.m**2 * self.g / (5 * self.q**2 * self.B**2)
        if self.H is not None and not self._check_numeric(self.H, H_expected, "H from Step 11"):
            success = False
        
        # Step 12: v = 2mg/(5qB)
        v_expected = 2 * self.m * self.g / (5 * self.q * self.B)
        if self.v is not None and not self._check_numeric(self.v, v_expected, "v from Step 12"):
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb input parameters and check expected trends"""
        success = True
        
        if None in [self.m, self.q, self.B, self.g, self.h, self.H, self.R]:
            self._log("Missing parameters for monotonicity test", False)
            return False
        
        # Test 1: Increase mass m → increase maximum height H
        m_perturbed = self.m * 1.1  # 10% increase
        R_perturbed = 3 * m_perturbed**2 * self.g / (5 * self.q**2 * self.B**2)
        H_perturbed = self.h + 2 * R_perturbed
        
        # H should increase with mass
        if H_perturbed > self.H:
            self._log(f"Monotonicity: H increases with mass ({self.H:.4f} → {H_perturbed:.4f})", True)
        else:
            self._log(f"Monotonicity: H should increase with mass", False)
            success = False
        
        # Test 2: Increase magnetic field B → decrease radius R
        B_perturbed = self.B * 1.1  # 10% increase
        R_B_perturbed = 3 * self.m**2 * self.g / (5 * self.q**2 * B_perturbed**2)
        
        # R should decrease with B
        if R_B_perturbed < self.R:
            self._log(f"Monotonicity: R decreases with B ({self.R:.4f} → {R_B_perturbed:.4f})", True)
        else:
            self._log(f"Monotonicity: R should decrease with B", False)
            success = False
        
        # Test 3: Increase charge q → decrease radius R
        q_perturbed = self.q * 1.1  # 10% increase
        R_q_perturbed = 3 * self.m**2 * self.g / (5 * q_perturbed**2 * self.B**2)
        
        # R should decrease with q
        if R_q_perturbed < self.R:
            self._log(f"Monotonicity: R decreases with q ({self.R:.4f} → {R_q_perturbed:.4f})", True)
        else:
            self._log(f"Monotonicity: R should decrease with q", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        
        self._log("=== Physics Checker Started ===", True)
        
        # Run individual tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary
        success = test1 and test2 and test3
        self._log(f"=== Summary: {'All tests passed' if success else 'Some tests failed'} ===", success)
        self._log(f"  Invariants: {'✓' if test1 else '✗'}", test1)
        self._log(f"  Trace Consistency: {'✓' if test2 else '✗'}", test2)
        self._log(f"  Monotonicity: {'✓' if test3 else '✗'}", test3)
        
        return success, self.logs