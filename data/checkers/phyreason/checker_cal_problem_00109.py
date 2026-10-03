import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.q_A = self.results.get('q_A', 0.0)
        self.q_B = self.results.get('q_B', 0.0)
        self.q_C = self.results.get('q_C', 0.0)
        
        # Geometry parameters (side length set to 2 for numerical stability)
        self.L = 2.0  # side length of equilateral triangle
        self.k = 1.0  # Coulomb constant (set to 1 for scaling)
        
        # Derived coordinates
        self.sqrt3 = math.sqrt(3)
        self.A = (0.0, 0.0)
        self.B = (self.L, 0.0)
        self.C = (self.L/2, self.sqrt3 * self.L/2)
        self.M = (self.L/2, 0.0)  # midpoint of AB
        self.N = (3*self.L/4, self.sqrt3 * self.L/4)  # midpoint of BC
        
        # Distance calculations
        self.AM = self._distance(self.A, self.M)
        self.BM = self._distance(self.B, self.M)
        self.CM = self._distance(self.C, self.M)
        self.AN = self._distance(self.A, self.N)
        self.BN = self._distance(self.B, self.N)
        self.CN = self._distance(self.C, self.N)
        
        # Test logs
        self.logs = []
        
    def _distance(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    
    def _electric_field_components(self, q, source, point):
        """Return (Ex, Ey) at point due to charge q at source"""
        dx = point[0] - source[0]
        dy = point[1] - source[1]
        r2 = dx*dx + dy*dy
        r3 = r2 * math.sqrt(r2)
        
        if abs(r3) < 1e-12:
            return (0.0, 0.0)
            
        Ex = self.k * q * dx / r3
        Ey = self.k * q * dy / r3
        return (Ex, Ey)
    
    def _test_condition_M(self):
        """Check electric field at midpoint M is vertically downward"""
        # Calculate net field at M
        Ex_net, Ey_net = 0.0, 0.0
        for q, source in [(self.q_A, self.A), (self.q_B, self.B), (self.q_C, self.C)]:
            Ex, Ey = self._electric_field_components(q, source, self.M)
            Ex_net += Ex
            Ey_net += Ey
        
        # Horizontal component should be zero (within tolerance)
        if not math.isclose(Ex_net, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: At point M, horizontal field Ex = {Ex_net} (expected ~0)")
            return False
        
        # Vertical component should be negative (downward)
        if Ey_net > 0:
            self.logs.append(f"FAIL: At point M, vertical field Ey = {Ey_net} (expected < 0)")
            return False
            
        self.logs.append(f"PASS: At M, Ex≈0, Ey={Ey_net:.3f}<0 (vertically downward)")
        return True
    
    def _test_condition_N(self):
        """Check electric field at midpoint N is vertically upward"""
        # Calculate net field at N
        Ex_net, Ey_net = 0.0, 0.0
        for q, source in [(self.q_A, self.A), (self.q_B, self.B), (self.q_C, self.C)]:
            Ex, Ey = self._electric_field_components(q, source, self.N)
            Ex_net += Ex
            Ey_net += Ey
        
        # Horizontal component should be zero
        if not math.isclose(Ex_net, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: At point N, horizontal field Ex = {Ex_net} (expected ~0)")
            return False
        
        # Vertical component should be positive (upward)
        if Ey_net < 0:
            self.logs.append(f"FAIL: At point N, vertical field Ey = {Ey_net} (expected > 0)")
            return False
            
        self.logs.append(f"PASS: At N, Ex≈0, Ey={Ey_net:.3f}>0 (vertically upward)")
        return True
    
    def _test_charge_equality(self):
        """Check q_A = q_B = q (with q positive)"""
        if not math.isclose(self.q_A, self.q_B, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: q_A = {self.q_A} ≠ q_B = {self.q_B}")
            return False
        
        if self.q_A <= 0 or self.q_B <= 0:
            self.logs.append(f"FAIL: q_A = {self.q_A}, q_B = {self.q_B} should be positive")
            return False
            
        self.logs.append(f"PASS: q_A = q_B = {self.q_A} (positive)")
        return True
    
    def _test_qC_value(self):
        """Check q_C = (3-√3)q/3"""
        expected_qC = (3 - self.sqrt3) * self.q_A / 3
        if not math.isclose(self.q_C, expected_qC, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: q_C = {self.q_C}, expected = {expected_qC:.4f}")
            return False
            
        if self.q_C <= 0:
            self.logs.append(f"FAIL: q_C = {self.q_C} should be positive")
            return False
            
        self.logs.append(f"PASS: q_C = {self.q_C:.4f} ≈ (3-√3)q/3")
        return True
    
    def _test_step4_equation(self):
        """Verify the relationship from step 4: kq/AN² = (√3/3)(kq/BN² - kq_C/CN²)"""
        # Calculate left side
        left = self.k * self.q_A / (self.AN**2)
        
        # Calculate right side
        term1 = self.k * self.q_A / (self.BN**2)
        term2 = self.k * self.q_C / (self.CN**2)
        right = (self.sqrt3/3) * (term1 - term2)
        
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Step 4 equation mismatch: {left:.4f} ≠ {right:.4f}")
            return False
            
        self.logs.append(f"PASS: Step 4 equation verified: {left:.4f} ≈ {right:.4f}")
        return True
    
    def _test_step5_geometry(self):
        """Verify geometric relationships: AN = √3 * BN = √3 * CN"""
        if not math.isclose(self.AN, self.sqrt3 * self.BN, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: AN = {self.AN}, √3*BN = {self.sqrt3 * self.BN}")
            return False
            
        if not math.isclose(self.AN, self.sqrt3 * self.CN, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: AN = {self.AN}, √3*CN = {self.sqrt3 * self.CN}")
            return False
            
        self.logs.append(f"PASS: AN = √3*BN = √3*CN = {self.AN:.3f}")
        return True
    
    def test_invariants(self):
        """Test physical invariants based on superposition principle"""
        success = True
        tests = [
            self._test_condition_M,
            self._test_condition_N,
            self._test_charge_equality,
            self._test_qC_value,
            self._test_step5_geometry,
            self._test_step4_equation
        ]
        
        for test in tests:
            try:
                if not test():
                    success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test.__name__}: {str(e)}")
                success = False
                
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        success = True
        
        # Check that all charges are positive
        if not (self.q_A > 0 and self.q_B > 0 and self.q_C > 0):
            self.logs.append("FAIL: All charges should be positive according to step 3")
            success = False
        else:
            self.logs.append("PASS: All charges positive (step 3 verified)")
        
        # Check that q_C formula yields the correct value
        symbolic_check = (3 - sp.sqrt(3)) * self.q_A / 3
        if not math.isclose(self.q_C, float(symbolic_check), rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: q_C doesn't match symbolic formula: {self.q_C} vs {symbolic_check}")
            success = False
        else:
            self.logs.append(f"PASS: q_C matches symbolic formula (step 6 verified)")
        
        return success
    
    def test_monotonicity(self):
        """Perturb q_A and check effect on field at N"""
        try:
            # Store original
            q_A_orig = self.q_A
            
            # Small perturbation (5%)
            perturbed_q_A = q_A_orig * 1.05
            
            # Calculate Ey at N for original and perturbed
            def calc_Ey_at_N(q_A_val):
                # Recalculate q_C based on perturbed q_A
                q_C_val = (3 - self.sqrt3) * q_A_val / 3
                Ey_net = 0.0
                for q, source in [(q_A_val, self.A), (q_A_val, self.B), (q_C_val, self.C)]:
                    _, Ey = self._electric_field_components(q, source, self.N)
                    Ey_net += Ey
                return Ey_net
            
            Ey_orig = calc_Ey_at_N(q_A_orig)
            Ey_perturbed = calc_Ey_at_N(perturbed_q_A)
            
            # Since all charges are positive and proportional to q_A, 
            # increasing q_A should increase the upward field at N
            if Ey_perturbed <= Ey_orig:
                self.logs.append(f"FAIL: Monotonicity violated - Increasing q_A from {q_A_orig} to {perturbed_q_A} should increase Ey at N")
                self.logs.append(f"      Ey_orig = {Ey_orig:.4f}, Ey_perturbed = {Ey_perturbed:.4f}")
                return False
                
            self.logs.append(f"PASS: Monotonicity holds - Ey at N increases with q_A")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in monotonicity test: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs"""
        self.logs = []
        success = True
        
        try:
            # Run all test suites
            if not self.test_invariants():
                success = False
                self.logs.append("--- Invariants test failed ---")
            
            if not self.test_trace_consistency():
                success = False
                self.logs.append("--- Trace consistency test failed ---")
            
            if not self.test_monotonicity():
                success = False
                self.logs.append("--- Monotonicity test failed ---")
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        final_message = "ALL TESTS PASSED" if success else "SOME TESTS FAILED"
        self.logs.insert(0, f"=== Physics Checker Results: {final_message} ===")
        
        return success, self.logs