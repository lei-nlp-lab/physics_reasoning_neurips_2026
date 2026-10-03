import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Physical constants
        self.g = 9.81  # gravity (though not used in this problem)
        
        # Extract input parameters safely
        self.m = getattr(solution_instance, 'm', self.inputs.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', self.inputs.get('q', 1.0))
        self.E = getattr(solution_instance, 'E', self.inputs.get('E', 1.0))
        self.L = getattr(solution_instance, 'L', self.inputs.get('L', 1.0))
        
        # Extract results safely using mapping
        self.a1 = self.results.get('a_1', 0.0)
        self.v1 = self.results.get('v_1', 0.0)
        self.t = self.results.get('t', 0.0)
        self.v = self.results.get('v', 0.0)
        self.r = self.results.get('r', 0.0)
        self.B = self.results.get('B', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        self.v2 = self.results.get('v_2', 0.0)
        self.s = self.results.get('s', 0.0)
        
        # Derived quantities for checking
        self.s_A = 2 * self.L  # from step_1
        self.theta = math.pi/4  # 45 degrees
        self.cos45 = math.sqrt(2)/2
        
        # Store logs
        self.logs = []

    def log(self, message, success=None):
        """Helper to log messages with optional success flag"""
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(f"  {message}")

    def test_invariants(self):
        """Test conservation laws and fundamental physics"""
        success = True
        
        try:
            # 1. Check Newton's 2nd Law for ball A
            # a1 should equal qE/m
            expected_a1 = self.q * self.E / self.m
            if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Newton's 2nd Law (A): a1={self.a1:.4e}, expected qE/m={expected_a1:.4e}", False)
                success = False
            else:
                self.log(f"Newton's 2nd Law (A) valid: a1 = qE/m = {self.a1:.4e}", True)
            
            # 2. Check momentum conservation during collision
            # m*v1 = (2m)*v  => v = v1/2
            expected_v = self.v1 / 2
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Momentum conservation: v={self.v:.4e}, expected v1/2={expected_v:.4e}", False)
                success = False
            else:
                self.log(f"Momentum conservation valid: v = v1/2 = {self.v:.4e}", True)
            
            # 3. Check Lorentz force = centripetal force for D in magnetic field
            # q*v*B = (2m)*v²/r => B = 2m*v/(q*r)
            if self.r > 0 and self.v > 0:
                expected_B = (2 * self.m * self.v) / (self.q * self.r)
                if not math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Lorentz force balance: B={self.B:.4e}, expected 2mv/(qr)={expected_B:.4e}", False)
                    success = False
                else:
                    self.log(f"Lorentz force balance valid: qvB = 2mv²/r", True)
            
            # 4. Check Newton's 2nd Law for object D in electric field
            # a2 should equal qE/(2m)
            expected_a2 = self.q * self.E / (2 * self.m)
            if not math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Newton's 2nd Law (D): a2={self.a2:.4e}, expected qE/(2m)={expected_a2:.4e}", False)
                success = False
            else:
                self.log(f"Newton's 2nd Law (D) valid: a2 = qE/(2m) = {self.a2:.4e}", True)
            
            # 5. Check geometry: r should be 4L (from step_6)
            expected_r = 4 * self.L
            if not math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Geometry check: r={self.r:.4e}, expected 4L={expected_r:.4e}", False)
                success = False
            else:
                self.log(f"Geometry valid: r = 4L = {self.r:.4e}", True)
                
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", False)
            success = False
            
        return success

    def test_trace_consistency(self):
        """Check that intermediate results are consistent"""
        success = True
        
        try:
            # 1. Check kinematics for ball A: s = 0.5*a1*t²
            expected_t = math.sqrt(2 * self.s_A / self.a1) if self.a1 > 0 else 0
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Kinematics (A): t={self.t:.4e}, expected √(2s/a1)={expected_t:.4e}", False)
                success = False
            else:
                self.log(f"Kinematics (A) consistent: t = √(2s/a1) = {self.t:.4e}", True)
            
            # 2. Check v1 = a1*t
            expected_v1 = self.a1 * self.t
            if not math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v1 calculation: v1={self.v1:.4e}, expected a1*t={expected_v1:.4e}", False)
                success = False
            else:
                self.log(f"v1 consistent: v1 = a1*t = {self.v1:.4e}", True)
            
            # 3. Check v = v1/2 (already in invariants, but double-check)
            if not math.isclose(self.v, self.v1/2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Collision velocity: v={self.v:.4e}, expected v1/2={self.v1/2:.4e}", False)
                success = False
            else:
                self.log(f"Collision velocity consistent: v = v1/2", True)
            
            # 4. Check v2 = √(v² + (a2*t2)²) with t2 = 2v/a2
            if self.a2 > 0:
                t2 = 2 * self.v / self.a2
                expected_v2 = math.sqrt(self.v**2 + (self.a2 * t2)**2)
                if not math.isclose(self.v2, expected_v2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"v2 calculation: v2={self.v2:.4e}, expected √(v²+(a2*t2)²)={expected_v2:.4e}", False)
                    success = False
                else:
                    self.log(f"v2 consistent: v2 = √(v²+(a2*t2)²) = {self.v2:.4e}", True)
                
                # 5. Check s = √2 * v * t2 = 4√2 L
                expected_s = 4 * math.sqrt(2) * self.L
                s_calc = math.sqrt(2) * self.v * t2
                if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2) or \
                   not math.isclose(s_calc, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Displacement s: s={self.s:.4e}, calc={s_calc:.4e}, expected 4√2L={expected_s:.4e}", False)
                    success = False
                else:
                    self.log(f"Displacement consistent: s = 4√2L = {self.s:.4e}", True)
                    
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", False)
            success = False
            
        return success

    def test_monotonicity(self):
        """Test that perturbing inputs produces expected trends"""
        success = True
        
        try:
            # Store original values
            q_orig = self.q
            perturb_factor = 1.1  # 10% increase
            
            # Perturb q
            self.q = q_orig * perturb_factor
            
            # Recalculate using formulas from steps
            # Step 1-2: a1 = qE/m
            a1_new = self.q * self.E / self.m
            
            # Step 3: t = √(2s/a1) with s = 2L
            t_new = math.sqrt(2 * self.s_A / a1_new)
            
            # Step 4: v1 = a1 * t
            v1_new = a1_new * t_new
            
            # Step 5: v = v1/2
            v_new = v1_new / 2
            
            # Step 6: r = 4L (geometry, unchanged)
            r_new = 4 * self.L
            
            # Step 8: B = 2mv/(qr)
            B_new = (2 * self.m * v_new) / (self.q * r_new)
            
            # Step 10: a2 = qE/(2m)
            a2_new = self.q * self.E / (2 * self.m)
            
            # Step 11-12: t2 = 2v/a2, v2 = √(v² + (a2*t2)²)
            t2_new = 2 * v_new / a2_new
            v2_new = math.sqrt(v_new**2 + (a2_new * t2_new)**2)
            
            # Restore original q
            self.q = q_orig
            
            # Check trends
            # Since q increased by 10%:
            # a1 should increase proportionally
            if not math.isclose(a1_new, self.a1 * perturb_factor, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity: a1 should increase by {perturb_factor}x with q", False)
                success = False
            
            # t should decrease (t ∝ 1/√q)
            expected_t_factor = 1 / math.sqrt(perturb_factor)
            if not math.isclose(t_new, self.t * expected_t_factor, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity: t should decrease by 1/√q factor", False)
                success = False
            
            # v1 should increase (v1 ∝ √q)
            expected_v1_factor = math.sqrt(perturb_factor)
            if not math.isclose(v1_new, self.v1 * expected_v1_factor, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity: v1 should increase by √q factor", False)
                success = False
            
            # v should increase (v ∝ √q)
            if not math.isclose(v_new, self.v * expected_v1_factor, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity: v should increase by √q factor", False)
                success = False
            
            # B should decrease (B ∝ v/q ∝ 1/√q)
            expected_B_factor = 1 / math.sqrt(perturb_factor)
            if not math.isclose(B_new, self.B * expected_B_factor, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity: B should decrease by 1/√q factor", False)
                success = False
            
            # v2 should increase (v2 = √5 * v ∝ √q)
            if not math.isclose(v2_new, self.v2 * expected_v1_factor, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity: v2 should increase by √q factor", False)
                success = False
            
            if success:
                self.log(f"Monotonicity test passed for q perturbation", True)
                
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}", False)
            success = False
            
        return success

    def run_all_tests(self):
        """Run all tests and return success flag with logs"""
        self.logs = []
        overall_success = True
        
        try:
            self.log("=== PhysicsChecker starting ===")
            
            # Test 1: Invariants
            self.log("\n1. Testing invariants (conservation laws, force balances):")
            if not self.test_invariants():
                overall_success = False
                self.log("  Failed some invariant tests", False)
            
            # Test 2: Trace consistency
            self.log("\n2. Testing trace consistency (intermediate calculations):")
            if not self.test_trace_consistency():
                overall_success = False
                self.log("  Failed some consistency tests", False)
            
            # Test 3: Monotonicity
            self.log("\n3. Testing monotonicity (parameter perturbation):")
            if not self.test_monotonicity():
                overall_success = False
                self.log("  Failed monotonicity test", False)
            
            # Summary
            self.log(f"\n=== Summary: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===")
            
        except Exception as e:
            self.log(f"Fatal error in run_all_tests: {str(e)}", False)
            self.log(traceback.format_exc())
            overall_success = False
        
        return overall_success, self.logs