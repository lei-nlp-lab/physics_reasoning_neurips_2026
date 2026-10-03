import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs from problem statement (fixed values)
        self.R = 0.5  # ohm
        self.l = 0.20  # m
        self.B = 0.50  # T
        self.m = 0.1  # kg
        self.r = 0.5  # ohm
        self.F = 0.6  # N
        self.f = 0.5  # N
        
        # Safely extract results from solution instance
        self.P = getattr(solution_instance, 'P', solution_instance.results.get('P', 0.0))
        self.I = getattr(solution_instance, 'I', solution_instance.results.get('I', 0.0))
        self.F_a = getattr(solution_instance, 'F_a', solution_instance.results.get('F_a', 0.0))
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        self.v_m = getattr(solution_instance, 'v_m', solution_instance.results.get('v_m', 0.0))
        
        # Store original values for monotonicity test
        self.original_results = {
            'P': self.P, 'I': self.I, 'F_a': self.F_a, 
            'E': self.E, 'v_m': self.v_m
        }
        
        # Tolerance for fuzzy matching (2% relative, 0.02 absolute)
        self.rel_tol = 0.02
        self.abs_tol = 0.02
        
        self.logs = []
        
    def log(self, message):
        """Helper to log messages."""
        self.logs.append(message)
    
    def check_close(self, a, b, label=""):
        """Check if two values are close with fuzzy matching."""
        is_close = math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        if not is_close:
            self.log(f"FAIL {label}: {a} != {b} (tolerance: rel={self.rel_tol}, abs={self.abs_tol})")
        return is_close
    
    def test_invariants(self):
        """Test fundamental physical laws."""
        success = True
        
        # 1. Newton's First Law (Equilibrium at maximum velocity)
        # Net force should be zero when velocity is constant (v_m)
        net_force = self.F - self.f - self.F_a
        if not self.check_close(net_force, 0.0, "Newton's First Law (F_net = 0)"):
            success = False
            self.log(f"  Net force = {net_force:.6f} N")
        
        # 2. Ohm's Law for the circuit
        # I = E / (R + r)
        expected_I = self.E / (self.R + self.r)
        if not self.check_close(self.I, expected_I, "Ohm's Law (I = E/(R+r))"):
            success = False
            self.log(f"  Expected I = {expected_I:.6f}, got {self.I:.6f}")
        
        # 3. Power in resistor R
        # P = I² * R
        expected_P = self.I**2 * self.R
        if not self.check_close(self.P, expected_P, "Power in resistor (P = I²R)"):
            success = False
            self.log(f"  Expected P = {expected_P:.6f}, got {self.P:.6f}")
        
        # 4. Motional EMF
        # E = B * l * v_m
        expected_E = self.B * self.l * self.v_m
        if not self.check_close(self.E, expected_E, "Motional EMF (E = B*l*v)"):
            success = False
            self.log(f"  Expected E = {expected_E:.6f}, got {self.E:.6f}")
        
        # 5. Ampere's Law (magnetic force on conductor)
        # F_a = B * I * l
        expected_F_a = self.B * self.I * self.l
        if not self.check_close(self.F_a, expected_F_a, "Ampere Force (F_a = B*I*l)"):
            success = False
            self.log(f"  Expected F_a = {expected_F_a:.6f}, got {self.F_a:.6f}")
        
        # 6. Check derived formula for F_a from STEP 4
        # F_a = (B² * l² * v_m) / (R + r)
        expected_F_a_derived = (self.B**2 * self.l**2 * self.v_m) / (self.R + self.r)
        if not self.check_close(self.F_a, expected_F_a_derived, "Derived F_a formula"):
            success = False
            self.log(f"  Expected F_a (derived) = {expected_F_a_derived:.6f}, got {self.F_a:.6f}")
        
        return success
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        success = True
        
        # Check that v_m produces the expected E, I, F_a, P chain
        # Recalculate from v_m using the formulas in steps
        v = self.v_m
        
        # Step 1: E = B*l*v
        E_calc = self.B * self.l * v
        
        # Step 2: I = E/(R+r)
        I_calc = E_calc / (self.R + self.r)
        
        # Step 3: F_a = B*I*l
        F_a_calc = self.B * I_calc * self.l
        
        # Step 4: Alternative F_a = (B²*l²*v)/(R+r)
        F_a_calc_alt = (self.B**2 * self.l**2 * v) / (self.R + self.r)
        
        # Step 9: P = I²*R
        P_calc = I_calc**2 * self.R
        
        # Check consistency
        checks = [
            ("E from v_m", self.E, E_calc),
            ("I from v_m", self.I, I_calc),
            ("F_a from v_m (direct)", self.F_a, F_a_calc),
            ("F_a from v_m (derived)", self.F_a, F_a_calc_alt),
            ("P from v_m", self.P, P_calc)
        ]
        
        for label, actual, calculated in checks:
            if not self.check_close(actual, calculated, f"Trace consistency: {label}"):
                success = False
        
        # Check that Newton's law holds with calculated values
        net_force_calc = self.F - self.f - F_a_calc
        if not self.check_close(net_force_calc, 0.0, "Newton's Law with calculated F_a"):
            success = False
            self.log(f"  Net force (calc) = {net_force_calc:.6f}")
        
        return success
    
    def test_monotonicity(self):
        """Test that perturbations in inputs produce expected trends."""
        success = True
        
        # Store original values
        original_v = self.v_m
        original_P = self.P
        
        # Test 1: Increase magnetic field B
        self.log("\nMonotonicity Test 1: Increase B by 10%")
        B_perturbed = self.B * 1.1
        
        # Calculate new v_m from equilibrium: F - f = (B²*l²*v)/(R+r)
        # => v = (F - f)*(R+r) / (B²*l²)
        v_new_B = ((self.F - self.f) * (self.R + self.r)) / (B_perturbed**2 * self.l**2)
        
        # v should decrease when B increases (stronger opposing magnetic force)
        if v_new_B >= original_v:
            success = False
            self.log(f"  FAIL: v_m increased from {original_v:.3f} to {v_new_B:.3f} when B increased")
        else:
            self.log(f"  PASS: v_m decreased from {original_v:.3f} to {v_new_B:.3f}")
        
        # Test 2: Increase pulling force F
        self.log("\nMonotonicity Test 2: Increase F by 10%")
        F_perturbed = self.F * 1.1
        
        # v = (F - f)*(R+r) / (B²*l²)
        v_new_F = ((F_perturbed - self.f) * (self.R + self.r)) / (self.B**2 * self.l**2)
        
        # v should increase when F increases
        if v_new_F <= original_v:
            success = False
            self.log(f"  FAIL: v_m decreased from {original_v:.3f} to {v_new_F:.3f} when F increased")
        else:
            self.log(f"  PASS: v_m increased from {original_v:.3f} to {v_new_F:.3f}")
        
        # Test 3: Increase resistance (R+r)
        self.log("\nMonotonicity Test 3: Increase total resistance by 20%")
        total_R_perturbed = (self.R + self.r) * 1.2
        
        # v = (F - f)*(R+r) / (B²*l²)
        v_new_R = ((self.F - self.f) * total_R_perturbed) / (self.B**2 * self.l**2)
        
        # v should increase when resistance increases (less current → smaller F_a)
        if v_new_R <= original_v:
            success = False
            self.log(f"  FAIL: v_m decreased from {original_v:.3f} to {v_new_R:.3f} when resistance increased")
        else:
            self.log(f"  PASS: v_m increased from {original_v:.3f} to {v_new_R:.3f}")
        
        # Test 4: Check power trend with B
        # P = I²*R, and I = E/(R+r) = (B*l*v)/(R+r)
        # With perturbed B, recalculate v, then I, then P
        I_new_B = (B_perturbed * self.l * v_new_B) / (self.R + self.r)
        P_new_B = I_new_B**2 * self.R
        
        # Power should increase with B? Actually complex relationship:
        # P = (B²*l²*v²/(R+r)²)*R = R*(F-f)²/(B²*l²) * (B²*l²/(R+r))? Let's check trend.
        # From v = (F-f)*(R+r)/(B²*l²), substitute into P = I²*R = [B*l*v/(R+r)]²*R
        # P = B²*l²*v²*R/(R+r)² = B²*l²*R*(F-f)²*(R+r)²/(B⁴*l⁴*(R+r)²) = R*(F-f)²/(B²*l²)
        # So P ∝ 1/B², so P should decrease when B increases
        if P_new_B >= original_P:
            success = False
            self.log(f"  FAIL: P increased from {original_P:.3f} to {P_new_B:.3f} when B increased")
        else:
            self.log(f"  PASS: P decreased from {original_P:.3f} to {P_new_B:.3f}")
        
        # Restore original values
        self.v_m = original_v
        self.P = original_P
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        all_success = True
        
        try:
            self.log("=" * 50)
            self.log("PhysicsChecker - Starting Tests")
            self.log("=" * 50)
            
            # Test 1: Invariants
            self.log("\n1. Testing Physical Invariants:")
            inv_success = self.test_invariants()
            if inv_success:
                self.log("  ✓ All invariants passed")
            else:
                self.log("  ✗ Some invariants failed")
            all_success = all_success and inv_success
            
            # Test 2: Trace Consistency
            self.log("\n2. Testing Trace Consistency:")
            trace_success = self.test_trace_consistency()
            if trace_success:
                self.log("  ✓ Trace consistency passed")
            else:
                self.log("  ✗ Trace consistency failed")
            all_success = all_success and trace_success
            
            # Test 3: Monotonicity
            self.log("\n3. Testing Monotonicity:")
            mono_success = self.test_monotonicity()
            if mono_success:
                self.log("  ✓ Monotonicity passed")
            else:
                self.log("  ✗ Monotonicity failed")
            all_success = all_success and mono_success
            
            # Test 4: Target Answers
            self.log("\n4. Checking Target Answers:")
            target_v = 10.0  # m/s
            target_P = 0.5   # W
            
            v_check = self.check_close(self.v_m, target_v, "Target v_m = 10 m/s")
            P_check = self.check_close(self.P, target_P, "Target P = 0.5 W")
            
            target_success = v_check and P_check
            if target_success:
                self.log("  ✓ Target answers matched")
            else:
                self.log("  ✗ Target answers mismatched")
            all_success = all_success and target_success
            
            # Summary
            self.log("\n" + "=" * 50)
            self.log("Test Summary:")
            self.log(f"  Invariants: {'PASS' if inv_success else 'FAIL'}")
            self.log(f"  Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
            self.log(f"  Monotonicity: {'PASS' if mono_success else 'FAIL'}")
            self.log(f"  Target Answers: {'PASS' if target_success else 'FAIL'}")
            self.log(f"\nOverall: {'ALL TESTS PASSED' if all_success else 'SOME TESTS FAILED'}")
            self.log("=" * 50)
            
        except Exception as e:
            all_success = False
            self.log(f"\nERROR during testing: {str(e)}")
            self.log(traceback.format_exc())
        
        return all_success, self.logs