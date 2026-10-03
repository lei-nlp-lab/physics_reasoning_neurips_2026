import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'inputs', {}).get('L', 0.40))
        self.theta_deg = getattr(solution_instance, 'theta', getattr(solution_instance, 'inputs', {}).get('theta', 37))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 0.50))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 0.040))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'inputs', {}).get('I', 1.5))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        
        # Convert theta to radians for calculations
        self.theta_rad = math.radians(self.theta_deg)
        # Use given sin/cos values explicitly for consistency
        self.sin_theta = 0.60  # sin37°
        self.cos_theta = 0.80  # cos37°
        
        # Safe extraction of results
        self.results = getattr(solution_instance, 'results', {})
        self.F_a = self.results.get('F_a', 0.0)
        self.F_1 = self.results.get('F_1', 0.0)
        self.f = self.results.get('f', 0.0)
        self.B_2 = self.results.get('B_2', 0.0)
        self.B_3 = self.results.get('B_3', 0.0)
        
        # Store logs
        self.logs = []
        
    def log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
        return success
        
    def safe_isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
            
    def test_invariants(self):
        """Test fundamental physics laws: force balance and Ampere's law."""
        self.log("=== Testing Physics Invariants ===")
        passes = []
        
        # 1. Force balance along incline (rod stationary)
        # Net force = F_a (up incline) - mg sinθ - f (down incline) should be 0
        F_a_val = float(self.F_a)
        mg_sinθ = float(self.m * self.g * self.sin_theta)
        f_val = float(self.f)
        net_force = F_a_val - mg_sinθ - f_val
        
        if self.safe_isclose(net_force, 0):
            self.log(f"Force balance satisfied: F_a({F_a_val}) - mg sinθ({mg_sinθ}) - f({f_val}) ≈ 0", True)
            passes.append(True)
        else:
            self.log(f"Force balance violated: net = {net_force:.4f} (should be 0)", False)
            passes.append(False)
        
        # 2. Ampere's law: F_a = B * I * L
        expected_F_a = self.B * self.I * self.L
        if self.safe_isclose(self.F_a, expected_F_a):
            self.log(f"Ampere's law satisfied: B({self.B})*I({self.I})*L({self.L}) = {expected_F_a:.3f} ≈ F_a({self.F_a})", True)
            passes.append(True)
        else:
            self.log(f"Ampere's law violated: expected F_a = {expected_F_a:.3f}, got {self.F_a}", False)
            passes.append(False)
        
        # 3. Friction direction consistency
        # Based on steps: F_a > mg sinθ so friction acts downward along incline
        if F_a_val > mg_sinθ and f_val > 0:
            self.log(f"Friction direction consistent: F_a({F_a_val}) > mg sinθ({mg_sinθ}) so f({f_val}) > 0 (down incline)", True)
            passes.append(True)
        else:
            self.log(f"Friction direction inconsistent: expected F_a > mg sinθ and f>0", False)
            passes.append(False)
            
        # 4. B_2 formula check (when B is vertical)
        # B_2 = (m*g*tanθ) / (I*L)
        expected_B_2 = (self.m * self.g * (self.sin_theta/self.cos_theta)) / (self.I * self.L)
        if self.safe_isclose(self.B_2, expected_B_2):
            self.log(f"B_2 formula satisfied: (m*g*tanθ)/(I*L) = {expected_B_2:.3f} ≈ B_2({self.B_2})", True)
            passes.append(True)
        else:
            self.log(f"B_2 formula violated: expected {expected_B_2:.3f}, got {self.B_2}", False)
            passes.append(False)
            
        # 5. B_3 formula check (minimum B when friction is zero)
        # B_3 = (m*g*sinθ) / (I*L)
        expected_B_3 = (self.m * self.g * self.sin_theta) / (self.I * self.L)
        if self.safe_isclose(self.B_3, expected_B_3):
            self.log(f"B_3 formula satisfied: (m*g*sinθ)/(I*L) = {expected_B_3:.3f} ≈ B_3({self.B_3})", True)
            passes.append(True)
        else:
            self.log(f"B_3 formula violated: expected {expected_B_3:.3f}, got {self.B_3}", False)
            passes.append(False)
            
        return all(passes)
        
    def test_trace_consistency(self):
        """Check that intermediate calculations match final answers."""
        self.log("\n=== Testing Trace Consistency ===")
        passes = []
        
        # From steps:
        # Step 1: F_a = 0.30 N
        expected_F_a = 0.30
        if self.safe_isclose(self.F_a, expected_F_a):
            self.log(f"Step 1 (F_a) consistent: expected {expected_F_a}, got {self.F_a}", True)
            passes.append(True)
        else:
            self.log(f"Step 1 (F_a) inconsistent: expected {expected_F_a}, got {self.F_a}", False)
            passes.append(False)
            
        # Step 2: F_1 = mg sinθ = 0.24 N
        expected_F_1 = self.m * self.g * self.sin_theta
        if self.safe_isclose(self.F_1, expected_F_1):
            self.log(f"Step 2 (F_1) consistent: mg sinθ = {expected_F_1:.3f} ≈ {self.F_1}", True)
            passes.append(True)
        else:
            self.log(f"Step 2 (F_1) inconsistent: expected {expected_F_1:.3f}, got {self.F_1}", False)
            passes.append(False)
            
        # Step 5: f = F_a - F_1 = 0.06 N
        expected_f = expected_F_a - expected_F_1
        if self.safe_isclose(self.f, expected_f):
            self.log(f"Step 5 (f) consistent: F_a - F_1 = {expected_f:.3f} ≈ {self.f}", True)
            passes.append(True)
        else:
            self.log(f"Step 5 (f) inconsistent: expected {expected_f:.3f}, got {self.f}", False)
            passes.append(False)
            
        # Step 7: B_2 = 0.5 T
        expected_B_2 = 0.5
        if self.safe_isclose(self.B_2, expected_B_2):
            self.log(f"Step 7 (B_2) consistent: expected {expected_B_2}, got {self.B_2}", True)
            passes.append(True)
        else:
            self.log(f"Step 7 (B_2) inconsistent: expected {expected_B_2}, got {self.B_2}", False)
            passes.append(False)
            
        # Step 9: B_3 = 0.4 T
        expected_B_3 = 0.4
        if self.safe_isclose(self.B_3, expected_B_3):
            self.log(f"Step 9 (B_3) consistent: expected {expected_B_3}, got {self.B_3}", True)
            passes.append(True)
        else:
            self.log(f"Step 9 (B_3) inconsistent: expected {expected_B_3}, got {self.B_3}", False)
            passes.append(False)
            
        return all(passes)
        
    def test_monotonicity(self):
        """Perturb inputs and check that results change in expected directions."""
        self.log("\n=== Testing Monotonicity ===")
        passes = []
        
        # Store original values
        orig_m = self.m
        
        # Perturb mass upward by 20%
        self.m *= 1.2
        
        # Recalculate expected values
        # F_a should remain same (depends on B, I, L, not m)
        # F_1 should increase by 20%
        # f should change by -0.2 * (mg sinθ) if F_a constant
        # B_2 should increase by 20%
        # B_3 should increase by 20%
        
        new_F_1 = self.m * self.g * self.sin_theta
        new_f = self.F_a - new_F_1
        new_B_2 = (self.m * self.g * (self.sin_theta/self.cos_theta)) / (self.I * self.L)
        new_B_3 = (self.m * self.g * self.sin_theta) / (self.I * self.L)
        
        # Check trends
        if new_F_1 > float(self.F_1):
            self.log(f"Monotonicity: Increasing mass increases F_1 ({self.F_1:.3f} -> {new_F_1:.3f})", True)
            passes.append(True)
        else:
            self.log(f"Monotonicity: F_1 did not increase with mass", False)
            passes.append(False)
            
        if new_f < float(self.f):
            self.log(f"Monotonicity: Increasing mass decreases f ({self.f:.3f} -> {new_f:.3f})", True)
            passes.append(True)
        else:
            self.log(f"Monotonicity: f did not decrease with mass increase", False)
            passes.append(False)
            
        if new_B_2 > float(self.B_2):
            self.log(f"Monotonicity: Increasing mass increases B_2 ({self.B_2:.3f} -> {new_B_2:.3f})", True)
            passes.append(True)
        else:
            self.log(f"Monotonicity: B_2 did not increase with mass", False)
            passes.append(False)
            
        if new_B_3 > float(self.B_3):
            self.log(f"Monotonicity: Increasing mass increases B_3 ({self.B_3:.3f} -> {new_B_3:.3f})", True)
            passes.append(True)
        else:
            self.log(f"Monotonicity: B_3 did not increase with mass", False)
            passes.append(False)
        
        # Restore original mass
        self.m = orig_m
        
        return all(passes)
        
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            overall = inv_ok and trace_ok and mono_ok
            
            self.log(f"\n=== SUMMARY ===")
            self.log(f"Invariants: {'PASS' if inv_ok else 'FAIL'}")
            self.log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
            self.log(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
            self.log(f"OVERALL: {'PASS' if overall else 'FAIL'}")
            
            return overall, self.logs
            
        except Exception as e:
            error_msg = f"Error during testing: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(f"ERROR: {error_msg}")
            return False, self.logs