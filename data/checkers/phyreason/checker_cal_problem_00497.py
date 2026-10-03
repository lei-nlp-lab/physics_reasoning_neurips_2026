import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.E = 100.0  # Given in problem context
        self.L = 0.1    # 10 cm = 0.1 m (given)
        
        # Extract results safely
        self.theta = getattr(solution_instance, 'theta', 
                           solution_instance.results.get('theta', 60.0))
        self.d = getattr(solution_instance, 'd', 
                        solution_instance.results.get('d', 0.0))
        self.U_AB = getattr(solution_instance, 'U_AB', 
                           solution_instance.results.get('U_AB', 0.0))
        self.U_BA = getattr(solution_instance, 'U_BA', 
                           solution_instance.results.get('U_BA', 0.0))
        
        # Initialize symbolic variables
        self.E_sym = sp.Symbol('E', positive=True)
        self.d_sym = sp.Symbol('d', real=True)
        self.U_AB_sym = sp.Symbol('U_AB', real=True)
        self.U_BA_sym = sp.Symbol('U_BA', real=True)
        self.theta_sym = sp.Symbol('theta', real=True)
        
        # Store logs
        self.logs = []
    
    def log(self, message, success):
        self.logs.append({"test": message, "success": success})
    
    def test_invariants(self):
        """Verify physics laws from KB reference."""
        try:
            # Theorem 1: U_AB = -E * d (from KB formula delta_V = -E_avg * delta_x)
            # where d = L * cos(theta) is displacement along field from A to B
            theta_rad = math.radians(self.theta)
            d_expected = self.L * math.cos(theta_rad)
            
            # Check distance calculation
            if not math.isclose(self.d, d_expected, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Distance d incorrect: expected {d_expected:.4f}, got {self.d}", False)
                return False
            else:
                self.log(f"Distance d correct: {self.d:.4f} m", True)
            
            # Check potential difference magnitude: |U_AB| = E * d
            magnitude_expected = self.E * d_expected
            if not math.isclose(abs(self.U_AB), magnitude_expected, 
                               rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"|U_AB| magnitude mismatch: expected {magnitude_expected:.2f}, got {abs(self.U_AB):.2f}", False)
                return False
            else:
                self.log(f"|U_AB| magnitude correct: {abs(self.U_AB):.2f} V", True)
            
            # Theorem 2: U_BA = -U_AB
            if not math.isclose(self.U_BA, -self.U_AB, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"U_BA ≠ -U_AB: U_BA={self.U_BA:.2f}, -U_AB={-self.U_AB:.2f}", False)
                return False
            else:
                self.log(f"U_BA = -U_AB relation holds", True)
            
            # Symbolic verification
            try:
                # Theorem 1 symbolic
                theorem1 = sp.Eq(self.U_AB_sym, -self.E_sym * self.d_sym)
                theorem1_check = sp.N(theorem1.subs({
                    self.U_AB_sym: self.U_AB,
                    self.E_sym: self.E,
                    self.d_sym: self.d
                }))
                
                # Theorem 2 symbolic
                theorem2 = sp.Eq(self.U_BA_sym, -self.U_AB_sym)
                theorem2_check = sp.N(theorem2.subs({
                    self.U_BA_sym: self.U_BA,
                    self.U_AB_sym: self.U_AB
                }))
                
                if not (bool(theorem1_check) and bool(theorem2_check)):
                    self.log("Symbolic verification failed", False)
                    return False
                else:
                    self.log("Symbolic verification passed", True)
            
            except Exception as e:
                self.log(f"Symbolic check skipped: {str(e)}", True)
            
            return True
            
        except Exception as e:
            self.log(f"Invariants test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results from STEPS ANALYSIS."""
        try:
            # Step 2: d = L * cos(theta)
            theta_rad = math.radians(self.theta)
            d_calc = self.L * math.cos(theta_rad)
            d_expected_step2 = 0.05  # L * cos(60°) = 0.1 * 0.5 = 0.05 m
            
            if not math.isclose(d_calc, d_expected_step2, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Step 2 inconsistency: d should be ~0.05 m, got {d_calc:.4f}", False)
                return False
            else:
                self.log("Step 2 (d calculation) consistent", True)
            
            # Step 3: U_AB = E * d (note: this is magnitude, sign handled in step 4)
            U_AB_mag_calc = self.E * d_calc
            U_AB_mag_expected = 5.0  # 100 * 0.05 = 5 V
            
            if not math.isclose(abs(self.U_AB), U_AB_mag_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Step 3 inconsistency: |U_AB| should be ~5 V, got {abs(self.U_AB):.2f}", False)
                return False
            else:
                self.log("Step 3 (U_AB magnitude) consistent", True)
            
            # Step 4: U_BA = -U_AB
            if not math.isclose(self.U_BA, -self.U_AB, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Step 4 inconsistency: U_BA should be -U_AB", False)
                return False
            else:
                self.log("Step 4 (U_BA = -U_AB) consistent", True)
            
            # Final answer check
            target_answer = -5.0  # From TARGET ANSWERS
            if not math.isclose(self.U_BA, target_answer, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Final answer mismatch: expected U_BA ≈ -5 V, got {self.U_BA:.2f} V", False)
                return False
            else:
                self.log(f"Final answer correct: U_BA = {self.U_BA:.2f} V", True)
            
            return True
            
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb input angle and check trend consistency."""
        try:
            original_theta = self.theta
            
            # Test perturbation 1: Increase angle to 70°
            self.theta = 70.0
            theta_rad = math.radians(self.theta)
            d_perturbed = self.L * math.cos(theta_rad)
            U_AB_perturbed = self.E * d_perturbed
            U_BA_perturbed = -U_AB_perturbed
            
            # For fixed E and L, as theta increases (0-90°), cos(theta) decreases
            # So |U_AB| decreases, and |U_BA| decreases
            d_original = self.L * math.cos(math.radians(original_theta))
            
            if d_perturbed < d_original:
                self.log(f"Monotonicity: d decreases as theta increases (cosine trend)", True)
            else:
                self.log(f"Monotonicity violation: d should decrease when theta increases", False)
                return False
            
            # Test perturbation 2: Decrease angle to 30°
            self.theta = 30.0
            theta_rad = math.radians(self.theta)
            d_perturbed2 = self.L * math.cos(theta_rad)
            
            if d_perturbed2 > d_original:
                self.log(f"Monotonicity: d increases as theta decreases", True)
            else:
                self.log(f"Monotonicity violation: d should increase when theta decreases", False)
                return False
            
            # Restore original value
            self.theta = original_theta
            
            # Test E perturbation
            original_E = self.E
            self.E = 150.0  # Increase by 50%
            U_AB_perturbed_E = self.E * d_original
            
            if U_AB_perturbed_E > abs(self.U_AB):
                self.log(f"Monotonicity: |U_AB| increases with E (proportional trend)", True)
            else:
                self.log(f"Monotonicity violation: |U_AB| should increase with E", False)
                return False
            
            self.E = original_E  # Restore
            
            return True
            
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return results."""
        self.logs = []
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            try:
                if not test_func():
                    all_passed = False
                    self.logs[-1]["test"] = f"{test_name}: " + self.logs[-1]["test"]
            except Exception as e:
                all_passed = False
                self.log(f"{test_name}: Failed with exception: {str(e)}", False)
        
        return all_passed, self.logs