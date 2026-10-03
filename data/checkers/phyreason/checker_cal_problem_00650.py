import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Results with safe access - default to 0 if not found
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract ONLY the required results, ignore any other keys
        self.F = float(self.results.get('F', 0.0))
        self.E = float(self.results.get('E', 0.0))
        
        # Direction flags if provided (optional, for vector checks)
        self.F_direction = self.results.get('F_direction', None)
        self.E_direction = self.results.get('E_direction', None)
        
        # Problem constants (from problem statement, NOT from solution_instance)
        # Use fresh variable names to avoid conflicts
        self.k_const = 9.0e9  # N·m²/C²
        self.q1_const = -1.0e-10  # C
        self.q2_const = 1.0e-9  # C
        self.r_const = 0.30  # m (30 cm)
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        print(f"[PhysicsChecker] {msg}")
    
    def test_invariants(self):
        """Check fundamental physics laws: Coulomb's Law and superposition principle."""
        success = True
        
        # 1. Coulomb's Law: Check force magnitude
        # Expected signed force: F = k * q1 * q2 / r²
        F_expected_signed = self.k_const * self.q1_const * self.q2_const / (self.r_const ** 2)
        # Expected magnitude: |F|
        F_expected_magnitude = abs(F_expected_signed)
        
        # Use tighter tolerance for very small numbers
        # The expected force is 1e-8, so we need abs_tol < 1e-8
        if not math.isclose(self.F, F_expected_magnitude, rel_tol=1e-2, abs_tol=1e-10):
            self.log(f"FAIL: Coulomb's Law mismatch. Got F={self.F:.2e}, expected |F|={F_expected_magnitude:.2e}")
            success = False
        else:
            self.log(f"PASS: Coulomb's Law satisfied. F={self.F:.2e}")
        
        # Check if force magnitude is reasonable
        if self.F <= 0:
            self.log("WARNING: Force magnitude should be positive.")
        
        # 2. Electric Field at point A due to q2 only
        E_expected = self.k_const * self.q2_const / (self.r_const ** 2)  # Positive since q2 > 0
        
        if not math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Electric field mismatch. Got E={self.E:.2f}, expected {E_expected:.2f}")
            success = False
        else:
            self.log(f"PASS: Electric field correct. E={self.E:.2f}")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check consistency between force and electric field via F = q1 * E
        # This checks if E = F/q1 (in magnitude, considering signs)
        if abs(self.q1_const) > 1e-20:  # Avoid division by near-zero
            # Compute what E should be based on F and q1
            # Note: F is magnitude, q1 is negative, so E_from_force should be positive
            # But first check if F is not zero to avoid 0/very_small issues
            if abs(self.F) < 1e-20:
                self.log("SKIP: F is too small for F = |q1|*E check")
            else:
                E_from_force = abs(self.F / self.q1_const)
                
                if not math.isclose(self.E, E_from_force, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Trace consistency F = |q1|*E failed. E={self.E:.2f}, F/|q1|={E_from_force:.2f}")
                    success = False
                else:
                    self.log(f"PASS: Trace consistency F = |q1|*E satisfied.")
        else:
            self.log("SKIP: q1 too small for F = q1*E check")
        
        # 2. Check that removing q1 doesn't change E
        # E should be k*q2/r²
        E_due_to_q2 = self.k_const * self.q2_const / (self.r_const ** 2)
        if not math.isclose(self.E, E_due_to_q2, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: E mismatch when q1 removed. Got {self.E:.2f}, expected k*q2/r²={E_due_to_q2:.2f}")
            success = False
        else:
            self.log(f"PASS: E consistent with superposition principle.")
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        success = True
        
        # Store original values
        orig_q2 = self.q2_const
        orig_r = self.r_const
        orig_F = self.F
        orig_E = self.E
        
        try:
            # Test 1: Increase |q2| by 10% → force and field should increase
            q2_test = orig_q2 * 1.1
            F_new_magnitude = self.k_const * abs(self.q1_const * q2_test) / (self.r_const ** 2)
            E_new = self.k_const * q2_test / (self.r_const ** 2)
            
            # For monotonicity, we need to handle the case where orig_F might be 0
            # If orig_F is 0, any positive F_new should be considered an increase
            F_increased = (F_new_magnitude > orig_F) or (abs(orig_F) < 1e-20 and F_new_magnitude > 1e-20)
            E_increased = E_new > orig_E
            
            if not (F_increased and E_increased):
                self.log(f"FAIL: Increasing |q2| should increase |F| and |E|.")
                self.log(f"  F_new={F_new_magnitude:.2e}, orig_F={orig_F:.2e}")
                self.log(f"  E_new={E_new:.2f}, orig_E={orig_E:.2f}")
                success = False
            else:
                self.log(f"PASS: Monotonicity in |q2| verified.")
            
            # Test 2: Increase r by 10% → force and field should decrease
            r_test = orig_r * 1.1
            F_new_magnitude = self.k_const * abs(self.q1_const * orig_q2) / (r_test ** 2)
            E_new = self.k_const * orig_q2 / (r_test ** 2)
            
            F_decreased = F_new_magnitude < orig_F
            E_decreased = E_new < orig_E
            
            if not (F_decreased and E_decreased):
                self.log(f"FAIL: Increasing r should decrease |F| and |E|.")
                self.log(f"  F_new={F_new_magnitude:.2e}, orig_F={orig_F:.2e}")
                self.log(f"  E_new={E_new:.2f}, orig_E={orig_E:.2f}")
                success = False
            else:
                self.log(f"PASS: Monotonicity in r verified.")
                
        except Exception as e:
            self.log(f"ERROR in monotonicity test: {e}\n{traceback.format_exc()}")
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs = []
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"CRASH in {test_name}: {e}\n{traceback.format_exc()}")
                overall_success = False
        
        self.log(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs