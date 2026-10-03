import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize all required attributes from the solution instance.
        """
        # Safely extract inputs with defaults from problem statement
        self.L = getattr(solution_instance, 'L', 0.2)  # length in meters
        self.I = getattr(solution_instance, 'I', 0.05)  # current in amperes
        self.F_mag = getattr(solution_instance, 'F', 4e-3)  # magnetic force in newtons
        self.theta = getattr(solution_instance, 'theta', 90.0)  # angle in degrees (perpendicular)
        
        # Safely extract results with fallback
        self.B = getattr(solution_instance, 'B', None)
        if self.B is None:
            self.B = getattr(solution_instance, 'results', {}).get('B', None)
        
        # Expected values from problem
        self.expected_B = 0.4  # Tesla
        self.expected_statement = "The magnetic induction intensity is only related to the magnetic field itself and does not change."
        
        # SymPy symbols for symbolic checks
        self.F_sym, self.I_sym, self.L_sym, self.B_sym, self.theta_sym = sp.symbols('F I L B theta')
        self.magnetic_force_formula = sp.Eq(self.F_sym, self.I_sym * self.L_sym * self.B_sym * sp.sin(self.theta_sym))
        
    def test_invariants(self):
        """
        Test fundamental physical invariants:
        1. Magnetic force formula F = I*L*B*sin(theta)
        2. Consistency of B value
        """
        logs = []
        success = True
        
        try:
            # Convert angle to radians for calculations
            theta_rad = math.radians(self.theta)
            
            # Test 1: Check if B value is provided and reasonable
            if self.B is None:
                logs.append("WARNING: No B value found in solution")
                success = False
            else:
                # Check B is positive
                if self.B <= 0:
                    logs.append(f"ERROR: B must be positive, got {self.B}")
                    success = False
                else:
                    logs.append(f"INFO: B value is {self.B} T")
            
            # Test 2: Verify magnetic force formula numerically
            # F_expected = I * L * B * sin(theta)
            if self.B is not None:
                F_calculated = self.I * self.L * self.B * math.sin(theta_rad)
                if not math.isclose(self.F_mag, F_calculated, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"ERROR: Magnetic force formula mismatch. F_given={self.F_mag}, F_calculated={F_calculated}")
                    success = False
                else:
                    logs.append(f"INFO: Magnetic force formula verified: F={self.F_mag} ≈ I*L*B*sinθ={F_calculated}")
            
            # Test 3: Verify angle is 90 degrees (perpendicular)
            if not math.isclose(self.theta, 90.0, rel_tol=1e-2, abs_tol=1.0):
                logs.append(f"WARNING: Angle θ={self.theta}°, expected 90° (perpendicular)")
            else:
                logs.append("INFO: Wire is perpendicular to magnetic field (θ=90°)")
            
            # Test 4: Symbolic check of magnetic force formula
            try:
                # Substitute known values
                subs_dict = {
                    self.I_sym: self.I,
                    self.L_sym: self.L,
                    self.B_sym: self.B if self.B is not None else self.expected_B,
                    self.theta_sym: theta_rad,
                    self.F_sym: self.F_mag
                }
                # Check if formula holds symbolically
                lhs = self.magnetic_force_formula.lhs.subs(subs_dict)
                rhs = self.magnetic_force_formula.rhs.subs(subs_dict)
                if not math.isclose(float(lhs), float(rhs), rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"ERROR: Symbolic formula check failed: {float(lhs)} ≠ {float(rhs)}")
                    success = False
                else:
                    logs.append("INFO: Symbolic magnetic force formula verified")
            except Exception as e:
                logs.append(f"WARNING: Symbolic check failed: {str(e)}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Check consistency between intermediate steps and final answer.
        Based on STEPS ANALYSIS.
        """
        logs = []
        success = True
        
        try:
            # Step 1: Check B calculation from formula B = F/(I*L) since sin(90°)=1
            B_calculated = self.F_mag / (self.I * self.L)
            logs.append(f"INFO: Calculated B from formula: {B_calculated} T")
            
            # Compare with expected value
            if not math.isclose(B_calculated, self.expected_B, rel_tol=1e-2, abs_tol=1e-3):
                logs.append(f"ERROR: B calculation mismatch: {B_calculated} ≠ {self.expected_B}")
                success = False
            
            # Compare with solution's B if available
            if self.B is not None:
                if not math.isclose(self.B, B_calculated, rel_tol=1e-2, abs_tol=1e-3):
                    logs.append(f"ERROR: Solution B={self.B} doesn't match calculated B={B_calculated}")
                    success = False
                else:
                    logs.append(f"INFO: Solution B={self.B} matches calculated value")
            
            # Step 2: Check consistency with magnetic field property statement
            # The second target answer states B only depends on magnetic field
            # We can check if changing wire properties would change B
            B_independent_check = True
            test_current = self.I * 2  # Double the current
            test_length = self.L * 2   # Double the length
            
            # With same force, B should remain the same if recalculated
            B_test1 = self.F_mag / (test_current * self.L)
            B_test2 = self.F_mag / (self.I * test_length)
            
            if not math.isclose(B_test1, B_calculated, rel_tol=1e-2, abs_tol=1e-3):
                logs.append("WARNING: B changes with current change - contradicts statement")
                B_independent_check = False
                
            if not math.isclose(B_test2, B_calculated, rel_tol=1e-2, abs_tol=1e-3):
                logs.append("WARNING: B changes with length change - contradicts statement")
                B_independent_check = False
            
            if B_independent_check:
                logs.append("INFO: B independence property verified")
            else:
                logs.append("INFO: B shows dependence on wire parameters in calculation")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Test monotonic relationships by perturbing inputs.
        """
        logs = []
        success = True
        
        try:
            # Store original values
            original_F = self.F_mag
            original_I = self.I
            original_L = self.L
            
            # Test 1: Increase force by 10%, B should increase proportionally
            perturbed_F = original_F * 1.1
            B_perturbed = perturbed_F / (original_I * original_L)
            B_original = original_F / (original_I * original_L)
            
            if B_perturbed <= B_original:
                logs.append(f"ERROR: B didn't increase when F increased: {B_perturbed} <= {B_original}")
                success = False
            else:
                logs.append(f"INFO: Monotonicity check 1 passed: F↑ → B↑ ({B_original}→{B_perturbed})")
            
            # Test 2: Increase current by 10%, B should decrease (B = F/(I*L))
            perturbed_I = original_I * 1.1
            B_perturbed = original_F / (perturbed_I * original_L)
            
            if B_perturbed >= B_original:
                logs.append(f"ERROR: B didn't decrease when I increased: {B_perturbed} >= {B_original}")
                success = False
            else:
                logs.append(f"INFO: Monotonicity check 2 passed: I↑ → B↓ ({B_original}→{B_perturbed})")
            
            # Test 3: Increase length by 10%, B should decrease (B = F/(I*L))
            perturbed_L = original_L * 1.1
            B_perturbed = original_F / (original_I * perturbed_L)
            
            if B_perturbed >= B_original:
                logs.append(f"ERROR: B didn't decrease when L increased: {B_perturbed} >= {B_original}")
                success = False
            else:
                logs.append(f"INFO: Monotonicity check 3 passed: L↑ → B↓ ({B_original}→{B_perturbed})")
            
            # Restore original values
            self.F_mag = original_F
            self.I = original_I
            self.L = original_L
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """
        Run all tests and return consolidated results.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                for log in logs:
                    all_logs.append(f"  {log}")
            except Exception as e:
                all_logs.append(f"  ERROR: Test crashed - {str(e)}")
                all_logs.append(f"  {traceback.format_exc()}")
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall success: {overall_success}")
        
        return overall_success, all_logs

# Example usage (commented out):
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.L = 0.2
#             self.I = 0.05
#             self.F = 4e-3
#             self.theta = 90.0
#             self.results = {'B': 0.4}
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)