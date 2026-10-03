import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallback values
        self.U = getattr(solution_instance, 'U', 220.0)  # Generator voltage (V)
        self.P = getattr(solution_instance, 'P', 44000.0)  # Output power (W)
        self.R = getattr(solution_instance, 'R', 0.4)  # Line resistance (Ω)
        
        # Safely extract results from solution_instance
        sol_results = getattr(solution_instance, 'results', {})
        
        # Without transformer case
        self.I = getattr(solution_instance, 'I', sol_results.get('I', 0.0))
        self.U_loss = getattr(solution_instance, 'U_loss', sol_results.get('U_loss', 0.0))
        self.P_loss = getattr(solution_instance, 'P_loss', sol_results.get('P_loss', 0.0))
        self.U_user = getattr(solution_instance, 'U_user', sol_results.get('U_user', 0.0))
        self.P_user = getattr(solution_instance, 'P_user', sol_results.get('P_user', 0.0))
        
        # With transformer case
        self.U_1 = getattr(solution_instance, 'U_1', sol_results.get('U_1', 0.0))
        self.I_1 = getattr(solution_instance, 'I_1', sol_results.get('I_1', 0.0))
        self.U_loss_prime = getattr(solution_instance, "U_loss'", sol_results.get("U_loss'", sol_results.get('U_loss_prime', 0.0)))
        self.P_loss_prime = getattr(solution_instance, "P_loss'", sol_results.get("P_loss'", sol_results.get('P_loss_prime', 0.0)))
        self.U_2 = getattr(solution_instance, 'U_2', sol_results.get('U_2', 0.0))
        self.P_2 = getattr(solution_instance, 'P_2', sol_results.get('P_2', 0.0))
        self.U_user_prime = getattr(solution_instance, "U_user'", sol_results.get("U_user'", sol_results.get('U_user_prime', 0.0)))
        self.P_user_prime = getattr(solution_instance, "P_user'", sol_results.get("P_user'", sol_results.get('P_user_prime', 0.0)))
        
        # Log storage
        self.logs = []
        
    def log(self, message):
        """Add message to logs."""
        self.logs.append(message)
        
    def check_close(self, a, b, name="", rel_tol=1e-2, abs_tol=1e-2):
        """Check if two values are close with tolerance."""
        if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            self.log(f"✓ {name}: {a} ≈ {b}")
            return True
        else:
            self.log(f"✗ {name}: {a} ≠ {b} (outside tolerance)")
            return False
    
    def test_invariants(self):
        """Check physical invariants from KB."""
        success = True
        
        # Symbolic variables
        I, U, R, P, U_loss, P_loss, U_user, P_user = sp.symbols('I U R P U_loss P_loss U_user P_user')
        U_1, I_1, U_loss_prime, P_loss_prime, U_2, P_2, U_user_prime, P_user_prime = sp.symbols(
            'U_1 I_1 U_loss_prime P_loss_prime U_2 P_2 U_user_prime P_user_prime'
        )
        
        try:
            # Without transformer invariants
            # 1. Ohm's Law: U_loss = I * R
            if not self.check_close(self.U_loss, self.I * self.R, "Ohm's Law (U_loss = I·R)"):
                success = False
                
            # 2. Power formulas: P = U * I
            if not self.check_close(self.P, self.U * self.I, "Power (P = U·I)"):
                success = False
                
            # 3. Power loss: P_loss = I² * R
            if not self.check_close(self.P_loss, self.I**2 * self.R, "Power loss (P_loss = I²·R)"):
                success = False
                
            # 4. Series voltage: U_user = U - U_loss
            if not self.check_close(self.U_user, self.U - self.U_loss, "Series voltage (U_user = U - U_loss)"):
                success = False
                
            # 5. Power conservation: P_user = P - P_loss
            if not self.check_close(self.P_user, self.P - self.P_loss, "Power conservation (P_user = P - P_loss)"):
                success = False
                
            # With transformer invariants
            # 6. Transformer power conservation (step-up): P ≈ U_1 * I_1
            # Note: Transformers are ideal in this problem
            if not self.check_close(self.P, self.U_1 * self.I_1, 
                                   "Transformer power conservation (P ≈ U_1·I_1)", rel_tol=5e-2):
                success = False
                
            # 7. Transformer turns ratio consistency
            # From step-up: U_1/U ≈ I/I_1 (since power is conserved)
            if self.I_1 != 0 and not self.check_close(self.U_1/self.U, self.I/self.I_1,
                                                     "Turns ratio (U_1/U ≈ I/I_1)", rel_tol=5e-2):
                success = False
                
            # 8. Ohm's Law for loss: U_loss' = I_1 * R
            if not self.check_close(self.U_loss_prime, self.I_1 * self.R, "Ohm's Law (U_loss' = I_1·R)"):
                success = False
                
            # 9. Power loss: P_loss' = I_1² * R
            if not self.check_close(self.P_loss_prime, self.I_1**2 * self.R, "Power loss (P_loss' = I_1²·R)"):
                success = False
                
            # 10. Series voltage: U_2 = U_1 - U_loss'
            if not self.check_close(self.U_2, self.U_1 - self.U_loss_prime, "Series voltage (U_2 = U_1 - U_loss')"):
                success = False
                
            # 11. Power conservation: P_2 = P - P_loss'
            if not self.check_close(self.P_2, self.P - self.P_loss_prime, "Power conservation (P_2 = P - P_loss')"):
                success = False
                
            # 12. Step-down transformer: U_user' should be consistent with U_2 and turns ratio
            # Since the step-down transformer reverses the step-up, we expect U_user' ≈ U
            # but accounting for losses: U_user' ≈ U_2 * (U/U_1) = U_2 * (1/10) if turns ratio is 10:1
            # Actually from steps: U_user' = U_2 * (U/U_1) = 2192 * (220/2200) = 219.2
            expected_U_user_prime = self.U_2 * (self.U / self.U_1)
            if not self.check_close(self.U_user_prime, expected_U_user_prime, 
                                   "Step-down transformer voltage ratio"):
                success = False
                
            # 13. Final power conservation: P_user' ≈ P_2
            if not self.check_close(self.P_user_prime, self.P_2, "Final power conservation (P_user' ≈ P_2)"):
                success = False
                
        except Exception as e:
            self.log(f"Error in invariants: {str(e)}")
            success = False
            
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Expected values from steps analysis
        expected = {
            'I': 200.0,
            'U_loss': 80.0,
            'P_loss': 16000.0,
            'U_user': 140.0,
            'P_user': 28000.0,
            'U_1': 2200.0,
            'I_1': 20.0,
            'U_loss_prime': 8.0,
            'P_loss_prime': 160.0,
            'U_2': 2192.0,
            'P_2': 43840.0,
            'U_user_prime': 219.2,
            'P_user_prime': 43840.0
        }
        
        # Compare each value with expected
        for key, exp_val in expected.items():
            actual = getattr(self, key, None)
            if actual is not None:
                if not self.check_close(actual, exp_val, f"Trace {key}"):
                    success = False
            else:
                self.log(f"✗ Trace {key}: Missing value")
                success = False
                
        # Check final answers match target
        target1 = (140.0, 28000.0)
        target2 = (219.2, 43840.0)
        
        if not (self.check_close(self.U_user, target1[0], "Final U_user") and 
                self.check_close(self.P_user, target1[1], "Final P_user")):
            success = False
            
        if not (self.check_close(self.U_user_prime, target2[0], "Final U_user'") and 
                self.check_close(self.P_user_prime, target2[1], "Final P_user'")):
            success = False
            
        return success
    
    def test_monotonicity(self):
        """Perturb resistance and check trends."""
        success = True
        original_R = self.R
        
        try:
            # Test 1: Increase resistance by 10%
            perturbed_R = original_R * 1.1
            
            # Without transformer case (I remains same since P and U fixed)
            original_P_loss = self.I**2 * original_R
            perturbed_P_loss = self.I**2 * perturbed_R
            
            if perturbed_P_loss > original_P_loss:
                self.log("✓ Monotonicity: P_loss increases with R (no transformer)")
            else:
                self.log(f"✗ Monotonicity: P_loss {perturbed_P_loss} ≤ {original_P_loss}")
                success = False
                
            # With transformer case (I_1 would change if we recalculated, but here we test the relationship)
            # Since P_loss' = I_1² * R and I_1 = P/U_1 (fixed if U_1 fixed), but U_1 depends on transformer ratio
            # Actually, with fixed transformer ratio, I_1 is fixed if P is fixed, so P_loss' ∝ R
            original_P_loss_prime = self.I_1**2 * original_R
            perturbed_P_loss_prime = self.I_1**2 * perturbed_R
            
            if perturbed_P_loss_prime > original_P_loss_prime:
                self.log("✓ Monotonicity: P_loss' increases with R (with transformer)")
            else:
                self.log(f"✗ Monotonicity: P_loss' {perturbed_P_loss_prime} ≤ {original_P_loss_prime}")
                success = False
                
        except Exception as e:
            self.log(f"Error in monotonicity test: {str(e)}")
            success = False
        finally:
            # Restore original value
            self.R = original_R
            
        return success
    
    def run_all_tests(self):
        """Run all tests and return results."""
        self.logs = []
        all_success = True
        
        try:
            self.log("=== PhysicsChecker Starting ===")
            
            # Run individual tests
            tests = [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]
            
            for test_name, test_func in tests:
                self.log(f"\n--- Running {test_name} ---")
                try:
                    if not test_func():
                        all_success = False
                        self.log(f"✗ {test_name} FAILED")
                except Exception as e:
                    self.log(f"✗ {test_name} ERROR: {str(e)}")
                    self.log(traceback.format_exc())
                    all_success = False
            
            self.log(f"\n=== PhysicsChecker Complete: {'PASS' if all_success else 'FAIL'} ===")
            
        except Exception as e:
            self.log(f"Critical error: {str(e)}")
            self.log(traceback.format_exc())
            all_success = False
            
        return all_success, self.logs