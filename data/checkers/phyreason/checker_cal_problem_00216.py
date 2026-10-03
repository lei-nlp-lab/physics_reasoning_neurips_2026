import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.n = getattr(solution_instance, 'n', 2.0)  # refractive index
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.C = self.results.get('C', 0.0)  # critical angle
        self.theta = self.results.get('theta', 0.0)  # maximum θ
        # Store original for monotonicity test
        self._original_n = self.n
        self._original_theta = self.theta
        
    def test_invariants(self):
        """Verify total internal reflection physics."""
        logs = []
        success = True
        
        try:
            # 1. Critical angle invariant: sin(C) = 1/n
            # Assume C is in degrees (as per problem statement and target answer)
            if self.C > 0:
                # Convert C to radians for sin calculation
                C_rad = math.radians(self.C)
                lhs = math.sin(C_rad)
                rhs = 1.0 / self.n
                if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Critical angle mismatch: sin({self.C}°)={lhs:.4f}, 1/n={rhs:.4f}")
                    success = False
                else:
                    logs.append(f"✓ Critical angle correct: sin({self.C}°) = {lhs:.4f} ≈ 1/{self.n} = {rhs:.4f}")
            
            # 2. Geometric invariant: C + θ = 90°
            # Both C and θ are in degrees
            angle_sum = self.C + self.theta
            target = 90.0  # Degrees
            if not math.isclose(angle_sum, target, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Geometric invariant fails: C + θ = {angle_sum:.2f}°, expected {target:.2f}°")
                success = False
            else:
                logs.append(f"✓ Geometric invariant: C + θ = {angle_sum:.2f}° ≈ {target:.2f}°")
                
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            # Compute critical angle from n (in degrees)
            C_calc = math.degrees(math.asin(1.0 / self.n))
            
            # Check if stored C matches computed
            if self.C > 0:
                if not math.isclose(self.C, C_calc, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Critical angle mismatch: stored {self.C:.2f}°, computed {C_calc:.2f}°")
                    success = False
                else:
                    logs.append(f"✓ Critical angle consistent: {self.C:.2f}°")
            
            # Check θ = 90° - C
            theta_from_C = 90.0 - C_calc
            if not math.isclose(self.theta, theta_from_C, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Theta mismatch: stored {self.theta:.2f}°, from C: {theta_from_C:.2f}°")
                success = False
            else:
                logs.append(f"✓ Theta consistent: θ = 90° - C = {self.theta:.2f}°")
                
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb refractive index and check trend."""
        logs = []
        success = True
        
        try:
            # Save original values
            original_n = self.n
            original_theta = self._original_theta
            
            # Test 1: Increase n (should INCREASE θ, not decrease)
            self.n = original_n * 1.1  # 10% increase
            C_new = math.degrees(math.asin(1.0 / self.n))
            theta_new = 90.0 - C_new
            
            # Compare the NEW theta with the ORIGINAL theta
            # CORRECTED: When n increases, θ should INCREASE
            if not theta_new > original_theta:
                logs.append(f"Monotonicity fails: n↑ from {original_n} to {self.n:.2f}, θ should increase but was {theta_new:.2f}° vs {original_theta:.2f}°")
                success = False
            else:
                logs.append(f"✓ n↑ trend correct: n={self.n:.2f} → θ={theta_new:.2f}° (increased from {original_theta:.2f}°)")
            
            # Test 2: Decrease n (should DECREASE θ, not increase)
            self.n = original_n * 0.9  # 10% decrease
            C_new = math.degrees(math.asin(1.0 / self.n))
            theta_new = 90.0 - C_new
            
            # Compare the NEW theta with the ORIGINAL theta
            # CORRECTED: When n decreases, θ should DECREASE
            if not theta_new < original_theta:
                logs.append(f"Monotonicity fails: n↓ from {original_n} to {self.n:.2f}, θ should decrease but was {theta_new:.2f}° vs {original_theta:.2f}°")
                success = False
            else:
                logs.append(f"✓ n↓ trend correct: n={self.n:.2f} → θ={theta_new:.2f}° (decreased from {original_theta:.2f}°)")
            
            # Restore original n
            self.n = original_n
            logs.append(f"Restored n to {self.n}")
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            # Ensure restoration
            self.n = self._original_n
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        all_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    all_success = False
                    all_logs.append(f"✗ {test_name} failed")
                else:
                    all_logs.append(f"✓ {test_name} passed")
            except Exception as e:
                all_logs.append(f"✗ {test_name} crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_success = False
        
        # Final check against target answer (60°)
        if not math.isclose(self.theta, 60.0, rel_tol=0.02, abs_tol=0.02):
            all_logs.append(f"\n✗ Final answer mismatch: θ={self.theta:.2f}°, expected 60.0°")
            all_success = False
        else:
            all_logs.append(f"\n✓ Final answer correct: θ={self.theta:.2f}°")
        
        return all_success, all_logs