import math
import sympy as sp
from sympy import symbols, Eq, solve

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get()
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters from problem statement
        self.a = 2.0  # m/s^2
        self.t1 = 10.0  # s
        self.t2 = 41.0  # s
        self.v0 = 340.0  # m/s (speed of sound)
        
        # Extract results with safe fallbacks
        self.v = self._extract_value('v')
        self.x = self._extract_value('x')
        
        # Expected target values
        self.expected_v = 20.0  # m/s
        self.expected_x = 680.0  # m
        
        # Logs for reporting
        self.logs = []
    
    def _extract_value(self, key):
        """Safely extract a value from results."""
        # First try direct key access
        val = getattr(self.solution, key, None)
        if val is not None:
            return float(val)
        
        # Then try results dictionary
        val = self.results.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        
        # Return 0.0 as safe default for missing values
        return 0.0
    
    def _log(self, message, success=None):
        """Add a log message with optional success flag."""
        if success is not None:
            prefix = "✓" if success else "✗"
            message = f"{prefix} {message}"
        self.logs.append(message)
        return success
    
    def test_invariants(self):
        """Test fundamental physical invariants."""
        success = True
        
        # 1. Check velocity from uniform acceleration (step 1)
        calculated_v = self.a * self.t1
        if math.isclose(self.v, calculated_v, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Velocity check: v = a*t1 = {calculated_v} ≈ {self.v}", True)
        else:
            success = False
            self._log(f"Velocity check FAILED: v = a*t1 = {calculated_v} ≠ {self.v}", False)
        
        # 2. Check consistency between ambulance motion and sound travel (steps 2-4)
        # Let Δt be the time from t1 to when siren stops
        # From ambulance motion: x = 0.5*a*t1² + v*Δt
        # From sound travel: x = v0*(t2 - t1 - Δt)
        # Equate and solve for Δt symbolically
        Δt = symbols('Δt')
        eq1 = Eq(0.5*self.a*self.t1**2 + self.v*Δt, self.x)
        eq2 = Eq(self.v0*(self.t2 - self.t1 - Δt), self.x)
        
        # Solve for Δt from both equations
        try:
            Δt1 = solve(eq1, Δt)[0]
            Δt2 = solve(eq2, Δt)[0]
            
            if math.isclose(float(Δt1), float(Δt2), rel_tol=0.02, abs_tol=0.02):
                self._log(f"Δt consistency: {float(Δt1):.2f}s ≈ {float(Δt2):.2f}s", True)
                
                # Check if Δt is positive and less than t2-t1
                Δt_val = float(Δt1)
                if 0 < Δt_val < (self.t2 - self.t1):
                    self._log(f"Δt = {Δt_val:.2f}s is physically valid (0 < Δt < {self.t2-self.t1}s)", True)
                else:
                    success = False
                    self._log(f"Δt = {Δt_val:.2f}s is not in valid range (0, {self.t2-self.t1})", False)
            else:
                success = False
                self._log(f"Δt inconsistency: {float(Δt1):.2f}s ≠ {float(Δt2):.2f}s", False)
        except Exception as e:
            success = False
            self._log(f"Symbolic check failed: {str(e)}", False)
        
        return success
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final answer."""
        success = True
        
        # Check if final velocity matches target
        if math.isclose(self.v, self.expected_v, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Velocity matches target: {self.v} ≈ {self.expected_v}", True)
        else:
            success = False
            self._log(f"Velocity mismatch: {self.v} ≠ {self.expected_v}", False)
        
        # Check if final position matches target
        if math.isclose(self.x, self.expected_x, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Position matches target: {self.x} ≈ {self.expected_x}", True)
        else:
            success = False
            self._log(f"Position mismatch: {self.x} ≠ {self.expected_x}", False)
        
        # Verify the equation from step 4: x = 680 m leads to consistent Δt
        if math.isclose(self.x, 680.0, rel_tol=0.02, abs_tol=0.02):
            # Calculate Δt from ambulance motion
            Δt_from_x = (self.x - 0.5*self.a*self.t1**2) / self.v
            
            # Calculate sound travel time
            sound_time = self.x / self.v0
            
            # Check time consistency: t1 + Δt + sound_time = t2
            total_time = self.t1 + Δt_from_x + sound_time
            if math.isclose(total_time, self.t2, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Time consistency: t1 + Δt + x/v0 = {total_time:.2f} ≈ t2 = {self.t2}", True)
            else:
                success = False
                self._log(f"Time inconsistency: {total_time:.2f} ≠ {self.t2}", False)
        
        return success
    
    def test_monotonicity(self):
        """Test if increasing acceleration increases distance traveled."""
        success = True
        
        try:
            # Store original values
            original_a = self.a
            original_v = self.v
            original_x = self.x
            
            # Perturb acceleration by +10%
            perturbed_a = original_a * 1.1
            
            # Calculate new velocity at t1
            perturbed_v = perturbed_a * self.t1
            
            # Solve for new Δt using symbolic equations
            Δt = symbols('Δt')
            # Equation: 0.5*perturbed_a*t1² + perturbed_v*Δt = v0*(t2 - t1 - Δt)
            equation = Eq(
                0.5*perturbed_a*self.t1**2 + perturbed_v*Δt,
                self.v0*(self.t2 - self.t1 - Δt)
            )
            
            Δt_solution = solve(equation, Δt)
            if Δt_solution:
                perturbed_Δt = float(Δt_solution[0])
                perturbed_x = 0.5*perturbed_a*self.t1**2 + perturbed_v*perturbed_Δt
                
                # Check monotonicity: increased a should increase x
                if perturbed_x > original_x:
                    self._log(f"Monotonicity: a↑ {original_a}→{perturbed_a:.2f} => x↑ {original_x:.2f}→{perturbed_x:.2f}", True)
                else:
                    success = False
                    self._log(f"Monotonicity violation: a↑ but x↓ {original_x:.2f}→{perturbed_x:.2f}", False)
                
                # Additional check: velocity should also increase
                if perturbed_v > original_v:
                    self._log(f"Velocity monotonic: v↑ {original_v:.2f}→{perturbed_v:.2f}", True)
                else:
                    success = False
                    self._log(f"Velocity monotonicity violation", False)
            else:
                success = False
                self._log("Could not solve perturbed equations", False)
                
        except Exception as e:
            success = False
            self._log(f"Monotonicity test failed: {str(e)}", False)
        
        # Restore original values (though we didn't modify instance)
        self.a = original_a
        self.v = original_v
        self.x = original_x
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Overall success
        overall_success = test1 and test2 and test3
        
        # Summary
        self._log("\n=== SUMMARY ===")
        self._log(f"Invariants test: {'PASS' if test1 else 'FAIL'}")
        self._log(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
        self._log(f"Monotonicity test: {'PASS' if test3 else 'FAIL'}")
        self._log(f"OVERALL: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)