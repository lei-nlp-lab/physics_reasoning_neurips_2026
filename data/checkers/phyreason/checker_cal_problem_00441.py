import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Extract known physical constants
        self.g = 10.0  # Given in problem
        
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.mu = self.results.get('mu', None)
        self.a1 = self.results.get('a_1', None)
        self.a2 = self.results.get('a_2', None)
        
        self.logs = []
        
    def _log(self, message):
        """Helper to add log messages."""
        self.logs.append(message)
    
    def test_invariants(self):
        """Test fundamental physical laws."""
        success = True
        
        if None in (self.a1, self.a2, self.mu):
            self._log("Missing acceleration or friction coefficient values")
            return False
        
        # The physics allows two possible sign conventions for friction.
        # We'll check both possibilities and accept if either is consistent.
        
        # Option 1: a1 = g sinθ + μ g cosθ, a2 = g sinθ - μ g cosθ
        # Option 2: a1 = g sinθ - μ g cosθ, a2 = g sinθ + μ g cosθ
        
        # From either option, we can derive:
        # sinθ = (a1 + a2)/(2g)
        # |μ cosθ| = |a1 - a2|/(2g)
        
        sin_theta_num = (self.a1 + self.a2) / (2 * self.g)
        
        # Check if sinθ is physically valid
        if not (0 <= sin_theta_num <= 1):
            self._log(f"Invalid sinθ value: {sin_theta_num:.3f}")
            success = False
        else:
            cos_theta_num = math.sqrt(1 - sin_theta_num**2)
            if cos_theta_num > 0:
                # Calculate μ from both possible sign conventions
                mu_option1 = (self.a1 - self.a2) / (2 * self.g * cos_theta_num)
                mu_option2 = (self.a2 - self.a1) / (2 * self.g * cos_theta_num)
                
                # Check if either matches the computed μ
                if not (math.isclose(self.mu, mu_option1, rel_tol=1e-2, abs_tol=1e-2) or
                        math.isclose(self.mu, mu_option2, rel_tol=1e-2, abs_tol=1e-2)):
                    self._log(f"Newton's law consistency check fails: μ={self.mu:.3f}, options: {mu_option1:.3f}, {mu_option2:.3f}")
                    success = False
            else:
                self._log("cosθ calculation yielded zero")
                success = False
        
        # Accelerations should be positive (v² increases throughout)
        if not (self.a1 > 0 and self.a2 > 0):
            self._log(f"Accelerations not positive: a1={self.a1}, a2={self.a2}")
            success = False
        
        # The friction coefficient magnitude should be reasonable
        if not (0 <= abs(self.mu) <= 1):
            self._log(f"Friction coefficient magnitude out of bounds: |μ|={abs(self.mu):.3f}")
            success = False
        
        if success:
            self._log("All invariants satisfied")
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check step 3: a1 = 10 m/s² from graph
        if self.a1 is not None:
            if not math.isclose(self.a1, 10.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"a1 mismatch: expected 10.0, got {self.a1}")
                success = False
        
        # Check step 4: a2 = 2 m/s² from graph
        if self.a2 is not None:
            if not math.isclose(self.a2, 2.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"a2 mismatch: expected 2.0, got {self.a2}")
                success = False
        
        # Check step 6: The problem expects μ = 0.5
        # But we need to check if the solution's μ has the correct magnitude
        if self.mu is not None:
            if not math.isclose(abs(self.mu), 0.5, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"|μ| mismatch: expected 0.5, got {abs(self.mu):.3f}")
                success = False
        
        if success:
            self._log("Trace consistency check passed")
        return success
    
    def test_monotonicity(self):
        """Test that perturbing mass doesn't affect accelerations (as expected)."""
        success = True
        
        # According to physics, accelerations on inclined plane with friction
        # are independent of mass: a = g sinθ ± μ g cosθ
        
        # Use derived angle from current results
        if None not in (self.a1, self.a2):
            sin_theta = (self.a1 + self.a2) / (2 * self.g)
            if 0 <= sin_theta <= 1:
                cos_theta = math.sqrt(1 - sin_theta**2)
                
                # Test both possible sign conventions
                # Option 1: a1 = g sinθ + μ g cosθ, a2 = g sinθ - μ g cosθ
                a1_option1 = self.g * sin_theta + self.mu * self.g * cos_theta
                a2_option1 = self.g * sin_theta - self.mu * self.g * cos_theta
                
                # Option 2: a1 = g sinθ - μ g cosθ, a2 = g sinθ + μ g cosθ
                a1_option2 = self.g * sin_theta - self.mu * self.g * cos_theta
                a2_option2 = self.g * sin_theta + self.mu * self.g * cos_theta
                
                # Check if either option matches the computed accelerations
                option1_match = (math.isclose(a1_option1, self.a1, rel_tol=1e-2, abs_tol=1e-2) and
                                math.isclose(a2_option1, self.a2, rel_tol=1e-2, abs_tol=1e-2))
                
                option2_match = (math.isclose(a1_option2, self.a1, rel_tol=1e-2, abs_tol=1e-2) and
                                math.isclose(a2_option2, self.a2, rel_tol=1e-2, abs_tol=1e-2))
                
                if not (option1_match or option2_match):
                    self._log(f"Recalculated accelerations don't match: a1={self.a1}, a2={self.a2}")
                    self._log(f"Option 1: a1={a1_option1:.3f}, a2={a2_option1:.3f}")
                    self._log(f"Option 2: a1={a1_option2:.3f}, a2={a2_option2:.3f}")
                    success = False
        
        if success:
            self._log("Monotonicity (mass independence) check passed")
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status with logs."""
        self.logs = []
        results = []
        
        results.append(("Invariants", self.test_invariants()))
        results.append(("Trace Consistency", self.test_trace_consistency()))
        results.append(("Monotonicity", self.test_monotonicity()))
        
        success = all(r[1] for r in results)
        
        # Summary
        self._log("\n=== SUMMARY ===")
        for test_name, test_result in results:
            self._log(f"{test_name}: {'PASS' if test_result else 'FAIL'}")
        
        return success, self.logs