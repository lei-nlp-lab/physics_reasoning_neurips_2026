import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs
        self.AC = 4.0  # km
        self.BC = 6.0  # km
        self.DE = 5.0  # km
        self.v = 60.0  # km/h
        
        # Extract results safely
        self.s_criminal = getattr(solution_instance, 's_criminal', self.results.get('s_criminal', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.CD = getattr(solution_instance, 'CD', self.results.get('CD', 0.0))
        self.s_police = getattr(solution_instance, 's_police', self.results.get('s_police', 0.0))
        self.NA = getattr(solution_instance, 'N/A', self.results.get('N/A', None))
        
        # Store logs for reporting
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(f"  {message}")
    
    def test_invariants(self):
        """Check physical invariants from KB theorems."""
        success = True
        self._log("=== Testing Invariants ===")
        
        try:
            # 1. Equal distance-speed-time relationship for equal speeds
            # Both vehicles travel at same speed and meet at E simultaneously
            if not math.isclose(self.s_police, self.s_criminal, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distances should be equal: s_police={self.s_police:.3f}, s_criminal={self.s_criminal:.3f}", False)
                success = False
            else:
                self._log(f"Equal distances check: s_police={self.s_police:.3f} ≈ s_criminal={self.s_criminal:.3f}", True)
            
            # 2. Verify distance calculations match Pythagorean theorem
            s_police_calc = math.sqrt(self.BC**2 + (self.CD + self.DE)**2)
            s_criminal_calc = math.sqrt(self.AC**2 + self.CD**2) + self.DE
            
            if not math.isclose(self.s_police, s_police_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Police distance mismatch: {self.s_police:.3f} vs calculated {s_police_calc:.3f}", False)
                success = False
            else:
                self._log(f"Police distance formula correct: {self.s_police:.3f} ≈ sqrt({self.BC}^2 + ({self.CD}+{self.DE})^2)", True)
            
            if not math.isclose(self.s_criminal, s_criminal_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Criminal distance mismatch: {self.s_criminal:.3f} vs calculated {s_criminal_calc:.3f}", False)
                success = False
            else:
                self._log(f"Criminal distance formula correct: {self.s_criminal:.3f} ≈ sqrt({self.AC}^2 + {self.CD}^2) + {self.DE}", True)
            
            # 3. Verify time calculation from distance and speed
            t_calc_hours = self.s_criminal / self.v
            t_calc_minutes = t_calc_hours * 60
            
            if not math.isclose(self.t, t_calc_minutes, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Time calculation mismatch: t={self.t:.3f} min vs calculated {t_calc_minutes:.3f} min", False)
                success = False
            else:
                self._log(f"Time calculation correct: t={self.t:.3f} min ≈ ({self.s_criminal:.3f} km / {self.v} km/h) × 60", True)
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answer."""
        success = True
        self._log("\n=== Testing Trace Consistency ===")
        
        try:
            # Check that CD satisfies the equation from step_2
            lhs = math.sqrt(self.BC**2 + (self.CD + self.DE)**2)
            rhs = math.sqrt(self.AC**2 + self.CD**2) + self.DE
            
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"CD={self.CD:.3f} doesn't satisfy equation: LHS={lhs:.3f}, RHS={rhs:.3f}", False)
                success = False
            else:
                self._log(f"CD={self.CD:.3f} satisfies equation: sqrt({self.BC}^2+({self.CD}+{self.DE})^2) = sqrt({self.AC}^2+{self.CD}^2)+{self.DE}", True)
            
            # Verify CD is positive and reasonable
            if self.CD <= 0:
                self._log(f"CD should be positive: {self.CD:.3f}", False)
                success = False
            
            # Verify t is positive and reasonable (should be ~10 minutes)
            if self.t <= 0 or self.t > 60:  # Should be around 10 min
                self._log(f"Time t={self.t:.3f} min seems unrealistic", False)
                success = False
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Test that perturbations to inputs produce expected trends."""
        success = True
        self._log("\n=== Testing Monotonicity ===")
        
        try:
            # Perturb AC (distance from crime scene to line CD)
            original_AC = self.AC
            perturbed_AC = original_AC * 1.1  # Increase by 10%
            
            # Recalculate CD for perturbed AC by solving the equation
            # sqrt(BC^2 + (CD + DE)^2) = sqrt(AC^2 + CD^2) + DE
            
            # Define the equation symbolically
            CD_sym = sp.Symbol('CD', real=True, positive=True)
            equation = sp.Eq(
                sp.sqrt(self.BC**2 + (CD_sym + self.DE)**2),
                sp.sqrt(perturbed_AC**2 + CD_sym**2) + self.DE
            )
            
            # Solve numerically
            solutions = sp.nsolve(equation, 3.0)  # Initial guess 3.0
            perturbed_CD = float(solutions)
            
            # Calculate time for perturbed case
            perturbed_s_criminal = math.sqrt(perturbed_AC**2 + perturbed_CD**2) + self.DE
            perturbed_t = (perturbed_s_criminal / self.v) * 60  # Convert to minutes
            
            self._log(f"Perturbed AC: {original_AC:.2f} → {perturbed_AC:.2f} km (+10%)", None)
            self._log(f"  CD changed: {self.CD:.3f} → {perturbed_CD:.3f} km", None)
            self._log(f"  Time changed: {self.t:.3f} → {perturbed_t:.3f} min", None)
            
            # Check trend: If AC increases, criminal's path gets longer, so time should increase
            if perturbed_t > self.t:
                self._log("  Trend correct: Increased AC → increased time", True)
            else:
                self._log(f"  Trend violation: Time decreased when AC increased", False)
                success = False
            
            # Restore original AC for any future tests
            # (not strictly needed as we don't modify self.AC)
            
            # Second perturbation: DE (road segment from D to E)
            original_DE = self.DE
            perturbed_DE = original_DE * 1.1  # Increase by 10%
            
            # Solve for new CD with perturbed DE
            equation2 = sp.Eq(
                sp.sqrt(self.BC**2 + (CD_sym + perturbed_DE)**2),
                sp.sqrt(self.AC**2 + CD_sym**2) + perturbed_DE
            )
            
            solutions2 = sp.nsolve(equation2, 3.0)
            perturbed_CD2 = float(solutions2)
            
            perturbed_s_criminal2 = math.sqrt(self.AC**2 + perturbed_CD2**2) + perturbed_DE
            perturbed_t2 = (perturbed_s_criminal2 / self.v) * 60
            
            self._log(f"\nPerturbed DE: {original_DE:.2f} → {perturbed_DE:.2f} km (+10%)", None)
            self._log(f"  CD changed: {self.CD:.3f} → {perturbed_CD2:.3f} km", None)
            self._log(f"  Time changed: {self.t:.3f} → {perturbed_t2:.3f} min", None)
            
            # Check trend: If DE increases, both paths get longer, so time should increase
            if perturbed_t2 > self.t:
                self._log("  Trend correct: Increased DE → increased time", True)
            else:
                self._log(f"  Trend violation: Time decreased when DE increased", False)
                success = False
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            self._log(traceback.format_exc(), False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)."""
        self.logs = []  # Clear previous logs
        
        overall_success = True
        
        try:
            # Run individual tests
            if not self.test_invariants():
                overall_success = False
            
            if not self.test_trace_consistency():
                overall_success = False
            
            if not self.test_monotonicity():
                overall_success = False
            
            # Summary
            self._log("\n=== Summary ===", None)
            if overall_success:
                self._log("All tests passed!", True)
            else:
                self._log("Some tests failed.", False)
            
        except Exception as e:
            self._log(f"Test suite crashed: {str(e)}", False)
            self._log(traceback.format_exc(), False)
            overall_success = False
        
        return overall_success, self.logs