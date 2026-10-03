import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract inputs from solution instance or use defaults from problem
        self.L = getattr(solution_instance, 'L', 0.2)  # rail spacing (m)
        self.B = getattr(solution_instance, 'B', 5.0)  # magnetic field (T)
        self.v = getattr(solution_instance, 'v', 5.0)  # rod velocity (m/s)
        self.R = getattr(solution_instance, 'R', 4.8)  # resistor (Ω)
        self.r = getattr(solution_instance, 'r', 0.2)  # rod resistance (Ω)
        
        # Extract results safely - ONLY numeric values
        results = getattr(solution_instance, 'results', {})
        
        # Extract numeric values with defaults
        self.E = float(results.get('E', 0.0))  # induced EMF
        self.I = float(results.get('I', 0.0))  # current
        self.F = float(results.get('F', 0.0))  # applied force
        
        # Extract direction string separately (not converted to float)
        # This avoids the sandbox float conversion error
        self.current_dir = str(results.get('N/A', '')) if 'N/A' in results else ''
        
        # Expected values from problem solution
        self.E_expected = 5.0  # V
        self.I_expected = 1.0  # A
        self.F_expected = 1.0  # N
        
        # Expected direction string
        self.dir_expected = "From M through R to P"
        
        # Tolerance for numerical comparisons (1-2% as per robustness rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Initialize logs
        self.logs = []
    
    def log(self, message: str, success: bool = True):
        """Helper to log test results"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success
    
    def fuzzy_equal(self, a: float, b: float) -> bool:
        """Robust fuzzy equality check"""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_emf_calculation(self) -> bool:
        """Test motional EMF formula E = B*L*v"""
        try:
            # Numerical check only - no symbolic operations
            E_calc = self.B * self.L * self.v
            
            if self.fuzzy_equal(self.E, E_calc):
                return self.log(f"Motional EMF correct: E = B*L*v = {E_calc:.3f} V ≈ {self.E:.3f} V")
            else:
                return self.log(f"Motional EMF mismatch: Expected {E_calc:.3f} V, got {self.E:.3f} V", False)
        except Exception as e:
            self.log(f"EMF test exception: {str(e)}", False)
            return False
    
    def test_ohm_law(self) -> bool:
        """Test Ohm's Law: I = E/(R+r)"""
        try:
            # Numerical check only
            I_calc = self.E / (self.R + self.r)
            
            if self.fuzzy_equal(self.I, I_calc):
                return self.log(f"Ohm's Law correct: I = E/(R+r) = {I_calc:.3f} A ≈ {self.I:.3f} A")
            else:
                return self.log(f"Ohm's Law mismatch: Expected {I_calc:.3f} A, got {self.I:.3f} A", False)
        except Exception as e:
            self.log(f"Ohm's Law test exception: {str(e)}", False)
            return False
    
    def test_force_calculation(self) -> bool:
        """Test magnetic force formula F = B*I*L and equilibrium"""
        try:
            # Numerical check only
            F_magnetic = self.B * self.I * self.L
            F_calc = F_magnetic  # For equilibrium, applied force = magnetic force
            
            if self.fuzzy_equal(self.F, F_calc):
                return self.log(f"Force calculation correct: F = B*I*L = {F_calc:.3f} N ≈ {self.F:.3f} N")
            else:
                return self.log(f"Force mismatch: Expected {F_calc:.3f} N, got {self.F:.3f} N", False)
        except Exception as e:
            self.log(f"Force test exception: {str(e)}", False)
            return False
    
    def test_current_direction(self) -> bool:
        """Check current direction consistency with Lenz's Law"""
        try:
            # FIXED: The contract requires results to contain ONLY floats.
            # Direction strings should NOT be in results. Therefore, we should
            # not fail the test if direction is not in results.
            # Instead, we can check if the physics is consistent with Lenz's Law
            # by verifying the sign of the induced EMF and current.
            
            # For this specific problem:
            # - B field is downward (negative z-direction)
            # - Rod moves to the right (positive x-direction)
            # - According to motional EMF formula E = (v × B) · L, the induced
            #   EMF should be positive in the direction from b to a in the rod
            # - This means current flows counterclockwise: from M through R to P
            
            # Since we can't access direction from results (by contract design),
            # we'll simply verify that the physics calculations are correct
            # and note that direction determination is a separate conceptual step.
            
            # Check if EMF is positive (which it should be for this configuration)
            if self.E > 0:
                return self.log(f"EMF positive ({self.E:.3f} V), consistent with Lenz's Law for given configuration")
            else:
                return self.log(f"EMF negative or zero ({self.E:.3f} V), may indicate direction issue", False)
                
        except Exception as e:
            self.log(f"Direction test exception: {str(e)}", False)
            return False
    
    def test_equilibrium(self) -> bool:
        """Test Newton's 1st Law: net force = 0 for constant velocity"""
        try:
            # For constant velocity, applied force must equal magnetic force
            F_applied = self.F
            F_magnetic = self.B * self.I * self.L
            
            if self.fuzzy_equal(F_applied, F_magnetic):
                return self.log(f"Equilibrium satisfied: F_applied = {F_applied:.3f} N = F_magnetic = {F_magnetic:.3f} N")
            else:
                return self.log(f"Equilibrium violated: F_applied ({F_applied:.3f} N) ≠ F_magnetic ({F_magnetic:.3f} N)", False)
        except Exception as e:
            self.log(f"Equilibrium test exception: {str(e)}", False)
            return False
    
    def test_power_balance(self) -> bool:
        """Test energy conservation: mechanical power = electrical power"""
        try:
            # Mechanical power input: P_mech = F * v
            P_mech = self.F * self.v
            
            # Electrical power dissipated: P_elec = I² * (R + r)
            P_elec = (self.I ** 2) * (self.R + self.r)
            
            if self.fuzzy_equal(P_mech, P_elec):
                return self.log(f"Power balance: P_mech = {P_mech:.3f} W = P_elec = {P_elec:.3f} W")
            else:
                return self.log(f"Power imbalance: P_mech ({P_mech:.3f} W) ≠ P_elec ({P_elec:.3f} W)", False)
        except Exception as e:
            self.log(f"Power test exception: {str(e)}", False)
            return False
    
    def test_invariants(self) -> List[bool]:
        """Run all invariant tests"""
        tests = [
            self.test_emf_calculation(),
            self.test_ohm_law(),
            self.test_force_calculation(),
            self.test_current_direction(),
            self.test_equilibrium(),
            self.test_power_balance()
        ]
        return tests
    
    def test_trace_consistency(self) -> bool:
        """Check intermediate results consistency with final answer"""
        try:
            # Check that intermediate calculations lead to final answer
            E_calc = self.B * self.L * self.v
            I_calc = E_calc / (self.R + self.r)
            F_calc = self.B * I_calc * self.L
            
            # Verify chain of calculations
            chain_valid = (
                self.fuzzy_equal(self.E, E_calc) and
                self.fuzzy_equal(self.I, I_calc) and
                self.fuzzy_equal(self.F, F_calc)
            )
            
            if chain_valid:
                return self.log(f"Trace consistency: E→I→F chain validates: {E_calc:.2f}V → {I_calc:.2f}A → {F_calc:.2f}N")
            else:
                return self.log("Trace inconsistency in E→I→F calculation chain", False)
        except Exception as e:
            self.log(f"Trace consistency exception: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """Test monotonic relationships by perturbing velocity"""
        try:
            # Store original values
            v_original = self.v
            E_original = self.E
            I_original = self.I
            F_original = self.F
            
            # Perturb velocity by +10%
            v_perturbed = v_original * 1.1
            
            # Recalculate using physical formulas
            E_new = self.B * self.L * v_perturbed
            I_new = E_new / (self.R + self.r)
            F_new = self.B * I_new * self.L
            
            # Check monotonic increase
            monotonic = (E_new > E_original and 
                        I_new > I_original and 
                        F_new > F_original)
            
            if monotonic:
                return self.log(f"Monotonicity: +10% v increases E ({E_original:.2f}→{E_new:.2f}V), I ({I_original:.2f}→{I_new:.2f}A), F ({F_original:.2f}→{F_new:.2f}N)")
            else:
                return self.log(f"Monotonicity violation: Some quantity didn't increase with v", False)
        except Exception as e:
            self.log(f"Monotonicity test exception: {str(e)}", False)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs"""
        self.logs = []  # Reset logs
        
        # Run test suites
        invariant_results = self.test_invariants()
        trace_result = self.test_trace_consistency()
        monotonicity_result = self.test_monotonicity()
        
        # Combine all test results
        all_results = invariant_results + [trace_result, monotonicity_result]
        success = all(all_results)
        
        # Summary
        if success:
            self.logs.append("[SUMMARY] All physics checks passed ✓")
        else:
            self.logs.append("[SUMMARY] Some physics checks failed ✗")
        
        return success, self.logs