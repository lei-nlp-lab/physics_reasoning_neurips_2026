import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    """
    Physics auditor for hydrogen balloon wind force problem.
    Verifies equilibrium conditions, trace consistency, and monotonicity.
    """
    
    def __init__(self, solution_instance):
        """
        Safely extract all required data using getattr to avoid AttributeError.
        
        Expected keys in solution_instance.results:
            - 'F_wind': Horizontal wind force (N)
            - 'T_rope': Tension in rope (N)
            - 'T_rope*cos60°': Horizontal tension component (N)
            - 'T_rope*sin60°': Vertical tension component (N)
        """
        # Safe extraction from solution instance
        results = getattr(solution_instance, 'results', {})
        
        # Final results
        self.F_wind = getattr(solution_instance, 'F_wind', results.get('F_wind', 0.0))
        self.T_rope = getattr(solution_instance, 'T_rope', results.get('T_rope', 0.0))
        
        # Intermediate components
        self.T_horiz = getattr(solution_instance, 'T_rope*cos60°', 
                              results.get('T_rope*cos60°', 0.0))
        self.T_vert = getattr(solution_instance, 'T_rope*sin60°', 
                             results.get('T_rope*sin60°', 0.0))
        
        # Problem constants (from problem statement)
        self.F_buoyancy = 20 * math.sqrt(3)  # 20√3 N
        self.theta = math.radians(60)  # 60° in radians
        
        # Trigonometric values
        self.cos60 = math.cos(self.theta)
        self.sin60 = math.sin(self.theta)
        
        # Store logs for reporting
        self.logs = []
        
    def _log(self, msg: str, success: bool) -> None:
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {msg}")
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants:
        1. Vertical equilibrium: T_vert = F_buoyancy
        2. Horizontal equilibrium: F_wind = T_horiz
        3. Component consistency: T_vert = T_rope * sin(60°), T_horiz = T_rope * cos(60°)
        """
        success = True
        
        # 1. Vertical equilibrium (neglecting gravity)
        expected_T_vert = self.F_buoyancy
        if math.isclose(self.T_vert, expected_T_vert, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Vertical equilibrium: T_vert = {self.T_vert:.2f} N ≈ F_buoyancy = {expected_T_vert:.2f} N", True)
        else:
            self._log(f"Vertical equilibrium violated: T_vert = {self.T_vert:.2f} N ≠ F_buoyancy = {expected_T_vert:.2f} N", False)
            success = False
        
        # 2. Horizontal equilibrium
        if math.isclose(self.F_wind, self.T_horiz, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Horizontal equilibrium: F_wind = {self.F_wind:.2f} N ≈ T_horiz = {self.T_horiz:.2f} N", True)
        else:
            self._log(f"Horizontal equilibrium violated: F_wind = {self.F_wind:.2f} N ≠ T_horiz = {self.T_horiz:.2f} N", False)
            success = False
        
        # 3. Component consistency with T_rope
        expected_T_horiz = self.T_rope * self.cos60
        expected_T_vert = self.T_rope * self.sin60
        
        if math.isclose(self.T_horiz, expected_T_horiz, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"T_horiz consistency: {self.T_horiz:.2f} N ≈ T_rope*cos60° = {expected_T_horiz:.2f} N", True)
        else:
            self._log(f"T_horiz inconsistent: {self.T_horiz:.2f} N ≠ T_rope*cos60° = {expected_T_horiz:.2f} N", False)
            success = False
        
        if math.isclose(self.T_vert, expected_T_vert, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"T_vert consistency: {self.T_vert:.2f} N ≈ T_rope*sin60° = {expected_T_vert:.2f} N", True)
        else:
            self._log(f"T_vert inconsistent: {self.T_vert:.2f} N ≠ T_rope*sin60° = {expected_T_vert:.2f} N", False)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Verify that intermediate calculations are consistent with final results.
        From steps analysis:
            - T_rope should satisfy: T_rope = F_buoyancy / sin(60°)
            - F_wind should satisfy: F_wind = T_rope * cos(60°)
        """
        success = True
        
        # Calculate expected values from formulas
        expected_T_rope = self.F_buoyancy / self.sin60
        expected_F_wind = expected_T_rope * self.cos60
        
        # Check T_rope
        if math.isclose(self.T_rope, expected_T_rope, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"T_rope trace: {self.T_rope:.2f} N ≈ F_buoyancy/sin60° = {expected_T_rope:.2f} N", True)
        else:
            self._log(f"T_rope trace mismatch: {self.T_rope:.2f} N ≠ F_buoyancy/sin60° = {expected_T_rope:.2f} N", False)
            success = False
        
        # Check F_wind
        if math.isclose(self.F_wind, expected_F_wind, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"F_wind trace: {self.F_wind:.2f} N ≈ T_rope*cos60° = {expected_F_wind:.2f} N", True)
        else:
            self._log(f"F_wind trace mismatch: {self.F_wind:.2f} N ≠ T_rope*cos60° = {expected_F_wind:.2f} N", False)
            success = False
        
        # Also verify the reported intermediate components match recalculated ones
        recalc_T_horiz = self.T_rope * self.cos60
        recalc_T_vert = self.T_rope * self.sin60
        
        if math.isclose(self.T_horiz, recalc_T_horiz, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Intermediate T_horiz consistent: {self.T_horiz:.2f} N ≈ {recalc_T_horiz:.2f} N", True)
        else:
            self._log(f"Intermediate T_horiz inconsistent: {self.T_horiz:.2f} N ≠ {recalc_T_horiz:.2f} N", False)
            success = False
        
        if math.isclose(self.T_vert, recalc_T_vert, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Intermediate T_vert consistent: {self.T_vert:.2f} N ≈ {recalc_T_vert:.2f} N", True)
        else:
            self._log(f"Intermediate T_vert inconsistent: {self.T_vert:.2f} N ≠ {recalc_T_vert:.2f} N", False)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Test how outputs change when buoyant force is perturbed.
        Perturb F_buoyancy by +10%, recalculate T_rope and F_wind using formulas,
        verify they change in the expected direction, then restore.
        """
        success = True
        
        # Original values
        F_b_original = self.F_buoyancy
        T_original = self.T_rope
        F_w_original = self.F_wind
        
        # Perturb buoyant force by +10%
        perturbation = 0.10
        F_b_perturbed = F_b_original * (1 + perturbation)
        
        # Recalculate using the physics formulas
        # From steps analysis: T_rope = F_buoyancy / sin(60°), F_wind = T_rope * cos(60°)
        T_perturbed = F_b_perturbed / self.sin60
        F_w_perturbed = T_perturbed * self.cos60
        
        # Check monotonicity: if F_buoyancy increases, T_rope should increase
        if T_perturbed > T_original:
            self._log(f"Monotonicity (T_rope): F_buoyancy ↑ {perturbation*100:.0f}% → T_rope ↑ ({T_original:.2f} → {T_perturbed:.2f} N)", True)
        else:
            self._log(f"Monotonicity violation (T_rope): F_buoyancy ↑ but T_rope ↓ ({T_original:.2f} → {T_perturbed:.2f} N)", False)
            success = False
        
        # Check monotonicity for F_wind (should also increase since it's proportional to T_rope)
        if F_w_perturbed > F_w_original:
            self._log(f"Monotonicity (F_wind): F_buoyancy ↑ {perturbation*100:.0f}% → F_wind ↑ ({F_w_original:.2f} → {F_w_perturbed:.2f} N)", True)
        else:
            self._log(f"Monotonicity violation (F_wind): F_buoyancy ↑ but F_wind ↓ ({F_w_original:.2f} → {F_w_perturbed:.2f} N)", False)
            success = False
        
        # Restore original values (though they weren't modified, just for clarity)
        self.F_buoyancy = F_b_original
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all audit tests and return overall success status with logs.
        """
        self.logs = []
        self.logs.append("Starting Physics Audit for Hydrogen Balloon Problem")
        
        # Run individual tests
        test_results = []
        
        test_results.append(("Invariants", self.test_invariants()))
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        test_results.append(("Monotonicity", self.test_monotonicity()))
        
        # Determine overall success
        all_pass = all(success for _, success in test_results)
        
        # Summary
        self.logs.append("\n" + "="*60)
        self.logs.append("TEST SUMMARY:")
        for name, passed in test_results:
            status = "PASS" if passed else "FAIL"
            self.logs.append(f"  {name}: {status}")
        
        if all_pass:
            self.logs.append("OVERALL: All physics checks passed ✓")
        else:
            self.logs.append("OVERALL: Some physics checks failed ✗")
        
        return all_pass, self.logs