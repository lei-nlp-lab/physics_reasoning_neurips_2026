import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # First try to get results attribute
        self.results = getattr(solution_instance, 'results', {})
        
        # If results is empty or doesn't have the required keys,
        # try to extract values directly from solution_instance attributes
        if not self.results or not all(key in self.results for key in ['v_t', 'E_k', 'eU', 'U']):
            # Try to get values directly from solution_instance
            self.v_t = getattr(solution_instance, 'v_t', 0.0)
            self.E_k = getattr(solution_instance, 'E_k', 0.0)
            self.eU = getattr(solution_instance, 'eU', 0.0)
            self.U = getattr(solution_instance, 'U', 0.0)
        else:
            # Extract from results dictionary
            self.v_t = self.results.get('v_t', 0.0)
            self.E_k = self.results.get('E_k', 0.0)
            self.eU = self.results.get('eU', 0.0)
            self.U = self.results.get('U', 0.0)
        
        # Extract input parameters from problem statement
        self.v0 = 4e6  # m/s
        self.e = 1.6e-19  # C
        self.m = 9.1e-31  # kg
        self.theta = math.radians(30)  # radians
        
        # Target answers for verification
        self.target_E_k = 9.7e-18  # J
        self.target_U = 15.2  # V
        
        # Initialize logs
        self.logs = []
        
    def _log(self, msg: str, success: bool) -> None:
        """Helper to log messages with success/failure status."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {msg}")
    
    def test_trigonometric_relation(self) -> bool:
        """Test the velocity relation v_t = v0 / cos(30°)."""
        try:
            v_t_expected = self.v0 / math.cos(self.theta)
            success = math.isclose(self.v_t, v_t_expected, rel_tol=1e-2, abs_tol=1e2)
            
            if success:
                self._log(f"Trigonometric relation: v_t = {self.v_t:.2e} ≈ v0/cos30° = {v_t_expected:.2e}", True)
            else:
                self._log(f"Trigonometric relation failed: {self.v_t:.2e} ≠ {v_t_expected:.2e}", False)
            
            return success
        except Exception as e:
            self._log(f"Trigonometric relation test failed with error: {str(e)}", False)
            return False
    
    def test_kinetic_energy(self) -> bool:
        """Test kinetic energy calculation using E_k = 0.5*m*v_t²."""
        try:
            E_k_expected = 0.5 * self.m * self.v_t**2
            success = math.isclose(self.E_k, E_k_expected, rel_tol=1e-2, abs_tol=1e-20)
            
            if success:
                self._log(f"Kinetic energy: E_k = {self.E_k:.2e} ≈ 0.5*m*v_t² = {E_k_expected:.2e}", True)
            else:
                self._log(f"Kinetic energy failed: {self.E_k:.2e} ≠ {E_k_expected:.2e}", False)
            
            # Check against target value
            if not math.isclose(self.E_k, self.target_E_k, rel_tol=1e-2, abs_tol=1e-20):
                self._log(f"E_k doesn't match target: {self.E_k:.2e} vs {self.target_E_k:.2e}", False)
                success = False
                
            return success
        except Exception as e:
            self._log(f"Kinetic energy test failed with error: {str(e)}", False)
            return False
    
    def test_work_energy_theorem(self) -> bool:
        """Test work-energy theorem: eU = ΔK = 0.5*m*(v_t² - v0²)."""
        try:
            delta_K = 0.5 * self.m * (self.v_t**2 - self.v0**2)
            success = math.isclose(self.eU, delta_K, rel_tol=1e-2, abs_tol=1e-20)
            
            if success:
                self._log(f"Work-energy: eU = {self.eU:.2e} ≈ ΔK = {delta_K:.2e}", True)
            else:
                self._log(f"Work-energy failed: {self.eU:.2e} ≠ {delta_K:.2e}", False)
            
            # Check that kinetic energy increased (electron gained energy from field)
            if not delta_K > 0:
                self._log("Work-energy sign error: ΔK should be positive (electron gains energy)", False)
                success = False
                
            return success
        except Exception as e:
            self._log(f"Work-energy test failed with error: {str(e)}", False)
            return False
    
    def test_potential_difference(self) -> bool:
        """Test U = (ΔK)/e = 0.5*m*(v_t² - v0²)/e."""
        try:
            U_expected = 0.5 * self.m * (self.v_t**2 - self.v0**2) / self.e
            success = math.isclose(self.U, U_expected, rel_tol=1e-2, abs_tol=1e-1)
            
            if success:
                self._log(f"Potential: U = {self.U:.2f} ≈ ΔK/e = {U_expected:.2f}", True)
            else:
                self._log(f"Potential failed: {self.U:.2f} ≠ {U_expected:.2f}", False)
            
            # Check against target value
            if not math.isclose(self.U, self.target_U, rel_tol=1e-2, abs_tol=1e-1):
                self._log(f"U doesn't match target: {self.U:.2f} vs {self.target_U:.2f}", False)
                success = False
                
            return success
        except Exception as e:
            self._log(f"Potential difference test failed with error: {str(e)}", False)
            return False
    
    def test_energy_conservation(self) -> bool:
        """Test overall energy conservation: Initial KE + Work = Final KE."""
        try:
            K_initial = 0.5 * self.m * self.v0**2
            K_final = self.E_k
            work_done = self.eU
            
            # Conservation: K_initial + work_done = K_final
            success = math.isclose(K_initial + work_done, K_final, rel_tol=1e-2, abs_tol=1e-20)
            
            if success:
                self._log(f"Energy conservation: {K_initial:.2e} + {work_done:.2e} ≈ {K_final:.2e}", True)
            else:
                self._log(f"Energy conservation failed: {K_initial:.2e} + {work_done:.2e} ≠ {K_final:.2e}", False)
            
            return success
        except Exception as e:
            self._log(f"Energy conservation test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results for consistency."""
        try:
            # Reconstruct steps from scratch using stored values
            v_t_calc = self.v0 / math.cos(self.theta)
            E_k_calc = 0.5 * self.m * v_t_calc**2
            eU_calc = E_k_calc - 0.5 * self.m * self.v0**2
            U_calc = eU_calc / self.e
            
            checks = [
                ("v_t", self.v_t, v_t_calc),
                ("E_k", self.E_k, E_k_calc),
                ("eU", self.eU, eU_calc),
                ("U", self.U, U_calc)
            ]
            
            all_passed = True
            for name, val, calc in checks:
                if name == "U":
                    # Use looser tolerance for voltage
                    passed = math.isclose(val, calc, rel_tol=1e-2, abs_tol=1e-1)
                else:
                    passed = math.isclose(val, calc, rel_tol=1e-2, abs_tol=1e-20)
                
                if passed:
                    self._log(f"Trace consistency {name}: {val:.2e} ≈ {calc:.2e}", True)
                else:
                    self._log(f"Trace consistency {name} failed: {val:.2e} ≠ {calc:.2e}", False)
                    all_passed = False
            
            return all_passed
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb initial velocity and check trends in results."""
        try:
            # Save original values
            original_v0 = self.v0
            original_v_t = self.v_t
            original_E_k = self.E_k
            original_U = self.U
            
            # Test 1: Increase v0 by 1%
            v0_new = self.v0 * 1.01
            v_t_new = v0_new / math.cos(self.theta)
            E_k_new = 0.5 * self.m * v_t_new**2
            U_new = 0.5 * self.m * (v_t_new**2 - v0_new**2) / self.e
            
            # Check trends
            # Increased v0 should increase v_t and E_k
            trend1_ok = v_t_new > original_v_t and E_k_new > original_E_k
            
            # For U: U = (0.5*m*(v_t² - v0²))/e 
            # Since v_t = v0/cosθ, v_t² - v0² = v0²(1/cos²θ - 1) = v0² * constant
            # So U ∝ v0², increased v0 should increase U
            trend2_ok = U_new > original_U
            
            if trend1_ok and trend2_ok:
                self._log("Monotonicity: All trends correct with increased v0", True)
                return True
            else:
                issues = []
                if not trend1_ok:
                    issues.append("v_t and E_k didn't increase with v0")
                if not trend2_ok:
                    issues.append("U didn't increase with v0")
                self._log(f"Monotonicity failed: {'; '.join(issues)}", False)
                return False
                
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
    
    def test_invariants(self) -> bool:
        """Run all invariant tests."""
        tests = [
            self.test_trigonometric_relation,
            self.test_kinetic_energy,
            self.test_work_energy_theorem,
            self.test_potential_difference,
            self.test_energy_conservation
        ]
        
        all_passed = True
        for test in tests:
            if not test():
                all_passed = False
        
        return all_passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics verification tests."""
        self.logs = []
        
        # Run test suites
        invariant_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        all_ok = invariant_ok and trace_ok and monotonicity_ok
        
        # Summary
        self._log(f"\n=== SUMMARY ===", True)
        self._log(f"Invariant tests: {'PASS' if invariant_ok else 'FAIL'}", invariant_ok)
        self._log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}", trace_ok)
        self._log(f"Monotonicity: {'PASS' if monotonicity_ok else 'FAIL'}", monotonicity_ok)
        self._log(f"OVERALL: {'PASS' if all_ok else 'FAIL'}", all_ok)
        
        return all_ok, self.logs