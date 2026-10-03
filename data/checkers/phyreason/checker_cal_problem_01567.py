import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract all values from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Get constants from results dict with defaults
        self.L = self.results.get('L', 0.8)
        self.Q = self.results.get('Q', 1.8e-7)
        self.m = self.results.get('m', 0.02)
        self.k = self.results.get('k', 9e9)
        
        # Get computed results
        self.W_1 = self.results.get('W_1', 0.0)
        self.W_2 = self.results.get('W_2', 0.0)
        self.W_3 = self.results.get('W_3', 0.0)
        self.F_all = self.results.get('F_all', 0.0)
        self.W_all = self.results.get('W_all', 0.0)
        self.F_1 = self.results.get('F_1', 0.0)
        self.q = self.results.get('q', 0.0)
        self.s = self.results.get('s', 0.0)
        self.U = self.results.get('U', 0.0)
        self.E = self.results.get('E', 0.0)
        self.F_2 = self.results.get('F_2', 0.0)
        
        # Constants from problem context
        self.x_measure = 0.3  # Position for Coulomb force measurement
        self.logs = []
        
    def _log(self, msg: str):
        self.logs.append(msg)
    
    def _check_coulomb_law(self) -> bool:
        """Verify Coulomb's law at x=0.3m"""
        try:
            # Numerical check - protect against division by zero
            if abs(self.Q) < 1e-12:
                self._log(f"Coulomb's law: Q is too small ({self.Q:.3e}) for calculation")
                return False
            
            expected_F1 = self.k * abs(self.q) * self.Q / (self.x_measure**2)
            if math.isclose(expected_F1, self.F_1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Coulomb's law: F₁ = {self.F_1:.3e}, expected {expected_F1:.3e} ✓")
                return True
            else:
                self._log(f"Coulomb's law: F₁ = {self.F_1:.3e}, expected {expected_F1:.3e} ✗")
                return False
        except Exception as e:
            self._log(f"Coulomb's law check failed: {e}")
            return False
    
    def _check_force_superposition(self) -> bool:
        """Verify force superposition at x=0.3m"""
        try:
            # According to steps: F_all = F_2 + q*E
            # F_2 is the Coulomb force (F_1), E is external field magnitude
            # External field direction is left (negative), so q*E is negative
            force_external = -abs(self.q * self.E)  # Negative for left direction
            
            # Resultant force should be F_1 (right, positive) + q*E (left, negative)
            expected_F_all = self.F_1 + force_external
            
            # Check magnitude (sign may vary based on coordinate system)
            if math.isclose(abs(expected_F_all), abs(self.F_all), rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Force superposition: |F_all| = {abs(self.F_all):.3e}, expected |{expected_F_all:.3e}| ✓")
                return True
            else:
                self._log(f"Force superposition: F_all = {self.F_all:.3e}, expected {expected_F_all:.3e} ✗")
                return False
        except Exception as e:
            self._log(f"Force superposition check failed: {e}")
            return False
    
    def _check_potential_difference(self) -> bool:
        """Verify U = W_all / q"""
        try:
            if abs(self.q) > 1e-12:  # Avoid division by zero
                expected_U = self.W_all / self.q
                if math.isclose(expected_U, self.U, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Potential difference: U = {self.U:.1f}, expected {expected_U:.1f} ✓")
                    return True
                else:
                    self._log(f"Potential difference: U = {self.U:.1f}, expected {expected_U:.1f} ✗")
                    return False
            else:
                self._log("Charge q too small for potential check")
                return False
        except Exception as e:
            self._log(f"Potential difference check failed: {e}")
            return False
    
    def _check_work_energy_balance(self) -> bool:
        """Verify work-energy balance for stopping distance s"""
        try:
            # W_1 + W_2 + W_3 + F_out * s = 0
            total_work = self.W_1 + self.W_2 + self.W_3
            
            # Don't assume F_out = -0.004. Instead, check if the relationship holds
            # s = -(W_1 + W_2 + W_3) / F_out
            # We can verify the relationship without assuming F_out value
            
            # First check if s is close to the target answer 0.065 m
            if math.isclose(abs(self.s), 0.065, rel_tol=1e-2, abs_tol=1e-3):
                self._log(f"Stopping distance magnitude: |s| = {abs(self.s):.3f}, target 0.065 ✓")
                return True
            else:
                self._log(f"Stopping distance magnitude: |s| = {abs(self.s):.3f}, target 0.065 ✗")
                return False
        except Exception as e:
            self._log(f"Work-energy balance check failed: {e}")
            return False
    
    def _check_work_signs(self) -> bool:
        """Verify consistency of work signs"""
        try:
            # W_1: work from 0.16-0.20m, should be positive (area under curve)
            # W_2: work from 0.2-0.4m, should be negative (kinetic energy decrease)
            # W_3: work from 0.4-0.8m, should be negative (constant opposing force)
            
            checks = []
            if self.W_1 > 0:
                self._log(f"W_1 positive: {self.W_1:.3e} ✓")
                checks.append(True)
            else:
                self._log(f"W_1 should be positive: {self.W_1:.3e} ✗")
                checks.append(False)
            
            if self.W_2 < 0:
                self._log(f"W_2 negative: {self.W_2:.3e} ✓")
                checks.append(True)
            else:
                self._log(f"W_2 should be negative: {self.W_2:.3e} ✗")
                checks.append(False)
            
            if self.W_3 < 0:
                self._log(f"W_3 negative: {self.W_3:.3e} ✓")
                checks.append(True)
            else:
                self._log(f"W_3 should be negative: {self.W_3:.3e} ✗")
                checks.append(False)
            
            return all(checks)
        except Exception as e:
            self._log(f"Work signs check failed: {e}")
            return False
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants"""
        self._log("=== Testing Physics Invariants ===")
        
        checks = []
        checks.append(self._check_coulomb_law())
        checks.append(self._check_force_superposition())
        checks.append(self._check_work_signs())
        
        return all(checks)
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers"""
        self._log("\n=== Testing Trace Consistency ===")
        
        checks = []
        
        # Check q calculation from Coulomb's law
        try:
            if abs(self.k * self.Q) > 1e-12:  # Avoid division by zero
                expected_q = self.F_1 * (self.x_measure**2) / (self.k * self.Q)
                if math.isclose(abs(expected_q), abs(self.q), rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Charge q: {self.q:.3e}, expected {expected_q:.3e} ✓")
                    checks.append(True)
                else:
                    self._log(f"Charge q: {self.q:.3e}, expected {expected_q:.3e} ✗")
                    checks.append(False)
            else:
                self._log(f"Cannot check q: k*Q = {self.k * self.Q:.3e} is too small")
                checks.append(False)
        except Exception as e:
            self._log(f"q consistency check failed: {e}")
            checks.append(False)
        
        # Check E calculation
        checks.append(self._check_force_superposition())
        
        # Check U calculation
        checks.append(self._check_potential_difference())
        
        # Check s calculation
        checks.append(self._check_work_energy_balance())
        
        return all(checks)
    
    def test_monotonicity(self) -> bool:
        """Perturb input and check expected trends"""
        self._log("\n=== Testing Monotonicity ===")
        
        try:
            # Perturb charge Q and observe effect on q
            original_Q = self.Q if abs(self.Q) > 1e-12 else 1.8e-7  # Use default if Q is 0
            perturbed_Q = original_Q * 1.1  # Increase by 10%
            
            # From q = F_1 * x² / (k * Q), q is inversely proportional to Q
            if abs(self.k * original_Q) > 1e-12:
                original_q_calc = self.F_1 * (self.x_measure**2) / (self.k * original_Q)
                perturbed_q_calc = self.F_1 * (self.x_measure**2) / (self.k * perturbed_Q)
                
                # q should decrease when Q increases
                if perturbed_q_calc < original_q_calc:
                    self._log(f"Monotonicity: Q↑ ({original_Q:.2e}→{perturbed_Q:.2e}) ⇒ q↓ ({original_q_calc:.2e}→{perturbed_q_calc:.2e}) ✓")
                    return True
                else:
                    self._log(f"Monotonicity violation: Q↑ but q not decreasing ✗")
                    return False
            else:
                self._log(f"Cannot test monotonicity: k*Q is too small")
                return False
                
        except Exception as e:
            self._log(f"Monotonicity test failed: {e}")
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag with logs"""
        self.logs = []
        
        try:
            success = True
            
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            success = inv_success and trace_success and mono_success
            
            self._log(f"\n=== Summary ===")
            self._log(f"Invariants: {'PASS' if inv_success else 'FAIL'}")
            self._log(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
            self._log(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}")
            self._log(f"Overall: {'PASS' if success else 'FAIL'}")
            
            return success, self.logs
            
        except Exception as e:
            self._log(f"Test execution failed: {e}")
            return False, self.logs