import math
import sympy as sp
from typing import Tuple, List, Optional, Any
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize checker with safe attribute access.
        solution_instance is the student's solution object.
        """
        self.solution = solution_instance
        
        # Extract inputs with fallbacks
        self.theta = getattr(solution_instance, 'theta', 
                            getattr(solution_instance, 'inputs', {}).get('theta', 60.0))
        self.S = getattr(solution_instance, 'S', 
                        getattr(solution_instance, 'inputs', {}).get('S', 0.4))
        self.B = getattr(solution_instance, 'B', 
                        getattr(solution_instance, 'inputs', {}).get('B', 0.6))
        
        # Extract results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.phi = self.results.get('Φ', getattr(solution_instance, 'phi', 0.0))
        
        # Convert to float if needed (handles sympy expressions)
        try:
            if hasattr(self.theta, 'evalf'):
                self.theta = float(self.theta.evalf())
            if hasattr(self.S, 'evalf'):
                self.S = float(self.S.evalf())
            if hasattr(self.B, 'evalf'):
                self.B = float(self.B.evalf())
            if hasattr(self.phi, 'evalf'):
                self.phi = float(self.phi.evalf())
        except:
            pass
            
        # Store original values for perturbation tests
        self.original_theta = float(self.theta)
        self.original_S = float(self.S)
        self.original_B = float(self.B)
        self.original_phi = float(self.phi)
        
        self.logs = []
    
    def _log(self, msg: str, success: bool = True) -> None:
        """Helper for logging test results."""
        self.logs.append((success, msg))
    
    def test_invariants(self) -> bool:
        """
        Verify magnetic flux formula: Φ = B * S * cos(θ)
        Uses both symbolic and numeric checks.
        """
        try:
            # Convert to radians for calculation
            theta_rad = math.radians(self.original_theta)
            
            # Calculate expected flux
            expected_phi = self.original_B * self.original_S * math.cos(theta_rad)
            
            # Fuzzy comparison with 2% tolerance
            if math.isclose(self.original_phi, expected_phi, rel_tol=0.02, abs_tol=0.02):
                self._log(f"✓ Flux invariant: Φ = B * S * cos(θ) holds. "
                         f"Got {self.original_phi:.3f}, expected {expected_phi:.3f}")
                return True
            else:
                self._log(f"✗ Flux invariant violated. "
                         f"Got {self.original_phi:.3f}, expected {expected_phi:.3f} "
                         f"with B={self.original_B}, S={self.original_S}, θ={self.original_theta}°", 
                         False)
                return False
                
        except Exception as e:
            self._log(f"✗ Flux invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self) -> bool:
        """
        Check if intermediate calculations (if any) are consistent with final answer.
        For this problem, we only have one step.
        """
        try:
            # If solution has steps attribute, check consistency
            steps = getattr(self.solution, 'steps', {})
            if steps:
                step1_result = steps.get('step_1', {}).get('result_quantity', [{}])[0].get('value', None)
                if step1_result:
                    try:
                        step_val = float(step1_result)
                        if math.isclose(step_val, self.original_phi, rel_tol=0.02, abs_tol=0.02):
                            self._log(f"✓ Trace consistency: Step value {step_val} matches final Φ {self.original_phi}")
                            return True
                        else:
                            self._log(f"✗ Trace inconsistency: Step value {step_val} ≠ final Φ {self.original_phi}", False)
                            return False
                    except:
                        pass
            
            # No steps to check - consider as passed
            self._log("✓ No intermediate steps to check for consistency")
            return True
            
        except Exception as e:
            self._log(f"✗ Trace consistency test failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and verify expected trends in flux.
        Tests: ∂Φ/∂B > 0, ∂Φ/∂S > 0, ∂Φ/∂θ < 0 (for θ in [0,90]°)
        """
        success = True
        
        try:
            # Test 1: Increase B by 10%, flux should increase
            B_perturbed = self.original_B * 1.1
            theta_rad = math.radians(self.original_theta)
            phi_perturbed_B = B_perturbed * self.original_S * math.cos(theta_rad)
            
            if phi_perturbed_B > self.original_phi:
                self._log(f"✓ Monotonicity: Increasing B increases flux "
                         f"({self.original_phi:.3f} → {phi_perturbed_B:.3f})")
            else:
                self._log(f"✗ Monotonicity violation: Increasing B should increase flux", False)
                success = False
            
            # Test 2: Increase S by 10%, flux should increase
            S_perturbed = self.original_S * 1.1
            phi_perturbed_S = self.original_B * S_perturbed * math.cos(theta_rad)
            
            if phi_perturbed_S > self.original_phi:
                self._log(f"✓ Monotonicity: Increasing S increases flux "
                         f"({self.original_phi:.3f} → {phi_perturbed_S:.3f})")
            else:
                self._log(f"✗ Monotonicity violation: Increasing S should increase flux", False)
                success = False
            
            # Test 3: Increase θ by 10%, flux should decrease (for θ < 90°)
            if self.original_theta < 90:
                theta_perturbed = min(self.original_theta * 1.1, 89.9)  # Keep < 90°
                theta_perturbed_rad = math.radians(theta_perturbed)
                phi_perturbed_theta = self.original_B * self.original_S * math.cos(theta_perturbed_rad)
                
                if phi_perturbed_theta < self.original_phi:
                    self._log(f"✓ Monotonicity: Increasing θ decreases flux "
                             f"({self.original_phi:.3f} → {phi_perturbed_theta:.3f})")
                else:
                    self._log(f"✗ Monotonicity violation: Increasing θ should decrease flux "
                             f"(for θ < 90°)", False)
                    success = False
            
            return success
            
        except Exception as e:
            self._log(f"✗ Monotonicity test failed: {str(e)}", False)
            return False
    
    def test_physical_bounds(self) -> bool:
        """
        Check that results are within physical bounds.
        Flux magnitude cannot exceed B*S (when cosθ=1).
        """
        try:
            max_flux = self.original_B * self.original_S
            abs_phi = abs(self.original_phi)
            
            if abs_phi <= max_flux * 1.01:  # Allow 1% tolerance for rounding
                self._log(f"✓ Physical bounds: |Φ| = {abs_phi:.3f} ≤ B*S = {max_flux:.3f}")
                return True
            else:
                self._log(f"✗ Physical bounds violated: |Φ| = {abs_phi:.3f} > B*S = {max_flux:.3f}", False)
                return False
                
        except Exception as e:
            self._log(f"✗ Physical bounds test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics consistency checks.
        Returns (success_bool, list_of_messages)
        """
        self.logs = []
        
        tests = [
            ("Physical Bounds Check", self.test_physical_bounds),
            ("Flux Invariant", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        all_success = True
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    all_success = False
            except Exception as e:
                self._log(f"✗ {test_name} crashed: {str(e)}", False)
                all_success = False
        
        # Compile messages
        messages = []
        for success, msg in self.logs:
            prefix = "PASS" if success else "FAIL"
            messages.append(f"{prefix}: {msg}")
        
        return all_success, messages