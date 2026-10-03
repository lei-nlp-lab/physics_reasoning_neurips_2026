import math
import sympy as sp
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        Extracts all necessary parameters from solution instance.
        """
        self.logs = []
        
        # Safe extraction of inputs
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', None))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', None))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', None))
        self.eps0 = getattr(solution_instance, 'eps0', 8.854187817e-12)  # Default SI value
        
        # Safe extraction of results
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', None))
        self.q0 = getattr(solution_instance, 'q_0', solution_instance.results.get('q_0', None))
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', None))
        self.Vtheta = getattr(solution_instance, 'Vtheta', solution_instance.results.get('Vtheta', None))
        self.L_theta = getattr(solution_instance, 'L_theta', solution_instance.results.get('L_theta', None))
        self.x_theta = getattr(solution_instance, 'x_theta', solution_instance.results.get('x_theta', None))
        self.fracqq_0 = getattr(solution_instance, 'fracqq_0', solution_instance.results.get('fracqq_0', None))
        
        # Define k = 1/(4πϵ₀)
        self.k = 1.0 / (4 * math.pi * self.eps0)
        
        # For symbolic calculations
        self.R_sym, self.L_sym, self.q_sym, self.x_sym, self.q0_sym = sp.symbols('R L q x q0', positive=True)
        self.theta = sp.symbols('theta', real=True)
        
    def log(self, message: str, test_name: str):
        """Helper to log messages with test context."""
        self.logs.append(f"[{test_name}] {message}")
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants:
        1. Boundary condition: V=0 on sphere surface
        2. Force direction: Attractive for opposite charges
        3. Charge ratio consistency
        """
        success = True
        test_logs = []
        
        try:
            # Test 1: Boundary condition (sphere potential = 0)
            if all(v is not None for v in [self.R, self.L, self.q, self.x, self.q0]):
                # Compute potential at two points: θ=0 and θ=π
                # At θ=0: cosθ=1
                x_theta_0 = math.sqrt(self.x**2 + self.R**2 - 2*self.x*self.R)
                L_theta_0 = math.sqrt(self.L**2 + self.R**2 - 2*self.L*self.R)
                V0 = self.k * (self.q0/x_theta_0 + self.q/L_theta_0)
                
                # At θ=π: cosθ=-1
                x_theta_pi = math.sqrt(self.x**2 + self.R**2 + 2*self.x*self.R)
                L_theta_pi = math.sqrt(self.L**2 + self.R**2 + 2*self.L*self.R)
                Vpi = self.k * (self.q0/x_theta_pi + self.q/L_theta_pi)
                
                if not (math.isclose(V0, 0.0, rel_tol=1e-2, abs_tol=1e-2) and 
                       math.isclose(Vpi, 0.0, rel_tol=1e-2, abs_tol=1e-2)):
                    success = False
                    msg = f"Boundary condition failed: V(0)={V0:.2e}, V(π)={Vpi:.2e}"
                    self.log(msg, "test_invariants")
                    test_logs.append(msg)
                else:
                    self.log("Boundary condition satisfied: V(sphere)=0", "test_invariants")
            
            # Test 2: Force direction (should be attractive)
            if self.F is not None:
                # Force should be negative (attractive) since charges are opposite
                if self.F > 0:
                    success = False
                    msg = f"Force should be attractive (negative) but got F={self.F}"
                    self.log(msg, "test_invariants")
                    test_logs.append(msg)
                else:
                    self.log("Force direction correct: attractive", "test_invariants")
            
            # Test 3: Charge ratio consistency
            if all(v is not None for v in [self.q, self.q0, self.R, self.L]):
                expected_q0 = -self.q * self.R / self.L
                if not math.isclose(self.q0, expected_q0, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    msg = f"Charge ratio mismatch: q0={self.q0}, expected={expected_q0}"
                    self.log(msg, "test_invariants")
                    test_logs.append(msg)
                else:
                    self.log(f"Charge ratio correct: q0 = -qR/L = {self.q0}", "test_invariants")
            
            # Test 4: Image position consistency
            if all(v is not None for v in [self.x, self.R, self.L]):
                expected_x = self.R**2 / self.L
                if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    msg = f"Image position mismatch: x={self.x}, expected={expected_x}"
                    self.log(msg, "test_invariants")
                    test_logs.append(msg)
                else:
                    self.log(f"Image position correct: x=R²/L = {self.x}", "test_invariants")
                    
        except Exception as e:
            success = False
            msg = f"Error in test_invariants: {str(e)}"
            self.log(msg, "test_invariants")
            test_logs.append(msg)
        
        return success, test_logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Check consistency between intermediate steps and final results.
        """
        success = True
        test_logs = []
        
        try:
            # Test 1: Check the condition for theta-independence
            if all(v is not None for v in [self.R, self.L, self.x]):
                lhs = (self.L**2 + self.R**2) / self.L
                rhs = (self.x**2 + self.R**2) / self.x
                
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    msg = f"Theta-independence condition failed: LHS={lhs}, RHS={rhs}"
                    self.log(msg, "test_trace_consistency")
                    test_logs.append(msg)
                else:
                    self.log("Theta-independence condition satisfied", "test_trace_consistency")
            
            # Test 2: Check force calculation from Coulomb's law
            if all(v is not None for v in [self.F, self.q, self.q0, self.L, self.x]):
                # Distance between charges
                d = abs(self.L - self.x)
                # Expected force from Coulomb's law
                F_expected = self.k * self.q * self.q0 / (d**2)
                
                if not math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    msg = f"Force mismatch: F={self.F}, Coulomb's law gives {F_expected}"
                    self.log(msg, "test_trace_consistency")
                    test_logs.append(msg)
                else:
                    self.log(f"Force consistent with Coulomb's law: F={self.F}", "test_trace_consistency")
            
            # Test 3: Check simplified charge ratio
            if all(v is not None for v in [self.q, self.q0, self.L, self.x]):
                expected_ratio = -math.sqrt(self.L / self.x)
                actual_ratio = self.q / self.q0  # Should be negative
                
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    msg = f"Charge ratio mismatch: q/q0={actual_ratio}, expected {expected_ratio}"
                    self.log(msg, "test_trace_consistency")
                    test_logs.append(msg)
                else:
                    self.log(f"Charge ratio sqrt(L/x) correct: {actual_ratio}", "test_trace_consistency")
                    
        except Exception as e:
            success = False
            msg = f"Error in test_trace_consistency: {str(e)}"
            self.log(msg, "test_trace_consistency")
            test_logs.append(msg)
        
        return success, test_logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Test monotonic relationships:
        1. Increasing |q| should increase |F|
        2. Increasing L should decrease |F|
        """
        success = True
        test_logs = []
        
        try:
            if all(v is not None for v in [self.R, self.L, self.q, self.F]):
                # Perturbation factor
                perturb = 0.01  # 1% change
                
                # Test 1: Increase |q| by 1%
                q_perturbed = self.q * (1 + perturb)
                F_perturbed = self.k * (q_perturbed**2 * self.R / self.L) / (self.L - self.R**2/self.L)**2
                
                # Check if |F| increases with |q|
                if abs(F_perturbed) <= abs(self.F):
                    success = False
                    msg = f"Monotonicity violated: |F| should increase with |q|"
                    self.log(msg, "test_monotonicity")
                    test_logs.append(msg)
                else:
                    self.log("Monotonicity: |F| increases with |q| ✓", "test_monotonicity")
                
                # Test 2: Increase L by 1%
                L_perturbed = self.L * (1 + perturb)
                F_L_perturbed = self.k * (self.q**2 * self.R / L_perturbed) / (L_perturbed - self.R**2/L_perturbed)**2
                
                # Check if |F| decreases with increasing L
                if abs(F_L_perturbed) >= abs(self.F):
                    success = False
                    msg = f"Monotonicity violated: |F| should decrease with increasing L"
                    self.log(msg, "test_monotonicity")
                    test_logs.append(msg)
                else:
                    self.log("Monotonicity: |F| decreases with increasing L ✓", "test_monotonicity")
            
            else:
                self.log("Skipping monotonicity tests: missing parameters", "test_monotonicity")
                
        except Exception as e:
            success = False
            msg = f"Error in test_monotonicity: {str(e)}"
            self.log(msg, "test_monotonicity")
            test_logs.append(msg)
        
        return success, test_logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and consolidated logs.
        """
        overall_success = True
        all_logs = []
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                overall_success = overall_success and test_success
                all_logs.extend(test_logs)
                
                status = "PASSED" if test_success else "FAILED"
                self.log(f"{test_name}: {status}", "run_all_tests")
                
            except Exception as e:
                overall_success = False
                error_msg = f"{test_name} crashed: {str(e)}\n{traceback.format_exc()}"
                self.log(error_msg, "run_all_tests")
                all_logs.append(error_msg)
        
        # Add summary
        summary = f"Overall: {'PASS' if overall_success else 'FAIL'}"
        self.log(summary, "run_all_tests")
        all_logs.append(summary)
        
        return overall_success, all_logs

# Example usage and test (commented out for production)
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.R = 0.1  # meters
            self.L = 0.5  # meters
            self.q = 1e-9  # Coulombs
            self.eps0 = 8.854187817e-12
            
            # Results
            self.results = {
                'x': 0.02,  # R²/L = 0.01/0.5 = 0.02
                'q_0': -2e-10,  # -qR/L = -1e-9 * 0.1/0.5 = -2e-10
                'F': -8.987551787e-9 * (1e-9 * -2e-10) / (0.48**2)  # k*q*q0/(L-x)²
            }
    
    # Test the checker
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    print(f"Success: {success}")
    for log in logs:
        print(log)