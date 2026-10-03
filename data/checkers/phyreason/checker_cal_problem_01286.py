import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the PhysicsChecker with a solution instance.
        Safely extracts inputs and results using getattr to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.mA = getattr(solution_instance, 'mA', getattr(solution_instance, 'inputs', {}).get('mA', 4.0))
        self.mB = getattr(solution_instance, 'mB', getattr(solution_instance, 'inputs', {}).get('mB', 5.0))
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'inputs', {}).get('F', 17.0))
        self.F0 = getattr(solution_instance, 'F0', getattr(solution_instance, 'inputs', {}).get('F0', 8.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        
        # Extract results with safe access pattern
        results = getattr(solution_instance, 'results', {})
        self.f = getattr(solution_instance, 'f', results.get('f', None))
        self.mu = getattr(solution_instance, 'mu', results.get('mu', None))
        self.mu_prime = getattr(solution_instance, 'mu_prime', results.get('mu\'', None))
        
        # Target values from problem statement
        self.target_f = 8.0
        self.target_mu = 0.2
        self.target_mu_prime = 0.1
        
        # Setup symbolic variables for invariant checking
        self.setup_symbolic()
    
    def setup_symbolic(self) -> None:
        """Initialize symbolic variables for physics laws."""
        # Define symbols
        mA, mB, F, F0, g, f, mu, mu_prime = sp.symbols('mA mB F F0 g f mu mu_prime', real=True, positive=True)
        
        # Define equations from physics laws
        self.eq1 = sp.Eq(f, F0)  # Spring scale reading equals friction on A
        self.eq2 = sp.Eq(f, mu * mA * g)  # Kinetic friction between A and B
        self.eq3 = sp.Eq(F, f + mu_prime * (mA + mB) * g)  # Force equilibrium for B
        
        # Store symbols
        self.symbols = {
            'mA': mA, 'mB': mB, 'F': F, 'F0': F0, 'g': g,
            'f': f, 'mu': mu, 'mu_prime': mu_prime
        }
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental physics invariants:
        1. Force equilibrium on block A (f = F0)
        2. Friction law between A and B (f = μ·mA·g)
        3. Force equilibrium on block B (F = f + μ'·(mA+mB)·g)
        """
        logs = []
        success = True
        
        # 1. Check f = F0 (spring force equals friction on A)
        if self.f is not None:
            if not math.isclose(self.f, self.F0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f = {self.f} ≠ F0 = {self.F0}")
                success = False
            else:
                logs.append(f"PASS: f = {self.f} ≈ F0 = {self.F0}")
        else:
            logs.append("SKIP: f not available for invariant check")
            success = False
        
        # 2. Check friction law: f = μ·mA·g
        if self.mu is not None:
            expected_f = self.mu * self.mA * self.g
            if self.f is not None:
                if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: f = {self.f} ≠ μ·mA·g = {expected_f}")
                    success = False
                else:
                    logs.append(f"PASS: f = {self.f} ≈ μ·mA·g = {expected_f}")
        
        # 3. Check force equilibrium on B: F = f + μ'·(mA+mB)·g
        if all(v is not None for v in [self.mu_prime, self.f]):
            lhs = self.F
            rhs = self.f + self.mu_prime * (self.mA + self.mB) * self.g
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F = {lhs} ≠ f + μ'·(mA+mB)·g = {rhs}")
                success = False
            else:
                logs.append(f"PASS: F = {lhs} ≈ f + μ'·(mA+mB)·g = {rhs}")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answers.
        Compares computed values with target values from steps analysis.
        """
        logs = []
        success = True
        
        # Check f against target
        if self.f is not None:
            if not math.isclose(self.f, self.target_f, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f = {self.f} ≠ target {self.target_f}")
                success = False
            else:
                logs.append(f"PASS: f = {self.f} ≈ target {self.target_f}")
        
        # Check μ against target
        if self.mu is not None:
            if not math.isclose(self.mu, self.target_mu, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: μ = {self.mu} ≠ target {self.target_mu}")
                success = False
            else:
                logs.append(f"PASS: μ = {self.mu} ≈ target {self.target_mu}")
        
        # Check μ' against target
        if self.mu_prime is not None:
            if not math.isclose(self.mu_prime, self.target_mu_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: μ' = {self.mu_prime} ≠ target {self.target_mu_prime}")
                success = False
            else:
                logs.append(f"PASS: μ' = {self.mu_prime} ≈ target {self.target_mu_prime}")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb input mass mA and check if coefficients change as expected.
        Theory: Increasing mA should increase normal force, thus:
        - μ = f/(mA·g) should DECREASE (since f is fixed by F0)
        - μ' = (F-f)/((mA+mB)·g) should DECREASE (denominator increases)
        """
        logs = []
        success = True
        
        # Store original values
        original_mA = self.mA
        original_mu = self.mu
        original_mu_prime = self.mu_prime
        
        # Skip if missing data
        if None in [original_mu, original_mu_prime]:
            logs.append("SKIP: Missing μ or μ' for monotonicity test")
            return False, logs
        
        # Apply small perturbation (+10% increase)
        perturbed_mA = original_mA * 1.1
        
        # Calculate expected perturbed values using physics formulas
        # Note: f remains fixed at F0 due to equilibrium condition
        perturbed_mu = self.F0 / (perturbed_mA * self.g)
        perturbed_mu_prime = (self.F - self.F0) / ((perturbed_mA + self.mB) * self.g)
        
        # Check expected trends
        trend_check1 = perturbed_mu < original_mu  # μ should decrease
        trend_check2 = perturbed_mu_prime < original_mu_prime  # μ' should decrease
        
        if not trend_check1:
            logs.append(f"FAIL: μ did not decrease when mA increased (from {original_mu} to {perturbed_mu})")
            success = False
        else:
            logs.append(f"PASS: μ decreased correctly when mA increased (from {original_mu} to {perturbed_mu})")
        
        if not trend_check2:
            logs.append(f"FAIL: μ' did not decrease when mA increased (from {original_mu_prime} to {perturbed_mu_prime})")
            success = False
        else:
            logs.append(f"PASS: μ' decreased correctly when mA increased (from {original_mu_prime} to {perturbed_mu_prime})")
        
        # Restore original value (though we only used local copy)
        self.mA = original_mA
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics audit tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs

# Example usage (commented out for the sandbox environment):
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.mA = 4.0
#             self.mB = 5.0
#             self.F = 17.0
#             self.F0 = 8.0
#             self.g = 10.0
#             self.results = {'f': 8.0, 'mu': 0.2, 'mu\'': 0.1}
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)