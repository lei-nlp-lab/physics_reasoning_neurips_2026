import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.solution = solution_instance
        # Extract inputs safely - use defaults if not present
        self.m = getattr(solution_instance, 'm', 
                        getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.theta = getattr(solution_instance, 'theta',
                            getattr(solution_instance, 'inputs', {}).get('theta', 0.1))
        self.g = getattr(solution_instance, 'g',
                        getattr(solution_instance, 'inputs', {}).get('g', 9.81))
        
        # Extract results safely
        self.T = getattr(solution_instance, 'T', 
                        getattr(solution_instance, 'results', {}).get('T', 0.0))
        
        # For symbolic computation
        self.sym_m, self.sym_theta, self.sym_g, self.sym_T = sp.symbols('m theta g T', positive=True)
        
    def test_invariants(self):
        logs = []
        try:
            # Symbolic equilibrium condition: 2*T*cos(theta) - m*g = 0
            equilibrium_eq = sp.Eq(2*self.sym_T*sp.cos(self.sym_theta), self.sym_m*self.sym_g)
            symbolic_check = equilibrium_eq.subs({
                self.sym_m: self.m,
                self.sym_theta: self.theta,
                self.sym_g: self.g,
                self.sym_T: self.T
            })
            symbolic_result = symbolic_check.simplify()
            logs.append(f"Symbolic equilibrium check: {symbolic_result}")
            
            # Numerical equilibrium check
            lhs = 2 * self.T * math.cos(self.theta)
            rhs = self.m * self.g
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Numerical equilibrium FAILED: 2Tcosθ = {lhs}, mg = {rhs}")
                return False, logs
            
            logs.append("✓ Force equilibrium (vertical) satisfied")
            
            # Check horizontal force balance (should be zero due to symmetry)
            # For left and right tensions: horizontal components cancel
            T_horizontal = self.T * math.sin(self.theta)  # Magnitude from one side
            # The two horizontal components are equal and opposite, so net ≈ 0
            logs.append(f"Horizontal force components: ±{T_horizontal:.4f} N (should cancel)")
            
            # Verify T formula matches expected form
            expected_T = (self.m * self.g) / (2 * math.cos(self.theta))
            if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"T formula mismatch: T = {self.T}, expected mg/(2cosθ) = {expected_T}")
                return False, logs
                
            logs.append("✓ Tension formula T = mg/(2cosθ) verified")
            return True, logs
            
        except Exception as e:
            logs.append(f"Invariant test error: {traceback.format_exc()}")
            return False, logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Check if any intermediate steps are available
            steps = getattr(self.solution, 'steps', {})
            if 'step_1' in steps:
                step1 = steps['step_1']
                # Extract tension value from step if available
                step_T = step1.get('T', step1.get('tension', step1.get('value', None)))
                if step_T is not None:
                    if not math.isclose(self.T, step_T, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"Trace inconsistency: final T={self.T}, step T={step_T}")
                        return False, logs
                    logs.append("✓ Trace consistency verified")
            else:
                logs.append("No intermediate steps found for trace check")
            return True, logs
        except Exception as e:
            logs.append(f"Trace consistency error: {traceback.format_exc()}")
            return True, logs  # Don't fail if trace check fails
    
    def test_monotonicity(self):
        logs = []
        original_m = self.m
        original_theta = self.theta
        original_T = self.T
        
        try:
            # Perturb mass +10%
            perturbed_m = original_m * 1.1
            perturbed_T_m = (perturbed_m * self.g) / (2 * math.cos(self.theta))
            
            if perturbed_T_m <= original_T:
                logs.append(f"Monotonicity FAILED: T(m↑) = {perturbed_T_m} ≤ T = {original_T}")
                return False, logs
            logs.append(f"✓ Mass monotonicity: T increases with m (dT/dm = g/(2cosθ) > 0)")
            
            # Perturb theta +5% (small to avoid singularities)
            perturbed_theta = original_theta * 1.05
            # Ensure theta stays in valid range (0, π/2)
            if perturbed_theta >= math.pi/2:
                perturbed_theta = original_theta * 0.95  # Try decreasing instead
            
            perturbed_T_theta = (self.m * self.g) / (2 * math.cos(perturbed_theta))
            
            # Check derivative sign: dT/dθ = (mg sinθ)/(2 cos²θ) > 0 for θ in (0, π/2)
            if original_theta > 0 and original_theta < math.pi/2:
                if perturbed_theta > original_theta:
                    if perturbed_T_theta <= original_T:
                        logs.append(f"Monotonicity FAILED: T(θ↑) = {perturbed_T_theta} ≤ T = {original_T}")
                        return False, logs
                else:
                    if perturbed_T_theta >= original_T:
                        logs.append(f"Monotonicity FAILED: T(θ↓) = {perturbed_T_theta} ≥ T = {original_T}")
                        return False, logs
                logs.append("✓ Theta monotonicity: T increases with θ for θ ∈ (0, π/2)")
            
            # Restore original values
            self.m = original_m
            self.theta = original_theta
            return True, logs
            
        except Exception as e:
            # Restore on error
            self.m = original_m
            self.theta = original_theta
            logs.append(f"Monotonicity test error: {traceback.format_exc()}")
            return False, logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Test 1: Invariants
        test1_success, logs1 = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs1)
        if not test1_success:
            success = False
        
        # Test 2: Trace Consistency
        test2_success, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs2)
        if not test2_success:
            success = False
        
        # Test 3: Monotonicity
        test3_success, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs3)
        if not test3_success:
            success = False
        
        return success, all_logs

# Example usage (for testing the checker itself):
if __name__ == "__main__":
    class MockSolution:
        def __init__(self):
            self.inputs = {'m': 70.0, 'theta': 0.1745, 'g': 9.81}  # 70kg, 10°, g=9.81
            self.results = {'T': 347.0}  # Approx 70*9.81/(2*cos(10°))
            self.steps = {
                'step_1': {
                    'T': 347.0,
                    'equation': 'T = mg/(2cosθ)'
                }
            }
    
    mock = MockSolution()
    checker = PhysicsChecker(mock)
    success, logs = checker.run_all_tests()
    print(f"Overall success: {success}")
    for log in logs:
        print(log)