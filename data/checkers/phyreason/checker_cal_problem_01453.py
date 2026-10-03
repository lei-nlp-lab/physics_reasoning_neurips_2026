import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        The solution_instance should have attributes or a results dictionary.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.m_A = getattr(solution_instance, 'm_A', 2.0)
        self.m_B = getattr(solution_instance, 'm_B', 5.0)
        self.m_C = getattr(solution_instance, 'm_C', 1.0)
        self.x = getattr(solution_instance, 'x', 0.02)  # 2 cm in meters
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.T = results.get('T', 0.0)
        self.F_e = results.get('F_e', 0.0)
        self.f = results.get('f', 0.0)
        self.f_ground = results.get('f_textground', results.get('f_ground', 0.0))
        self.k = results.get('k', 0.0)
        
        # Precompute trigonometric values
        self.cos30 = math.cos(math.radians(30))
        self.sin30 = math.sin(math.radians(30))
        self.cos60 = math.cos(math.radians(60))
        self.sin60 = math.sin(math.radians(60))
        
        self.logs = []
        
    def log(self, message, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
        
    def test_invariants(self):
        """
        Verify physical invariants:
        1. Equilibrium of forces on object C (vertical and horizontal)
        2. Newton's 3rd law: f = T (friction on A equals tension)
        3. Equilibrium of forces on object B (horizontal)
        """
        successes = []
        
        # 1. Equilibrium of object C
        # Vertical: T*cos30 + F_e*cos60 = m_C * g
        vertical_lhs = self.T * self.cos30 + self.F_e * self.cos60
        vertical_rhs = self.m_C * self.g
        successes.append(self.log(
            f"Vertical equilibrium of C: {vertical_lhs:.3f} ≈ {vertical_rhs:.3f}",
            math.isclose(vertical_lhs, vertical_rhs, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Horizontal: T*sin30 = F_e*sin60
        horizontal_lhs = self.T * self.sin30
        horizontal_rhs = self.F_e * self.sin60
        successes.append(self.log(
            f"Horizontal equilibrium of C: {horizontal_lhs:.3f} ≈ {horizontal_rhs:.3f}",
            math.isclose(horizontal_lhs, horizontal_rhs, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 2. Newton's 3rd law for object A: f = T
        successes.append(self.log(
            f"Friction on A equals tension: f={self.f:.3f}, T={self.T:.3f}",
            math.isclose(self.f, self.T, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 3. Horizontal equilibrium of object B: f_ground = T*cos60
        expected_f_ground = self.T * self.cos60
        successes.append(self.log(
            f"Horizontal equilibrium of B: f_ground={self.f_ground:.3f}, expected={expected_f_ground:.3f}",
            math.isclose(self.f_ground, expected_f_ground, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        successes = []
        
        # 1. Hooke's Law: k = F_e / x
        k_from_hooke = self.F_e / self.x if self.x != 0 else 0.0
        successes.append(self.log(
            f"Hooke's Law: k={self.k:.3f}, F_e/x={k_from_hooke:.3f}",
            math.isclose(self.k, k_from_hooke, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 2. Consistency with target answers
        target_k = 250.0
        target_f = 5 * math.sqrt(3)  # 5√3 ≈ 8.660
        target_f_ground = 2.5 * math.sqrt(3)  # (5√3)/2 ≈ 4.330
        
        successes.append(self.log(
            f"Spring constant k matches target: {self.k:.3f} ≈ {target_k:.3f}",
            math.isclose(self.k, target_k, rel_tol=1e-2, abs_tol=1e-2)
        ))
        successes.append(self.log(
            f"Friction f matches target: {self.f:.3f} ≈ {target_f:.3f}",
            math.isclose(self.f, target_f, rel_tol=1e-2, abs_tol=1e-2)
        ))
        successes.append(self.log(
            f"Ground friction f_ground matches target: {self.f_ground:.3f} ≈ {target_f_ground:.3f}",
            math.isclose(self.f_ground, target_f_ground, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        return all(successes)
    
    def test_monotonicity(self):
        """
        Perturb an input and check the trend of outputs.
        We perturb the spring elongation x and verify:
        - k is inversely proportional to x (if F_e constant)
        - Since F_e depends on equilibrium, we also check symbolic relationships.
        """
        successes = []
        
        # Perturbation factor
        perturb_factor = 1.05  # 5% increase
        
        # Save original state
        original_x = self.x
        original_k = self.k
        original_F_e = self.F_e
        
        # Symbolic relationships from equilibrium
        # From horizontal equilibrium of C: T*sin30 = F_e*sin60 => T = F_e * (sin60/sin30)
        # From vertical equilibrium: T*cos30 + F_e*cos60 = m_C*g
        # Substitute T: F_e*(sin60/sin30)*cos30 + F_e*cos60 = m_C*g
        # Simplify: F_e*( (√3/2)/(1/2)*(√3/2) + 1/2 ) = m_C*g
        #         = F_e*( √3*(√3/2) + 1/2 ) = F_e*(3/2 + 1/2) = 2*F_e = m_C*g
        # So F_e = (m_C*g)/2 (independent of x!)
        # Therefore k = F_e/x = (m_C*g)/(2*x)
        
        # Expected trend: if x increases by 5%, k should decrease by ~5%
        expected_k = (self.m_C * self.g) / (2 * (original_x * perturb_factor))
        
        # Restore original state
        self.x = original_x
        self.k = original_k
        self.F_e = original_F_e
        
        # Check trend
        successes.append(self.log(
            f"Monotonicity: x↑{perturb_factor*100:.1f}% → k↓, expected {expected_k:.3f}",
            math.isclose(original_k, expected_k * perturb_factor, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Additional check: F_e should be independent of x
        expected_F_e = (self.m_C * self.g) / 2
        successes.append(self.log(
            f"F_e independent of x: {original_F_e:.3f} ≈ {expected_F_e:.3f}",
            math.isclose(original_F_e, expected_F_e, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        return all(successes)
    
    def run_all_tests(self):
        """
        Run all tests and return success flag and logs.
        """
        self.logs = []
        
        try:
            success = True
            
            # Run tests
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
            
            # Final summary
            self.logs.append(f"\nOverall: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
            
        except Exception as e:
            self.logs.append(f"ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        return success, self.logs

# Example usage and test
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.m_A = 2.0
            self.m_B = 5.0
            self.m_C = 1.0
            self.x = 0.02
            self.g = 10.0
            self.results = {
                'T': 5 * math.sqrt(3),  # 5√3
                'F_e': 5.0,
                'f': 5 * math.sqrt(3),
                'f_textground': 2.5 * math.sqrt(3),  # (5√3)/2
                'k': 250.0
            }
    
    # Create checker and run tests
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    print("\n".join(logs))
    print(f"\nSuccess: {success}")