import math
import sympy as sp
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters
        self.m = getattr(solution_instance, 'm', 0.5)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.F_N = getattr(solution_instance, 'F_N', 40.0)  # Horizontal pressure
        self.mu_board = getattr(solution_instance, 'mu_board', 0.4)  # Board-book friction
        
        # Extract results with safe fallbacks
        self.f = getattr(solution_instance, 'f', results.get('f', 0.0))
        self.mu = getattr(solution_instance, 'mu', results.get('mu', 0.0))
        self.F_1 = getattr(solution_instance, 'F_1', results.get('F_1', 0.0))
        self.f_12 = getattr(solution_instance, 'f_12', results.get('f_12', 0.0))
        self.two_f = getattr(solution_instance, 'two_f', results.get('2f', 0.0))
        
        # Derived quantities
        self.total_mass = 6 * self.m
        self.total_weight = self.total_mass * self.g
        
        # For symbolic calculations
        self.m_sym = sp.symbols('m', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        self.F_sym = sp.symbols('F', positive=True)  # F_N
        self.mu_sym = sp.symbols('mu', positive=True)
        
        self.logs = []
    
    def _log(self, message: str, success: bool):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def test_invariants(self) -> List[bool]:
        """Verify physics laws and consistency with KB theorems."""
        tests = []
        
        # 1. Equilibrium of Concurrent Forces (vertical equilibrium)
        # Total friction from both boards = total weight
        expected_two_f = self.total_weight  # 6*m*g
        actual_two_f = self.two_f if self.two_f != 0.0 else 2 * self.f
        tests.append(self._log(
            f"Total friction 2f = {actual_two_f:.2f} N, should equal total weight {expected_two_f:.2f} N",
            math.isclose(actual_two_f, expected_two_f, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 2. Symmetry check: friction per board
        expected_f = 3 * self.m * self.g  # 3*m*g
        tests.append(self._log(
            f"Friction per board f = {self.f:.2f} N, should equal {expected_f:.2f} N",
            math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 3. Equilibrium for leftmost book: mg + f_12 = f
        left_equation = self.m * self.g + self.f_12
        tests.append(self._log(
            f"Left book equilibrium: mg + f_12 = {self.m*self.g:.1f} + {self.f_12:.1f} = {left_equation:.1f} N, should equal f = {self.f:.1f} N",
            math.isclose(left_equation, self.f, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 4. Friction definition: f_12 = μ * F_N
        expected_mu = self.f_12 / self.F_N
        tests.append(self._log(
            f"Coefficient μ = {self.mu:.3f}, should equal f_12/F_N = {self.f_12:.1f}/{self.F_N:.1f} = {expected_mu:.3f}",
            math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 5. Force composition for pulling second book
        expected_F1 = self.m * self.g + 2 * self.mu * self.F_N
        tests.append(self._log(
            f"Pull force F1 = {self.F_1:.1f} N, should equal mg + 2μF = {self.m*self.g:.1f} + 2*{self.mu:.3f}*{self.F_N:.1f} = {expected_F1:.1f} N",
            math.isclose(self.F_1, expected_F1, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 6. Board friction limit check (should be less than max static friction)
        max_board_friction = self.mu_board * self.F_N
        tests.append(self._log(
            f"Board friction f = {self.f:.1f} N ≤ μ_board*F_N = {self.mu_board:.1f}*{self.F_N:.1f} = {max_board_friction:.1f} N",
            self.f <= max_board_friction + 1e-2  # Allow small tolerance
        ))
        
        return tests
    
    def test_trace_consistency(self) -> List[bool]:
        """Cross-check intermediate results with final answers."""
        tests = []
        
        # Check target answers from problem statement
        target_f = 15.0
        target_mu = 0.25
        target_F1 = 25.0
        
        tests.append(self._log(
            f"Friction f = {self.f:.1f} N matches target {target_f:.1f} N",
            math.isclose(self.f, target_f, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        tests.append(self._log(
            f"Coefficient μ = {self.mu:.3f} matches target {target_mu:.3f}",
            math.isclose(self.mu, target_mu, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        tests.append(self._log(
            f"Pull force F1 = {self.F_1:.1f} N matches target {target_F1:.1f} N",
            math.isclose(self.F_1, target_F1, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Check f_12 from step 3
        expected_f12 = self.f - self.m * self.g
        tests.append(self._log(
            f"Book-book friction f_12 = {self.f_12:.1f} N, should equal f - mg = {self.f:.1f} - {self.m*self.g:.1f} = {expected_f12:.1f} N",
            math.isclose(self.f_12, expected_f12, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        return tests
    
    def test_monotonicity(self) -> List[bool]:
        """Perturb mass and check trend, then restore."""
        original_m = self.m
        tests = []
        
        try:
            # Increase mass by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected values using formulas
            perturbed_f = 3 * perturbed_m * self.g
            perturbed_f12 = perturbed_f - perturbed_m * self.g
            perturbed_mu = perturbed_f12 / self.F_N
            perturbed_F1 = perturbed_m * self.g + 2 * perturbed_mu * self.F_N
            
            # Check monotonic relationships
            # 1. If mass increases, friction f should increase
            tests.append(self._log(
                f"Mass increase: f increases from {self.f:.1f} to {perturbed_f:.1f}",
                perturbed_f > self.f
            ))
            
            # 2. If mass increases, f_12 should increase (but check formula)
            tests.append(self._log(
                f"Mass increase: f_12 increases from {self.f_12:.1f} to {perturbed_f12:.1f}",
                perturbed_f12 > self.f_12
            ))
            
            # 3. μ should remain constant since it's independent of mass in this setup
            # Actually μ = f_12/F_N, and f_12 changes with mass, so μ changes
            expected_mu_change = (perturbed_f12 - self.f_12) / self.F_N
            tests.append(self._log(
                f"Mass increase: μ changes from {self.mu:.3f} to {perturbed_mu:.3f}",
                abs(perturbed_mu - self.mu) > 1e-3  # Should change
            ))
            
            # 4. F1 should increase with mass
            tests.append(self._log(
                f"Mass increase: F1 increases from {self.F_1:.1f} to {perturbed_F1:.1f}",
                perturbed_F1 > self.F_1
            ))
            
        except Exception as e:
            tests.append(self._log(f"Monotonicity test failed: {str(e)}", False))
        finally:
            # Restore original mass
            self.m = original_m
            
        return tests
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        self.logs = []
        all_passed = True
        
        try:
            self.logs.append("=== Physics Checker Starting ===")
            
            # Run invariant tests
            self.logs.append("\n1. Testing Physics Invariants:")
            inv_tests = self.test_invariants()
            if not all(inv_tests):
                all_passed = False
            
            # Run trace consistency tests
            self.logs.append("\n2. Testing Trace Consistency:")
            trace_tests = self.test_trace_consistency()
            if not all(trace_tests):
                all_passed = False
            
            # Run monotonicity tests
            self.logs.append("\n3. Testing Monotonicity:")
            mono_tests = self.test_monotonicity()
            if not all(mono_tests):
                all_passed = False
            
            # Summary
            self.logs.append("\n=== Summary ===")
            total_tests = len(inv_tests) + len(trace_tests) + len(mono_tests)
            passed_tests = sum(inv_tests) + sum(trace_tests) + sum(mono_tests)
            self.logs.append(f"Passed {passed_tests}/{total_tests} tests")
            
            if all_passed:
                self.logs.append("✓ All tests passed!")
            else:
                self.logs.append("✗ Some tests failed")
                
        except Exception as e:
            self.logs.append(f"✗ Checker crashed: {str(e)}")
            self.logs.append(traceback.format_exc())
            all_passed = False
        
        return all_passed, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)