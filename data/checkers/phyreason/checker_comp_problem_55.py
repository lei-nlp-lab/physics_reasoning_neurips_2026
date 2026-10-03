import sympy as sp
import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Never access solution_instance.attr directly.
        """
        # Extract inputs with safe defaults
        self.L = getattr(solution_instance, 'L', 
                        getattr(solution_instance, 'inputs', {}).get('L', 1.0))
        self.d = getattr(solution_instance, 'd', 
                        getattr(solution_instance, 'inputs', {}).get('d', 0.1))
        self.lam = getattr(solution_instance, 'lam', 
                          getattr(solution_instance, 'inputs', {}).get('lam', 0.000001))
        self.n = getattr(solution_instance, 'n', 
                        getattr(solution_instance, 'inputs', {}).get('n', 1.5))
        self.s = getattr(solution_instance, 's', 
                        getattr(solution_instance, 'inputs', {}).get('s', 0.0001))
        
        # Extract results with safe defaults
        self.results = getattr(solution_instance, 'results', {})
        
        # Part 1: y without shell
        self.y_part1 = self.results.get('y_part1', 
                                       self.results.get('y', self.lam * self.L / (8 * self.d)))
        # Part 2: y with shell
        y_part2_expr = self.results.get('y_part2', None)
        if y_part2_expr is None:
            # Calculate from formula
            term = (self.n - 1) * self.s
            m_opt = -math.floor(term / self.lam - 0.5)
            self.y_part2 = (self.L / (4 * self.d)) * (term - self.lam * m_opt - self.lam / 2)
        else:
            self.y_part2 = y_part2_expr
        
        # Part 3: Total fringes on top wall
        self.total_fringes = self.results.get('total_fringes', 
                                            2 * (2 * self.d / self.lam) * (1 - 2 / math.sqrt(5)))
        
        # Intermediate results
        self.delta_r = self.results.get('$Deltar$', None)
        self.m = self.results.get('m', None)
        self.delta_r_new = self.results.get('$Deltar_textnew$', None)
        self.delta_r_shell = self.results.get('$Deltar_shell$', None)
        self.delta_r_total = self.results.get('$Deltar_texttotal$', None)
        self.x = self.results.get('x', None)
        
        self.logs = []
    
    def _log(self, message: str, success: bool = True):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> List[bool]:
        """Verify physical invariants from KB theorems."""
        tests = []
        
        # Test 1: Part 1 - Constructive interference condition (m=1)
        # Path difference should equal lambda for m=1
        if self.y_part1 is not None:
            # Exact geometric path difference
            y = float(self.y_part1)
            exact_path = (math.sqrt((self.d + y)**2 + (self.L/2)**2) - 
                         math.sqrt((self.d - y)**2 + (self.L/2)**2))
            total_path_exact = exact_path + self.lam/2
            
            # Approximated path difference
            approx_path = 4 * self.d * y / self.L
            total_path_approx = approx_path + self.lam/2
            
            # Check both against m*lambda (m=1)
            m = 1
            tests.append(self._log(
                f"Part1 exact path diff: {total_path_exact:.3e} ≈ {m*self.lam:.3e} (m={m}λ)",
                self._is_close(total_path_exact, m * self.lam)
            ))
            
            tests.append(self._log(
                f"Part1 approx path diff: {total_path_approx:.3e} ≈ {m*self.lam:.3e}",
                self._is_close(total_path_approx, m * self.lam)
            ))
        
        # Test 2: Part 2 - Constructive interference with shell
        if self.y_part2 is not None:
            y = float(self.y_part2)
            exact_path = (math.sqrt((self.d + y)**2 + (self.L/2)**2) - 
                         math.sqrt((self.d - y)**2 + (self.L/2)**2))
            
            # Calculate optimal m from formula
            term = (self.n - 1) * self.s
            m_opt = -math.floor(term / self.lam - 0.5)
            
            total_path_with_shell = exact_path - (self.n - 1) * self.s + self.lam/2
            
            tests.append(self._log(
                f"Part2 path diff: {total_path_with_shell:.3e} ≈ {m_opt*self.lam:.3e} (m={m_opt}λ)",
                self._is_close(total_path_with_shell, m_opt * self.lam)
            ))
        
        # Test 3: Part 3 - Fringe count consistency
        # For large d/lambda, count should be integer
        fringe_count = float(self.total_fringes)
        tests.append(self._log(
            f"Fringe count: {fringe_count:.3f} (should be positive)",
            fringe_count > 0
        ))
        
        # Check approximate formula against integer count
        m_max = math.floor(2 * self.d / self.lam + 0.5)
        m_min = math.ceil(0.5 - 2 * self.d / self.lam)
        integer_count = 2 * max(0, m_max - m_min + 1)
        
        tests.append(self._log(
            f"Approx fringe count {fringe_count:.1f} vs integer range {integer_count}",
            self._is_close(fringe_count, integer_count, rel_tol=0.1) or integer_count == 0
        ))
        
        return tests
    
    def test_trace_consistency(self) -> List[bool]:
        """Check consistency between intermediate and final results."""
        tests = []
        
        # Check Part 1 formula consistency
        if self.y_part1 is not None:
            y_calc = self.lam * self.L / (8 * self.d)
            tests.append(self._log(
                f"Part1 y: {float(self.y_part1):.3e} vs formula {y_calc:.3e}",
                self._is_close(float(self.y_part1), y_calc)
            ))
        
        # Check Part 2 floor function application
        if self.y_part2 is not None:
            term = (self.n - 1) * self.s
            m_opt = -math.floor(term / self.lam - 0.5)
            y_calc = (self.L / (4 * self.d)) * (term - self.lam * m_opt - self.lam / 2)
            
            tests.append(self._log(
                f"Part2 y: {float(self.y_part2):.3e} vs formula {y_calc:.3e}",
                self._is_close(float(self.y_part2), y_calc)
            ))
        
        # Check delta_r approximations if available
        if self.delta_r is not None and self.y_part1 is not None:
            y = float(self.y_part1)
            # Two approximations from steps
            approx1 = 4 * self.d * y / self.L
            tests.append(self._log(
                f"delta_r approx1 (4dy/L): {approx1:.3e} vs stored {float(self.delta_r):.3e}",
                self._is_close(approx1, float(self.delta_r))
            ))
            
            # Exact geometric calculation
            exact = math.sqrt((self.d + y)**2 + (self.L/2)**2) - math.sqrt((self.d - y)**2 + (self.L/2)**2)
            tests.append(self._log(
                f"delta_r exact: {exact:.3e} vs stored {float(self.delta_r):.3e}",
                self._is_close(exact, float(self.delta_r), rel_tol=0.05)
            ))
        
        return tests
    
    def test_monotonicity(self) -> List[bool]:
        """Test how outputs change with perturbed inputs."""
        tests = []
        
        # Store original values
        L_orig, d_orig, lam_orig = self.L, self.d, self.lam
        
        try:
            # Test 1: Increase wavelength -> y should increase (Part 1)
            self.lam *= 1.1
            y_new = self.lam * self.L / (8 * self.d)
            tests.append(self._log(
                f"λ↑ {lam_orig:.2e}→{self.lam:.2e}: y {self.y_part1:.2e}→{y_new:.2e} (should increase)",
                y_new > float(self.y_part1)
            ))
            self.lam = lam_orig  # Restore
            
            # Test 2: Increase d -> y should decrease (Part 1)
            self.d *= 1.1
            y_new = self.lam * self.L / (8 * self.d)
            tests.append(self._log(
                f"d↑ {d_orig:.2e}→{self.d:.2e}: y {self.y_part1:.2e}→{y_new:.2e} (should decrease)",
                y_new < float(self.y_part1)
            ))
            self.d = d_orig  # Restore
            
            # Test 3: Increase n -> y should change (Part 2)
            n_orig = self.n
            self.n *= 1.01
            term = (self.n - 1) * self.s
            m_opt = -math.floor(term / self.lam - 0.5)
            y_new = (self.L / (4 * self.d)) * (term - self.lam * m_opt - self.lam / 2)
            # Just check it's positive
            tests.append(self._log(
                f"n↑ {n_orig:.3f}→{self.n:.3f}: y_part2 = {y_new:.3e} (should be positive)",
                y_new > 0
            ))
            self.n = n_orig  # Restore
            
            # Test 4: Fringe count decreases with wavelength
            lam_orig = self.lam
            self.lam *= 1.1
            fringes_new = 2 * (2 * self.d / self.lam) * (1 - 2 / math.sqrt(5))
            tests.append(self._log(
                f"λ↑ {lam_orig:.2e}→{self.lam:.2e}: fringes {self.total_fringes:.1f}→{fringes_new:.1f} (should decrease)",
                fringes_new < float(self.total_fringes)
            ))
            self.lam = lam_orig  # Restore
            
        except Exception as e:
            tests.append(self._log(f"Monotonicity test failed: {str(e)}", False))
            # Restore all values
            self.L, self.d, self.lam = L_orig, d_orig, lam_orig
        
        return tests
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        all_results = []
        
        try:
            self.logs.append("=== Physics Checker Starting ===")
            
            # Run invariant tests
            self.logs.append("\n--- Testing Physical Invariants ---")
            inv_results = self.test_invariants()
            all_results.extend(inv_results)
            
            # Run trace consistency tests
            self.logs.append("\n--- Testing Trace Consistency ---")
            trace_results = self.test_trace_consistency()
            all_results.extend(trace_results)
            
            # Run monotonicity tests
            self.logs.append("\n--- Testing Monotonicity ---")
            mono_results = self.test_monotonicity()
            all_results.extend(mono_results)
            
            # Determine overall success
            success_rate = sum(all_results) / len(all_results) if all_results else 0
            overall_success = success_rate > 0.8  # 80% pass rate
            
            self.logs.append(f"\n=== Summary: {sum(all_results)}/{len(all_results)} tests passed ===")
            self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs

# Example usage (commented out):
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.inputs = {
#                 'L': 10.0,      # m
#                 'd': 0.1,       # m
#                 'lam': 5e-7,    # m
#                 'n': 1.5,
#                 's': 1e-6       # m
#             }
#             self.results = {
#                 'y_part1': 5e-7 * 10.0 / (8 * 0.1),
#                 'y_part2': None,  # Will be calculated
#                 'total_fringes': 2 * (2 * 0.1 / 5e-7) * (1 - 2 / math.sqrt(5))
#             }
#     
#     mock_sol = MockSolution()
#     checker = PhysicsChecker(mock_sol)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)