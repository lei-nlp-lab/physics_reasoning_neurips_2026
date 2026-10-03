import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract inputs and results from solution_instance.
        Never access solution_instance.attr directly.
        """
        self.solution = solution_instance
        
        # Safe extraction of inputs (M, m, k, g)
        self.M = self._safe_get('M', default=1.0)
        self.m = self._safe_get('m', default=1.0)
        self.k = self._safe_get('k', default=1.0)
        self.g = self._safe_get('g', default=9.8)
        
        # Safe extraction of results
        self.N = self._safe_get('N')
        self.two_kx = self._safe_get('2kx')
        self.x = self._safe_get('x')
        self.A = self._safe_get('A')
        
        # Derived symbolic variables for symbolic checks
        self._init_symbols()
        
    def _safe_get(self, key: str, default: float = None) -> float:
        """Safe attribute access with fallback to results dictionary."""
        # First try to get from results dict
        if hasattr(self.solution, 'results') and isinstance(self.solution.results, dict):
            val = self.solution.results.get(key, default)
            if val is not None:
                return float(val)
        
        # Then try as direct attribute
        val = getattr(self.solution, key, default)
        if val is not None:
            return float(val)
        
        return default
    
    def _init_symbols(self):
        """Initialize symbolic variables for physics laws."""
        self.M_sym, self.m_sym, self.k_sym, self.g_sym = sp.symbols('M m k g', positive=True)
        self.N_sym, self.x_sym, self.A_sym = sp.symbols('N x A', positive=True)
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify physics invariants from KB:
        1. Newton's Third Law consistency
        2. Force balance in equilibrium
        3. Newton's Second Law for SHM
        """
        logs = []
        success = True
        
        # 1. Check equilibrium condition: 2kx = 2mg (from step 4)
        if self.two_kx is not None:
            expected_2kx = 2 * self.m * self.g
            if not math.isclose(self.two_kx, expected_2kx, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: 2kx = {self.two_kx} != 2mg = {expected_2kx}")
                success = False
            else:
                logs.append(f"PASS: 2kx = 2mg = {expected_2kx}")
        
        # 2. Check spring deformation: x = mg/k (from step 4)
        if self.x is not None:
            expected_x = self.m * self.g / self.k if self.k != 0 else 0
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x = {self.x} != mg/k = {expected_x}")
                success = False
            else:
                logs.append(f"PASS: x = mg/k = {expected_x}")
        
        # 3. Check amplitude equals deformation: A = x (from step 5)
        if self.A is not None and self.x is not None:
            if not math.isclose(self.A, self.x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: A = {self.A} != x = {self.x}")
                success = False
            else:
                logs.append(f"PASS: A = x = {self.x}")
        
        # 4. Check normal force: N = Mg (from steps 6 & 7)
        if self.N is not None:
            expected_N = self.M * self.g
            if not math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: N = {self.N} != Mg = {expected_N}")
                success = False
            else:
                logs.append(f"PASS: N = Mg = {expected_N}")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answer.
        Ensure all steps are mathematically consistent.
        """
        logs = []
        success = True
        
        # Check that if we have x, we can compute 2kx and compare
        if self.x is not None and self.two_kx is not None:
            computed_2kx = 2 * self.k * self.x
            if not math.isclose(self.two_kx, computed_2kx, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: 2kx = {self.two_kx} != 2*k*x = {computed_2kx}")
                success = False
            else:
                logs.append(f"PASS: 2kx consistency check passed")
        
        # Check that N is independent of m (should only depend on M and g)
        if self.N is not None:
            # Symbolic check: N = M*g (no m dependence)
            symbolic_N = sp.Eq(self.N_sym, self.M_sym * self.g_sym)
            # Check if m appears in the equation (it shouldn't)
            if self.m_sym in symbolic_N.rhs.free_symbols:
                logs.append(f"FAIL: N depends on m, violating step 6")
                success = False
            else:
                logs.append(f"PASS: N correctly independent of m")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb inputs and check expected trends based on physics laws.
        We perturb m (mass of blocks) and check how results change.
        """
        logs = []
        success = True
        
        # Store original values
        original_m = self.m
        
        # Perturb m by +10%
        perturbed_m = original_m * 1.10
        
        # Expected changes based on formulas:
        # 1. x = mg/k should increase by 10%
        # 2. 2kx = 2mg should increase by 10%
        # 3. A = x should increase by 10%
        # 4. N = Mg should NOT change (independent of m)
        
        if self.x is not None:
            expected_x_perturbed = perturbed_m * self.g / self.k
            expected_ratio = expected_x_perturbed / self.x
            if not math.isclose(expected_ratio, 1.10, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x should increase by 10% when m increases by 10%")
                success = False
            else:
                logs.append(f"PASS: x monotonicity check (proportional to m)")
        
        if self.two_kx is not None:
            expected_2kx_perturbed = 2 * perturbed_m * self.g
            expected_ratio = expected_2kx_perturbed / self.two_kx
            if not math.isclose(expected_ratio, 1.10, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: 2kx should increase by 10% when m increases by 10%")
                success = False
            else:
                logs.append(f"PASS: 2kx monotonicity check (proportional to m)")
        
        if self.N is not None:
            # N should remain unchanged
            expected_N_perturbed = self.M * self.g
            if not math.isclose(expected_N_perturbed, self.N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: N should not change when m changes")
                success = False
            else:
                logs.append(f"PASS: N monotonicity check (independent of m)")
        
        # Restore original value (just in case)
        self.m = original_m
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and combined logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== Overall Result ===")
        all_logs.append(f"All tests {'PASSED' if overall_success else 'FAILED'}")
        
        return overall_success, all_logs