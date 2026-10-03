import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Constants from problem statement
        self.m = 1.0  # kg
        self.θ = 37.0  # degrees
        self.sinθ = 0.6
        self.cosθ = 0.8
        self.μ = 0.5
        self.g = 10.0  # m/s²
        self.t1 = 2.0  # s (first phase duration)
        self.t3 = 1.0  # s (downward phase after highest point)
        
        # Safely extract results using getattr with fallback to results dict
        self.results = {}
        expected_keys = ['x', 'a_1', 'x_3', 'F', 't_2', 'x_1', 'a_2', 'x_2', 'a_3', 'v_1']
        
        for key in expected_keys:
            # First try attribute, then results dict, then None
            val = getattr(solution_instance, key, None)
            if val is None and hasattr(solution_instance, 'results'):
                val = solution_instance.results.get(key, None)
            self.results[key] = val
            
        # Extract critical values with defaults
        self.x = self.results.get('x', 0.0)
        self.a1 = self.results.get('a_1', 0.0)
        self.x3 = self.results.get('x_3', 0.0)
        self.F = self.results.get('F', 0.0)
        self.t2 = self.results.get('t_2', 0.0)
        self.x1 = self.results.get('x_1', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.a3 = self.results.get('a_3', 0.0)
        self.v1 = self.results.get('v_1', 0.0)
        
        # Expected values from steps analysis
        self.expected = {
            'a_1': 4.0,
            'F': 14.0,
            'v_1': 8.0,
            'x_1': 8.0,
            'a_2': -10.0,
            't_2': 0.8,
            'x_2': 3.2,
            'a_3': 2.0,
            'x_3': 1.0,
            'x': 10.2
        }

    def is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical matching with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physical laws based on KB and steps analysis."""
        logs = []
        success = True
        
        # 1. Check uniformly accelerated motion for first phase
        # v1 = a1 * t1 (since v0 = 0)
        v1_from_a1 = self.a1 * self.t1
        if not self.is_close(self.v1, v1_from_a1):
            logs.append(f"FAIL: v1 = {self.v1}, but a1*t1 = {v1_from_a1}")
            success = False
        else:
            logs.append("PASS: v1 = a1 * t1 (uniform acceleration)")
            
        # 2. Check Newton's 2nd Law for first phase
        # F_net = F - mgsinθ - μmgcosθ = m*a1
        F_net_expected = self.m * self.a1
        F_net_calculated = self.F - self.m * self.g * self.sinθ - self.μ * self.m * self.g * self.cosθ
        if not self.is_close(F_net_calculated, F_net_expected):
            logs.append(f"FAIL: Newton's 2nd Law (phase 1). Net force = {F_net_calculated}, expected {F_net_expected}")
            success = False
        else:
            logs.append("PASS: Newton's 2nd Law (phase 1)")
            
        # 3. Check displacement in first phase using average velocity
        # x1 = (v1/2) * t1
        x1_from_v1 = (self.v1 / 2.0) * self.t1
        if not self.is_close(self.x1, x1_from_v1):
            logs.append(f"FAIL: x1 = {self.x1}, but (v1/2)*t1 = {x1_from_v1}")
            success = False
        else:
            logs.append("PASS: x1 = (v1/2)*t1 (average velocity)")
            
        # 4. Check Newton's 2nd Law for upward phase after force removal
        # -mgsinθ - μmgcosθ = m*a2
        F_net_upward = -self.m * self.g * self.sinθ - self.μ * self.m * self.g * self.cosθ
        a2_expected = F_net_upward / self.m
        if not self.is_close(self.a2, a2_expected):
            logs.append(f"FAIL: a2 = {self.a2}, but expected {a2_expected} from Newton's 2nd Law")
            success = False
        else:
            logs.append("PASS: Newton's 2nd Law (upward phase)")
            
        # 5. Check time to highest point: t2 = -v1 / a2
        t2_expected = -self.v1 / self.a2
        if not self.is_close(self.t2, t2_expected):
            logs.append(f"FAIL: t2 = {self.t2}, but -v1/a2 = {t2_expected}")
            success = False
        else:
            logs.append("PASS: t2 = -v1/a2")
            
        # 6. Check displacement during upward phase: 0 - v1² = 2*a2*x2
        if abs(self.a2) > 1e-9:
            x2_expected = -self.v1**2 / (2.0 * self.a2)
            if not self.is_close(self.x2, x2_expected):
                logs.append(f"FAIL: x2 = {self.x2}, but -v1²/(2a2) = {x2_expected}")
                success = False
            else:
                logs.append("PASS: x2 = -v1²/(2a2)")
                
        # 7. Check Newton's 2nd Law for downward phase
        # mgsinθ - μmgcosθ = m*a3
        F_net_downward = self.m * self.g * self.sinθ - self.μ * self.m * self.g * self.cosθ
        a3_expected = F_net_downward / self.m
        if not self.is_close(self.a3, a3_expected):
            logs.append(f"FAIL: a3 = {self.a3}, but expected {a3_expected} from Newton's 2nd Law")
            success = False
        else:
            logs.append("PASS: Newton's 2nd Law (downward phase)")
            
        # 8. Check displacement in downward phase: x3 = 0.5*a3*t3²
        x3_expected = 0.5 * self.a3 * self.t3**2
        if not self.is_close(self.x3, x3_expected):
            logs.append(f"FAIL: x3 = {self.x3}, but 0.5*a3*t3² = {x3_expected}")
            success = False
        else:
            logs.append("PASS: x3 = 0.5*a3*t3²")
            
        # 9. Check final position: x = x1 + x2 - x3
        x_expected = self.x1 + self.x2 - self.x3
        if not self.is_close(self.x, x_expected):
            logs.append(f"FAIL: Final x = {self.x}, but x1 + x2 - x3 = {x_expected}")
            success = False
        else:
            logs.append("PASS: x = x1 + x2 - x3 (vector addition)")
            
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        # Check against expected values from steps analysis
        for key, expected_val in self.expected.items():
            actual_val = getattr(self, key.replace('_', ''), self.results.get(key, None))
            if actual_val is not None and not self.is_close(actual_val, expected_val):
                logs.append(f"FAIL: {key} = {actual_val}, expected {expected_val}")
                success = False
            elif actual_val is not None:
                logs.append(f"PASS: {key} = {actual_val} matches expected")
                
        # Additional consistency: a1 should be positive, a2 negative, a3 positive
        if self.a1 <= 0:
            logs.append(f"WARN: a1 = {self.a1} should be positive")
        if self.a2 >= 0:
            logs.append(f"WARN: a2 = {self.a2} should be negative")
        if self.a3 <= 0:
            logs.append(f"WARN: a3 = {self.a3} should be positive")
            
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb coefficient of friction and check trend."""
        logs = []
        success = True
        
        # Save original μ
        μ_original = self.μ
        
        # Perturb μ by +10%
        self.μ = μ_original * 1.1
        
        # Recalculate accelerations using formulas
        a2_perturbed = -self.g * self.sinθ - self.μ * self.g * self.cosθ
        a3_perturbed = self.g * self.sinθ - self.μ * self.g * self.cosθ
        
        # Check trends
        # With higher μ, upward deceleration should be more negative (larger magnitude)
        if not (a2_perturbed < self.a2):
            logs.append(f"FAIL: Increasing μ should make a2 more negative. Original: {self.a2}, Perturbed: {a2_perturbed}")
            success = False
        else:
            logs.append(f"PASS: a2 becomes more negative with higher μ ({self.a2} → {a2_perturbed})")
            
        # With higher μ, downward acceleration should decrease
        if not (a3_perturbed < self.a3):
            logs.append(f"FAIL: Increasing μ should decrease a3. Original: {self.a3}, Perturbed: {a3_perturbed}")
            success = False
        else:
            logs.append(f"PASS: a3 decreases with higher μ ({self.a3} → {a3_perturbed})")
            
        # Restore original μ
        self.μ = μ_original
        
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Testing Physical Invariants ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Testing Trace Consistency ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Testing Monotonicity (μ perturbation) ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs

# Example usage (commented out for safety):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'a_1': 4.0,
#                 'F': 14.0,
#                 'v_1': 8.0,
#                 'x_1': 8.0,
#                 'a_2': -10.0,
#                 't_2': 0.8,
#                 'x_2': 3.2,
#                 'a_3': 2.0,
#                 'x_3': 1.0,
#                 'x': 10.2
#             }
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)