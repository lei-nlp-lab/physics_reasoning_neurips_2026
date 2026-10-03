import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract physical parameters (hardcoded from problem statement)
        self.R = 90.0  # m
        self.r = 40.0  # m
        self.L = 100.0  # m
        self.k = 2.25  # friction coefficient multiplier
        self.g = 10.0  # m/s²
        
        # Extract results with safe access
        self.v = self._safe_get('v', 0.0)  # speed on small arc
        self.v_m = self._safe_get('v_m', 0.0)  # speed on large arc
        self.x = self._safe_get('x', 0.0)  # straight section length
        self.a = self._safe_get('a', 0.0)  # acceleration magnitude
        self.alpha = self._safe_get('alpha', 0.0)  # angle in degrees
        self.t_small = self._safe_get('t', 0.0)  # time on small arc
        self.t_large = self._safe_get("t'", 0.0)  # time on large arc
        self.t_straight = self._safe_get('t_straight', 0.0)  # time on straights
        self.t_all = self._safe_get('t_all', 0.0)  # total lap time
        
        # For symbolic calculations
        self.symbols = self._init_symbols()
    
    def _safe_get(self, key: str, default: float) -> float:
        """Safely extract a numerical value from results."""
        val = self.results.get(key, default)
        if isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, str):
            try:
                # Handle expressions like "50*sqrt(3)" or "8*pi/9"
                expr = sp.sympify(val.replace('\\', ''))
                return float(expr.evalf())
            except:
                return default
        return default
    
    def _init_symbols(self):
        """Initialize symbolic variables for consistency checks."""
        R, r, L, k, g = sp.symbols('R r L k g', positive=True)
        v, v_m, x, a = sp.symbols('v v_m x a', positive=True)
        alpha = sp.symbols('alpha', positive=True)
        t_small, t_large, t_straight, t_all = sp.symbols('t_small t_large t_straight t_all', positive=True)
        
        return {
            'R': R, 'r': r, 'L': L, 'k': k, 'g': g,
            'v': v, 'v_m': v_m, 'x': x, 'a': a, 'alpha': alpha,
            't_small': t_small, 't_large': t_large, 't_straight': t_straight, 't_all': t_all
        }
    
    def _num_is_close(self, a: float, b: float, tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Check physical invariants from KB."""
        logs = []
        success = True
        
        # 1. Check centripetal force conditions
        # For small arc: v = sqrt(k*g*r)
        v_expected = math.sqrt(self.k * self.g * self.r)
        if not self._num_is_close(self.v, v_expected):
            logs.append(f"FAIL: Small arc speed v={self.v:.3f}, expected {v_expected:.3f} from v=√(k*g*r)")
            success = False
        else:
            logs.append(f"PASS: Small arc speed v={self.v:.3f} matches √(k*g*r)={v_expected:.3f}")
        
        # For large arc: v_m = sqrt(k*g*R)
        v_m_expected = math.sqrt(self.k * self.g * self.R)
        if not self._num_is_close(self.v_m, v_m_expected):
            logs.append(f"FAIL: Large arc speed v_m={self.v_m:.3f}, expected {v_m_expected:.3f} from v_m=√(k*g*R)")
            success = False
        else:
            logs.append(f"PASS: Large arc speed v_m={self.v_m:.3f} matches √(k*g*R)={v_m_expected:.3f}")
        
        # 2. Check Pythagorean theorem for straight section
        x_expected = math.sqrt(self.L**2 - (self.R - self.r)**2)
        if not self._num_is_close(self.x, x_expected):
            logs.append(f"FAIL: Straight length x={self.x:.3f}, expected √(L²-(R-r)²)={x_expected:.3f}")
            success = False
        else:
            logs.append(f"PASS: Straight length x={self.x:.3f} matches √(L²-(R-r)²)={x_expected:.3f}")
        
        # 3. Check kinematics for acceleration (constant acceleration formula)
        if abs(self.x) > 1e-9:
            a_expected = (self.v_m**2 - self.v**2) / (2 * self.x)
            if not self._num_is_close(self.a, a_expected):
                logs.append(f"FAIL: Acceleration a={self.a:.3f}, expected (v_m²-v²)/(2x)={a_expected:.3f}")
                success = False
            else:
                logs.append(f"PASS: Acceleration a={self.a:.3f} matches (v_m²-v²)/(2x)={a_expected:.3f}")
        
        # 4. Check angle alpha from geometry
        alpha_expected_rad = math.acos((self.R - self.r) / self.L)
        alpha_expected_deg = math.degrees(alpha_expected_rad)
        if not self._num_is_close(self.alpha, alpha_expected_deg):
            logs.append(f"FAIL: Angle α={self.alpha:.3f}°, expected arccos((R-r)/L)={alpha_expected_deg:.3f}°")
            success = False
        else:
            logs.append(f"PASS: Angle α={self.alpha:.3f}° matches arccos((R-r)/L)={alpha_expected_deg:.3f}°")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate and final results."""
        logs = []
        success = True
        
        # 1. Check arc times from geometry and speeds
        # Small arc: central angle = 120° = 2π/3 rad
        arc_small = (2 * math.pi * self.r) * (120 / 360)
        t_small_expected = arc_small / self.v if abs(self.v) > 1e-9 else 0
        if not self._num_is_close(self.t_small, t_small_expected):
            logs.append(f"FAIL: Small arc time t={self.t_small:.3f}s, expected (2πr*120°/360°)/v={t_small_expected:.3f}s")
            success = False
        else:
            logs.append(f"PASS: Small arc time t={self.t_small:.3f}s matches arc_length/v")
        
        # Large arc: central angle = 240° = 4π/3 rad
        arc_large = (2 * math.pi * self.R) * (240 / 360)
        t_large_expected = arc_large / self.v_m if abs(self.v_m) > 1e-9 else 0
        if not self._num_is_close(self.t_large, t_large_expected):
            logs.append(f"FAIL: Large arc time t'={self.t_large:.3f}s, expected (2πR*240°/360°)/v_m={t_large_expected:.3f}s")
            success = False
        else:
            logs.append(f"PASS: Large arc time t'={self.t_large:.3f}s matches arc_length/v_m")
        
        # 2. Check straight section time (constant acceleration)
        if abs(self.a) > 1e-9:
            t_straight_expected = 2 * (self.v_m - self.v) / self.a
            if not self._num_is_close(self.t_straight, t_straight_expected):
                logs.append(f"FAIL: Straight time t_straight={self.t_straight:.3f}s, expected 2*(v_m-v)/a={t_straight_expected:.3f}s")
                success = False
            else:
                logs.append(f"PASS: Straight time t_straight={self.t_straight:.3f}s matches 2*(v_m-v)/a")
        
        # 3. Check total time summation
        t_all_sum = self.t_small + self.t_large + self.t_straight
        if not self._num_is_close(self.t_all, t_all_sum):
            logs.append(f"FAIL: Total time t_all={self.t_all:.3f}s ≠ t+t'+t_straight={t_all_sum:.3f}s")
            success = False
        else:
            logs.append(f"PASS: Total time t_all={self.t_all:.3f}s equals sum of components")
        
        # 4. Verify numerical values against expected expressions
        # Expected values from steps analysis
        expected_v = 30.0
        expected_v_m = 45.0
        expected_x = 50 * math.sqrt(3)  # ≈ 86.6025
        expected_a = (15 * math.sqrt(3)) / 4  # ≈ 6.4952
        expected_alpha = 60.0  # degrees
        expected_t_small = (8 * math.pi) / 9  # ≈ 2.7925
        expected_t_large = (8 * math.pi) / 3  # ≈ 8.3776
        expected_t_straight = (8 * math.sqrt(3)) / 3  # ≈ 4.6188
        expected_t_all = (32 * math.pi + 24 * math.sqrt(3)) / 9  # ≈ 15.7889
        
        checks = [
            ('v', self.v, expected_v),
            ('v_m', self.v_m, expected_v_m),
            ('x', self.x, expected_x),
            ('a', self.a, expected_a),
            ('alpha', self.alpha, expected_alpha),
            ('t_small', self.t_small, expected_t_small),
            ('t_large', self.t_large, expected_t_large),
            ('t_straight', self.t_straight, expected_t_straight),
            ('t_all', self.t_all, expected_t_all)
        ]
        
        for name, actual, expected in checks:
            if not self._num_is_close(actual, expected):
                logs.append(f"WARN: {name}={actual:.5f} differs from expected {expected:.5f}")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test monotonic trends by perturbing parameters."""
        logs = []
        success = True
        
        # Test 1: Increasing r should increase v (v = √(k*g*r))
        original_r = self.r
        perturbed_r = original_r * 1.01  # +1%
        original_v = math.sqrt(self.k * self.g * original_r)
        perturbed_v = math.sqrt(self.k * self.g * perturbed_r)
        
        if perturbed_v <= original_v:
            logs.append(f"FAIL: Monotonicity: v should increase with r, but {perturbed_v:.3f} ≤ {original_v:.3f}")
            success = False
        else:
            logs.append(f"PASS: Monotonicity: v increases with r ({original_v:.3f} → {perturbed_v:.3f})")
        
        # Test 2: Increasing R should increase v_m
        original_R = self.R
        perturbed_R = original_R * 1.01
        original_v_m = math.sqrt(self.k * self.g * original_R)
        perturbed_v_m = math.sqrt(self.k * self.g * perturbed_R)
        
        if perturbed_v_m <= original_v_m:
            logs.append(f"FAIL: Monotonicity: v_m should increase with R, but {perturbed_v_m:.3f} ≤ {original_v_m:.3f}")
            success = False
        else:
            logs.append(f"PASS: Monotonicity: v_m increases with R ({original_v_m:.3f} → {perturbed_v_m:.3f})")
        
        # Test 3: Increasing L (while keeping R-r constant) should increase x
        original_L = self.L
        perturbed_L = original_L * 1.01
        original_x = math.sqrt(original_L**2 - (self.R - self.r)**2)
        perturbed_x = math.sqrt(perturbed_L**2 - (self.R - self.r)**2)
        
        if perturbed_x <= original_x:
            logs.append(f"FAIL: Monotonicity: x should increase with L, but {perturbed_x:.3f} ≤ {original_x:.3f}")
            success = False
        else:
            logs.append(f"PASS: Monotonicity: x increases with L ({original_x:.3f} → {perturbed_x:.3f})")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return combined results."""
        all_logs = []
        overall_success = True
        
        # Run individual tests
        for test_name, test_func in [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            success, logs = test_func()
            overall_success = overall_success and success
            all_logs.extend(logs)
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All physics checks passed within tolerances.")
        else:
            all_logs.append("WARNING: Some physics checks failed.")
        
        return overall_success, all_logs

# Example usage (commented out for clarity):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'v': 30.0,
#                 'v_m': 45.0,
#                 'x': 50*math.sqrt(3),
#                 'a': (15*math.sqrt(3))/4,
#                 'alpha': 60.0,
#                 't': (8*math.pi)/9,
#                 "t'": (8*math.pi)/3,
#                 't_straight': (8*math.sqrt(3))/3,
#                 't_all': (32*math.pi + 24*math.sqrt(3))/9
#             }
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)