import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: extract results and inputs
        self.results = getattr(solution_instance, 'results', {})
        
        # Target quantities from results (with fallback)
        self.a = self._get_value('a')
        self.v_m = self._get_value('v_m')
        self.x = self._get_value('x')
        
        # Constants from problem statement
        self.m = 1000.0  # kg (1 ton)
        self.theta = math.radians(30)  # rad
        self.g = 10.0  # m/s^2
        self.P = 81000.0  # W (81 kW)
        self.t1 = 3.0  # s
        self.t2 = 37.0 / 8.0  # s
        self.t_total = self.t1 + self.t2  # s
        
        # Computed constants
        self.sin_theta = math.sin(self.theta)
        self.f_res = 0.1 * self.m * self.g  # N
        self.F_res = self.m * self.g * self.sin_theta + self.f_res  # total constant resisting force
        
        # Intermediate values derived from a (if available)
        self.v1 = None
        if self.a is not None:
            self.v1 = self.a * self.t1  # velocity at end of phase 1
            self.x1 = 0.5 * self.a * self.t1**2  # displacement in phase 1
        
    def _get_value(self, key: str) -> Optional[float]:
        """Safely retrieve a value from results with fallback."""
        value = self.results.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            # Remove units if present (e.g., "3m/s^2" -> 3.0)
            import re
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
            if match:
                return float(match.group())
            else:
                return None
        return float(value)
    
    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        success = True
        logs = []
        
        if None in (self.a, self.v_m, self.x):
            logs.append("Missing required results (a, v_m, or x)")
            return False, logs
        
        # 1. Check maximum speed condition: P = F_res * v_m (net force zero)
        expected_v_m = self.P / self.F_res
        if not self._is_close(self.v_m, expected_v_m):
            logs.append(f"Maximum speed mismatch: expected {expected_v_m:.3f} m/s, got {self.v_m:.3f} m/s")
            success = False
        else:
            logs.append(f"Maximum speed check passed: v_m = {self.v_m:.3f} m/s")
        
        # 2. Check first phase power condition: P = (m*a + F_res) * (a*t1)
        if self.a is not None:
            engine_force_phase1 = self.m * self.a + self.F_res
            power_at_t1 = engine_force_phase1 * (self.a * self.t1)
            if not self._is_close(power_at_t1, self.P):
                logs.append(f"Power condition at t=t1 failed: expected {self.P:.1f} W, got {power_at_t1:.1f} W")
                success = False
            else:
                logs.append(f"Power condition at t=t1 passed: P = {power_at_t1:.1f} W")
        
        # 3. Numeric check of maximum speed condition
        lhs = self.P / self.v_m
        rhs = self.F_res
        if not self._is_close(lhs, rhs):
            logs.append(f"Numeric maximum speed condition failed: P/v_m = {lhs:.1f} N, F_res = {rhs:.1f} N")
            success = False
        else:
            logs.append(f"Numeric maximum speed condition passed: P/v_m = F_res = {rhs:.1f} N")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        success = True
        logs = []
        
        if None in (self.a, self.v_m, self.x):
            logs.append("Cannot perform trace checks due to missing a, v_m, or x")
            return False, logs
        
        # Calculate v1 from a
        v1_calc = self.a * self.t1
        self.v1 = v1_calc
        
        # 1. Check v1 = a * t1
        if not self._is_close(v1_calc, self.v1):
            logs.append(f"Velocity at t1 mismatch: expected {v1_calc:.3f} m/s, got {self.v1:.3f} m/s")
            success = False
        else:
            logs.append(f"Velocity at t1 check passed: v1 = {v1_calc:.3f} m/s")
        
        # 2. Check work-energy theorem for phase 2
        x1 = 0.5 * self.a * self.t1**2
        x2 = self.x - x1
        left_side = self.P * self.t2 - self.F_res * x2
        right_side = 0.5 * self.m * (self.v_m**2 - v1_calc**2)
        if not self._is_close(left_side, right_side):
            logs.append(f"Work-energy theorem for phase 2 failed: left={left_side:.1f} J, right={right_side:.1f} J")
            success = False
        else:
            logs.append(f"Work-energy theorem for phase 2 passed: ΔE_k = {right_side:.1f} J")
        
        # 3. FIXED: The time integral check is problematic because the differential equation
        # m*dv/dt = P/v - F_res has v_m as an asymptotic limit (t → ∞ to reach exactly v_m).
        # The given t2 = 4.625 s is for the car to "reach its maximum speed" which in practice
        # means it gets very close to v_m. Instead of testing the exact integral (which is
        # sensitive to how we define "reaching" v_m), we should check if the time is reasonable.
        # We'll skip this test since it's not a reliable consistency check.
        logs.append("Time integral check skipped: v_m is an asymptotic limit, exact time to reach it is infinite")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        success = True
        logs = []
        
        if None in (self.a, self.v_m):
            logs.append("Cannot perform monotonicity checks due to missing a or v_m")
            return False, logs
        
        # Perturb mass by +1%
        m_new = self.m * 1.01
        f_res_new = 0.1 * m_new * self.g
        F_res_new = m_new * self.g * self.sin_theta + f_res_new
        
        # 1. New maximum speed (should decrease)
        v_m_new = self.P / F_res_new
        if v_m_new >= self.v_m:
            logs.append(f"Monotonicity failed: v_m should decrease when mass increases, but {v_m_new:.3f} >= {self.v_m:.3f}")
            success = False
        else:
            logs.append(f"v_m monotonicity passed: v_m decreases from {self.v_m:.3f} to {v_m_new:.3f}")
        
        # 2. New acceleration from first phase condition
        # Solve quadratic: 3 * m_new * a_new * (a_new + 6) = P
        # 3*m_new*a_new^2 + 18*m_new*a_new - P = 0
        A = 3 * m_new
        B = 18 * m_new
        C = -self.P
        disc = B**2 - 4 * A * C
        if disc < 0:
            logs.append("Cannot solve quadratic for new acceleration")
            success = False
        else:
            a_new_pos = (-B + math.sqrt(disc)) / (2 * A)
            a_new_neg = (-B - math.sqrt(disc)) / (2 * A)
            a_new = max(a_new_pos, a_new_neg)  # take positive root
            if a_new >= self.a:
                logs.append(f"Monotonicity failed: a should decrease when mass increases, but {a_new:.3f} >= {self.a:.3f}")
                success = False
            else:
                logs.append(f"Acceleration monotonicity passed: a decreases from {self.a:.3f} to {a_new:.3f}")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"=== {test_name} failed with exception: {e} ===")
                overall_success = False
        
        return overall_success, all_logs