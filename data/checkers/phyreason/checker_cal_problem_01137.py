import math
import sympy as sp
from typing import Tuple, List, Optional, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize checker by safely extracting inputs and results.
        Uses safe attribute access to avoid sandbox errors.
        """
        self.results = {}
        
        # Extract results safely from solution_instance
        expected_keys = ['x', 'I_F', 'F', 'I_A', 'q', 'v', 'I', 'F_A', 'E']
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, None))
        
        # Extract problem parameters safely
        self.L = getattr(solution_instance, 'L', 0.5)  # rail separation = rod length (m)
        self.R = getattr(solution_instance, 'R', 0.4)  # resistor resistance (Ω)
        self.m = getattr(solution_instance, 'm', 0.5)  # rod mass (kg)
        self.r = getattr(solution_instance, 'r', 0.1)  # rod resistance (Ω)
        self.B = getattr(solution_instance, 'B', 1.0)  # magnetic field strength (T)
        self.a = getattr(solution_instance, 'a', 1.0)  # acceleration (m/s²)
        self.t = getattr(solution_instance, 't', 4.0)  # time from step analysis (s)
        
        # Constants
        self.g = 9.8  # gravity (m/s²), though not used in this problem
        
        # Initialize logs
        self.logs = []
        
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental physics laws:
        1. Newton's 2nd Law: F_net = m*a
        2. EMF formula: E = B*L*v
        3. Ohm's Law: I = E/(R+r)
        4. Ampere Force: F_A = B*I*L
        5. Impulse-Momentum: I_F - I_A = m*v (for t=4s)
        """
        logs = []
        success = True
        
        # Get values from results
        x = self.results.get('x')
        I_F = self.results.get('I_F')
        F = self.results.get('F')
        I_A = self.results.get('I_A')
        q = self.results.get('q')
        v = self.results.get('v')
        I = self.results.get('I')
        F_A = self.results.get('F_A')
        E = self.results.get('E')
        
        # 1. Check kinematics consistency (v = a*t at t=4s)
        expected_v = self.a * self.t  # 1 * 4 = 4 m/s
        if v is not None:
            if not self._is_close(v, expected_v):
                logs.append(f"Velocity invariant failed: v={v}, expected {expected_v}")
                success = False
            else:
                logs.append(f"✓ Velocity correct: v = {v} m/s")
        else:
            logs.append("⚠ Velocity not found in results")
        
        # 2. Check EMF formula: E = B*L*v
        if E is not None and v is not None:
            expected_E = self.B * self.L * v
            if not self._is_close(E, expected_E):
                logs.append(f"EMF invariant failed: E={E}, expected {expected_E}")
                success = False
            else:
                logs.append(f"✓ EMF correct: E = {E} V")
        elif E is None:
            logs.append("⚠ EMF not found in results")
        
        # 3. Check Ohm's Law: I = E/(R+r)
        if I is not None and E is not None:
            expected_I = E / (self.R + self.r)
            if not self._is_close(I, expected_I):
                logs.append(f"Ohm's Law failed: I={I}, expected {expected_I}")
                success = False
            else:
                logs.append(f"✓ Current correct: I = {I} A")
        elif I is None:
            logs.append("⚠ Current not found in results")
        
        # 4. Check Ampere Force: F_A = B*I*L
        if F_A is not None and I is not None:
            expected_F_A = self.B * I * self.L
            if not self._is_close(F_A, expected_F_A):
                logs.append(f"Ampere force failed: F_A={F_A}, expected {expected_F_A}")
                success = False
            else:
                logs.append(f"✓ Ampere force correct: F_A = {F_A} N")
        elif F_A is None:
            logs.append("⚠ Ampere force not found in results")
        
        # 5. Check Newton's 2nd Law: F_net = F - F_A = m*a
        if F is not None and F_A is not None:
            net_force = F - F_A
            expected_net = self.m * self.a
            if not self._is_close(net_force, expected_net):
                logs.append(f"Newton's 2nd Law failed: F-F_A={net_force}, expected {expected_net}")
                success = False
            else:
                logs.append(f"✓ Newton's 2nd Law satisfied: F - F_A = {net_force} N")
        elif F is None:
            logs.append("⚠ External force F not found in results")
        
        # 6. Check displacement: x = 0.5*a*t²
        if x is not None:
            expected_x = 0.5 * self.a * self.t**2
            if not self._is_close(x, expected_x):
                logs.append(f"Displacement failed: x={x}, expected {expected_x}")
                success = False
            else:
                logs.append(f"✓ Displacement correct: x = {x} m")
        
        # 7. Check charge: q = (B*L*x)/(R+r)
        if q is not None and x is not None:
            expected_q = (self.B * self.L * x) / (self.R + self.r)
            if not self._is_close(q, expected_q):
                logs.append(f"Charge formula failed: q={q}, expected {expected_q}")
                success = False
            else:
                logs.append(f"✓ Charge correct: q = {q} C")
        
        # 8. Check Ampere impulse: I_A = B*L*q
        if I_A is not None and q is not None:
            expected_I_A = self.B * self.L * q
            if not self._is_close(I_A, expected_I_A):
                logs.append(f"Ampere impulse failed: I_A={I_A}, expected {expected_I_A}")
                success = False
            else:
                logs.append(f"✓ Ampere impulse correct: I_A = {I_A} N·s")
        
        # 9. Check impulse-momentum theorem: I_F - I_A = m*v
        if I_F is not None and I_A is not None and v is not None:
            net_impulse = I_F - I_A
            expected_mv = self.m * v
            if not self._is_close(net_impulse, expected_mv):
                logs.append(f"Impulse-momentum failed: I_F-I_A={net_impulse}, expected m*v={expected_mv}")
                success = False
            else:
                logs.append(f"✓ Impulse-momentum satisfied: I_F - I_A = {net_impulse} N·s = m*v")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Verify that intermediate results from steps analysis match final results.
        """
        logs = []
        success = True
        
        # Expected values from step-by-step analysis
        step_values = {
            'v': 4.0,      # step 1: v = a*t = 1*4 = 4 m/s
            'E': 2.0,      # step 2: E = B*L*v = 1*0.5*4 = 2 V
            'I': 4.0,      # step 3: I = E/(R+r) = 2/(0.4+0.1) = 4 A
            'F_A': 2.0,    # step 4: F_A = B*I*L = 1*4*0.5 = 2 N
            'F': 2.5,      # step 5: F = m*a + F_A = 0.5*1 + 2 = 2.5 N
            'x': 8.0,      # step 9: x = 0.5*a*t² = 0.5*1*16 = 8 m
            'I_F': 6.0     # step 10: I_F = 6 N·s
        }
        
        for key, expected in step_values.items():
            actual = self.results.get(key)
            if actual is not None:
                if not self._is_close(actual, expected):
                    logs.append(f"Trace inconsistency: {key}={actual}, expected {expected}")
                    success = False
                else:
                    logs.append(f"✓ {key} matches step analysis: {actual}")
            else:
                logs.append(f"⚠ {key} not found in results")
        
        # Verify charge and Ampere impulse consistency
        x = self.results.get('x')
        q = self.results.get('q')
        I_A = self.results.get('I_A')
        
        if x is not None and q is not None:
            expected_q = (self.B * self.L * x) / (self.R + self.r)
            if not self._is_close(q, expected_q):
                logs.append(f"Charge inconsistency: q={q}, expected from x: {expected_q}")
                success = False
        
        if q is not None and I_A is not None:
            expected_I_A = self.B * self.L * q
            if not self._is_close(I_A, expected_I_A):
                logs.append(f"Ampere impulse inconsistency: I_A={I_A}, expected from q: {expected_I_A}")
                success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb mass (m) and check if force and impulse change in expected direction.
        Restores original mass after test.
        """
        logs = []
        
        # Store original values
        m_original = self.m
        F_original = self.results.get('F')
        I_F_original = self.results.get('I_F')
        
        if F_original is None or I_F_original is None:
            logs.append("⚠ Cannot test monotonicity: F or I_F missing")
            return False, logs
        
        # Increase mass by 10%
        self.m = m_original * 1.1
        
        # Recalculate using physics formulas
        # All calculations follow the same step-by-step logic
        v_new = self.a * self.t  # step 1
        E_new = self.B * self.L * v_new  # step 2
        I_new = E_new / (self.R + self.r)  # step 3
        F_A_new = self.B * I_new * self.L  # step 4
        F_new = self.m * self.a + F_A_new  # step 5: F = m*a + F_A
        
        x_new = 0.5 * self.a * self.t**2  # step 9
        q_new = (self.B * self.L * x_new) / (self.R + self.r)  # step 8
        I_A_new = self.B * self.L * q_new  # step 7
        
        # From impulse-momentum: I_F = I_A + m*v
        I_F_new = I_A_new + self.m * v_new  # step 10
        
        # Check monotonic trends
        trend_checks = []
        
        # 1. F should increase with mass (since F = m*a + constant F_A)
        if F_new > F_original:
            trend_checks.append(True)
            logs.append(f"✓ F increases with mass: {F_original:.3f} → {F_new:.3f}")
        else:
            trend_checks.append(False)
            logs.append(f"✗ F monotonicity failed: {F_original:.3f} → {F_new:.3f}")
        
        # 2. I_F should increase with mass (since I_F = constant I_A + m*v)
        if I_F_new > I_F_original:
            trend_checks.append(True)
            logs.append(f"✓ I_F increases with mass: {I_F_original:.3f} → {I_F_new:.3f}")
        else:
            trend_checks.append(False)
            logs.append(f"✗ I_F monotonicity failed: {I_F_original:.3f} → {I_F_new:.3f}")
        
        # Restore original mass
        self.m = m_original
        
        success = all(trend_checks)
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success status with logs.
        """
        all_logs = []
        test_results = []
        
        # Run invariant tests
        success, logs = self.test_invariants()
        test_results.append(success)
        all_logs.append("=== Physics Invariants Test ===")
        all_logs.extend(logs)
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        test_results.append(success)
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs)
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        test_results.append(success)
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs)
        
        # Overall success
        overall_success = all(test_results)
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs

# For demonstration purposes, here's a mock solution class
class MockSolution:
    def __init__(self):
        self.results = {
            'x': 8.0,      # displacement (m)
            'I_F': 6.0,    # impulse of pulling force (N·s)
            'F': 2.5,      # external force (N)
            'I_A': 2.0,    # impulse of Ampere force (N·s)
            'q': 8.0,      # charge (C)
            'v': 4.0,      # velocity (m/s)
            'I': 4.0,      # current (A)
            'F_A': 2.0,    # Ampere force (N)
            'E': 2.0       # EMF (V)
        }
        self.L = 0.5
        self.R = 0.4
        self.m = 0.5
        self.r = 0.1
        self.B = 1.0
        self.a = 1.0
        self.t = 4.0

if __name__ == "__main__":
    # Example usage
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)
    print(f"\nOverall success: {success}")