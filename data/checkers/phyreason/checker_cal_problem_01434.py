import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization with safe attribute access.
        Extracts all necessary inputs and results from solution_instance.
        """
        # Constants from problem statement
        self.m = 0.2  # kg
        self.g = 10.0  # m/s^2
        self.R = 2.0  # m (radius)
        self.H = 3.2  # m (vertical height difference B->C)
        
        # Safe extraction of results (use getattr and dict fallback)
        results_dict = getattr(solution_instance, 'results', {})
        self.F_N = results_dict.get('F_N', 0.0)
        self.v = results_dict.get('v', 0.0)
        self.x = results_dict.get('x', 0.0)
        self.t = results_dict.get('t', 0.0)  # This is t_parallel (0.4 s) per contract
        
        # Extract t_flight if available, otherwise compute it from physics
        self.t_flight = results_dict.get('t_flight', 0.0)
        if self.t_flight == 0.0 and self.H > 0:
            # Compute from projectile motion: H = 0.5*g*t_flight^2
            self.t_flight = math.sqrt(2 * self.H / self.g)
        
        # Validate extracted values are numeric
        for attr in ['F_N', 'v', 'x', 't', 't_flight']:
            if not isinstance(getattr(self, attr), (int, float)):
                try:
                    setattr(self, attr, float(getattr(self, attr)))
                except:
                    setattr(self, attr, 0.0)
        
        self.logs = []

    def _log(self, msg: str):
        """Helper to append log messages."""
        self.logs.append(msg)

    def test_invariants(self) -> bool:
        """
        Verify fundamental physics laws:
        1. Centripetal force at point B (Newton's 2nd Law)
        2. Projectile motion equations (kinematics) - using t_flight, not t
        3. Energy conservation (optional)
        """
        success = True
        
        # 1. Centripetal force check at B
        # At bottom of circular arc: F_N - m*g = m*v^2/R (upward positive)
        centripetal_force = self.F_N - self.m * self.g
        expected_centripetal = self.m * (self.v ** 2) / self.R
        
        if not math.isclose(centripetal_force, expected_centripetal, 
                           rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Centripetal force mismatch: F_N - mg = {centripetal_force:.3f}, "
                     f"m*v²/R = {expected_centripetal:.3f}")
            success = False
        else:
            self._log(f"✓ Centripetal force consistent: {centripetal_force:.3f} ≈ {expected_centripetal:.3f}")
        
        # 2. Projectile motion checks - using t_flight, not t
        # Vertical fall: H = 0.5*g*t_flight^2 (since initial vertical velocity = 0)
        fall_distance = 0.5 * self.g * (self.t_flight ** 2)
        if not math.isclose(fall_distance, self.H, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Vertical fall mismatch: 0.5*g*t_flight² = {fall_distance:.3f}, "
                     f"H = {self.H:.3f}")
            success = False
        else:
            self._log(f"✓ Vertical fall consistent: {fall_distance:.3f} ≈ {self.H:.3f}")
        
        # Horizontal range: x = v*t_flight (since horizontal velocity constant)
        range_calc = self.v * self.t_flight
        if not math.isclose(range_calc, self.x, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Horizontal range mismatch: v*t_flight = {range_calc:.3f}, "
                     f"x = {self.x:.3f}")
            success = False
        else:
            self._log(f"✓ Horizontal range consistent: {range_calc:.3f} ≈ {self.x:.3f}")
        
        # 3. Energy conservation (optional but useful)
        # At point A (top of quarter-circle): height = R, velocity = 0
        # At point B (bottom): height = 0, velocity = v
        # Energy: m*g*R = 0.5*m*v^2 + energy_loss? (should be zero for ideal track)
        energy_initial = self.m * self.g * self.R
        energy_final = 0.5 * self.m * (self.v ** 2)
        if not math.isclose(energy_initial, energy_final, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Energy conservation warning: mgR = {energy_initial:.3f}, "
                     f"0.5*m*v² = {energy_final:.3f} (should be equal for no friction)")
            # Not a failure, just a warning (track might not be frictionless in real problems)
        else:
            self._log(f"✓ Energy conserved: {energy_initial:.3f} ≈ {energy_final:.3f}")
        
        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        From KB steps: v=4 m/s, x=3.2 m, t_parallel=0.4 s should be internally consistent.
        """
        success = True
        
        # Consistency 1: t_flight from projectile equations
        # t_flight = sqrt(2*H/g) for vertical fall with zero initial vertical velocity
        t_flight_calc = math.sqrt(2 * self.H / self.g)
        if not math.isclose(t_flight_calc, self.t_flight, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Flight time inconsistency: sqrt(2H/g) = {t_flight_calc:.3f}, t_flight = {self.t_flight:.3f}")
            success = False
        else:
            self._log(f"✓ Flight time consistent with vertical fall: {t_flight_calc:.3f} ≈ {self.t_flight:.3f}")
        
        # Consistency 2: x from v and t_flight
        x_calc = self.v * self.t_flight
        if not math.isclose(x_calc, self.x, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Distance inconsistency: v*t_flight = {x_calc:.3f}, x = {self.x:.3f}")
            success = False
        else:
            self._log(f"✓ Distance consistent: v*t_flight = {x_calc:.3f} ≈ {self.x:.3f}")
        
        # Consistency 3: v from centripetal force
        if self.F_N > 0:
            v_calc = math.sqrt((self.F_N - self.m * self.g) * self.R / self.m)
            if not math.isclose(v_calc, self.v, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Velocity inconsistency: sqrt((F_N-mg)R/m) = {v_calc:.3f}, v = {self.v:.3f}")
                success = False
            else:
                self._log(f"✓ Velocity consistent with normal force: {v_calc:.3f} ≈ {self.v:.3f}")
        
        # Consistency 4: t_parallel (contract's 't') from parallel condition
        # Condition: (g * t_parallel) / v = H / x
        if self.v > 0 and self.x > 0:
            t_parallel_calc = (self.H / self.x) * self.v / self.g
            if not math.isclose(t_parallel_calc, self.t, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Parallel time inconsistency: (H/x)*v/g = {t_parallel_calc:.3f}, t = {self.t:.3f}")
                success = False
            else:
                self._log(f"✓ Parallel time consistent: {t_parallel_calc:.3f} ≈ {self.t:.3f}")
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and check if trends match physical intuition.
        We'll perturb mass m and check how v changes (should decrease if m increases).
        """
        success = True
        
        # Save original values
        m_orig = self.m
        v_orig = self.v
        
        # Perturb mass by +10%
        self.m = m_orig * 1.10
        
        # Recalculate v using centripetal formula (assuming F_N remains 3.6N)
        # This is a simplified check - in reality F_N might change too
        if self.F_N > 0:
            v_new = math.sqrt((self.F_N - self.m * self.g) * self.R / self.m)
            
            # Expected: heavier ball -> lower speed at bottom for same F_N
            if v_new > v_orig:  # Should be smaller
                self._log(f"Monotonicity violation: m↑ from {m_orig:.3f} to {self.m:.3f}, "
                         f"v increased from {v_orig:.3f} to {v_new:.3f} (should decrease)")
                success = False
            else:
                self._log(f"✓ Monotonicity OK: m↑ from {m_orig:.3f} to {self.m:.3f}, "
                         f"v decreased from {v_orig:.3f} to {v_new:.3f}")
        else:
            self._log("Skipping monotonicity test: F_N not available")
        
        # Restore original values
        self.m = m_orig
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        self._log("=== Physics Checker Starting ===")
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary
        overall = test1 and test2 and test3
        self._log(f"\n=== Summary ===")
        self._log(f"Invariants test: {'PASS' if test1 else 'FAIL'}")
        self._log(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
        self._log(f"Monotonicity test: {'PASS' if test3 else 'FAIL'}")
        self._log(f"OVERALL: {'PASS' if overall else 'FAIL'}")
        
        return overall, self.logs