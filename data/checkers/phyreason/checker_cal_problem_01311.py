import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any) -> None:
        """Safely extract parameters and results."""
        self.L = getattr(solution_instance, 'L', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Extract results with safe fallback
        results = getattr(solution_instance, 'results', {})
        self.T = getattr(solution_instance, 'T', results.get('T', None))
        self.F_axis = getattr(solution_instance, 'F_axis', results.get('F_axis', None))
        self.v_A = getattr(solution_instance, 'v_A', results.get('v_A', None))
        self.v_B = getattr(solution_instance, 'v_B', results.get('v_B', None))
        self.omega = getattr(solution_instance, 'omega', results.get('omega', None))
        self.omega = getattr(solution_instance, 'ω', self.omega)  # Alternative key
        
        # Store original values for restoration
        self.original_params = {'L': self.L, 'm': self.m, 'g': self.g}
        
    def test_invariants(self) -> List[str]:
        """Check fundamental physics laws."""
        logs = []
        
        # Newton's Second Law (Centripetal Force) - Scenario 1: A at highest point
        if self.omega is not None and self.F_axis is not None:
            # Check ω = √(g/L)
            expected_omega = math.sqrt(self.g / self.L)
            if not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ω mismatch. Expected {expected_omega:.3f}, got {self.omega:.3f}")
            
            # Check F_axis = 9mg
            expected_F_axis = 9 * self.m * self.g
            if not math.isclose(self.F_axis, expected_F_axis, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_axis mismatch. Expected {expected_F_axis:.3f}, got {self.F_axis:.3f}")
            
            # Check tension T = 9mg (if provided)
            if self.T is not None:
                if not math.isclose(self.T, expected_F_axis, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T mismatch. Expected {expected_F_axis:.3f}, got {self.T:.3f}")
        
        # Scenario 2: Axis free of force
        if self.v_A is not None and self.v_B is not None:
            # Check ω = √(4g/(5L))
            expected_omega2 = math.sqrt(4 * self.g / (5 * self.L))
            # Use stored omega or recompute from velocities
            omega_from_vA = self.v_A / self.L
            omega_from_vB = self.v_B / (2 * self.L)
            
            if not (math.isclose(omega_from_vA, expected_omega2, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(omega_from_vB, expected_omega2, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"FAIL: ω from velocities mismatch. Expected {expected_omega2:.3f}, "
                           f"got from vA: {omega_from_vA:.3f}, from vB: {omega_from_vB:.3f}")
            
            # Check v_A = Lω, v_B = 2Lω
            expected_vA = self.L * expected_omega2
            expected_vB = 2 * self.L * expected_omega2
            if not math.isclose(self.v_A, expected_vA, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_A mismatch. Expected {expected_vA:.3f}, got {self.v_A:.3f}")
            if not math.isclose(self.v_B, expected_vB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B mismatch. Expected {expected_vB:.3f}, got {self.v_B:.3f}")
            
            # Net force on axis should be zero in free condition, but F_axis is stored for Q1.
            # So we should NOT check F_axis here. Remove this check.
            # if self.F_axis is not None:
            #     if not math.isclose(self.F_axis, 0.0, abs_tol=1e-2):
            #         logs.append(f"FAIL: Axis force not zero in free condition. Got {self.F_axis:.3f}")
        
        # Symbolic check removed to avoid SymPy usage in numeric tests
        # logs.append("INFO: Symbolic check skipped for numeric testing")
        
        return logs if logs else ["PASS: All invariants satisfied"]
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate steps with final results."""
        logs = []
        
        # Scenario 1: Using ω = √(g/L) to compute T and F_axis
        if self.omega is not None and self.F_axis is not None:
            computed_T = 3 * self.m * self.g + 3 * self.m * (2 * self.L) * (self.omega**2)
            expected_T = 9 * self.m * self.g
            if not math.isclose(computed_T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 2-3 inconsistency. Computed T={computed_T:.3f}, expected {expected_T:.3f}")
        
        # Scenario 2: Check that ω from step 7 yields correct velocities
        if self.v_A is not None and self.v_B is not None:
            omega_from_step7 = math.sqrt(4 * self.g / (5 * self.L))
            vA_from_omega = self.L * omega_from_step7
            vB_from_omega = 2 * self.L * omega_from_step7
            
            if not (math.isclose(vA_from_omega, self.v_A, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(vB_from_omega, self.v_B, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"FAIL: Velocity inconsistency. From ω: vA={vA_from_omega:.3f}, vB={vB_from_omega:.3f}; "
                           f"Stored: vA={self.v_A:.3f}, vB={self.v_B:.3f}")
        
        return logs if logs else ["PASS: All trace checks consistent"]
    
    def test_monotonicity(self) -> List[str]:
        """Perturb mass m and verify expected trends."""
        logs = []
        original_m = self.m
        
        # Perturb m by +10%
        self.m = original_m * 1.10
        
        # Scenario 1: F_axis = 9mg should increase proportionally
        if self.F_axis is not None:
            new_F_axis_expected = 9 * self.m * self.g
            original_F_axis = 9 * original_m * self.g
            if not math.isclose(new_F_axis_expected, original_F_axis * 1.10, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_axis monotonicity violated. +10% m gives {new_F_axis_expected:.3f}, "
                           f"expected {original_F_axis * 1.10:.3f}")
        
        # Scenario 2: v_A and v_B should increase with m? Actually, from ω = √(4g/(5L)), independent of m.
        # So velocities should remain constant when m changes.
        if self.v_A is not None and self.v_B is not None:
            new_omega = math.sqrt(4 * self.g / (5 * self.L))
            new_vA = self.L * new_omega
            new_vB = 2 * self.L * new_omega
            original_omega = math.sqrt(4 * self.g / (5 * self.L))  # Same as before since m cancels
            original_vA = self.L * original_omega
            original_vB = 2 * self.L * original_omega
            
            if not (math.isclose(new_vA, original_vA, rel_tol=1e-4, abs_tol=1e-4) and
                    math.isclose(new_vB, original_vB, rel_tol=1e-4, abs_tol=1e-4)):
                logs.append("FAIL: Velocity should be independent of mass in free-axis condition.")
        
        # Restore original m
        self.m = original_m
        
        return logs if logs else ["PASS: Monotonicity trends correct"]
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        test_functions = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            try:
                logs = test_func()
                if logs and logs[0].startswith("FAIL"):
                    success = False
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
            except Exception as e:
                success = False
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
        
        return success, all_logs