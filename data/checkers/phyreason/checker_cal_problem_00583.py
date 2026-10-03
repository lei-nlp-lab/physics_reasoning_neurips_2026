import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 0.0))
        self.x_boat = getattr(solution_instance, 'x_boat', solution_instance.results.get('x_boat', 0.0))
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', 0.0))
        self.L_relative = getattr(solution_instance, 'L_relative', solution_instance.results.get('L_relative', 0.0))
        
        # Store original for monotonicity test
        self._original_L = float(self.L)
        self._original_x = float(self.x)
        
        # Physics constants from problem
        self.ship_displacement_total = 1000.0  # meters
        self.tol = 1e-2  # 1% tolerance for fuzzy matching

    def test_invariants(self) -> List[str]:
        logs = []
        
        # 1. Check total relative distance L_relative = 2*L (from steps)
        if not math.isclose(self.L_relative, 2 * self.L, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Invariant failed: L_relative ({self.L_relative}) ≠ 2*L ({2*self.L})")
        
        # 2. Check boat displacement symmetry (equal time for both legs)
        # From steps: x_boat = total_ship_displacement / 2
        expected_x_boat = self.ship_displacement_total / 2
        if not math.isclose(self.x_boat, expected_x_boat, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Invariant failed: x_boat ({self.x_boat}) ≠ total_displacement/2 ({expected_x_boat})")
        
        # 3. Check relative motion formula: x = x_boat - L (from step 4)
        expected_x = self.x_boat - self.L
        if not math.isclose(self.x, expected_x, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Invariant failed: x ({self.x}) ≠ x_boat - L ({expected_x})")
        
        # 4. Symbolic check: Conservation of time
        # Let v = tourist speed relative to ship, u = ship speed
        # Time for first leg: t1 = L/v
        # Time for second leg: t2 = L/v (constant relative speed)
        # Total time: T = 2L/v
        # Ship displacement: u*T = 1000 => u/v = 1000/(2L)
        if self.L > 0:
            uv_ratio = self.ship_displacement_total / (2 * self.L)  # u/v from total displacement
            # Also from step 4: x = (u - v)*t2 = (u/v - 1)*L
            uv_from_x = self.x / self.L + 1 if self.L > 0 else 0
            if not math.isclose(uv_ratio, uv_from_x, rel_tol=self.tol, abs_tol=self.tol):
                logs.append(f"Symbolic invariant failed: u/v from total displacement ({uv_ratio}) ≠ u/v from x ({uv_from_x})")
        
        return logs

    def test_trace_consistency(self) -> List[str]:
        logs = []
        
        # Check against target answers from problem
        target_L_relative = 200.0
        target_x = 400.0
        
        if not math.isclose(self.L_relative, target_L_relative, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Trace mismatch: L_relative ({self.L_relative}) ≠ target ({target_L_relative})")
        
        if not math.isclose(self.x, target_x, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Trace mismatch: x ({self.x}) ≠ target ({target_x})")
        
        # Cross-check intermediate steps
        # Step 1: L_relative = 2*L
        if not math.isclose(self.L_relative, 2 * self.L, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Step 1 inconsistency: L_relative ≠ 2*L")
        
        # Step 2: x_boat = 500
        if not math.isclose(self.x_boat, 500.0, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Step 2 inconsistency: x_boat ({self.x_boat}) ≠ 500")
        
        # Step 4: x = x_boat - L
        if not math.isclose(self.x, self.x_boat - self.L, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Step 4 inconsistency: x ≠ x_boat - L")
        
        return logs

    def test_monotonicity(self) -> List[str]:
        logs = []
        
        # Perturb L by +10% and check trend in x
        perturbed_L = self._original_L * 1.10
        
        # Using formulas from steps to recalculate
        # x_boat remains half of total ship displacement (symmetry)
        perturbed_x_boat = self.ship_displacement_total / 2
        # x = x_boat - L (from step 4)
        perturbed_x = perturbed_x_boat - perturbed_L
        
        # Check monotonic relationship: as L increases, x decreases (negative correlation)
        expected_trend = self._original_x - (perturbed_L - self._original_L)
        
        if not math.isclose(perturbed_x, expected_trend, rel_tol=self.tol, abs_tol=self.tol):
            logs.append(f"Monotonicity failed: Expected x={expected_trend} for L={perturbed_L}, got {perturbed_x}")
        
        # Verify the trend sign: dx/dL = -1
        delta_x = perturbed_x - self._original_x
        delta_L = perturbed_L - self._original_L
        if delta_L != 0:
            derivative = delta_x / delta_L
            if not math.isclose(derivative, -1.0, rel_tol=self.tol, abs_tol=self.tol):
                logs.append(f"Monotonicity derivative failed: dx/dL = {derivative}, expected -1")
        
        # Restore original values (just in case)
        self.L = self._original_L
        self.x = self._original_x
        
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        
        all_logs.extend(self.test_invariants())
        all_logs.extend(self.test_trace_consistency())
        all_logs.extend(self.test_monotonicity())
        
        success = len(all_logs) == 0
        return success, all_logs