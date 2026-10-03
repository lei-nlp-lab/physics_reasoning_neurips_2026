import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Extract inputs and results safely without direct attribute access."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with fallbacks
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.E = getattr(solution_instance, 'E', 1.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        
        # Extract results with fallbacks
        self.v_y = self.results.get('v_y', 0.0)
        self.twoR = self.results.get('2R', 0.0)
        self.theta = self.results.get('theta', 0.0)
        self.v_0 = self.results.get('v_0', 0.0)
        self.v = self.results.get('v', 0.0)
        self.R = self.results.get('R', 0.0)
        self.B = self.results.get('B', 0.0)
        self.twoL = self.results.get('2L', 2.0 * self.L)
        
        # Extract boundary values for collision conditions
        self.B_avoid = self.results.get('B_avoid', 0.0)
        self.B_single_low = self.results.get('B_single_low', 0.0)  # Should be lower bound for single collision
        self.B_single_high = self.results.get('B_single_high', 0.0)  # Should be upper bound for single collision
        
        # Constants - extract n from B formula
        self.n = self._extract_n_from_B()
        
    def _extract_n_from_B(self) -> int:
        """Try to extract integer n from B expression."""
        try:
            # Expected form: B = 2n√(2mE/qL)
            theoretical_B = 2 * math.sqrt(2 * self.m * self.E / (self.q * self.L))
            if math.isclose(theoretical_B, 0.0, abs_tol=1e-12):
                return 1
            if math.isclose(self.B, 0.0, abs_tol=1e-12):
                return 1
            n_approx = self.B / theoretical_B
            # Round to nearest integer
            return int(round(n_approx))
        except:
            return 1
    
    def test_invariants(self) -> List[str]:
        """Verify conservation laws and physics principles."""
        logs = []
        tol = {'rel_tol': 1e-2, 'abs_tol': 1e-2}
        
        # 1. Check kinematic relationship from electric field region
        expected_v0_sq = 2 * self.q * self.E * self.L / self.m
        actual_v0_sq = self.v_0**2
        
        if not math.isclose(actual_v0_sq, expected_v0_sq, **tol):
            logs.append(f"Kinematics invariant violated: v_0² expected {expected_v0_sq:.4f}, got {actual_v0_sq:.4f}")
        
        # 2. Check vertical velocity consistency
        if not math.isclose(self.v_y, self.v_0, **tol):
            logs.append(f"Vertical velocity inconsistent: v_y={self.v_y:.4f}, v_0={self.v_0:.4f}")
        
        # 3. Check speed composition
        expected_v = math.sqrt(2) * self.v_0
        if not math.isclose(self.v, expected_v, **tol):
            logs.append(f"Speed composition violated: v expected {expected_v:.4f}, got {self.v:.4f}")
        
        # 4. Check angle θ = 45°
        expected_theta_deg = 45.0
        theta_deg = math.degrees(self.theta) if abs(self.theta) > 0.1 else self.theta
        if not math.isclose(theta_deg, expected_theta_deg, **tol):
            logs.append(f"Angle violation: θ expected {expected_theta_deg}°, got {theta_deg:.2f}°")
        
        # 5. Check Lorentz force for circular motion
        if self.B > 0 and self.R > 0:
            expected_R = self.m * self.v / (self.q * self.B)
            if not math.isclose(self.R, expected_R, **tol):
                logs.append(f"Lorentz force violation: R expected {expected_R:.4f}, got {self.R:.4f}")
        
        # 6. Check geometry relationship for n=1 case
        if self.twoR > 0 and self.n > 0:
            expected_2R = math.sqrt(2) * self.L / self.n
            if not math.isclose(self.twoR, expected_2R, **tol):
                logs.append(f"Geometry violation: 2R expected {expected_2R:.4f}, got {self.twoR:.4f}")
        
        # 7. Check magnetic field formula for reaching point c
        if self.B > 0:
            expected_B = 2 * self.n * math.sqrt(2 * self.m * self.E / (self.q * self.L))
            if not math.isclose(self.B, expected_B, **tol):
                logs.append(f"Magnetic field formula violation: B expected {expected_B:.4f}, got {self.B:.4f}")
        
        if not logs:
            logs.append("All invariants satisfied ✓")
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answers."""
        logs = []
        tol = {'rel_tol': 1e-2, 'abs_tol': 1e-2}
        
        # Check that 2L is consistent with L
        expected_2L = 2 * self.L
        if not math.isclose(self.twoL, expected_2L, **tol):
            logs.append(f"Length inconsistency: 2L should be {expected_2L:.4f}, got {self.twoL:.4f}")
        
        # Check that R and 2R are consistent
        if self.R > 0 and self.twoR > 0:
            if not math.isclose(self.twoR, 2 * self.R, **tol):
                logs.append(f"Radius inconsistency: 2R should be {2*self.R:.4f}, got {self.twoR:.4f}")
        
        # Verify the three target answers
        # Target 1: v_0 = √(2qEL/m)
        target1 = math.sqrt(2 * self.q * self.E * self.L / self.m)
        if not math.isclose(self.v_0, target1, **tol):
            logs.append(f"Target 1 mismatch: v_0={self.v_0:.4f}, expected √(2qEL/m)={target1:.4f}")
        
        # Target 2: B = 2n√(2mE/qL) for n=1,2,3...
        target2 = 2 * self.n * math.sqrt(2 * self.m * self.E / (self.q * self.L))
        if not math.isclose(self.B, target2, **tol) and self.B > 0:
            logs.append(f"Target 2 mismatch: B={self.B:.4f}, expected 2n√(2mE/qL)={target2:.4f}")
        
        # Target 3: Check the boundary values for collision conditions
        # Compute expected boundaries
        expected_B_avoid = (4/3) * math.sqrt(2 * self.m * self.E / (self.q * self.L))
        expected_B_single_low = 2 * math.sqrt(2 * self.m * self.E / (self.q * self.L))  # Lower bound for single collision
        expected_B_single_high = (8/3) * math.sqrt(2 * self.m * self.E / (self.q * self.L))  # Upper bound for single collision
        
        # Check B_avoid boundary (for avoiding collision)
        if self.B_avoid > 0 and not math.isclose(self.B_avoid, expected_B_avoid, **tol):
            logs.append(f"B_avoid boundary mismatch: {self.B_avoid:.4f} vs expected {expected_B_avoid:.4f}")
        
        # Check B_single_low boundary (lower bound for single collision)
        if self.B_single_low > 0 and not math.isclose(self.B_single_low, expected_B_single_low, **tol):
            logs.append(f"B_single_low boundary mismatch: {self.B_single_low:.4f} vs expected {expected_B_single_low:.4f}")
        
        # Check B_single_high boundary (upper bound for single collision)
        if self.B_single_high > 0 and not math.isclose(self.B_single_high, expected_B_single_high, **tol):
            logs.append(f"B_single_high boundary mismatch: {self.B_single_high:.4f} vs expected {expected_B_single_high:.4f}")
        
        # The stored B value (for reaching point c with n=1) should be equal to expected_B_single_low
        if not math.isclose(self.B, expected_B_single_low, **tol):
            logs.append(f"B (for reaching c) should equal lower bound of single collision range: B={self.B:.4f}, expected {expected_B_single_low:.4f}")
        
        if not logs:
            logs.append("Trace consistency satisfied ✓")
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Perturb mass and check expected trends."""
        logs = []
        
        # Store original values
        orig_m = self.m
        orig_v0 = self.v_0
        orig_v = self.v
        orig_B = self.B
        
        # Perturb mass (increase by 10%)
        perturb_factor = 1.1
        self.m *= perturb_factor
        
        # Recalculate expected values using formulas
        new_v0 = math.sqrt(2 * self.q * self.E * self.L / self.m)
        new_v = math.sqrt(2) * new_v0
        new_B = 2 * self.n * math.sqrt(2 * self.m * self.E / (self.q * self.L))
        
        # Check trends
        # v_0 ∝ 1/√m
        if not (new_v0 < orig_v0):
            logs.append(f"Monotonicity violation: v_0 should decrease when m increases")
        
        # v ∝ 1/√m
        if not (new_v < orig_v):
            logs.append(f"Monotonicity violation: v should decrease when m increases")
        
        # B ∝ √m
        if orig_B > 0 and not (new_B > orig_B):
            logs.append(f"Monotonicity violation: B should increase when m increases")
        
        # Restore original mass
        self.m = orig_m
        
        if not logs:
            logs.append("Monotonicity tests passed ✓")
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success flag with logs."""
        all_logs = []
        
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = not any("violation" in log or "mismatch" in log or "inconsistency" in log 
                         for log in all_logs if "✓" not in log)
        
        return success, all_logs