import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access using getattr.
        """
        # Safely extract inputs and results from solution instance
        self.solution = solution_instance
        
        # Extract results with safe fallback to defaults
        self.S = getattr(solution_instance, 'S', solution_instance.results.get('S', None))
        self.phi1 = getattr(solution_instance, 'phi1', solution_instance.results.get('Φ_1', None))
        self.phi2 = getattr(solution_instance, 'phi2', solution_instance.results.get('Φ_2', None))
        self.delta_phi = getattr(solution_instance, 'delta_phi', solution_instance.results.get('delta_Φ', None))
        
        # Constants from problem statement
        self.L = 0.1  # m (10 cm)
        self.B = 0.2  # T
        self.N = 10   # number of turns
        self.theta1 = 0  # degrees, initial perpendicular
        self.theta2 = 60  # degrees, rotated 60 degrees
        self.theta3 = 180  # degrees, rotated 180 degrees
        
        # Store logs
        self.logs = []
    
    def log(self, message: str):
        """Helper to append log messages."""
        self.logs.append(message)
        print(f"[PhysicsChecker] {message}")
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants: 
        1. Area formula (S = L^2)
        2. Magnetic flux formulas (Φ = B * S * cosθ for single loop)
        3. Change in flux (ΔΦ = Φ2 - Φ1)
        Uses numerical tolerance for comparisons.
        """
        success = True
        
        # 1. Area invariant: S = L^2
        expected_S = self.L ** 2
        if self.S is not None:
            if not math.isclose(self.S, expected_S, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Area invariant failed: S={self.S}, expected L^2={expected_S}")
                success = False
            else:
                self.log(f"Area invariant passed: S={self.S}")
        else:
            self.log("Area S not found in results")
            success = False
        
        # 2. Magnetic flux invariants
        # For single loop: Φ = B * S * cosθ
        # For coil with N turns: Φ = N * B * S * cosθ
        # From steps, they're computing per loop (single loop)
        
        # Initial flux (θ=0°, cos0=1)
        expected_phi1_single = self.B * self.L**2  # single loop
        expected_phi1_coil = self.N * self.B * self.L**2  # full coil
        
        if self.phi1 is not None:
            # Check against single-loop calculation (as per steps)
            if math.isclose(self.phi1, expected_phi1_single, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Φ1 invariant passed (single loop): {self.phi1}")
            elif math.isclose(self.phi1, expected_phi1_coil, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Φ1 invariant passed (full coil): {self.phi1}")
            else:
                self.log(f"Φ1 invariant failed: {self.phi1}, expected single={expected_phi1_single}, coil={expected_phi1_coil}")
                success = False
        
        # Flux after 60° rotation (cos60=0.5)
        expected_phi2_single = self.B * self.L**2 * math.cos(math.radians(self.theta2))
        expected_phi2_coil = self.N * self.B * self.L**2 * math.cos(math.radians(self.theta2))
        
        if self.phi2 is not None:
            if math.isclose(self.phi2, expected_phi2_single, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Φ2 invariant passed (single loop): {self.phi2}")
            elif math.isclose(self.phi2, expected_phi2_coil, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Φ2 invariant passed (full coil): {self.phi2}")
            else:
                self.log(f"Φ2 invariant failed: {self.phi2}, expected single={expected_phi2_single}, coil={expected_phi2_coil}")
                success = False
        
        # 3. Change in flux for 180° rotation
        # From perpendicular (θ=0°) to opposite (θ=180°): ΔΦ = B*S*(cos180 - cos0) = B*S*(-1 - 1) = -2*B*S
        # Magnitude should be 2*B*S
        expected_delta_single = 2 * self.B * self.L**2
        expected_delta_coil = self.N * 2 * self.B * self.L**2
        
        if self.delta_phi is not None:
            delta_magnitude = abs(self.delta_phi)
            # Accept either signed (-2*B*S) or magnitude (2*B*S)
            if (math.isclose(delta_magnitude, expected_delta_single, rel_tol=1e-2, abs_tol=1e-4) or
                math.isclose(self.delta_phi, -expected_delta_single, rel_tol=1e-2, abs_tol=1e-4)):
                self.log(f"ΔΦ invariant passed (single loop): {self.delta_phi}")
            elif (math.isclose(delta_magnitude, expected_delta_coil, rel_tol=1e-2, abs_tol=1e-4) or
                  math.isclose(self.delta_phi, -expected_delta_coil, rel_tol=1e-2, abs_tol=1e-4)):
                self.log(f"ΔΦ invariant passed (full coil): {self.delta_phi}")
            else:
                self.log(f"ΔΦ invariant failed: {self.delta_phi}, expected single magnitude={expected_delta_single}, coil magnitude={expected_delta_coil}")
                success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with the final answer.
        Verify the step-by-step logic from the steps analysis.
        """
        success = True
        
        # Step 1: Area calculation
        if self.S is not None:
            expected_S = self.L ** 2
            if not math.isclose(self.S, expected_S, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Step 1 (Area) inconsistent: S={self.S}, expected {expected_S}")
                success = False
        
        # Step 2: Initial flux (θ=0°)
        if self.phi1 is not None and self.S is not None:
            # Check if phi1 = B * S (for single loop assumption)
            expected_phi1 = self.B * self.S
            if not math.isclose(self.phi1, expected_phi1, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Step 2 (Φ1) inconsistent: Φ1={self.phi1}, expected B*S={expected_phi1}")
                success = False
        
        # Step 3: Flux after 60° rotation
        if self.phi2 is not None and self.S is not None:
            expected_phi2 = self.B * self.S * math.cos(math.radians(60))
            if not math.isclose(self.phi2, expected_phi2, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"Step 3 (Φ2) inconsistent: Φ2={self.phi2}, expected B*S*cos60={expected_phi2}")
                success = False
        
        # Step 4: Change in flux for 180° rotation
        if self.delta_phi is not None and self.S is not None:
            # Accept either signed change (-2*B*S) or magnitude (2*B*S)
            expected_delta_signed = -2 * self.B * self.S
            expected_delta_magnitude = 2 * self.B * self.S
            if (not math.isclose(self.delta_phi, expected_delta_signed, rel_tol=1e-2, abs_tol=1e-4) and
                not math.isclose(abs(self.delta_phi), expected_delta_magnitude, rel_tol=1e-2, abs_tol=1e-4)):
                self.log(f"Step 4 (ΔΦ) inconsistent: ΔΦ={self.delta_phi}, expected signed={expected_delta_signed} or magnitude={expected_delta_magnitude}")
                success = False
        
        if success:
            self.log("Trace consistency passed")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and check expected trends:
        1. Increase B -> flux should increase proportionally
        2. Increase area -> flux should increase proportionally
        3. Increase angle -> flux should decrease (cosθ decreases)
        Restores original values after perturbation.
        """
        success = True
        
        # Store original values
        original_B = self.B
        original_L = self.L
        original_S = self.L**2
        
        # Use single-loop values for comparison (as per solution)
        if self.phi1 is not None:
            phi1_ref = self.phi1
        else:
            phi1_ref = original_B * original_S
        
        if self.phi2 is not None:
            phi2_ref = self.phi2
        else:
            phi2_ref = original_B * original_S * math.cos(math.radians(60))
        
        if self.delta_phi is not None:
            delta_ref = abs(self.delta_phi)
        else:
            delta_ref = 2 * original_B * original_S
        
        # Test 1: Increase B by 10%
        self.B = original_B * 1.1
        phi1_perturbed = self.B * original_S  # θ=0°
        phi2_perturbed = self.B * original_S * math.cos(math.radians(60))  # θ=60°
        delta_perturbed = 2 * self.B * original_S  # magnitude for 180° rotation
        
        # Check monotonic increase
        if phi1_perturbed <= phi1_ref:
            self.log(f"Monotonicity test 1 failed: B increased but Φ1 didn't increase ({phi1_perturbed} <= {phi1_ref})")
            success = False
        
        if phi2_perturbed <= phi2_ref:
            self.log(f"Monotonicity test 1 failed: B increased but Φ2 didn't increase ({phi2_perturbed} <= {phi2_ref})")
            success = False
        
        if delta_perturbed <= delta_ref:
            self.log(f"Monotonicity test 1 failed: B increased but |ΔΦ| didn't increase ({delta_perturbed} <= {delta_ref})")
            success = False
        
        # Test 2: Increase area by 10% (via L)
        self.B = original_B  # restore B
        self.L = original_L * 1.1
        S_perturbed = self.L**2
        phi1_perturbed = self.B * S_perturbed
        phi2_perturbed = self.B * S_perturbed * math.cos(math.radians(60))
        delta_perturbed = 2 * self.B * S_perturbed
        
        if phi1_perturbed <= phi1_ref:
            self.log(f"Monotonicity test 2 failed: Area increased but Φ1 didn't increase ({phi1_perturbed} <= {phi1_ref})")
            success = False
        
        if phi2_perturbed <= phi2_ref:
            self.log(f"Monotonicity test 2 failed: Area increased but Φ2 didn't increase ({phi2_perturbed} <= {phi2_ref})")
            success = False
        
        if delta_perturbed <= delta_ref:
            self.log(f"Monotonicity test 2 failed: Area increased but |ΔΦ| didn't increase ({delta_perturbed} <= {delta_ref})")
            success = False
        
        # Test 3: Increase angle from 60° to 70° (flux should decrease)
        self.L = original_L  # restore L
        phi2_original = self.B * original_S * math.cos(math.radians(60))
        phi2_larger_angle = self.B * original_S * math.cos(math.radians(70))
        
        if phi2_larger_angle >= phi2_original:
            self.log(f"Monotonicity test 3 failed: Angle increased but Φ2 didn't decrease ({phi2_larger_angle} >= {phi2_original})")
            success = False
        
        # Restore all original values
        self.B = original_B
        self.L = original_L
        
        if success:
            self.log("Monotonicity tests passed")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        self.logs = []  # Reset logs
        
        self.log("Starting PhysicsChecker tests...")
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        success = test1 and test2 and test3
        
        self.log(f"Test summary: invariants={test1}, trace={test2}, monotonicity={test3}")
        self.log(f"Overall result: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs