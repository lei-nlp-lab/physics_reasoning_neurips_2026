import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        # Inputs from problem statement
        self.L = 0.5  # m
        self.R = 3.0  # Ω
        self.B = 4.0  # T
        self.m = 0.2  # kg
        self.r = 1.0  # Ω
        self.v0 = 8.0  # m/s
        
        # Extract results from solution_instance.results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        # Results with safe access - use .get() with appropriate defaults
        # Only extract numeric values; direction is optional string
        self.Q_total = float(results_dict.get('Q_total', 0.0))
        self.x = float(results_dict.get('x', 0.0))
        self.q = float(results_dict.get('q', 0.0))
        self.Q_R = float(results_dict.get('Q_R', 0.0))
        self.I = float(results_dict.get('I', 0.0))
        self.E = float(results_dict.get('E', 0.0))
        
        # Direction is optional - don't convert to float
        self.direction = results_dict.get('direction', '')
        
        self.results = {
            'Q_total': self.Q_total,
            'x': self.x,
            'q': self.q,
            'direction': self.direction,
            'Q_R': self.Q_R,
            'I': self.I,
            'E': self.E
        }
        
        # Physics constants
        self.g = 9.8
        
        # Log storage
        self.logs = []

    def log(self, msg: str) -> None:
        """Add message to logs."""
        self.logs.append(msg)

    def test_invariants(self) -> bool:
        """Test conservation laws and physical invariants."""
        passed = True
        
        # 1. Energy conservation: Initial KE = Total Joule heat
        initial_KE = 0.5 * self.m * self.v0**2
        if not math.isclose(self.Q_total, initial_KE, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Energy conservation FAILED: Q_total={self.Q_total:.3f}, initial_KE={initial_KE:.3f}")
            passed = False
        else:
            self.log(f"Energy conservation PASSED: Q_total={self.Q_total:.3f} ≈ initial_KE={initial_KE:.3f}")
        
        # 2. Ohm's law consistency: I = E/(R+r) with E = BLv0 at initial moment
        E_initial = self.B * self.L * self.v0
        I_expected = E_initial / (self.R + self.r)
        if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Ohm's law FAILED: I={self.I:.3f}, expected I=BLv0/(R+r)={I_expected:.3f}")
            passed = False
        else:
            self.log(f"Ohm's law PASSED: I={self.I:.3f} ≈ {I_expected:.3f}")
        
        # 3. Current direction (from steps analysis: Q to P)
        # Direction is not required to be in results per contract, so don't fail if missing
        if self.direction:
            if self.direction not in ["from Q to P", "Q to P"]:
                self.log(f"Direction mismatch: got '{self.direction}', expected 'from Q to P'")
                passed = False
            else:
                self.log(f"Direction PASSED: {self.direction}")
        else:
            self.log(f"Direction: not provided in results (allowed per contract)")
        
        # 4. Heat division: Q_R / Q_total = R/(R+r)
        if self.Q_total == 0:
            self.log("Heat division SKIPPED: Q_total is zero")
        else:
            expected_ratio = self.R / (self.R + self.r)
            actual_ratio = self.Q_R / self.Q_total
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Heat division FAILED: Q_R/Q_total={actual_ratio:.3f}, expected {expected_ratio:.3f}")
                passed = False
            else:
                self.log(f"Heat division PASSED: Q_R={self.Q_R:.3f}, Q_total={self.Q_total:.3f}, ratio={actual_ratio:.3f}")
        
        return passed

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer."""
        passed = True
        
        # 1. Check impulse-momentum relationship: BLq = mv0
        impulse_left = self.B * self.L * self.q
        momentum_change = self.m * self.v0  # From v0 to 0
        if not math.isclose(impulse_left, momentum_change, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Impulse-momentum FAILED: BLq={impulse_left:.3f}, mv0={momentum_change:.3f}")
            passed = False
        else:
            self.log(f"Impulse-momentum PASSED: BLq={impulse_left:.3f} ≈ mv0={momentum_change:.3f}")
        
        # 2. Check charge-flux relationship: q = (BLx)/(R+r)
        if (self.R + self.r) == 0:
            self.log("Charge-flux SKIPPED: R+r is zero")
        else:
            q_expected = (self.B * self.L * self.x) / (self.R + self.r)
            if not math.isclose(self.q, q_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Charge-flux FAILED: q={self.q:.3f}, expected (BLx)/(R+r)={q_expected:.3f}")
                passed = False
            else:
                self.log(f"Charge-flux PASSED: q={self.q:.3f} ≈ {q_expected:.3f}")
        
        # 3. Solve for x from the two relationships above
        # From BLq = mv0 and q = (BLx)/(R+r), we get:
        # B*L*(B*L*x)/(R+r) = m*v0 → B²L²x/(R+r) = m*v0 → x = m*v0*(R+r)/(B²L²)
        if (self.B**2 * self.L**2) == 0:
            self.log("Stopping distance SKIPPED: B²L² is zero")
        else:
            x_expected = self.m * self.v0 * (self.R + self.r) / (self.B**2 * self.L**2)
            if not math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Stopping distance FAILED: x={self.x:.3f}, expected from physics={x_expected:.3f}")
                passed = False
            else:
                self.log(f"Stopping distance PASSED: x={self.x:.3f} ≈ {x_expected:.3f}")
        
        return passed

    def test_monotonicity(self) -> bool:
        """Perturb mass and check effect on stopping distance."""
        original_mass = self.m
        perturbed_mass = original_mass * 1.1  # Increase by 10%
        
        # Compute expected x for perturbed mass using physics formula
        # x = m*v0*(R+r)/(B²L²)
        if (self.B**2 * self.L**2) == 0:
            self.log("Monotonicity SKIPPED: B²L² is zero")
            return True
            
        x_original = self.x
        x_perturbed = perturbed_mass * self.v0 * (self.R + self.r) / (self.B**2 * self.L**2)
        
        # Trend: Increasing mass should increase stopping distance (both have positive momentum)
        if x_perturbed > x_original:
            self.log(f"Monotonicity PASSED: m↑ from {original_mass:.3f} to {perturbed_mass:.3f} → x↑ from {x_original:.3f} to {x_perturbed:.3f}")
            passed = True
        else:
            self.log(f"Monotonicity FAILED: m↑ from {original_mass:.3f} to {perturbed_mass:.3f} → x from {x_original:.3f} to {x_perturbed:.3f}")
            passed = False
        
        # Restore original mass (important for subsequent tests)
        self.m = original_mass
        
        return passed

    def run_all_tests(self) -> tuple[bool, list[str]]:
        """Execute all tests and return results."""
        self.logs = []  # Clear previous logs
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary - direction test should not cause overall failure if not provided
        all_passed = test1 and test2 and test3
        self.log(f"\n=== SUMMARY ===")
        self.log(f"Invariants: {'PASS' if test1 else 'FAIL'}")
        self.log(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
        self.log(f"Monotonicity: {'PASS' if test3 else 'FAIL'}")
        self.log(f"Overall: {'PASS' if all_passed else 'FAIL'}")
        
        return all_passed, self.logs