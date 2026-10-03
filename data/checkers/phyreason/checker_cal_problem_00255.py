import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization - never access solution_instance.attr directly.
        All inputs and results are extracted with fallbacks.
        """
        # Extract inputs with safe defaults from problem statement
        self.AC = getattr(solution_instance, 'AC', 3.0)  # km
        self.CD = getattr(solution_instance, 'CD', 4.0)  # km
        self.BC = getattr(solution_instance, 'BC', 6.0)  # km
        self.DE = getattr(solution_instance, 'DE', 4.0)  # km
        self.police_speed = getattr(solution_instance, 'police_speed', 72.0)  # km/h
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))  # seconds
        self.x1 = getattr(solution_instance, 'x_1', results.get('x_1', 0.0))  # km
        self.x2 = getattr(solution_instance, 'x_2', results.get('x_2', 0.0))  # km
        
        # Expected values from problem context
        self.expected_x1 = 9.0  # km
        self.expected_x2 = 8.5  # km (given approximation)
        self.expected_t = 500.0  # seconds
        
        # Store logs for test results
        self.logs = []
    
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def _fuzzy_equal(self, a: float, b: float, desc: str = "") -> bool:
        """Fuzzy numerical comparison with 1-2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """Test fundamental physical invariants."""
        success = True
        
        # 1. Distance invariants from Pythagorean theorem
        # AD = sqrt(AC² + CD²)
        AD_calc = math.sqrt(self.AC**2 + self.CD**2)
        if not self._fuzzy_equal(AD_calc, 5.0, "AD calculation"):
            success = False
            self.logs.append(f"  Calculated AD = {AD_calc:.3f} km, expected 5.000 km")
        
        # x1 = AD + DE
        x1_calc = AD_calc + self.DE
        if not self._fuzzy_equal(x1_calc, self.expected_x1):
            success = False
            self.logs.append(f"  Calculated x1 = {x1_calc:.3f} km, expected {self.expected_x1} km")
        
        # x2 = sqrt(AC² + (CD + DE)²)
        x2_calc = math.sqrt(self.AC**2 + (self.CD + self.DE)**2)
        # Accept both the approximation (8.5) and exact value (sqrt(73) ≈ 8.544)
        if not (self._fuzzy_equal(x2_calc, self.expected_x2) or 
                self._fuzzy_equal(x2_calc, math.sqrt(73))):
            success = False
            self.logs.append(f"  Calculated x2 = {x2_calc:.3f} km, expected ~{self.expected_x2} km")
        
        # 2. Time invariant from uniform motion
        # Police distance BE = sqrt(BC² + CE²) where CE = CD + DE
        CE = self.CD + self.DE
        BE = math.sqrt(self.BC**2 + CE**2)
        if not self._fuzzy_equal(BE, 10.0, "BE calculation"):
            success = False
            self.logs.append(f"  Calculated BE = {BE:.3f} km, expected 10.000 km")
        
        # Convert police speed to m/s for time calculation
        # 72 km/h = 72 * 1000 / 3600 = 20 m/s
        police_speed_ms = self.police_speed * 1000 / 3600  # m/s
        
        # Time = distance / speed
        # BE in km = 10 km = 10000 m
        t_calc = (BE * 1000) / police_speed_ms  # seconds
        
        if not self._fuzzy_equal(t_calc, self.expected_t):
            success = False
            self.logs.append(f"  Calculated t = {t_calc:.1f} s, expected {self.expected_t} s")
        
        # 3. Check that time is positive
        if self.t <= 0:
            success = False
            self.logs.append(f"  Time must be positive, got t = {self.t}")
        
        return self._log("Invariants test", success)
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Uses the steps analysis to verify consistency.
        """
        success = True
        
        # Step 1: Check x1 consistency
        if not self._fuzzy_equal(self.x1, self.expected_x1):
            success = False
            self.logs.append(f"  x1 = {self.x1:.3f} km doesn't match expected {self.expected_x1} km")
        
        # Step 2: Check x2 consistency (allow both 8.5 and sqrt(73))
        x2_ok = (self._fuzzy_equal(self.x2, self.expected_x2) or 
                 self._fuzzy_equal(self.x2, math.sqrt(73)))
        if not x2_ok:
            success = False
            self.logs.append(f"  x2 = {self.x2:.3f} km doesn't match expected ~{self.expected_x2} km")
        
        # Step 3: Check time consistency
        # Calculate expected time from BE and police speed
        CE = self.CD + self.DE
        BE = math.sqrt(self.BC**2 + CE**2)
        police_speed_ms = self.police_speed * 1000 / 3600
        t_expected = (BE * 1000) / police_speed_ms
        
        if not self._fuzzy_equal(self.t, t_expected):
            success = False
            self.logs.append(f"  t = {self.t:.1f} s doesn't match calculated {t_expected:.1f} s")
        
        # Also check that the target answers are consistent
        target_distance_str = f"Distance: {self.x1} km, Displacement: {self.x2} km"
        target_time_str = f"{self.t}s"
        
        # Verify format and values
        if not (f"{self.x1:.1f}" in target_distance_str and f"{self.x2:.1f}" in target_distance_str):
            success = False
            self.logs.append("  Output format for distance/displacement incorrect")
        
        if not f"{self.t:.0f}" in target_time_str:
            success = False
            self.logs.append("  Output format for time incorrect")
        
        return self._log("Trace consistency test", success)
    
    def test_monotonicity(self) -> bool:
        """
        Perturb police speed and verify monotonic trend.
        Restores original value after test.
        """
        success = True
        
        # Store original values
        original_speed = self.police_speed
        original_t = self.t
        
        # Perturb police speed by +10%
        perturbed_speed = original_speed * 1.1
        
        # Calculate expected time with perturbed speed
        CE = self.CD + self.DE
        BE = math.sqrt(self.BC**2 + CE**2)
        perturbed_speed_ms = perturbed_speed * 1000 / 3600
        t_perturbed = (BE * 1000) / perturbed_speed_ms
        
        # Check monotonicity: Higher speed → lower time
        if not (t_perturbed < original_t or self._fuzzy_equal(t_perturbed, original_t)):
            success = False
            self.logs.append(f"  Monotonicity violated: t({perturbed_speed:.1f} km/h) = {t_perturbed:.1f} s, "
                           f"t({original_speed:.1f} km/h) = {original_t:.1f} s")
        
        # Restore original value
        self.police_speed = original_speed
        
        # Additional test: perturb DE (criminal's path length)
        original_DE = self.DE
        perturbed_DE = original_DE * 1.1  # +10%
        
        # Calculate BE with perturbed DE
        perturbed_CE = self.CD + perturbed_DE
        perturbed_BE = math.sqrt(self.BC**2 + perturbed_CE**2)
        t_perturbed_DE = (perturbed_BE * 1000) / (original_speed * 1000 / 3600)
        
        # Longer DE → longer BE → longer time (police travel further)
        if not (t_perturbed_DE > original_t or self._fuzzy_equal(t_perturbed_DE, original_t)):
            success = False
            self.logs.append(f"  Monotonicity violated with DE: t(DE={perturbed_DE:.1f} km) = {t_perturbed_DE:.1f} s, "
                           f"t(DE={original_DE:.1f} km) = {original_t:.1f} s")
        
        # Restore DE
        self.DE = original_DE
        
        return self._log("Monotonicity test", success)
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []  # Reset logs
        
        try:
            tests = [
                self.test_invariants,
                self.test_trace_consistency,
                self.test_monotonicity
            ]
            
            success = True
            for test in tests:
                if not test():
                    success = False
            
            # Add summary
            self.logs.append(f"\n{'='*40}")
            self.logs.append(f"Overall result: {'PASS' if success else 'FAIL'}")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Error during testing: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs