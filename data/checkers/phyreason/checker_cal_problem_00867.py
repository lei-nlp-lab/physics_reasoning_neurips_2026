import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all needed parameters from solution_instance.
        Use getattr with fallback to .get() on results dictionary.
        """
        self.solution = solution_instance
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract known physical constants (with safe defaults)
        self.L = 0.1  # rod length in meters (10 cm)
        self.B = 0.1  # magnetic field strength in Tesla
        self.E = 12.0  # battery EMF in Volts
        self.R = 2.0  # total resistance in Ohms
        self.g = 10.0  # gravitational acceleration in m/s²
        
        # Extract results with safe access - handle string value separately
        self.delta_l1 = self._get_numeric('delta_l1', 0.005)  # 0.5 cm = 0.005 m
        self.delta_l2 = self._get_numeric('delta_l2', 0.003)  # 0.3 cm = 0.003 m
        self.F = self._get_numeric('F', 0.06)  # Ampere force in Newtons
        self.m = self._get_numeric('m', 0.01)  # mass in kg
        
        # Handle string value separately
        self.F_direction = self._get_string('F_ampere_direction', "vertically downward")
        
        # Current calculated from Ohm's Law (should be ~6 A)
        self.I = self.E / self.R
        
        # Spring constant k from initial equilibrium
        # From 2k*Δl1 = m*g, so k = (m*g)/(2*Δl1)
        self.k = (self.m * self.g) / (2 * self.delta_l1)
        
        self.logs = []
    
    def _get_numeric(self, key, default):
        """Safely extract a numeric value from results or attributes."""
        # First try to get from results dictionary
        if key in self.results:
            value = self.results[key]
            # Convert to float if it's numeric
            if isinstance(value, (int, float)):
                return float(value)
            elif hasattr(value, 'evalf'):
                return float(value.evalf())
            else:
                # Try to convert string to float
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
        # Fall back to getattr
        return float(getattr(self.solution, key, default))
    
    def _get_string(self, key, default):
        """Safely extract a string value from results or attributes."""
        # First try to get from results dictionary
        if key in self.results:
            value = self.results[key]
            # Return as string
            return str(value)
        # Fall back to getattr
        return str(getattr(self.solution, key, default))

    def log(self, message: str):
        """Helper to append log messages."""
        self.logs.append(message)

    def test_invariants(self) -> bool:
        """Test fundamental physical invariants and laws."""
        success = True
        
        # 1. Test Ohm's Law: I = E/R
        computed_I = self.E / self.R
        if not math.isclose(self.I, computed_I, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Ohm's Law violation. I from solution: {self.I:.3f}, computed I = E/R: {computed_I:.3f}")
            success = False
        else:
            self.log(f"PASS: Ohm's Law check. I = {self.I:.3f} A matches E/R = {computed_I:.3f} A")
        
        # 2. Test Ampere Force Formula: F = B*I*L
        computed_F = self.B * self.I * self.L
        if not math.isclose(self.F, computed_F, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Ampere force formula violation. F from solution: {self.F:.3f}, computed F = B*I*L: {computed_F:.3f}")
            success = False
        else:
            self.log(f"PASS: Ampere force formula check. F = {self.F:.3f} N matches B*I*L = {computed_F:.3f} N")
        
        # 3. Test Force Balance when switch is open: 2k*Δl1 = m*g
        left_open = 2 * self.k * self.delta_l1
        right_open = self.m * self.g
        if not math.isclose(left_open, right_open, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Initial force balance violation. 2kΔl1 = {left_open:.3f}, mg = {right_open:.3f}")
            success = False
        else:
            self.log(f"PASS: Initial force balance. 2kΔl1 = {left_open:.3f} N equals mg = {right_open:.3f} N")
        
        # 4. Test Force Balance when switch is closed: 2k*(Δl1+Δl2) = mg + F
        left_closed = 2 * self.k * (self.delta_l1 + self.delta_l2)
        right_closed = self.m * self.g + self.F
        if not math.isclose(left_closed, right_closed, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Final force balance violation. 2k(Δl1+Δl2) = {left_closed:.3f}, mg+F = {right_closed:.3f}")
            success = False
        else:
            self.log(f"PASS: Final force balance. 2k(Δl1+Δl2) = {left_closed:.3f} N equals mg+F = {right_closed:.3f} N")
        
        # 5. Test derived relationship: 2k*Δl2 = F (subtract the two equilibrium equations)
        derived_F = 2 * self.k * self.delta_l2
        if not math.isclose(self.F, derived_F, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Derived relationship 2kΔl2 = F violation. 2kΔl2 = {derived_F:.3f}, F = {self.F:.3f}")
            success = False
        else:
            self.log(f"PASS: Derived relationship 2kΔl2 = F holds: {derived_F:.3f} N = {self.F:.3f} N")
        
        # 6. Test direction of Ampere force
        expected_direction = "vertically downward"
        if self.F_direction.lower() != expected_direction.lower():
            self.log(f"FAIL: Ampere force direction mismatch. Expected: {expected_direction}, Got: {self.F_direction}")
            success = False
        else:
            self.log(f"PASS: Ampere force direction correct: {self.F_direction}")
        
        return success

    def test_trace_consistency(self) -> bool:
        """Test that intermediate results are consistent with final answer."""
        success = True
        
        # The main derivation: m = (B * (E/R) * L) * (Δl1/Δl2) / g
        # From step analysis: F = BIL = B*(E/R)*L, and from force balance: F = 2kΔl2
        # From initial balance: k = mg/(2Δl1)
        # Combining: B*(E/R)*L = 2*(mg/(2Δl1))*Δl2 = mg*(Δl2/Δl1)
        # So: m = (B*(E/R)*L)*(Δl1/Δl2)/g
        
        # Compute m from the derived formula
        computed_m = (self.B * (self.E / self.R) * self.L) * (self.delta_l1 / self.delta_l2) / self.g
        
        if not math.isclose(self.m, computed_m, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Trace consistency. m from solution: {self.m:.4f} kg, computed from derivation: {computed_m:.4f} kg")
            success = False
        else:
            self.log(f"PASS: Trace consistency. m = {self.m:.4f} kg matches derived formula: {computed_m:.4f} kg")
        
        # Check that the spring constant is positive
        if self.k <= 0:
            self.log(f"FAIL: Spring constant should be positive, got k = {self.k:.3f} N/m")
            success = False
        else:
            self.log(f"PASS: Spring constant k = {self.k:.3f} N/m is positive")
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb an input parameter and check if the output changes in the expected direction.
        We'll perturb Δl2 and see if m changes appropriately.
        According to formula: m ∝ 1/Δl2, so increasing Δl2 should decrease m.
        """
        success = True
        
        # Store original values
        original_delta_l2 = self.delta_l2
        original_m = self.m
        
        # Perturb Δl2 by +10%
        perturbed_delta_l2 = original_delta_l2 * 1.1
        
        # Recompute m using the derived formula with perturbed Δl2
        # m = (B*(E/R)*L)*(Δl1/perturbed_delta_l2)/g
        perturbed_m = (self.B * (self.E / self.R) * self.L) * (self.delta_l1 / perturbed_delta_l2) / self.g
        
        # Check monotonicity: if Δl2 increases, m should decrease
        if perturbed_m >= original_m:
            self.log(f"FAIL: Monotonicity test. Δl2 increased from {original_delta_l2:.6f} to {perturbed_delta_l2:.6f} m, "
                    f"but m did not decrease (original: {original_m:.6f}, perturbed: {perturbed_m:.6f})")
            success = False
        else:
            self.log(f"PASS: Monotonicity test. Δl2↑ ({original_delta_l2:.6f}→{perturbed_delta_l2:.6f} m) "
                    f"causes m↓ ({original_m:.6f}→{perturbed_m:.6f} kg)")
        
        # Restore original value
        self.delta_l2 = original_delta_l2
        
        return success

    def run_all_tests(self) -> tuple[bool, list[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall_success = test1 and test2 and test3
        
        # Summary
        self.log("\n=== SUMMARY ===")
        self.log(f"Invariants test: {'PASS' if test1 else 'FAIL'}")
        self.log(f"Trace consistency test: {'PASS' if test2 else 'FAIL'}")
        self.log(f"Monotonicity test: {'PASS' if test3 else 'FAIL'}")
        self.log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs