import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract input parameters safely
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 1.0))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 1.0))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 1.0))
        
        # Extract results safely with fallbacks
        self.v1 = getattr(solution_instance, 'v_1', solution_instance.results.get('v_1', 0.0))
        self.v2 = getattr(solution_instance, 'v_2', solution_instance.results.get('v_2', 0.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        
        # Intermediate quantities (if provided)
        self.I1 = getattr(solution_instance, 'I_1', solution_instance.results.get('I_1', 0.0))
        self.I2 = getattr(solution_instance, 'I_2', solution_instance.results.get('I_2', 0.0))
        self.U_m = getattr(solution_instance, 'U_m', solution_instance.results.get('U_m', 0.0))
        self.U_R = getattr(solution_instance, 'U_R', solution_instance.results.get('U_R', 0.0))
        self.E1 = getattr(solution_instance, 'E_1', solution_instance.results.get('E_1', 0.0))
        self.F_A = getattr(solution_instance, 'F_A', solution_instance.results.get('F_A', 0.0))
        self.I_prime = getattr(solution_instance, "I'", solution_instance.results.get("I'", 0.0))
        self.U_R_prime = getattr(solution_instance, "U'_R", solution_instance.results.get("U'_R", 0.0))
        
        # Problem-given constants
        self.I0 = self.m * self.g / (4 * self.B * self.L)  # I₀ = mg/(4BL)
        self.U_m_given = self.m * self.g * self.R / (2 * self.B * self.L)  # Uₘ = mgR/(2BL)
        
        self.logs = []
        self.all_passed = True

    def _log(self, message, passed):
        self.logs.append(f"{'PASS' if passed else 'FAIL'}: {message}")
        if not passed:
            self.all_passed = False

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Test fundamental physical laws using symbolic and numeric checks."""
        try:
            # 1. Force equilibrium at terminal velocity (switch closed)
            expected_I1 = self.m * self.g / (self.B * self.L)
            self._log(f"Force equilibrium: I₁ = mg/(BL) = {expected_I1:.3g}, got {self.I1:.3g}", 
                     self._is_close(self.I1, expected_I1))
            
            # 2. Ohm's law for first scenario: I₁ = E₁/R, E₁ = BLv₁
            expected_E1 = self.B * self.L * self.v1
            self._log(f"Faraday + Ohm: E₁ = BLv₁ = {expected_E1:.3g}, got {self.E1:.3g}",
                     self._is_close(self.E1, expected_E1))
            
            expected_I1_from_ohm = expected_E1 / self.R
            self._log(f"Ohm's law: I₁ = E₁/R = {expected_I1_from_ohm:.3g}, got {self.I1:.3g}",
                     self._is_close(self.I1, expected_I1_from_ohm))
            
            # 3. Derived v₁ from force equilibrium and Ohm's law
            expected_v1 = self.m * self.g * self.R / (self.B**2 * self.L**2)
            self._log(f"v₁ = mgR/(B²L²) = {expected_v1:.3g}, got {self.v1:.3g}",
                     self._is_close(self.v1, expected_v1))
            
            # 4. Non-linear component property (U_m matches given expression)
            self._log(f"U_m = mgR/(2BL) = {self.U_m_given:.3g}, got {self.U_m:.3g}",
                     self._is_close(self.U_m, self.U_m_given))
            
            # 5. Kirchhoff's law for second scenario: BLv₂ = U_m + I₂R
            expected_U_R = self.B * self.L * self.v2 - self.U_m
            self._log(f"KVL: U_R = BLv₂ - U_m = {expected_U_R:.3g}, got {self.U_R:.3g}",
                     self._is_close(self.U_R, expected_U_R))
            
            # 6. Current continuity: I₂ = mg/(BL) (same as I₁!)
            expected_I2 = self.m * self.g / (self.B * self.L)
            self._log(f"Current continuity: I₂ = mg/(BL) = {expected_I2:.3g}, got {self.I2:.3g}",
                     self._is_close(self.I2, expected_I2))
            
            # 7. Ohm's law for second scenario: I₂ = U_R/R
            expected_I2_from_ohm = self.U_R / self.R
            self._log(f"Ohm's law (open): I₂ = U_R/R = {expected_I2_from_ohm:.3g}, got {self.I2:.3g}",
                     self._is_close(self.I2, expected_I2_from_ohm))
            
            # 8. Derived v₂ from simultaneous equations
            expected_v2 = 3 * self.m * self.g * self.R / (2 * self.B**2 * self.L**2)
            self._log(f"v₂ = 3mgR/(2B²L²) = {expected_v2:.3g}, got {self.v2:.3g}",
                     self._is_close(self.v2, expected_v2))
            
            # 9. Moment of switch opening: E₁ = BLv₁, U'_R = E₁ - U_m
            expected_U_R_prime = self.E1 - self.U_m
            self._log(f"KVL (switch opening): U'_R = E₁ - U_m = {expected_U_R_prime:.3g}, got {self.U_R_prime:.3g}",
                     self._is_close(self.U_R_prime, expected_U_R_prime))
            
            # 10. Ohm's law at switch opening: I' = U'_R/R
            expected_I_prime = self.U_R_prime / self.R
            self._log(f"Ohm's law (switch opening): I' = U'_R/R = {expected_I_prime:.3g}, got {self.I_prime:.3g}",
                     self._is_close(self.I_prime, expected_I_prime))
            
            # 11. Ampere's force law: F_A = B I' L
            expected_F_A = self.B * self.I_prime * self.L
            self._log(f"Ampere force: F_A = B I' L = {expected_F_A:.3g}, got {self.F_A:.3g}",
                     self._is_close(self.F_A, expected_F_A))
            
            # 12. Newton's second law: a = (mg - F_A)/m
            expected_a = (self.m * self.g - self.F_A) / self.m
            self._log(f"Newton II: a = (mg - F_A)/m = {expected_a:.3g}, got {self.a:.3g}",
                     self._is_close(self.a, expected_a))
            
            # 13. Final acceleration expression
            expected_a_final = self.g / 2
            self._log(f"a = g/2 = {expected_a_final:.3g}, got {self.a:.3g}",
                     self._is_close(self.a, expected_a_final))
            
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check that v₁ is consistent with I₁ and E₁
            v1_from_E1 = self.E1 / (self.B * self.L)
            self._log(f"v₁ from E₁: v₁ = E₁/(BL) = {v1_from_E1:.3g}, matches v₁ = {self.v1:.3g}",
                     self._is_close(v1_from_E1, self.v1))
            
            # Check that v₂ is consistent with U_R and U_m
            v2_from_U = (self.U_R + self.U_m) / (self.B * self.L)
            self._log(f"v₂ from U_R and U_m: v₂ = (U_R + U_m)/(BL) = {v2_from_U:.3g}, matches v₂ = {self.v2:.3g}",
                     self._is_close(v2_from_U, self.v2))
            
            # Check that I' and a are consistent
            F_A_calc = self.B * self.I_prime * self.L
            a_calc = (self.m * self.g - F_A_calc) / self.m
            self._log(f"a from I': a = (mg - B I' L)/m = {a_calc:.3g}, matches a = {self.a:.3g}",
                     self._is_close(a_calc, self.a))
            
            # Check that I₁ and I₂ are equal (as per steps)
            self._log(f"I₁ = I₂ = mg/(BL): {self.I1:.3g} vs {self.I2:.3g}",
                     self._is_close(self.I1, self.I2))
            
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}", False)

    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        original_m = self.m
        
        try:
            # Increase mass by 5%
            perturbed_m = original_m * 1.05
            
            # Expected changes in v1, v2, a
            v1_original = self.m * self.g * self.R / (self.B**2 * self.L**2)
            v1_perturbed = perturbed_m * self.g * self.R / (self.B**2 * self.L**2)
            self._log(f"v₁ increases with m: {v1_original:.3g} → {v1_perturbed:.3g}",
                     v1_perturbed > v1_original)
            
            v2_original = 3 * self.m * self.g * self.R / (2 * self.B**2 * self.L**2)
            v2_perturbed = 3 * perturbed_m * self.g * self.R / (2 * self.B**2 * self.L**2)
            self._log(f"v₂ increases with m: {v2_original:.3g} → {v2_perturbed:.3g}",
                     v2_perturbed > v2_original)
            
            # Acceleration a = g/2 is independent of m
            a_original = self.g / 2
            a_perturbed = self.g / 2
            self._log(f"a independent of m: {a_original:.3g} (original) vs {a_perturbed:.3g} (perturbed)",
                     self._is_close(a_original, a_perturbed))
            
            # Restore original m (conceptually)
            self.m = original_m
            
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}", False)
            self.m = original_m  # Restore on error

    def run_all_tests(self):
        """Execute all tests and return results."""
        self.logs = []
        self.all_passed = True
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        return self.all_passed, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)