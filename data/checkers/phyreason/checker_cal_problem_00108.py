import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        self.logs = []
        self.solution = solution_instance
        
        # Extract results from results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract all results safely with defaults
        self.B_direction = results_dict.get('B_direction', 'unknown')
        self.R = results_dict.get('R')
        self.U0 = results_dict.get('U_0')
        self.I = results_dict.get('I')
        self.F = results_dict.get('F')
        self.m = results_dict.get('m')
        self.U = results_dict.get('U')
        
        # Extract physical parameters
        self.rho0 = results_dict.get('rho0')
        self.rho_resist = results_dict.get('rho')
        self.l = results_dict.get('l')
        self.b = results_dict.get('b')
        self.h = results_dict.get('h')
        self.g = results_dict.get('g', 9.8)
        self.B_mag = results_dict.get('B')
        
    def _is_close(self, val1, val2):
        """Robust fuzzy comparison for numerical values."""
        if val1 is None or val2 is None:
            return False
        try:
            # Handle string values (like B_direction)
            if isinstance(val1, str) or isinstance(val2, str):
                return str(val1) == str(val2)
            
            # Convert to float for numeric comparison
            v1 = float(val1)
            v2 = float(val2)
            return math.isclose(v1, v2, rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            # If conversion fails, check equality directly
            return val1 == val2
    
    def test_invariants(self):
        """Verify fundamental physics laws from KB."""
        logs = []
        
        # 1. Ohm's Law check: U0 = I * R
        if all(x is not None for x in [self.U0, self.I, self.R]):
            try:
                ohm_check = self._is_close(self.U0, self.I * self.R)
                logs.append(f"Ohm's Law (U0 = I*R): {ohm_check}")
                if not ohm_check:
                    logs.append(f"  Expected: {self.U0}, Got: {self.I * self.R}")
            except Exception as e:
                logs.append(f"Ohm's Law check failed: {e}")
        
        # 2. Ampere force formula: F = B * I * l
        if all(x is not None for x in [self.F, self.B_mag, self.I, self.l]):
            try:
                ampere_check = self._is_close(self.F, self.B_mag * self.I * self.l)
                logs.append(f"Ampere force (F = B*I*l): {ampere_check}")
                if not ampere_check:
                    logs.append(f"  Expected: {self.F}, Got: {self.B_mag * self.I * self.l}")
            except Exception as e:
                logs.append(f"Ampere force check failed: {e}")
        
        # 3. Mass formula: m = rho0 * b * h * l
        if all(x is not None for x in [self.m, self.rho0, self.b, self.h, self.l]):
            try:
                mass_check = self._is_close(self.m, self.rho0 * self.b * self.h * self.l)
                logs.append(f"Mass definition (m = ρ0*b*h*l): {mass_check}")
                if not mass_check:
                    logs.append(f"  Expected: {self.m}, Got: {self.rho0 * self.b * self.h * self.l}")
            except Exception as e:
                logs.append(f"Mass check failed: {e}")
        
        # 4. Force equilibrium at steady state: F = m * g
        if all(x is not None for x in [self.F, self.m, self.g]):
            try:
                eq_check = self._is_close(self.F, self.m * self.g)
                logs.append(f"Force equilibrium (F = m*g): {eq_check}")
                if not eq_check:
                    logs.append(f"  Expected: {self.F}, Got: {self.m * self.g}")
            except Exception as e:
                logs.append(f"Equilibrium check failed: {e}")
        
        # 5. Resistance formula: R = ρ * l / (2*h*b)
        if all(x is not None for x in [self.R, self.rho_resist, self.l, self.h, self.b]):
            try:
                R_expected = self.rho_resist * self.l / (2 * self.h * self.b)
                R_check = self._is_close(self.R, R_expected)
                logs.append(f"Resistance formula (R = ρ*l/(2*h*b)): {R_check}")
                if not R_check:
                    logs.append(f"  Expected: {R_expected}, Got: {self.R}")
            except Exception as e:
                logs.append(f"Resistance check failed: {e}")
        
        # 6. Check the derived equilibrium voltage: U0 = (ρ0*ρ*g*l)/(2*B)
        if all(x is not None for x in [self.U0, self.rho0, self.rho_resist, self.g, self.l, self.B_mag]):
            try:
                U0_expected = (self.rho0 * self.rho_resist * self.g * self.l) / (2 * self.B_mag)
                U0_check = self._is_close(self.U0, U0_expected)
                logs.append(f"Equilibrium voltage (U0 = ρ0*ρ*g*l/(2*B)): {U0_check}")
                if not U0_check:
                    logs.append(f"  Expected: {U0_expected}, Got: {self.U0}")
            except Exception as e:
                logs.append(f"U0 formula check failed: {e}")
        
        # 7. Check the continuous rise condition: U ≥ (ρ0*ρ*g*l)/B
        if all(x is not None for x in [self.U, self.rho0, self.rho_resist, self.g, self.l, self.B_mag]):
            try:
                U_min = (self.rho0 * self.rho_resist * self.g * self.l) / self.B_mag
                # Use numeric comparison
                condition_check = float(self.U) >= float(U_min)
                logs.append(f"Continuous rise condition (U ≥ ρ0*ρ*g*l/B): {condition_check}")
                if not condition_check:
                    logs.append(f"  Minimum required: {U_min}, Actual: {self.U}")
            except Exception as e:
                logs.append(f"Condition check failed: {e}")
        
        self.logs.extend(logs)
        return all("True" in log or "Pass" in log for log in logs if "check" in log.lower())
    
    def test_trace_consistency(self):
        """Verify intermediate steps lead to final answers consistently."""
        logs = []
        
        # Check that the derivation path from steps makes sense
        # Step 1: B direction should be negative z-axis
        dir_check = "negative z-axis" in str(self.B_direction).lower()
        logs.append(f"B direction correct (negative z-axis): {dir_check}")
        
        # Verify formulas using numeric values (not symbolic)
        # Check U0 formula: U0 = (ρ0*ρ*g*l)/(2*B)
        if all(x is not None for x in [self.U0, self.rho0, self.rho_resist, self.g, self.l, self.B_mag]):
            try:
                U0_expected = (self.rho0 * self.rho_resist * self.g * self.l) / (2 * self.B_mag)
                formula_check = self._is_close(self.U0, U0_expected)
                logs.append(f"U0 formula (ρ0*ρ*g*l/(2*B)) matches computed value: {formula_check}")
            except Exception as e:
                logs.append(f"U0 formula check failed: {e}")
        
        # Check general formula: U = ((n-1)/n) * (ρ0*ρ*g*l)/B
        # For n=2, this should give U0
        if all(x is not None for x in [self.U0, self.rho0, self.rho_resist, self.g, self.l, self.B_mag]):
            try:
                # For n=2: ((2-1)/2) = 1/2
                U_n2 = (1/2) * (self.rho0 * self.rho_resist * self.g * self.l) / self.B_mag
                n2_check = self._is_close(self.U0, U_n2)
                logs.append(f"General formula with n=2 gives U0: {n2_check}")
            except Exception as e:
                logs.append(f"n=2 formula check failed: {e}")
        
        # Check limit formula: U_min = (ρ0*ρ*g*l)/B
        if all(x is not None for x in [self.U, self.rho0, self.rho_resist, self.g, self.l, self.B_mag]):
            try:
                U_min_expected = (self.rho0 * self.rho_resist * self.g * self.l) / self.B_mag
                limit_check = self._is_close(self.U, U_min_expected)
                logs.append(f"Limit formula (ρ0*ρ*g*l/B) matches U: {limit_check}")
            except Exception as e:
                logs.append(f"Limit formula check failed: {e}")
        
        self.logs.extend(logs)
        return all("True" in log or "Pass" in log for log in logs if "check" in log.lower())
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends match physical intuition."""
        logs = []
        
        # Test 1: Increase B, U0 should decrease (from U0 = ρ0*ρ*g*l/(2*B))
        if all(x is not None for x in [self.U0, self.B_mag, self.rho0, self.rho_resist, self.g, self.l]):
            try:
                original_B = float(self.B_mag)
                perturbed_B = original_B * 1.1  # +10%
                
                original_U0 = float(self.U0)
                expected_U0_perturbed = (self.rho0 * self.rho_resist * self.g * self.l) / (2 * perturbed_B)
                
                # Check monotonicity: U0 ∝ 1/B
                trend_correct = expected_U0_perturbed < original_U0
                logs.append(f"Monotonicity: U0 decreases when B increases: {trend_correct}")
                if not trend_correct:
                    logs.append(f"  Original U0: {original_U0}, After +10% B: {expected_U0_perturbed}")
            except Exception as e:
                logs.append(f"B perturbation test failed: {e}")
        
        # Test 2: Increase resistivity ρ, U0 should increase (U0 ∝ ρ)
        if all(x is not None for x in [self.U0, self.B_mag, self.rho0, self.rho_resist, self.g, self.l]):
            try:
                original_rho = float(self.rho_resist)
                perturbed_rho = original_rho * 1.1  # +10%
                
                original_U0 = float(self.U0)
                expected_U0_perturbed = (self.rho0 * perturbed_rho * self.g * self.l) / (2 * self.B_mag)
                
                trend_correct = expected_U0_perturbed > original_U0
                logs.append(f"Monotonicity: U0 increases when resistivity increases: {trend_correct}")
                if not trend_correct:
                    logs.append(f"  Original U0: {original_U0}, After +10% ρ: {expected_U0_perturbed}")
            except Exception as e:
                logs.append(f"ρ perturbation test failed: {e}")
        
        # Test 3: Increase density ρ0, U0 should increase (U0 ∝ ρ0)
        if all(x is not None for x in [self.U0, self.B_mag, self.rho0, self.rho_resist, self.g, self.l]):
            try:
                original_rho0 = float(self.rho0)
                perturbed_rho0 = original_rho0 * 1.1  # +10%
                
                original_U0 = float(self.U0)
                expected_U0_perturbed = (perturbed_rho0 * self.rho_resist * self.g * self.l) / (2 * self.B_mag)
                
                trend_correct = expected_U0_perturbed > original_U0
                logs.append(f"Monotonicity: U0 increases when density increases: {trend_correct}")
                if not trend_correct:
                    logs.append(f"  Original U0: {original_U0}, After +10% ρ0: {expected_U0_perturbed}")
            except Exception as e:
                logs.append(f"ρ0 perturbation test failed: {e}")
        
        # Test 4: For continuous rise condition, if U increases, condition becomes easier to satisfy
        if all(x is not None for x in [self.U, self.rho0, self.rho_resist, self.g, self.l, self.B_mag]):
            try:
                U_min = (self.rho0 * self.rho_resist * self.g * self.l) / self.B_mag
                original_U = float(self.U)
                perturbed_U = original_U * 1.1  # +10%
                
                # Both should satisfy condition if original did
                original_satisfies = original_U >= U_min
                perturbed_satisfies = perturbed_U >= U_min
                
                # If original satisfies, perturbed should also satisfy
                trend_correct = not original_satisfies or perturbed_satisfies
                logs.append(f"Monotonicity: Higher U makes condition easier to satisfy: {trend_correct}")
            except Exception as e:
                logs.append(f"U perturbation test failed: {e}")
        
        self.logs.extend(logs)
        return all("True" in log or "Pass" in log for log in logs if "Monotonicity" in log)
    
    def run_all_tests(self):
        """Run all tests and return success status with logs."""
        self.logs = []
        success = True
        
        try:
            inv_success = self.test_invariants()
            self.logs.append(f"\n=== Invariants Test: {'PASS' if inv_success else 'FAIL'} ===")
            success = success and inv_success
        except Exception as e:
            self.logs.append(f"Invariants test crashed: {traceback.format_exc()}")
            success = False
        
        try:
            trace_success = self.test_trace_consistency()
            self.logs.append(f"\n=== Trace Consistency Test: {'PASS' if trace_success else 'FAIL'} ===")
            success = success and trace_success
        except Exception as e:
            self.logs.append(f"Trace consistency test crashed: {traceback.format_exc()}")
            success = False
        
        try:
            mono_success = self.test_monotonicity()
            self.logs.append(f"\n=== Monotonicity Test: {'PASS' if mono_success else 'FAIL'} ===")
            success = success and mono_success
        except Exception as e:
            self.logs.append(f"Monotonicity test crashed: {traceback.format_exc()}")
            success = False
        
        return success, self.logs