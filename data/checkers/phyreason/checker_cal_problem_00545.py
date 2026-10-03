import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters
        self.R = 3000.0 * 1000.0  # Convert km to meters
        self.n_peak = 500  # oscillations at peak
        self.n_sea = 501   # oscillations at sea level
        
        # Extracted results with safe fallbacks
        self.H = getattr(solution_instance, 'H', self.results.get('H', None))
        self.T0 = getattr(solution_instance, 'T0', self.results.get('T0', None))
        self.T = getattr(solution_instance, 'T', self.results.get('T', None))
        self.g = getattr(solution_instance, 'g', self.results.get('g', None))
        self.g_H = getattr(solution_instance, 'g_H', self.results.get('g_H', None))
        
        # Physics constants
        self.G = 6.67430e-11
        self.pi = math.pi
        
        # Test logs
        self.logs = []
    
    def _log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        checks = []
        
        # 1. Check period ratio from oscillation counts
        if self.T0 is not None and self.T is not None:
            period_ratio_expected = self.n_sea / self.n_peak  # T/T0 = n_sea/n_peak
            period_ratio_actual = self.T / self.T0
            check1 = math.isclose(period_ratio_actual, period_ratio_expected, rel_tol=1e-2, abs_tol=1e-2)
            checks.append(self._log(f"Period ratio T/T0 = {period_ratio_actual:.4f} ≈ {period_ratio_expected:.4f} (n_sea/n_peak)", check1))
        else:
            checks.append(self._log("Skipped period ratio check (T0 or T missing)", True))
        
        # 2. Check g ratio from period formula
        if self.g is not None and self.g_H is not None:
            # From T ∝ 1/√g, so g_H/g = (T0/T)² = (n_peak/n_sea)²
            g_ratio_expected = (self.n_peak / self.n_sea) ** 2
            g_ratio_actual = self.g_H / self.g
            check2 = math.isclose(g_ratio_actual, g_ratio_expected, rel_tol=1e-2, abs_tol=1e-2)
            checks.append(self._log(f"Gravity ratio g_H/g = {g_ratio_actual:.4f} ≈ {g_ratio_expected:.4f} (n_peak/n_sea)²", check2))
        else:
            checks.append(self._log("Skipped gravity ratio check (g or g_H missing)", True))
        
        # 3. Check inverse square law for gravity
        if self.H is not None and self.g is not None and self.g_H is not None:
            # g_H/g = R²/(R+H)²
            expected_g_ratio_sq = (self.R ** 2) / ((self.R + self.H) ** 2)
            actual_g_ratio = self.g_H / self.g
            check3 = math.isclose(actual_g_ratio, expected_g_ratio_sq, rel_tol=1e-2, abs_tol=1e-2)
            checks.append(self._log(f"Inverse square law: g_H/g = {actual_g_ratio:.4f} ≈ R²/(R+H)² = {expected_g_ratio_sq:.4f}", check3))
        else:
            checks.append(self._log("Skipped inverse square law check (H, g, or g_H missing)", True))
        
        # 4. Check H = R/500 relationship (simplified formula)
        if self.H is not None:
            H_expected = self.R / 500.0
            check4 = math.isclose(self.H, H_expected, rel_tol=1e-2, abs_tol=1e-2)
            checks.append(self._log(f"Height H = {self.H:.0f}m ≈ R/500 = {H_expected:.0f}m", check4))
        else:
            checks.append(self._log("Skipped height check (H missing)", True))
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        checks = []
        
        # 1. Verify H from period ratio
        if self.T0 is not None and self.T is not None:
            # Correct formula: T/T0 = (R+H)/R => H = R*(T/T0 - 1)
            H_from_T = self.R * (self.T / self.T0 - 1)
            if self.H is not None:
                check1 = math.isclose(H_from_T, self.H, rel_tol=1e-2, abs_tol=1e-2)
                checks.append(self._log(f"H from periods: {H_from_T:.0f}m ≈ stored H: {self.H:.0f}m", check1))
            else:
                checks.append(self._log("Skipped H from periods (H missing)", True))
        else:
            checks.append(self._log("Skipped H from periods (T0 or T missing)", True))
        
        # 2. Verify H from gravity ratio
        if self.g is not None and self.g_H is not None:
            # g_H/g = R²/(R+H)² => (R+H)/R = sqrt(g/g_H) => H = R*(sqrt(g/g_H) - 1)
            H_from_g = self.R * (math.sqrt(self.g / self.g_H) - 1)
            if self.H is not None:
                check2 = math.isclose(H_from_g, self.H, rel_tol=1e-2, abs_tol=1e-2)
                checks.append(self._log(f"H from gravity: {H_from_g:.0f}m ≈ stored H: {self.H:.0f}m", check2))
            else:
                checks.append(self._log("Skipped H from gravity (H missing)", True))
        else:
            checks.append(self._log("Skipped H from gravity (g or g_H missing)", True))
        
        # 3. Check g and g_H consistency with periods (if all available)
        if all(v is not None for v in [self.T0, self.T, self.g, self.g_H]):
            # From T = 2π√(L/g), so T² ∝ 1/g
            ratio_from_T = (self.T0 / self.T) ** 2  # = g_H/g
            ratio_direct = self.g_H / self.g
            check3 = math.isclose(ratio_from_T, ratio_direct, rel_tol=1e-2, abs_tol=1e-2)
            checks.append(self._log(f"Period-gravity consistency: (T0/T)² = {ratio_from_T:.4f} ≈ g_H/g = {ratio_direct:.4f}", check3))
        else:
            checks.append(self._log("Skipped period-gravity consistency (missing values)", True))
        
        return all(checks)
    
    def test_monotonicity(self):
        """Test that perturbations produce expected trends."""
        checks = []
        
        # Backup original values
        original_H = self.H
        
        # 1. Perturb n_peak (increase by 1)
        new_n_peak = self.n_peak + 1  # 501 oscillations
        # New ratio: (n_sea / new_n_peak) = 501/501 = 1
        H_perturbed1 = self.R * (self.n_sea / new_n_peak - 1)  # Should be 0
        
        # Since original H > 0, perturbed H should decrease to 0
        if original_H is not None:
            check1 = H_perturbed1 < original_H
            checks.append(self._log(f"Perturb n_peak↑ → H↓ (H_perturbed={H_perturbed1:.0f}m < H_original={original_H:.0f}m)", check1))
        else:
            checks.append(self._log("Skipped n_peak perturbation (H missing)", True))
        
        # 2. Perturb n_sea (decrease by 1)
        new_n_sea = self.n_sea - 1  # 500 oscillations
        # New ratio: (new_n_sea / n_peak) = 500/500 = 1
        H_perturbed2 = self.R * (new_n_sea / self.n_peak - 1)  # Should be 0
        
        # H should decrease
        if original_H is not None:
            check2 = H_perturbed2 < original_H
            checks.append(self._log(f"Perturb n_sea↓ → H↓ (H_perturbed={H_perturbed2:.0f}m < H_original={original_H:.0f}m)", check2))
        else:
            checks.append(self._log("Skipped n_sea perturbation (H missing)", True))
        
        # 3. Perturb R (increase by 10%)
        R_perturbed = self.R * 1.10
        H_perturbed3 = R_perturbed * (self.n_sea / self.n_peak - 1)  # R_perturbed/500
        
        # H should increase proportionally
        if original_H is not None:
            ratio_original = original_H / self.R
            ratio_perturbed = H_perturbed3 / R_perturbed
            check3 = math.isclose(ratio_original, ratio_perturbed, rel_tol=1e-2, abs_tol=1e-2)
            checks.append(self._log(f"Perturb R↑10% → H↑ proportionally (ΔH/H ≈ ΔR/R)", check3))
        else:
            checks.append(self._log("Skipped R perturbation (H missing)", True))
        
        # 4. Verify monotonic relationship: H decreases as n_peak increases
        # Use numeric derivative approximation instead of symbolic
        if original_H is not None:
            # Small change in n_peak
            delta_n = 1
            H_plus = self.R * (self.n_sea / (self.n_peak + delta_n) - 1)
            derivative_approx = (H_plus - original_H) / delta_n
            check4 = derivative_approx < 0
            checks.append(self._log(f"Monotonicity: dH/dn_peak ≈ {derivative_approx:.1f} < 0 (H decreases with more oscillations at peak)", check4))
        else:
            checks.append(self._log("Skipped monotonicity check (H missing)", True))
        
        return all(checks)
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary
        success = all([test1, test2, test3])
        self.logs.append(f"\nSUMMARY: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
        self.logs.append(f"  Invariants: {'PASS' if test1 else 'FAIL'}")
        self.logs.append(f"  Trace consistency: {'PASS' if test2 else 'FAIL'}")
        self.logs.append(f"  Monotonicity: {'PASS' if test3 else 'FAIL'}")
        
        return success, self.logs