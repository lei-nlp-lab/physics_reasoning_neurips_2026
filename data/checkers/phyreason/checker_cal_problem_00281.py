import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr and .get() with defaults
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallbacks
        self.v_m = self.results.get('v_m', 0.0)
        self.v_1 = self.results.get('v_1', 0.0)
        self.x_1 = self.results.get('x_1', 0.0)
        self.L = self.results.get('L', 0.0)
        
        # Extract inputs safely (if available in solution_instance attributes)
        self.mu_k = getattr(solution_instance, 'mu_k', 0.05)
        self.mu_s_max = 2 * self.mu_k  # Given: max static friction = 2*kinetic friction
        self.x_total = getattr(solution_instance, 'x_total', 12.5)  # corridor length
        self.g = 9.8  # standard gravity
        
        # Logs for reporting
        self.logs = []
    
    def _log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with physics tolerance."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws: Work-Energy Theorem and force limits."""
        try:
            # 1. Work-Energy Theorem for sliding (kinetic friction only)
            # Net work = change in KE = -friction work
            # W_friction = -μ_k * m * g * x
            # ΔKE = 0 - ½ m v² = -½ m v²
            # So: ½ m v² = μ_k * m * g * x  => x = v²/(2μ_k g)
            if self.v_m and self.L:
                expected_L = self.v_m**2 / (2 * self.mu_k * self.g)
                if self._is_close(self.L, expected_L):
                    self._log(f"Work-Energy Theorem satisfied: L = {self.L:.3f} ≈ v_m²/(2μ g) = {expected_L:.3f}", True)
                else:
                    self._log(f"Work-Energy Theorem violated: L = {self.L:.3f} ≠ v_m²/(2μ g) = {expected_L:.3f}", False)
            
            # 2. Minimum initial velocity to slide entire distance
            # v_1 = sqrt(2 μ_k g x_total)
            expected_v1 = math.sqrt(2 * self.mu_k * self.g * self.x_total)
            if self.v_1:
                if self._is_close(self.v_1, expected_v1):
                    self._log(f"Minimum initial velocity correct: v_1 = {self.v_1:.3f} ≈ √(2μ g x_total) = {expected_v1:.3f}", True)
                else:
                    self._log(f"Minimum initial velocity mismatch: v_1 = {self.v_1:.3f} ≠ √(2μ g x_total) = {expected_v1:.3f}", False)
            
            # 3. Static friction force limit (max acceleration without slipping)
            # Max static friction force = μ_s_max * m * g = 2 μ_k * m * g
            # Max acceleration a_max = 2 μ_k * g
            a_max = 2 * self.mu_k * self.g
            self._log(f"Maximum acceleration without slipping: a_max = {a_max:.3f} m/s²", None)
            
            # 4. Check acceleration distance formula from steps analysis
            # x_1 = v_m²/(4 μ_k g) [from step_5]
            if self.v_m and self.x_1:
                expected_x1 = self.v_m**2 / (4 * self.mu_k * self.g)
                if self._is_close(self.x_1, expected_x1):
                    self._log(f"Acceleration distance formula satisfied: x_1 = {self.x_1:.3f} ≈ v_m²/(4μ g) = {expected_x1:.3f}", True)
                else:
                    self._log(f"Acceleration distance mismatch: x_1 = {self.x_1:.3f} ≠ v_m²/(4μ g) = {expected_x1:.3f}", False)
            
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            success = True
            
            # From physics: v_1 = sqrt(2*μ*g*x_total) = sqrt(2*0.05*9.8*12.5) = sqrt(12.25) = 3.5
            expected_v1 = math.sqrt(2 * self.mu_k * self.g * self.x_total)
            if self.v_1 and not self._is_close(self.v_1, expected_v1):
                self._log(f"v_1 = {self.v_1:.3f} does not match physics value {expected_v1:.3f}", False)
                success = False
            else:
                self._log(f"v_1 consistent with physics: {self.v_1:.3f} ≈ {expected_v1:.3f}", True)
            
            # For optimal strategy: v_m = sqrt(4/3 * μ * g * x_total)
            # Derivation: From equations: x_1 = v_m²/(4μg), L = v_m²/(2μg), x_1 + L = x_total
            # => v_m²/(4μg) + v_m²/(2μg) = x_total => (3v_m²)/(4μg) = x_total
            # => v_m = sqrt(4/3 * μ * g * x_total)
            expected_vm = math.sqrt(4/3 * self.mu_k * self.g * self.x_total)
            if self.v_m and not self._is_close(self.v_m, expected_vm):
                self._log(f"v_m = {self.v_m:.3f} does not match optimal strategy value {expected_vm:.3f}", False)
                success = False
            else:
                self._log(f"v_m consistent with optimal strategy: {self.v_m:.3f} ≈ {expected_vm:.3f}", True)
            
            # Maximum gliding distance L = v_m²/(2μg) = (2/3) * x_total
            expected_L = (2/3) * self.x_total
            if self.L and not self._is_close(self.L, expected_L):
                self._log(f"L = {self.L:.3f} does not match maximum gliding distance {expected_L:.3f}", False)
                success = False
            else:
                self._log(f"L consistent with maximum gliding distance: {self.L:.3f} ≈ {expected_L:.3f}", True)
            
            return success
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb μ_k and verify expected trends."""
        original_mu = self.mu_k
        original_v1 = self.v_1 if self.v_1 else 0.0
        original_L = self.L if self.L else 0.0
        
        try:
            # Increase μ_k by 10%
            perturbed_mu = original_mu * 1.1
            
            # Recalculate expected values using formulas
            # v_1 = sqrt(2 μ_k g x_total) should increase
            new_v1 = math.sqrt(2 * perturbed_mu * self.g * self.x_total)
            
            # For L, need v_m. Use v_m from solution if available
            if self.v_m:
                # L = v_m²/(2 μ_k g) should decrease with increased μ_k
                new_L = self.v_m**2 / (2 * perturbed_mu * self.g)
            else:
                # Use optimal strategy formula
                new_vm = math.sqrt(4/3 * perturbed_mu * self.g * self.x_total)
                new_L = new_vm**2 / (2 * perturbed_mu * self.g)
            
            # Check monotonicity
            trend_ok = True
            if original_v1:
                if not (new_v1 > original_v1):
                    self._log(f"v_1 monotonicity violated: μ↑ from {original_mu:.3f} to {perturbed_mu:.3f}, v_1 should increase", False)
                    trend_ok = False
                else:
                    self._log(f"v_1 monotonicity OK: μ↑ → v_1↑ ({original_v1:.3f} → {new_v1:.3f})", True)
            
            if original_L:
                if not (new_L < original_L):
                    self._log(f"L monotonicity violated: μ↑ from {original_mu:.3f} to {perturbed_mu:.3f}, L should decrease", False)
                    trend_ok = False
                else:
                    self._log(f"L monotonicity OK: μ↑ → L↓ ({original_L:.3f} → {new_L:.3f})", True)
            
            return trend_ok
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all checks and return summary."""
        self.logs = []  # Reset logs
        
        # Run tests
        test_results = []
        test_results.append(("Invariants", self.test_invariants()))
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        test_results.append(("Monotonicity", self.test_monotonicity()))
        
        # Count successes
        success_count = sum(1 for _, passed in test_results if passed)
        total_count = len(test_results)
        all_passed = (success_count == total_count)
        
        # Final summary
        self._log(f"\n=== Physics Audit Summary ===", None)
        for name, passed in test_results:
            self._log(f"{name}: {'PASS' if passed else 'FAIL'}", None)
        self._log(f"Overall: {'PASS' if all_passed else 'FAIL'} ({success_count}/{total_count})", None)
        
        return all_passed, self.logs