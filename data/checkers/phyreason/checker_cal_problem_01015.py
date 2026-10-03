import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.L = getattr(solution_instance, 'L', 0.3)  # m (total height)
        self.S = getattr(solution_instance, 'S', 5e-4)  # m^2
        self.m = getattr(solution_instance, 'm', 1.0)  # kg
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s^2
        self.p0 = getattr(solution_instance, 'p0', 1e5)  # Pa
        self.T0 = getattr(solution_instance, 'T0', 300.0)  # K
        self.k = getattr(solution_instance, 'k', 600.0)  # N/m
        
        # Extract results safely (use provided key mapping)
        results = getattr(solution_instance, 'results', {})
        self.p1 = results.get('p_1', 0.0)
        self.p2 = results.get('p_2', 0.0)
        self.T1 = results.get('T_1', 0.0)
        
        # Log storage
        self.logs = []
    
    def _log(self, msg, is_error=False):
        """Helper to append log messages."""
        self.logs.append((msg, is_error))
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify key physical laws from the KB and steps analysis."""
        success = True
        
        try:
            # --- Step 1: Initial pressure p1 = p0 + 2mg/S ---
            p1_expected = self.p0 + 2 * self.m * self.g / self.S
            if not self._is_close(self.p1, p1_expected):
                self._log(f"Step 1 invariant failed: p1 = {self.p1} vs expected {p1_expected}", True)
                success = False
            else:
                self._log(f"Step 1 invariant passed: p1 = {self.p1} matches p0 + 2mg/S")
            
            # --- Step 2: Pressure p2 = p0 + (12mg + k*L)/(6S) ---
            # Note: L is total height = 0.30 m, not initial distance between pistons
            p2_expected = self.p0 + (12 * self.m * self.g + self.k * self.L) / (6 * self.S)
            if not self._is_close(self.p2, p2_expected):
                self._log(f"Step 2 invariant failed: p2 = {self.p2} vs expected {p2_expected}", True)
                success = False
            else:
                self._log(f"Step 2 invariant passed: p2 = {self.p2} matches p0 + (12mg + k*L)/(6S)")
            
            # --- Step 3: Temperature T1 from ideal gas law ---
            # Correct volumes: V1 = (L/3)*S, V2 = (5*L/6)*S
            V1 = (self.L/3) * self.S
            V2 = (5*self.L/6) * self.S
            T1_expected = self.p2 * V2 * self.T0 / (self.p1 * V1)
            if not self._is_close(self.T1, T1_expected):
                self._log(f"Step 3 invariant failed: T1 = {self.T1} vs expected {T1_expected}", True)
                success = False
            else:
                self._log(f"Step 3 invariant passed: T1 = {self.T1} matches (p2*V2*T0)/(p1*V1)")
            
            # --- Ideal gas law consistency ---
            left = self.p1 * V1 / self.T0
            right = self.p2 * V2 / self.T1
            if not self._is_close(left, right, rel_tol=2e-2):
                self._log(f"Ideal gas law failed: (p1*V1)/T0 = {left}, (p2*V2)/T1 = {right}", True)
                success = False
            else:
                self._log(f"Ideal gas law passed: (p1*V1)/T0 ≈ (p2*V2)/T1")
                
        except Exception as e:
            self._log(f"Exception in test_invariants: {e}\n{traceback.format_exc()}", True)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        try:
            # The target answers are p1 = 1.4e5 Pa and T1 = 1071 K
            # Check p1 against target
            if not self._is_close(self.p1, 1.4e5):
                self._log(f"p1 trace consistency failed: {self.p1} vs target 1.4e5", True)
                success = False
            else:
                self._log(f"p1 trace consistency passed: {self.p1} ≈ 1.4e5")
            
            # Check T1 against target
            if not self._is_close(self.T1, 1071.0, abs_tol=1.0):
                self._log(f"T1 trace consistency failed: {self.T1} vs target 1071", True)
                success = False
            else:
                self._log(f"T1 trace consistency passed: {self.T1} ≈ 1071")
            
            # Verify p2 from step 2 is consistent with T1 via ideal gas law
            V1 = (self.L/3) * self.S
            V2 = (5*self.L/6) * self.S
            T1_from_p2 = self.p2 * V2 * self.T0 / (self.p1 * V1)
            if not self._is_close(self.T1, T1_from_p2, rel_tol=2e-2):
                self._log(f"Trace consistency p2->T1 failed: T1={self.T1}, computed={T1_from_p2}", True)
                success = False
            else:
                self._log(f"Trace consistency p2->T1 passed: p2 leads to T1 ≈ {self.T1}")
                
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {e}\n{traceback.format_exc()}", True)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and check that p1, p2, T1 change in expected directions."""
        success = True
        
        try:
            original_m = self.m
            perturbed_m = original_m * 1.01  # Increase by 1%
            
            # Recompute expected values using formulas
            p1_original = self.p0 + 2 * original_m * self.g / self.S
            p1_perturbed = self.p0 + 2 * perturbed_m * self.g / self.S
            
            p2_original = self.p0 + (12 * original_m * self.g + self.k * self.L) / (6 * self.S)
            p2_perturbed = self.p0 + (12 * perturbed_m * self.g + self.k * self.L) / (6 * self.S)
            
            V1 = (self.L/3) * self.S
            V2 = (5*self.L/6) * self.S
            T1_original = p2_original * V2 * self.T0 / (p1_original * V1)
            T1_perturbed = p2_perturbed * V2 * self.T0 / (p1_perturbed * V1)
            
            # Check monotonicity: p1 and p2 should increase with mass
            if not (p1_perturbed > p1_original and p2_perturbed > p2_original):
                self._log(f"Monotonicity failed: p1 or p2 did not increase with mass", True)
                success = False
            else:
                self._log(f"Monotonicity passed: p1 and p2 increase with mass")
            
            # For T1, compute ratio
            ratio_original = p2_original / p1_original
            ratio_perturbed = p2_perturbed / p1_perturbed
            if ratio_perturbed > ratio_original:
                self._log(f"T1 monotonicity: ratio p2/p1 increases with mass, so T1 increases")
            else:
                self._log(f"T1 monotonicity: ratio p2/p1 decreases with mass, so T1 decreases")
            
            # Check consistency
            if not self._is_close(T1_perturbed, (p2_perturbed / p1_perturbed) * (5/2) * self.T0):
                self._log(f"Monotonicity consistency failed: T1 formula violated after perturbation", True)
                success = False
                
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {e}\n{traceback.format_exc()}", True)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self._log(f"\n--- Running {name} Test ---")
            if not test_func():
                overall_success = False
        
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs