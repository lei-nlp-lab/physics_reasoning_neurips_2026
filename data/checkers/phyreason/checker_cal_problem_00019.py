import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        self.solution = solution_instance
        
        # Extract input parameters with safe access
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'results', {}).get('v0', 0.0))
        self.delta_t = getattr(solution_instance, 'delta_t', getattr(solution_instance, 'dt', getattr(solution_instance, 'results', {}).get('delta_t', 0.0)))
        self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 0.0))
        self.M = getattr(solution_instance, 'M', getattr(solution_instance, 'results', {}).get('M', 0.0))
        
        # Extract computed results with safe access
        self.v1 = getattr(solution_instance, 'v_1', getattr(solution_instance, 'results', {}).get('v_1', 0.0))
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 0.0))
        
        # Setup symbolic variables for invariant checks
        self.sym_v0, self.sym_dt, self.sym_v, self.sym_m, self.sym_M = sp.symbols('v0 dt v m M')
        
        self.logs = []
    
    def _log(self, message, status="INFO"):
        """Helper to log messages with status."""
        self.logs.append(f"[{status}] {message}")
    
    def test_invariants(self):
        """Test conservation of momentum and impulse-momentum theorem."""
        checks_passed = True
        
        # 1. Conservation of momentum: p_initial = p_final
        p_initial = (self.m + self.M) * self.v0
        p_final = self.m * self.v1 + self.M * self.v
        
        if not math.isclose(p_initial, p_final, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Momentum conservation failed: p_initial={p_initial:.4f}, p_final={p_final:.4f}", "FAIL")
            checks_passed = False
        else:
            self._log(f"Momentum conservation passed: p_initial={p_initial:.4f} ≈ p_final={p_final:.4f}", "PASS")
        
        # 2. Impulse-momentum theorem for composite B: M*(v - v0) = F*Δt
        delta_p_B = self.M * (self.v - self.v0)
        impulse = self.F * self.delta_t
        
        if not math.isclose(delta_p_B, impulse, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Impulse-momentum theorem failed: Δp_B={delta_p_B:.4f}, F*Δt={impulse:.4f}", "FAIL")
            checks_passed = False
        else:
            self._log(f"Impulse-momentum theorem passed: Δp_B={delta_p_B:.4f} ≈ F*Δt={impulse:.4f}", "PASS")
        
        # 3. Symbolic check of conservation of momentum formula
        try:
            sym_v1 = ((self.sym_m + self.sym_M) * self.sym_v0 - self.sym_M * self.sym_v) / self.sym_m
            momentum_eq = sp.Eq((self.sym_m + self.sym_M) * self.sym_v0, self.sym_m * sym_v1 + self.sym_M * self.sym_v)
            if not momentum_eq:
                self._log("Symbolic momentum check failed", "WARNING")
        except Exception as e:
            self._log(f"Symbolic momentum check error: {e}", "WARNING")
        
        # 4. Symbolic check of impulse-momentum theorem
        try:
            sym_F = self.sym_M * (self.sym_v - self.sym_v0) / self.sym_dt
            impulse_eq = sp.Eq(self.sym_M * (self.sym_v - self.sym_v0), sym_F * self.sym_dt)
            if not impulse_eq:
                self._log("Symbolic impulse check failed", "WARNING")
        except Exception as e:
            self._log(f"Symbolic impulse check error: {e}", "WARNING")
        
        return checks_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        checks_passed = True
        
        # Check v1 formula: v1 = ((m+M)v0 - Mv)/m
        expected_v1 = ((self.m + self.M) * self.v0 - self.M * self.v) / self.m if abs(self.m) > 1e-12 else 0.0
        
        if not math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"v1 consistency failed: computed={self.v1:.4f}, expected={expected_v1:.4f}", "FAIL")
            checks_passed = False
        else:
            self._log(f"v1 consistency passed: {self.v1:.4f} ≈ {expected_v1:.4f}", "PASS")
        
        # Check F formula: F = M(v - v0)/Δt
        if abs(self.delta_t) > 1e-12:
            expected_F = self.M * (self.v - self.v0) / self.delta_t
        else:
            expected_F = 0.0
            self._log("Δt is zero or extremely small, F formula invalid", "WARNING")
        
        if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"F consistency failed: computed={self.F:.4f}, expected={expected_F:.4f}", "FAIL")
            checks_passed = False
        else:
            self._log(f"F consistency passed: {self.F:.4f} ≈ {expected_F:.4f}", "PASS")
        
        return checks_passed
    
    def test_monotonicity(self):
        """Perturb input v and check response trends, then restore."""
        checks_passed = True
        
        # Save original state
        original_v = self.v
        original_v1 = self.v1
        original_F = self.F
        
        # Perturb v by 2% (relative perturbation)
        perturb_factor = 1.02
        perturbed_v = original_v * perturb_factor
        
        # Recalculate using formulas (not calling solution methods)
        perturbed_v1 = ((self.m + self.M) * self.v0 - self.M * perturbed_v) / self.m if abs(self.m) > 1e-12 else 0.0
        perturbed_F = self.M * (perturbed_v - self.v0) / self.delta_t if abs(self.delta_t) > 1e-12 else 0.0
        
        # Check expected trends:
        # 1. v1 should decrease when v increases (since derivative ∂v1/∂v = -M/m < 0)
        # 2. F should increase when v increases (since ∂F/∂v = M/Δt > 0)
        
        v_change = perturbed_v - original_v
        v1_change = perturbed_v1 - original_v1
        F_change = perturbed_F - original_F
        
        # Trend 1: v1 should change opposite to v
        if v_change > 0 and v1_change > 0:
            self._log(f"Monotonicity fail: v↑ ({v_change:.4f}) but v1↑ ({v1_change:.4f}), expected v1↓", "FAIL")
            checks_passed = False
        elif v_change < 0 and v1_change < 0:
            self._log(f"Monotonicity fail: v↓ ({v_change:.4f}) but v1↓ ({v1_change:.4f}), expected v1↑", "FAIL")
            checks_passed = False
        else:
            self._log(f"Monotonicity passed for v1: v_change={v_change:.4f}, v1_change={v1_change:.4f} (opposite signs)", "PASS")
        
        # Trend 2: F should change same as v (if Δt > 0)
        if abs(self.delta_t) > 1e-12:
            if v_change > 0 and F_change < 0:
                self._log(f"Monotonicity fail: v↑ ({v_change:.4f}) but F↓ ({F_change:.4f}), expected F↑", "FAIL")
                checks_passed = False
            elif v_change < 0 and F_change > 0:
                self._log(f"Monotonicity fail: v↓ ({v_change:.4f}) but F↑ ({F_change:.4f}), expected F↓", "FAIL")
                checks_passed = False
            else:
                self._log(f"Monotonicity passed for F: v_change={v_change:.4f}, F_change={F_change:.4f} (same signs)", "PASS")
        else:
            self._log("Skipping F monotonicity check (Δt ≈ 0)", "WARNING")
        
        # Restore original state
        self.v = original_v
        
        return checks_passed
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...", "INFO")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", "ERROR")
                success = False
        
        self._log(f"Overall result: {'PASS' if success else 'FAIL'}", "FINAL")
        return success, self.logs