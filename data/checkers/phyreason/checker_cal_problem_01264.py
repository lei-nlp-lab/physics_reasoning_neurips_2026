import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        
        # Safe extraction of common inputs
        self.R = getattr(solution_instance, 'R', 1.0)  # default radius
        self.g = getattr(solution_instance, 'g', 9.8)  # default gravity
        self.m = getattr(solution_instance, 'm', 1.0)  # default mass
        
        # Safe extraction of friction coefficients (given in problem)
        self.mu1 = 0.5  # kinetic friction with disk
        self.mu2 = 0.25  # kinetic friction with table
        
        # Extract results with fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Target answers - check multiple possible key formats
        # First try exact contract symbols (without LaTeX markup), then LaTeX keys, then alternative names
        self.omega = self._safe_get(results, 'omega', '$\\omega$', 'omega', 0.0)
        self.omega1 = self._safe_get(results, 'omega_1', '$\\omega_1$', 'omega1', 0.0)
        self.v1 = self._safe_get(results, 'v_1', '$v_1$', 'v1', 0.0)
        self.x1 = self._safe_get(results, 'x_1', '$x_1$', 'x1', 0.0)
        self.R1 = self._safe_get(results, 'R_1', '$R_1$', 'R1', 0.0)
        self.x2 = self._safe_get(results, 'x_2', '$x_2$', 'x2', 0.0)
        self.t1 = self._safe_get(results, 't_1', '$t_1$', 't1', 0.0)
        self.t2 = self._safe_get(results, 't_2', '$t_2$', 't2', 0.0)
        self.t = self._safe_get(results, 't', '$t$', 't', 0.0)
        self.a = self._safe_get(results, 'a', '$a$', 'a', 0.0)
        
        # Initialize logs
        self.logs = []
    
    def _safe_get(self, results, contract_key, latex_key, alt_key, default):
        """Helper to safely extract a value from results dict."""
        # Try contract key first (exact symbol from interface), then LaTeX key, then alternative
        value = results.get(contract_key, results.get(latex_key, results.get(alt_key, default)))
        # Convert sympy expressions to float if needed
        if hasattr(value, 'evalf'):
            value = float(value.evalf())
        return float(value) if value is not None else default
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physical laws based on KB theorems."""
        logs = []
        try:
            # 1. Check ω from centripetal condition (step_2)
            omega_expected = math.sqrt(self.mu1 * self.g / self.R)
            if not self._is_close(self.omega, omega_expected):
                logs.append(f"FAIL: ω mismatch. Expected {omega_expected:.4f}, got {self.omega:.4f}")
            
            # 2. Check ω1 = ω (step_3)
            if not self._is_close(self.omega1, self.omega):
                logs.append(f"FAIL: ω1 ≠ ω. ω1={self.omega1:.4f}, ω={self.omega:.4f}")
            
            # 3. Check v1 = ω1 * R (step_3)
            v1_expected = self.omega1 * self.R
            if not self._is_close(self.v1, v1_expected):
                logs.append(f"FAIL: v1 mismatch. Expected {v1_expected:.4f}, got {self.v1:.4f}")
            
            # 4. Check acceleration on table (step_4)
            a_expected = -self.mu2 * self.g  # negative since friction opposes motion
            if not self._is_close(abs(self.a), abs(a_expected)):
                logs.append(f"FAIL: Acceleration mismatch. Expected |{a_expected:.4f}|, got |{self.a:.4f}|")
            
            # 5. Check x1 = v1²/(2μ₂g) (step_5)
            x1_expected = self.v1**2 / (2 * self.mu2 * self.g)
            if not self._is_close(self.x1, x1_expected):
                logs.append(f"FAIL: x1 mismatch. Expected {x1_expected:.4f}, got {self.x1:.4f}")
            
            # 6. Check R1 = √(x1² + R²) (step_6)
            R1_expected = math.sqrt(self.x1**2 + self.R**2)
            if not self._is_close(self.R1, R1_expected):
                logs.append(f"FAIL: R1 mismatch. Expected {R1_expected:.4f}, got {self.R1:.4f}")
            
            # 7. Check t1 from kinematics (step_9): solve v1*t1 + 0.5*a*t1² = x2
            # From step_7, x2 = (3/4)R (given in steps analysis)
            x2_expected = 0.75 * self.R
            if not self._is_close(self.x2, x2_expected):
                logs.append(f"FAIL: x2 mismatch. Expected {x2_expected:.4f}, got {self.x2:.4f}")
            
            # 8. Check t1 = √(2R/g) (step_9)
            t1_expected = math.sqrt(2 * self.R / self.g)
            if not self._is_close(self.t1, t1_expected):
                logs.append(f"FAIL: t1 mismatch. Expected {t1_expected:.4f}, got {self.t1:.4f}")
            
            # 9. Check t2 = √(2R/g) (step_11)
            t2_expected = math.sqrt(2 * self.R / self.g)
            if not self._is_close(self.t2, t2_expected):
                logs.append(f"FAIL: t2 mismatch. Expected {t2_expected:.4f}, got {self.t2:.4f}")
            
            # 10. Check t = t1 + t2 (step_12)
            t_expected = self.t1 + self.t2
            if not self._is_close(self.t, t_expected):
                logs.append(f"FAIL: Total time mismatch. Expected {t_expected:.4f}, got {self.t:.4f}")
            
            if not logs:
                logs.append("PASS: All invariants satisfied.")
        except Exception as e:
            logs.append(f"ERROR in invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that final answers match derived values
            # Target 1: ω = √(g/(2R))
            target1 = math.sqrt(self.g / (2 * self.R))
            if not self._is_close(self.omega, target1):
                logs.append(f"FAIL: Target ω mismatch. Expected {target1:.4f}, got {self.omega:.4f}")
            
            # Target 2: R1 = √2 * R
            target2 = math.sqrt(2) * self.R
            if not self._is_close(self.R1, target2):
                logs.append(f"FAIL: Target R1 mismatch. Expected {target2:.4f}, got {self.R1:.4f}")
            
            # Target 3: t = 2√(2R/g)
            target3 = 2 * math.sqrt(2 * self.R / self.g)
            if not self._is_close(self.t, target3):
                logs.append(f"FAIL: Target t mismatch. Expected {target3:.4f}, got {self.t:.4f}")
            
            # Consistency between ω and v1
            if not self._is_close(self.v1, self.omega * self.R):
                logs.append(f"FAIL: v1 ≠ ω*R. v1={self.v1:.4f}, ω*R={self.omega*self.R:.4f}")
            
            # Consistency between x1 and v1
            if not self._is_close(self.x1, self.v1**2 / (2 * self.mu2 * self.g)):
                logs.append(f"FAIL: x1 ≠ v1²/(2μ₂g). x1={self.x1:.4f}, expected={self.v1**2/(2*self.mu2*self.g):.4f}")
            
            if not logs:
                logs.append("PASS: All trace consistency checks passed.")
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb input R and verify expected trends."""
        logs = []
        original_R = self.R
        
        try:
            # Perturb R by 10% increase
            perturb_factor = 1.10
            self.R *= perturb_factor
            
            # Recalculate expected values using formulas
            omega_new = math.sqrt(self.mu1 * self.g / self.R)
            R1_new = math.sqrt(2) * self.R
            t_new = 2 * math.sqrt(2 * self.R / self.g)
            
            # Restore original R
            self.R = original_R
            
            # Check trends
            # ω should decrease as R increases (ω ∝ 1/√R)
            if not (omega_new < self.omega):
                logs.append(f"FAIL: ω monotonicity. ω(new)={omega_new:.4f} should be < ω(orig)={self.omega:.4f}")
            
            # R1 should increase as R increases (R1 ∝ R)
            if not (R1_new > self.R1):
                logs.append(f"FAIL: R1 monotonicity. R1(new)={R1_new:.4f} should be > R1(orig)={self.R1:.4f}")
            
            # t should increase as R increases (t ∝ √R)
            if not (t_new > self.t):
                logs.append(f"FAIL: t monotonicity. t(new)={t_new:.4f} should be > t(orig)={self.t:.4f}")
            
            if not logs:
                logs.append("PASS: All monotonicity checks passed.")
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
            self.R = original_R  # Ensure restoration even on error
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        self.logs = []
        
        # Run tests
        self.logs.append("=== Testing Invariants ===")
        self.logs.extend(self.test_invariants())
        
        self.logs.append("\n=== Testing Trace Consistency ===")
        self.logs.extend(self.test_trace_consistency())
        
        self.logs.append("\n=== Testing Monotonicity ===")
        self.logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in self.logs:
            if log.startswith("FAIL:") or log.startswith("ERROR:"):
                success = False
                break
        
        return success, self.logs