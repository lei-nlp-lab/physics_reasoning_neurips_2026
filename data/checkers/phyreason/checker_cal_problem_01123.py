import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants
        self.g = 10.0  # m/s^2
        self.t1 = 4.0  # s
        self.t2 = 7.0  # s
        self.v2 = 5.0  # m/s
        
        # Extract results safely
        self.v_1 = float(getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0)))
        self.a = float(getattr(solution_instance, 'a', self.results.get('a', 0.0)))
        self.h_1 = float(getattr(solution_instance, 'h_1', self.results.get('h_1', 0.0)))
        self.h_2 = float(getattr(solution_instance, 'h_2', self.results.get('h_2', 0.0)))
        self.h = float(getattr(solution_instance, 'h', self.results.get('h', 0.0)))
        
        # Expected values from steps analysis
        self.expected_v1 = self.g * self.t1  # 40 m/s
        self.expected_a = (self.expected_v1 - self.v2) / self.t2  # 5 m/s²
        self.expected_h1 = 0.5 * self.g * self.t1**2  # 80 m
        self.expected_h2 = (self.v2**2 - self.expected_v1**2) / (-2 * self.expected_a)  # 157.5 m
        self.expected_h = self.expected_h1 + self.expected_h2  # 237.5 m
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
        
    def test_invariants(self):
        """Check kinematic consistency using KB theorems."""
        tests = []
        
        # Free fall motion (Theorem 1,2): v1 = g*t1
        if math.isclose(self.v_1, self.expected_v1, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Free fall: v1 = {self.v_1:.1f} ≈ g*t1 = {self.expected_v1:.1f} m/s", True))
        else:
            tests.append(self._log(f"Free fall: v1 = {self.v_1:.1f} ≠ g*t1 = {self.expected_v1:.1f} m/s", False))
        
        # Uniform deceleration (Theorem 4): a = (v1 - v2)/t2
        if math.isclose(abs(self.a), self.expected_a, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Deceleration: |a| = {abs(self.a):.1f} ≈ (v1-v2)/t2 = {self.expected_a:.1f} m/s²", True))
        else:
            tests.append(self._log(f"Deceleration: |a| = {abs(self.a):.1f} ≠ (v1-v2)/t2 = {self.expected_a:.1f} m/s²", False))
        
        # Displacement consistency: h1 from free fall
        if math.isclose(self.h_1, self.expected_h1, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Free fall displacement: h1 = {self.h_1:.1f} ≈ ½gt1² = {self.expected_h1:.1f} m", True))
        else:
            tests.append(self._log(f"Free fall displacement: h1 = {self.h_1:.1f} ≠ ½gt1² = {self.expected_h1:.1f} m", False))
        
        # Displacement consistency: h2 from deceleration
        if math.isclose(self.h_2, self.expected_h2, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Deceleration displacement: h2 = {self.h_2:.1f} ≈ (v2²-v1²)/(-2a) = {self.expected_h2:.1f} m", True))
        else:
            tests.append(self._log(f"Deceleration displacement: h2 = {self.h_2:.1f} ≠ (v2²-v1²)/(-2a) = {self.expected_h2:.1f} m", False))
        
        # Energy conservation check: mgh = ½mv2² + work_against_drag (but we don't have drag force)
        # Instead, check total displacement consistency
        if math.isclose(self.h, self.expected_h, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Total height: h = {self.h:.1f} ≈ h1+h2 = {self.expected_h:.1f} m", True))
        else:
            tests.append(self._log(f"Total height: h = {self.h:.1f} ≠ h1+h2 = {self.expected_h:.1f} m", False))
        
        return all(tests)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        tests = []
        
        # Check if intermediate values yield the final height
        h_sum = self.h_1 + self.h_2
        if math.isclose(self.h, h_sum, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Trace: h = {self.h:.1f} = h1+h2 = {h_sum:.1f} m", True))
        else:
            tests.append(self._log(f"Trace: h = {self.h:.1f} ≠ h1+h2 = {h_sum:.1f} m", False))
        
        # Alternative calculation for h2 using v1*t2 + ½*a*t2² (with a negative)
        h2_alt = self.v_1 * self.t2 + 0.5 * (-abs(self.a)) * self.t2**2
        if math.isclose(self.h_2, h2_alt, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Trace: h2 = {self.h_2:.1f} ≈ v1*t2 + ½*a*t2² = {h2_alt:.1f} m", True))
        else:
            tests.append(self._log(f"Trace: h2 = {self.h_2:.1f} ≠ v1*t2 + ½*a*t2² = {h2_alt:.1f} m", False))
        
        # Verify that deceleration brings v1 down to v2 correctly
        v2_calc = self.v_1 + (-abs(self.a)) * self.t2
        if math.isclose(self.v2, v2_calc, rel_tol=1e-2, abs_tol=1e-2):
            tests.append(self._log(f"Trace: v2 = {self.v2:.1f} ≈ v1 + a*t2 = {v2_calc:.1f} m/s", True))
        else:
            tests.append(self._log(f"Trace: v2 = {self.v2:.1f} ≠ v1 + a*t2 = {v2_calc:.1f} m/s", False))
        
        return all(tests)
    
    def test_monotonicity(self):
        """Perturb t1 and check if total height changes monotonically."""
        original_t1 = self.t1
        perturbed_t1 = original_t1 + 0.1  # Small increase
        
        # Recalculate using physics formulas
        v1_pert = self.g * perturbed_t1
        a_pert = (v1_pert - self.v2) / self.t2
        h1_pert = 0.5 * self.g * perturbed_t1**2
        h2_pert = (self.v2**2 - v1_pert**2) / (-2 * a_pert)
        h_pert = h1_pert + h2_pert
        
        # Check monotonicity: increasing t1 should increase total height
        if h_pert > self.expected_h:
            success = self._log(
                f"Monotonicity: t1↑ ({original_t1}→{perturbed_t1}s) ⇒ h↑ ({self.expected_h:.1f}→{h_pert:.1f}m)",
                True
            )
        else:
            success = self._log(
                f"Monotonicity: t1↑ ({original_t1}→{perturbed_t1}s) ⇒ h↓ ({self.expected_h:.1f}→{h_pert:.1f}m) - unexpected!",
                False
            )
        
        # Restore original value (conceptual)
        perturbed_t1 = original_t1
        
        return success
    
    def run_all_tests(self):
        """Execute all checks and return success status with logs."""
        self.logs = []
        self.logs.append("=== Physics Audit Report ===")
        
        try:
            # Run tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            # Summary
            success = inv_ok and trace_ok and mono_ok
            self.logs.append(f"\nSummary: {'ALL PASS' if success else 'FAILURES DETECTED'}")
            self.logs.append(f"  Invariants: {'PASS' if inv_ok else 'FAIL'}")
            self.logs.append(f"  Trace: {'PASS' if trace_ok else 'FAIL'}")
            self.logs.append(f"  Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"✗ Runtime error: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        return success, self.logs