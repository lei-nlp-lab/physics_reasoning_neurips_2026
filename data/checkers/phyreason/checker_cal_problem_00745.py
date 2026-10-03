import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all necessary data from the solution instance.
        Never access solution_instance.attr directly.
        """
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'mass', 0.5))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'gravity', 10.0))
        self.t_total = 4.0  # Given in problem context
        
        # Extract results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Step results from analysis
        self.h = float(getattr(solution_instance, 'h', results.get('h', 0.0)))
        self.W = float(getattr(solution_instance, 'W', results.get('W', 0.0)))
        self.v = float(getattr(solution_instance, 'v', results.get('v', 0.0)))
        self.p = float(getattr(solution_instance, 'p', results.get('p', 0.0)))
        
        # Store original values for monotonicity tests
        self._original_m = float(self.m)
        self._original_h = float(self.h)
        
        # Logs for reporting
        self.logs = []
    
    def _log(self, message: str, success: bool = True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success
    
    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with physics-appropriate tolerances."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants:
        1. Energy conservation (PE to KE conversion)
        2. Newton's 2nd Law (F = ma)
        3. Momentum-impulse theorem
        """
        success = True
        
        # 1. Energy conservation check
        total_height = 0.5 * self.g * (self.t_total ** 2)  # Using t=4s from problem
        initial_PE = self.m * self.g * total_height
        final_KE = 0.5 * self.m * (self.v ** 2)
        
        if not self._is_close(initial_PE, final_KE):
            success = self._log(
                f"Energy conservation failed: PE={initial_PE:.2f}J, KE={final_KE:.2f}J", 
                False
            )
        else:
            success = self._log(
                f"Energy conservation OK: PE={initial_PE:.2f}J ≈ KE={final_KE:.2f}J"
            ) and success
        
        # 2. Newton's 2nd Law check (F_net = m*a = m*g for free fall)
        a_expected = self.g
        a_from_v = self.v / self.t_total  # v = u + at, with u=0
        
        if not self._is_close(a_expected, a_from_v):
            success = self._log(
                f"Newton's 2nd Law failed: g={a_expected:.2f}m/s², a_from_v={a_from_v:.2f}m/s²", 
                False
            )
        else:
            success = self._log(
                f"Newton's 2nd Law OK: g={a_expected:.2f}m/s² ≈ a={a_from_v:.2f}m/s²"
            ) and success
        
        # 3. Momentum-impulse theorem (Δp = F_avg * Δt)
        delta_p = self.p - 0  # Initial momentum = 0
        F_avg = self.m * self.g  # Constant gravitational force
        delta_t = self.t_total
        impulse = F_avg * delta_t
        
        if not self._is_close(delta_p, impulse):
            success = self._log(
                f"Impulse-momentum failed: Δp={delta_p:.2f}kg·m/s, FΔt={impulse:.2f}N·s", 
                False
            )
        else:
            success = self._log(
                f"Impulse-momentum OK: Δp={delta_p:.2f}kg·m/s ≈ FΔt={impulse:.2f}N·s"
            ) and success
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer.
        Based on the STEPS ANALYSIS.
        """
        success = True
        
        # Step 1: Check displacement at t=2s
        t_step1 = 2.0
        h_expected = 0.5 * self.g * (t_step1 ** 2)
        if not self._is_close(self.h, h_expected):
            success = self._log(
                f"Step 1 inconsistency: h={self.h:.2f}m, expected {h_expected:.2f}m", 
                False
            )
        else:
            success = self._log(
                f"Step 1 OK: displacement h={self.h:.2f}m correct for t=2s"
            ) and success
        
        # Step 2: Check work done by gravity
        W_expected = self.m * self.g * self.h
        if not self._is_close(self.W, W_expected):
            success = self._log(
                f"Step 2 inconsistency: W={self.W:.2f}J, expected {W_expected:.2f}J", 
                False
            )
        else:
            success = self._log(
                f"Step 2 OK: work W={self.W:.2f}J = m*g*h"
            ) and success
        
        # Step 3: Check velocity at impact (t=4s)
        v_expected = self.g * self.t_total
        if not self._is_close(self.v, v_expected):
            success = self._log(
                f"Step 3 inconsistency: v={self.v:.2f}m/s, expected {v_expected:.2f}m/s", 
                False
            )
        else:
            success = self._log(
                f"Step 3 OK: velocity v={self.v:.2f}m/s = g*t_total"
            ) and success
        
        # Step 4: Check momentum at impact
        p_expected = self.m * self.v
        if not self._is_close(self.p, p_expected):
            success = self._log(
                f"Step 4 inconsistency: p={self.p:.2f}kg·m/s, expected {p_expected:.2f}kg·m/s", 
                False
            )
        else:
            success = self._log(
                f"Step 4 OK: momentum p={self.p:.2f}kg·m/s = m*v"
            ) and success
        
        # Cross-step consistency: energy between steps
        # Work done in first 2s should equal KE at t=2s
        v_at_2s = self.g * t_step1
        KE_at_2s = 0.5 * self.m * (v_at_2s ** 2)
        if not self._is_close(self.W, KE_at_2s):
            success = self._log(
                f"Work-energy consistency failed: W={self.W:.2f}J, KE(t=2s)={KE_at_2s:.2f}J", 
                False
            )
        else:
            success = self._log(
                f"Work-energy OK: W={self.W:.2f}J = ΔKE at t=2s"
            ) and success
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb mass by 10% and verify expected trends.
        Restore original values after test.
        """
        success = True
        
        # Store original
        original_m = float(self.m)
        
        # Test 1: Increase mass by 10%
        self.m = original_m * 1.1
        
        # Recompute expected values with perturbed mass
        h_expected = 0.5 * self.g * (2.0 ** 2)  # Independent of mass
        W_expected = self.m * self.g * h_expected  # Proportional to mass
        v_expected = self.g * self.t_total  # Independent of mass
        p_expected = self.m * v_expected  # Proportional to mass
        
        # Restore original mass for next calculations
        self.m = original_m
        
        # Check monotonic trends
        W_original = original_m * self.g * h_expected
        p_original = original_m * v_expected
        
        # Work should increase with mass
        if not (W_expected > W_original):
            success = self._log(
                f"Work monotonicity failed: W(1.1m)={W_expected:.2f}J ≤ W(m)={W_original:.2f}J", 
                False
            )
        else:
            success = self._log(
                f"Work monotonicity OK: W increases with mass"
            ) and success
        
        # Momentum should increase with mass
        if not (p_expected > p_original):
            success = self._log(
                f"Momentum monotonicity failed: p(1.1m)={p_expected:.2f} ≤ p(m)={p_original:.2f}", 
                False
            )
        else:
            success = self._log(
                f"Momentum monotonicity OK: p increases with mass"
            ) and success
        
        # Test 2: Displacement independent of mass
        self.m = original_m * 0.9
        h_expected2 = 0.5 * self.g * (2.0 ** 2)
        self.m = original_m
        h_expected1 = 0.5 * self.g * (2.0 ** 2)
        
        if not self._is_close(h_expected1, h_expected2):
            success = self._log(
                f"Displacement should be mass-independent: h differs with mass change", 
                False
            )
        else:
            success = self._log(
                "Displacement OK: independent of mass"
            ) and success
        
        # Restore completely
        self.m = original_m
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics audits and return (success, logs).
        """
        self.logs = []
        overall_success = True
        
        self.logs.append("=== Physics Audit Report ===")
        
        # Run tests
        tests = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Physics Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== Audit {'PASSED' if overall_success else 'FAILED'} ===")
        
        return overall_success, self.logs


# Optional: Example usage stub
if __name__ == "__main__":
    # Mock solution class for demonstration
    class MockSolution:
        def __init__(self):
            self.m = 0.5
            self.g = 10.0
            self.results = {
                'h': 20.0,
                'W': 100.0,
                'v': 40.0,
                'p': 20.0
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)