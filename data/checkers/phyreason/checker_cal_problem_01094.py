import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.H = getattr(solution_instance, 'H', 47.0)
        self.L1 = getattr(solution_instance, 'L1', 8.0)
        self.h1 = getattr(solution_instance, 'h1', 2.0)
        self.v0 = getattr(solution_instance, 'v0', 9.0)
        self.L2 = getattr(solution_instance, 'L2', 24.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Reaction time from steps analysis (Δt = 1s, derived from x_2 = 9m)
        self.delta_t = getattr(solution_instance, 'delta_t', 1.0)
        
        # Safe extraction of results
        self.results = getattr(solution_instance, 'results', {})
        self.h = self.results.get('h', 0.0)
        self.t = self.results.get('t', 0.0)
        self.x = self.results.get('x', 0.0)
        self.x_2 = self.results.get('x_2', 0.0)
        self.d = self.results.get('d', 0.0)
        self.a = self.results.get('a', 0.0)
        
        self.logs = []
        
    def _log(self, message: str, success: bool = True):
        """Helper for logging test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy comparison for physics quantities."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Verify physics invariants: energy conservation, kinematic laws."""
        success = True
        
        # 1. Free-fall time invariant: t = sqrt(2h/g)
        h_calc = self.H - self.h1
        t_calc = math.sqrt(2 * h_calc / self.g)
        if not self._log(f"Free-fall time: calc={t_calc:.2f}, result={self.t:.2f}", 
                        self._is_close(self.t, t_calc)):
            success = False
        
        # 2. Energy conservation for flowerpot: v_final^2 = 2*g*h
        v_final = self.g * self.t
        energy_check = v_final**2 - 2 * self.g * h_calc
        if not self._log(f"Energy conservation: v_final²-2gh = {energy_check:.2f} ≈ 0",
                        self._is_close(energy_check, 0.0)):
            success = False
            
        # 3. Uniform motion during reaction: x_2 = v0 * Δt
        x2_calc = self.v0 * self.delta_t
        if not self._log(f"Reaction displacement: calc={x2_calc:.2f}, result={self.x_2:.2f}",
                        self._is_close(self.x_2, x2_calc)):
            success = False
            
        # 4. Distance after reaction: d = L2 - x_2
        d_calc = self.L2 - self.x_2
        if not self._log(f"Distance after reaction: calc={d_calc:.2f}, result={self.d:.2f}",
                        self._is_close(self.d, d_calc)):
            success = False
            
        # 5. Acceleration equation symbolic check
        t_remaining = self.t - self.delta_t
        # Equation: d + L1 = v0*t_remaining + 0.5*a*t_remaining²
        a_calc = 2 * (self.d + self.L1 - self.v0 * t_remaining) / (t_remaining**2) if t_remaining > 0 else 0
        if not self._log(f"Acceleration from equation: calc={a_calc:.2f}, result={self.a:.2f}",
                        self._is_close(self.a, a_calc)):
            success = False
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer."""
        success = True
        
        # Check step-by-step consistency from steps analysis
        
        # Step 1: h = H - h1 = 45
        h_step = self.H - self.h1
        if not self._log(f"Step1: h = H - h1 = {h_step:.2f}", 
                        self._is_close(h_step, 45.0)):
            success = False
            
        # Step 3: t = sqrt(2h/g) = 3
        t_step = math.sqrt(2 * h_step / self.g)
        if not self._log(f"Step3: t = sqrt(2h/g) = {t_step:.2f}",
                        self._is_close(t_step, 3.0)):
            success = False
            
        # Step 4: x = v0 * t = 27
        x_step = self.v0 * t_step
        if not self._log(f"Step4: x = v0*t = {x_step:.2f}",
                        self._is_close(x_step, 27.0)):
            success = False
            
        # Step 6: x_2 = v0 * Δt = 9
        x2_step = self.v0 * self.delta_t
        if not self._log(f"Step6: x_2 = v0*Δt = {x2_step:.2f}",
                        self._is_close(x2_step, 9.0)):
            success = False
            
        # Step 7: d = L2 - x_2 = 15
        d_step = self.L2 - x2_step
        if not self._log(f"Step7: d = L2 - x_2 = {d_step:.2f}",
                        self._is_close(d_step, 15.0)):
            success = False
            
        # Step 9: a from d+L1 = v0(t-Δt) + 0.5*a*(t-Δt)²
        t_rem = t_step - self.delta_t
        a_step = 2 * (d_step + self.L1 - self.v0 * t_rem) / (t_rem**2) if t_rem > 0 else 0
        if not self._log(f"Step9: a = {a_step:.2f} m/s²",
                        self._is_close(a_step, 2.5)):
            success = False
            
        # Final answer cross-check
        if not self._log(f"Final acceleration matches step analysis: {self.a:.2f} vs 2.50",
                        self._is_close(self.a, 2.5)):
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb input parameters and check expected trends."""
        success = True
        original_delta_t = self.delta_t
        
        try:
            # Perturb reaction time (increase by 10%)
            perturbed_delta_t = original_delta_t * 1.10
            
            # Recalculate chain of dependencies
            perturbed_x2 = self.v0 * perturbed_delta_t
            perturbed_d = self.L2 - perturbed_x2
            perturbed_t_rem = self.t - perturbed_delta_t
            
            # Ensure positive time remaining
            if perturbed_t_rem <= 0:
                self._log("Monotonicity: Perturbed reaction time too large, skipping", True)
                return True
                
            perturbed_a = 2 * (perturbed_d + self.L1 - self.v0 * perturbed_t_rem) / (perturbed_t_rem**2)
            
            # Expectation: Longer reaction time → less time to accelerate → higher required acceleration
            if perturbed_delta_t > original_delta_t:
                if perturbed_a > self.a:
                    self._log(f"Monotonicity: Δt↑ ({original_delta_t:.2f}→{perturbed_delta_t:.2f}) ⇒ a↑ ({self.a:.2f}→{perturbed_a:.2f})", True)
                else:
                    self._log(f"Monotonicity FAIL: Δt↑ but a↓ ({self.a:.2f}→{perturbed_a:.2f})", False)
                    success = False
                    
            # Test opposite perturbation (decrease reaction time)
            perturbed_delta_t2 = original_delta_t * 0.90
            perturbed_x2_2 = self.v0 * perturbed_delta_t2
            perturbed_d_2 = self.L2 - perturbed_x2_2
            perturbed_t_rem_2 = self.t - perturbed_delta_t2
            perturbed_a_2 = 2 * (perturbed_d_2 + self.L1 - self.v0 * perturbed_t_rem_2) / (perturbed_t_rem_2**2)
            
            if perturbed_delta_t2 < original_delta_t:
                if perturbed_a_2 < self.a:
                    self._log(f"Monotonicity: Δt↓ ({original_delta_t:.2f}→{perturbed_delta_t2:.2f}) ⇒ a↓ ({self.a:.2f}→{perturbed_a_2:.2f})", True)
                else:
                    self._log(f"Monotonicity FAIL: Δt↓ but a↑ ({self.a:.2f}→{perturbed_a_2:.2f})", False)
                    success = False
                    
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", False)
            success = False
        finally:
            # Restore original value
            self.delta_t = original_delta_t
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success."""
        self.logs.clear()
        self.logs.append("=== PhysicsChecker Started ===")
        
        overall_success = True
        
        # Run individual test suites
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.logs.append(f"\n--- {name} Test ---")
            if not test_func():
                overall_success = False
                
        self.logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs