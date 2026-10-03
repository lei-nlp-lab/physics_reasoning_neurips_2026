import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Inputs with defaults from problem statement
        self.v0 = getattr(solution_instance, 'v0', 30.0)
        self.vt = getattr(solution_instance, 'vt', 6.0)
        self.a1 = getattr(solution_instance, 'a1', 4.0)
        
        # Results with safe access
        results = getattr(solution_instance, 'results', {})
        self.x = results.get('x', None)
        self.t = results.get('t', None)
        self.t_prime = results.get("t'", results.get('t\'', None))  # t'
        self.t2 = results.get('t_2', None)
        self.delta_t = results.get('Deltat', results.get('Δt', None))
        self.x2 = results.get('x_2', None)
        self.t1 = results.get('t_1', None)
        self.x1 = results.get('x_1', None)
        
        # Target answers for comparison
        self.targets = [108.0, 10.0, 4.0]
        
        # SymPy symbols for symbolic checks
        self.v0_sym, self.vt_sym, self.a1_sym = sp.symbols('v0 vt a1', positive=True)
        self.x1_sym, self.t1_sym, self.x2_sym, self.t2_sym = sp.symbols('x1 t1 x2 t2', positive=True)
        
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical matching with physics tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify kinematic conservation laws."""
        logs = []
        success = True
        
        # 1. Check deceleration phase (x1 calculation)
        if self.x1 is not None:
            # Symbolic: v_t^2 = v_0^2 + 2*a*x1, where a = -a1
            eq1 = sp.Eq(self.vt_sym**2, self.v0_sym**2 + 2*(-self.a1_sym)*self.x1_sym)
            lhs_num = self.vt**2
            rhs_num = self.v0**2 + 2*(-self.a1)*self.x1
            
            if not self._is_close(lhs_num, rhs_num):
                logs.append(f"Deceleration invariant failed: vt²={lhs_num:.2f}, v0²+2(-a1)x1={rhs_num:.2f}")
                success = False
        
        # 2. Check acceleration phase (x2 calculation if available)
        if self.x2 is not None:
            # Need a2 from step_4 and step_6
            # From step_4: t2 = (v0 - vt)/a2 => a2 = (v0 - vt)/t2
            # From step_6: v0² - vt² = 2*a2*x2
            if self.t2 is not None:
                a2_calc = (self.v0 - self.vt) / self.t2 if self.t2 != 0 else 0
                lhs_acc = self.v0**2 - self.vt**2
                rhs_acc = 2 * a2_calc * self.x2
                
                if not self._is_close(lhs_acc, rhs_acc):
                    logs.append(f"Acceleration invariant failed: v0²-vt²={lhs_acc:.2f}, 2*a2*x2={rhs_acc:.2f}")
                    success = False
        
        # 3. Time addition consistency
        if self.t1 is not None and self.t2 is not None and self.t is not None:
            total_t = self.t1 + self.t2
            if not self._is_close(total_t, self.t):
                logs.append(f"Time addition failed: t1+t2={total_t:.2f}, t={self.t:.2f}")
                success = False
        
        # 4. Displacement addition consistency
        if self.x1 is not None and self.x2 is not None and self.x is not None:
            total_x = self.x1 + self.x2
            if not self._is_close(total_x, self.x):
                logs.append(f"Displacement addition failed: x1+x2={total_x:.2f}, x={self.x:.2f}")
                success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate and final results."""
        logs = []
        success = True
        
        # Check step_1: x1 from v_t^2 - v_0^2 = 2*a1*x1 (with a = -a1)
        if self.x1 is not None:
            x1_calc = (self.vt**2 - self.v0**2) / (2 * (-self.a1))
            if not self._is_close(x1_calc, self.x1):
                logs.append(f"Trace: x1 inconsistent: calc={x1_calc:.2f}, result={self.x1:.2f}")
                success = False
        
        # Check step_3: t1 = (vt - v0)/(-a1)
        if self.t1 is not None:
            t1_calc = (self.vt - self.v0) / (-self.a1)
            if not self._is_close(t1_calc, self.t1):
                logs.append(f"Trace: t1 inconsistent: calc={t1_calc:.2f}, result={self.t1:.2f}")
                success = False
        
        # Check step_4: t2 = (v0 - vt)/a2, but need a2
        # From step_6: a2 = (v0² - vt²)/(2*x2)
        if self.x2 is not None and self.t2 is not None:
            a2_calc = (self.v0**2 - self.vt**2) / (2 * self.x2) if self.x2 != 0 else 0
            t2_calc = (self.v0 - self.vt) / a2_calc if a2_calc != 0 else 0
            if not self._is_close(t2_calc, self.t2):
                logs.append(f"Trace: t2 inconsistent: calc={t2_calc:.2f}, result={self.t2:.2f}")
                success = False
        
        # Check step_8: t' = (x1 + x2)/v0
        if self.t_prime is not None and self.x1 is not None and self.x2 is not None:
            t_prime_calc = (self.x1 + self.x2) / self.v0
            if not self._is_close(t_prime_calc, self.t_prime):
                logs.append(f"Trace: t' inconsistent: calc={t_prime_calc:.2f}, result={self.t_prime:.2f}")
                success = False
        
        # Check step_9: Δt = t - t'
        if self.delta_t is not None and self.t is not None and self.t_prime is not None:
            delta_t_calc = self.t - self.t_prime
            if not self._is_close(delta_t_calc, self.delta_t):
                logs.append(f"Trace: Δt inconsistent: calc={delta_t_calc:.2f}, result={self.delta_t:.2f}")
                success = False
        
        # Final answer check against targets
        if self.x1 is not None:
            if not self._is_close(self.x1, self.targets[0]):
                logs.append(f"Target x1 mismatch: {self.x1:.2f} vs {self.targets[0]:.2f}")
                success = False
        
        if self.t is not None:
            if not self._is_close(self.t, self.targets[1]):
                logs.append(f"Target t mismatch: {self.t:.2f} vs {self.targets[1]:.2f}")
                success = False
        
        if self.delta_t is not None:
            if not self._is_close(self.delta_t, self.targets[2]):
                logs.append(f"Target Δt mismatch: {self.delta_t:.2f} vs {self.targets[2]:.2f}")
                success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb inputs and verify physical trends."""
        logs = []
        success = True
        
        # Perturb deceleration a1 by +10%
        original_a1 = self.a1
        perturbed_a1 = original_a1 * 1.10
        
        # Recalculate x1 with perturbed a1 using kinematic formula
        x1_original = (self.vt**2 - self.v0**2) / (2 * (-original_a1))
        x1_perturbed = (self.vt**2 - self.v0**2) / (2 * (-perturbed_a1))
        
        # Trend: Larger deceleration magnitude (more negative a) → smaller braking distance
        if not (x1_perturbed < x1_original):
            logs.append(f"Monotonicity failed: x1 should decrease with increased deceleration")
            logs.append(f"  Original a1={original_a1:.2f}, x1={x1_original:.2f}")
            logs.append(f"  Perturbed a1={perturbed_a1:.2f}, x1={x1_perturbed:.2f}")
            success = False
        
        # Restore original value implicitly (we only used local copies)
        
        # Perturb initial velocity v0 by +5%
        original_v0 = self.v0
        perturbed_v0 = original_v0 * 1.05
        
        # Recalculate braking distance x1 with perturbed v0
        x1_v0_original = (self.vt**2 - original_v0**2) / (2 * (-original_a1))
        x1_v0_perturbed = (self.vt**2 - perturbed_v0**2) / (2 * (-original_a1))
        
        # Trend: Higher initial velocity → larger braking distance (vt fixed)
        if not (x1_v0_perturbed > x1_v0_original):
            logs.append(f"Monotonicity failed: x1 should increase with increased v0")
            logs.append(f"  Original v0={original_v0:.2f}, x1={x1_v0_original:.2f}")
            logs.append(f"  Perturbed v0={perturbed_v0:.2f}, x1={x1_v0_perturbed:.2f}")
            success = False
        
        if success:
            logs.append("Monotonicity tests passed: trends match physical expectations")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all audit tests."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs