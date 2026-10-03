import math
import sympy as sp
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks
        self.t = getattr(solution_instance, 't', 
                        getattr(solution_instance, 'inputs', {}).get('t', 5.0))
        self.g = getattr(solution_instance, 'g', 
                        getattr(solution_instance, 'inputs', {}).get('g', 9.8))
        
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {})
        self.h = self.results.get('h', getattr(solution_instance, 'h', 0.0))
        self.v = self.results.get('v', getattr(solution_instance, 'v', 0.0))
        
        # Store steps if available
        self.steps = getattr(solution_instance, 'steps', None)
        
        # Symbolic variables for invariant checking
        self.t_sym = sp.Symbol('t')
        self.g_sym = sp.Symbol('g')
        self.v0_sym = sp.Symbol('v0')
        self.v1_sym = sp.Symbol('v1')
        self.delta_y_sym = sp.Symbol('delta_y')
        
        # KB formulas
        self.kb_formulas = [
            sp.Eq(self.v1_sym, self.v0_sym + self.g_sym * self.t_sym),
            sp.Eq(self.delta_y_sym, ((self.v0_sym + self.v1_sym) / 2) * self.t_sym),
            sp.Eq(self.delta_y_sym, self.v0_sym * self.t_sym + (self.g_sym * self.t_sym**2) / 2),
            sp.Eq(self.v1_sym**2, self.v0_sym**2 + 2 * self.g_sym * self.delta_y_sym)
        ]
        
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
        
    def test_invariants(self):
        """Verify energy conservation and kinematic laws"""
        success = True
        
        # Check displacement formula: h = 0.5*g*t² (v0=0)
        expected_h = 0.5 * self.g * self.t**2
        if not math.isclose(self.h, expected_h, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Displacement mismatch: h={self.h:.2f}, expected {expected_h:.2f} "
                f"(0.5*{self.g:.1f}*{self.t:.1f}²)", 
                False
            )
        else:
            self._log(f"Displacement correct: h={self.h:.2f} m", True)
        
        # Check velocity formula: v = g*t (v0=0)
        expected_v = self.g * self.t
        if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Velocity mismatch: v={self.v:.2f}, expected {expected_v:.2f} "
                f"({self.g:.1f}*{self.t:.1f})", 
                False
            )
        else:
            self._log(f"Velocity correct: v={self.v:.2f} m/s", True)
        
        # Check energy conservation: v² = 2*g*h (v0=0)
        v_squared = self.v**2
        expected_v_squared = 2 * self.g * self.h
        if not math.isclose(v_squared, expected_v_squared, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Energy conservation mismatch: v²={v_squared:.2f}, "
                f"2*g*h={expected_v_squared:.2f}", 
                False
            )
        else:
            self._log(f"Energy conservation: v² = 2*g*h holds", True)
        
        # Check kinematic consistency using KB formulas
        try:
            # Test formula 3 with v0=0
            formula_check = self.delta_y_sym - (self.v0_sym * self.t_sym + 
                                               (self.g_sym * self.t_sym**2) / 2)
            numeric_check = self.h - (0 + (self.g * self.t**2) / 2)
            if not math.isclose(numeric_check, 0, rel_tol=0.02, abs_tol=0.02):
                success = self._log(
                    f"Kinematic formula mismatch: h - 0.5*g*t² = {numeric_check:.2f} ≠ 0", 
                    False
                )
            else:
                self._log("Kinematic formula h = 0.5*g*t² validated", True)
        except Exception as e:
            self._log(f"Symbolic check failed: {str(e)}", False)
            success = False
            
        return success
        
    def test_trace_consistency(self):
        """Check intermediate steps against final results"""
        success = True
        
        if self.steps:
            # Check step 1: displacement calculation
            step1_val = None
            for step_name, step_data in self.steps.items():
                if 'step_1' in step_name:
                    for quantity in step_data.get('result_quantity', []):
                        if quantity.get('symbol') == 'h':
                            step1_val = float(quantity.get('value', 0))
                            break
            
            if step1_val is not None:
                if not math.isclose(step1_val, self.h, rel_tol=0.02, abs_tol=0.02):
                    success = self._log(
                        f"Step 1 mismatch: step_h={step1_val:.2f}, final_h={self.h:.2f}", 
                        False
                    )
                else:
                    self._log(f"Step 1 consistent: h={step1_val:.2f} m", True)
            
            # Check step 3: velocity calculation
            step3_val = None
            for step_name, step_data in self.steps.items():
                if 'step_3' in step_name:
                    for quantity in step_data.get('result_quantity', []):
                        if quantity.get('symbol') == 'v':
                            step3_val = float(quantity.get('value', 0))
                            break
            
            if step3_val is not None:
                if not math.isclose(step3_val, self.v, rel_tol=0.02, abs_tol=0.02):
                    success = self._log(
                        f"Step 3 mismatch: step_v={step3_val:.2f}, final_v={self.v:.2f}", 
                        False
                    )
                else:
                    self._log(f"Step 3 consistent: v={step3_val:.2f} m/s", True)
        else:
            self._log("No steps available for trace consistency check", True)
            
        return success
        
    def test_monotonicity(self):
        """Verify that increasing time increases displacement and velocity"""
        success = True
        
        # Store original values
        original_t = self.t
        
        # Perturb time by +10%
        perturbed_t = original_t * 1.1
        
        # Calculate expected values with perturbed time
        expected_h_perturbed = 0.5 * self.g * perturbed_t**2
        expected_v_perturbed = self.g * perturbed_t
        
        # Check monotonic increase
        if not expected_h_perturbed > self.h:
            success = self._log(
                f"Non-monotonic displacement: h({perturbed_t:.2f}s)={expected_h_perturbed:.2f} "
                f"≤ h({original_t:.2f}s)={self.h:.2f}", 
                False
            )
        else:
            self._log(
                f"Displacement monotonic: h increases with time "
                f"({self.h:.2f} → {expected_h_perturbed:.2f})", 
                True
            )
            
        if not expected_v_perturbed > self.v:
            success = self._log(
                f"Non-monotonic velocity: v({perturbed_t:.2f}s)={expected_v_perturbed:.2f} "
                f"≤ v({original_t:.2f}s)={self.v:.2f}", 
                False
            )
        else:
            self._log(
                f"Velocity monotonic: v increases with time "
                f"({self.v:.2f} → {expected_v_perturbed:.2f})", 
                True
            )
            
        # Restore original time (not strictly needed as we don't modify instance)
        self.t = original_t
        
        return success
        
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        test_results = []
        
        # Run tests
        test_results.append(("Invariants", self.test_invariants()))
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        test_results.append(("Monotonicity", self.test_monotonicity()))
        
        # Summary
        all_passed = all(result for _, result in test_results)
        
        self.logs.append("\n=== Summary ===")
        for test_name, passed in test_results:
            self.logs.append(f"{test_name}: {'PASS' if passed else 'FAIL'}")
        
        self.logs.append(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        return all_passed, self.logs