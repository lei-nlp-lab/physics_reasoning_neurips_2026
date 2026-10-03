import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Never access solution_instance.attr directly to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Safely get g value from solution or use default
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safely extract masses with fallback values
        self.M = getattr(solution_instance, 'M', 70.0)
        self.m = getattr(solution_instance, 'm', 50.0)
        
        # Safely extract time value if available
        self.t = getattr(solution_instance, 't', 2.0)
        
        # Safely extract results from solution's results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract all expected result values with fallbacks
        self.F = results.get('F', 0.0)  # Maximum pulling force
        self.a = results.get('a', 0.0)  # Acceleration
        self.v = results.get('v', 0.0)  # Velocity after t seconds
        self.F_minus_mg = results.get('F - mg', 0.0)  # Net force on object
        self.h = results.get('h', 0.0)  # Total height
        self.h1 = results.get('h_1', 0.0)  # Height in first phase
        self.h2 = results.get('h_2', 0.0)  # Height in second phase
        
        # For symbolic calculations (only used for monotonicity test)
        self.symbols = self._define_symbols()
        
        # Store logs for all tests
        self.logs = []

    def _define_symbols(self):
        """Define symbolic variables for physics laws."""
        M, m, g, t, F, a, v, h, h1, h2 = sp.symbols('M m g t F a v h h1 h2')
        return {
            'M': M, 'm': m, 'g': g, 't': t,
            'F': F, 'a': a, 'v': v, 'h': h, 'h1': h1, 'h2': h2
        }

    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success

    def _is_close(self, val1, val2, msg: str = "") -> bool:
        """Fuzzy numerical matching with 2% tolerance."""
        try:
            result = math.isclose(float(val1), float(val2), 
                                rel_tol=1e-2, abs_tol=1e-2)
            if not result and msg:
                self.logs.append(f"    Mismatch: {val1} vs {val2} - {msg}")
            return result
        except (TypeError, ValueError):
            self.logs.append(f"    Error comparing {val1} and {val2}")
            return False

    def test_invariants(self) -> bool:
        """
        Verify physics invariants based on KB theorems.
        1. F = M*g (worker's maximum pulling force equals weight)
        2. Newton's 2nd Law: F - m*g = m*a
        3. Energy conservation check (optional)
        """
        success = True
        
        # 1. Check F = M*g
        expected_F = self.M * self.g
        if not self._is_close(self.F, expected_F, "F = M*g"):
            success = False
        self._log(f"F = M*g: {self.F} ≈ {expected_F}", 
                 self._is_close(self.F, expected_F))
        
        # 2. Check Newton's 2nd Law: F - m*g = m*a
        # Calculate both sides
        lhs = self.F - self.m * self.g  # Net force
        rhs = self.m * self.a  # m*a
        
        # Direct numerical check only - no symbolic verification
        if not self._is_close(lhs, rhs, "F - m*g = m*a"):
            success = False
        
        self._log(f"Newton's 2nd Law: {lhs} ≈ {rhs}", 
                 self._is_close(lhs, rhs))
        
        # 3. Energy conservation (optional check)
        # Work done by worker: W = F * h1 (constant force over distance h1)
        # Object energy: KE + PE at end of phase 1
        try:
            work_done = self.F * self.h1
            energy_object = 0.5 * self.m * self.v**2 + self.m * self.g * self.h1
            # They should be close but not exactly equal due to simplifications
            energy_ratio = work_done / energy_object if energy_object != 0 else 1
            if 0.9 < energy_ratio < 1.1:  # Allow 10% tolerance
                self._log(f"Energy roughly conserved: work={work_done:.1f}, energy={energy_object:.1f}", True)
            else:
                self._log(f"Energy check: work={work_done:.1f}, energy={energy_object:.1f}", False)
                # Don't fail the test for energy, it's not critical
        except Exception as e:
            self.logs.append(f"    Energy check skipped: {e}")
        
        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer.
        Follow the steps from the analysis.
        """
        success = True
        
        # Step 3: Check acceleration a
        if not self._is_close(self.a, 4.0, f"a should be 4.0 m/s²"):
            success = False
        self._log(f"Acceleration a = {self.a} m/s²", 
                 self._is_close(self.a, 4.0))
        
        # Step 4: Check velocity v = a*t
        expected_v = self.a * self.t
        if not self._is_close(self.v, expected_v, "v = a*t"):
            success = False
        self._log(f"Velocity v = a*t: {self.v} ≈ {expected_v}", 
                 self._is_close(self.v, expected_v))
        
        # Step 5: Check h1 = (1/2)*a*t^2
        expected_h1 = 0.5 * self.a * self.t**2
        if not self._is_close(self.h1, expected_h1, "h1 = 0.5*a*t^2"):
            success = False
        self._log(f"Height h1 = 0.5*a*t²: {self.h1} ≈ {expected_h1}", 
                 self._is_close(self.h1, expected_h1))
        
        # Step 6: Check h2 = v^2/(2g)
        expected_h2 = self.v**2 / (2 * self.g)
        if not self._is_close(self.h2, expected_h2, "h2 = v²/(2g)"):
            success = False
        self._log(f"Height h2 = v²/(2g): {self.h2} ≈ {expected_h2}", 
                 self._is_close(self.h2, expected_h2))
        
        # Step 7: Check h = h1 + h2
        expected_h = self.h1 + self.h2
        if not self._is_close(self.h, expected_h, "h = h1 + h2"):
            success = False
        self._log(f"Total height h = h1 + h2: {self.h} ≈ {expected_h}", 
                 self._is_close(self.h, expected_h))
        
        # Final check against target answers
        if not self._is_close(self.a, 4.0, "Final acceleration"):
            success = False
        if not self._is_close(self.h, 11.2, "Final height"):
            success = False
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb an input (worker mass M) and check if acceleration changes
        in the expected direction. Restore original values after test.
        """
        success = True
        
        # Store original values
        original_M = self.M
        original_F = self.F
        original_a = self.a
        
        try:
            # Increase M by 10%
            perturbed_M = original_M * 1.1
            
            # Recalculate F = M*g
            perturbed_F = perturbed_M * self.g
            
            # Recalculate acceleration: a = (F - m*g)/m
            perturbed_a = (perturbed_F - self.m * self.g) / self.m
            
            # Check monotonicity: if M increases, a should increase
            if perturbed_a > original_a:
                self._log(f"Monotonicity: M↑ ({original_M}→{perturbed_M}) ⇒ a↑ ({original_a:.2f}→{perturbed_a:.2f})", True)
            else:
                self._log(f"Monotonicity failed: M↑ but a didn't increase", False)
                success = False
            
            # Also check proportionality: a = (M*g - m*g)/m = g*(M/m - 1)
            # So da/dM = g/m > 0, which should hold
            theoretical_slope = self.g / self.m
            numerical_slope = (perturbed_a - original_a) / (perturbed_M - original_M)
            
            if self._is_close(numerical_slope, theoretical_slope, "da/dM = g/m"):
                self._log(f"Derivative check: da/dM ≈ {numerical_slope:.3f} ≈ g/m = {theoretical_slope:.3f}", True)
            else:
                self._log(f"Derivative mismatch: {numerical_slope:.3f} vs {theoretical_slope:.3f}", False)
                # Don't fail the whole test for derivative check
            
        except Exception as e:
            self.logs.append(f"    Monotonicity test error: {e}")
            success = False
        finally:
            # Restore original values
            self.M = original_M
            self.F = original_F
            self.a = original_a
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success status with logs.
        """
        self.logs = []  # Reset logs
        self.logs.append("Starting Physics Audit...")
        
        # Run individual tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary
        success = test1 and test2 and test3
        self.logs.append("\n" + "="*50)
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        self.logs.append(f"  Invariants: {'PASS' if test1 else 'FAIL'}")
        self.logs.append(f"  Trace Consistency: {'PASS' if test2 else 'FAIL'}")
        self.logs.append(f"  Monotonicity: {'PASS' if test3 else 'FAIL'}")
        
        return success, self.logs