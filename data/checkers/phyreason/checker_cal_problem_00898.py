import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.logs = []
        
        # Try multiple extraction strategies
        self.U = self._extract_value('U', 6.0)
        self.R = self._extract_value('R', 2.0)
        self.R1 = self._extract_value('R1', 6.0)
        
        self.R_L = self._extract_value('R_L', None)
        self.I_0 = self._extract_value('I_0', None)
        self.E = self._extract_value('E', None)
        self.I = self._extract_value('I', None)
        
        # For direction, we need to be more flexible
        self.direction = self._extract_direction()
    
    def _extract_value(self, name, default):
        """Extract a numeric value using multiple strategies."""
        # Strategy 1: Try to get from results dict
        results_dict = getattr(self.solution, 'results', {})
        if name in results_dict:
            val = results_dict[name]
            if isinstance(val, (int, float)):
                return float(val)
            elif hasattr(val, 'evalf'):
                return float(val.evalf())
        
        # Strategy 2: Try to get as attribute
        val = getattr(self.solution, name, None)
        if val is not None:
            if isinstance(val, (int, float)):
                return float(val)
            elif hasattr(val, 'evalf'):
                return float(val.evalf())
        
        # Strategy 3: Try common alternative names
        alt_names = {
            'U': ['U_val', 'u'],
            'R': ['R_val', 'r'],
            'R1': ['R1_val', 'r1'],
            'R_L': ['R_L_val'],
            'I_0': ['I_0_val'],
            'E': ['E_val'],
            'I': ['I_val']
        }
        
        if name in alt_names:
            for alt_name in alt_names[name]:
                # Try as attribute
                val = getattr(self.solution, alt_name, None)
                if val is not None:
                    if isinstance(val, (int, float)):
                        return float(val)
                    elif hasattr(val, 'evalf'):
                        return float(val.evalf())
                
                # Try in results dict
                if alt_name in results_dict:
                    val = results_dict[alt_name]
                    if isinstance(val, (int, float)):
                        return float(val)
                    elif hasattr(val, 'evalf'):
                        return float(val.evalf())
        
        return default
    
    def _extract_direction(self):
        """Extract direction statement using multiple strategies."""
        results_dict = getattr(self.solution, 'results', {})
        
        # Check for the exact key used in the generated code
        if 'lamp_current_direction' in results_dict:
            val = results_dict['lamp_current_direction']
            if isinstance(val, str):
                return val
        
        # Also check for other possible keys
        direction_keys = ['direction', 'current_direction', 'lamp_direction']
        for key in direction_keys:
            if key in results_dict:
                val = results_dict[key]
                if isinstance(val, str):
                    return val
        
        # Strategy 2: Try to get as attribute
        direction = getattr(self.solution, 'direction', None)
        if isinstance(direction, str):
            return direction
        
        # Strategy 3: Try non_numeric_results if it exists
        non_numeric = getattr(self.solution, 'non_numeric_results', None)
        if isinstance(non_numeric, dict):
            for key in ['direction', 'lamp_current_direction']:
                if key in non_numeric:
                    val = non_numeric[key]
                    if isinstance(val, str):
                        return val
        
        return ''  # Return empty string if not found
    
    def _log(self, message: str):
        self.logs.append(message)

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Safe numeric comparison."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self) -> bool:
        success = True
        
        # 1. Ohm's law for initial steady state: U = I_0 * (R_L + R)
        if None in (self.I_0, self.R_L):
            self._log("Missing I_0 or R_L for initial Ohm's law check.")
            success = False
        else:
            expected_U = self.I_0 * (self.R_L + self.R)
            if not self._is_close(self.U, expected_U):
                self._log(f"Initial Ohm's law violation: U={self.U}, I_0*(R_L+R)={expected_U}")
                success = False
            else:
                self._log("Initial Ohm's law satisfied.")
        
        # 2. Ohm's law for induced circuit: E = I * (R_L + R + R1)
        if None in (self.E, self.I, self.R_L):
            self._log("Missing E, I, or R_L for induced circuit Ohm's law check.")
            success = False
        else:
            expected_E = self.I * (self.R_L + self.R + self.R1)
            if not self._is_close(self.E, expected_E):
                self._log(f"Induced circuit Ohm's law violation: E={self.E}, I*(R_L+R+R1)={expected_E}")
                success = False
            else:
                self._log("Induced circuit Ohm's law satisfied.")
        
        # 3. Direction consistency with Lenz's law
        expected_direction = "The direction of the current in the electric lamp is from right to left."
        # Normalize strings: lower case, strip extra spaces
        if not isinstance(self.direction, str):
            self._log(f"Direction is not a string: {type(self.direction)}")
            success = False
        elif self.direction.strip().lower() != expected_direction.strip().lower():
            self._log(f"Direction mismatch. Expected: '{expected_direction}', Got: '{self.direction}'")
            success = False
        else:
            self._log("Direction statement matches expected.")
        
        return success

    def test_trace_consistency(self) -> bool:
        success = True
        
        # Step 1 and Step 3: R_L = U / I_0 - R
        if None in (self.I_0, self.R_L):
            self._log("Missing I_0 or R_L for trace consistency.")
            success = False
        else:
            # Protect against division by zero
            if abs(self.I_0) < 1e-10:
                self._log("I_0 is too close to zero for R_L calculation.")
                success = False
            else:
                computed_R_L = self.U / self.I_0 - self.R
                if not self._is_close(self.R_L, computed_R_L):
                    self._log(f"Trace inconsistency in R_L: From steps: {computed_R_L}, Result: {self.R_L}")
                    success = False
                else:
                    self._log(f"R_L consistent with steps: {self.R_L} Ω")
        
        # Step 5 and Step 6: E = I * (R_L + R + R1)
        if None in (self.I, self.E, self.R_L):
            self._log("Missing I, E, or R_L for trace consistency.")
            success = False
        else:
            computed_E = self.I * (self.R_L + self.R + self.R1)
            if not self._is_close(self.E, computed_E):
                self._log(f"Trace inconsistency in E: From steps: {computed_E}, Result: {self.E}")
                success = False
            else:
                self._log(f"E consistent with steps: {self.E} V")
        
        return success

    def test_monotonicity(self) -> bool:
        success = True
        
        # Perturb I_0 by +1% and check R_L trend
        if None in (self.I_0, self.R_L):
            self._log("Missing I_0 or R_L for monotonicity check (I_0 perturbation).")
            success = False
        else:
            # Protect against division by zero
            if abs(self.I_0) < 1e-10:
                self._log("I_0 is too close to zero for monotonicity check.")
                success = False
            else:
                perturbed_I_0 = self.I_0 * 1.01
                new_R_L = self.U / perturbed_I_0 - self.R
                # Expected: if I_0 increases, R_L should decrease
                if not new_R_L < self.R_L:
                    self._log(f"Monotonicity violation: I_0 increased but R_L did not decrease. Original R_L: {self.R_L}, New R_L: {new_R_L}")
                    success = False
                else:
                    # Check the magnitude of change is roughly consistent with derivative
                    delta_I_0 = perturbed_I_0 - self.I_0
                    expected_delta_R_L = -self.U / (self.I_0**2) * delta_I_0
                    actual_delta_R_L = new_R_L - self.R_L
                    if not self._is_close(actual_delta_R_L, expected_delta_R_L, rel_tol=0.05):
                        self._log(f"Monotonicity: delta_R_L mismatch. Expected: {expected_delta_R_L}, Actual: {actual_delta_R_L}")
                        # Not a failure, just a warning
                    self._log("Monotonicity check for I_0 -> R_L passed.")
        
        # Perturb R1 by +1% and check E trend
        if None in (self.I, self.E, self.R_L):
            self._log("Missing I, E, or R_L for monotonicity check (R1 perturbation).")
            success = False
        else:
            perturbed_R1 = self.R1 * 1.01
            new_E = self.I * (self.R_L + self.R + perturbed_R1)
            # Expected: if R1 increases, E should increase (since I > 0)
            if not new_E > self.E:
                self._log(f"Monotonicity violation: R1 increased but E did not increase. Original E: {self.E}, New E: {new_E}")
                success = False
            else:
                # Check the magnitude of change
                delta_R1 = perturbed_R1 - self.R1
                expected_delta_E = self.I * delta_R1
                actual_delta_E = new_E - self.E
                if not self._is_close(actual_delta_E, expected_delta_E, rel_tol=0.05):
                    self._log(f"Monotonicity: delta_E mismatch. Expected: {expected_delta_E}, Actual: {actual_delta_E}")
                self._log("Monotonicity check for R1 -> E passed.")
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Error in {test_name}: {e}")
                self._log(traceback.format_exc())
                overall_success = False
        
        self._log(f"--- Overall: {'PASS' if overall_success else 'FAIL'} ---")
        return overall_success, self.logs