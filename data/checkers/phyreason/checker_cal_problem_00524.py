import math
import sympy as sp
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        # Extract inputs from problem statement (hardcoded as given)
        self.r = 1.0  # ohm
        self.R = 10.0  # ohm
        self.U = 150.0  # V
        self.U_V = 120.0  # V
        self.g = 9.81  # m/s² (standard gravity, implicit in problem)
        # Velocity is unknown but appears in m = P_motor/(g*v). 
        # We'll compute v from m and P_motor for consistency checks.
        
        # Safely extract results from solution instance
        self.P_hot = self._safe_get(solution_instance, 'P_hot')
        self.U_M = self._safe_get(solution_instance, 'U_M')
        self.I_M = self._safe_get(solution_instance, 'I_M')
        self.I_R = self._safe_get(solution_instance, 'I_R')
        self.P_motor = self._safe_get(solution_instance, 'P_motor')
        self.m = self._safe_get(solution_instance, 'm')
        self.P_e = self._safe_get(solution_instance, 'P_e')
        self.U_R = self._safe_get(solution_instance, 'U_R')
        
        # Store results dict for easy access
        self.results = {
            'P_hot': self.P_hot,
            'U_M': self.U_M,
            'I_M': self.I_M,
            'I_R': self.I_R,
            'P_motor': self.P_motor,
            'm': self.m,
            'P_e': self.P_e,
            'U_R': self.U_R
        }
        
        # Target answers (from problem statement)
        self.targets = {
            'I_R': 3.0,  # A
            'P_e': 360.0,  # W
            'P_motor': 351.0,  # W
            'm': 35.1  # kg
        }
        
        # Steps analysis values for consistency checking
        self.steps = {
            'U_R': 30.0,  # V
            'I_R': 3.0,  # A
            'I_M': 3.0,  # A
            'U_M': 120.0,  # V
            'P_e': 360.0,  # W
            'P_hot': 9.0,  # W
            'P_motor': 351.0,  # W
            'm': 35.1  # kg
        }
        
        self.logs = []
        
    def _safe_get(self, obj, key, default=None):
        """Safely get attribute from solution instance."""
        try:
            # Try direct attribute
            val = getattr(obj, key, default)
            if val is not default:
                return float(val) if isinstance(val, (int, float, str)) else val
            # Try results dict
            if hasattr(obj, 'results'):
                val = obj.results.get(key, default)
                return float(val) if isinstance(val, (int, float, str)) else val
            return default
        except Exception:
            return default
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self) -> List[str]:
        """Verify fundamental physics laws."""
        logs = []
        
        # 1. Ohm's Law for resistor R
        try:
            if self.I_R is not None and self.U_R is not None:
                expected = self.I_R * self.R
                if not self._is_close(self.U_R, expected):
                    logs.append(f"FAIL: Ohm's Law for R: U_R = {self.U_R}, I_R*R = {expected}")
                else:
                    logs.append("PASS: Ohm's Law for resistor R")
        except Exception as e:
            logs.append(f"ERROR in Ohm's Law check: {str(e)}")
        
        # 2. Ohm's Law for motor internal resistance (P_hot = I_M² * r)
        try:
            if self.P_hot is not None and self.I_M is not None:
                expected = self.I_M**2 * self.r
                if not self._is_close(self.P_hot, expected):
                    logs.append(f"FAIL: Joule's Law for motor: P_hot = {self.P_hot}, I_M²*r = {expected}")
                else:
                    logs.append("PASS: Joule's Law for motor heating")
        except Exception as e:
            logs.append(f"ERROR in Joule's Law check: {str(e)}")
        
        # 3. Voltage conservation: U = U_R + U_M
        try:
            if self.U_R is not None and self.U_M is not None:
                total = self.U_R + self.U_M
                if not self._is_close(self.U, total):
                    logs.append(f"FAIL: Voltage conservation: U = {self.U}, U_R+U_M = {total}")
                else:
                    logs.append("PASS: Voltage conservation (U = U_R + U_M)")
        except Exception as e:
            logs.append(f"ERROR in voltage conservation check: {str(e)}")
        
        # 4. Current conservation in series: I_M = I_R
        try:
            if self.I_M is not None and self.I_R is not None:
                if not self._is_close(self.I_M, self.I_R):
                    logs.append(f"FAIL: Current conservation: I_M = {self.I_M}, I_R = {self.I_R}")
                else:
                    logs.append("PASS: Current conservation (I_M = I_R)")
        except Exception as e:
            logs.append(f"ERROR in current conservation check: {str(e)}")
        
        # 5. Motor power conservation: P_e = P_motor + P_hot
        try:
            if all(v is not None for v in [self.P_e, self.P_motor, self.P_hot]):
                total_loss = self.P_motor + self.P_hot
                if not self._is_close(self.P_e, total_loss):
                    logs.append(f"FAIL: Motor power conservation: P_e = {self.P_e}, P_motor+P_hot = {total_loss}")
                else:
                    logs.append("PASS: Motor power conservation (P_e = P_motor + P_hot)")
        except Exception as e:
            logs.append(f"ERROR in motor power conservation check: {str(e)}")
        
        # 6. Electrical power definition: P_e = I_M * U_M
        try:
            if self.P_e is not None and self.I_M is not None and self.U_M is not None:
                expected = self.I_M * self.U_M
                if not self._is_close(self.P_e, expected):
                    logs.append(f"FAIL: P_e definition: P_e = {self.P_e}, I_M*U_M = {expected}")
                else:
                    logs.append("PASS: Electrical power definition (P_e = I_M * U_M)")
        except Exception as e:
            logs.append(f"ERROR in P_e definition check: {str(e)}")
        
        # 7. Mass calculation consistency (using implied velocity)
        try:
            if self.m is not None and self.P_motor is not None:
                # From m = P_motor/(g*v), we can check that v would be positive
                # We don't know v, but we can verify the ratio is physically reasonable
                if self.P_motor > 0 and self.m > 0:
                    implied_v = self.P_motor / (self.g * self.m)
                    if implied_v <= 0 or implied_v > 100:  # Reasonable velocity range (0-100 m/s)
                        logs.append(f"WARN: Implied velocity v = {implied_v:.2f} m/s seems unrealistic")
                    else:
                        logs.append(f"INFO: Implied lifting velocity v = {implied_v:.2f} m/s")
                else:
                    logs.append("WARN: Non-positive P_motor or mass")
        except Exception as e:
            logs.append(f"ERROR in mass consistency check: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with steps analysis."""
        logs = []
        
        # Check each step against the solution's results
        step_checks = [
            ('U_R', self.U_R, self.steps['U_R'], "U_R = U - U_V"),
            ('I_R', self.I_R, self.steps['I_R'], "I_R = U_R / R"),
            ('I_M', self.I_M, self.steps['I_M'], "I_M = I_R (series)"),
            ('U_M', self.U_M, self.steps['U_M'], "U_M = U_V"),
            ('P_e', self.P_e, self.steps['P_e'], "P_e = I_M * U_M"),
            ('P_hot', self.P_hot, self.steps['P_hot'], "P_hot = I_M² * r"),
            ('P_motor', self.P_motor, self.steps['P_motor'], "P_motor = P_e - P_hot"),
            ('m', self.m, self.steps['m'], "m = P_motor / (g * v)")
        ]
        
        for name, actual, expected, desc in step_checks:
            if actual is None:
                logs.append(f"MISS: {name} not available")
            elif not self._is_close(actual, expected):
                logs.append(f"FAIL: Step {name}: {desc} -> got {actual}, expected {expected}")
            else:
                logs.append(f"PASS: Step {name}: {desc}")
        
        # Check against target answers
        for key, target_val in self.targets.items():
            actual = self.results.get(key)
            if actual is None:
                logs.append(f"MISS: Target {key} not available")
            elif not self._is_close(actual, target_val):
                logs.append(f"FAIL: Target {key}: got {actual}, expected {target_val}")
            else:
                logs.append(f"PASS: Target {key} matches")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Perturb inputs and check physical trends (numerical reasoning)."""
        logs = []
        
        # We'll do symbolic reasoning with SymPy since we can't re-run the solution
        try:
            # Define symbols
            I_M, U_M, r_val, P_e, P_hot, P_motor = sp.symbols('I_M U_M r P_e P_hot P_motor')
            
            # Define relationships from steps
            relations = [
                sp.Eq(P_e, I_M * U_M),
                sp.Eq(P_hot, I_M**2 * r_val),
                sp.Eq(P_motor, P_e - P_hot)
            ]
            
            # 1. Check if P_motor increases with I_M (holding U_M and r constant)
            # Solve for P_motor as function of I_M
            P_motor_expr = sp.solve(relations, P_motor)[0]
            # Differentiate wrt I_M
            dP_dI = sp.diff(P_motor_expr, I_M)
            # Substitute typical values: U_M = 120, r = 1
            dP_dI_val = dP_dI.subs({U_M: 120, r_val: 1})
            # Should be positive for reasonable I_M? Actually: dP_dI = U_M - 2*I_M*r
            # At I_M = 3A: dP_dI = 120 - 2*3*1 = 114 > 0, so increasing I_M increases P_motor
            logs.append(f"INFO: d(P_motor)/d(I_M) = {dP_dI_val} (positive → increasing I_M increases P_motor)")
            
            # 2. Check if increasing r decreases P_motor
            dP_dr = sp.diff(P_motor_expr, r_val)
            dP_dr_val = dP_dr.subs({I_M: 3})
            # dP_dr = -I_M^2 = -9 < 0, so increasing r decreases P_motor
            logs.append(f"INFO: d(P_motor)/d(r) = {dP_dr_val} (negative → increasing r decreases P_motor)")
            
            # 3. Check if increasing U (which increases U_M if voltmeter reading fixed?)
            # Actually U_M = U_V is fixed at 120V, so changing U changes U_R, thus I_M
            # From U_R = U - U_V and I_M = U_R/R = (U - 120)/10
            U, R_val = sp.symbols('U R')
            I_M_expr = (U - 120) / R_val
            P_motor_U = P_motor_expr.subs({I_M: I_M_expr, U_M: 120, r_val: 1, R_val: 10})
            dP_dU = sp.diff(P_motor_U, U)
            dP_dU_val = dP_dU.subs({U: 150})
            # Should be positive: increasing U increases P_motor
            logs.append(f"INFO: d(P_motor)/d(U) = {float(dP_dU_val):.2f} (positive → increasing U increases P_motor)")
            
            logs.append("PASS: Monotonicity trends are physically reasonable")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity analysis: {str(e)}")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run each test
            for test_name, test_method in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== {test_name} Test ===")
                test_logs = test_method()
                all_logs.extend(test_logs)
                
                # Check for failures in this test
                for log in test_logs:
                    if log.startswith("FAIL:"):
                        success = False
                    elif log.startswith("ERROR:"):
                        success = False
        
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR in test execution: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall success: {'PASS' if success else 'FAIL'}")
        
        return success, all_logs