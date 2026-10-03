import math
import re
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs using getattr and fallback to results dict
        self.N = getattr(solution_instance, 'N', solution_instance.results.get('N', 100))
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 1.0))
        self.ab = getattr(solution_instance, 'ab', solution_instance.results.get('ab', 0.30))
        self.ad = getattr(solution_instance, 'ad', solution_instance.results.get('ad', 0.20))
        self.omega = getattr(solution_instance, 'omega', solution_instance.results.get('omega', 100.0))
        
        # Safe extraction of results
        self.E_m = getattr(solution_instance, 'E_m', solution_instance.results.get('E_m', 0.0))
        # e could be either a numeric value (amplitude) or a string expression
        e_value = getattr(solution_instance, 'e', solution_instance.results.get('e', ''))
        
        # Store e appropriately
        if isinstance(e_value, (int, float)):
            self.e_amplitude = float(e_value)
            self.e_expr = f"{self.e_amplitude}*cos({self.omega}*t)"
        else:
            self.e_amplitude = None  # Will extract from string
            self.e_expr = str(e_value)
        
        # Derived quantities
        self.A = self.ab * self.ad  # Area in m^2
        
        # Logs
        self.logs = []
        
    def test_invariants(self):
        """Verify the fundamental physics laws."""
        try:
            # 1. Check maximum emf formula: E_m = N * B * A * omega
            expected_E_m = self.N * self.B * self.A * self.omega
            if not math.isclose(self.E_m, expected_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: E_m mismatch. Got {self.E_m}, expected {expected_E_m}")
                return False
            
            # 2. Check instantaneous emf expression
            # If we have amplitude directly, use it
            if self.e_amplitude is not None:
                amp = self.e_amplitude
                freq = self.omega
            else:
                # Parse the string expression to extract amplitude and angular frequency
                pattern = r'([+-]?\d*\.?\d+)\s*[×*]?\s*cos\s*\(?\s*([+-]?\d*\.?\d+)\s*[×*]?\s*t\s*\)?'
                match = re.search(pattern, self.e_expr.replace(' ', ''))
                if not match:
                    self.logs.append(f"FAIL: Cannot parse e expression: {self.e_expr}")
                    return False
                
                amp = float(match.group(1))
                freq = float(match.group(2))
            
            # Check amplitude matches E_m
            if not math.isclose(amp, self.E_m, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Amplitude in e mismatch. Got {amp}, expected {self.E_m}")
                return False
            
            # Check angular frequency matches omega (if we parsed from string)
            if self.e_amplitude is None and not math.isclose(freq, self.omega, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Angular frequency in e mismatch. Got {freq}, expected {self.omega}")
                return False
            
            # 3. Numerical verification at sample time points
            t_vals = [0.0, 0.01, 0.05, 0.1]
            for t_val in t_vals:
                expected_val = self.E_m * math.cos(self.omega * t_val)
                # Evaluate the expression if it's a string, otherwise use amplitude*cos(omega*t)
                if self.e_amplitude is not None:
                    actual_val = self.E_m * math.cos(self.omega * t_val)
                else:
                    try:
                        actual_val = eval(self.e_expr.replace('t', str(t_val)).replace('cos', 'math.cos'))
                    except:
                        self.logs.append(f"FAIL: Could not evaluate e expression at t={t_val}")
                        return False
                
                if not math.isclose(actual_val, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"FAIL: e(t) mismatch at t={t_val}. Got {actual_val}, expected {expected_val}")
                    return False
            
            self.logs.append("PASS: All invariants satisfied.")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # The steps analysis shows step_1 computes E_m, step_2 uses it in e expression
            # We already checked the formula in invariants, but also verify the value 600V
            if not math.isclose(self.E_m, 600.0, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: E_m should be 600V, got {self.E_m}")
                return False
            
            # Verify the expression has the right amplitude
            if self.e_amplitude is not None:
                if not math.isclose(self.e_amplitude, self.E_m, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"FAIL: e amplitude mismatch. Got {self.e_amplitude}, expected {self.E_m}")
                    return False
            else:
                # Verify the expression string matches expected format
                expected_format = f"{int(self.E_m)}cos({int(self.omega)}t)"
                cleaned = self.e_expr.replace(' ', '').replace('*', '').replace('×', '')
                if not cleaned.lower().startswith(expected_format.lower()):
                    self.logs.append(f"FAIL: e expression format mismatch. Got {self.e_expr}, expected like {expected_format}")
                    return False
            
            self.logs.append("PASS: Trace consistency verified.")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        try:
            # Save original values
            orig_omega = self.omega
            orig_E_m = self.E_m
            
            # Test 1: Double omega, E_m should double
            self.omega = 2 * orig_omega
            new_E_m = self.N * self.B * self.A * self.omega
            if not math.isclose(new_E_m, 2 * orig_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Doubling omega should double E_m. Got {new_E_m}, expected {2*orig_E_m}")
                self.omega = orig_omega  # Restore
                return False
            
            # Test 2: Double B, E_m should double (restore omega first)
            self.omega = orig_omega
            orig_B = self.B
            self.B = 2 * orig_B
            new_E_m = self.N * self.B * self.A * self.omega
            if not math.isclose(new_E_m, 2 * orig_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Doubling B should double E_m. Got {new_E_m}, expected {2*orig_E_m}")
                self.B = orig_B  # Restore
                return False
            
            # Restore all
            self.B = orig_B
            self.omega = orig_omega
            
            self.logs.append("PASS: Monotonicity trends verified.")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all checks and return summary."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.logs.append(f"\n--- Running {name} Test ---")
            if not test_func():
                success = False
        
        return success, self.logs