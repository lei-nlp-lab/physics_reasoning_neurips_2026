import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        # Safe extraction of inputs
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 0.4))
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', 0.5))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 10.0))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', None))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', None))
        
        # Safe extraction of results - handle both numeric and string values
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', None))
        self.I = getattr(solution_instance, 'I', solution_instance.results.get('I', None))
        self.direction = getattr(solution_instance, 'direction', 
                                solution_instance.results.get('N/A', None))
        
        # Store original state for perturbation tests
        self.original_state = {
            'B': self.B, 'l': self.l, 'v': self.v,
            'E': self.E, 'I': self.I, 'direction': self.direction
        }

    def test_invariants(self):
        logs = []
        try:
            # Faraday's law: E = B*l*v
            if self.E is not None and isinstance(self.E, (int, float)):
                E_expected = self.B * self.l * self.v
                if not math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Faraday's law violated: E={self.E}, expected B*l*v={E_expected}")
            elif self.E is None:
                logs.append("EMF (E) not found in results")
            
            # Ohm's law: I = E/(R+r) if both resistances available
            if (self.I is not None and isinstance(self.I, (int, float)) and 
                self.E is not None and isinstance(self.E, (int, float))):
                if self.R is not None and self.r is not None:
                    I_expected = self.E / (self.R + self.r)
                    if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"Ohm's law violated: I={self.I}, expected E/(R+r)={I_expected}")
                # Symbolic check for consistency
                elif self.R is not None or self.r is not None:
                    logs.append("Partial resistance data; Ohm's law check incomplete")
            
            # Direction consistency (qualitative)
            if self.direction is not None:
                expected_dir = "from b to a"
                if isinstance(self.direction, str) and self.direction.lower() != expected_dir.lower():
                    logs.append(f"Direction mismatch: {self.direction} vs expected {expected_dir}")
            else:
                logs.append("Current direction not specified")
                
        except Exception as e:
            logs.append(f"Error in invariants test: {str(e)}")
        
        return len(logs) == 0, logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Check step-by-step consistency from KB
            if (self.E is not None and isinstance(self.E, (int, float)) and 
                self.I is not None and isinstance(self.I, (int, float))):
                # From step 1: E = B*l*v = 2V
                E_calculated = self.B * self.l * self.v
                if not math.isclose(self.E, E_calculated, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"EMF inconsistent: result {self.E} vs calculated {E_calculated}")
                
                # From step 4: I = E/(R+r) = 0.5A if resistances allow
                if self.R is not None and self.r is not None:
                    I_calculated = self.E / (self.R + self.r)
                    if not math.isclose(self.I, I_calculated, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"Current inconsistent: result {self.I} vs calculated {I_calculated}")
            
        except Exception as e:
            logs.append(f"Error in trace consistency: {str(e)}")
        
        return len(logs) == 0, logs

    def test_monotonicity(self):
        logs = []
        original_v = self.v
        
        try:
            # Perturb velocity by 10%
            self.v = original_v * 1.1
            
            # Recalculate using physics formulas
            E_perturbed = self.B * self.l * self.v
            
            # Check EMF increases with velocity
            if (self.original_state['E'] is not None and 
                isinstance(self.original_state['E'], (int, float))):
                if not (E_perturbed > self.original_state['E']):
                    logs.append(f"Monotonicity violation: E doesn't increase with v (original E={self.original_state['E']}, perturbed E={E_perturbed})")
            
            # If resistances available, check current monotonicity
            if (self.R is not None and self.r is not None and 
                self.original_state['I'] is not None and 
                isinstance(self.original_state['I'], (int, float))):
                I_perturbed = E_perturbed / (self.R + self.r)
                if not (I_perturbed > self.original_state['I']):
                    logs.append(f"Monotonicity violation: I doesn't increase with v (original I={self.original_state['I']}, perturbed I={I_perturbed})")
            
            # Restore original state
            self.v = original_v
            
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            # Restore on error
            self.v = original_v
        
        return len(logs) == 0, logs

    def run_all_tests(self):
        all_logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                if not test_success:
                    success = False
                all_logs.append(f"{test_name}: {'PASS' if test_success else 'FAIL'}")
                all_logs.extend([f"  {log}" for log in logs])
            except Exception as e:
                success = False
                all_logs.append(f"{test_name}: ERROR - {str(e)}")
                all_logs.append(traceback.format_exc())
        
        return success, all_logs