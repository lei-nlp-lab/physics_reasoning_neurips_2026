import sympy as sp
import math
import traceback
import re

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results using robust attribute access."""
        self.solution = solution_instance
        self.logs = []
        
        # Input parameters from problem context
        self.B = getattr(solution_instance, 'B', 
                        getattr(solution_instance, 'inputs', {}).get('B', 0.5))
        self.L = getattr(solution_instance, 'L', 
                        getattr(solution_instance, 'inputs', {}).get('L', 0.5))
        self.I = getattr(solution_instance, 'I', 
                        getattr(solution_instance, 'inputs', {}).get('I', 2.0))
        
        # Result extraction with fallback
        self.F_raw = getattr(solution_instance, 'F', 
                            solution_instance.results.get('F', 0.0))
        
        # Convert string result like "0.5N" to float if needed
        if isinstance(self.F_raw, str):
            try:
                self.F_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", self.F_raw)[0])
            except:
                self.F_val = 0.0
        else:
            self.F_val = float(self.F_raw)
    
    def test_invariants(self):
        """Verify Ampere's law F = BIL with perpendicular orientation."""
        logs = []
        success = True
        
        try:
            # Symbolic verification
            B, I, L, F = sp.symbols('B I L F')
            law = sp.Eq(F, B * I * L)
            
            # Numerical check with 2% tolerance
            expected = self.B * self.I * self.L
            if not math.isclose(self.F_val, expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Ampere's law invariant violated. F = {self.F_val}, BIL = {expected}")
                success = False
            else:
                logs.append(f"PASS: Ampere's law F = BIL holds within 2% tolerance")
                
        except Exception as e:
            logs.append(f"ERROR in invariant test: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check if intermediate steps align with final result."""
        logs = []
        success = True
        
        try:
            # Since steps analysis only has one step with F = BIL, verify consistency
            if hasattr(self.solution, 'steps'):
                for step_key, step_data in self.solution.steps.items():
                    if 'result_quantity' in step_data:
                        for quantity in step_data['result_quantity']:
                            if quantity.get('symbol') == 'F':
                                step_val = float(quantity.get('value', 0))
                                if not math.isclose(step_val, self.F_val, rel_tol=0.02, abs_tol=0.02):
                                    logs.append(f"WARN: Step {step_key} F={step_val} differs from final F={self.F_val}")
                                    success = False
            
            # Direct comparison with target answer
            expected_final = 0.5
            if not math.isclose(self.F_val, expected_final, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Final result {self.F_val} differs from target {expected_final}")
                success = False
            else:
                logs.append("PASS: Final result consistent with target answer")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and verify proportional relationships from KB."""
        logs = []
        success = True
        
        try:
            original_F = self.F_val
            
            # Test 1: Increase current I by 10%
            perturbed_I = self.I * 1.1
            new_F_I = self.B * perturbed_I * self.L
            if not (new_F_I > original_F):
                logs.append(f"FAIL: Monotonicity violated for I↑. F={original_F} → {new_F_I} with I×1.1")
                success = False
            else:
                logs.append(f"PASS: I↑10% → F↑ ({original_F:.3f} → {new_F_I:.3f})")
            
            # Test 2: Increase magnetic field B by 10%
            perturbed_B = self.B * 1.1
            new_F_B = perturbed_B * self.I * self.L
            if not (new_F_B > original_F):
                logs.append(f"FAIL: Monotonicity violated for B↑. F={original_F} → {new_F_B} with B×1.1")
                success = False
            else:
                logs.append(f"PASS: B↑10% → F↑ ({original_F:.3f} → {new_F_B:.3f})")
            
            # Test 3: Increase length L by 10%
            perturbed_L = self.L * 1.1
            new_F_L = self.B * self.I * perturbed_L
            if not (new_F_L > original_F):
                logs.append(f"FAIL: Monotonicity violated for L↑. F={original_F} → {new_F_L} with L×1.1")
                success = False
            else:
                logs.append(f"PASS: L↑10% → F↑ ({original_F:.3f} → {new_F_L:.3f})")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all checks and return comprehensive results."""
        all_logs = []
        overall_success = True
        
        # Test 1: Physical invariants
        success1, logs1 = self.test_invariants()
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        if overall_success:
            all_logs.insert(0, "PhysicsChecker: All tests passed")
        else:
            all_logs.insert(0, "PhysicsChecker: Some tests failed")
        
        return overall_success, all_logs