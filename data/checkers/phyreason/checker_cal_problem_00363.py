import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract the namespace where the code ran
        if hasattr(solution_instance, '__dict__'):
            namespace = solution_instance.__dict__
        else:
            namespace = {}
        
        # Get results from the namespace
        self.results = namespace.get('results', {})
        
        # Extract input values from the code's namespace
        # Look for the num_vals dictionary or individual variables
        self.inputs = {}
        
        # Try to get num_vals dictionary
        num_vals = namespace.get('num_vals', {})
        if num_vals:
            # Convert sympy symbols to float values
            for key, val in num_vals.items():
                if hasattr(val, 'evalf'):
                    self.inputs[str(key)] = float(val.evalf())
                else:
                    self.inputs[str(key)] = float(val)
        
        # Also check for individual variables
        for var_name in ['U', 'I', 't', 'C', 'R', 'I_2']:
            if var_name in namespace:
                val = namespace[var_name]
                if hasattr(val, 'evalf'):
                    self.inputs[var_name] = float(val.evalf())
                else:
                    self.inputs[var_name] = float(val)
        
        # Ensure all required keys exist with defaults from results if available
        required_inputs = ['U', 'I', 't', 'C', 'R', 'I_2']
        for key in required_inputs:
            if key not in self.inputs:
                # Try to get from results
                if key in self.results:
                    self.inputs[key] = self.results[key]
                else:
                    self.inputs[key] = 0.0
        
        # Ensure all required results exist
        required_results = ['U_1', 'U_2', 'Q', 'I_2', 'U_R']
        for key in required_results:
            if key not in self.results:
                self.results[key] = 0.0
        
        self.logs = []
    
    def _log(self, message, success=True):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
    
    def _numerical_check(self, val1, val2, description):
        try:
            if math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"{description}: {val1:.3f} ≈ {val2:.3f}", True)
                return True
            else:
                self._log(f"{description}: {val1:.3f} ≠ {val2:.3f} (difference: {abs(val1-val2):.3f})", False)
                return False
        except Exception as e:
            self._log(f"Numerical check failed for {description}: {e}", False)
            return False
    
    def test_invariants(self):
        success = True
        
        # 1. Check capacitance definition: U_1 = Q/C
        try:
            if abs(self.inputs['C']) > 1e-10:  # Avoid division by near-zero
                numerical_result = self.results['Q'] / self.inputs['C']
                if not self._numerical_check(self.results['U_1'], numerical_result, "Capacitance definition (U_1 = Q/C)"):
                    success = False
            else:
                self._log("Capacitance definition check skipped (C≈0)", True)
        except Exception as e:
            self._log(f"Capacitance definition check failed: {e}", False)
            success = False
        
        # 2. Check Ohm's Law for resistor: U_R = I_2 * R
        try:
            numerical_result = self.results['I_2'] * self.inputs['R']
            if not self._numerical_check(self.results['U_R'], numerical_result, "Ohm's Law (U_R = I_2*R)"):
                success = False
        except Exception as e:
            self._log(f"Ohm's Law check failed: {e}", False)
            success = False
        
        # 3. Check Kirchhoff's Voltage Law: U = U_2 + U_R
        try:
            numerical_result = self.results['U_2'] + self.results['U_R']
            if not self._numerical_check(self.inputs['U'], numerical_result, "Kirchhoff's Voltage Law (U = U_2 + U_R)"):
                success = False
        except Exception as e:
            self._log(f"KVL check failed: {e}", False)
            success = False
        
        # 4. Check current definition consistency: Q = I*t during constant current phase
        try:
            numerical_result = self.inputs['I'] * self.inputs['t']
            if not self._numerical_check(self.results['Q'], numerical_result, "Current definition (Q = I*t)"):
                success = False
        except Exception as e:
            self._log(f"Current definition check failed: {e}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        success = True
        
        # Check that intermediate results are consistent with final answers
        # From steps: U_1 = I*t/C (from steps 1 and 2)
        try:
            if abs(self.inputs['C']) > 1e-10:
                expected_u1 = self.inputs['I'] * self.inputs['t'] / self.inputs['C']
                if not self._numerical_check(self.results['U_1'], expected_u1, "Step consistency for U_1 (I*t/C)"):
                    success = False
            else:
                self._log("U_1 consistency check skipped (C≈0)", True)
        except Exception as e:
            self._log(f"U_1 consistency check failed: {e}", False)
            success = False
        
        # From steps: U_2 = U - I_2*R (from steps 4 and 5)
        try:
            expected_u2 = self.inputs['U'] - self.results['I_2'] * self.inputs['R']
            if not self._numerical_check(self.results['U_2'], expected_u2, "Step consistency for U_2 (U - I_2*R)"):
                success = False
        except Exception as e:
            self._log(f"U_2 consistency check failed: {e}", False)
            success = False
        
        # Check that I_2 matches input if provided
        if abs(self.inputs['I_2']) > 1e-10:
            if not self._numerical_check(self.results['I_2'], self.inputs['I_2'], "I_2 matches input value"):
                success = False
        
        return success
    
    def test_monotonicity(self):
        success = True
        
        # Test 1: Increase current I, U_1 should increase (U_1 = I*t/C)
        try:
            original_I = self.inputs['I']
            perturbed_I = original_I * 1.1  # Increase by 10%
            
            original_u1 = self.results['U_1']
            if abs(self.inputs['C']) > 1e-10:
                perturbed_u1 = perturbed_I * self.inputs['t'] / self.inputs['C']
                if perturbed_u1 > original_u1:
                    self._log(f"Monotonicity: Increasing I from {original_I:.3f} to {perturbed_I:.3f} increases U_1 from {original_u1:.3f} to {perturbed_u1:.3f}", True)
                else:
                    self._log(f"Monotonicity: Increasing I did not increase U_1 as expected", False)
                    success = False
            else:
                self._log("Monotonicity test 1 skipped (C≈0)", True)
        except Exception as e:
            self._log(f"Monotonicity test 1 failed: {e}", False)
            success = False
        
        # Test 2: Increase current I, U_2 should decrease (U_2 = U - I_2*R)
        try:
            # Note: In constant current stage, I_2 = I
            original_u2 = self.results['U_2']
            perturbed_u2 = self.inputs['U'] - perturbed_I * self.inputs['R']
            
            if perturbed_u2 < original_u2:
                self._log(f"Monotonicity: Increasing I from {original_I:.3f} to {perturbed_I:.3f} decreases U_2 from {original_u2:.3f} to {perturbed_u2:.3f}", True)
            else:
                self._log(f"Monotonicity: Increasing I did not decrease U_2 as expected", False)
                success = False
        except Exception as e:
            self._log(f"Monotonicity test 2 failed: {e}", False)
            success = False
        
        # Test 3: Increase resistance R, U_R should increase (U_R = I_2*R)
        try:
            original_R = self.inputs['R']
            perturbed_R = original_R * 1.1  # Increase by 10%
            
            original_ur = self.results['U_R']
            perturbed_ur = self.results['I_2'] * perturbed_R
            
            if perturbed_ur > original_ur:
                self._log(f"Monotonicity: Increasing R from {original_R:.3f} to {perturbed_R:.3f} increases U_R from {original_ur:.3f} to {perturbed_ur:.3f}", True)
            else:
                self._log(f"Monotonicity: Increasing R did not increase U_R as expected", False)
                success = False
        except Exception as e:
            self._log(f"Monotonicity test 3 failed: {e}", False)
            success = False
        
        # Test 4: Increase capacitance C, U_1 should decrease (U_1 = I*t/C)
        try:
            original_C = self.inputs['C']
            perturbed_C = original_C * 1.1  # Increase by 10%
            
            original_u1 = self.results['U_1']
            if abs(perturbed_C) > 1e-10:
                perturbed_u1_c = self.inputs['I'] * self.inputs['t'] / perturbed_C
                if perturbed_u1_c < original_u1:
                    self._log(f"Monotonicity: Increasing C from {original_C:.3f} to {perturbed_C:.3f} decreases U_1 from {original_u1:.3f} to {perturbed_u1_c:.3f}", True)
                else:
                    self._log(f"Monotonicity: Increasing C did not decrease U_1 as expected", False)
                    success = False
            else:
                self._log("Monotonicity test 4 skipped (C≈0)", True)
        except Exception as e:
            self._log(f"Monotonicity test 4 failed: {e}", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        self.logs = []
        overall_success = True
        
        self._log("=== Starting Physics Audit ===", True)
        
        try:
            # Test 1: Physical invariants
            self._log("\n1. Testing Physical Invariants:", True)
            if not self.test_invariants():
                overall_success = False
            
            # Test 2: Trace consistency
            self._log("\n2. Testing Trace Consistency:", True)
            if not self.test_trace_consistency():
                overall_success = False
            
            # Test 3: Monotonicity
            self._log("\n3. Testing Monotonicity:", True)
            if not self.test_monotonicity():
                overall_success = False
            
        except Exception as e:
            self._log(f"Critical error during testing: {e}", False)
            self._log(traceback.format_exc(), False)
            overall_success = False
        
        self._log(f"\n=== Audit {'PASSED' if overall_success else 'FAILED'} ===", overall_success)
        
        return overall_success, self.logs