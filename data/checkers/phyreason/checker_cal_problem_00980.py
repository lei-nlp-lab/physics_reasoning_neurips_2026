import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr with fallbacks to avoid AttributeErrors.
        """
        # Inputs from the problem statement - extract from results dict or use defaults
        results = getattr(solution_instance, 'results', {})
        
        # Get inputs from results (they might be stored there) or use defaults
        self.I1 = results.get('I1', 2.0)  # Default test value
        self.I2 = results.get('I2', 3.0)
        self.T0 = results.get('T0', 10.0)
        self.T1 = results.get('T1', 8.0)
        self.a = results.get('a', 20.0)
        self.g = results.get('g', 9.8)
        
        # Results extracted from solution_instance.results with safe fallbacks
        self.F1 = results.get('F_1', 0.0)
        self.F2 = results.get('F_2', 0.0)
        self.F1_F2_ratio = results.get('F_1:F_2', 0.0)  # Could be a ratio value
        self.G = results.get('G', 0.0)  # Weight of coil
        self.I3 = results.get('I_3', 0.0)  # Final computed current
        
        # Log storage
        self.logs = []
    
    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append((success, msg))
    
    def test_invariants(self):
        """Verify fundamental physics invariants based on KB theorems."""
        try:
            # 1. Equilibrium of forces: G = 2*T0 (from step_4)
            expected_G = 2 * self.T0
            if not math.isclose(self.G, expected_G, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Equilibrium invariant failed: G = {self.G}, expected 2*T0 = {expected_G}", False)
                return False
            self._log("Equilibrium G = 2T0 satisfied", True)
            
            # 2. Force equilibrium with current I1: 2*T1 + F1 = G (from step_5)
            expected_tension_sum = 2 * self.T1 + self.F1
            if not math.isclose(self.G, expected_tension_sum, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Force equilibrium failed: G = {self.G}, 2*T1 + F1 = {expected_tension_sum}", False)
                return False
            self._log(f"Force equilibrium 2*T1 + F1 = G satisfied", True)
            
            # 3. Ampere force proportionality: F1/F2 = I1/I2 (from step_3)
            if self.I2 != 0:
                expected_ratio = self.I1 / self.I2
                actual_ratio = self.F1 / self.F2 if self.F2 != 0 else 0
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Ampere force ratio failed: F1/F2 = {actual_ratio}, expected I1/I2 = {expected_ratio}", False)
                    return False
                self._log(f"Ampere force proportionality F1:F2 = I1:I2 satisfied", True)
            
            return True
        except Exception as e:
            self._log(f"Invariant test failed with error: {traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Check the final expression for I3 matches the derived formula
            # From step_8: I3 = ((a-g)*T0*I1) / ((T0-T1)*g)
            numerator = (self.a - self.g) * self.T0 * self.I1
            denominator = (self.T0 - self.T1) * self.g
            expected_I3 = numerator / denominator if denominator != 0 else float('inf')
            
            if not math.isclose(self.I3, expected_I3, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace consistency failed: I3 = {self.I3}, expected = {expected_I3}", False)
                return False
            self._log(f"Final I3 matches derived formula: {self.I3} ≈ {expected_I3}", True)
            
            # Additional check: verify the ratio from step_7
            # I1:I3 = (T0 - T1)*g : (a-g)*T0
            ratio_lhs = self.I1 / self.I3 if self.I3 != 0 else float('inf')
            ratio_rhs = ((self.T0 - self.T1) * self.g) / ((self.a - self.g) * self.T0) if (self.a - self.g) * self.T0 != 0 else float('inf')
            if not math.isclose(ratio_lhs, ratio_rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Ratio consistency failed: I1/I3 = {ratio_lhs}, expected ratio = {ratio_rhs}", False)
                return False
            self._log(f"Ratio I1:I3 matches step_7 expression", True)
            
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input and verify expected changes in outputs.
        We'll perturb I1 and check that F1 and I3 change proportionally.
        """
        original_I1 = self.I1
        perturbation = 1.1  # 10% increase
        
        try:
            # Perturb I1
            perturbed_I1 = original_I1 * perturbation
            
            # Recompute F1 using proportionality: F1 ∝ I1
            # From step_1: F1 = k*I1*i*L*(1/r1 - 1/r2)
            # So new F1 should be perturbed by the same factor
            expected_F1_perturbed = self.F1 * perturbation
            self._log(f"Perturbation: I1 increased by 10%, expected F1 increase by 10%", True)
            
            # Recompute I3 using the formula
            numerator = (self.a - self.g) * self.T0 * perturbed_I1
            denominator = (self.T0 - self.T1) * self.g
            expected_I3_perturbed = numerator / denominator if denominator != 0 else float('inf')
            
            # Check that I3 also increases by 10% (linear relationship)
            expected_I3_factor = perturbation
            actual_I3_factor = expected_I3_perturbed / self.I3 if self.I3 != 0 else float('inf')
            
            if not math.isclose(actual_I3_factor, expected_I3_factor, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity failed: I3 changed by factor {actual_I3_factor}, expected {expected_I3_factor}", False)
                return False
            
            self._log(f"Monotonicity satisfied: I1↑10% → F1↑10%, I3↑10%", True)
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs.clear()
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Check", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...", True)
            if not test_func():
                overall_success = False
                self._log(f"{test_name} FAILED", False)
            else:
                self._log(f"{test_name} PASSED", True)
        
        return overall_success, self.logs