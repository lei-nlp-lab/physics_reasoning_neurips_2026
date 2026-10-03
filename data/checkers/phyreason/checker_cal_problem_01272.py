import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Inputs from problem statement (safe direct assignment)
        self.d1 = 347.5
        self.d2 = 337.5
        self.v_ultra = 340.0
        
        # Extract results with fallbacks
        self.x = getattr(solution_instance, 'x', getattr(solution_instance, 'results', {}).get('x', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'results', {}).get('a', 0.0))
        self.v0 = getattr(solution_instance, 'v_0', getattr(solution_instance, 'results', {}).get('v_0', 0.0))
        
        # For monotonicity perturbation
        self._original_inputs = {'d1': self.d1, 'd2': self.d2, 'v_ultra': self.v_ultra}
        self._original_results = {'x': self.x, 't': self.t, 'a': self.a, 'v0': self.v0}
        
        # Setup symbolic variables
        self.symbols = self._setup_symbols()
    
    def _setup_symbols(self):
        """Define SymPy symbols for physical quantities."""
        d1, d2, v, x, t, a, v0 = sp.symbols('d1 d2 v x t a v0', real=True, positive=True)
        return {'d1': d1, 'd2': d2, 'v': v, 'x': x, 't': t, 'a': a, 'v0': v0}
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison with default 1% tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """
        Verify kinematic relationships from the KB.
        Returns: (bool, list_of_logs)
        """
        logs = []
        all_pass = True
        
        try:
            # 1. Check step2 ratio: (d1 - x) / (x - d2) = 3/1
            ratio = (self.d1 - self.x) / (self.x - self.d2) if self.x != self.d2 else float('inf')
            if not self._is_close(ratio, 3.0):
                logs.append(f"FAIL: Step2 ratio mismatch. Expected ~3.0, got {ratio:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: Step2 ratio = {ratio:.3f} ≈ 3.0")
            
            # 2. Check step3: t = 2*x / v_ultra
            t_expected = 2 * self.x / self.v_ultra
            if not self._is_close(self.t, t_expected):
                logs.append(f"FAIL: Step3 time mismatch. Expected {t_expected:.3f}, got {self.t:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: Step3 t = {self.t:.3f} ≈ 2*x/v = {t_expected:.3f}")
            
            # 3. Check step4: d1 - d2 = 0.5 * |a| * t^2 (use absolute value for magnitude)
            displacement = self.d1 - self.d2
            expected_displacement = 0.5 * abs(self.a) * self.t**2
            if not self._is_close(displacement, expected_displacement):
                logs.append(f"FAIL: Step4 displacement mismatch. Expected {expected_displacement:.3f}, got {displacement:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: Step4 d1-d2 = {displacement:.3f} ≈ 0.5*|a|*t² = {expected_displacement:.3f}")
            
            # 4. Check step5: v0 = |a| * t (since a is deceleration magnitude)
            v0_expected = abs(self.a) * self.t
            if not self._is_close(self.v0, v0_expected):
                logs.append(f"FAIL: Step5 v0 mismatch. Expected {v0_expected:.3f}, got {self.v0:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: Step5 v0 = {self.v0:.3f} ≈ |a|*t = {v0_expected:.3f}")
            
            # 5. Energy/work-energy theorem check (optional): v0^2 = 2*|a|*(d1-d2)
            v0_sq_expected = 2 * abs(self.a) * (self.d1 - self.d2)
            v0_sq_actual = self.v0**2
            if not self._is_close(v0_sq_actual, v0_sq_expected, rel_tol=5e-2):
                logs.append(f"WARN: Work-energy check: v0² = {v0_sq_actual:.3f}, 2|a|Δd = {v0_sq_expected:.3f}")
            else:
                logs.append(f"PASS: Work-energy check consistent")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            all_pass = False
        
        return all_pass, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        all_pass = True
        
        try:
            # Expected values from steps analysis
            expected_x = 340.0
            expected_t = 2.0
            expected_a = 5.0
            expected_v0 = 10.0
            
            checks = [
                ('x', self.x, expected_x),
                ('t', self.t, expected_t),
                ('a', abs(self.a), expected_a),  # Use magnitude
                ('v0', self.v0, expected_v0)
            ]
            
            for name, actual, expected in checks:
                if not self._is_close(actual, expected):
                    logs.append(f"FAIL: {name} = {actual:.3f}, expected ≈ {expected}")
                    all_pass = False
                else:
                    logs.append(f"PASS: {name} = {actual:.3f} ≈ {expected}")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            all_pass = False
        
        return all_pass, logs
    
    def test_monotonicity(self):
        """
        Perturb d1 (increase by 1%), recalculate using formulas, check trends.
        Restore original values after test.
        """
        logs = []
        all_pass = True
        
        try:
            # Perturb d1 by +1%
            d1_perturbed = self.d1 * 1.01  # 347.5 -> ~350.975
            
            # Recalculate using step formulas (symbolic derivation)
            # From step2: (d1 - x)/(x - d2) = 3 => x = (d1 + 3*d2)/4
            x_perturbed = (d1_perturbed + 3 * self.d2) / 4
            
            # From step3: t = 2*x/v_ultra
            t_perturbed = 2 * x_perturbed / self.v_ultra
            
            # From step4: a = 2*(d1 - d2)/t^2 (magnitude, negative sign for deceleration)
            a_perturbed_mag = 2 * (d1_perturbed - self.d2) / (t_perturbed**2)
            
            # From step5: v0 = a*t (using magnitude)
            v0_perturbed = a_perturbed_mag * t_perturbed
            
            # Check monotonic trends (all should increase with increased d1)
            original = [self.x, self.t, abs(self.a), self.v0]
            perturbed = [x_perturbed, t_perturbed, a_perturbed_mag, v0_perturbed]
            names = ['x', 't', '|a|', 'v0']
            
            for i, (name, orig, pert) in enumerate(zip(names, original, perturbed)):
                if pert > orig:
                    logs.append(f"PASS: {name} increases with d1: {orig:.3f} -> {pert:.3f}")
                else:
                    logs.append(f"FAIL: {name} does not increase: {orig:.3f} -> {pert:.3f}")
                    all_pass = False
            
            # Verify physical consistency of perturbed values
            # Check ratio condition
            ratio_perturbed = (d1_perturbed - x_perturbed) / (x_perturbed - self.d2)
            if not self._is_close(ratio_perturbed, 3.0, rel_tol=5e-3):
                logs.append(f"WARN: Perturbed ratio = {ratio_perturbed:.3f}, expected ~3.0")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            all_pass = False
        
        # Restore original values (important for subsequent tests)
        self.d1 = self._original_inputs['d1']
        self.d2 = self._original_inputs['d2']
        self.v_ultra = self._original_inputs['v_ultra']
        
        return all_pass, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} Test ===")
            success, logs = test_method()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs

# Example usage (commented out for production):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'x': 340.0,
#                 't': 2.0,
#                 'a': -5.0,  # Negative for deceleration
#                 'v_0': 10.0
#             }
#     
#     mock_solution = MockSolution()
#     checker = PhysicsChecker(mock_solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)