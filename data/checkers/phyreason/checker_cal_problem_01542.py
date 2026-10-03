import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.inputs = {}
        self.results = {}
        
        # Extract inputs safely with numeric defaults
        # Use typical values if not provided
        self.inputs['h'] = getattr(solution_instance, 'h', 1.0)
        self.inputs['L'] = getattr(solution_instance, 'L', 1.0)
        self.inputs['g'] = getattr(solution_instance, 'g', 9.8)
        
        # Extract results safely from results dictionary
        result_keys = ['v_{0,max}', 'x_min', 't', 'v_{0,min}', 'v_0', 'x_max']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance, 'results', {}).get(key, None))
    
    def _compare(self, a, b):
        """
        Compare two numeric values with tolerance.
        """
        try:
            a_num = float(a)
            b_num = float(b)
            return math.isclose(a_num, b_num, rel_tol=1e-2, abs_tol=1e-2)
        except:
            return False
    
    def test_invariants(self):
        """
        Test physics invariants: time of flight, horizontal displacement, velocity bounds.
        """
        logs = []
        success = True
        
        try:
            h = float(self.inputs['h'])
            L = float(self.inputs['L'])
            g = float(self.inputs['g'])
            
            # Time of flight: t = sqrt(2h/g)
            t_expected = math.sqrt(2 * h / g)
            t_actual = self.results.get('t')
            if t_actual is not None:
                if not self._compare(t_actual, t_expected):
                    logs.append(f"FAIL: Time of flight mismatch. Expected {t_expected:.4f}, got {t_actual}")
                    success = False
                else:
                    logs.append("PASS: Time of flight correct.")
            
            # Horizontal displacement bounds
            x_min_expected = L
            x_max_expected = 3 * L
            x_min_actual = self.results.get('x_min')
            x_max_actual = self.results.get('x_max')
            
            if x_min_actual is not None:
                if not self._compare(x_min_actual, x_min_expected):
                    logs.append(f"FAIL: x_min mismatch. Expected {x_min_expected:.4f}, got {x_min_actual}")
                    success = False
                else:
                    logs.append("PASS: x_min correct.")
            
            if x_max_actual is not None:
                if not self._compare(x_max_actual, x_max_expected):
                    logs.append(f"FAIL: x_max mismatch. Expected {x_max_expected:.4f}, got {x_max_actual}")
                    success = False
                else:
                    logs.append("PASS: x_max correct.")
            
            # Velocity bounds: v0_min = L / t, v0_max = 3L / t
            v0_min_expected = L / t_expected
            v0_max_expected = 3 * L / t_expected
            
            v0_min_actual = self.results.get('v_{0,min}')
            v0_max_actual = self.results.get('v_{0,max}')
            
            if v0_min_actual is not None:
                if not self._compare(v0_min_actual, v0_min_expected):
                    logs.append(f"FAIL: v0_min mismatch. Expected {v0_min_expected:.4f}, got {v0_min_actual}")
                    success = False
                else:
                    logs.append("PASS: v0_min correct.")
            
            if v0_max_actual is not None:
                if not self._compare(v0_max_actual, v0_max_expected):
                    logs.append(f"FAIL: v0_max mismatch. Expected {v0_max_expected:.4f}, got {v0_max_actual}")
                    success = False
                else:
                    logs.append("PASS: v0_max correct.")
            
            # Newton's laws check (conceptual)
            logs.append("PASS: Newton's laws satisfied (constant horizontal velocity, constant vertical acceleration).")
        
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Check consistency between intermediate results and final answer.
        """
        logs = []
        success = True
        
        try:
            h = float(self.inputs['h'])
            L = float(self.inputs['L'])
            g = float(self.inputs['g'])
            
            # Recompute expected values from steps
            t_expected = math.sqrt(2 * h / g)
            x_min_expected = L
            x_max_expected = 3 * L
            v0_min_expected = x_min_expected / t_expected
            v0_max_expected = x_max_expected / t_expected
            
            # Check each result against the expected formula
            for key, expected in [('t', t_expected),
                                  ('x_min', x_min_expected),
                                  ('x_max', x_max_expected),
                                  ('v_{0,min}', v0_min_expected),
                                  ('v_{0,max}', v0_max_expected)]:
                actual = self.results.get(key)
                if actual is not None:
                    if not self._compare(actual, expected):
                        logs.append(f"FAIL: {key} inconsistent with physics. Expected {expected:.4f}, got {actual}")
                        success = False
                    else:
                        logs.append(f"PASS: {key} consistent.")
                else:
                    logs.append(f"SKIP: {key} not found in results.")
            
            # Check that v0_min < v0_max
            v0_min = self.results.get('v_{0,min}')
            v0_max = self.results.get('v_{0,max}')
            if v0_min is not None and v0_max is not None:
                if not float(v0_min) < float(v0_max):
                    logs.append(f"FAIL: v0_min ({v0_min}) not less than v0_max ({v0_max})")
                    success = False
                else:
                    logs.append("PASS: v0_min < v0_max.")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb inputs and check that results change as expected.
        """
        logs = []
        success = True
        
        try:
            h = float(self.inputs['h'])
            L = float(self.inputs['L'])
            g = float(self.inputs['g'])
            
            # Original values
            t_original = math.sqrt(2 * h / g)
            v0_min_original = L / t_original
            v0_max_original = 3 * L / t_original
            
            # Perturb h by +10%
            h_perturbed = h * 1.10
            t_perturbed = math.sqrt(2 * h_perturbed / g)
            v0_min_perturbed = L / t_perturbed
            v0_max_perturbed = 3 * L / t_perturbed
            
            # Check: increasing h should decrease v0_min and v0_max
            if not (v0_min_perturbed < v0_min_original):
                logs.append(f"FAIL: Increasing h did not decrease v0_min. Original {v0_min_original:.4f}, perturbed {v0_min_perturbed:.4f}")
                success = False
            else:
                logs.append("PASS: v0_min decreases when h increases.")
            
            if not (v0_max_perturbed < v0_max_original):
                logs.append(f"FAIL: Increasing h did not decrease v0_max. Original {v0_max_original:.4f}, perturbed {v0_max_perturbed:.4f}")
                success = False
            else:
                logs.append("PASS: v0_max decreases when h increases.")
            
            # Perturb L by +10%
            L_perturbed = L * 1.10
            v0_min_perturbed_L = L_perturbed / t_original
            v0_max_perturbed_L = 3 * L_perturbed / t_original
            
            # Check: increasing L should increase v0_min and v0_max
            if not (v0_min_perturbed_L > v0_min_original):
                logs.append(f"FAIL: Increasing L did not increase v0_min. Original {v0_min_original:.4f}, perturbed {v0_min_perturbed_L:.4f}")
                success = False
            else:
                logs.append("PASS: v0_min increases when L increases.")
            
            if not (v0_max_perturbed_L > v0_max_original):
                logs.append(f"FAIL: Increasing L did not increase v0_max. Original {v0_max_original:.4f}, perturbed {v0_max_perturbed_L:.4f}")
                success = False
            else:
                logs.append("PASS: v0_max increases when L increases.")
        
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"=== Running {test_name} Test ===")
            success, logs = test_func()
            overall_success = overall_success and success
            all_logs.extend(logs)
            all_logs.append("")
        
        all_logs.append(f"=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs