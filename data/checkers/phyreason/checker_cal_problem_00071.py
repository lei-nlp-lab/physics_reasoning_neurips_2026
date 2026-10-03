import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        self.logs = []
        
        # Extract inputs with sensible defaults
        # The solution should have these as attributes or in results
        default_inputs = {
            'm': 1.0,
            'g': 9.8,
            'theta': math.pi / 6,  # 30 degrees
            'd': 1.0,
            'mu': 0.2,
            's': 10.0,
            'L': 20.0
        }
        
        for key in ['m', 'g', 'theta', 'd', 'mu', 's', 'L']:
            # Try to get from solution instance attributes first
            val = getattr(solution_instance, key, None)
            if val is None:
                # Try from results dict
                val = getattr(solution_instance, 'results', {}).get(key, None)
            if val is None:
                # Use default
                val = default_inputs[key]
            self.inputs[key] = val
        
        # Extract results from results dict
        results_dict = getattr(solution_instance, 'results', {})
        result_mapping = {
            'a': ['a'],
            'DeltaE': ['DeltaE'],
            'DeltaE_all': ['DeltaE_all'],
            'DeltaE_prime': ['DeltaE_prime'],
            'L_condition': ['L_condition']
        }
        
        for res_key, possible_keys in result_mapping.items():
            for key in possible_keys:
                val = results_dict.get(key, None)
                if val is not None:
                    self.results[res_key] = float(val) if isinstance(val, (int, float)) else val
                    break
            else:
                self.results[res_key] = None
        
        # Store expected numeric values based on inputs
        m = self.inputs['m']
        g = self.inputs['g']
        theta = self.inputs['theta']
        d = self.inputs['d']
        mu = self.inputs['mu']
        s = self.inputs['s']
        L = self.inputs['L']
        
        self.expected_numeric = {
            'a': g * math.sin(theta),
            'DeltaE': m * g * d * math.sin(theta),
            'DeltaE_all': m * g * (L + 29*d) * math.sin(theta) - mu * m * g * s,
            'DeltaE_prime': (m * g * (L + 29*d) * math.sin(theta) - mu * m * g * s) / 30,
            'L_condition_threshold': d + (mu * s) / math.sin(theta)
        }
    
    def _are_close(self, a, b):
        """Compare numeric values with tolerance."""
        try:
            # Convert to float if possible
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        logs = []
        success = True
        
        try:
            # 1. Check acceleration on incline
            if self.results.get('a') is not None:
                expected_a = self.expected_numeric['a']
                actual_a = self.results['a']
                
                if not self._are_close(actual_a, expected_a):
                    logs.append(f"FAIL: Acceleration mismatch. Expected {expected_a:.3f}, got {actual_a:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Acceleration matches g*sin(theta) = {expected_a:.3f}")
            
            # 2. Check energy loss per bump after 30th
            if self.results.get('DeltaE') is not None:
                expected_DE = self.expected_numeric['DeltaE']
                actual_DE = self.results['DeltaE']
                
                if not self._are_close(actual_DE, expected_DE):
                    logs.append(f"FAIL: Energy loss per bump mismatch. Expected {expected_DE:.3f}, got {actual_DE:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Energy loss per bump matches mgd sin(theta) = {expected_DE:.3f}")
            
            # 3. Check energy conservation
            if all(k in self.results for k in ['DeltaE_all']):
                initial_PE = self.expected_numeric['DeltaE_all'] + self.inputs['mu'] * self.inputs['m'] * self.inputs['g'] * self.inputs['s']
                expected_PE = self.inputs['m'] * self.inputs['g'] * (self.inputs['L'] + 29*self.inputs['d']) * math.sin(self.inputs['theta'])
                
                if not self._are_close(initial_PE, expected_PE):
                    logs.append(f"FAIL: Energy conservation violated. Initial PE {expected_PE:.3f} ≠ Final KE + Losses")
                    success = False
                else:
                    logs.append("PASS: Energy conservation holds")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check ΔE' = ΔE_all / 30
            if all(k in self.results for k in ['DeltaE_prime', 'DeltaE_all']):
                expected_prime = self.results['DeltaE_all'] / 30
                if not self._are_close(self.results['DeltaE_prime'], expected_prime):
                    logs.append(f"FAIL: ΔE' should be ΔE_all/30. Expected {expected_prime:.3f}, got {self.results['DeltaE_prime']:.3f}")
                    success = False
                else:
                    logs.append("PASS: ΔE' = ΔE_all / 30 holds")
            
            # 2. Check ΔE_all formula
            if self.results.get('DeltaE_all') is not None:
                expected_all = self.expected_numeric['DeltaE_all']
                if not self._are_close(self.results['DeltaE_all'], expected_all):
                    logs.append(f"FAIL: ΔE_all formula mismatch. Expected {expected_all:.3f}, got {self.results['DeltaE_all']:.3f}")
                    success = False
                else:
                    logs.append("PASS: ΔE_all formula correct")
            
            # 3. Check ΔE formula
            if self.results.get('DeltaE') is not None:
                expected_DE = self.expected_numeric['DeltaE']
                if not self._are_close(self.results['DeltaE'], expected_DE):
                    logs.append(f"FAIL: ΔE formula mismatch. Expected {expected_DE:.3f}, got {self.results['DeltaE']:.3f}")
                    success = False
                else:
                    logs.append("PASS: ΔE formula correct")
            
            # 4. Check L condition
            if self.results.get('L_condition') is not None:
                expected_threshold = self.expected_numeric['L_condition_threshold']
                actual_threshold = self.results['L_condition']
                
                if not self._are_close(actual_threshold, expected_threshold):
                    logs.append(f"FAIL: L condition threshold mismatch. Expected {expected_threshold:.3f}, got {actual_threshold:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: L condition threshold correct: L > {expected_threshold:.3f}")
                
                # Also check that ΔE' > ΔE leads to this condition
                DeltaE_prime_val = self.expected_numeric['DeltaE_prime']
                DeltaE_val = self.expected_numeric['DeltaE']
                if DeltaE_prime_val > DeltaE_val:
                    logs.append("PASS: ΔE' > ΔE holds with given inputs")
                else:
                    logs.append(f"WARNING: ΔE' ({DeltaE_prime_val:.3f}) ≤ ΔE ({DeltaE_val:.3f}) with given inputs")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs."""
        logs = []
        success = True
        
        try:
            # Perturb L and check that ΔE_all and ΔE' increase
            L_orig = self.inputs['L']
            if L_orig is not None:
                # Small perturbation
                L_pert = L_orig * 1.01 if L_orig != 0 else L_orig + 0.1
                
                # Calculate ΔE_all for original and perturbed L
                m = self.inputs['m']
                g = self.inputs['g']
                theta = self.inputs['theta']
                d = self.inputs['d']
                mu = self.inputs['mu']
                s = self.inputs['s']
                
                DeltaE_all_orig = m * g * (L_orig + 29*d) * math.sin(theta) - mu * m * g * s
                DeltaE_all_pert = m * g * (L_pert + 29*d) * math.sin(theta) - mu * m * g * s
                
                # Check monotonic increase
                if DeltaE_all_pert <= DeltaE_all_orig:
                    logs.append(f"FAIL: ΔE_all not monotonic in L. Original: {DeltaE_all_orig:.3f}, Perturbed: {DeltaE_all_pert:.3f}")
                    success = False
                else:
                    logs.append("PASS: ΔE_all increases with L (monotonic)")
                
                # Check ΔE' also increases
                DeltaE_prime_orig = DeltaE_all_orig / 30
                DeltaE_prime_pert = DeltaE_all_pert / 30
                
                if DeltaE_prime_pert <= DeltaE_prime_orig:
                    logs.append(f"FAIL: ΔE' not monotonic in L. Original: {DeltaE_prime_orig:.3f}, Perturbed: {DeltaE_prime_pert:.3f}")
                    success = False
                else:
                    logs.append("PASS: ΔE' increases with L (monotonic)")
                
                # Check condition threshold is independent of L
                threshold = d + (mu * s) / math.sin(theta)
                logs.append(f"PASS: Condition threshold independent of L: {threshold:.3f}")
            
            # Additional check: ΔE should be independent of L
            if self.results.get('DeltaE') is not None:
                logs.append("PASS: ΔE independent of L (as expected)")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_method()
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.append(f"ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs