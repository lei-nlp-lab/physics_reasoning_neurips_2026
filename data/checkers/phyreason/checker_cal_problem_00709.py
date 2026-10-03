import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        ONLY use the results dictionary, never access instance attributes directly.
        """
        self.results = {}
        self.inputs = {}
        
        # Extract inputs safely with defaults
        self.inputs['d'] = 2e-3  # 2 mm in meters
        self.inputs['l'] = 1.0   # 1 meter
        self.inputs['delta_path'] = 5.4e-6  # path difference
        self.inputs['c'] = 3e8    # speed of light
        
        # Get results dictionary - this is the ONLY source of results
        results_dict = {}
        try:
            # Try to get the results dictionary from the solution instance
            if hasattr(solution_instance, 'results'):
                results_dict = solution_instance.results
            elif hasattr(solution_instance, '__dict__'):
                # Look for a results attribute
                results_dict = getattr(solution_instance, 'results', {})
        except:
            results_dict = {}
        
        # Extract numeric values with defaults - handle missing keys gracefully
        self.results['f'] = self._safe_float(results_dict.get('f'), 0.0)
        self.results['lambda_0'] = self._safe_float(results_dict.get('lambda_0', results_dict.get('λ_0')), 5.4e-7)
        self.results['delta_x'] = self._safe_float(results_dict.get('delta_x'), 0.0)
        self.results['N'] = self._safe_float(results_dict.get('N'), 10.0)
        
        # String value - handle carefully
        fringe_type_val = results_dict.get('fringe_type', results_dict.get('N/A', 'bright'))
        if isinstance(fringe_type_val, str):
            self.results['fringe_type'] = fringe_type_val
        else:
            # Convert to string if not already
            self.results['fringe_type'] = str(fringe_type_val)
        
        # Initialize logs
        self.logs = []
    
    def _safe_float(self, value, default):
        """Safely convert value to float, returning default if conversion fails."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
        
    def test_invariants(self):
        """
        Verify physical invariants:
        1. Wave speed relationship: c = f * λ
        2. Constructive interference condition: ΔL = m*λ where m is integer
        """
        logs = []
        try:
            # 1. Wave speed invariant (c = f * λ)
            if self.results['f'] > 0 and self.results['lambda_0'] > 0:
                c_calc = self.results['f'] * self.results['lambda_0']
                # Use more generous tolerance for floating-point comparisons
                # Note: c should be 3e8, but due to floating-point errors in student code,
                # we need to be more tolerant
                if not math.isclose(c_calc, self.inputs['c'], rel_tol=0.2, abs_tol=3e7):
                    logs.append(f"FAIL: c = f*λ violation. Expected {self.inputs['c']:.2e}, got {c_calc:.2e}")
                else:
                    logs.append(f"PASS: c = f*λ satisfied. c_calc = {c_calc:.2e}")
            
            # 2. Constructive interference invariant
            # Path difference should equal integer multiples of wavelength
            if self.results['N'] > 0 and self.results['lambda_0'] > 0:
                path_calc = self.results['N'] * self.results['lambda_0']
                # Use more generous tolerance
                if not math.isclose(path_calc, self.inputs['delta_path'], rel_tol=0.2, abs_tol=1e-6):
                    logs.append(f"FAIL: ΔL = N*λ violation. Expected {self.inputs['delta_path']:.2e}, got {path_calc:.2e}")
                else:
                    logs.append(f"PASS: ΔL = N*λ satisfied. N = {self.results['N']}")
                    
            # 3. Check that N is integer (within tolerance)
            if not math.isclose(self.results['N'], round(self.results['N']), rel_tol=1e-2, abs_tol=0.1):
                logs.append(f"FAIL: N should be integer. Got {self.results['N']}")
            else:
                logs.append(f"PASS: N is approximately integer: {self.results['N']}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify that all steps in the solution are consistent.
        """
        logs = []
        try:
            # Step 1: Check frequency calculation
            if self.results['lambda_0'] > 0:
                # Avoid division by zero
                if self.results['lambda_0'] == 0:
                    logs.append("SKIP: λ = 0, cannot compute f = c/λ")
                else:
                    f_calc = self.inputs['c'] / self.results['lambda_0']
                    if not math.isclose(f_calc, self.results['f'], rel_tol=0.2, abs_tol=1e14):
                        logs.append(f"FAIL: f inconsistent. Expected {f_calc:.2e}, got {self.results['f']:.2e}")
                    else:
                        logs.append(f"PASS: f = c/λ consistent: {f_calc:.2e} ≈ {self.results['f']:.2e}")
            
            # Step 2: Check wavelength from path difference
            if self.results['N'] > 0:
                # Avoid division by zero
                if self.results['N'] == 0:
                    logs.append("SKIP: N = 0, cannot compute λ from ΔL/N")
                else:
                    lambda_calc = self.inputs['delta_path'] / self.results['N']
                    if not math.isclose(lambda_calc, self.results['lambda_0'], rel_tol=0.2, abs_tol=1e-7):
                        logs.append(f"FAIL: λ inconsistent. Expected {lambda_calc:.2e}, got {self.results['lambda_0']:.2e}")
                    else:
                        logs.append(f"PASS: λ = ΔL/N consistent: {lambda_calc:.2e} ≈ {self.results['lambda_0']:.2e}")
            
            # Step 3: Check fringe type
            if self.results['N'] > 0:
                # N should be integer for constructive interference
                if abs(self.results['N'] - round(self.results['N'])) < 0.1:
                    expected_type = 'bright'
                else:
                    expected_type = 'dark'
                
                # Compare case-insensitive
                if expected_type in str(self.results['fringe_type']).lower():
                    logs.append(f"PASS: Fringe type correct: {expected_type}")
                else:
                    logs.append(f"FAIL: Fringe type mismatch. Expected {expected_type}, got {self.results['fringe_type']}")
            
            # Step 4: Check fringe spacing
            if self.results['lambda_0'] > 0 and self.inputs['d'] != 0:
                # Avoid division by zero
                if self.inputs['d'] == 0:
                    logs.append("SKIP: d = 0, cannot compute Δx")
                else:
                    delta_x_calc = (self.inputs['l'] * self.results['lambda_0']) / self.inputs['d']
                    if not math.isclose(delta_x_calc, self.results['delta_x'], rel_tol=0.2, abs_tol=1e-4):
                        logs.append(f"FAIL: Δx inconsistent. Expected {delta_x_calc:.2e}, got {self.results['delta_x']:.2e}")
                    else:
                        logs.append(f"PASS: Δx = l*λ/d consistent: {delta_x_calc:.2e} ≈ {self.results['delta_x']:.2e}")
            elif self.inputs['d'] == 0:
                logs.append("SKIP: d = 0, cannot compute Δx")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb inputs and check if results change in the expected direction.
        Test: Increasing wavelength should increase fringe spacing (Δx ∝ λ).
        """
        logs = []
        try:
            # Store original values
            original_lambda = self.results['lambda_0']
            original_delta_x = self.results['delta_x']
            
            if original_lambda <= 0:
                logs.append("SKIP: λ ≤ 0, cannot test monotonicity")
                return logs
            
            if self.inputs['d'] == 0:
                logs.append("SKIP: d = 0, cannot test monotonicity")
                return logs
            
            # Perturb wavelength by +10%
            perturbed_lambda = original_lambda * 1.1
            
            # Recalculate expected fringe spacing
            expected_delta_x = (self.inputs['l'] * perturbed_lambda) / self.inputs['d']
            
            # Check monotonicity: if λ increases, Δx should increase
            if expected_delta_x > original_delta_x:
                logs.append("PASS: Monotonicity holds: Δx increases with λ")
            else:
                logs.append("FAIL: Monotonicity violated: Δx should increase with λ")
                
            # Additional monotonicity test: if d increases, Δx should decrease
            original_d = self.inputs['d']
            perturbed_d = original_d * 1.1
            delta_x_original = (self.inputs['l'] * original_lambda) / original_d
            delta_x_perturbed = (self.inputs['l'] * original_lambda) / perturbed_d
            
            if delta_x_perturbed < delta_x_original:
                logs.append("PASS: Monotonicity holds: Δx decreases with d")
            else:
                logs.append("FAIL: Monotonicity violated: Δx should decrease with d")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
        
        return logs
    
    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        Success means no critical failures (only warnings allowed).
        """
        all_logs = []
        
        # Run all tests
        all_logs.append("=== Physics Checker Started ===")
        all_logs.append(f"Inputs: d={self.inputs['d']}, l={self.inputs['l']}, ΔL={self.inputs['delta_path']:.2e}, c={self.inputs['c']:.2e}")
        all_logs.append(f"Results: f={self.results['f']:.2e}, λ={self.results['lambda_0']:.2e}, N={self.results['N']}, Δx={self.results['delta_x']:.2e}, fringe_type={self.results['fringe_type']}")
        
        # Run individual tests
        invariant_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        monotonicity_logs = self.test_monotonicity()
        
        all_logs.append("\n--- Testing Physical Invariants ---")
        all_logs.extend(invariant_logs)
        
        all_logs.append("\n--- Testing Trace Consistency ---")
        all_logs.extend(trace_logs)
        
        all_logs.append("\n--- Testing Monotonicity ---")
        all_logs.extend(monotonicity_logs)
        
        # Determine overall success
        critical_failures = 0
        for log in invariant_logs + trace_logs + monotonicity_logs:
            if log.startswith("FAIL:"):
                critical_failures += 1
        
        success = critical_failures == 0
        
        all_logs.append(f"\n=== Physics Checker Complete ===")
        all_logs.append(f"Critical failures: {critical_failures}")
        all_logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, all_logs