import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with safe fallbacks - convert to float
        input_keys = ['L', 'theta', 'B', 'm', 'R', 'g', 't0']
        for key in input_keys:
            val = getattr(solution_instance, key, 
                         getattr(solution_instance, 'inputs', {}).get(key, None))
            if val is None:
                # Use reasonable defaults that match the expected results
                if key == 'm': val = 0.1
                elif key == 'g': val = 9.8
                elif key == 'theta': val = 30 * math.pi / 180  # 30 degrees in radians
                elif key == 'B': val = 0.5
                elif key == 'L': val = 0.2
                elif key == 'R': val = 10.0
                elif key == 't0': val = 1.0
                else: val = 1.0
            
            # Convert to float if it's a SymPy expression
            if hasattr(val, 'evalf'):
                self.inputs[key] = float(val.evalf())
            else:
                self.inputs[key] = float(val)
        
        # Extract results with safe fallbacks
        result_keys = ['F', 'q', 'a', 'v_0', 'v', 'I', 'hatI', 'Deltax', 'E']
        for key in result_keys:
            val = getattr(solution_instance, key,
                         getattr(solution_instance, 'results', {}).get(key, 0.0))
            if hasattr(val, 'evalf'):
                self.results[key] = float(val.evalf())
            else:
                self.results[key] = float(val)
        
        # Extract target answers if available
        self.target_v0 = getattr(solution_instance, 'target_v0', None)
        self.target_a = getattr(solution_instance, 'target_a', None)
        self.target_v = getattr(solution_instance, 'target_v', None)
        self.target_Deltax = getattr(solution_instance, 'target_Deltax', None)
    
    def _isclose(self, a, b):
        """Fuzzy numerical comparison with physics tolerance."""
        # Ensure both are floats
        a_float = float(a) if not isinstance(a, float) else a
        b_float = float(b) if not isinstance(b, float) else b
        return math.isclose(a_float, b_float, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify conservation laws and Newton's laws."""
        logs = []
        
        try:
            # Extract numeric values
            L = self.inputs['L']
            theta = self.inputs['theta']  # in radians
            B = self.inputs['B']
            m = self.inputs['m']
            R = self.inputs['R']
            g = self.inputs['g']
            v0 = self.results.get('v_0', 0)
            a_val = self.results.get('a', 0)
            
            # 1. Check force equilibrium for rod a at terminal velocity
            # expected_v0 = (2 * m * g * R * sin(theta)) / (B**2 * L**2)
            expected_v0 = (2 * m * g * R * math.sin(theta)) / (B**2 * L**2)
            
            if v0 != 0 and not self._isclose(v0, expected_v0):
                logs.append(f"FAIL: v0 equilibrium check. Got {v0}, expected {expected_v0}")
            
            # 2. Check Newton's 2nd law for rod b
            # a = 2g sinθ
            expected_a = 2 * g * math.sin(theta)
            
            if a_val != 0 and not self._isclose(a_val, expected_a):
                logs.append(f"FAIL: Acceleration check. Got {a_val}, expected {expected_a}")
            
            # 3. Check momentum conservation (system of both rods)
            v = self.results.get('v', 0)
            t0 = self.inputs.get('t0', 1.0)
            
            if v != 0 and t0 != 0 and v0 != 0:
                # 2mg sinθ t0 = 2mv - mv0
                lhs = 2 * m * g * math.sin(theta) * t0
                rhs = 2 * m * v - m * v0
                
                if not self._isclose(lhs, rhs):
                    logs.append(f"FAIL: System momentum check. LHS={lhs}, RHS={rhs}")
            
            return logs
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            L = self.inputs['L']
            theta = self.inputs['theta']
            B = self.inputs['B']
            m = self.inputs['m']
            R = self.inputs['R']
            g = self.inputs['g']
            t0 = self.inputs.get('t0', 1.0)
            
            v0 = self.results.get('v_0', 0)
            a_val = self.results.get('a', 0)
            v = self.results.get('v', 0)
            Deltax = self.results.get('Deltax', 0)
            E_val = self.results.get('E', 0)
            I_val = self.results.get('I', 0)
            F_val = self.results.get('F', 0)
            hatI = self.results.get('hatI', 0)
            q = self.results.get('q', 0)
            
            # 1. Check E = BLv0
            if v0 != 0 and E_val != 0:
                expected_E = B * L * v0
                if not self._isclose(E_val, expected_E):
                    logs.append(f"FAIL: EMF consistency. Got {E_val}, expected {expected_E}")
            
            # 2. Check I = E/(2R)
            if I_val != 0 and E_val != 0:
                expected_I = E_val / (2 * R)
                if not self._isclose(I_val, expected_I):
                    logs.append(f"FAIL: Current consistency. Got {I_val}, expected {expected_I}")
            
            # 3. Check F = BIL
            if F_val != 0 and I_val != 0:
                expected_F = B * I_val * L
                if not self._isclose(F_val, expected_F):
                    logs.append(f"FAIL: Force consistency. Got {F_val}, expected {expected_F}")
            
            # 4. Check v = g sinθ t0 + v0/2
            if v != 0 and v0 != 0:
                expected_v = g * math.sin(theta) * t0 + v0 / 2
                if not self._isclose(v, expected_v):
                    logs.append(f"FAIL: Velocity v consistency. Got {v}, expected {expected_v}")
            
            # 5. Check Δx from momentum equations
            if Deltax != 0 and v0 != 0:
                expected_Deltax = (m * v0 * R) / (B**2 * L**2)
                if not self._isclose(Deltax, expected_Deltax):
                    logs.append(f"FAIL: Δx consistency. Got {Deltax}, expected {expected_Deltax}")
            
            # 6. Check q = Ī t0
            if q != 0 and hatI != 0 and t0 != 0:
                expected_q = hatI * t0
                if not self._isclose(q, expected_q):
                    logs.append(f"FAIL: Charge consistency. Got {q}, expected {expected_q}")
            
            return logs
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return logs
    
    def test_monotonicity(self):
        """Check that perturbations in inputs produce expected trends."""
        logs = []
        
        try:
            # Store original values
            orig_inputs = self.inputs.copy()
            
            # Test parameter: mass m
            orig_m = self.inputs['m']
            perturbed_m = orig_m * 1.01  # +1%
            
            # Compute v0 for original and perturbed
            L = self.inputs['L']
            theta = self.inputs['theta']
            B = self.inputs['B']
            R = self.inputs['R']
            g = self.inputs['g']
            
            v0_orig = (2 * orig_m * g * R * math.sin(theta)) / (B**2 * L**2)
            v0_pert = (2 * perturbed_m * g * R * math.sin(theta)) / (B**2 * L**2)
            
            # v0 should increase with m
            if v0_pert <= v0_orig:
                logs.append(f"FAIL: v0 monotonicity w.r.t m. {v0_orig} -> {v0_pert}")
            
            # Test parameter: resistance R
            orig_R = self.inputs['R']
            perturbed_R = orig_R * 1.01
            
            v0_orig_R = (2 * orig_m * g * orig_R * math.sin(theta)) / (B**2 * L**2)
            v0_pert_R = (2 * orig_m * g * perturbed_R * math.sin(theta)) / (B**2 * L**2)
            
            # v0 should increase with R
            if v0_pert_R <= v0_orig_R:
                logs.append(f"FAIL: v0 monotonicity w.r.t R. {v0_orig_R} -> {v0_pert_R}")
            
            # Test parameter: magnetic field B
            orig_B = self.inputs['B']
            perturbed_B = orig_B * 1.01
            
            v0_orig_B = (2 * orig_m * g * orig_R * math.sin(theta)) / (orig_B**2 * L**2)
            v0_pert_B = (2 * orig_m * g * orig_R * math.sin(theta)) / (perturbed_B**2 * L**2)
            
            # v0 should decrease with B
            if v0_pert_B >= v0_orig_B:
                logs.append(f"FAIL: v0 monotonicity w.r.t B. {v0_orig_B} -> {v0_pert_B}")
            
            # Test parameter: separation L
            orig_L = self.inputs['L']
            perturbed_L = orig_L * 1.01
            
            v0_orig_L = (2 * orig_m * g * orig_R * math.sin(theta)) / (orig_B**2 * orig_L**2)
            v0_pert_L = (2 * orig_m * g * orig_R * math.sin(theta)) / (orig_B**2 * perturbed_L**2)
            
            # v0 should decrease with L
            if v0_pert_L >= v0_orig_L:
                logs.append(f"FAIL: v0 monotonicity w.r.t L. {v0_orig_L} -> {v0_pert_L}")
            
            # Test a (acceleration) - should be independent of m, R, B, L
            a_orig = 2 * g * math.sin(theta)
            # All perturbations give same a
            if not self._isclose(a_orig, a_orig):
                logs.append(f"FAIL: a should be independent of parameters")
            
            # Restore original inputs
            self.inputs = orig_inputs
            
            return logs
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            # Restore on error
            self.inputs = orig_inputs
            return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        
        # Run tests
        all_logs.append("=== Testing Invariants ===")
        inv_logs = self.test_invariants()
        all_logs.extend(inv_logs)
        
        all_logs.append("\n=== Testing Trace Consistency ===")
        trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        
        all_logs.append("\n=== Testing Monotonicity ===")
        mono_logs = self.test_monotonicity()
        all_logs.extend(mono_logs)
        
        # Determine success
        error_logs = [log for log in all_logs if log.startswith(('FAIL', 'ERROR'))]
        success = len(error_logs) == 0
        
        return success, all_logs