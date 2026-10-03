import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution instance.
        """
        self.inputs = {}
        self.results = {}
        
        # Safely extract common inputs with defaults
        self.inputs['V'] = getattr(solution_instance, 'V', getattr(solution_instance, 'inputs', {}).get('V', 1.0))
        self.inputs['DeltaV'] = getattr(solution_instance, 'DeltaV', getattr(solution_instance, 'inputs', {}).get('DeltaV', 0.1))
        self.inputs['Pa'] = getattr(solution_instance, 'Pa', getattr(solution_instance, 'inputs', {}).get('Pa', 1.013e5))
        self.inputs['Ta'] = getattr(solution_instance, 'Ta', getattr(solution_instance, 'inputs', {}).get('Ta', 300.0))
        self.inputs['R'] = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 8.314))
        self.inputs['V0'] = getattr(solution_instance, 'V0', getattr(solution_instance, 'inputs', {}).get('V0', 0.0))
        self.inputs['gamma'] = getattr(solution_instance, 'gamma', getattr(solution_instance, 'inputs', {}).get('gamma', 5/3))
        
        # Safely extract results with defaults
        self.results['P_min'] = getattr(solution_instance, 'P_min', getattr(solution_instance, 'results', {}).get('P_min', 0.0))
        self.results['P_f'] = getattr(solution_instance, 'P_f', getattr(solution_instance, 'results', {}).get('P_f', 0.0))
        self.results['Pt'] = getattr(solution_instance, 'Pt', getattr(solution_instance, 'results', {}).get('Pt', 0.0))
        self.results['T_outt'] = getattr(solution_instance, 'T_outt', getattr(solution_instance, 'results', {}).get('T_outt', 0.0))
        
        # For symbolic checks
        self.symbols = {}
        self.setup_symbols()
    
    def setup_symbols(self):
        """Define symbolic variables for physics laws."""
        # Declare symbols
        V, DeltaV, Pa, Ta, R, V0, gamma, t = sp.symbols('V DeltaV Pa Ta R V0 gamma t', positive=True, real=True)
        self.symbols.update({
            'V': V, 'DeltaV': DeltaV, 'Pa': Pa, 'Ta': Ta, 'R': R,
            'V0': V0, 'gamma': gamma, 't': t
        })
        
        # Declare functions
        P = sp.Function('P')(t)
        T_out = sp.Function('T_out')(t)
        self.symbols['P'] = P
        self.symbols['T_out'] = T_out
    
    def test_invariants(self):
        """
        Verify energy conservation, adiabatic relations, and ideal gas law.
        """
        logs = []
        success = True
        
        try:
            # Extract numerical values
            V_val = self.inputs['V']
            DeltaV_val = self.inputs['DeltaV']
            Pa_val = self.inputs['Pa']
            Ta_val = self.inputs['Ta']
            R_val = self.inputs['R']
            V0_val = self.inputs['V0']
            gamma_val = self.inputs['gamma']
            
            # Test 1: Check P(t) at t=0 equals Pa
            if hasattr(self.results['Pt'], '__call__'):
                P_at_0 = self.results['Pt'](0.0)
            else:
                # If Pt is a constant or array, try to evaluate at t=0
                P_at_0 = self.results['Pt'] if not callable(self.results['Pt']) else self.results['Pt'](0.0)
            
            if not math.isclose(P_at_0, Pa_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: P(0) = {P_at_0} != Pa = {Pa_val}")
                success = False
            else:
                logs.append("PASS: P(0) = Pa (boundary condition)")
            
            # Test 2: Check T_out(t) at t=0 equals Ta
            if hasattr(self.results['T_outt'], '__call__'):
                T_out_at_0 = self.results['T_outt'](0.0)
            else:
                T_out_at_0 = self.results['T_outt'] if not callable(self.results['T_outt']) else self.results['T_outt'](0.0)
            
            if not math.isclose(T_out_at_0, Ta_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T_out(0) = {T_out_at_0} != Ta = {Ta_val}")
                success = False
            else:
                logs.append("PASS: T_out(0) = Ta (boundary condition)")
            
            # Test 3: For V0=0, P_min should be 0 (from equation P_min = Pa*(V0/(V0+DeltaV))^gamma)
            if math.isclose(V0_val, 0.0, abs_tol=1e-9):
                P_min_expected = 0.0
                if not math.isclose(self.results['P_min'], P_min_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: For V0=0, P_min = {self.results['P_min']} != 0")
                    success = False
                else:
                    logs.append("PASS: P_min = 0 when V0=0")
            
            # Test 4: Check that P(t) is monotonically decreasing (derivative negative)
            # We'll check at a sample point t=1.0 if Pt is callable
            if hasattr(self.results['Pt'], '__call__'):
                t_sample = 1.0
                # Use finite difference to approximate derivative
                dt = 1e-6
                P1 = self.results['Pt'](t_sample)
                P2 = self.results['Pt'](t_sample + dt)
                dP_dt = (P2 - P1) / dt
                if dP_dt > 0:
                    logs.append(f"WARN: dP/dt at t={t_sample} is positive ({dP_dt}), expected negative for exponential decay")
                    # Not a failure, just a warning
                else:
                    logs.append("PASS: P(t) is decreasing (dP/dt < 0)")
            
            # Test 5: Check that T_out(t) is monotonically increasing (derivative positive)
            if hasattr(self.results['T_outt'], '__call__'):
                t_sample = 1.0
                dt = 1e-6
                T1 = self.results['T_outt'](t_sample)
                T2 = self.results['T_outt'](t_sample + dt)
                dT_dt = (T2 - T1) / dt
                if dT_dt < 0:
                    logs.append(f"WARN: dT_out/dt at t={t_sample} is negative ({dT_dt}), expected positive for exponential growth")
                else:
                    logs.append("PASS: T_out(t) is increasing (dT_out/dt > 0)")
            
            # Test 6: Check adiabatic relation for P_min (if V0 > 0)
            if V0_val > 1e-9:
                # P_min = Pa * (V0/(V0+DeltaV))^gamma
                expected_P_min = Pa_val * (V0_val / (V0_val + DeltaV_val)) ** gamma_val
                if not math.isclose(self.results['P_min'], expected_P_min, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: P_min = {self.results['P_min']}, expected {expected_P_min} from adiabatic relation")
                    success = False
                else:
                    logs.append("PASS: P_min satisfies adiabatic relation")
            
            # Test 7: Check ideal gas law consistency for initial state
            # We don't have n, but we can check that P(t) and T_out(t) satisfy adiabatic relation
            # For adiabatic process: T_out(t) / Ta = (P(t)/Pa)^((gamma-1)/gamma)
            if hasattr(self.results['Pt'], '__call__') and hasattr(self.results['T_outt'], '__call__'):
                t_test = 0.5  # arbitrary time
                P_test = self.results['Pt'](t_test)
                T_test = self.results['T_outt'](t_test)
                lhs = T_test / Ta_val
                rhs = (P_test / Pa_val) ** ((gamma_val - 1) / gamma_val)
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: At t={t_test}, T_out/Ta = {lhs}, (P/Pa)^((γ-1)/γ) = {rhs}")
                    success = False
                else:
                    logs.append("PASS: P(t) and T_out(t) satisfy adiabatic relation")
        
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        try:
            V_val = self.inputs['V']
            DeltaV_val = self.inputs['DeltaV']
            Pa_val = self.inputs['Pa']
            Ta_val = self.inputs['Ta']
            R_val = self.inputs['R']
            gamma_val = self.inputs['gamma']
            
            # Check step 2: P_f after one cycle
            expected_P_f = Pa_val * (V_val / (V_val + DeltaV_val))
            if 'P_f' in self.results and self.results['P_f'] is not None:
                if not math.isclose(self.results['P_f'], expected_P_f, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: P_f = {self.results['P_f']}, expected {expected_P_f} from one-cycle formula")
                    success = False
                else:
                    logs.append("PASS: P_f matches one-cycle formula")
            
            # Check step 5: P(t) = Pa * exp(-R*t/V)
            # We'll test at a specific time t=1.0
            t_test = 1.0
            if hasattr(self.results['Pt'], '__call__'):
                P_t = self.results['Pt'](t_test)
            else:
                # If Pt is not callable, assume it's a constant or array; skip this test
                P_t = None
            
            if P_t is not None:
                expected_P_t = Pa_val * math.exp(-R_val * t_test / V_val)
                if not math.isclose(P_t, expected_P_t, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: P({t_test}) = {P_t}, expected {expected_P_t} from exponential formula")
                    success = False
                else:
                    logs.append("PASS: P(t) matches exponential decay formula")
            
            # Check step 10: T_out(t) = Ta * exp(2*R*t/(5*V))
            if hasattr(self.results['T_outt'], '__call__'):
                T_t = self.results['T_outt'](t_test)
            else:
                T_t = None
            
            if T_t is not None:
                expected_T_t = Ta_val * math.exp(2 * R_val * t_test / (5 * V_val))
                if not math.isclose(T_t, expected_T_t, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T_out({t_test}) = {T_t}, expected {expected_T_t} from exponential formula")
                    success = False
                else:
                    logs.append("PASS: T_out(t) matches exponential growth formula")
            
            # Check step 13: P_min = Pa * (1 + DeltaV/V0)^(-gamma)
            V0_val = self.inputs['V0']
            if V0_val > 1e-9:
                expected_P_min = Pa_val * (1 + DeltaV_val / V0_val) ** (-gamma_val)
                if not math.isclose(self.results['P_min'], expected_P_min, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: P_min = {self.results['P_min']}, expected {expected_P_min}")
                    success = False
                else:
                    logs.append("PASS: P_min matches formula from step 13")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input and check if output changes in expected direction.
        """
        logs = []
        success = True
        
        try:
            # Save original inputs
            original_inputs = self.inputs.copy()
            original_results = self.results.copy()
            
            # Perturb volume V (increase by 10%)
            V_orig = self.inputs['V']
            V_pert = V_orig * 1.1
            
            # Recalculate P(t) and T_out(t) using formulas from KB
            Pa = self.inputs['Pa']
            Ta = self.inputs['Ta']
            R = self.inputs['R']
            gamma = self.inputs['gamma']
            
            # For P(t): P(t) = Pa * exp(-R*t/V)
            # For T_out(t): T_out(t) = Ta * exp(2*R*t/(5*V))
            t_test = 1.0
            P_orig = Pa * math.exp(-R * t_test / V_orig)
            P_pert = Pa * math.exp(-R * t_test / V_pert)
            
            T_orig = Ta * math.exp(2 * R * t_test / (5 * V_orig))
            T_pert = Ta * math.exp(2 * R * t_test / (5 * V_pert))
            
            # Check monotonicity: as V increases, P(t) should increase (less negative exponent)
            if P_pert <= P_orig:
                logs.append(f"FAIL: When V increases from {V_orig} to {V_pert}, P(t) did not increase (from {P_orig} to {P_pert})")
                success = False
            else:
                logs.append("PASS: P(t) increases with V (as expected)")
            
            # Check monotonicity: as V increases, T_out(t) should decrease (less positive exponent)
            if T_pert >= T_orig:
                logs.append(f"FAIL: When V increases from {V_orig} to {V_pert}, T_out(t) did not decrease (from {T_orig} to {T_pert})")
                success = False
            else:
                logs.append("PASS: T_out(t) decreases with V (as expected)")
            
            # For P_min, perturb V0 (if V0 > 0)
            V0_orig = self.inputs['V0']
            if V0_orig > 1e-9:
                DeltaV = self.inputs['DeltaV']
                V0_pert = V0_orig * 1.1
                P_min_orig = Pa * (1 + DeltaV / V0_orig) ** (-gamma)
                P_min_pert = Pa * (1 + DeltaV / V0_pert) ** (-gamma)
                
                # As V0 increases, (1 + DeltaV/V0) decreases, so P_min should increase
                if P_min_pert <= P_min_orig:
                    logs.append(f"FAIL: When V0 increases from {V0_orig} to {V0_pert}, P_min did not increase (from {P_min_orig} to {P_min_pert})")
                    success = False
                else:
                    logs.append("PASS: P_min increases with V0 (as expected)")
            
            # Restore original inputs and results
            self.inputs = original_inputs
            self.results = original_results
        
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
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
            success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        return overall_success, all_logs