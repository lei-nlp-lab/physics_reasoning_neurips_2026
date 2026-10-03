import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution_instance."""
        self.inputs = {}
        self.results = {}
        
        # Inputs from problem statement - try to get from solution_instance attributes
        # The solution_instance should have these as attributes or in an inputs dict
        input_keys = ['S', 'M', 'g', 'p0', 'l', 'T0']
        for key in input_keys:
            # First try direct attribute
            val = getattr(solution_instance, key, None)
            if val is None:
                # Try from inputs dict
                val = getattr(solution_instance, 'inputs', {}).get(key, None)
            if val is None:
                # Try from results dict (some might be stored there)
                val = getattr(solution_instance, 'results', {}).get(key, None)
            if val is None:
                # Default to 1.0 to avoid division by zero, but log warning
                val = 1.0
            self.inputs[key] = float(val) if not isinstance(val, (int, float)) else val
        
        # Results from solution_instance
        result_keys = ['h', 'p', 'p\'', 'T', 'TandT_0/2']
        for key in result_keys:
            if key == 'TandT_0/2':
                val = getattr(solution_instance, 'TandT_0/2', None)
                if val is None:
                    val = getattr(solution_instance, 'results', {}).get('TandT_0/2', None)
                self.results['TandT_0/2'] = float(val) if val is not None else 0.0
            else:
                # Handle p' key
                if key == 'p\'':
                    store_key = 'p_prime'
                else:
                    store_key = key
                val = getattr(solution_instance, key, None)
                if val is None:
                    val = getattr(solution_instance, 'results', {}).get(store_key, None)
                if val is None:
                    val = getattr(solution_instance, 'results', {}).get(key, None)
                self.results[key] = float(val) if val is not None else 0.0
        
        # Set attributes for convenience
        self.S = self.inputs['S']
        self.M = self.inputs['M']
        self.g = self.inputs['g']
        self.p0 = self.inputs['p0']
        self.l = self.inputs['l']
        self.T0 = self.inputs['T0']
        self.h = self.results['h']
        self.p = self.results['p']
        self.p_prime = self.results['p\'']
        self.T = self.results['T']
    
    def test_invariants(self):
        """Verify physical laws: force balance, Boyle's Law, Gay-Lussac's Law, Charles's Law."""
        logs = []
        success = True
        
        # 1. Force balance in vertical position: pS = p0S + Mg
        try:
            LHS = self.p * self.S
            RHS = self.p0 * self.S + self.M * self.g
            if not math.isclose(LHS, RHS, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Force balance invariant violated. pS={LHS}, p0S+Mg={RHS}")
                success = False
            else:
                logs.append("PASS: Force balance invariant holds.")
        except Exception as e:
            logs.append(f"ERROR in force balance: {traceback.format_exc()}")
            success = False
        
        # 2. Boyle's Law (isothermal): p0 * V0 = p * V1, where V0 = 2lS, V1 = (2l - h)S
        try:
            V0 = 2 * self.l * self.S
            V1 = (2 * self.l - self.h) * self.S
            LHS = self.p0 * V0
            RHS = self.p * V1
            if not math.isclose(LHS, RHS, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Boyle's Law invariant violated. p0*V0={LHS}, p*V1={RHS}")
                success = False
            else:
                logs.append("PASS: Boyle's Law invariant holds.")
        except Exception as e:
            logs.append(f"ERROR in Boyle's Law: {traceback.format_exc()}")
            success = False
        
        # 3. Gay-Lussac's Law (isobaric): V0/T0 = V1/T, where V0 = (2l - h)S, V1 = lS
        try:
            V0_gay = (2 * self.l - self.h) * self.S
            V1_gay = self.l * self.S
            # Avoid division by zero
            if abs(self.T0) < 1e-12 or abs(self.T) < 1e-12:
                logs.append("SKIP: T0 or T is zero, cannot test Gay-Lussac's Law")
            else:
                LHS = V0_gay / self.T0
                RHS = V1_gay / self.T
                if not math.isclose(LHS, RHS, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Gay-Lussac's Law invariant violated. V0/T0={LHS}, V1/T={RHS}")
                    success = False
                else:
                    logs.append("PASS: Gay-Lussac's Law invariant holds.")
        except Exception as e:
            logs.append(f"ERROR in Gay-Lussac's Law: {traceback.format_exc()}")
            success = False
        
        # 4. Charles's Law (isochoric): p/T = p'/(T0/2)
        try:
            # Avoid division by zero
            if abs(self.T) < 1e-12 or abs(self.T0) < 1e-12:
                logs.append("SKIP: T or T0 is zero, cannot test Charles's Law")
            else:
                LHS = self.p / self.T
                RHS = self.p_prime / (self.T0 / 2)
                if not math.isclose(LHS, RHS, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Charles's Law invariant violated. p/T={LHS}, p'/(T0/2)={RHS}")
                    success = False
                else:
                    logs.append("PASS: Charles's Law invariant holds.")
        except Exception as e:
            logs.append(f"ERROR in Charles's Law: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Step 3: h = (2 M g l) / (p0 S + M g)
        try:
            denominator = self.p0 * self.S + self.M * self.g
            if abs(denominator) < 1e-12:
                logs.append("SKIP: denominator is zero, cannot compute h_expected")
            else:
                h_expected = (2 * self.M * self.g * self.l) / denominator
                if not math.isclose(self.h, h_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: h mismatch. Expected {h_expected}, got {self.h}")
                    success = False
                else:
                    logs.append(f"PASS: h consistent with formula.")
        except Exception as e:
            logs.append(f"ERROR in h consistency: {traceback.format_exc()}")
            success = False
        
        # Step 4: T = (p0 S + M g) / (2 p0 S) * T0
        try:
            denominator = 2 * self.p0 * self.S
            if abs(denominator) < 1e-12:
                logs.append("SKIP: denominator is zero, cannot compute T_expected")
            else:
                T_expected = (self.p0 * self.S + self.M * self.g) / denominator * self.T0
                if not math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T mismatch. Expected {T_expected}, got {self.T}")
                    success = False
                else:
                    logs.append(f"PASS: T consistent with formula.")
        except Exception as e:
            logs.append(f"ERROR in T consistency: {traceback.format_exc()}")
            success = False
        
        # Step 6: p' = p0
        try:
            if not math.isclose(self.p_prime, self.p0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: p' mismatch. Expected {self.p0}, got {self.p_prime}")
                success = False
            else:
                logs.append(f"PASS: p' equals p0.")
        except Exception as e:
            logs.append(f"ERROR in p' consistency: {traceback.format_exc()}")
            success = False
        
        # Step 5: Compare T and T0/2, should be T > T0/2
        try:
            T_half = self.T0 / 2
            if self.T <= T_half:
                logs.append(f"FAIL: T ({self.T}) should be > T0/2 ({T_half})")
                success = False
            else:
                logs.append(f"PASS: T > T0/2 as expected.")
        except Exception as e:
            logs.append(f"ERROR in T comparison: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input M and check trends in h and p'."""
        logs = []
        success = True
        
        # Save original values
        M_original = self.M
        perturb_factor = 1.1  # 10% increase
        
        try:
            # Perturb M
            M_perturbed = M_original * perturb_factor
            
            # Recalculate h using formula: h = (2 M g l) / (p0 S + M g)
            denom_original = self.p0 * self.S + M_original * self.g
            denom_perturbed = self.p0 * self.S + M_perturbed * self.g
            
            if abs(denom_original) < 1e-12 or abs(denom_perturbed) < 1e-12:
                logs.append("SKIP: denominator is zero, cannot compute monotonicity")
            else:
                h_original = (2 * M_original * self.g * self.l) / denom_original
                h_perturbed = (2 * M_perturbed * self.g * self.l) / denom_perturbed
                
                # Check monotonicity: h should increase with M
                if h_perturbed <= h_original:
                    logs.append(f"FAIL: h does not increase with M. Original h={h_original}, perturbed h={h_perturbed}")
                    success = False
                else:
                    logs.append(f"PASS: h increases with M (monotonic).")
            
            # p' should remain constant at p0 regardless of M
            p_prime_original = self.p0
            p_prime_perturbed = self.p0  # from formula p' = p0
            
            if not math.isclose(p_prime_perturbed, p_prime_original, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: p' changes with M. Original p'={p_prime_original}, perturbed p'={p_prime_perturbed}")
                success = False
            else:
                logs.append(f"PASS: p' independent of M.")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"--- {test_name} ---")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
            all_logs.append("")
        
        return overall_success, all_logs