import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.B = getattr(solution_instance, 'B', 0.1)
        self.n = getattr(solution_instance, 'n', 100)
        self.l_AB = getattr(solution_instance, 'l_AB', 0.2)
        self.l_BC = getattr(solution_instance, 'l_BC', 0.5)
        self.omega = getattr(solution_instance, 'omega', 100 * math.pi)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.E_m = results.get('E_m', results.get('E_m', 0.0))
        self.e_expr = results.get('e', None)  # Could be sympy expression or string
        self.overlineE = results.get('overlineE', results.get('\\overline{E}', 0.0))
        self.S = results.get('S', results.get('S_area', 0.0))
        
        # Symbolic variables for verification
        self.t = sp.symbols('t', real=True)
        self.expected_S = self.l_AB * self.l_BC
        self.expected_E_m = self.n * self.B * self.expected_S * self.omega
        
        # Expected expressions based on KB
        self.expected_e = self.expected_E_m * sp.sin(self.omega * self.t)
        
        # For average EMF from t=0 to T/4 (T = 2π/ω)
        T = 2 * math.pi / self.omega
        self.expected_overlineE = (2 * self.n * self.B * self.expected_S * self.omega) / math.pi
        
    def test_invariants(self):
        logs = []
        success = True
        
        # 1. Check Faraday's law symbolic form: e = -n * d(Φ)/dt
        try:
            phi = self.n * self.B * self.expected_S * sp.cos(self.omega * self.t)
            dphi_dt = sp.diff(phi, self.t)
            expected_emf_law = -dphi_dt
            # Should simplify to n*B*S*ω*sin(ωt)
            expected_simplified = self.expected_E_m * sp.sin(self.omega * self.t)
            
            if sp.simplify(expected_emf_law - expected_simplified) != 0:
                logs.append(f"FAIL: Faraday's law doesn't match expected form.")
                logs.append(f"  From -dΦ/dt: {expected_emf_law}")
                logs.append(f"  Expected: {expected_simplified}")
                success = False
            else:
                logs.append("PASS: Faraday's law symbolic check")
        except Exception as e:
            logs.append(f"ERROR in Faraday symbolic check: {e}")
            success = False
        
        # 2. Check maximum EMF consistency: E_m = n*B*S*ω
        if not math.isclose(self.expected_E_m, self.E_m, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Maximum EMF mismatch")
            logs.append(f"  Computed: {self.E_m}")
            logs.append(f"  Expected from formula: {self.expected_E_m}")
            success = False
        else:
            logs.append("PASS: Maximum EMF consistent with formula")
        
        # 3. Check average EMF formula: (2/π) * E_m
        expected_from_E_m = (2 / math.pi) * self.E_m
        if not math.isclose(self.expected_overlineE, self.overlineE, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Average EMF mismatch")
            logs.append(f"  Computed: {self.overlineE}")
            logs.append(f"  Expected from formula: {self.expected_overlineE}")
            success = False
        elif not math.isclose(expected_from_E_m, self.overlineE, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Average EMF not consistent with (2/π)*E_m")
            logs.append(f"  (2/π)*E_m = {expected_from_E_m}")
            logs.append(f"  Computed average: {self.overlineE}")
            success = False
        else:
            logs.append("PASS: Average EMF consistent with both direct formula and (2/π)*E_m")
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check area S
        if not math.isclose(self.S, self.expected_S, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Area S mismatch")
            logs.append(f"  Computed: {self.S}")
            logs.append(f"  Expected: {self.expected_S}")
            success = False
        else:
            logs.append("PASS: Area S consistent")
        
        # Check if e expression matches expected form
        if self.e_expr is not None:
            try:
                # Convert to sympy if it's a string
                if isinstance(self.e_expr, str):
                    # Try to parse - might contain "314sin(100πt)" or similar
                    # We'll compare numerical coefficients
                    # Extract numerical coefficient and frequency
                    coeff_match = None
                    if 'sin' in self.e_expr:
                        # Simple parsing for format like "314sin(100πt)"
                        import re
                        match = re.search(r'([\d.]+)sin', self.e_expr)
                        if match:
                            coeff = float(match.group(1))
                            coeff_match = coeff
                
                # Compare with expected numerically at sample points
                test_times = [0.001, 0.01, 0.1]
                for t_val in test_times:
                    expected_val = float(self.expected_E_m * math.sin(self.omega * t_val))
                    
                    # Try to evaluate self.e_expr
                    actual_val = None
                    if hasattr(self.e_expr, 'subs'):  # sympy expression
                        actual_val = float(self.e_expr.subs(self.t, t_val))
                    elif isinstance(self.e_expr, str):
                        # Try to evaluate as Python expression
                        try:
                            # Replace π with math.pi and t with t_val
                            expr = self.e_expr.replace('π', 'math.pi').replace('t', str(t_val))
                            actual_val = eval(expr, {"math": math, "__builtins__": {}})
                        except:
                            # If parsing fails, use coefficient match
                            if coeff_match is not None:
                                actual_val = coeff_match * math.sin(self.omega * t_val)
                    
                    if actual_val is not None and not math.isclose(expected_val, actual_val, 
                                                                   rel_tol=0.05, abs_tol=0.05):
                        logs.append(f"FAIL: Instantaneous EMF mismatch at t={t_val}")
                        logs.append(f"  Expected: {expected_val}")
                        logs.append(f"  Actual: {actual_val}")
                        success = False
                        break
                
                if success:
                    logs.append("PASS: Instantaneous EMF consistent at sample points")
            except Exception as e:
                logs.append(f"ERROR checking e expression: {e}")
                logs.append(traceback.format_exc())
                success = False
        else:
            logs.append("WARNING: No e expression found in results")
        
        # Check that E_m appears as coefficient in e expression
        if self.e_expr is not None and coeff_match is not None:
            if not math.isclose(coeff_match, self.E_m, rel_tol=0.05, abs_tol=0.05):
                logs.append(f"FAIL: Coefficient in e expression doesn't match E_m")
                logs.append(f"  Coefficient: {coeff_match}")
                logs.append(f"  E_m: {self.E_m}")
                success = False
            else:
                logs.append("PASS: Coefficient in e expression matches E_m")
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        # Perturb B and check if E_m changes in correct direction
        original_B = self.B
        
        try:
            # Increase B by 10%
            perturbed_B = original_B * 1.1
            perturbed_E_m = self.n * perturbed_B * self.expected_S * self.omega
            
            if not (perturbed_E_m > self.E_m):
                logs.append("FAIL: E_m doesn't increase when B increases")
                logs.append(f"  Original E_m: {self.E_m}")
                logs.append(f"  Perturbed E_m: {perturbed_E_m}")
                success = False
            else:
                logs.append("PASS: E_m increases with B (monotonicity check 1)")
            
            # Decrease B by 10%
            perturbed_B = original_B * 0.9
            perturbed_E_m = self.n * perturbed_B * self.expected_S * self.omega
            
            if not (perturbed_E_m < self.E_m):
                logs.append("FAIL: E_m doesn't decrease when B decreases")
                logs.append(f"  Original E_m: {self.E_m}")
                logs.append(f"  Perturbed E_m: {perturbed_E_m}")
                success = False
            else:
                logs.append("PASS: E_m decreases with B (monotonicity check 2)")
            
            # Check area monotonicity
            original_S = self.expected_S
            perturbed_S = original_S * 1.1
            perturbed_E_m = self.n * self.B * perturbed_S * self.omega
            
            if not (perturbed_E_m > self.E_m):
                logs.append("FAIL: E_m doesn't increase when area increases")
                success = False
            else:
                logs.append("PASS: E_m increases with area")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                for log in logs:
                    all_logs.append(f"  {log}")
                if not success:
                    overall_success = False
                    all_logs.append(f"  Result: FAILED")
                else:
                    all_logs.append(f"  Result: PASSED")
            except Exception as e:
                all_logs.append(f"  ERROR during {test_name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All tests passed!")
        else:
            all_logs.append("Some tests failed.")
        
        return overall_success, all_logs