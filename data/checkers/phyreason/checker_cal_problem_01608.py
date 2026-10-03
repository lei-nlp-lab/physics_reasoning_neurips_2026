import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get() fallbacks
        self.solution = solution_instance
        self.inputs = {
            'l': getattr(solution_instance, 'l', 1.0),
            'm': getattr(solution_instance, 'm', 1.0),
            'E': getattr(solution_instance, 'E', 1.0),
            'alpha': getattr(solution_instance, 'alpha', 0.5),
            'g': getattr(solution_instance, 'g', 9.8)
        }
        # Extract results safely
        results_dict = getattr(solution_instance, 'results', {})
        self.results = {
            'q': results_dict.get('q', 0.0),
            'charge_type': results_dict.get('charge_type', 1.0),  # Default to 1.0 (positive) as float
            'W': results_dict.get('W', 0.0)
        }
        self.logs = []
        
    def log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            prefix = "PASS" if success else "FAIL"
            self.logs.append(f"{prefix}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        success = True
        
        # Extract values with safe defaults
        m, E, alpha, g = self.inputs['m'], self.inputs['E'], self.inputs['alpha'], self.inputs['g']
        q, charge_type, W = self.results['q'], self.results['charge_type'], self.results['W']
        
        # 1. Charge type must be positive (from KB)
        # Accept either string "positive" or float 1.0
        if isinstance(charge_type, str):
            if 'positive' not in charge_type.lower():
                self.log(f"Charge type should be positive, got string: {charge_type}", False)
                success = False
            else:
                self.log("Charge type positive check passed (string)", True)
        elif isinstance(charge_type, (int, float)):
            # Positive charge should be > 0, negative < 0
            if charge_type > 0:
                self.log(f"Charge type positive check passed (float: {charge_type})", True)
            else:
                self.log(f"Charge type should be positive, got float: {charge_type}", False)
                success = False
        else:
            self.log(f"Charge type has unexpected type: {type(charge_type)}", False)
            success = False
        
        # 2. Force equilibrium check (Newton's 2nd Law)
        # Symbolic equilibrium condition: T sin(α) = qE, T cos(α) = mg
        # Combining: tan(α) = qE / (mg) => q = (mg tan(α))/E
        try:
            # Numeric check
            mg = m * g
            tan_alpha = math.tan(alpha)
            expected_q = (mg * tan_alpha) / E if E != 0 else 0.0
            
            if not math.isclose(q, expected_q, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Force equilibrium check failed: q={q}, expected={expected_q}", False)
                success = False
            else:
                self.log("Force equilibrium check passed", True)
        except Exception as e:
            self.log(f"Force equilibrium check error: {e}", False)
            success = False
        
        # 3. Work-energy consistency
        # From steps: W = E*q*l = m*g*l*tan(α)
        try:
            l = self.inputs['l']
            expected_W = m * g * l * tan_alpha
            
            if not math.isclose(W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Work-energy consistency failed: W={W}, expected={expected_W}", False)
                success = False
            else:
                self.log("Work-energy consistency passed", True)
        except Exception as e:
            self.log(f"Work-energy check error: {e}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        m, E, alpha, g, l = (self.inputs[k] for k in ['m', 'E', 'alpha', 'g', 'l'])
        q, W = self.results['q'], self.results['W']
        
        # 1. Check q formula consistency
        try:
            tan_alpha = math.tan(alpha)
            q_from_formula = (m * g * tan_alpha) / E if E != 0 else 0.0
            
            if not math.isclose(q, q_from_formula, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"q trace inconsistency: q={q}, formula gives={q_from_formula}", False)
                success = False
            else:
                self.log("q trace consistency passed", True)
        except Exception as e:
            self.log(f"q trace check error: {e}", False)
            success = False
        
        # 2. Check W formula consistency
        try:
            tan_alpha = math.tan(alpha)
            W_from_formula = m * g * l * tan_alpha
            W_from_q_formula = E * q * l
            
            # Both formulas should give same result
            if (not math.isclose(W, W_from_formula, rel_tol=1e-2, abs_tol=1e-2) or
                not math.isclose(W, W_from_q_formula, rel_tol=1e-2, abs_tol=1e-2)):
                self.log(f"W trace inconsistency: W={W}, formula1={W_from_formula}, formula2={W_from_q_formula}", False)
                success = False
            else:
                self.log("W trace consistency passed", True)
        except Exception as e:
            self.log(f"W trace check error: {e}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Test how q and W change with perturbations in inputs."""
        success = True
        
        # Original values
        m, E, alpha, g, l = (self.inputs[k] for k in ['m', 'E', 'alpha', 'g', 'l'])
        q_orig, W_orig = self.results['q'], self.results['W']
        
        # Test 1: Increase mass m by 10%
        m_new = m * 1.1
        tan_alpha = math.tan(alpha)
        
        q_new = (m_new * g * tan_alpha) / E if E != 0 else 0.0
        W_new = m_new * g * l * tan_alpha
        
        # Both should increase with mass
        if not (q_new > q_orig and W_new > W_orig):
            self.log(f"Monotonicity w.r.t mass failed: q {q_orig}->{q_new}, W {W_orig}->{W_new}", False)
            success = False
        else:
            self.log("Monotonicity w.r.t mass passed", True)
        
        # Test 2: Increase angle α by 10% (but keep < π/2)
        alpha_new = min(alpha * 1.1, 0.99 * math.pi / 2)
        tan_alpha_new = math.tan(alpha_new)
        
        q_new = (m * g * tan_alpha_new) / E if E != 0 else 0.0
        W_new = m * g * l * tan_alpha_new
        
        # Both should increase with angle (tan is increasing for 0 < α < π/2)
        if not (q_new > q_orig and W_new > W_orig):
            self.log(f"Monotonicity w.r.t angle failed: q {q_orig}->{q_new}, W {W_orig}->{W_new}", False)
            success = False
        else:
            self.log("Monotonicity w.r.t angle passed", True)
        
        # Test 3: Increase electric field E by 10%
        E_new = E * 1.1
        q_new = (m * g * tan_alpha) / E_new if E_new != 0 else 0.0
        W_new = m * g * l * tan_alpha  # W independent of E per formula
        
        # q should decrease, W should remain same (per formula)
        if not (q_new < q_orig and math.isclose(W_new, W_orig, rel_tol=1e-2, abs_tol=1e-2)):
            self.log(f"Monotonicity w.r.t E failed: q {q_orig}->{q_new}, W {W_orig}->{W_new}", False)
            success = False
        else:
            self.log("Monotonicity w.r.t E passed", True)
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self.log(f"Running {test_name} test...")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"{test_name} test crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        self.log(f"Overall test result: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs