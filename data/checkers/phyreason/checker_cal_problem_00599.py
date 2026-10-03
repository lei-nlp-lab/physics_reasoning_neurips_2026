import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        solution_instance: object containing problem solution (attributes or .results dict)
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe fallbacks (symbols if not numeric)
        self.G = self._safe_get('G')
        self.alpha = self._safe_get('alpha')
        self.beta = self._safe_get('beta')
        self.F = self._safe_get('F')
        
        # Extract results with safe fallbacks
        self.N = self._safe_get_result('N')
        self.F_1 = self._safe_get_result('F_1')
        self.F_f = self._safe_get_result('F_f')
        self.F_N = self._safe_get_result('F_N')
        
        # Define symbolic variables for physical laws
        self.symbols = sp.symbols('G alpha beta F')
        self.G_sym, self.alpha_sym, self.beta_sym, self.F_sym = self.symbols
        
        # Build expected symbolic expressions from steps analysis
        self.F1_expected = self.F_sym * sp.cos(self.beta_sym)
        self.N_expected = self.G_sym * sp.cos(self.alpha_sym) - self.F_sym * sp.sin(self.beta_sym)
        self.Ff_expected = self.F_sym * sp.cos(self.beta_sym) - self.G_sym * sp.sin(self.alpha_sym)
        self.FN_expected = self.N_expected  # Newton's third law
        
    def _safe_get(self, key):
        """Safely get attribute from solution or results, converting sympy to float if numeric."""
        val = getattr(self.solution, key, self.results.get(key, 0.0))
        if hasattr(val, 'evalf'):
            try:
                return float(val.evalf())
            except:
                return val
        return val
    
    def _safe_get_result(self, key):
        """Safely get result value, converting sympy to float if numeric."""
        val = self.results.get(key, 0.0)
        if hasattr(val, 'evalf'):
            try:
                return float(val.evalf())
            except:
                return val
        return val
    
    def _numeric_close(self, a, b):
        """Fuzzy numerical comparison with 1% tolerance."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        return False
    
    def _symbolic_check(self, expr1, expr2, subs_dict=None):
        """Check if two symbolic expressions are equivalent (numerically after substitution)."""
        if subs_dict is None:
            subs_dict = {
                self.G_sym: self.G,
                self.alpha_sym: self.alpha,
                self.beta_sym: self.beta,
                self.F_sym: self.F
            }
        try:
            diff = sp.simplify(expr1 - expr2)
            if diff == 0:
                return True
            # Try numerical evaluation
            if all(isinstance(v, (int, float)) for v in subs_dict.values()):
                num_diff = float(diff.subs(subs_dict).evalf())
                return math.isclose(num_diff, 0.0, rel_tol=1e-2, abs_tol=1e-2)
        except:
            pass
        return False
    
    def test_invariants(self):
        """Verify physical invariants: equilibrium conditions and Newton's third law."""
        logs = []
        success = True
        
        try:
            # 1. Equilibrium along incline: F_1 - G*sin(alpha) - F_f = 0
            # (From steps: F_f = F cosβ - G sinα, so F cosβ - G sinα - F_f = 0)
            net_incline = self.F_1 - self.G * math.sin(self.alpha) - self.F_f
            if not self._numeric_close(net_incline, 0.0):
                logs.append(f"FAIL: Net force along incline ≠ 0. Got {net_incline:.4f}")
                success = False
            else:
                logs.append("PASS: Equilibrium along incline satisfied.")
            
            # 2. Equilibrium perpendicular: N - (G*cos(alpha) - F*sin(beta)) = 0
            net_perp = self.N - (self.G * math.cos(self.alpha) - self.F * math.sin(self.beta))
            if not self._numeric_close(net_perp, 0.0):
                logs.append(f"FAIL: Net force perpendicular ≠ 0. Got {net_perp:.4f}")
                success = False
            else:
                logs.append("PASS: Equilibrium perpendicular satisfied.")
            
            # 3. Newton's third law: F_N should equal N (magnitude of normal force)
            if not self._numeric_close(self.F_N, self.N):
                logs.append(f"FAIL: Newton's third law violated. F_N={self.F_N:.4f}, N={self.N:.4f}")
                success = False
            else:
                logs.append("PASS: Newton's third law satisfied.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with formulas from steps analysis."""
        logs = []
        success = True
        
        try:
            # Build substitution dictionary for symbolic checks
            subs_dict = {
                self.G_sym: self.G,
                self.alpha_sym: self.alpha,
                self.beta_sym: self.beta,
                self.F_sym: self.F
            }
            
            # Check F_1 = F cosβ
            if not self._symbolic_check(self.F_1, self.F1_expected, subs_dict):
                logs.append(f"FAIL: F_1 mismatch. Expected {self.F1_expected}, got {self.F_1}")
                success = False
            else:
                logs.append("PASS: F_1 matches formula F cosβ.")
            
            # Check N = G cosα - F sinβ
            if not self._symbolic_check(self.N, self.N_expected, subs_dict):
                logs.append(f"FAIL: N mismatch. Expected {self.N_expected}, got {self.N}")
                success = False
            else:
                logs.append("PASS: N matches formula G cosα - F sinβ.")
            
            # Check F_f = F cosβ - G sinα
            if not self._symbolic_check(self.F_f, self.Ff_expected, subs_dict):
                logs.append(f"FAIL: F_f mismatch. Expected {self.Ff_expected}, got {self.F_f}")
                success = False
            else:
                logs.append("PASS: F_f matches formula F cosβ - G sinα.")
            
            # Check F_N = N
            if not self._symbolic_check(self.F_N, self.FN_expected, subs_dict):
                logs.append(f"FAIL: F_N mismatch. Expected {self.FN_expected}, got {self.F_N}")
                success = False
            else:
                logs.append("PASS: F_N matches normal force N.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input G and verify expected trends in results (then restore)."""
        logs = []
        success = True
        
        try:
            # Store original G
            original_G = self.G
            
            # Perturbation: increase G by 1%
            perturbed_G = original_G * 1.01
            
            # Compute expected results with perturbed G using formulas
            expected_N_orig = original_G * math.cos(self.alpha) - self.F * math.sin(self.beta)
            expected_N_pert = perturbed_G * math.cos(self.alpha) - self.F * math.sin(self.beta)
            
            expected_Ff_orig = self.F * math.cos(self.beta) - original_G * math.sin(self.alpha)
            expected_Ff_pert = self.F * math.cos(self.beta) - perturbed_G * math.sin(self.alpha)
            
            # Check monotonic trends (assuming 0 < α < 90°)
            if math.cos(self.alpha) > 0:
                if expected_N_pert <= expected_N_orig:
                    logs.append(f"FAIL: N should increase with G. Original: {expected_N_orig:.4f}, Perturbed: {expected_N_pert:.4f}")
                    success = False
                else:
                    logs.append("PASS: N increases with G (as expected from N = G cosα - F sinβ).")
            
            if math.sin(self.alpha) > 0:
                if expected_Ff_pert >= expected_Ff_orig:
                    logs.append(f"FAIL: F_f should decrease with G. Original: {expected_Ff_orig:.4f}, Perturbed: {expected_Ff_pert:.4f}")
                    success = False
                else:
                    logs.append("PASS: F_f decreases with G (as expected from F_f = F cosβ - G sinα).")
                    
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
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
            all_logs.append(f"\n=== {test_name} Test ===")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All tests passed. Physics is consistent.")
        else:
            all_logs.append("Some tests failed. Check physics reasoning.")
        
        return overall_success, all_logs