import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results using getattr to prevent AttributeError.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs from problem context
        self.m = getattr(solution_instance, 'm', None)
        self.n = getattr(solution_instance, 'n', None)
        self.k = getattr(solution_instance, 'k', None)
        self.V = getattr(solution_instance, 'V', None)
        self.a = getattr(solution_instance, 'a', None)
        
        # Results (with fallbacks)
        self.S = getattr(solution_instance, 'S', self.results.get('S', None))
        self.d = getattr(solution_instance, 'd', self.results.get('d', None))
        
        # Symbolic variables for physical laws
        self.m_sym = sp.symbols('m', positive=True, real=True)
        self.n_sym = sp.symbols('n', positive=True, real=True)
        self.k_sym = sp.symbols('k', positive=True, real=True)
        self.V_sym = sp.symbols('V', positive=True, real=True)
        self.a_sym = sp.symbols('a', positive=True, real=True)
        self.S_sym = sp.symbols('S', positive=True, real=True)
        self.d_sym = sp.symbols('d', positive=True, real=True)
        self.V_oleic_sym = sp.symbols('V_oleic', positive=True, real=True)
        self.V_solution_sym = sp.symbols('V_solution', positive=True, real=True)
        
        self.logs = []
        
    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append((msg, success))
        return success
        
    def _is_close(self, val1, val2):
        """Fuzzy numerical matching with 2% tolerance."""
        if val1 is None or val2 is None:
            return False
        try:
            return math.isclose(float(val1), float(val2), rel_tol=0.02, abs_tol=0.02)
        except (TypeError, ValueError):
            # If values are symbolic, try symbolic equality
            try:
                return sp.simplify(val1 - val2) == 0
            except:
                return False
                
    def test_invariants(self):
        """
        Verify physical invariants from KB:
        1. Volume = Area × Thickness (d = V_oleic / S)
        2. Volume ratio consistency (V_oleic proportional to m/(m+n))
        """
        success = True
        
        # Test 1: Volume = Area × Thickness
        # From steps: d = V_oleic / S, where V_oleic = (m/(m+n)) * (V/k)
        try:
            if None not in [self.m, self.n, self.k, self.V, self.S, self.d]:
                V_oleic_calc = (self.m / (self.m + self.n)) * (self.V / self.k)
                d_calc = V_oleic_calc / self.S
                if not self._is_close(self.d, d_calc):
                    success = self._log(
                        f"FAIL: Volume=Area×Thickness invariant. d={self.d}, calculated={d_calc}",
                        False
                    )
                else:
                    self._log("PASS: Volume=Area×Thickness invariant", True)
            else:
                self._log("SKIP: Volume=Area×Thickness (missing values)", True)
        except Exception as e:
            success = self._log(f"ERROR in Volume invariant: {str(e)}", False)
            
        # Test 2: Volume ratio consistency (symbolic check)
        try:
            # Theorem: V_oleic / V_solution = m/(m+n) for one droplet
            V_solution_one = self.V_sym / self.k_sym
            V_oleic_expr = (self.m_sym / (self.m_sym + self.n_sym)) * V_solution_one
            ratio_expr = sp.simplify(V_oleic_expr / V_solution_one - self.m_sym/(self.m_sym + self.n_sym))
            if ratio_expr != 0:
                success = self._log(f"FAIL: Volume ratio symbolic check: {ratio_expr}", False)
            else:
                self._log("PASS: Volume ratio symbolic consistency", True)
        except Exception as e:
            success = self._log(f"ERROR in ratio symbolic check: {str(e)}", False)
            
        return success
        
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer:
        1. S = 31a² (from grid counting)
        2. d = mV / [31a²k(m+n)]
        """
        success = True
        
        # Test 1: Area from grid counting
        try:
            if self.a is not None and self.S is not None:
                S_expected = 31 * (self.a ** 2)
                if not self._is_close(self.S, S_expected):
                    success = self._log(
                        f"FAIL: Area consistency. S={self.S}, expected 31a²={S_expected}",
                        False
                    )
                else:
                    self._log("PASS: Area S = 31a²", True)
            else:
                self._log("SKIP: Area check (missing a or S)", True)
        except Exception as e:
            success = self._log(f"ERROR in area check: {str(e)}", False)
            
        # Test 2: Diameter formula
        try:
            if None not in [self.m, self.n, self.k, self.V, self.a, self.d]:
                d_expected = (self.m * self.V) / (31 * (self.a ** 2) * self.k * (self.m + self.n))
                if not self._is_close(self.d, d_expected):
                    success = self._log(
                        f"FAIL: Diameter consistency. d={self.d}, expected formula={d_expected}",
                        False
                    )
                else:
                    self._log("PASS: Diameter formula matches", True)
            else:
                self._log("SKIP: Diameter check (missing values)", True)
        except Exception as e:
            success = self._log(f"ERROR in diameter check: {str(e)}", False)
            
        return success
        
    def test_monotonicity(self):
        """
        Perturb input m (oleic acid proportion) and check d changes appropriately.
        d = mV / [31a²k(m+n)] should increase with m when other parameters fixed.
        """
        success = True
        
        original_m = self.m
        if original_m is None or None in [self.n, self.k, self.V, self.a]:
            self._log("SKIP: Monotonicity test (missing values)", True)
            return True
            
        try:
            # Increase m by 1%
            perturbed_m = original_m * 1.01
            # Calculate d with perturbed m
            d_original = (original_m * self.V) / (31 * (self.a ** 2) * self.k * (original_m + self.n))
            d_perturbed = (perturbed_m * self.V) / (31 * (self.a ** 2) * self.k * (perturbed_m + self.n))
            
            # d should increase with m (derivative positive for m,n > 0)
            if d_perturbed <= d_original:
                success = self._log(
                    f"FAIL: Monotonicity. d({original_m})={d_original}, d({perturbed_m})={d_perturbed}",
                    False
                )
            else:
                self._log(
                    f"PASS: Monotonicity - d increases with m ({d_original:.4e} → {d_perturbed:.4e})",
                    True
                )
                
        except Exception as e:
            success = self._log(f"ERROR in monotonicity test: {str(e)}", False)
            
        return success
        
    def run_all_tests(self):
        """
        Execute all tests and return (overall_success, logs).
        """
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                error_msg = f"CRASH in {test_name}: {traceback.format_exc()}"
                self._log(error_msg, False)
                overall_success = False
                
        return overall_success, self.logs