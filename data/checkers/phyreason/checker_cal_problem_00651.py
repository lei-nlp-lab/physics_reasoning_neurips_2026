import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', 37))
        
        # Safe extraction of results
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', 0.0))
        self.F_all = getattr(solution_instance, 'F_all', solution_instance.results.get('F_all', 0.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        
        # Time parameter (if exists)
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 1.0))
        
        # Trigonometric values for 37°
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = 0.6/0.8
        
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def _is_close(self, a, b):
        """Robust fuzzy numerical comparison."""
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        tests = []
        
        # 1. Static equilibrium condition (before string is cut)
        # x-direction: qE = T*sinθ
        # y-direction: mg = T*cosθ
        
        # Symbolic check
        T_symbolic = self.m * self.g / self.cos37
        qE_expected = T_symbolic * self.sin37
        mg_expected = T_symbolic * self.cos37
        
        # Numerical check
        qE_actual = self.q * self.E
        mg_actual = self.m * self.g
        
        eq1 = self._is_close(qE_actual, qE_expected)
        eq2 = self._is_close(mg_actual, mg_expected)
        
        self._log(f"Static equilibrium x-dir: qE={qE_actual:.3f} vs T*sinθ={qE_expected:.3f}", eq1)
        self._log(f"Static equilibrium y-dir: mg={mg_actual:.3f} vs T*cosθ={mg_expected:.3f}", eq2)
        tests.append(eq1 and eq2)
        
        # 2. Net force after string cut equals vector sum of qE and mg
        # F_all = sqrt((qE)² + (mg)²)
        F_expected = math.sqrt((self.q * self.E)**2 + (self.m * self.g)**2)
        net_eq = self._is_close(self.F_all, F_expected)
        self._log(f"Net force magnitude: {self.F_all:.3f} vs sqrt((qE)²+(mg)²)={F_expected:.3f}", net_eq)
        tests.append(net_eq)
        
        # 3. Newton's Second Law: F_all = m*a
        newton_eq = self._is_close(self.F_all, self.m * self.a)
        self._log(f"Newton's 2nd Law: F_all={self.F_all:.3f} vs m*a={self.m*self.a:.3f}", newton_eq)
        tests.append(newton_eq)
        
        return all(tests)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests = []
        
        # 1. Check electric field E from step 1
        E_expected = (self.m * self.g * self.tan37) / self.q
        E_check = self._is_close(self.E, E_expected)
        self._log(f"Electric field: {self.E:.3f} vs (mg tanθ)/q={E_expected:.3f}", E_check)
        tests.append(E_check)
        
        # 2. Check tension F from step 2
        F_expected = self.m * self.g / self.cos37
        F_check = self._is_close(self.F, F_expected)
        self._log(f"Tension: {self.F:.3f} vs mg/cosθ={F_expected:.3f}", F_check)
        tests.append(F_check)
        
        # 3. Check net force F_all from step 3
        F_all_expected = self.m * self.g / self.cos37
        F_all_check = self._is_close(self.F_all, F_all_expected)
        self._log(f"Net force: {self.F_all:.3f} vs mg/cosθ={F_all_expected:.3f}", F_all_check)
        tests.append(F_all_check)
        
        # 4. Check acceleration a from step 4
        a_expected = self.g / self.cos37
        a_check = self._is_close(self.a, a_expected)
        self._log(f"Acceleration: {self.a:.3f} vs g/cosθ={a_expected:.3f}", a_check)
        tests.append(a_check)
        
        # 5. Check velocity v from step 5
        v_expected = self.a * self.t
        v_check = self._is_close(self.v, v_expected)
        self._log(f"Velocity: v={self.v:.3f} vs a*t={v_expected:.3f}", v_check)
        tests.append(v_check)
        
        return all(tests)
    
    def test_monotonicity(self):
        """Test how results change when mass is perturbed."""
        tests = []
        
        # Store original values
        m_orig = float(self.m)
        orig_results = {
            'E': float(self.E),
            'F': float(self.F),
            'F_all': float(self.F_all),
            'a': float(self.a),
            'v': float(self.v)
        }
        
        # Perturb mass by +10%
        self.m = m_orig * 1.1
        
        # Recalculate expected values using formulas
        E_new = (self.m * self.g * self.tan37) / self.q
        F_new = self.m * self.g / self.cos37
        F_all_new = self.m * self.g / self.cos37
        a_new = self.g / self.cos37  # Independent of mass!
        v_new = a_new * self.t
        
        # Check monotonic trends
        # E, F, F_all should increase with mass
        E_trend = E_new > orig_results['E']
        F_trend = F_new > orig_results['F']
        F_all_trend = F_all_new > orig_results['F_all']
        
        # a should remain constant (independent of mass)
        a_trend = self._is_close(a_new, orig_results['a'])
        
        # v should increase because a*t (but a constant, so depends on t)
        # Actually v increases with t, but for fixed t, v remains same since a constant
        v_trend = self._is_close(v_new, orig_results['v'])
        
        self._log(f"E trend with +10% mass: {E_new:.3f} > {orig_results['E']:.3f}", E_trend)
        self._log(f"F trend with +10% mass: {F_new:.3f} > {orig_results['F']:.3f}", F_trend)
        self._log(f"F_all trend with +10% mass: {F_all_new:.3f} > {orig_results['F_all']:.3f}", F_all_trend)
        self._log(f"a constant with mass: {a_new:.3f} ≈ {orig_results['a']:.3f}", a_trend)
        self._log(f"v constant with mass (fixed t): {v_new:.3f} ≈ {orig_results['v']:.3f}", v_trend)
        
        tests.extend([E_trend, F_trend, F_all_trend, a_trend, v_trend])
        
        # Restore original values
        self.m = m_orig
        self.E = orig_results['E']
        self.F = orig_results['F']
        self.F_all = orig_results['F_all']
        self.a = orig_results['a']
        self.v = orig_results['v']
        
        return all(tests)
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        self._log("Starting Physics Audit", None)
        
        try:
            # Run all test suites
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            # Overall success
            success = inv_ok and trace_ok and mono_ok
            
            # Summary
            self._log("\n=== SUMMARY ===", None)
            self._log(f"Invariants: {'PASS' if inv_ok else 'FAIL'}", inv_ok)
            self._log(f"Trace Consistency: {'PASS' if trace_ok else 'FAIL'}", trace_ok)
            self._log(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}", mono_ok)
            self._log(f"OVERALL: {'PASS' if success else 'FAIL'}", success)
            
            return success, self.logs
            
        except Exception as e:
            self._log(f"Error during testing: {str(e)}", False)
            self._log(traceback.format_exc(), False)
            return False, self.logs