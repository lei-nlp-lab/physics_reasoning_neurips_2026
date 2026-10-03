import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        self.solution = solution_instance
        
        # Extract inputs safely
        self.h = self._safe_extract('h', 5.0)  # default 5.0 for fallback
        self.v0 = self._safe_extract('v0', 10.0)
        self.g = self._safe_extract('g', 9.8)
        
        # Extract results safely
        self.t = self._safe_extract_result('t', 0.0)
        self.v = self._safe_extract_result('v', 0.0)
        self.v_y = self._safe_extract_result('v_y', 0.0)
        
        # Expected formulas from KB
        self.expected_t = sp.sqrt(2 * self.h / self.g)
        self.expected_v_y = sp.sqrt(2 * self.g * self.h)
        self.expected_v = sp.sqrt(self.v0**2 + 2 * self.g * self.h)
        
    def _safe_extract(self, attr, default):
        """Safely extract attribute from solution instance with multiple fallbacks."""
        try:
            # Try direct attribute
            value = getattr(self.solution, attr, None)
            if value is not None:
                return float(value)
            
            # Try inputs dictionary
            inputs = getattr(self.solution, 'inputs', {})
            if isinstance(inputs, dict) and attr in inputs:
                return float(inputs[attr])
            
            # Try results dictionary
            results = getattr(self.solution, 'results', {})
            if isinstance(results, dict) and attr in results:
                return float(results[attr])
        except (TypeError, ValueError, AttributeError):
            pass
        return default
    
    def _safe_extract_result(self, key, default):
        """Safely extract result from results dictionary."""
        results = getattr(self.solution, 'results', {})
        if isinstance(results, dict) and key in results:
            try:
                return float(results[key])
            except (TypeError, ValueError):
                pass
        return default
    
    def test_invariants(self):
        """Verify kinematic laws and conservation principles."""
        logs = []
        success = True
        
        try:
            # 1. Check flight time formula (free fall from height h)
            t_expected = math.sqrt(2 * self.h / self.g)
            if not math.isclose(self.t, t_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Flight time mismatch. Got {self.t:.3f}, expected {t_expected:.3f}")
                success = False
            else:
                logs.append(f"PASS: Flight time t = √(2h/g) verified within 2%")
            
            # 2. Check vertical velocity (v_y = g*t = √(2gh))
            v_y_from_t = self.g * self.t
            v_y_from_h = math.sqrt(2 * self.g * self.h)
            
            if not (math.isclose(self.v_y, v_y_from_t, rel_tol=0.02, abs_tol=0.02) and 
                    math.isclose(self.v_y, v_y_from_h, rel_tol=0.02, abs_tol=0.02)):
                logs.append(f"FAIL: Vertical velocity mismatch. Got {self.v_y:.3f}, expected g*t={v_y_from_t:.3f} or √(2gh)={v_y_from_h:.3f}")
                success = False
            else:
                logs.append(f"PASS: Vertical velocity v_y = gt = √(2gh) verified within 2%")
            
            # 3. Check final speed (vector addition)
            v_expected = math.sqrt(self.v0**2 + self.v_y**2)
            if not math.isclose(self.v, v_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Final speed mismatch. Got {self.v:.3f}, expected √(v0²+v_y²)={v_expected:.3f}")
                success = False
            else:
                logs.append(f"PASS: Final speed v = √(v0²+v_y²) verified within 2%")
            
            # 4. Symbolic verification using SymPy
            t_sym = sp.symbols('t')
            v_y_sym = sp.symbols('v_y')
            v_sym = sp.symbols('v')
            h_sym, v0_sym, g_sym = sp.symbols('h v0 g')
            
            eq1 = sp.Eq(t_sym, sp.sqrt(2 * h_sym / g_sym))
            eq2 = sp.Eq(v_y_sym, sp.sqrt(2 * g_sym * h_sym))
            eq3 = sp.Eq(v_sym, sp.sqrt(v0_sym**2 + v_y_sym**2))
            
            # Substitute numerical values
            eq1_sub = eq1.subs({h_sym: self.h, g_sym: self.g})
            eq2_sub = eq2.subs({h_sym: self.h, g_sym: self.g})
            eq3_sub = eq3.subs({v0_sym: self.v0, v_y_sym: self.v_y})
            
            # Evaluate left and right sides numerically
            lhs1 = float(self.t)
            rhs1 = float(sp.N(eq1_sub.rhs))
            lhs2 = float(self.v_y)
            rhs2 = float(sp.N(eq2_sub.rhs))
            lhs3 = float(self.v)
            rhs3 = float(sp.N(eq3_sub.rhs))
            
            symbolic_success = (math.isclose(lhs1, rhs1, rel_tol=0.02, abs_tol=0.02) and
                               math.isclose(lhs2, rhs2, rel_tol=0.02, abs_tol=0.02) and
                               math.isclose(lhs3, rhs3, rel_tol=0.02, abs_tol=0.02))
            
            if symbolic_success:
                logs.append("PASS: All symbolic equations verified within 2%")
            else:
                logs.append("FAIL: Symbolic equation mismatch")
                success = False
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Verify intermediate results are consistent with final answer."""
        logs = []
        success = True
        
        try:
            # Check that t and v_y yield the correct v when combined
            v_calculated = math.sqrt(self.v0**2 + (self.g * self.t)**2)
            
            if not math.isclose(self.v, v_calculated, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Trace inconsistency. Using t={self.t:.3f}, v0={self.v0:.3f}, calculated v={v_calculated:.3f}, but result has v={self.v:.3f}")
                success = False
            else:
                logs.append(f"PASS: Trace consistent: v = √(v0² + (g*t)²) verified")
            
            # Check that v_y and t satisfy v_y = g*t
            if not math.isclose(self.v_y, self.g * self.t, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_y and t inconsistent: v_y={self.v_y:.3f} ≠ g*t={self.g*self.t:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_y = g*t relationship holds")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        logs = []
        success = True
        
        try:
            original_h = self.h
            
            # Test 1: Increase h by 10%
            self.h *= 1.1
            t_increased = math.sqrt(2 * self.h / self.g)
            v_increased = math.sqrt(self.v0**2 + 2 * self.g * self.h)
            
            # Check monotonicity: both t and v should increase with h
            if t_increased <= self.t:
                logs.append(f"FAIL: t should increase with h, but {t_increased:.3f} ≤ {self.t:.3f}")
                success = False
            
            if v_increased <= self.v:
                logs.append(f"FAIL: v should increase with h, but {v_increased:.3f} ≤ {self.v:.3f}")
                success = False
            
            if success:
                logs.append(f"PASS: Monotonicity holds: increasing h increases both t and v")
            
            # Restore original h
            self.h = original_h
            
            # Test 2: Decrease v0 by 10%
            original_v0 = self.v0
            self.v0 *= 0.9
            v_decreased = math.sqrt(self.v0**2 + 2 * self.g * self.h)
            
            # v should decrease when v0 decreases
            if v_decreased >= self.v:
                logs.append(f"FAIL: v should decrease with v0, but {v_decreased:.3f} ≥ {self.v:.3f}")
                success = False
            
            if success:
                logs.append(f"PASS: Monotonicity holds: decreasing v0 decreases v")
            
            # Restore original v0
            self.v0 = original_v0
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return aggregate results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*40}")
            all_logs.append(f"Running {test_name}")
            all_logs.append('='*40)
            
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRITICAL ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n{'='*40}")
        all_logs.append("SUMMARY")
        all_logs.append('='*40)
        all_logs.append(f"Overall Status: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append(f"Inputs: h={self.h:.3f}, v0={self.v0:.3f}, g={self.g:.3f}")
        all_logs.append(f"Results: t={self.t:.3f}, v={self.v:.3f}, v_y={self.v_y:.3f}")
        
        return overall_success, all_logs