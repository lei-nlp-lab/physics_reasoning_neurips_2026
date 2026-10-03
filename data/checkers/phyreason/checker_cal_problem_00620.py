import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe extraction with fallbacks
        self.m = float(getattr(solution_instance, 'm', self.results.get('m', 1500.0)))
        self.v0 = float(getattr(solution_instance, 'v_0', self.results.get('v_0', 10.0)))
        self.a1 = float(getattr(solution_instance, 'a_1', self.results.get('a_1', -1.0)))  # Negative for deceleration
        self.F_f = float(getattr(solution_instance, 'F_f', self.results.get('F_f', 12000.0)))
        self.x_total = float(getattr(solution_instance, 'x', self.results.get('x', 22.0)))
        self.v = float(self.results.get('v', None)) if self.results.get('v') is not None else None
        self.x2 = float(self.results.get('x_2', None)) if self.results.get('x_2') is not None else None
        self.a2 = float(self.results.get('a_2', None)) if self.results.get('a_2') is not None else None
        self.t_total = float(self.results.get('t', None)) if self.results.get('t') is not None else None
        self.t2 = float(self.results.get('t_2', None)) if self.results.get('t_2') is not None else None
        self.v_avg = float(self.results.get('overlinev', None)) if self.results.get('overlinev') is not None else None
        
        # Derived quantities from steps
        self.x1 = self.x_total - self.x2 if self.x2 is not None else None
        
        # Tolerances for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Log storage
        self.logs = []
    
    def _log(self, message, success=True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success
    
    def _isclose(self, a, b):
        """Safe numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """
        Verify physical invariants: Newton's 2nd Law and kinematic consistency.
        """
        successes = []
        
        # 1. Newton's 2nd Law for emergency braking: a2 = -F_f / m
        if self.a2 is not None:
            expected_a2 = -self.F_f / self.m  # Negative for deceleration
            if self._isclose(self.a2, expected_a2):
                successes.append(self._log(f"Newton's 2nd Law: a2 = {self.a2:.2f}, F_f/m = {expected_a2:.2f}"))
            else:
                successes.append(self._log(f"Newton's 2nd Law mismatch: a2={self.a2:.2f}, F_f/m={expected_a2:.2f}", False))
        
        # 2. Kinematic consistency: v^2 = v0^2 + 2*a1*x1
        if self.v is not None and self.x1 is not None:
            lhs = self.v**2
            rhs = self.v0**2 + 2 * self.a1 * self.x1
            if self._isclose(lhs, rhs):
                successes.append(self._log(f"Kinematic eq1: v²={lhs:.2f}, v0²+2a1x1={rhs:.2f}"))
            else:
                successes.append(self._log(f"Kinematic eq1 mismatch: v²={lhs:.2f}, v0²+2a1x1={rhs:.2f}", False))
        
        # 3. Kinematic consistency: 0 = v^2 + 2*a2*x2 (since final velocity is 0)
        if self.v is not None and self.x2 is not None and self.a2 is not None:
            lhs = 0.0
            rhs = self.v**2 + 2 * self.a2 * self.x2
            if self._isclose(lhs, rhs):
                successes.append(self._log(f"Kinematic eq2: 0={lhs:.2f}, v²+2a2x2={rhs:.2f}"))
            else:
                successes.append(self._log(f"Kinematic eq2 mismatch: 0={lhs:.2f}, v²+2a2x2={rhs:.2f}", False))
        
        # 4. Total distance constraint: x_total = x1 + x2
        if self.x1 is not None and self.x2 is not None:
            if self._isclose(self.x_total, self.x1 + self.x2):
                successes.append(self._log(f"Distance sum: x_total={self.x_total:.2f}, x1+x2={self.x1+self.x2:.2f}"))
            else:
                successes.append(self._log(f"Distance sum mismatch: x_total={self.x_total:.2f}, x1+x2={self.x1+self.x2:.2f}", False))
        
        # 5. Average velocity definition: v_avg = x_total / t_total
        if self.v_avg is not None and self.t_total is not None and self.t_total > 0:
            expected_v_avg = self.x_total / self.t_total
            if self._isclose(self.v_avg, expected_v_avg):
                successes.append(self._log(f"Avg velocity: v_avg={self.v_avg:.2f}, x/t={expected_v_avg:.2f}"))
            else:
                successes.append(self._log(f"Avg velocity mismatch: v_avg={self.v_avg:.2f}, x/t={expected_v_avg:.2f}", False))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers and steps.
        """
        successes = []
        
        # Expected values from steps
        expected_a2 = -8.0  # From step_4: 12000/1500 = 8, negative for deceleration
        expected_v = 8.0    # From step_10
        expected_x2 = 4.0   # From step_10
        expected_t1 = 2.0   # From step_11: (10-8)/1 = 2 (phase 1 time)
        expected_t2 = 1.0   # From step_12: 8/8 = 1
        expected_total_t = 3.0  # From step_13: 2+1 = 3
        expected_v_avg = 22.0/3.0  # From step_14
        
        # Calculate phase 1 time from available data
        t1_calculated = None
        if self.v is not None:
            t1_calculated = (self.v - self.v0) / self.a1  # Should be positive
        
        checks = [
            ("a2", self.a2, expected_a2),
            ("v", self.v, expected_v),
            ("x2", self.x2, expected_x2),
            ("t1 (phase1)", t1_calculated, expected_t1),
            ("t2", self.t2, expected_t2),
            ("total t", self.t_total, expected_total_t),
            ("v_avg", self.v_avg, expected_v_avg)
        ]
        
        for name, actual, expected in checks:
            if actual is not None:
                if self._isclose(actual, expected):
                    successes.append(self._log(f"{name}: {actual:.2f} ≈ {expected:.2f}"))
                else:
                    successes.append(self._log(f"{name} mismatch: {actual:.2f} vs {expected:.2f}", False))
            else:
                successes.append(self._log(f"{name} not available", False))
        
        return all(successes)
    
    def test_monotonicity(self):
        """
        Perturb mass and verify expected trends.
        Mass increase → |a2| decreases → braking distance x2 increases → v at braking increases.
        """
        try:
            # Original values
            m_original = self.m
            F_f = self.F_f
            v0 = self.v0
            a1 = self.a1
            x_total = self.x_total
            
            # Store original results
            v_original = self.v
            x2_original = self.x2
            
            if v_original is None or x2_original is None:
                return self._log("Missing v or x2 for monotonicity test", False)
            
            # Perturb mass (increase by 10%)
            m_perturbed = m_original * 1.10
            
            # Recalculate using derived equations from steps
            # From step_10 derivation: v^2 = 56 / (1 - m/F_f)
            # Where 56 comes from: x_total = 22, v0=10, a1=-1
            # Derivation: 
            #   x1 = (v0^2 - v^2) / (2*|a1|) = (100 - v^2)/2
            #   x2 = v^2 / (2*|a2|) = v^2 / (2*(F_f/m)) = (v^2 * m) / (2*F_f)
            #   x1 + x2 = 22
            #   => (100 - v^2)/2 + (v^2 * m)/(2*F_f) = 22
            #   => 100 - v^2 + (v^2 * m)/F_f = 44
            #   => 56 = v^2 * (1 - m/F_f)
            #   => v^2 = 56 / (1 - m/F_f)
            
            v2_perturbed = 56.0 / (1.0 - m_perturbed / F_f)
            if v2_perturbed < 0:
                return self._log(f"Negative v^2 for m={m_perturbed}: not physical", False)
            
            v_perturbed = math.sqrt(v2_perturbed)
            a2_perturbed = -F_f / m_perturbed  # Negative
            x2_perturbed = v2_perturbed / (2.0 * abs(a2_perturbed))
            
            # Expected trends: 
            # Since m increased, |a2| decreased → x2 should increase, v should increase
            trend_ok = (v_perturbed > v_original) and (x2_perturbed > x2_original)
            
            if trend_ok:
                return self._log(f"Mass ↑ {m_original:.0f}→{m_perturbed:.0f}kg: v {v_original:.2f}→{v_perturbed:.2f}m/s, x2 {x2_original:.2f}→{x2_perturbed:.2f}m")
            else:
                return self._log(f"Unexpected trend: mass ↑ but v or x2 didn't increase", False)
                
        except Exception as e:
            return self._log(f"Monotonicity test error: {str(e)}", False)
    
    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []
        overall_success = True
        
        try:
            # Test 1: Physical invariants
            self.logs.append("=== Testing Physical Invariants ===")
            if not self.test_invariants():
                overall_success = False
            
            # Test 2: Trace consistency
            self.logs.append("\n=== Testing Trace Consistency ===")
            if not self.test_trace_consistency():
                overall_success = False
            
            # Test 3: Monotonicity
            self.logs.append("\n=== Testing Monotonicity ===")
            if not self.test_monotonicity():
                overall_success = False
                
        except Exception as e:
            self.logs.append(f"\n[ERROR] Test execution failed: {str(e)}")
            self.logs.append(traceback.format_exc())
            overall_success = False
        
        return overall_success, self.logs