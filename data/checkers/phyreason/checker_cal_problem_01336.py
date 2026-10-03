import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Extract inputs with safe attribute access
        self.F = self._extract_value(solution_instance, 'F')
        self.m = self._extract_value(solution_instance, 'm')
        self.t1 = self._extract_value(solution_instance, 't1')
        
        # Extract results using key mapping
        self.v1 = self._extract_value(solution_instance, 'v_1', result_key='v_1')
        self.P1 = self._extract_value(solution_instance, 'P_1', result_key='P_1')
        self.overlinev = self._extract_value(solution_instance, 'overlinev', result_key='overlinev')
        self.overlineP = self._extract_value(solution_instance, 'overlineP', result_key='overlineP')
    
    def _extract_value(self, obj, attr_name, result_key=None):
        """
        Safely extract a numeric value from the solution instance.
        Priority: direct attribute > inputs dict > results dict.
        Returns a float or 0.0 if not found/invalid.
        """
        # Try direct attribute
        value = getattr(obj, attr_name, None)
        if value is not None:
            return self._to_float(value)
        
        # Try inputs dictionary
        inputs = getattr(obj, 'inputs', {})
        if attr_name in inputs:
            return self._to_float(inputs[attr_name])
        
        # Try results dictionary with provided key
        results = getattr(obj, 'results', {})
        key = result_key if result_key else attr_name
        if key in results:
            return self._to_float(results[key])
        
        # Fallback: try to get as attribute with underscore
        if '_' in attr_name:
            alt_key = attr_name.replace('_', '')
            value = getattr(obj, alt_key, None)
            if value is not None:
                return self._to_float(value)
        
        return 0.0
    
    def _to_float(self, value):
        """
        Convert any value to a float, handling sympy expressions.
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        # Try direct conversion
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        
        # Try sympify and substitution
        try:
            expr = sp.sympify(str(value))
            symbols = {'F': self.F, 'm': self.m, 't1': self.t1}
            return float(expr.subs(symbols).evalf())
        except:
            return 0.0
    
    def test_invariants(self):
        """
        Verify fundamental physical laws:
        1. Newton's Second Law: v1 = (F/m)*t1
        2. Instantaneous power: P1 = F * v1
        3. Average velocity: overlinev = v1/2
        4. Average power: overlineP = F * overlinev
        """
        logs = []
        
        # 1. Newton's Second Law
        v1_expected = (self.F / self.m) * self.t1
        if not math.isclose(self.v1, v1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v1 fails Newton's Second Law: expected {v1_expected}, got {self.v1}")
        
        # 2. Instantaneous power definition
        P1_expected = self.F * self.v1
        if not math.isclose(self.P1, P1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"P1 fails instantaneous power definition: expected {P1_expected}, got {self.P1}")
        
        # 3. Average velocity for constant acceleration
        overlinev_expected = self.v1 / 2
        if not math.isclose(self.overlinev, overlinev_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"overlinev fails average velocity relation: expected {overlinev_expected}, got {self.overlinev}")
        
        # 4. Average power definition
        overlineP_expected = self.F * self.overlinev
        if not math.isclose(self.overlineP, overlineP_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"overlineP fails average power definition: expected {overlineP_expected}, got {self.overlineP}")
        
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        
        # Check ratio between instantaneous and average values
        if not math.isclose(self.P1, 2 * self.overlineP, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"P1 should be twice overlineP: P1={self.P1}, 2*overlineP={2*self.overlineP}")
        
        if not math.isclose(self.v1, 2 * self.overlinev, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v1 should be twice overlinev: v1={self.v1}, 2*overlinev={2*self.overlinev}")
        
        # Check derived formulas directly
        P1_formula = self.F**2 * self.t1 / self.m
        if not math.isclose(self.P1, P1_formula, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"P1 doesn't match derived formula F²t/m: expected {P1_formula}, got {self.P1}")
        
        overlineP_formula = self.F**2 * self.t1 / (2 * self.m)
        if not math.isclose(self.overlineP, overlineP_formula, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"overlineP doesn't match derived formula F²t/(2m): expected {overlineP_formula}, got {self.overlineP}")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb inputs and verify expected trends.
        """
        logs = []
        
        # Perturb force F (increase by 10%)
        F_pert = self.F * 1.1
        v1_pert = (F_pert / self.m) * self.t1
        P1_pert = F_pert * v1_pert
        overlinev_pert = v1_pert / 2
        overlineP_pert = F_pert * overlinev_pert
        
        if not (v1_pert > self.v1):
            logs.append(f"v1 should increase with F: {v1_pert} <= {self.v1}")
        if not (P1_pert > self.P1):
            logs.append(f"P1 should increase with F: {P1_pert} <= {self.P1}")
        if not (overlinev_pert > self.overlinev):
            logs.append(f"overlinev should increase with F: {overlinev_pert} <= {self.overlinev}")
        if not (overlineP_pert > self.overlineP):
            logs.append(f"overlineP should increase with F: {overlineP_pert} <= {self.overlineP}")
        
        # Perturb mass m (increase by 10%)
        m_pert = self.m * 1.1
        v1_pert_m = (self.F / m_pert) * self.t1
        P1_pert_m = self.F * v1_pert_m
        overlinev_pert_m = v1_pert_m / 2
        overlineP_pert_m = self.F * overlinev_pert_m
        
        if not (v1_pert_m < self.v1):
            logs.append(f"v1 should decrease with m: {v1_pert_m} >= {self.v1}")
        if not (P1_pert_m < self.P1):
            logs.append(f"P1 should decrease with m: {P1_pert_m} >= {self.P1}")
        if not (overlinev_pert_m < self.overlinev):
            logs.append(f"overlinev should decrease with m: {overlinev_pert_m} >= {self.overlinev}")
        if not (overlineP_pert_m < self.overlineP):
            logs.append(f"overlineP should decrease with m: {overlineP_pert_m} >= {self.overlineP}")
        
        return logs
    
    def run_all_tests(self):
        """
        Execute all tests and return success status and logs.
        """
        all_logs = []
        success = True
        
        # Run each test
        invariants_logs = self.test_invariants()
        if invariants_logs:
            success = False
            all_logs.append("=== Invariants Test ===")
            all_logs.extend(invariants_logs)
        else:
            all_logs.append("Invariants test passed.")
        
        trace_logs = self.test_trace_consistency()
        if trace_logs:
            success = False
            all_logs.append("=== Trace Consistency Test ===")
            all_logs.extend(trace_logs)
        else:
            all_logs.append("Trace consistency test passed.")
        
        monotonicity_logs = self.test_monotonicity()
        if monotonicity_logs:
            success = False
            all_logs.append("=== Monotonicity Test ===")
            all_logs.extend(monotonicity_logs)
        else:
            all_logs.append("Monotonicity test passed.")
        
        return success, all_logs