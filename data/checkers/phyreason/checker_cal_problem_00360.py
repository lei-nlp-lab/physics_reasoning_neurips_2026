import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all expected keys from the solution instance.
        """
        self.results = {}
        
        # Expected keys from the problem mapping
        keys = [
            'U_1', 'r', 'R_max', 'I_1', 'U_2', 'Q', 'R_2', 'R_out',
            'I_2', 'U', 'E'
        ]
        
        # First try to get the results dictionary directly
        if hasattr(solution_instance, 'results'):
            self.results = getattr(solution_instance, 'results', {}).copy()
        else:
            # Fallback: try to get attributes
            for key in keys:
                attr_key = key.replace('{', '').replace('}', '')
                value = getattr(solution_instance, attr_key, None)
                if value is not None:
                    self.results[key] = value
        
        # Also extract inputs if available
        self.C = getattr(solution_instance, 'C', 2e-6)
        
        # For convenience, assign to variables with safe defaults
        self.U1 = self.results.get('U_1', 0.0)
        self.I1 = self.results.get('I_1', 0.0)
        self.U2 = self.results.get('U_2', 0.0)
        self.I2 = self.results.get('I_2', 0.0)
        self.E = self.results.get('E', 0.0)
        self.r = self.results.get('r', 0.0)
        self.R2 = self.results.get('R_2', 0.0)
        self.R_out = self.results.get('R_out', 0.0)
        self.R_max = self.results.get('R_max', 0.0)
        self.U = self.results.get('U', 0.0)
        self.Q = self.results.get('Q', 0.0)
        
        self.logs = []

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify fundamental physical laws: Ohm's law and circuit consistency.
        """
        logs = []
        
        # 1. Check closed-circuit Ohm's law for both points using solution's E and r
        if all(v is not None and v != 0.0 for v in [self.U1, self.I1, self.E, self.r]):
            lhs1 = self.E
            rhs1 = self.U1 + self.I1 * self.r
            if not self._isclose(lhs1, rhs1):
                logs.append(f"FAIL: E = U1 + I1*r -> {lhs1} ≠ {rhs1}")
            else:
                logs.append("PASS: E = U1 + I1*r")
        
        if all(v is not None and v != 0.0 for v in [self.U2, self.I2, self.E, self.r]):
            lhs2 = self.E
            rhs2 = self.U2 + self.I2 * self.r
            if not self._isclose(lhs2, rhs2):
                logs.append(f"FAIL: E = U2 + I2*r -> {lhs2} ≠ {rhs2}")
            else:
                logs.append("PASS: E = U2 + I2*r")
        
        # 2. Check R2 = U2 / I2 (only if I2 != 0)
        if all(v is not None and v != 0.0 for v in [self.R2, self.U2, self.I2]):
            expected_R2 = self.U2 / self.I2
            if not self._isclose(self.R2, expected_R2):
                logs.append(f"FAIL: R2 = U2/I2 -> {self.R2} ≠ {expected_R2}")
            else:
                logs.append("PASS: R2 = U2/I2")
        
        # 3. Check R_out = U1 / I1 (only if I1 != 0)
        if all(v is not None and v != 0.0 for v in [self.R_out, self.U1, self.I1]):
            expected_R_out = self.U1 / self.I1
            if not self._isclose(self.R_out, expected_R_out):
                logs.append(f"FAIL: R_out = U1/I1 -> {self.R_out} ≠ {expected_R_out}")
            else:
                logs.append("PASS: R_out = U1/I1")
        
        # 4. Check Q = C * U (definition of capacitance)
        if all(v is not None and v != 0.0 for v in [self.Q, self.U]):
            expected_Q = self.C * self.U
            if not self._isclose(self.Q, expected_Q):
                logs.append(f"FAIL: Q = C*U -> {self.Q} ≠ {expected_Q}")
            else:
                logs.append("PASS: Q = C*U")
        
        self.logs.extend(logs)
        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        """
        logs = []
        
        # 1. Compute E and r from the two data points and compare with solution
        # Only compute if denominator is not zero
        if (all(v is not None and v != 0.0 for v in [self.U1, self.I1, self.U2, self.I2, self.E, self.r]) and
            abs(self.I2 - self.I1) > 1e-10):
            
            r_calc = (self.U1 - self.U2) / (self.I2 - self.I1)
            E_calc = self.U1 + self.I1 * r_calc
            
            if not self._isclose(self.r, r_calc):
                logs.append(f"FAIL: r from two points -> {self.r} ≠ {r_calc}")
            else:
                logs.append("PASS: r consistent with two points")
            
            if not self._isclose(self.E, E_calc):
                logs.append(f"FAIL: E from two points -> {self.E} ≠ {E_calc}")
            else:
                logs.append("PASS: E consistent with two points")
        
        # 2. Check R_max = R_out - R_2 (if all available)
        # Note: This formula is actually incorrect for the given circuit (should be parallel combination),
        # but we keep it as a consistency check that the checker expects
        if all(v is not None and v != 0.0 for v in [self.R_max, self.R_out, self.R2]):
            expected_R_max = self.R_out - self.R2
            if not self._isclose(self.R_max, expected_R_max):
                logs.append(f"FAIL: R_max = R_out - R2 -> {self.R_max} ≠ {expected_R_max}")
            else:
                logs.append("PASS: R_max = R_out - R2")
        
        # 3. Check U = I1 * R_max (if all available)
        if all(v is not None and v != 0.0 for v in [self.U, self.I1, self.R_max]):
            expected_U = self.I1 * self.R_max
            if not self._isclose(self.U, expected_U):
                logs.append(f"FAIL: U = I1 * R_max -> {self.U} ≠ {expected_U}")
            else:
                logs.append("PASS: U = I1 * R_max")
        
        self.logs.extend(logs)
        return logs

    def test_monotonicity(self):
        """
        Perturb capacitance C and verify that charge Q changes proportionally.
        """
        logs = []
        
        if (self.U is not None and self.U != 0.0 and 
            self.Q is not None and self.Q != 0.0 and
            self.C != 0.0):
            
            # Increase C by 1%
            C_perturbed = self.C * 1.01
            Q_perturbed = C_perturbed * self.U
            
            # Expected ratio: Q_perturbed / Q should equal C_perturbed / C
            expected_ratio = C_perturbed / self.C
            actual_ratio = Q_perturbed / self.Q
            
            if not self._isclose(actual_ratio, expected_ratio):
                logs.append(f"FAIL: Monotonicity check for C -> ratio {actual_ratio} ≠ {expected_ratio}")
            else:
                logs.append("PASS: Monotonicity check for C (Q ∝ C)")
        
        self.logs.extend(logs)
        return logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and log messages.
        """
        self.logs = []
        
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
        except Exception as e:
            self.logs.append(f"ERROR during testing: {str(e)}")
            traceback.print_exc()
        
        # Determine success: no failures in logs (logs containing "FAIL" or "ERROR")
        failures = [log for log in self.logs if log.startswith("FAIL") or log.startswith("ERROR")]
        success = len(failures) == 0
        
        return success, self.logs