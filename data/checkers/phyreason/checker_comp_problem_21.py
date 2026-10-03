import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        self.solution = solution_instance
        
        # Extract results safely from results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Map expected keys from problem context - store as is (could be float, str, or symbolic)
        self.n0 = results.get('n_0', None)
        self.a = results.get('a', None)
        self.b = results.get('b', None)
        self.n_e = results.get('n_e', None)
        self.I = results.get('I', None)
        self.rho = results.get('rho', None)
        self.A1 = results.get('A_1', None)
        self.A2 = results.get('A_2', None)
        self.B1 = results.get('B_1', None)
        self.B2 = results.get('B_2', None)
        self.I_iL = results.get('I_iL', None)
        self.I_e0 = results.get('I_e0', None)
        self.I_self = results.get('I_self', None)
        self.L_cr = results.get('L_cr', None)
        
        # Additional intermediate results
        self.n_e1 = results.get('n_e1', None)
        self.n_e2 = results.get('n_e2', None)
        self.ab = results.get('ab', None)
        self.ba_ratio = results.get('b/a', None)
        self.C1 = results.get('C_1', None)
        self.C2 = results.get('C_2', None)
        self.I_half = results.get('I/2', None)
        
        # Get input parameters with defaults
        self.Z_ext = getattr(solution_instance, 'Z_ext', 1.0)
        self.r = getattr(solution_instance, 'r', 1.0)
        self.U = getattr(solution_instance, 'U', 1.0)
        self.S = getattr(solution_instance, 'S', 1.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.beta = getattr(solution_instance, 'beta', 1.0)
        self.alpha = getattr(solution_instance, 'alpha', 1.0)
        self.gamma = getattr(solution_instance, 'gamma', 1.0)
        self.e = getattr(solution_instance, 'e', 1.602e-19)
        
        self.logs = []
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        # If either is not a number, return True if they are equal as strings (for formulas)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return str(a) == str(b)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _is_numeric(self, val):
        """Check if value is numeric."""
        return isinstance(val, (int, float))
    
    def test_invariants(self):
        """Verify physical invariants from KB theorems."""
        logs = []
        success = True
        
        # 1. Check superposition of electron densities (only numeric check)
        if self._is_numeric(self.n_e1) and self._is_numeric(self.n_e2) and self._is_numeric(self.n_e):
            expected = math.sqrt(self.n_e1**2 + self.n_e2**2)
            if not math.isclose(self.n_e, expected, rel_tol=1e-2):
                logs.append(f"Superposition violation: n_e = {self.n_e} != sqrt(n_e1^2+n_e2^2) = {expected}")
                success = False
        
        # 2. Check self-sustainment condition (numeric only)
        if self._is_numeric(self.L_cr) and self._is_numeric(self.alpha) and self._is_numeric(self.gamma):
            expected_L_cr = (1/self.alpha) * math.log(1 + 1/self.gamma)
            if not math.isclose(self.L_cr, expected_L_cr, rel_tol=1e-2):
                logs.append(f"Self-sustainment condition violation: L_cr = {self.L_cr} != (1/alpha)*ln(1+1/gamma) = {expected_L_cr}")
                success = False
        
        self.logs.extend(logs)
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check step_2: n0 should be 0 (numeric)
        if self._is_numeric(self.n0) and not math.isclose(self.n0, 0.0, abs_tol=1e-10):
            logs.append(f"Trace inconsistency: n0 = {self.n0} != 0")
            success = False
        
        # Check step_4: ab = Z_ext and b/a = r (symbolic/formula checks)
        if self.ab is not None and self.Z_ext is not None:
            if str(self.ab) != str(self.Z_ext):
                logs.append(f"Trace inconsistency: ab = {self.ab} != Z_ext = {self.Z_ext}")
                success = False
        
        if self.ba_ratio is not None and self.r is not None:
            if str(self.ba_ratio) != str(self.r):
                logs.append(f"Trace inconsistency: b/a = {self.ba_ratio} != r = {self.r}")
                success = False
        
        # Check step_5: a = sqrt(Z_ext/r), b = sqrt(r*Z_ext) (formula checks)
        if self.a is not None and self.Z_ext is not None and self.r is not None:
            expected_a = f"sqrt({self.Z_ext}/{self.r})" if isinstance(self.Z_ext, (int, float)) and isinstance(self.r, (int, float)) else "sqrt(Z_ext/r)"
            if str(self.a) != expected_a and not (isinstance(self.a, (int, float)) and isinstance(self.Z_ext, (int, float)) and isinstance(self.r, (int, float)) and math.isclose(self.a, math.sqrt(self.Z_ext/self.r), rel_tol=1e-2)):
                logs.append(f"Trace inconsistency: a = {self.a} != sqrt(Z_ext/r)")
                success = False
        
        if self.b is not None and self.Z_ext is not None and self.r is not None:
            expected_b = f"sqrt({self.r}*{self.Z_ext})" if isinstance(self.r, (int, float)) and isinstance(self.Z_ext, (int, float)) else "sqrt(r*Z_ext)"
            if str(self.b) != expected_b and not (isinstance(self.b, (int, float)) and isinstance(self.r, (int, float)) and isinstance(self.Z_ext, (int, float)) and math.isclose(self.b, math.sqrt(self.r*self.Z_ext), rel_tol=1e-2)):
                logs.append(f"Trace inconsistency: b = {self.b} != sqrt(r*Z_ext)")
                success = False
        
        # Check step_7: n_e = sqrt(n_e1^2 + n_e2^2) already in invariants
        
        # Check step_12: I formula - skip if I is string formula
        if self._is_numeric(self.I) and all(self._is_numeric(v) for v in [self.e, self.beta, self.U, self.S, self.r, self.L, self.Z_ext]):
            term = (4 * self.r * self.Z_ext * self.L**4) / (self.beta**2 * self.U**2)
            expected_I = (self.e * self.beta**2 * self.U**2 * self.S) / (self.r * self.L**3) * (math.sqrt(1 + term) - 1)
            if not math.isclose(self.I, expected_I, rel_tol=1e-2):
                logs.append(f"Trace inconsistency: I = {self.I} != formula gives {expected_I}")
                success = False
        
        # Check step_14: rho formula - skip if rho is string
        if self._is_numeric(self.rho) and all(self._is_numeric(v) for v in [self.e, self.beta, self.r, self.Z_ext]) and self.Z_ext != 0:
            expected_rho = (1/(2*self.e*self.beta)) * math.sqrt(self.r/self.Z_ext)
            if not math.isclose(self.rho, expected_rho, rel_tol=1e-2):
                logs.append(f"Trace inconsistency: rho = {self.rho} != formula gives {expected_rho}")
                success = False
        
        # Check step_18: A1 = alpha, A2 = -e*Z_ext*S/alpha (formula checks)
        if self.A1 is not None and self.alpha is not None:
            if str(self.A1) != str(self.alpha):
                logs.append(f"Trace inconsistency: A1 = {self.A1} != alpha = {self.alpha}")
                success = False
        
        if self.A2 is not None and self.e is not None and self.Z_ext is not None and self.S is not None and self.alpha is not None:
            expected_A2 = f"-{self.e}*{self.Z_ext}*{self.S}/{self.alpha}" if all(isinstance(v, (int, float)) for v in [self.e, self.Z_ext, self.S, self.alpha]) else "-e*Z_ext*S/alpha"
            if str(self.A2) != expected_A2:
                logs.append(f"Trace inconsistency: A2 = {self.A2} != -e*Z_ext*S/alpha")
                success = False
        
        # Check step_22: B1 = -C1, B2 = alpha (formula checks)
        if self.B1 is not None and self.C1 is not None:
            if str(self.B1) != f"-{self.C1}":
                logs.append(f"Trace inconsistency: B1 = {self.B1} != -C1")
                success = False
        
        if self.B2 is not None and self.alpha is not None:
            if str(self.B2) != str(self.alpha):
                logs.append(f"Trace inconsistency: B2 = {self.B2} != alpha = {self.alpha}")
                success = False
        
        # Check step_23: I_i(L) = 0 (numeric)
        if self._is_numeric(self.I_iL) and not math.isclose(self.I_iL, 0.0, abs_tol=1e-10):
            logs.append(f"Trace inconsistency: I_i(L) = {self.I_iL} != 0")
            success = False
        
        # Check step_29: Self-sustained current formula - skip if string
        if self._is_numeric(self.I_self) and all(self._is_numeric(v) for v in [self.e, self.Z_ext, self.S, self.alpha, self.gamma, self.L]):
            numerator = math.exp(self.alpha * self.L)
            denominator = 1 - self.gamma + self.gamma * numerator
            expected_I_self = (self.e * self.Z_ext * self.S / self.alpha) * (numerator/denominator - 1)
            if not math.isclose(self.I_self, expected_I_self, rel_tol=1e-2):
                logs.append(f"Trace inconsistency: I_self = {self.I_self} != formula gives {expected_I_self}")
                success = False
        
        self.logs.extend(logs)
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends - numeric only."""
        logs = []
        success = True
        
        # Only run if we have numeric values
        try:
            # 1. Perturb Z_ext and check n_e increases (if numeric)
            if all(self._is_numeric(v) for v in [self.Z_ext, self.r, self.n_e]):
                perturbed_Z_ext = self.Z_ext * 1.1
                perturbed_n_e = math.sqrt(perturbed_Z_ext / self.r)
                if not perturbed_n_e > self.n_e:
                    logs.append("Monotonicity violation: n_e should increase with Z_ext")
                    success = False
            
            # 2. Perturb U and check I increases (if numeric)
            if all(self._is_numeric(v) for v in [self.U, self.I, self.e, self.beta, self.S, self.r, self.L, self.Z_ext]):
                perturbed_U = self.U * 1.1
                term = (4 * self.r * self.Z_ext * self.L**4) / (self.beta**2 * perturbed_U**2)
                perturbed_I = (self.e * self.beta**2 * perturbed_U**2 * self.S) / (self.r * self.L**3) * (math.sqrt(1 + term) - 1)
                if not perturbed_I > self.I:
                    logs.append("Monotonicity violation: I should increase with U")
                    success = False
            
            # 3. Perturb r and check n_e decreases (if numeric)
            if all(self._is_numeric(v) for v in [self.r, self.Z_ext, self.n_e]):
                perturbed_r = self.r * 1.1
                perturbed_n_e = math.sqrt(self.Z_ext / perturbed_r)
                if not perturbed_n_e < self.n_e:
                    logs.append("Monotonicity violation: n_e should decrease with r")
                    success = False
            
            # 4. Perturb L and check I decreases (low voltage approx, if numeric)
            if all(self._is_numeric(v) for v in [self.L, self.e, self.beta, self.U, self.Z_ext, self.r, self.S]):
                perturbed_L = self.L * 1.1
                perturbed_I = (2*self.e*self.beta*self.U*math.sqrt(self.Z_ext*self.r))/perturbed_L
                original_approx = (2*self.e*self.beta*self.U*math.sqrt(self.Z_ext*self.r))/self.L
                if not perturbed_I < original_approx:
                    logs.append("Monotonicity violation: I should decrease with L (low voltage approx)")
                    success = False
            
            # 5. Perturb gamma and check L_cr decreases (if numeric)
            if all(self._is_numeric(v) for v in [self.gamma, self.L_cr, self.alpha]):
                perturbed_gamma = self.gamma * 1.1
                perturbed_L_cr = (1/self.alpha) * math.log(1 + 1/perturbed_gamma)
                if not perturbed_L_cr < self.L_cr:
                    logs.append("Monotonicity violation: L_cr should decrease with gamma")
                    success = False
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {e}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def run_all_tests(self):
        """Run all checks and return success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
                    self.logs.append(f"{test_name} failed.")
                else:
                    self.logs.append(f"{test_name} passed.")
            except Exception as e:
                self.logs.append(f"{test_name} error: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, self.logs