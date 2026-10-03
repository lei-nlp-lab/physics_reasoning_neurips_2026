import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr to avoid AttributeError in the sandbox.
        """
        self.inputs = {}
        self.results = {}
        
        # Expected inputs (from problem context)
        input_keys = ['t1', 'x_OP', 'x_PC']
        for key in input_keys:
            self.inputs[key] = getattr(solution_instance, key, None)
            if self.inputs[key] is None:
                # Fallback to results dict if present
                self.inputs[key] = getattr(solution_instance, 'results', {}).get(key, None)
        
        # Expected results (from key mapping)
        result_keys = ['T', 'v', 'lambda', 't_2']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, None)
            if self.results[key] is None:
                # Fallback to results dict if present
                self.results[key] = getattr(solution_instance, 'results', {}).get(key, None)
        
        # Convert to float if they are sympy objects
        for key in self.inputs:
            if isinstance(self.inputs[key], sp.Basic):
                self.inputs[key] = float(self.inputs[key])
        for key in self.results:
            if isinstance(self.results[key], sp.Basic):
                self.results[key] = float(self.results[key])
        
        # Store symbolic variables for invariant checks
        self.symbols = {
            't1': sp.symbols('t1'),
            'x_OP': sp.symbols('x_OP'),
            'x_PC': sp.symbols('x_PC'),
            'T': sp.symbols('T'),
            'v': sp.symbols('v'),
            'lambda': sp.symbols('lambda'),
            't_2': sp.symbols('t_2')
        }
        
        self.logs = []

    def log(self, message, level="INFO"):
        """Add a log message."""
        self.logs.append(f"{level}: {message}")

    def safe_isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify physical invariants from the knowledge base."""
        logs = []
        try:
            # Extract values
            t1 = self.inputs.get('t1')
            x_OP = self.inputs.get('x_OP')
            x_PC = self.inputs.get('x_PC')
            v = self.results.get('v')
            lam = self.results.get('lambda')
            T = self.results.get('T')
            t2 = self.results.get('t_2')
            
            # Check 1: Wave speed from uniform motion (v = x_OP / t1)
            if None not in (v, x_OP, t1):
                expected_v = x_OP / t1
                if self.safe_isclose(v, expected_v):
                    logs.append("PASS: Wave speed v = x_OP / t1 holds.")
                else:
                    logs.append(f"FAIL: Wave speed invariant. v = {v}, but x_OP/t1 = {expected_v}.")
            else:
                logs.append("SKIP: Missing data for wave speed invariant.")
            
            # Check 2: Wavelength from geometry (x_OP = (3/2) * lambda)
            if None not in (lam, x_OP):
                expected_lam = (2/3) * x_OP
                if self.safe_isclose(lam, expected_lam):
                    logs.append("PASS: Wavelength relation x_OP = (3/2) * lambda holds.")
                else:
                    logs.append(f"FAIL: Wavelength invariant. lambda = {lam}, but (2/3)*x_OP = {expected_lam}.")
            else:
                logs.append("SKIP: Missing data for wavelength invariant.")
            
            # Check 3: Period from v and lambda (T = lambda / v)
            if None not in (T, lam, v):
                expected_T = lam / v
                if self.safe_isclose(T, expected_T):
                    logs.append("PASS: Period relation T = lambda / v holds.")
                else:
                    logs.append(f"FAIL: Period invariant. T = {T}, but lambda/v = {expected_T}.")
            else:
                logs.append("SKIP: Missing data for period invariant.")
            
            # Check 4: Time t2 from propagation (t2 = t1 + (lambda/4 + x_PC) / v)
            if None not in (t2, t1, lam, x_PC, v):
                expected_t2 = t1 + (lam/4 + x_PC) / v
                if self.safe_isclose(t2, expected_t2):
                    logs.append("PASS: Time t2 propagation formula holds.")
                else:
                    logs.append(f"FAIL: t2 invariant. t2 = {t2}, but t1+(lambda/4+x_PC)/v = {expected_t2}.")
            else:
                logs.append("SKIP: Missing data for t2 invariant.")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return all("PASS" in log or "SKIP" in log for log in logs), logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with the final answer."""
        logs = []
        try:
            # The steps analysis gives explicit values: v=20, lambda=4, T=0.2, t2=0.8
            # We compare with the results (within tolerance)
            v = self.results.get('v')
            lam = self.results.get('lambda')
            T = self.results.get('T')
            t2 = self.results.get('t_2')
            
            expected = {'v': 20.0, 'lambda': 4.0, 'T': 0.2, 't_2': 0.8}
            for key, exp_val in expected.items():
                val = self.results.get(key)
                if val is not None:
                    if self.safe_isclose(val, exp_val):
                        logs.append(f"PASS: {key} matches expected value {exp_val}.")
                    else:
                        logs.append(f"FAIL: {key} = {val} differs from expected {exp_val}.")
                else:
                    logs.append(f"SKIP: {key} not available.")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return all("PASS" in log or "SKIP" in log for log in logs), logs

    def test_monotonicity(self):
        """
        Perturb an input (e.g., x_OP) and check the monotonic trend of dependent quantities.
        Use formulas to recompute, then restore the original.
        """
        logs = []
        try:
            # Original values
            x_OP_orig = self.inputs.get('x_OP')
            if x_OP_orig is None:
                logs.append("SKIP: x_OP not available for monotonicity test.")
                self.logs.extend(logs)
                return True, logs
            
            # Perturb by +10%
            perturbed_x_OP = x_OP_orig * 1.1
            
            # Recompute using formulas (as per steps analysis)
            t1 = self.inputs.get('t1')
            if t1 is None:
                logs.append("SKIP: t1 not available for monotonicity test.")
                self.logs.extend(logs)
                return True, logs
            
            # Formula 1: v = x_OP / t1 (but t1 is fixed, so v increases with x_OP)
            new_v = perturbed_x_OP / t1
            
            # Formula 2: lambda = (2/3) * x_OP (direct proportionality)
            new_lambda = (2/3) * perturbed_x_OP
            
            # Formula 3: T = lambda / v
            new_T = new_lambda / new_v
            
            # Formula 4: t2 = t1 + (lambda/4 + x_PC) / v
            x_PC = self.inputs.get('x_PC')
            if x_PC is None:
                logs.append("SKIP: x_PC not available for monotonicity test.")
                self.logs.extend(logs)
                return True, logs
            new_t2 = t1 + (new_lambda/4 + x_PC) / new_v
            
            # Check monotonic trends (all should increase because x_OP increased)
            v_orig = self.results.get('v')
            lam_orig = self.results.get('lambda')
            T_orig = self.results.get('T')
            t2_orig = self.results.get('t_2')
            
            if v_orig is not None and new_v > v_orig:
                logs.append("PASS: v increases with x_OP (as expected).")
            elif v_orig is not None:
                logs.append(f"FAIL: v monotonicity. Original {v_orig}, perturbed {new_v}.")
            
            if lam_orig is not None and new_lambda > lam_orig:
                logs.append("PASS: lambda increases with x_OP (direct proportionality).")
            elif lam_orig is not None:
                logs.append(f"FAIL: lambda monotonicity. Original {lam_orig}, perturbed {new_lambda}.")
            
            # T = lambda / v, but lambda and v both increase proportionally (lambda = (2/3)x_OP, v = x_OP/t1)
            # So T = ((2/3)x_OP) / (x_OP/t1) = (2/3) * t1, which is constant. Thus T should not change.
            if T_orig is not None and self.safe_isclose(new_T, T_orig):
                logs.append("PASS: T remains constant when x_OP changes (as per theory).")
            elif T_orig is not None:
                logs.append(f"FAIL: T should be constant. Original {T_orig}, perturbed {new_T}.")
            
            # t2 = t1 + (lambda/4 + x_PC)/v. Since lambda and v both increase proportionally, 
            # the term (lambda/4)/v = (1/4)*(lambda/v) = (1/4)* ( (2/3)x_OP ) / (x_OP/t1) ) = (1/4)*(2/3 t1) = t1/6 constant.
            # So t2 = t1 + constant + x_PC/v. Since v increases, x_PC/v decreases, so t2 decreases.
            if t2_orig is not None and new_t2 < t2_orig:
                logs.append("PASS: t2 decreases with increasing x_OP (as expected from v increase).")
            elif t2_orig is not None:
                logs.append(f"FAIL: t2 monotonicity. Original {t2_orig}, perturbed {new_t2}.")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return all("PASS" in log or "SKIP" in log for log in logs), logs

    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        self.logs = []  # Reset logs
        success = True
        all_logs = []
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            self.log(f"Running {test_name} test...")
            test_success, logs = test_method()
            all_logs.extend(logs)
            if not test_success:
                success = False
        
        return success, all_logs