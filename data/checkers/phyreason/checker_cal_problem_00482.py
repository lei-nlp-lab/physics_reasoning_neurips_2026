import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results with robust fallbacks
        self.inputs = {}
        self.results = {}
        
        # Inputs from problem context (use FIXED constants from problem statement)
        self.inputs['E'] = 2e4  # Fixed from problem: E = 2×10^4 N/C
        self.inputs['q'] = 1e-8  # Fixed from problem: q = +1×10^-8 C
        self.inputs['x'] = 0.2  # Fixed from problem: x = 0.2 m
        self.inputs['theta'] = 60  # Fixed from problem: θ = 60°
        
        # Extract required output variables ONLY from results dict
        if hasattr(solution_instance, 'results'):
            results_dict = solution_instance.results
            # Ensure we get float values, handle None or other types
            self.results['F'] = float(results_dict.get('F', 0.0))
            self.results['W'] = float(results_dict.get('W', 0.0))
            self.results['delta_E_P'] = float(results_dict.get('delta_E_P', 0.0))
            self.results['U_AB'] = float(results_dict.get('U_AB', 0.0))
        else:
            # Fallback to direct attributes
            self.results['F'] = float(getattr(solution_instance, 'F', 0.0))
            self.results['W'] = float(getattr(solution_instance, 'W', 0.0))
            self.results['delta_E_P'] = float(getattr(solution_instance, 'delta_E_P', 0.0))
            self.results['U_AB'] = float(getattr(solution_instance, 'U_AB', 0.0))
        
        # Expected values from steps analysis (for trace consistency)
        self.expected_F = 2e-4
        self.expected_W = 2e-5
        self.expected_delta_E_P = -2e-5
        self.expected_U_AB = 2e3
        
        # Constants
        self.theta_rad = math.radians(self.inputs['theta'])
        
        # Logs
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)

    def test_invariants(self):
        """Verify energy conservation and force relationships based on KB."""
        try:
            # Numerical checks with fuzzy matching
            # 1. Electric force invariant: F = qE
            F_calc = self.inputs['q'] * self.inputs['E']
            if not math.isclose(self.results['F'], F_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Electric force invariant F = qE. Expected ~{F_calc:.2e}, got {self.results['F']:.2e}")
                return False
            
            # 2. Work-energy relationship: W = -ΔE_P
            if not math.isclose(self.results['W'], -self.results['delta_E_P'], rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Work-energy invariant W = -ΔE_P. W={self.results['W']:.2e}, ΔE_P={self.results['delta_E_P']:.2e}")
                return False
            
            # 3. Potential difference relationship: U_AB = W/q
            # Handle division by zero carefully
            if abs(self.inputs['q']) < 1e-12:
                self._log("ERROR: Charge q is zero, cannot compute U_AB = W/q")
                return False
            
            # Check if W is approximately zero to avoid division issues
            if abs(self.results['W']) < 1e-12:
                # If W is zero, U_AB should also be zero (or very small)
                if not math.isclose(self.results['U_AB'], 0.0, rel_tol=1e-2, abs_tol=1e-12):
                    self._log(f"FAIL: When W≈0, U_AB should be ≈0, got {self.results['U_AB']:.2e}")
                    return False
            else:
                U_calc = self.results['W'] / self.inputs['q']
                if not math.isclose(self.results['U_AB'], U_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: Potential difference invariant U_AB = W/q. Expected ~{U_calc:.2e}, got {self.results['U_AB']:.2e}")
                    return False
            
            # 4. Force direction logic (from KB: positive charge => force along E field)
            # Since we can't access direction directly, we check the sign of work:
            # For positive q moving at 60° to E field (cos60=0.5), work should be positive.
            if self.inputs['q'] > 0 and self.inputs['theta'] == 60:
                if self.results['W'] <= 0:
                    self._log(f"WARNING: For positive q at 60°, work should be positive, got {self.results['W']:.2e}")
                    # This is just a warning, not a failure
            
            self._log("PASS: All invariants verified.")
            return True
        except Exception as e:
            self._log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with expected steps."""
        try:
            # Compare each result to expected values from steps analysis
            checks = [
                ('F', self.expected_F, self.results['F']),
                ('W', self.expected_W, self.results['W']),
                ('delta_E_P', self.expected_delta_E_P, self.results['delta_E_P']),
                ('U_AB', self.expected_U_AB, self.results['U_AB'])
            ]
            
            for name, expected, actual in checks:
                if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: Trace inconsistency for {name}. Expected ~{expected:.2e}, got {actual:.2e}")
                    return False
            
            # Additionally, verify the step-by-step calculations
            F_step = self.inputs['q'] * self.inputs['E']
            W_step = F_step * self.inputs['x'] * math.cos(self.theta_rad)
            delta_E_P_step = -W_step
            U_AB_step = W_step / self.inputs['q']
            
            step_checks = [
                ('F_step', F_step, self.results['F']),
                ('W_step', W_step, self.results['W']),
                ('delta_E_P_step', delta_E_P_step, self.results['delta_E_P']),
                ('U_AB_step', U_AB_step, self.results['U_AB'])
            ]
            
            for name, step_val, actual in step_checks:
                if not math.isclose(step_val, actual, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: Step calculation mismatch for {name}. Expected ~{step_val:.2e}, got {actual:.2e}")
                    return False
            
            self._log("PASS: Trace consistency verified.")
            return True
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb an input and check expected trends."""
        try:
            original_q = self.inputs['q']
            perturbed_q = original_q * 1.1  # Increase by 10%
            
            # Calculate expected perturbed results using formulas
            perturbed_F = perturbed_q * self.inputs['E']
            perturbed_W = perturbed_F * self.inputs['x'] * math.cos(self.theta_rad)
            perturbed_delta_E_P = -perturbed_W
            perturbed_U_AB = perturbed_W / perturbed_q if abs(perturbed_q) > 1e-12 else 0.0
            
            # Check monotonic relationships from KB:
            # 1. Force should increase with q (linear)
            if perturbed_q > original_q:
                if not (perturbed_F > self.results['F']):
                    self._log(f"FAIL: Monotonicity - Force should increase with q. Original F={self.results['F']:.2e}, perturbed F={perturbed_F:.2e}")
                    return False
                # 2. Work should increase with q (linear)
                if not (perturbed_W > self.results['W']):
                    self._log(f"FAIL: Monotonicity - Work should increase with q. Original W={self.results['W']:.2e}, perturbed W={perturbed_W:.2e}")
                    return False
                # 3. |ΔE_P| should increase with q (linear)
                if not (abs(perturbed_delta_E_P) > abs(self.results['delta_E_P'])):
                    self._log(f"FAIL: Monotonicity - |ΔE_P| should increase with q. Original |ΔE_P|={abs(self.results['delta_E_P']):.2e}, perturbed |ΔE_P|={abs(perturbed_delta_E_P):.2e}")
                    return False
                # 4. U_AB should remain constant (independent of q for uniform field)
                if not math.isclose(perturbed_U_AB, self.results['U_AB'], rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: Monotonicity - U_AB should be independent of q. Original U_AB={self.results['U_AB']:.2e}, perturbed U_AB={perturbed_U_AB:.2e}")
                    return False
            
            self._log("PASS: Monotonicity checks verified.")
            return True
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            if not test_func():
                success = False
                self._log(f"{test_name} failed.")
            else:
                self._log(f"{test_name} passed.")
        
        return success, self.logs