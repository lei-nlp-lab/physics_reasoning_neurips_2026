import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.logs = []
        self.solution_instance = solution_instance
        
        # Extract inputs with safe fallbacks
        self.m = self._safe_get('m')
        self.g = self._safe_get('g')
        self.L = self._safe_get('L')
        self.theta = self._safe_get('theta')
        self.F = self._safe_get('F')
        self.s = self._safe_get('s')
        
        # Extract results with safe fallbacks
        self.mu = self._safe_get('mu')
        self.delta_E = self._safe_get('delta_E')
        self.W_f = self._safe_get('W_f')
        
        # Store original values for perturbation tests
        self.original_inputs = {
            'm': self.m,
            'g': self.g,
            'L': self.L,
            'theta': self.theta,
            'F': self.F,
            's': self.s
        }
        
        # Validate that all required values are present
        required = [self.m, self.g, self.L, self.theta, self.F, self.s, 
                   self.mu, self.delta_E, self.W_f]
        if any(v is None for v in required):
            self.logs.append("WARNING: Some required inputs or results are missing.")
    
    def _safe_get(self, key):
        """Safe attribute access with fallback to results dict."""
        value = getattr(self.solution_instance, key, None)
        if value is None:
            value = getattr(self.solution_instance, 'results', {}).get(key, None)
        return value
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """
        Verify physical invariants:
        1. Energy loss matches potential energy difference: ΔE = mgL cosθ
        2. Work done by friction equals negative energy loss: W_f = -ΔE
        3. Work done by friction matches definition: W_f = -μ F s
        """
        logs = []
        success = True
        
        try:
            # 1. Energy loss invariant
            expected_delta_E = self.m * self.g * self.L * math.cos(self.theta)
            if not self._isclose(self.delta_E, expected_delta_E):
                logs.append(f"FAIL: Energy loss mismatch. Got {self.delta_E}, expected {expected_delta_E}")
                success = False
            else:
                logs.append("PASS: Energy loss matches potential energy difference.")
            
            # 2. Work-energy theorem invariant
            if not self._isclose(self.W_f, -self.delta_E):
                logs.append(f"FAIL: Work-energy theorem mismatch. W_f = {self.W_f}, -ΔE = {-self.delta_E}")
                success = False
            else:
                logs.append("PASS: Work done by friction equals negative energy loss.")
            
            # 3. Friction work definition invariant
            expected_W_f = -self.mu * self.F * self.s
            if not self._isclose(self.W_f, expected_W_f):
                logs.append(f"FAIL: Friction work definition mismatch. Got {self.W_f}, expected {expected_W_f}")
                success = False
            else:
                logs.append("PASS: Friction work matches definition W_f = -μ F s.")
        
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def test_trace_consistency(self):
        """
        Check consistency between intermediate results and final answer.
        From steps analysis: μ = ΔE / (F s) = -W_f / (F s)
        """
        logs = []
        success = True
        
        try:
            # Method 1: μ from energy loss
            mu_from_delta_E = self.delta_E / (self.F * self.s)
            if not self._isclose(self.mu, mu_from_delta_E):
                logs.append(f"FAIL: μ mismatch from ΔE. Got {self.mu}, expected {mu_from_delta_E}")
                success = False
            else:
                logs.append("PASS: μ consistent with ΔE/(F s).")
            
            # Method 2: μ from friction work
            mu_from_W_f = -self.W_f / (self.F * self.s)
            if not self._isclose(self.mu, mu_from_W_f):
                logs.append(f"FAIL: μ mismatch from W_f. Got {self.mu}, expected {mu_from_W_f}")
                success = False
            else:
                logs.append("PASS: μ consistent with -W_f/(F s).")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def test_monotonicity(self):
        """
        Perturb mass m and verify that μ changes in the expected direction.
        μ = (m g L cosθ) / (F s) is directly proportional to m.
        """
        logs = []
        success = True
        
        try:
            # Original μ
            mu_original = self.mu
            
            # Perturb m by +1%
            m_perturbed = self.m * 1.01
            mu_perturbed_expected = (m_perturbed * self.g * self.L * math.cos(self.theta)) / (self.F * self.s)
            
            # Check direction of change
            if mu_perturbed_expected > mu_original:
                logs.append("PASS: μ increases with increased m (as expected).")
            else:
                logs.append("FAIL: μ does not increase with increased m.")
                success = False
            
            # Restore original value
            self.m = self.original_inputs['m']
        
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} Test ---")
            if not test_func():
                success = False
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs