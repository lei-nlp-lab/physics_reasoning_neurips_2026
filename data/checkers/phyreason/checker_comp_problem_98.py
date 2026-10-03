import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        """
        self.logs = []
        try:
            # Extract inputs safely using getattr only
            self.m = getattr(solution_instance, 'm', 1.0)
            self.v = getattr(solution_instance, 'v', 1.0)
            self.theta = getattr(solution_instance, 'theta', 0.0)
            self.mu = getattr(solution_instance, 'mu', 0.0)
            
            # Extract results safely from results dict
            results = getattr(solution_instance, 'results', {})
            self.p_n = results.get('p_n', 0.0)
            self.u = results.get('u', 0.0)
            self.p_f = results.get('p_f', 0.0)
            
            # Store for restoration in monotonicity test
            self._original_mu = self.mu
            
        except Exception as e:
            self.logs.append(f"Initialization error: {str(e)}")
            raise
    
    def test_invariants(self):
        """
        Verify fundamental physical laws:
        1. Normal impulse p_n = m*v*cos(theta)
        2. Friction impulse p_f = mu * p_n
        3. Momentum conservation along ramp: m*v*sin(theta) - p_f = m*u
        """
        logs = []
        try:
            # 1. Normal impulse invariant
            expected_p_n = self.m * self.v * math.cos(self.theta)
            if math.isclose(self.p_n, expected_p_n, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Normal impulse invariant: p_n = {self.p_n:.4f}, expected {expected_p_n:.4f}")
            else:
                logs.append(f"✗ Normal impulse invariant failed: p_n = {self.p_n:.4f}, expected {expected_p_n:.4f}")
                return False, logs
            
            # 2. Friction impulse invariant
            expected_p_f = self.mu * self.p_n
            if math.isclose(self.p_f, expected_p_f, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Friction impulse invariant: p_f = {self.p_f:.4f}, expected {expected_p_f:.4f}")
            else:
                logs.append(f"✗ Friction impulse invariant failed: p_f = {self.p_f:.4f}, expected {expected_p_f:.4f}")
                return False, logs
            
            # 3. Momentum conservation along ramp
            lhs = self.m * self.v * math.sin(self.theta) - self.p_f
            rhs = self.m * self.u
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Momentum conservation: {lhs:.4f} = {rhs:.4f}")
            else:
                logs.append(f"✗ Momentum conservation failed: {lhs:.4f} ≠ {rhs:.4f}")
                return False, logs
            
            return True, logs
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            return False, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        From steps: u = v*(sin(theta) - mu*cos(theta))
        """
        logs = []
        try:
            # Direct formula check
            expected_u = self.v * (math.sin(self.theta) - self.mu * math.cos(self.theta))
            
            if math.isclose(self.u, expected_u, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Trace consistency: u = {self.u:.4f}, expected {expected_u:.4f}")
            else:
                logs.append(f"✗ Trace consistency failed: u = {self.u:.4f}, expected {expected_u:.4f}")
                return False, logs
            
            # Check via p_n and p_f relationships
            u_from_p = self.v * math.sin(self.theta) - self.p_f / self.m
            if math.isclose(self.u, u_from_p, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Intermediate consistency: u = {self.u:.4f}, from p_f gives {u_from_p:.4f}")
            else:
                logs.append(f"✗ Intermediate consistency failed: u = {self.u:.4f}, from p_f gives {u_from_p:.4f}")
                return False, logs
            
            return True, logs
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            return False, logs
    
    def test_monotonicity(self):
        """
        Perturb mu and verify that u decreases when mu increases (since friction opposes motion).
        Restore original mu after test.
        """
        logs = []
        try:
            # Perturb mu by +10%
            perturbed_mu = self.mu * 1.1
            
            # Recalculate u using the derived formula
            perturbed_u = self.v * (math.sin(self.theta) - perturbed_mu * math.cos(self.theta))
            
            # Check monotonic relationship: if mu increases, u should decrease (for positive cos(theta))
            if math.cos(self.theta) > 0:
                if perturbed_u < self.u:
                    logs.append(f"✓ Monotonicity: u decreases from {self.u:.4f} to {perturbed_u:.4f} when mu increases from {self.mu:.4f} to {perturbed_mu:.4f}")
                else:
                    logs.append(f"✗ Monotonicity failed: u should decrease but changed from {self.u:.4f} to {perturbed_u:.4f}")
                    return False, logs
            else:
                logs.append("Note: cos(theta) ≤ 0, monotonicity test skipped (unphysical ramp angle)")
            
            # Restore original mu
            self.mu = self._original_mu
            return True, logs
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            # Restore original mu on error
            self.mu = self._original_mu
            return False, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        success = True
        
        # Test 1: Invariants
        test_success, logs = self.test_invariants()
        all_logs.extend(logs)
        if not test_success:
            success = False
        
        # Test 2: Trace consistency
        test_success, logs = self.test_trace_consistency()
        all_logs.extend(logs)
        if not test_success:
            success = False
        
        # Test 3: Monotonicity
        test_success, logs = self.test_monotonicity()
        all_logs.extend(logs)
        if not test_success:
            success = False
        
        return success, all_logs