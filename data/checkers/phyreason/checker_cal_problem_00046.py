import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all inputs and results from the solution instance.
        Uses getattr with fallbacks to avoid AttributeError in sandbox.
        """
        # Input parameters
        self.l1 = getattr(solution_instance, 'l1', 4.0)
        self.mu = getattr(solution_instance, 'mu', 2.0/9.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin24 = getattr(solution_instance, 'sin24', 0.4)
        self.cos24 = getattr(solution_instance, 'cos24', 0.9)

        # Results - extract from results dict or attributes
        results = getattr(solution_instance, 'results', {})
        self.a1 = getattr(solution_instance, 'a_1', results.get('a_1', None))
        self.v = getattr(solution_instance, 'v', results.get('v', None))
        self.a2 = getattr(solution_instance, 'a_2', results.get('a_2', None))
        self.l2 = getattr(solution_instance, 'l_2', results.get('l_2', None))

        # Store original values for restoration
        self._original_mu = self.mu
        self._original_results = {
            'a1': self.a1,
            'v': self.v,
            'a2': self.a2,
            'l2': self.l2
        }

        # Tolerance for numerical comparisons
        self.rtol = 1e-2
        self.atol = 1e-2

    def test_invariants(self):
        """
        Verify Newton's 2nd law on both chute segments.
        Uses numerical calculations only.
        """
        logs = []
        try:
            # Check incline acceleration: a1 = g*sin24 - μ*g*cos24
            expected_a1 = self.g * self.sin24 - self.mu * self.g * self.cos24
            if self.a1 is not None and math.isclose(self.a1, expected_a1, rel_tol=self.rtol, abs_tol=self.atol):
                logs.append(f"✓ Incline Newton's 2nd law: a1 = {self.a1:.3f}, expected = {expected_a1:.3f}")
            elif self.a1 is None:
                logs.append("✗ Incline acceleration a1 not found")
            else:
                logs.append(f"✗ Incline Newton's 2nd law failed: a1 = {self.a1:.3f}, expected = {expected_a1:.3f}")
                logs.append(f"  Equation: g sinθ - μ g cosθ = a1")
            
            # Check horizontal acceleration: a2 = -μ*g (deceleration, negative)
            expected_a2 = -self.mu * self.g
            if self.a2 is not None and math.isclose(self.a2, expected_a2, rel_tol=self.rtol, abs_tol=self.atol):
                logs.append(f"✓ Horizontal Newton's 2nd law: a2 = {self.a2:.3f}, expected = {expected_a2:.3f}")
            elif self.a2 is None:
                logs.append("✗ Horizontal acceleration a2 not found")
            else:
                logs.append(f"✗ Horizontal Newton's 2nd law failed: a2 = {self.a2:.3f}, expected = {expected_a2:.3f}")
                logs.append(f"  Equation: -μ g = a2")
                
        except Exception as e:
            logs.append(f"✗ Invariant test failed with error: {e}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with kinematic equations.
        """
        logs = []
        try:
            # Check v^2 = 2*a1*l1
            if self.a1 is not None and self.v is not None:
                expected_v = math.sqrt(2 * self.a1 * self.l1)
                if math.isclose(self.v, expected_v, rel_tol=self.rtol, abs_tol=self.atol):
                    logs.append(f"✓ Kinematic consistency: v = {self.v:.3f}, expected = {expected_v:.3f}")
                else:
                    logs.append(f"✗ Kinematic consistency failed: v = {self.v:.3f}, expected = {expected_v:.3f}")
                    logs.append(f"  Equation: v² = 2*a1*l1")
            elif self.a1 is None or self.v is None:
                logs.append("✗ Missing a1 or v for kinematic check")
            
            # Check l2 from 0 = v^2 + 2*a2*l2 (since a2 is negative)
            if self.a2 is not None and self.l2 is not None and self.v is not None:
                # Using 0 = v² + 2*a2*l2 => l2 = -v²/(2*a2)
                expected_l2 = -(self.v**2) / (2 * self.a2)
                if math.isclose(self.l2, expected_l2, rel_tol=self.rtol, abs_tol=self.atol):
                    logs.append(f"✓ Horizontal length consistency: l2 = {self.l2:.3f}, expected = {expected_l2:.3f}")
                else:
                    logs.append(f"✗ Horizontal length consistency failed: l2 = {self.l2:.3f}, expected = {expected_l2:.3f}")
                    logs.append(f"  Equation: 0 = v² + 2*a2*l2")
            elif self.l2 is None:
                logs.append("✗ Missing l2 for horizontal kinematic check")
                
        except Exception as e:
            logs.append(f"✗ Trace consistency test failed with error: {e}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_monotonicity(self):
        """
        Perturb friction coefficient μ and check expected trend.
        Restores original values after test.
        """
        logs = []
        original_mu = self.mu
        original_a1 = self.a1
        
        try:
            # Perturb μ upward
            perturbed_mu = self.mu * 1.05  # +5%
            
            # Recompute a1 with perturbed μ (using physical formula)
            # a1 = g sinθ - μ g cosθ
            perturbed_a1 = self.g * self.sin24 - perturbed_mu * self.g * self.cos24
            
            # Expected trend: increasing μ should decrease a1
            if perturbed_a1 < original_a1:
                logs.append(f"✓ Monotonicity: Increasing μ from {original_mu:.3f} to {perturbed_mu:.3f}")
                logs.append(f"  decreases a1 from {original_a1:.3f} to {perturbed_a1:.3f}")
            else:
                logs.append(f"✗ Monotonicity violated: Increasing μ should decrease a1")
                logs.append(f"  Original: μ={original_mu:.3f}, a1={original_a1:.3f}")
                logs.append(f"  Perturbed: μ={perturbed_mu:.3f}, a1={perturbed_a1:.3f}")
            
            # Restore original μ
            self.mu = original_mu
            
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed with error: {e}")
            logs.append(traceback.format_exc())
            # Ensure restoration even on error
            self.mu = original_mu
        
        return logs

    def run_all_tests(self):
        """
        Execute all physics verification tests.
        Returns (success_bool, logs_list).
        """
        all_logs = []
        
        # Run tests
        all_logs.append("=== Physics Invariants Test ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("✗"):
                success = False
                break
        
        return success, all_logs