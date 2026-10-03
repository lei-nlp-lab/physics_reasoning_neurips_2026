import sympy as sp
import math
import copy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs (with defaults from problem context)
        self.F = getattr(solution_instance, 'F', self.results.get('F', 900.0))
        self.delta_t = getattr(solution_instance, 'delta_t', self.results.get('delta_t', 3.0))
        self.delta_v = getattr(solution_instance, 'delta_v', self.results.get('delta_v', 0.9))
        
        # Extract results
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.m = getattr(solution_instance, 'm', self.results.get('m', 0.0))
        
        # Tolerances for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for test results
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log test results"""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
        return success
    
    def test_invariants(self):
        """Test fundamental physics invariants"""
        success = True
        
        # 1. Check Newton's Second Law (F = m*a)
        try:
            # Symbolic check
            F, m, a = sp.symbols('F m a')
            newton_eq = sp.Eq(F, m * a)
            
            # Numerical check
            computed_F = self.m * self.a
            if not math.isclose(self.F, computed_F, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = self._log(
                    f"Newton's Second Law violated: F={self.F:.3f}, m*a={computed_F:.3f}", 
                    False
                ) and success
            else:
                self._log(f"Newton's Second Law satisfied: F = m*a", True)
        except Exception as e:
            self._log(f"Error in Newton's Second Law check: {e}", False)
            success = False
        
        # 2. Check kinematic relationship (a = Δv/Δt)
        try:
            computed_a = self.delta_v / self.delta_t
            if not math.isclose(self.a, computed_a, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = self._log(
                    f"Kinematics violated: a={self.a:.3f}, Δv/Δt={computed_a:.3f}", 
                    False
                ) and success
            else:
                self._log(f"Kinematics satisfied: a = Δv/Δt", True)
        except ZeroDivisionError:
            self._log("Time interval is zero", False)
            success = False
        except Exception as e:
            self._log(f"Error in kinematics check: {e}", False)
            success = False
            
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results from steps analysis"""
        success = True
        
        # Expected values from steps analysis
        expected_a = 0.3
        expected_m = 3000.0
        
        # Check acceleration
        if not math.isclose(self.a, expected_a, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            success = self._log(
                f"Acceleration mismatch: computed={self.a:.3f}, expected={expected_a:.3f}", 
                False
            ) and success
        else:
            self._log(f"Acceleration consistent with step 1: a={self.a:.3f} m/s²", True)
        
        # Check mass
        if not math.isclose(self.m, expected_m, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            success = self._log(
                f"Mass mismatch: computed={self.m:.3f}, expected={expected_m:.3f}", 
                False
            ) and success
        else:
            self._log(f"Mass consistent with step 2: m={self.m:.3f} kg", True)
        
        # Check that using steps in sequence gives same result
        try:
            a_from_steps = self.delta_v / self.delta_t
            m_from_steps = self.F / a_from_steps
            
            if not math.isclose(self.a, a_from_steps, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = self._log(
                    f"Step sequence error in acceleration", 
                    False
                ) and success
            if not math.isclose(self.m, m_from_steps, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = self._log(
                    f"Step sequence error in mass", 
                    False
                ) and success
        except Exception as e:
            self._log(f"Error in step sequence check: {e}", False)
            success = False
            
        return success
    
    def test_monotonicity(self):
        """Test how outputs change with perturbed inputs"""
        success = True
        
        # Store original state
        original_state = {
            'F': self.F,
            'delta_v': self.delta_v,
            'a': self.a,
            'm': self.m
        }
        
        try:
            # Perturbation 1: Increase force by 2%
            perturbed_F = self.F * 1.02
            # Recalculate using physics formulas
            a_const = self.delta_v / self.delta_t  # a should remain constant
            m_new = perturbed_F / a_const
            
            # Mass should increase with increased force (for constant acceleration)
            if not m_new > self.m:
                success = self._log(
                    f"Monotonicity violated: m did not increase when F increased", 
                    False
                ) and success
            else:
                self._log(
                    f"Monotonicity: m increases with F (ΔF=+2% → Δm≈{100*(m_new/self.m-1):+.1f}%)", 
                    True
                )
            
            # Restore
            self.F = original_state['F']
            
            # Perturbation 2: Increase Δv by 2%
            perturbed_delta_v = self.delta_v * 1.02
            a_new = perturbed_delta_v / self.delta_t
            m_new = self.F / a_new
            
            # Acceleration should increase with increased Δv
            if not a_new > self.a:
                success = self._log(
                    f"Monotonicity violated: a did not increase when Δv increased", 
                    False
                ) and success
            else:
                self._log(
                    f"Monotonicity: a increases with Δv", 
                    True
                )
            
            # Mass should decrease with increased acceleration (for constant force)
            if not m_new < self.m:
                success = self._log(
                    f"Monotonicity violated: m did not decrease when a increased", 
                    False
                ) and success
            else:
                self._log(
                    f"Monotonicity: m decreases with a (Δa=+2% → Δm≈{100*(m_new/self.m-1):+.1f}%)", 
                    True
                )
                
        except Exception as e:
            self._log(f"Error in monotonicity test: {e}", False)
            success = False
        finally:
            # Restore original state
            for key, value in original_state.items():
                setattr(self, key, value)
            
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs"""
        self.logs = []
        
        # Add header
        self._log("=" * 50)
        self._log("Physics Auditor Report")
        self._log("=" * 50)
        
        # Run tests
        test_results = []
        
        self._log("Test 1: Physics Invariants")
        test_results.append(self.test_invariants())
        
        self._log("\nTest 2: Trace Consistency")
        test_results.append(self.test_trace_consistency())
        
        self._log("\nTest 3: Monotonicity")
        test_results.append(self.test_monotonicity())
        
        # Summary
        self._log("\n" + "=" * 50)
        success = all(test_results)
        if success:
            self._log("ALL TESTS PASSED ✓", True)
        else:
            self._log("SOME TESTS FAILED ✗", False)
        self._log("=" * 50)
        
        return success, self.logs

# Example usage (commented out):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.results = {'a': 0.3, 'm': 3000.0}
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)