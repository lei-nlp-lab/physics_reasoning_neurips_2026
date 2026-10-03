import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize checker with safe attribute access."""
        try:
            # Extract inputs with fallbacks (problem context)
            self.q = getattr(solution_instance, 'q', 300.0)  # Charge in Coulombs
            self.t = getattr(solution_instance, 't', 0.005)  # Time in seconds
            
            # Extract result with fallback to results dictionary
            self.I = getattr(solution_instance, 'I', 
                           getattr(solution_instance, 'results', {}).get('I', 0.0))
            
            # Expected target (from problem statement)
            self.target_I = 60000.0  # 300 / 0.005 = 60000 A
            
            self.logs = []
            self.success = True
            
        except Exception as e:
            self.logs.append(f"Initialization error: {str(e)}")
            self.success = False

    def test_invariants(self):
        """Test definition of electric current: I = q/t."""
        try:
            # Symbolic check
            I_sym, q_sym, t_sym = sp.symbols('I q t')
            eq = sp.Eq(I_sym, q_sym / t_sym)
            
            # Numerical check
            computed_I = self.q / self.t
            is_correct = math.isclose(self.I, computed_I, rel_tol=1e-2, abs_tol=1e-2)
            
            if is_correct:
                self.logs.append("✓ Current definition invariant passed: I = q/t")
            else:
                self.logs.append(f"✗ Current definition invariant failed. Expected {computed_I:.1f}A, got {self.I:.1f}A")
                self.success = False
                
            return is_correct
            
        except Exception as e:
            self.logs.append(f"Invariant test error: {str(e)}")
            self.success = False
            return False

    def test_trace_consistency(self):
        """Check intermediate steps against final answer."""
        try:
            # Check against target answer
            is_consistent = math.isclose(self.I, self.target_I, rel_tol=1e-2, abs_tol=1e-2)
            
            if is_consistent:
                self.logs.append("✓ Trace consistency passed: matches target 60000A")
            else:
                self.logs.append(f"✗ Trace consistency failed. Expected {self.target_I:.1f}A, got {self.I:.1f}A")
                self.success = False
                
            return is_consistent
            
        except Exception as e:
            self.logs.append(f"Trace consistency error: {str(e)}")
            self.success = False
            return False

    def test_monotonicity(self):
        """Test monotonicity by perturbing charge and time."""
        original_I = self.I
        original_q = self.q
        original_t = self.t
        
        try:
            # Test 1: Increase charge by 10%, current should increase proportionally
            self.q *= 1.1
            new_I_q = self.q / self.t
            trend_q_correct = (new_I_q > original_I)
            
            # Restore original q
            self.q = original_q
            
            # Test 2: Increase time by 10%, current should decrease
            self.t *= 1.1
            new_I_t = self.q / self.t
            trend_t_correct = (new_I_t < original_I)
            
            # Restore original t
            self.t = original_t
            
            if trend_q_correct and trend_t_correct:
                self.logs.append("✓ Monotonicity test passed: I ∝ q and I ∝ 1/t")
            else:
                if not trend_q_correct:
                    self.logs.append("✗ Monotonicity failed: Current didn't increase with charge")
                if not trend_t_correct:
                    self.logs.append("✗ Monotonicity failed: Current didn't decrease with time")
                self.success = False
                
            return trend_q_correct and trend_t_correct
            
        except Exception as e:
            self.logs.append(f"Monotonicity test error: {str(e)}")
            # Restore values in case of error
            self.q = original_q
            self.t = original_t
            self.success = False
            return False

    def run_all_tests(self):
        """Execute all tests and return final status."""
        if not self.success:
            return False, self.logs
            
        try:
            # Run tests in order
            tests = [
                self.test_invariants,
                self.test_trace_consistency,
                self.test_monotonicity
            ]
            
            for test in tests:
                if not self.success:  # Early exit if any test fails
                    break
                test()
            
            # Final validation
            if self.success:
                self.logs.append("="*50)
                self.logs.append("✓ ALL PHYSICS CHECKS PASSED")
                self.logs.append(f"  Final current: {self.I:.1f}A")
                self.logs.append(f"  Expected range: 59400-60600A (1% tolerance)")
            else:
                self.logs.append("="*50)
                self.logs.append("✗ PHYSICS CHECKS FAILED")
            
            return self.success, self.logs
            
        except Exception as e:
            error_msg = f"Runtime error in test suite: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs