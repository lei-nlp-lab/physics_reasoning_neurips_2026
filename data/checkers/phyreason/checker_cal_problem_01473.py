import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs (with fallback to problem defaults)
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'results', {}).get('v0', 1.5))
        self.v1 = getattr(solution_instance, 'v1', getattr(solution_instance, 'results', {}).get('v1', -1.3))
        self.delta_t = getattr(solution_instance, 'delta_t', getattr(solution_instance, 'results', {}).get('delta_t', 0.7))
        
        # Safe access for results (with fallback to defaults)
        self.delta_v = getattr(solution_instance, 'delta_v', getattr(solution_instance, 'results', {}).get('delta_v', 0.0))
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'results', {}).get('a', 0.0))
        
        # Constants for validation
        self.expected_delta_v = -2.8
        self.expected_a = -4.0
        
        # Storage for logs
        self.logs = []
        self.all_passed = True

    def _log(self, message, passed):
        """Helper to log test results"""
        status = "PASS" if passed else "FAIL"
        self.logs.append(f"[{status}] {message}")
        if not passed:
            self.all_passed = False
        return passed

    def test_invariants(self):
        """Test physical invariants (definitions of Δv and a)"""
        try:
            # 1. Check Δv = v1 - v0
            computed_delta_v = self.v1 - self.v0
            passed_delta_v = math.isclose(self.delta_v, computed_delta_v, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Δv definition: {self.delta_v:.4f} ≈ {computed_delta_v:.4f} (v1 - v0)", passed_delta_v)
            
            # 2. Check a = Δv / Δt
            if abs(self.delta_t) > 1e-12:  # Avoid division by zero
                computed_a = computed_delta_v / self.delta_t
                passed_a = math.isclose(self.a, computed_a, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Acceleration definition: {self.a:.4f} ≈ {computed_a:.4f} (Δv/Δt)", passed_a)
            else:
                self._log("Δt is zero or too small - skipping acceleration test", False)
                
            # 3. Check trend: if a is negative, velocity decreases (becomes more negative)
            if passed_a:
                if self.a > 0 and not math.isclose(self.a, 0.0, abs_tol=1e-2):
                    passed_trend = self.v1 > self.v0
                    self._log(f"Positive a: velocity increased from {self.v0:.2f} to {self.v1:.2f}", passed_trend)
                elif self.a < 0 and not math.isclose(self.a, 0.0, abs_tol=1e-2):
                    passed_trend = self.v1 < self.v0
                    self._log(f"Negative a: velocity decreased from {self.v0:.2f} to {self.v1:.2f}", passed_trend)
                else:
                    passed_trend = math.isclose(self.v1, self.v0, rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"Zero a: velocity constant at {self.v0:.2f}", passed_trend)
            
            return all([passed_delta_v, passed_a])
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with expected values"""
        try:
            # Check Δv against expected from steps analysis
            passed_delta_v = math.isclose(self.delta_v, self.expected_delta_v, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Δv consistency: {self.delta_v:.4f} ≈ {self.expected_delta_v:.4f}", passed_delta_v)
            
            # Check a against expected from steps analysis
            passed_a = math.isclose(self.a, self.expected_a, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Acceleration consistency: {self.a:.4f} ≈ {self.expected_a:.4f}", passed_a)
            
            # Verify the chain: a should equal Δv/Δt exactly
            if abs(self.delta_t) > 1e-12:
                expected_from_chain = self.expected_delta_v / self.delta_t
                passed_chain = math.isclose(self.a, expected_from_chain, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Chain consistency: a = Δv/Δt = {self.expected_delta_v:.2f}/{self.delta_t:.2f} ≈ {expected_from_chain:.4f}", 
                         passed_chain)
            else:
                passed_chain = False
                self._log("Δt too small for chain test", False)
            
            return all([passed_delta_v, passed_a, passed_chain])
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            return False

    def test_monotonicity(self):
        """Perturb an input and check trend"""
        try:
            # Save original values
            original_v0 = self.v0
            original_v1 = self.v1
            original_delta_t = self.delta_t
            
            # Test 1: Increase contact time by 10% -> magnitude of acceleration should decrease
            perturbed_delta_t = original_delta_t * 1.10
            perturbed_a = (original_v1 - original_v0) / perturbed_delta_t
            
            # |a| should decrease when Δt increases (since Δv is constant)
            passed_time_perturb = abs(perturbed_a) < abs(self.a) * 1.05  # Allow 5% tolerance for magnitude
            self._log(f"Perturb Δt from {original_delta_t:.2f} to {perturbed_delta_t:.2f}: " +
                     f"|a| decreases from {abs(self.a):.4f} to {abs(perturbed_a):.4f}", 
                     passed_time_perturb)
            
            # Test 2: Increase initial speed by 10% -> Δv magnitude should increase (more negative)
            perturbed_v0 = original_v0 * 1.10
            perturbed_delta_v = original_v1 - perturbed_v0
            passed_speed_perturb = abs(perturbed_delta_v) > abs(self.expected_delta_v) * 1.05  # 5% tolerance
            self._log(f"Perturb v0 from {original_v0:.2f} to {perturbed_v0:.2f}: " +
                     f"|Δv| increases from {abs(self.expected_delta_v):.4f} to {abs(perturbed_delta_v):.4f}",
                     passed_speed_perturb)
            
            # Values are restored automatically (we never modified instance attributes)
            
            return all([passed_time_perturb, passed_speed_perturb])
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            return False

    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)"""
        self.logs = []
        self.all_passed = True
        
        test_methods = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_methods:
            try:
                self._log(f"--- Starting {test_name} ---", True)
                test_func()
            except Exception as e:
                self._log(f"CRASH in {test_name}: {traceback.format_exc()}", False)
        
        # Summary
        if self.all_passed:
            self._log("=== All tests PASSED ===", True)
        else:
            self._log("=== Some tests FAILED ===", False)
        
        return self.all_passed, self.logs

# Optional: Add a symbolic verification method for completeness
def symbolic_verification():
    """Symbolic verification of the physics formulas"""
    v0, v1, delta_t = sp.symbols('v0 v1 delta_t', real=True)
    delta_v = v1 - v0
    a_avg = delta_v / delta_t
    
    print("Symbolic formulas:")
    print(f"Δv = {delta_v}")
    print(f"a_avg = {a_avg}")
    
    # Numerical substitution
    numeric_delta_v = delta_v.subs({v0: 1.5, v1: -1.3})
    numeric_a = a_avg.subs({v0: 1.5, v1: -1.3, delta_t: 0.7})
    
    print(f"\nNumerical verification:")
    print(f"Δv = {float(numeric_delta_v):.2f}")
    print(f"a_avg = {float(numeric_a):.2f}")
    
    return float(numeric_delta_v), float(numeric_a)

# For standalone testing
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'delta_v': -2.8,
                'a': -4.0,
                'v0': 1.5,
                'v1': -1.3,
                'delta_t': 0.7
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    print("\n".join(logs))
    print(f"\nOverall success: {success}")
    
    # Run symbolic verification
    print("\n" + "="*50)
    symbolic_verification()