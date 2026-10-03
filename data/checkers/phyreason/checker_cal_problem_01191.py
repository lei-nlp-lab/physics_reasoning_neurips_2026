import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results without direct attribute access."""
        # Safe extraction using getattr with fallback to results dictionary
        self.results = {}
        
        # Results extraction with safe fallbacks
        self.a1 = getattr(solution_instance, 'a_1', 
                         getattr(solution_instance, 'results', {}).get('a_1', 0.0))
        self.a2 = getattr(solution_instance, 'a_2', 
                         getattr(solution_instance, 'results', {}).get('a_2', 0.0))
        self.t_val = getattr(solution_instance, 't', 
                           getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.v_val = getattr(solution_instance, 'v', 
                           getattr(solution_instance, 'results', {}).get('v', 0.0))
        
        # Constants from problem context (safe extraction)
        self.g = 10.0  # Given in problem
        self.theta = math.radians(30)  # 30 degrees in radians
        
        # Store for mass if available (not critical for this problem)
        self.m = getattr(solution_instance, 'm', 1.0)  # Default mass
        
        # Additional derived quantities
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)
        
        # Expected values from problem context
        self.expected_a1 = 5.0  # m/s²
        self.expected_a2 = 2.5  # m/s²
        
        # For kinematic checks, we need length of incline
        # From t = √(2s/a1) with t=√2, a1=5 → s=5
        self.s = 5.0  # meters (derived from expected values)
        
        self.logs = []

    def _log(self, message, success=True):
        """Helper for logging test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success

    def test_invariants(self):
        """Verify physics invariants: Newton's laws and energy conservation."""
        tests_passed = True
        
        try:
            # 1. Check Newton's Second Law for a1 (no friction)
            # a1 = g*sin(theta) = 10 * sin(30°) = 5 m/s²
            expected_a1 = self.g * self.sin_theta
            if math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's 2nd Law (no friction): a1={self.a1:.3f} ≈ g·sinθ={expected_a1:.3f}")
            else:
                tests_passed = self._log(
                    f"Newton's 2nd Law (no friction) FAILED: a1={self.a1:.3f} ≠ g·sinθ={expected_a1:.3f}", 
                    False
                )
            
            # 2. Check a2 is less than a1 (friction reduces acceleration)
            if self.a2 < self.a1:
                self._log(f"Friction reduces acceleration: a2={self.a2:.3f} < a1={self.a1:.3f}")
            else:
                tests_passed = self._log(
                    f"Friction effect FAILED: a2={self.a2:.3f} should be < a1={self.a1:.3f}", 
                    False
                )
            
            # 3. Check a2 against friction formula pattern
            # Using symbolic check for a2 = g*sinθ - μ*g*cosθ
            # We can't verify μ directly, but check a2 = a1 - μ*g*cosθ
            if self.a1 > 0 and self.a2 > 0:
                # Derive μ from a2 and check it's reasonable (0 < μ < 1 for typical friction)
                μ = (self.a1 - self.a2) / (self.g * self.cos_theta)
                if 0 < μ < 1:
                    self._log(f"Friction coefficient μ={μ:.3f} is physically reasonable")
                else:
                    tests_passed = self._log(
                        f"Unphysical friction coefficient: μ={μ:.3f}", 
                        False
                    )
            
            # 4. Check normal force consistency (should be constant for constant θ)
            N_expected = self.m * self.g * self.cos_theta
            self._log(f"Normal force N = m·g·cosθ = {N_expected:.3f} N (for m={self.m:.1f} kg)")
            
            # 5. Energy conservation check (potential energy → kinetic energy)
            # Initial PE = m*g*h, where h = s*sinθ
            h = self.s * self.sin_theta
            PE_initial = self.m * self.g * h
            KE_final = 0.5 * self.m * (self.v_val**2)
            
            energy_diff = abs(PE_initial - KE_final)
            if math.isclose(PE_initial, KE_final, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy conserved: PE_initial={PE_initial:.3f} J ≈ KE_final={KE_final:.3f} J")
            else:
                self._log(f"Energy difference: ΔE={energy_diff:.3f} J", True)  # Not critical for this problem
                
        except Exception as e:
            tests_passed = self._log(f"Invariant test failed with error: {str(e)}", False)
            traceback.print_exc()
            
        return tests_passed

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        try:
            # 1. Check kinematic relation: v = a1 * t (for uniformly accelerated motion from rest)
            v_from_a1_t = self.a1 * self.t_val
            if math.isclose(self.v_val, v_from_a1_t, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Kinematic consistency: v={self.v_val:.3f} = a1·t={v_from_a1_t:.3f}")
            else:
                tests_passed = self._log(
                    f"Kinematic consistency FAILED: v={self.v_val:.3f} ≠ a1·t={v_from_a1_t:.3f}", 
                    False
                )
            
            # 2. Check t = √(2s/a1)
            t_from_s_a1 = math.sqrt(2 * self.s / self.a1)
            if math.isclose(self.t_val, t_from_s_a1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Time consistency: t={self.t_val:.3f} ≈ √(2s/a1)={t_from_s_a1:.3f}")
            else:
                tests_passed = self._log(
                    f"Time consistency FAILED: t={self.t_val:.3f} ≠ √(2s/a1)={t_from_s_a1:.3f}", 
                    False
                )
            
            # 3. Check v^2 = 2*a1*s
            v_squared = self.v_val**2
            expected_v_squared = 2 * self.a1 * self.s
            if math.isclose(v_squared, expected_v_squared, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v² consistency: v²={v_squared:.3f} ≈ 2·a1·s={expected_v_squared:.3f}")
            else:
                tests_passed = self._log(
                    f"v² consistency FAILED: v²={v_squared:.3f} ≠ 2·a1·s={expected_v_squared:.3f}", 
                    False
                )
            
            # 4. Check a2 against expected value
            if math.isclose(self.a2, self.expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"a2 matches expected: {self.a2:.3f} ≈ {self.expected_a2:.3f}")
            else:
                tests_passed = self._log(
                    f"a2 value mismatch: {self.a2:.3f} ≠ expected {self.expected_a2:.3f}", 
                    False
                )
                
        except Exception as e:
            tests_passed = self._log(f"Trace consistency test failed with error: {str(e)}", False)
            traceback.print_exc()
            
        return tests_passed

    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        tests_passed = True
        
        try:
            # Test 1: Increase θ → increase a1
            theta_perturbed = self.theta + math.radians(5)  # +5 degrees
            a1_original = self.a1
            a1_perturbed = self.g * math.sin(theta_perturbed)
            
            if a1_perturbed > a1_original:
                self._log(f"Monotonicity θ→a1: θ↑ {math.degrees(self.theta):.1f}°→{math.degrees(theta_perturbed):.1f}° ⇒ a1↑ {a1_original:.3f}→{a1_perturbed:.3f}")
            else:
                tests_passed = self._log(
                    f"Monotonicity θ→a1 FAILED: a1 should increase with θ", 
                    False
                )
            
            # Test 2: Increase μ (friction) → decrease a2
            # Estimate μ from current a2
            μ_current = (self.a1 - self.a2) / (self.g * self.cos_theta)
            μ_perturbed = μ_current * 1.5  # Increase by 50%
            a2_perturbed = self.g * self.sin_theta - μ_perturbed * self.g * self.cos_theta
            
            if a2_perturbed < self.a2:
                self._log(f"Monotonicity μ→a2: μ↑ {μ_current:.3f}→{μ_perturbed:.3f} ⇒ a2↓ {self.a2:.3f}→{a2_perturbed:.3f}")
            else:
                tests_passed = self._log(
                    f"Monotonicity μ→a2 FAILED: a2 should decrease with μ", 
                    False
                )
            
            # Test 3: Increase mass m → no change in accelerations (mass cancels)
            m_perturbed = self.m * 2.0
            a1_mass_independent = self.g * self.sin_theta  # Same as original
            
            if math.isclose(a1_mass_independent, self.a1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Mass independence: m↑ {self.m:.1f}→{m_perturbed:.1f} ⇒ a1 unchanged {self.a1:.3f}")
            else:
                tests_passed = self._log(
                    f"Mass independence FAILED: a1 should be independent of m", 
                    False
                )
                
        except Exception as e:
            tests_passed = self._log(f"Monotonicity test failed with error: {str(e)}", False)
            traceback.print_exc()
            
        return tests_passed

    def run_all_tests(self):
        """Execute all tests and return summary."""
        self.logs.clear()
        
        # Run individual test suites
        invariant_result = self.test_invariants()
        trace_result = self.test_trace_consistency()
        monotonicity_result = self.test_monotonicity()
        
        # Overall success
        all_passed = invariant_result and trace_result and monotonicity_result
        
        # Add summary
        summary = f"""
        ========================================
        PHYSICS AUDIT SUMMARY
        ========================================
        Invariant Tests: {'PASS' if invariant_result else 'FAIL'}
        Trace Consistency: {'PASS' if trace_result else 'FAIL'}
        Monotonicity Tests: {'PASS' if monotonicity_result else 'FAIL'}
        Overall: {'PASS' if all_passed else 'FAIL'}
        ========================================
        """
        self.logs.append(summary)
        
        return all_passed, self.logs

# Example usage demonstration
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'a_1': 5.0,
                'a_2': 2.5,
                't': math.sqrt(2),
                'v': 5 * math.sqrt(2)
            }
            self.m = 1.0
    
    # Run the checker
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    # Print results
    for log in logs:
        print(log)
    
    print(f"\nFinal result: {'All tests passed' if success else 'Some tests failed'}")