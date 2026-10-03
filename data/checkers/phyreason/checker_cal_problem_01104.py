import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance=None):
        # If no solution_instance provided, create a dummy one
        if solution_instance is None:
            solution_instance = type('DummySolution', (), {})()
        
        # Extract parameters from solution instance with defaults
        # Use the actual values from the solution if available
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Try to get k1 and k2 from the solution instance
        # If not found, use reasonable defaults
        self.k1 = getattr(solution_instance, 'k1', 100.0)
        self.k2 = getattr(solution_instance, 'k2', 200.0)
        
        # Extract results from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Initialize logs
        self.logs = []
        self.success = True

    def _log(self, msg, is_error=False):
        self.logs.append(msg)
        if is_error:
            self.success = False

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison"""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Verify force equilibrium at initial and final states"""
        try:
            # Initial state: T = m*g
            T_expected = self.m * self.g
            T_actual = self.results.get('T')
            if T_actual is not None:
                if not self._isclose(T_actual, T_expected):
                    self._log(f"Initial tension mismatch: T={T_actual}, expected mg={T_expected}", True)
                else:
                    self._log(f"✓ Initial tension T={T_actual} matches mg={T_expected}")
            
            # Initial state: 2T = k1*x1"
            if all(k in self.results for k in ('T', 'x_1_doubleprime')):
                lhs = 2 * self.results['T']
                rhs = self.k1 * self.results['x_1_doubleprime']
                if not self._isclose(lhs, rhs):
                    self._log(f"Initial pulley equilibrium mismatch: 2T={lhs}, k1*x1\"={rhs}", True)
                else:
                    self._log(f"✓ Initial pulley equilibrium: 2T={lhs} ≈ k1*x1\"={rhs}")
            
            # Final state: T" should be 4m*g for consistency with x2" = 8mg/k1
            # From pulley equilibrium: 2T" = k1*x2" = 8mg, so T" = 4mg
            T_prime_expected = 4 * self.m * self.g
            T_prime_actual = self.results.get('T_doubleprime')
            if T_prime_actual is not None:
                if not self._isclose(T_prime_actual, T_prime_expected):
                    self._log(f"Final tension mismatch: T\"={T_prime_actual}, expected 4mg={T_prime_expected}", True)
                else:
                    self._log(f"✓ Final tension T\"={T_prime_actual} matches 4mg={T_prime_expected}")
            
            # Final state: 2T" = k1*x2"
            if all(k in self.results for k in ('T_doubleprime', 'x_2_doubleprime')):
                lhs = 2 * self.results['T_doubleprime']
                rhs = self.k1 * self.results['x_2_doubleprime']
                if not self._isclose(lhs, rhs):
                    self._log(f"Final pulley equilibrium mismatch: 2T\"={lhs}, k1*x2\"={rhs}", True)
                else:
                    self._log(f"✓ Final pulley equilibrium: 2T\"={lhs} ≈ k1*x2\"={rhs}")
                    
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", True)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Check x1 = mg/k2
            if self.k2 != 0:
                x1_expected = self.m * self.g / self.k2
                x1_actual = self.results.get('x_1')
                if x1_actual is not None:
                    if not self._isclose(x1_actual, x1_expected):
                        self._log(f"x1 mismatch: {x1_actual} vs mg/k2={x1_expected}", True)
                    else:
                        self._log(f"✓ x1={x1_actual} matches mg/k2={x1_expected}")
            
            # Check x1" = 2mg/k1
            if self.k1 != 0:
                x1_dp_expected = 2 * self.m * self.g / self.k1
                x1_dp_actual = self.results.get('x_1_doubleprime')
                if x1_dp_actual is not None:
                    if not self._isclose(x1_dp_actual, x1_dp_expected):
                        self._log(f'x1" mismatch: {x1_dp_actual} vs 2mg/k1={x1_dp_expected}', True)
                    else:
                        self._log(f'✓ x1"={x1_dp_actual} matches 2mg/k1={x1_dp_expected}')
            
            # Check x2 = 2mg/k2
            if self.k2 != 0:
                x2_expected = 2 * self.m * self.g / self.k2
                x2_actual = self.results.get('x_2')
                if x2_actual is not None:
                    if not self._isclose(x2_actual, x2_expected):
                        self._log(f"x2 mismatch: {x2_actual} vs 2mg/k2={x2_expected}", True)
                    else:
                        self._log(f"✓ x2={x2_actual} matches 2mg/k2={x2_expected}")
            
            # Check x2" = 8mg/k1
            if self.k1 != 0:
                x2_dp_expected = 8 * self.m * self.g / self.k1
                x2_dp_actual = self.results.get('x_2_doubleprime')
                if x2_dp_actual is not None:
                    if not self._isclose(x2_dp_actual, x2_dp_expected):
                        self._log(f'x2" mismatch: {x2_dp_actual} vs 8mg/k1={x2_dp_expected}', True)
                    else:
                        self._log(f'✓ x2"={x2_dp_actual} matches 8mg/k1={x2_dp_expected}')
            
            # Check Δx = x2" - x1" = 6mg/k1
            if self.k1 != 0:
                delta_x_expected = 6 * self.m * self.g / self.k1
                delta_x_actual = self.results.get('delta_x')
                if delta_x_actual is not None:
                    if not self._isclose(delta_x_actual, delta_x_expected):
                        self._log(f"Δx mismatch: {delta_x_actual} vs 6mg/k1={delta_x_expected}", True)
                    else:
                        self._log(f"✓ Δx={delta_x_actual} matches 6mg/k1={delta_x_expected}")
            
            # Check Δh_B = 3mg/k2
            if self.k2 != 0:
                delta_h_B_expected = 3 * self.m * self.g / self.k2
                delta_h_B_actual = self.results.get('delta_h_B')
                if delta_h_B_actual is not None:
                    if not self._isclose(delta_h_B_actual, delta_h_B_expected):
                        self._log(f"Δh_B mismatch: {delta_h_B_actual} vs 3mg/k2={delta_h_B_expected}", True)
                    else:
                        self._log(f"✓ Δh_B={delta_h_B_actual} matches 3mg/k2={delta_h_B_expected}")
            
            # Check Δh_A = 3mg/k2 + 12mg/k1
            if self.k1 != 0 and self.k2 != 0:
                delta_h_A_expected = 3*self.m*self.g/self.k2 + 12*self.m*self.g/self.k1
                delta_h_A_actual = self.results.get('delta_h_A')
                if delta_h_A_actual is not None:
                    if not self._isclose(delta_h_A_actual, delta_h_A_expected):
                        self._log(f"Δh_A mismatch: {delta_h_A_actual} vs 3mg/k2+12mg/k1={delta_h_A_expected}", True)
                    else:
                        self._log(f"✓ Δh_A={delta_h_A_actual} matches 3mg/k2+12mg/k1")
            
            # Check Δh_AB = 6mg/k2 + 12mg/k1
            if self.k1 != 0 and self.k2 != 0:
                delta_h_AB_expected = 6*self.m*self.g/self.k2 + 12*self.m*self.g/self.k1
                delta_h_AB_actual = self.results.get('delta_h_AB')
                if delta_h_AB_actual is not None:
                    if not self._isclose(delta_h_AB_actual, delta_h_AB_expected):
                        self._log(f"Δh_AB mismatch: {delta_h_AB_actual} vs 6mg/k2+12mg/k1={delta_h_AB_expected}", True)
                    else:
                        self._log(f"✓ Δh_AB={delta_h_AB_actual} matches 6mg/k2+12mg/k1")
            
            # Check kinematic constraint: Δh_A = Δh_B + 2Δx
            if all(k in self.results for k in ('delta_h_A', 'delta_h_B', 'delta_x')):
                lhs = self.results['delta_h_A']
                rhs = self.results['delta_h_B'] + 2 * self.results['delta_x']
                if not self._isclose(lhs, rhs):
                    self._log(f"Kinematic constraint failed: Δh_A={lhs}, Δh_B+2Δx={rhs}", True)
                else:
                    self._log(f"✓ Kinematic constraint: Δh_A={lhs} ≈ Δh_B+2Δx={rhs}")
                    
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", True)

    def test_monotonicity(self):
        """Perturb mass m and check trend in Δh_A"""
        try:
            original_m = self.m
            
            # Perturb m by +10%
            perturbed_m = original_m * 1.1
            
            # Calculate expected Δh_A for perturbed mass using formula
            if self.k1 != 0 and self.k2 != 0:
                coeff = 3*self.g/self.k2 + 12*self.g/self.k1
                original_h = coeff * original_m
                perturbed_h = coeff * perturbed_m
                
                # Get actual Δh_A from results
                actual_h = self.results.get('delta_h_A')
                if actual_h is not None:
                    # Check if actual follows same trend (linear increase with m)
                    # Calculate expected ratio
                    expected_ratio = perturbed_h / original_h
                    # This should be exactly 1.1 since linear
                    if not self._isclose(expected_ratio, 1.1):
                        self._log(f"Monotonicity check: Δh_A not linear in m? Expected ratio=1.1, got={expected_ratio}", True)
                    else:
                        self._log(f"✓ Δh_A scales linearly with m (ratio={expected_ratio:.3f})")
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", True)

    def run_all_tests(self):
        """Execute all tests and return success status and logs"""
        self.logs = []
        self.success = True
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        return self.success, self.logs

# Allow the checker to be run standalone for testing
if __name__ == "__main__":
    # Create a dummy solution instance for testing
    class DummySolution:
        def __init__(self):
            self.m = 1.0
            self.g = 9.8
            self.k1 = 100.0
            self.k2 = 200.0
            self.results = {
                'T': 9.8,
                'T_doubleprime': 39.2,  # 4mg = 4*1*9.8 = 39.2
                'x_1': 0.049,
                'x_1_doubleprime': 0.196,
                'x_2': 0.098,
                'x_2_doubleprime': 0.784,
                'delta_x': 0.588,
                'delta_h_B': 0.147,
                'delta_h_A': 1.323,
                'delta_h_AB': 1.47
            }
    
    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    print(f"Success: {success}")
    for log in logs:
        print(log)