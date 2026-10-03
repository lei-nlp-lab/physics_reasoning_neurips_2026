import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using results dict or instance attributes
        results = getattr(solution_instance, 'results', {})
        
        # Extract quantities with safe defaults
        self.d = getattr(solution_instance, 'd', results.get('d', 0.0))
        self.phi_C = getattr(solution_instance, 'phi_C', results.get('phi_C', 0.0))
        self.U_CD = getattr(solution_instance, 'U_CD', results.get('U_CD', 0.0))
        self.d_DB = getattr(solution_instance, 'd_DB', results.get('d_DB', 0.0))
        self.phi_D = getattr(solution_instance, 'phi_D', results.get('phi_D', 0.0))
        self.C = getattr(solution_instance, 'C', results.get('C', 0.0))
        self.d_CB = getattr(solution_instance, 'd_CB', results.get('d_CB', 0.0))
        self.U = getattr(solution_instance, 'U', results.get('U', 0.0))
        self.phi_B = getattr(solution_instance, 'phi_B', results.get('phi_B', 0.0))
        
        # Constants from problem
        self.E = 120.0  # V/m
        self.d_CA = 0.005  # m (distance from C to plate A)
        
        # Optional quantities for capacitance check
        self.Q = getattr(solution_instance, 'Q', results.get('Q', None))
        self.A = getattr(solution_instance, 'A', results.get('A', None))
        self.epsilon_0 = getattr(solution_instance, 'epsilon_0', results.get('epsilon_0', 8.854e-12))
        
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
        
    def test_invariants(self):
        """Test physical laws and relationships."""
        tests = []
        
        # 1. Ground condition: plate B at 0V
        tests.append(self._log(
            f"φ_B = {self.phi_B} V (should be 0)",
            math.isclose(self.phi_B, 0.0, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 2. Uniform E-field relationship: U = E*d between plates
        expected_U = self.E * self.d
        tests.append(self._log(
            f"U = E*d: {self.U} vs {expected_U} V",
            math.isclose(self.U, expected_U, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 3. Potential from field: φ_D = -E*d_DB (since field points from high to low potential)
        expected_phi_D = -self.E * self.d_DB
        tests.append(self._log(
            f"φ_D = -E*d_DB: {self.phi_D} vs {expected_phi_D} V",
            math.isclose(self.phi_D, expected_phi_D, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 4. Geometry: d_CB = d - d_CA
        expected_d_CB = self.d - self.d_CA
        tests.append(self._log(
            f"d_CB = d - d_CA: {self.d_CB} vs {expected_d_CB} m",
            math.isclose(self.d_CB, expected_d_CB, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 5. Potential from field: φ_C = -E*d_CB
        expected_phi_C = -self.E * self.d_CB
        tests.append(self._log(
            f"φ_C = -E*d_CB: {self.phi_C} vs {expected_phi_C} V",
            math.isclose(self.phi_C, expected_phi_C, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 6. Potential difference definition: U_CD = φ_C - φ_D
        expected_U_CD = self.phi_C - self.phi_D
        tests.append(self._log(
            f"U_CD = φ_C - φ_D: {self.U_CD} vs {expected_U_CD} V",
            math.isclose(self.U_CD, expected_U_CD, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 7. Capacitance checks (multiple possible relationships)
        cap_tests = []
        
        # Option 1: C = Q/U if Q available
        if self.Q is not None:
            expected_C1 = self.Q / self.U if abs(self.U) > 1e-12 else 0.0
            cap_tests.append(math.isclose(self.C, expected_C1, rel_tol=1e-2, abs_tol=1e-2))
            self._log(f"C = Q/U: {self.C} vs {expected_C1} F", cap_tests[-1])
        
        # Option 2: C = ε₀*A/d if A available
        if self.A is not None:
            expected_C2 = self.epsilon_0 * self.A / self.d
            cap_tests.append(math.isclose(self.C, expected_C2, rel_tol=1e-2, abs_tol=1e-2))
            self._log(f"C = ε₀*A/d: {self.C} vs {expected_C2} F", cap_tests[-1])
        
        # Option 3: Compare with target value 1e-8 F
        target_C = 1e-8
        cap_tests.append(math.isclose(self.C, target_C, rel_tol=1e-2, abs_tol=1e-2))
        self._log(f"C ≈ 1e-8 F: {self.C} vs {target_C} F", cap_tests[-1])
        
        # At least one capacitance test should pass
        if cap_tests:
            tests.append(any(cap_tests))
            self._log("At least one capacitance relationship holds", any(cap_tests))
        else:
            tests.append(False)
            self._log("No capacitance check possible", False)
        
        return all(tests)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests = []
        
        # Check against expected values from steps analysis
        expected_values = {
            'd_DB': 0.005,
            'phi_D': -0.6,
            'd_CB': 0.045,
            'phi_C': -5.4,
            'U_CD': -4.8,
            'd': 0.05,
            'U': 6.0,
            'C': 1e-8
        }
        
        for key, expected in expected_values.items():
            actual = getattr(self, key, None)
            if actual is not None:
                tests.append(self._log(
                    f"{key}: {actual} vs {expected}",
                    math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2)
                ))
        
        # Check sign consistency: φ_C should be more negative than φ_D
        tests.append(self._log(
            f"φ_C ({self.phi_C}) < φ_D ({self.phi_D})",
            self.phi_C < self.phi_D
        ))
        
        # Check U_CD sign matches φ_C - φ_D
        sign_consistent = (self.U_CD > 0) == ((self.phi_C - self.phi_D) > 0)
        tests.append(self._log(
            f"U_CD sign consistent with φ_C - φ_D",
            sign_consistent
        ))
        
        return all(tests)
    
    def test_monotonicity(self):
        """Perturb input and check trend."""
        try:
            # Perturb distance d by +10%
            original_d = self.d
            perturbed_d = original_d * 1.1
            
            # Expected changes using formulas
            new_U = self.E * perturbed_d
            new_d_CB = perturbed_d - self.d_CA
            new_phi_C = -self.E * new_d_CB
            
            # Check monotonic relationships
            tests = []
            
            # U should increase with d (U = E*d, E constant)
            tests.append(self._log(
                f"U increases with d: {self.U} → {new_U}",
                new_U > self.U
            ))
            
            # φ_C should become more negative with larger d (since d_CB increases)
            tests.append(self._log(
                f"|φ_C| increases with d: {self.phi_C} → {new_phi_C}",
                abs(new_phi_C) > abs(self.phi_C)
            ))
            
            # Restore original value (conceptually)
            self.d = original_d
            
            return all(tests)
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        success = True
        
        try:
            self.logs.append("=== PhysicsChecker Starting ===")
            
            # Run individual test suites
            invariants_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            monotonic_ok = self.test_monotonicity()
            
            success = invariants_ok and trace_ok and monotonic_ok
            
            self.logs.append(f"\nSummary:")
            self.logs.append(f"  Invariants: {'PASS' if invariants_ok else 'FAIL'}")
            self.logs.append(f"  Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
            self.logs.append(f"  Monotonicity: {'PASS' if monotonic_ok else 'FAIL'}")
            self.logs.append(f"\nOverall: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
            
        return success, self.logs