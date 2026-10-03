import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Use getattr and .get() to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        
        # Extract inputs with defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.F1 = getattr(solution_instance, 'F1', 10.0)
        
        # Trigonometric values (given in problem)
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        self.a = self.results.get('a', 0.0)
        self.f_2 = self.results.get('f_2', 0.0)
        self.mu = self.results.get('mu', 0.0)
        self.N_1 = self.results.get('N_1', 0.0)
        self.N_2 = self.results.get('N_2', 0.0)
        self.f_1 = self.results.get('f_1', 0.0)
        
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Force equilibrium in first case (constant velocity)
        2. Newton's Second Law in second case
        3. Friction formula consistency
        """
        logs = []
        success = True
        
        try:
            # 1. Check first case equilibrium: F1 = mg sinθ + f1
            mg_sin_theta = self.m * self.g * self.sin_theta  # 1*10*0.6 = 6
            expected_f1 = self.F1 - mg_sin_theta  # 10 - 6 = 4
            if not math.isclose(self.f_1, expected_f1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: First case friction f1={self.f_1}, expected {expected_f1}")
                success = False
            else:
                logs.append(f"PASS: First case friction f1={self.f_1} matches equilibrium")
            
            # 2. Check normal forces: N1 = N2 = mg cosθ
            mg_cos_theta = self.m * self.g * self.cos_theta  # 1*10*0.8 = 8
            for N_name, N_val in [('N_1', self.N_1), ('N_2', self.N_2)]:
                if not math.isclose(N_val, mg_cos_theta, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: {N_name}={N_val}, expected {mg_cos_theta}")
                    success = False
                else:
                    logs.append(f"PASS: {N_name}={N_val} matches mg cosθ")
            
            # 3. Check friction formula: f1 = μ N1
            expected_f1_from_mu = self.mu * self.N_1
            if not math.isclose(self.f_1, expected_f1_from_mu, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f1={self.f_1} ≠ μ N1 = {expected_f1_from_mu}")
                success = False
            else:
                logs.append(f"PASS: f1 = μ N1 holds")
            
            # 4. Check second case friction: f2 = μ N2
            expected_f2 = self.mu * self.N_2
            if not math.isclose(self.f_2, expected_f2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f2={self.f_2} ≠ μ N2 = {expected_f2}")
                success = False
            else:
                logs.append(f"PASS: f2 = μ N2 holds")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Ensure coefficient μ is consistent between steps.
        """
        logs = []
        success = True
        
        try:
            # Check μ from first case: μ = f1 / N1
            mu_from_first = self.f_1 / self.N_1 if abs(self.N_1) > 1e-12 else 0.0
            if not math.isclose(self.mu, mu_from_first, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: μ={self.mu} but f1/N1={mu_from_first}")
                success = False
            else:
                logs.append(f"PASS: μ consistent from first case")
            
            # Check second case acceleration matches expected value (7 m/s²)
            if not math.isclose(self.a, 7.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration a={self.a} ≠ 7.0 from steps")
                success = False
            else:
                logs.append(f"PASS: Acceleration a={self.a} matches steps analysis")
            
            # Numeric check of equilibrium in first case
            lhs = self.F1
            rhs = self.m * self.g * self.sin_theta + self.mu * self.m * self.g * self.cos_theta
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Numeric equilibrium check: LHS={lhs}, RHS={rhs}")
                success = False
            else:
                logs.append(f"PASS: Numeric equilibrium holds")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb mass m and verify expected trends in results.
        μ is a surface property and should remain constant regardless of mass.
        However, with increased mass, the equilibrium condition changes (F1 fixed).
        We test that μ computed from the definition (f1/N1) using the ORIGINAL
        equilibrium values remains valid as a surface property, i.e., 
        μ = constant, and normal force increases with mass.
        """
        logs = []
        success = True
        
        try:
            # Store original values
            m_original = self.m
            f1_original = self.f_1
            N1_original = self.N_1
            mu_original = self.mu
            
            # Perturb mass by +10%
            self.m = m_original * 1.1
            
            # Recompute normal force with new mass (should increase)
            new_N1 = self.m * self.g * self.cos_theta
            
            # Check trends
            # 1. Normal force should increase with mass
            if not (new_N1 > N1_original):
                logs.append(f"FAIL: N1 didn't increase with mass (was {N1_original}, now {new_N1})")
                success = False
            else:
                logs.append(f"PASS: N1 increases with mass: {N1_original} -> {new_N1}")
            
            # 2. With increased mass, if we assume same μ, friction force would be μ*new_N1
            expected_f1_if_same_mu = mu_original * new_N1
            # But equilibrium would require F1 = m*g*sinθ + f1, so f1 = F1 - m*g*sinθ
            f1_for_equilibrium = self.F1 - self.m * self.g * self.sin_theta
            # These two are not equal because F1 is fixed. That's fine.
            # We just verify that μ as a surface property is constant.
            # Compute μ from the definition using the original scenario's values.
            # μ should be independent of mass.
            # We already have mu_original. Let's verify that if we recompute μ
            # using the formula μ = (F1 - m*g*sinθ) / (m*g*cosθ) for the ORIGINAL mass,
            # it matches mu_original.
            mu_recomputed_original = (self.F1 - m_original * self.g * self.sin_theta) / (m_original * self.g * self.cos_theta)
            if not math.isclose(mu_original, mu_recomputed_original, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: μ from original mass formula mismatch: {mu_original} vs {mu_recomputed_original}")
                success = False
            else:
                logs.append(f"PASS: μ formula consistent for original mass")
            
            # 3. Coefficient μ should remain constant as a surface property.
            # This means if we had a different mass but the same surface, μ would be the same.
            # We can't test that directly because we don't have data for different mass.
            # Instead, we verify that μ doesn't change when mass changes in our results?
            # Actually, our results only have one mass. So we just note that μ is constant.
            logs.append(f"INFO: μ is a surface property and should remain constant for same surfaces")
            
            # Restore original values
            self.m = m_original
            self.f_1 = f1_original
            self.N_1 = N1_original
            self.mu = mu_original
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            # Attempt to restore on error
            self.m = m_original if 'm_original' in locals() else self.m
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and aggregated logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_method()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
                overall_success = overall_success and success
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        return overall_success, all_logs