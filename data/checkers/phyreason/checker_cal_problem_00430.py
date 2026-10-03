import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', 30.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.F1 = getattr(solution_instance, 'F1', 80.0)
        self.F2 = getattr(solution_instance, 'F2', 120.0)
        
        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        self.f1 = getattr(solution_instance, 'f_1', results.get('f_1', results.get('f_1', 0.0)))
        self.mu = getattr(solution_instance, 'mu', results.get('mu', results.get('μ', 0.0)))
        self.Ff = getattr(solution_instance, 'F_f', results.get('F_f', results.get('F_f', 0.0)))
        self.f2 = getattr(solution_instance, 'f_2', results.get('f_2', results.get('f_2', 0.0)))
        
        self.logs = []
    
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify force balance laws and friction formulas."""
        success = True
        
        # 1. Static equilibrium: f1 must equal F1
        if not math.isclose(self.f1, self.F1, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Static equilibrium violated: f1={self.f1} N, F1={self.F1} N")
            success = False
        else:
            self.log(f"Static equilibrium OK: f1 = F1 = {self.f1} N")
        
        # 2. Uniform motion: Ff (kinetic friction) must equal F2
        if not math.isclose(self.Ff, self.F2, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Uniform motion violated: Ff={self.Ff} N, F2={self.F2} N")
            success = False
        else:
            self.log(f"Uniform motion OK: Ff = F2 = {self.Ff} N")
        
        # 3. Kinetic friction formula: μ = Ff / (m*g)
        expected_mu = self.F2 / (self.m * self.g)
        if not math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Friction formula violated: μ={self.mu}, expected {expected_mu}")
            success = False
        else:
            self.log(f"Friction formula OK: μ = {self.mu}")
        
        # 4. f2 should equal μ*m*g and match F2
        expected_f2 = self.mu * self.m * self.g
        if not (math.isclose(self.f2, expected_f2, rel_tol=1e-2, abs_tol=1e-2) and 
                math.isclose(self.f2, self.F2, rel_tol=1e-2, abs_tol=1e-2)):
            self.log(f"Sliding friction consistency violated: f2={self.f2} N, μmg={expected_f2} N, F2={self.F2} N")
            success = False
        else:
            self.log(f"Sliding friction consistency OK: f2 = μmg = F2 = {self.f2} N")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # μ from step_6 should match the formula μ = Ff / (m*g)
        mu_from_formula = self.Ff / (self.m * self.g)
        if not math.isclose(self.mu, mu_from_formula, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Trace inconsistency in μ: {self.mu} vs computed {mu_from_formula}")
            success = False
        else:
            self.log(f"μ trace consistency OK: {self.mu}")
        
        # f2 from step_8 should equal μ*m*g
        f2_from_mu = self.mu * self.m * self.g
        if not math.isclose(self.f2, f2_from_mu, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Trace inconsistency in f2: {self.f2} vs computed {f2_from_mu}")
            success = False
        else:
            self.log(f"f2 trace consistency OK: {self.f2} N")
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb mass m and verify friction forces change correctly.
        Restore original mass after test.
        """
        success = True
        original_m = self.m
        
        # Increase mass by 10%
        self.m *= 1.1
        # Recompute expected values
        new_mu = self.F2 / (self.m * self.g)  # μ should decrease
        new_f2 = new_mu * self.m * self.g     # f2 should remain F2 (constant velocity condition)
        
        # Check monotonicity: μ is inversely proportional to m
        if not (new_mu < self.mu):
            self.log(f"Monotonicity violated: μ did not decrease when m increased. Original μ={self.mu}, new μ={new_mu}")
            success = False
        else:
            self.log(f"μ monotonicity OK: μ decreases when m increases")
        
        # f2 should remain equal to F2 (120 N) because it's set by the constant velocity condition
        if not math.isclose(new_f2, self.F2, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Monotonicity violated: f2 changed with m. Original f2={self.f2} N, new f2={new_f2} N")
            success = False
        else:
            self.log(f"f2 monotonicity OK: f2 remains at F2={self.F2} N when m changes")
        
        # Restore original mass
        self.m = original_m
        return success
    
    def run_all_tests(self):
        """Execute all checks and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.log(f"\n--- Running {name} Test ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"Error in {name}: {traceback.format_exc()}")
                overall_success = False
        
        self.log(f"\n=== Overall Test Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs