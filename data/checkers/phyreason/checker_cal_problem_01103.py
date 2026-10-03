import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance (with fallbacks)
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 100.0))
        self.theta_deg = getattr(solution_instance, 'theta', getattr(solution_instance, 'results', {}).get('theta', 37.0))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 38.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 10.0))
        
        # Safely extract results (with fallbacks)
        self.N = getattr(solution_instance, 'N', getattr(solution_instance, 'results', {}).get('N', 0.0))
        self.f = getattr(solution_instance, 'f', getattr(solution_instance, 'results', {}).get('f', 0.0))
        self.mu = getattr(solution_instance, 'mu', getattr(solution_instance, 'results', {}).get('mu', 0.0))
        
        # Precompute trig values from problem context
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.logs = []
        
    def _log(self, msg):
        self.logs.append(msg)
        
    def test_invariants(self):
        """Verify force equilibrium and friction formula."""
        success = True
        
        # 1. Vertical force equilibrium: N + F*sinθ - m*g = 0
        vertical_net = self.N + self.F * self.sin37 - self.m * self.g
        if not math.isclose(vertical_net, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Vertical equilibrium violated: N + Fsinθ - mg = {vertical_net:.2f}")
            success = False
        else:
            self._log("✓ Vertical force equilibrium satisfied")
            
        # 2. Horizontal force equilibrium: f = F*cosθ (since constant velocity)
        if not math.isclose(self.f, self.F * self.cos37, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Horizontal equilibrium violated: f = {self.f:.2f}, Fcosθ = {self.F * self.cos37:.2f}")
            success = False
        else:
            self._log("✓ Horizontal force equilibrium satisfied")
            
        # 3. Kinetic friction formula: f = μ*N
        if not math.isclose(self.f, self.mu * self.N, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Friction formula violated: f = {self.f:.2f}, μN = {self.mu * self.N:.2f}")
            success = False
        else:
            self._log("✓ Kinetic friction formula satisfied")
            
        return success
        
    def test_trace_consistency(self):
        """Cross-check intermediate results from steps analysis."""
        success = True
        
        # Step 3: N = 320 N
        expected_N = 320.0
        if not math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Normal force mismatch: N = {self.N:.2f}, expected {expected_N:.2f}")
            success = False
        else:
            self._log(f"✓ Normal force matches step 3 (N = {expected_N} N)")
            
        # Step 4: f = 80 N
        expected_f = 80.0
        if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Friction force mismatch: f = {self.f:.2f}, expected {expected_f:.2f}")
            success = False
        else:
            self._log(f"✓ Friction force matches step 4 (f = {expected_f} N)")
            
        # Step 5: μ = 0.25
        expected_mu = 0.25
        if not math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Friction coefficient mismatch: μ = {self.mu:.4f}, expected {expected_mu:.4f}")
            success = False
        else:
            self._log(f"✓ Friction coefficient matches step 5 (μ = {expected_mu})")
            
        return success
        
    def test_monotonicity(self):
        """Perturb mass and check trend in normal force and friction coefficient."""
        success = True
        
        # Store original values
        original_m = self.m
        delta = 1.0  # Small perturbation (1 kg)
        
        # Perturb mass upward
        self.m = original_m + delta
        
        # Recalculate using physics formulas (not relying on solution methods)
        N_perturbed = self.m * self.g - self.F * self.sin37
        f_perturbed = self.F * self.cos37  # Unchanged by mass perturbation
        mu_perturbed = f_perturbed / N_perturbed
        
        # Expected trends:
        # 1. Normal force should increase with mass
        if not (N_perturbed > self.N):
            self._log(f"Normal force monotonicity failed: N({original_m+delta}) = {N_perturbed:.2f} ≤ N({original_m}) = {self.N:.2f}")
            success = False
        else:
            self._log(f"✓ Normal force increases with mass: {self.N:.2f} → {N_perturbed:.2f}")
            
        # 2. Friction force should remain constant (depends only on F and θ)
        if not math.isclose(f_perturbed, self.f, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Friction force changed unexpectedly: f({original_m+delta}) = {f_perturbed:.2f} vs f({original_m}) = {self.f:.2f}")
            success = False
        else:
            self._log(f"✓ Friction force constant with mass: {self.f:.2f} → {f_perturbed:.2f}")
            
        # 3. Friction coefficient should decrease (μ = f/N, f constant, N increases)
        if not (mu_perturbed < self.mu):
            self._log(f"Friction coefficient monotonicity failed: μ({original_m+delta}) = {mu_perturbed:.4f} ≥ μ({original_m}) = {self.mu:.4f}")
            success = False
        else:
            self._log(f"✓ Friction coefficient decreases with mass: {self.mu:.4f} → {mu_perturbed:.4f}")
            
        # Restore original mass
        self.m = original_m
        
        return success
        
    def run_all_tests(self):
        """Execute all checks and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        self._log("=== Physics Checker Started ===")
        
        # Run individual tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Testing {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' failed with exception: {str(e)}")
                overall_success = False
                
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs