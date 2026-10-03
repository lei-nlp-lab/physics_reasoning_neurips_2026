import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.m = getattr(solution_instance, 'm', 4.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.F = getattr(solution_instance, 'F', 10.0)
        self.theta = getattr(solution_instance, 'theta', math.radians(30))
        self.F0 = getattr(solution_instance, 'F0', 20.0)  # For verge of motion
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.N = getattr(solution_instance, 'N', results.get('N', None))
        self.f = getattr(solution_instance, 'f', results.get('f', None))
        self.mu = getattr(solution_instance, 'mu', results.get('mu', None))
        self.N0 = getattr(solution_instance, 'N_0', results.get('N_0', None))
        self.f0 = getattr(solution_instance, 'f_0', results.get('f_0', None))
        
        # For symbolic computations
        self.m_sym = sp.symbols('m', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        self.F_sym = sp.symbols('F', positive=True)
        self.theta_sym = sp.symbols('theta', positive=True)
        self.mu_sym = sp.symbols('mu', positive=True)
        
        self.logs = []
    
    def _log(self, message, success):
        self.logs.append((success, message))
    
    def test_invariants(self):
        """Verify physics laws from KB: equilibrium and friction relationships."""
        try:
            # Part 1: Equilibrium with F=10N
            # Vertical equilibrium: N = m*g - F*sin(theta)
            if self.N is not None:
                expected_N = self.m * self.g - self.F * math.sin(self.theta)
                if math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Vertical equilibrium: N={self.N:.3f} = m*g - F*sinθ={expected_N:.3f}", True)
                else:
                    self._log(f"Vertical equilibrium FAILED: N={self.N:.3f} ≠ m*g - F*sinθ={expected_N:.3f}", False)
            
            # Horizontal equilibrium: f = F*cos(theta)
            if self.f is not None:
                expected_f = self.F * math.cos(self.theta)
                if math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Horizontal equilibrium: f={self.f:.3f} = F*cosθ={expected_f:.3f}", True)
                else:
                    self._log(f"Horizontal equilibrium FAILED: f={self.f:.3f} ≠ F*cosθ={expected_f:.3f}", False)
            
            # Part 2: At verge of motion with F0=20N
            # Vertical equilibrium: N0 = m*g - F0*sin(theta)
            if self.N0 is not None:
                expected_N0 = self.m * self.g - self.F0 * math.sin(self.theta)
                if math.isclose(self.N0, expected_N0, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Verge equilibrium (vertical): N0={self.N0:.3f} = m*g - F0*sinθ={expected_N0:.3f}", True)
                else:
                    self._log(f"Verge equilibrium (vertical) FAILED: N0={self.N0:.3f} ≠ m*g - F0*sinθ={expected_N0:.3f}", False)
            
            # Maximum static friction: f0 = μ * N0 = F0*cos(theta)
            if self.mu is not None and self.N0 is not None and self.f0 is not None:
                # Check both relationships
                expected_f0_from_mu = self.mu * self.N0
                expected_f0_from_F0 = self.F0 * math.cos(self.theta)
                
                check1 = math.isclose(self.f0, expected_f0_from_mu, rel_tol=1e-2, abs_tol=1e-2)
                check2 = math.isclose(self.f0, expected_f0_from_F0, rel_tol=1e-2, abs_tol=1e-2)
                
                if check1 and check2:
                    self._log(f"Friction limit: f0={self.f0:.3f} = μ*N0={expected_f0_from_mu:.3f} = F0*cosθ={expected_f0_from_F0:.3f}", True)
                else:
                    msg = f"Friction limit FAILED: f0={self.f0:.3f}, μ*N0={expected_f0_from_mu:.3f}, F0*cosθ={expected_f0_from_F0:.3f}"
                    self._log(msg, False)
            
            # Also check μ directly from the derived formula
            if self.mu is not None:
                expected_mu = (self.F0 * math.cos(self.theta)) / (self.m * self.g - self.F0 * math.sin(self.theta))
                if math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"μ formula: μ={self.mu:.3f} = F0*cosθ/(m*g - F0*sinθ)={expected_mu:.3f}", True)
                else:
                    self._log(f"μ formula FAILED: μ={self.mu:.3f} ≠ F0*cosθ/(m*g - F0*sinθ)={expected_mu:.3f}", False)
            
        except Exception as e:
            self._log(f"test_invariants crashed: {traceback.format_exc()}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check part1 answers consistency
            if self.N is not None and self.f is not None:
                # N should be 35, f should be 5√3 ≈ 8.660
                expected_N = 35.0
                expected_f = 5 * math.sqrt(3)  # ≈ 8.660254
                
                if math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Part1 N matches target: {self.N:.3f} ≈ {expected_N}", True)
                else:
                    self._log(f"Part1 N mismatch: {self.N:.3f} ≠ {expected_N}", False)
                
                if math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Part1 f matches target: {self.f:.3f} ≈ {expected_f:.3f}", True)
                else:
                    self._log(f"Part1 f mismatch: {self.f:.3f} ≠ {expected_f:.3f}", False)
            
            # Check part2 answer consistency
            if self.mu is not None:
                expected_mu = math.sqrt(3) / 3  # ≈ 0.57735
                if math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Part2 μ matches target: {self.mu:.3f} ≈ {expected_mu:.3f}", True)
                else:
                    self._log(f"Part2 μ mismatch: {self.mu:.3f} ≠ {expected_mu:.3f}", False)
            
            # Check intermediate step consistency for part2
            if self.N0 is not None:
                expected_N0 = self.m * self.g - self.F0 * math.sin(self.theta)  # Should be 30
                if math.isclose(self.N0, expected_N0, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Intermediate N0 consistent: {self.N0:.3f} ≈ {expected_N0:.3f}", True)
                else:
                    self._log(f"Intermediate N0 inconsistent: {self.N0:.3f} ≠ {expected_N0:.3f}", False)
            
            if self.f0 is not None:
                expected_f0 = self.F0 * math.cos(self.theta)  # Should be 10√3 ≈ 17.3205
                if math.isclose(self.f0, expected_f0, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Intermediate f0 consistent: {self.f0:.3f} ≈ {expected_f0:.3f}", True)
                else:
                    self._log(f"Intermediate f0 inconsistent: {self.f0:.3f} ≠ {expected_f0:.3f}", False)
        
        except Exception as e:
            self._log(f"test_trace_consistency crashed: {traceback.format_exc()}", False)
    
    def test_monotonicity(self):
        """Perturb mass m and check physics trends, then restore."""
        original_m = self.m
        original_logs = len(self.logs)
        
        try:
            # Test 1: Increase mass by 10%
            self.m = original_m * 1.1
            
            # Recompute expected values using formulas
            new_N = self.m * self.g - self.F * math.sin(self.theta)
            new_f = self.F * math.cos(self.theta)  # Should be unchanged
            new_mu = (self.F0 * math.cos(self.theta)) / (self.m * self.g - self.F0 * math.sin(self.theta))
            
            # Check trends
            if self.N is not None and new_N > self.N:
                self._log(f"Mass increase → N increases ({self.N:.3f} → {new_N:.3f})", True)
            elif self.N is not None:
                self._log(f"Mass increase but N didn't increase as expected", False)
            
            # f should not change with m
            if self.f is not None and math.isclose(new_f, self.f, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Mass increase → f unchanged ({self.f:.3f} → {new_f:.3f})", True)
            elif self.f is not None:
                self._log(f"Mass increase but f changed unexpectedly", False)
            
            # μ should decrease when m increases (denominator increases)
            if self.mu is not None and new_mu < self.mu:
                self._log(f"Mass increase → μ decreases ({self.mu:.3f} → {new_mu:.3f})", True)
            elif self.mu is not None:
                self._log(f"Mass increase but μ didn't decrease as expected", False)
            
            # Test 2: Decrease mass by 10%
            self.m = original_m * 0.9
            new_N = self.m * self.g - self.F * math.sin(self.theta)
            new_mu = (self.F0 * math.cos(self.theta)) / (self.m * self.g - self.F0 * math.sin(self.theta))
            
            if self.N is not None and new_N < self.N:
                self._log(f"Mass decrease → N decreases ({self.N:.3f} → {new_N:.3f})", True)
            elif self.N is not None:
                self._log(f"Mass decrease but N didn't decrease as expected", False)
            
            if self.mu is not None and new_mu > self.mu:
                self._log(f"Mass decrease → μ increases ({self.mu:.3f} → {new_mu:.3f})", True)
            elif self.mu is not None:
                self._log(f"Mass decrease but μ didn't increase as expected", False)
            
        except Exception as e:
            self._log(f"test_monotonicity crashed: {traceback.format_exc()}", False)
        finally:
            # Restore original mass
            self.m = original_m
        
        # If no new logs were added, the test was skipped
        if len(self.logs) == original_logs:
            self._log("Monotonicity test skipped (missing data)", True)
    
    def run_all_tests(self):
        """Execute all checks and return success status with logs."""
        self.logs = []
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success: all tests must pass
        success = all(log[0] for log in self.logs)
        
        return success, self.logs