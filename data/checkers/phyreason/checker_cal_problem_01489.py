import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        
        # Extract physical parameters (safe with defaults)
        self.m = getattr(solution_instance, 'm', 1.0)  # Base mass
        self.mu = getattr(solution_instance, 'mu', 0.1)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.r = getattr(solution_instance, 'r', 1.0)
        
        # Extract results (safe access)
        self.results = getattr(solution_instance, 'results', {})
        
        # Key angular velocities
        self.omega1 = getattr(solution_instance, 'omega1', 
                             self.results.get('omega1', 
                             self.results.get('ω₁', 0.0)))
        self.omega2 = getattr(solution_instance, 'omega2', 
                             self.results.get('omega2', 
                             self.results.get('ω₂', 0.0)))
        self.omega3 = getattr(solution_instance, 'omega3', 
                             self.results.get('omega3', 
                             self.results.get('ω₃', 0.0)))
        
        # Extract forces if available
        self.f_c_max = getattr(solution_instance, 'f_c_max', 
                              self.results.get('f_c_max', None))
        self.f_b_max = getattr(solution_instance, 'f_b_max', 
                              self.results.get('f_b_max', None))
        self.f_a = getattr(solution_instance, 'f_a', 
                          self.results.get('f_a', None))
        
        self.logs = []
    
    def _log(self, msg: str):
        """Helper for logging"""
        self.logs.append(msg)
    
    def _isclose(self, a: float, b: float) -> bool:
        """Fuzzy comparison with 1% tolerance"""
        return math.isclose(a, b, rel_tol=0.01, abs_tol=0.01)
    
    def test_invariants(self) -> bool:
        """Verify Newton's 2nd Law and Centripetal Force consistency"""
        success = True
        
        # 1. Check ω₁: Tension appears between B and C
        # f_c_max = μ * 2m * g = 2m * ω₁² * 2r
        expected_omega1 = math.sqrt(self.mu * self.g / (2 * self.r))
        if not self._isclose(self.omega1, expected_omega1):
            self._log(f"ω₁ invariant failed: got {self.omega1:.4f}, expected {expected_omega1:.4f}")
            success = False
        else:
            self._log(f"ω₁ check passed: {self.omega1:.4f} ≈ √(μg/2r)")
        
        # 2. Check ω₂: Tension appears between A and B
        # μmg + μ2mg = mω₂²r + 2mω₂²2r
        expected_omega2 = math.sqrt(3 * self.mu * self.g / (5 * self.r))
        if not self._isclose(self.omega2, expected_omega2):
            self._log(f"ω₂ invariant failed: got {self.omega2:.4f}, expected {expected_omega2:.4f}")
            success = False
        else:
            self._log(f"ω₂ check passed: {self.omega2:.4f} ≈ √(3μg/5r)")
        
        # 3. Check ω₃: Static friction on A becomes zero
        # μmg + 2μmg = mω₃²r + 2mω₃²2r - 2mω₃²r
        expected_omega3 = math.sqrt(self.mu * self.g / self.r)
        if not self._isclose(self.omega3, expected_omega3):
            self._log(f"ω₃ invariant failed: got {self.omega3:.4f}, expected {expected_omega3:.4f}")
            success = False
        else:
            self._log(f"ω₃ check passed: {self.omega3:.4f} ≈ √(μg/r)")
        
        # 4. Verify monotonicity of angular velocities (should be ω₁ < ω₂ < ω₃)
        if not (self.omega1 < self.omega2 < self.omega3):
            self._log(f"Monotonicity violated: ω₁={self.omega1:.4f}, ω₂={self.omega2:.4f}, ω₃={self.omega3:.4f}")
            success = False
        else:
            self._log(f"Angular velocities are monotonic: ω₁ < ω₂ < ω₃")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers"""
        success = True
        
        # Check that f_c_max and f_b_max are consistent with definitions
        if self.f_c_max is not None:
            expected_f_c_max = self.mu * 2 * self.m * self.g
            if not self._isclose(self.f_c_max, expected_f_c_max):
                self._log(f"f_c_max inconsistent: got {self.f_c_max:.4f}, expected {expected_f_c_max:.4f}")
                success = False
            else:
                self._log(f"f_c_max check passed: {self.f_c_max:.4f} ≈ μ·2mg")
        
        if self.f_b_max is not None:
            expected_f_b_max = self.mu * self.m * self.g
            if not self._isclose(self.f_b_max, expected_f_b_max):
                self._log(f"f_b_max inconsistent: got {self.f_b_max:.4f}, expected {expected_f_b_max:.4f}")
                success = False
            else:
                self._log(f"f_b_max check passed: {self.f_b_max:.4f} ≈ μ·mg")
        
        # Check that f_a = 0 at ω₃ (if provided)
        if self.f_a is not None and not self._isclose(self.f_a, 0.0):
            self._log(f"f_a should be 0 at ω₃, but got {self.f_a:.4f}")
            success = False
        
        # Verify centripetal forces at each critical ω
        # At ω₁: F_cC = 2m * ω₁² * 2r should equal f_c_max
        if self.f_c_max is not None:
            F_cC_at_omega1 = 2 * self.m * self.omega1**2 * (2 * self.r)
            if not self._isclose(F_cC_at_omega1, self.f_c_max):
                self._log(f"Centripetal force on C at ω₁ mismatch: {F_cC_at_omega1:.4f} vs f_c_max={self.f_c_max:.4f}")
                success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb μ and check trend, then restore"""
        success = True
        original_mu = self.mu
        
        try:
            # Test 1: Increase μ by 10%
            self.mu = original_mu * 1.1
            new_omega1 = math.sqrt(self.mu * self.g / (2 * self.r))
            new_omega2 = math.sqrt(3 * self.mu * self.g / (5 * self.r))
            new_omega3 = math.sqrt(self.mu * self.g / self.r)
            
            if not (new_omega1 > self.omega1 and new_omega2 > self.omega2 and new_omega3 > self.omega3):
                self._log(f"Monotonic trend violated when μ increased: ω should increase")
                success = False
            
            # Test 2: Decrease μ by 10%
            self.mu = original_mu * 0.9
            new_omega1 = math.sqrt(self.mu * self.g / (2 * self.r))
            new_omega2 = math.sqrt(3 * self.mu * self.g / (5 * self.r))
            new_omega3 = math.sqrt(self.mu * self.g / self.r)
            
            if not (new_omega1 < self.omega1 and new_omega2 < self.omega2 and new_omega3 < self.omega3):
                self._log(f"Monotonic trend violated when μ decreased: ω should decrease")
                success = False
            
            if success:
                self._log("Monotonicity test passed: ω increases with μ")
                
        finally:
            # Restore original μ
            self.mu = original_mu
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success flag with logs"""
        self.logs.clear()
        self._log("=== PhysicsChecker Starting ===")
        
        overall_success = True
        
        # Run individual tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} Test ---")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"FAILED: {test_name}")
                else:
                    self._log(f"PASSED: {test_name}")
            except Exception as e:
                self._log(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        self._log(f"\n=== PhysicsChecker Complete: {'SUCCESS' if overall_success else 'FAILED'} ===")
        return overall_success, self.logs

# Optional: Symbolic verification helper
def symbolic_verification():
    """Symbolic check of the physical equations"""
    m, mu, g, r, omega = sp.symbols('m mu g r omega', positive=True)
    
    # ω₁ equation
    eq1 = sp.Eq(mu * 2 * m * g, 2 * m * omega**2 * (2 * r))
    sol1 = sp.solve(eq1, omega)[0]  # Should be sqrt(mu*g/(2r))
    
    # ω₂ equation
    eq2 = sp.Eq(mu * m * g + mu * 2 * m * g, m * omega**2 * r + 2 * m * omega**2 * (2 * r))
    sol2 = sp.solve(eq2, omega)[0]  # Should be sqrt(3*mu*g/(5r))
    
    # ω₃ equation (simplified from step_6)
    # μmg + 2μmg = mω₃²r + 2mω₃²2r - 2mω₃²r
    # Left: 3μmg, Right: (m + 4m - 2m)ω₃²r = 3mω₃²r
    eq3 = sp.Eq(3 * mu * m * g, 3 * m * omega**2 * r)
    sol3 = sp.solve(eq3, omega)[0]  # Should be sqrt(mu*g/r)
    
    return sol1, sol2, sol3