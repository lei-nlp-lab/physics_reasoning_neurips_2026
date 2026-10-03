import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get() with defaults
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.d = getattr(solution_instance, 'd', 2.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.alpha = getattr(solution_instance, 'alpha', math.pi/6)  # 30 degrees
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.I0 = results.get('I_0', 0.0)
        self.I1 = results.get('I_1', 0.0)
        self.v1 = results.get('v_1', 0.0)
        self.Q = results.get('Q', 0.0)
        self.B = results.get('B', 0.0)
        self.E1 = results.get('E_1', 0.0)
        self.E0 = results.get('E_0', 0.0)
        self.U = results.get('U', 0.0)
        
        # Symbolic variables for checks
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'g': sp.symbols('g', positive=True),
            'R': sp.symbols('R', positive=True),
            'v0': sp.symbols('v0', positive=True),
            'd': sp.symbols('d', positive=True),
            'L': sp.symbols('L', positive=True),
            'alpha': sp.symbols('alpha'),
            'B': sp.symbols('B'),
            'I0': sp.symbols('I0'),
            'I1': sp.symbols('I1'),
            'E0': sp.symbols('E0'),
            'E1': sp.symbols('E1'),
            'v1': sp.symbols('v1'),
            'U': sp.symbols('U'),
            'Q': sp.symbols('Q')
        }
        
        # Logs for test results
        self.logs = []
        
    def _log(self, message: str, is_error: bool = False):
        """Helper to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Check fundamental physics invariants."""
        success = True
        
        try:
            # 1. Check equilibrium when cd enters (constant velocity v0)
            # From step 3: m*g*sin(alpha) = B*I0*L
            sin_alpha = math.sin(self.alpha)
            lhs_eq = self.m * self.g * sin_alpha
            rhs_eq = self.B * self.I0 * self.L
            
            if not self._is_close(lhs_eq, rhs_eq):
                self._log(f"Equilibrium check failed: m*g*sin(alpha)={lhs_eq:.3f}, B*I0*L={rhs_eq:.3f}", True)
                success = False
            else:
                self._log(f"Equilibrium check passed: {lhs_eq:.3f} ≈ {rhs_eq:.3f}")
            
            # 2. Check Newton's second law when ab reaches PQ
            # From step 9: B*I1*L - m*g*sin(alpha) = m*(g/2)
            # This gives net force = m*a with a = g/2
            net_force = self.B * self.I1 * self.L - self.m * self.g * sin_alpha
            expected_force = self.m * (self.g / 2)
            
            if not self._is_close(net_force, expected_force):
                self._log(f"Newton's second law check failed: net_force={net_force:.3f}, m*(g/2)={expected_force:.3f}", True)
                success = False
            else:
                self._log(f"Newton's second law check passed: {net_force:.3f} ≈ {expected_force:.3f}")
            
            # 3. Check energy conservation (step 11)
            # m*g*(d+L)*sin(alpha) = 0.5*m*v1^2 - 0.5*m*v0^2 + Q
            mech_energy = self.m * self.g * (self.d + self.L) * sin_alpha
            delta_ke = 0.5 * self.m * (self.v1**2 - self.v0**2)
            total_energy = delta_ke + self.Q
            
            if not self._is_close(mech_energy, total_energy):
                self._log(f"Energy conservation check failed: mech={mech_energy:.3f}, ΔKE+Q={total_energy:.3f}", True)
                success = False
            else:
                self._log(f"Energy conservation check passed: {mech_energy:.3f} ≈ {total_energy:.3f}")
                
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", True)
            success = False
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate steps and final results."""
        success = True
        
        try:
            # 1. Check B formula from step 4: B = (1/L)*sqrt(m*g*R/(2*v0))
            B_expected = (1/self.L) * math.sqrt(self.m * self.g * self.R / (2 * self.v0))
            if not self._is_close(self.B, B_expected):
                self._log(f"B formula check failed: computed={self.B:.3f}, expected={B_expected:.3f}", True)
                success = False
            else:
                self._log(f"B formula check passed: {self.B:.3f} ≈ {B_expected:.3f}")
            
            # 2. Check U formula from step 6: U = (3/8)*sqrt(2*m*g*R*v0)
            U_expected = (3/8) * math.sqrt(2 * self.m * self.g * self.R * self.v0)
            if not self._is_close(self.U, U_expected):
                self._log(f"U formula check failed: computed={self.U:.3f}, expected={U_expected:.3f}", True)
                success = False
            else:
                self._log(f"U formula check passed: {self.U:.3f} ≈ {U_expected:.3f}")
            
            # 3. Check v1 from step 10: v1 = 2*v0
            if not self._is_close(self.v1, 2 * self.v0):
                self._log(f"v1 check failed: computed={self.v1:.3f}, expected={2*self.v0:.3f}", True)
                success = False
            else:
                self._log(f"v1 check passed: {self.v1:.3f} ≈ {2*self.v0:.3f}")
            
            # 4. Check Q formula from step 12: Q = [m*g*(d+L) - 3*m*v0^2]/2
            Q_expected = (self.m * self.g * (self.d + self.L) - 3 * self.m * self.v0**2) / 2
            if not self._is_close(self.Q, Q_expected):
                self._log(f"Q formula check failed: computed={self.Q:.3f}, expected={Q_expected:.3f}", True)
                success = False
            else:
                self._log(f"Q formula check passed: {self.Q:.3f} ≈ {Q_expected:.3f}")
            
            # 5. Check Faraday's law relationships
            # Step 1: E0 = B*L*v0
            E0_expected = self.B * self.L * self.v0
            if not self._is_close(self.E0, E0_expected):
                self._log(f"E0 check failed: computed={self.E0:.3f}, expected={E0_expected:.3f}", True)
                success = False
            else:
                self._log(f"E0 check passed: {self.E0:.3f} ≈ {E0_expected:.3f}")
            
            # Step 7: E1 = B*L*v1
            E1_expected = self.B * self.L * self.v1
            if not self._is_close(self.E1, E1_expected):
                self._log(f"E1 check failed: computed={self.E1:.3f}, expected={E1_expected:.3f}", True)
                success = False
            else:
                self._log(f"E1 check passed: {self.E1:.3f} ≈ {E1_expected:.3f}")
            
            # 6. Check Ohm's law relationships
            # Step 2: I0 = E0/R
            I0_expected = self.E0 / self.R
            if not self._is_close(self.I0, I0_expected):
                self._log(f"I0 check failed: computed={self.I0:.3f}, expected={I0_expected:.3f}", True)
                success = False
            else:
                self._log(f"I0 check passed: {self.I0:.3f} ≈ {I0_expected:.3f}")
            
            # Step 8: I1 = E1/R
            I1_expected = self.E1 / self.R
            if not self._is_close(self.I1, I1_expected):
                self._log(f"I1 check failed: computed={self.I1:.3f}, expected={I1_expected:.3f}", True)
                success = False
            else:
                self._log(f"I1 check passed: {self.I1:.3f} ≈ {I1_expected:.3f}")
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", True)
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """Test how results change with input perturbations."""
        success = True
        
        try:
            # Store original values
            original_m = self.m
            
            # Perturb mass by +10%
            self.m = original_m * 1.1
            
            # Recalculate expected B using formula from step 4
            B_perturbed = (1/self.L) * math.sqrt(self.m * self.g * self.R / (2 * self.v0))
            
            # Check if B increases with increased mass (B ∝ sqrt(m))
            if B_perturbed <= self.B:
                self._log(f"Monotonicity check failed: B did not increase with mass", True)
                success = False
            else:
                self._log(f"Monotonicity check passed: B increases with mass")
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", True)
            success = False
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []  # Reset logs
        
        try:
            self._log("Starting PhysicsChecker tests...")
            
            # Run tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            # Overall success
            overall_success = test1 and test2 and test3
            
            # Summary
            self._log(f"\nTest summary:")
            self._log(f"  test_invariants: {'PASS' if test1 else 'FAIL'}")
            self._log(f"  test_trace_consistency: {'PASS' if test2 else 'FAIL'}")
            self._log(f"  test_monotonicity: {'PASS' if test3 else 'FAIL'}")
            self._log(f"  Overall: {'PASS' if overall_success else 'FAIL'}")
            
            return overall_success, self.logs
            
        except Exception as e:
            self._log(f"Fatal error in run_all_tests: {str(e)}\n{traceback.format_exc()}", True)
            return False, self.logs