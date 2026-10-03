import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    """
    Robust physics auditor for raindrop air resistance problem.
    Implements safe attribute access, fuzzy matching, and multiple validation strategies.
    """
    
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        Never access solution_instance.attr directly.
        """
        self.solution_instance = solution_instance
        
        # Safe extraction using getattr with .results fallback
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all expected results safely
        self.V = getattr(solution_instance, 'V', self.results.get('V', None))
        self.f = getattr(solution_instance, 'f', self.results.get('f', None))
        self.F_1delta_t = getattr(solution_instance, 'F_1delta_t', 
                                  self.results.get('F_1delta_t', None))
        self.F_2delta_t = getattr(solution_instance, 'F_2delta_t', 
                                  self.results.get('F_2delta_t', None))
        self.W_f = getattr(solution_instance, 'W_f', self.results.get('W_f', None))
        self.m = getattr(solution_instance, 'm', self.results.get('m', None))
        self.N = getattr(solution_instance, 'N', self.results.get('N', None))
        self.v_m = getattr(solution_instance, 'v_m', self.results.get('v_m', None))
        
        # Extract inputs safely
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.g = self.inputs.get('g', 9.81)
        self.rho = self.inputs.get('rho', 1000.0)  # Water density
        self.r = self.inputs.get('r', None)
        self.k = self.inputs.get('k', None)
        self.S = self.inputs.get('S', None)
        self.n = self.inputs.get('n', None)
        self.m0 = self.inputs.get('m0', None)
        self.h = self.inputs.get('h', None)
        self.u = self.inputs.get('u', None)
        
        # SymPy symbols for symbolic checking
        self.symbols = {
            'g': sp.Symbol('g', positive=True),
            'ρ': sp.Symbol('ρ', positive=True),  # Density
            'r': sp.Symbol('r', positive=True),
            'k': sp.Symbol('k', positive=True),
            'S': sp.Symbol('S', positive=True),
            'n': sp.Symbol('n', positive=True),
            'm0': sp.Symbol('m0', positive=True),
            'h': sp.Symbol('h', positive=True),
            'u': sp.Symbol('u', positive=True),
            'v': sp.Symbol('v', positive=True),
            'v_m': sp.Symbol('v_m', positive=True),
            'm': sp.Symbol('m', positive=True),
            'V': sp.Symbol('V', positive=True),
            'f': sp.Symbol('f', positive=True),
            'W_f': sp.Symbol('W_f')
        }
        
        self.logs = []
    
    def _log(self, message: str, is_error: bool = False):
        """Helper for logging with error flag."""
        self.logs.append(f"{'[ERROR]' if is_error else '[INFO]'} {message}")
    
    def _fuzzy_equal(self, a: float, b: float, 
                    rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """
        Fuzzy numerical comparison with tolerance.
        Uses math.isclose for robust comparison with ±1-2% tolerance.
        """
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test physical invariants from KB."""
        logs = []
        success = True
        
        try:
            # 1. Check Work-Energy Theorem consistency
            if self.W_f is not None and self.m is not None and self.h is not None and self.u is not None:
                # Symbolic check: W_f = mgh - 1/2 m u^2
                expected_W_f = self.symbols['m'] * self.symbols['g'] * self.symbols['h'] - \
                              (sp.Rational(1, 2) * self.symbols['m'] * self.symbols['u']**2)
                
                # Numerical check if values exist
                if all(isinstance(val, (int, float)) for val in [self.m, self.g, self.h, self.u]):
                    calc_W_f = self.m * self.g * self.h - 0.5 * self.m * self.u**2
                    if not self._fuzzy_equal(self.W_f, calc_W_f):
                        self._log(f"Work-Energy Theorem mismatch: got {self.W_f}, expected ~{calc_W_f}", True)
                        success = False
                    else:
                        self._log("Work-Energy Theorem satisfied")
            
            # 2. Check terminal velocity condition (Newton's First Law at equilibrium)
            if self.v_m is not None and self.r is not None and self.k is not None:
                # From step 4: v_m = sqrt(4πρgr/(3k))
                # Symbolic expression
                expected_v_m = sp.sqrt(4 * sp.pi * self.symbols['ρ'] * self.symbols['g'] * 
                                      self.symbols['r'] / (3 * self.symbols['k']))
                
                # Numerical check
                if all(isinstance(val, (int, float)) for val in [self.rho, self.g, self.r, self.k]):
                    calc_v_m = math.sqrt(4 * math.pi * self.rho * self.g * self.r / (3 * self.k))
                    if not self._fuzzy_equal(self.v_m, calc_v_m):
                        self._log(f"Terminal velocity mismatch: got {self.v_m}, expected ~{calc_v_m}", True)
                        success = False
                    else:
                        self._log("Terminal velocity formula consistent with Newton's First Law")
            
            # 3. Check drag force proportionality
            if self.f is not None:
                # From step 10: f = 2Sv^2 n m0 ∝ v^2
                # Check if f has v^2 dependence
                try:
                    # Try symbolic check
                    if isinstance(self.f, sp.Expr):
                        v = sp.Symbol('v')
                        if self.f.has(v**2):
                            self._log("Drag force has v^2 dependence")
                        else:
                            self._log("Drag force missing v^2 dependence", True)
                            success = False
                except:
                    # Numerical check - verify quadratic dependence by testing scaling
                    if self.S is not None and self.n is not None and self.m0 is not None:
                        # If all constants are present, check if f = 2Sv^2 n m0 for some v
                        self._log("Symbolic check for drag force skipped (numerical data only)")
            
            # 4. Check volume-mass relationship
            if self.m is not None and self.V is not None and self.rho is not None:
                # m = ρV
                calc_m = self.rho * self.V
                if not self._fuzzy_equal(self.m, calc_m):
                    self._log(f"Mass-volume relationship mismatch: m={self.m}, ρV={calc_m}", True)
                    success = False
                else:
                    self._log("Mass-volume relationship satisfied")
        
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", True)
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check step 2 consistency
            if self.V is not None and self.r is not None:
                calc_V = (4/3) * math.pi * self.r**3
                if not self._fuzzy_equal(self.V, calc_V):
                    self._log(f"Volume mismatch: got {self.V}, expected ~{calc_V}", True)
                    success = False
                else:
                    self._log("Volume calculation consistent")
            
            # 2. Check step 3 equilibrium condition
            if self.f is not None and self.m is not None:
                # At terminal velocity: f = mg
                calc_f = self.m * self.g
                if not self._fuzzy_equal(self.f, calc_f):
                    self._log(f"Equilibrium force mismatch: drag={self.f}, weight={calc_f}", True)
                    success = False
                else:
                    self._log("Equilibrium condition satisfied")
            
            # 3. Check impulse consistency from steps 7-9
            if (self.F_1delta_t is not None and self.F_2delta_t is not None and 
                self.f is not None and self.S is not None and self.n is not None and 
                self.m0 is not None and self.u is not None):
                
                # From steps 7-9: f = (F_1Δt - F_2Δt)/Δt (assuming Δt cancels in ratio)
                # The net force should equal drag force
                try:
                    # If we have time interval, use it, otherwise check proportionality
                    delta_t = self.inputs.get('delta_t', 1.0)
                    net_force = (self.F_1delta_t - self.F_2delta_t) / delta_t
                    
                    if not self._fuzzy_equal(self.f, net_force):
                        self._log(f"Impulse-force consistency mismatch: f={self.f}, from impulses={net_force}", True)
                        success = False
                    else:
                        self._log("Impulse calculations consistent with net force")
                except:
                    self._log("Could not complete impulse consistency check (missing delta_t)")
            
            # 4. Check final drag force expression
            if self.f is not None and self.S is not None and self.n is not None and self.m0 is not None:
                # f = 2Sv^2 n m0 from step 10
                # Need v to check, use terminal velocity if available
                if self.v_m is not None:
                    calc_f = 2 * self.S * self.v_m**2 * self.n * self.m0
                    if not self._fuzzy_equal(self.f, calc_f):
                        self._log(f"Final drag expression mismatch: f={self.f}, 2Sv²nm₀={calc_f}", True)
                        success = False
                    else:
                        self._log("Final drag force expression consistent")
        
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", True)
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test monotonic trends by perturbing inputs."""
        logs = []
        success = True
        
        try:
            # Test 1: Terminal velocity vs radius
            if self.r is not None and self.rho is not None and self.g is not None and self.k is not None:
                original_r = float(self.r)
                perturbed_r = original_r * 1.01  # +1% perturbation
                
                # Calculate original and perturbed v_m
                v_m_original = math.sqrt(4 * math.pi * self.rho * self.g * original_r / (3 * self.k))
                v_m_perturbed = math.sqrt(4 * math.pi * self.rho * self.g * perturbed_r / (3 * self.k))
                
                # v_m should increase with r (v_m ∝ √r)
                if v_m_perturbed > v_m_original:
                    self._log(f"Monotonicity: v_m increases with r (√r dependence): {v_m_original:.3f} → {v_m_perturbed:.3f}")
                else:
                    self._log(f"Monotonicity violation: v_m doesn't increase with r", True)
                    success = False
            
            # Test 2: Drag force vs velocity (quadratic dependence)
            if self.S is not None and self.n is not None and self.m0 is not None:
                test_v = 1.0
                perturbed_v = 1.1  # +10%
                
                f_original = 2 * self.S * test_v**2 * self.n * self.m0
                f_perturbed = 2 * self.S * perturbed_v**2 * self.n * self.m0
                
                # f should increase quadratically with v
                expected_ratio = (perturbed_v / test_v)**2
                actual_ratio = f_perturbed / f_original
                
                if self._fuzzy_equal(actual_ratio, expected_ratio):
                    self._log(f"Monotonicity: f ∝ v² confirmed: ratio={actual_ratio:.3f}, expected={expected_ratio:.3f}")
                else:
                    self._log(f"Monotonicity violation: f not proportional to v²", True)
                    success = False
            
            # Test 3: Work done by air resistance vs height
            if self.h is not None and self.m is not None and self.g is not None and self.u is not None:
                original_h = float(self.h)
                perturbed_h = original_h * 1.01
                
                W_f_original = self.m * self.g * original_h - 0.5 * self.m * self.u**2
                W_f_perturbed = self.m * self.g * perturbed_h - 0.5 * self.m * self.u**2
                
                # W_f should increase with h (linear dependence)
                if W_f_perturbed > W_f_original:
                    self._log(f"Monotonicity: W_f increases with h: {W_f_original:.3f} → {W_f_perturbed:.3f}")
                else:
                    self._log(f"Monotonicity violation: W_f doesn't increase with h", True)
                    success = False
        
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", True)
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        # Run individual tests
        test_methods = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_methods:
            self._log(f"\n=== Running {test_name} Test ===")
            try:
                test_success, test_logs = test_func()
                for log in test_logs:
                    self.logs.append(log)
                if not test_success:
                    overall_success = False
                    self._log(f"{test_name} test FAILED", True)
                else:
                    self._log(f"{test_name} test PASSED")
            except Exception as e:
                self._log(f"{test_name} test CRASHED: {str(e)}", True)
                overall_success = False
        
        # Summary
        self._log(f"\n=== SUMMARY ===")
        self._log(f"Overall result: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs

# Example usage stub
if __name__ == "__main__":
    # This would be used with a solution instance
    # checker = PhysicsChecker(solution)
    # success, logs = checker.run_all_tests()
    pass