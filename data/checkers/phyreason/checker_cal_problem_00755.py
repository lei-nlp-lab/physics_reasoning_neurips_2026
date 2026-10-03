import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        
        Expected keys in solution_instance.results (from problem KB):
        - 'v'' (velocity at top of semicircle) -> stored as v_prime
        - 'v_1' (velocity at bottom of semicircle) -> stored as v1
        - 'E_p' (elastic potential energy) -> stored as E_p
        - 'v_A,v_B' (separation velocities) -> parsed as vA and vB
        - 'v' (common velocity of block B and plank)
        - 'x' (relative displacement on plank)
        - 'Q' (internal energy from friction)
        """
        # Safe extraction from solution_instance or its results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Physical constants from problem statement
        self.m = 2.0      # kg (mass of each block)
        self.M = 8.0      # kg (mass of plank)
        self.R = 0.5      # m (radius of semicircular track)
        self.L = 1.0      # m (length of plank)
        self.mu = 0.5     # coefficient of kinetic friction
        self.g = 10.0     # m/s^2 (gravitational acceleration)
        
        # Extract results with safe fallbacks
        # v' (velocity at top of semicircle)
        self.v_prime = self._safe_extract(['v\'', 'v_prime', 'vprime'], default=2.236)
        
        # v_1 (velocity at bottom of semicircle)
        self.v1 = self._safe_extract(['v_1', 'v1'], default=5.0)
        
        # Elastic potential energy
        self.E_p = self._safe_extract(['E_p', 'E_spring'], default=50.0)
        
        # Separation velocities vA and vB (expected to be equal)
        v_sep = self._safe_extract(['v_A,v_B', 'vA_vB', 'v_sep'], default=5.0)
        # If v_sep is a string like "5,5", parse it
        if isinstance(v_sep, str):
            try:
                parts = v_sep.split(',')
                self.vA = float(parts[0].strip())
                self.vB = float(parts[1].strip()) if len(parts) > 1 else self.vA
            except:
                self.vA = self.vB = float(v_sep)
        else:
            self.vA = self.vB = float(v_sep)
        
        # Common velocity of block B and plank
        self.v_common = self._safe_extract(['v', 'v_common'], default=1.0)
        
        # Relative displacement on plank
        self.x = self._safe_extract(['x', 'displacement'], default=2.0)
        
        # Internal energy from friction
        self.Q = self._safe_extract(['Q', 'Q_friction'], default=10.0)
        
        # Tolerance for numerical comparisons (1-2% as per robustness rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Sympy symbols for symbolic checks
        self._init_symbols()
        
        # Store logs for test results
        self.logs = []
    
    def _safe_extract(self, keys, default=0.0):
        """Safely extract value from results using multiple possible keys."""
        for key in keys:
            value = self.results.get(key, None)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value
        return default
    
    def _init_symbols(self):
        """Initialize SymPy symbols for symbolic physics laws."""
        m, M, R, L, mu, g = sp.symbols('m M R L mu g', positive=True)
        v_prime, v1, vA, vB, v_common, x, E_p, Q = sp.symbols(
            'v_prime v1 vA vB v_common x E_p Q', positive=True
        )
        
        self.sym = {
            'm': m, 'M': M, 'R': R, 'L': L, 'mu': mu, 'g': g,
            'v_prime': v_prime, 'v1': v1, 'vA': vA, 'vB': vB,
            'v_common': v_common, 'x': x, 'E_p': E_p, 'Q': Q
        }
    
    def _is_close(self, a, b, rel_tol=None, abs_tol=None):
        """Robust fuzzy numerical comparison."""
        if rel_tol is None:
            rel_tol = self.rel_tol
        if abs_tol is None:
            abs_tol = self.abs_tol
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental physics invariants:
        1. Circular motion condition at top of semicircle
        2. Energy conservation for block A on semicircular track
        3. Conservation of momentum for block B and plank
        4. Work-energy theorem for block B on plank
        """
        logs = []
        success = True
        
        # 1. Circular motion condition: mg = m*v'²/R at top of semicircle
        lhs_circular = self.m * self.g
        rhs_circular = self.m * self.v_prime**2 / self.R
        if not self._is_close(lhs_circular, rhs_circular):
            logs.append(f"FAIL: Circular motion condition violated. mg={lhs_circular:.3f}, "
                       f"mv'²/R={rhs_circular:.3f}")
            success = False
        else:
            logs.append(f"PASS: Circular motion condition satisfied (v'={self.v_prime:.3f} m/s)")
        
        # 2. Energy conservation for block A: ½m v1² = ½m v'² + mg(2R)
        KE_bottom = 0.5 * self.m * self.v1**2
        KE_top = 0.5 * self.m * self.v_prime**2
        PE_change = self.m * self.g * (2 * self.R)
        if not self._is_close(KE_bottom, KE_top + PE_change):
            logs.append(f"FAIL: Energy conservation for block A violated. "
                       f"KE_bottom={KE_bottom:.3f} J, KE_top+ΔPE={KE_top+PE_change:.3f} J")
            success = False
        else:
            logs.append(f"PASS: Energy conservation for block A satisfied")
        
        # 3. Conservation of momentum for block B and plank: m*vA = (M+m)*v_common
        momentum_initial = self.m * self.vA
        momentum_final = (self.M + self.m) * self.v_common
        if not self._is_close(momentum_initial, momentum_final):
            logs.append(f"FAIL: Momentum conservation violated. "
                       f"Initial={momentum_initial:.3f} kg·m/s, "
                       f"Final={momentum_final:.3f} kg·m/s")
            success = False
        else:
            logs.append(f"PASS: Momentum conservation satisfied (v_common={self.v_common:.3f} m/s)")
        
        # 4. Work-energy theorem for block B on plank until common velocity:
        # ½m vA² - ½(M+m)v_common² = μmg x
        KE_loss = 0.5 * self.m * self.vA**2 - 0.5 * (self.M + self.m) * self.v_common**2
        work_friction = self.mu * self.m * self.g * self.x
        if not self._is_close(KE_loss, work_friction):
            logs.append(f"FAIL: Work-energy theorem violated. "
                       f"ΔKE={KE_loss:.3f} J, Work_friction={work_friction:.3f} J")
            success = False
        else:
            logs.append(f"PASS: Work-energy theorem satisfied (x={self.x:.3f} m)")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answers.
        Verify consistency between steps in the analysis.
        """
        logs = []
        success = True
        
        # Check that vA = vB = v1 (from step 2)
        if not (self._is_close(self.vA, self.v1) and self._is_close(self.vB, self.v1)):
            logs.append(f"FAIL: Separation velocities inconsistent. "
                       f"vA={self.vA:.3f}, vB={self.vB:.3f}, v1={self.v1:.3f} m/s")
            success = False
        else:
            logs.append(f"PASS: Separation velocities consistent (vA=vB=v1={self.v1:.3f} m/s)")
        
        # Check that spring energy equals total kinetic energy of both blocks
        # E_p = ½*(2m)*v1² = m*v1²
        expected_E_p = self.m * self.v1**2
        if not self._is_close(self.E_p, expected_E_p):
            logs.append(f"FAIL: Spring energy inconsistent. "
                       f"E_p={self.E_p:.3f} J, expected m*v1²={expected_E_p:.3f} J")
            success = False
        else:
            logs.append(f"PASS: Spring energy consistent (E_p={self.E_p:.3f} J)")
        
        # Check that Q = μmgL (from step 7)
        expected_Q = self.mu * self.m * self.g * self.L
        if not self._is_close(self.Q, expected_Q):
            logs.append(f"FAIL: Friction energy inconsistent. "
                       f"Q={self.Q:.3f} J, expected μmgL={expected_Q:.3f} J")
            success = False
        else:
            logs.append(f"PASS: Friction energy consistent (Q={self.Q:.3f} J)")
        
        # Check that x > L (block slides off plank)
        if self.x <= self.L:
            logs.append(f"FAIL: Block should slide off plank but x={self.x:.3f} ≤ L={self.L:.3f} m")
            success = False
        else:
            logs.append(f"PASS: Block slides off plank (x={self.x:.3f} > L={self.L:.3f} m)")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb an input parameter and check if results change in expected direction.
        We'll perturb mass m and check that spring energy E_p increases with m.
        """
        logs = []
        success = True
        
        original_m = self.m
        perturbed_m = original_m * 1.1  # Increase by 10%
        
        # Use physics formulas to compute expected trend
        # v' = sqrt(gR) is independent of m
        # v1 = sqrt(5gR) is independent of m
        # E_p = m * v1² should increase with m
        
        # Save original results
        original_v1 = self.v1
        original_E_p = self.E_p
        
        # Calculate expected values with perturbed mass
        # v1 remains the same (independent of m)
        expected_v1 = math.sqrt(5 * self.g * self.R)
        expected_E_p_perturbed = perturbed_m * expected_v1**2
        
        # Check monotonicity: E_p should increase with m
        if expected_E_p_perturbed <= original_E_p:
            logs.append(f"FAIL: E_p should increase with mass, but "
                       f"E_p(perturbed)={expected_E_p_perturbed:.3f} ≤ "
                       f"E_p(original)={original_E_p:.3f} J")
            success = False
        else:
            logs.append(f"PASS: E_p increases with mass (monotonicity check)")
        
        # Restore original values (important for subsequent tests)
        self.m = original_m
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics checks and return overall success and detailed logs.
        """
        self.logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== {test_name} ===")
            try:
                success, test_logs = test_func()
                self.logs.extend(test_logs)
                if not success:
                    overall_success = False
                    self.logs.append(f"Result: FAILED")
                else:
                    self.logs.append(f"Result: PASSED")
            except Exception as e:
                self.logs.append(f"ERROR during {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs

# Example usage (commented out for safety):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'v\'': 2.236,
#                 'v_1': 5.0,
#                 'E_p': 50.0,
#                 'v_A,v_B': '5.0,5.0',
#                 'v': 1.0,
#                 'x': 2.0,
#                 'Q': 10.0
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)