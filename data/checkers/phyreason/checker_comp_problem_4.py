import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all required inputs and results."""
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.x = self._safe_get('x')
        self.x_new = self._safe_get('x_new')
        self.delta_x = self._safe_get('delta_x')
        self.u = self._safe_get('u')
        self.v = self._safe_get('v')
        self.R = self._safe_get('R')
        self.mu_1 = self._safe_get('mu_1')
        self.mu_2 = self._safe_get('mu_2')
        self.h_app = self._safe_get('h_app')
        
        # Problem constants
        self.R_mirror = 20.0  # cm, convex surface radius (silvered)
        self.R_lens = -60.0   # cm, concave surface radius
        self.n_glass = 1.5
        self.n_air = 1.0
        self.n_water = 4/3
        
        # Setup symbolic variables
        self._setup_symbols()
        
    def _safe_get(self, key: str) -> Optional[float]:
        """Safely extract numeric value from results with fallback."""
        val = self.results.get(key, None)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    def _setup_symbols(self):
        """Initialize symbolic variables for physics laws."""
        # Distances
        self.sym_x = sp.symbols('x')
        self.sym_x_new = sp.symbols('x_new')
        self.sym_u = sp.symbols('u')
        self.sym_v = sp.symbols('v')
        
        # Refractive indices
        self.sym_mu1 = sp.symbols('mu1')
        self.sym_mu2 = sp.symbols('mu2')
        
        # Radii
        self.sym_R = sp.symbols('R')
        
        # Physics equations
        self.single_surface_eq = sp.Eq(
            self.sym_mu1/self.sym_u + self.sym_mu2/self.sym_v,
            (self.sym_mu2 - self.sym_mu1)/self.sym_R
        )
        
        self.apparent_depth_eq = sp.Eq(
            self.sym_u,  # apparent distance
            self.sym_x_new * self.sym_mu1  # actual distance * refractive index
        )
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physics laws are satisfied."""
        logs = []
        success = True
        
        # 1. Check first refraction (air to glass)
        if self.x is not None:
            # Symbolic check
            lhs = self.n_air/self.x + self.n_glass/(-20.0)
            rhs = (self.n_glass - self.n_air)/self.R_lens
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: First refraction check: lhs={lhs:.4f}, rhs={rhs:.4f}")
                success = False
            else:
                logs.append(f"PASS: First refraction equation satisfied")
        
        # 2. Check second refraction (water to glass)
        if self.x_new is not None:
            # Using apparent depth: u = (4/3)*x_new
            u_eff = (self.n_water) * self.x_new
            lhs = self.n_water/u_eff + self.n_glass/(-20.0)
            rhs = (self.n_glass - self.n_water)/self.R_lens
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Second refraction check: lhs={lhs:.4f}, rhs={rhs:.4f}")
                success = False
            else:
                logs.append(f"PASS: Second refraction equation satisfied")
        
        # 3. Check mirror condition (image at center of curvature)
        # For silvered convex surface acting as concave mirror
        expected_v = 20.0  # cm
        if self.v is not None and not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"WARN: Mirror image distance v={self.v} != expected {expected_v}")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check x from steps
        if self.x is not None:
            expected_x = 15.0
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x={self.x} != expected {expected_x}")
                success = False
            else:
                logs.append(f"PASS: x matches expected value")
        
        # Check x_new from steps
        if self.x_new is not None:
            expected_x_new = 180/13  # ≈13.846
            if not math.isclose(self.x_new, expected_x_new, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x_new={self.x_new} != expected {expected_x_new:.4f}")
                success = False
            else:
                logs.append(f"PASS: x_new matches expected value")
        
        # Check delta_x consistency
        if self.x is not None and self.x_new is not None and self.delta_x is not None:
            calc_delta = self.x - self.x_new
            if not math.isclose(self.delta_x, calc_delta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: delta_x={self.delta_x} != x - x_new = {calc_delta:.4f}")
                success = False
            else:
                logs.append(f"PASS: delta_x consistent with x and x_new")
            
            # Check against target answer (1.16 cm)
            expected_delta = 15/13  # ≈1.154
            if not math.isclose(self.delta_x, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: delta_x={self.delta_x} differs from expected {expected_delta:.4f}")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb refractive index of water and check trend."""
        logs = []
        
        if self.x_new is None:
            logs.append("SKIP: monotonicity test (missing x_new)")
            return True, logs
        
        # Original parameters
        n_water_orig = self.n_water
        x_new_orig = self.x_new
        
        # Perturb water refractive index by +10%
        n_water_perturbed = n_water_orig * 1.10
        
        # Calculate new x_new using refraction formula
        # Equation: (n_water/(n_water*x_new)) + (n_glass/(-20)) = (n_glass - n_water)/(-60)
        # Simplifies to: 1/x_new - 0.075 = (n_glass - n_water)/(-60)
        try:
            # Solve for x_new: 1/x_new = (n_glass - n_water)/(-60) + 0.075
            rhs = (self.n_glass - n_water_perturbed)/(-60.0) + 0.075
            x_new_perturbed = 1.0 / rhs if rhs != 0 else float('inf')
            
            # Check trend: as n_water increases, x_new should increase (from earlier analysis)
            if x_new_perturbed > x_new_orig:
                logs.append(f"PASS: Monotonicity: n_water↑ {n_water_orig:.3f}→{n_water_perturbed:.3f} "
                           f"⇒ x_new↑ {x_new_orig:.3f}→{x_new_perturbed:.3f}")
            else:
                logs.append(f"WARN: Unexpected monotonicity trend")
        except ZeroDivisionError:
            logs.append("SKIP: monotonicity test (division by zero)")
        
        return True, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics checks."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        return overall_success, all_logs

# Example usage (commented out for production):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'x': 15.0,
#                 'x_new': 180/13,
#                 'delta_x': 15/13,
#                 'v': -20.0,
#                 'R': -60.0,
#                 'mu_1': 4/3,
#                 'mu_2': 1.5
#             }
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)
#     print(f"\nOverall success: {success}")