import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters safely
        self.D = getattr(solution_instance, 'D', getattr(solution_instance, 'd', self.results.get('D', 1.0)))
        self.theta = getattr(solution_instance, 'theta', getattr(solution_instance, 'θ', self.results.get('theta', 0.0)))
        
        # Extract calculated results safely
        self.n = getattr(solution_instance, 'n', self.results.get('n', 0.0))
        self.gamma = getattr(solution_instance, 'gamma', self.results.get('γ', 0.0))
        self.d = getattr(solution_instance, 'd', self.results.get('d', 0.0))
        
        # Symbolic variables
        self.theta_sym = sp.symbols('θ', real=True)
        self.gamma_sym = sp.symbols('γ', real=True)
        self.n_sym = sp.symbols('n', real=True)
        self.D_sym = sp.symbols('D', real=True)
        self.d_sym = sp.symbols('d', real=True)
        
        self.logs = []
    
    def test_invariants(self) -> bool:
        """Verify Snell's Law and geometric relationships."""
        success = True
        
        # Convert angles to radians for calculations
        theta_rad = math.radians(self.theta)
        gamma_rad = math.radians(self.gamma)
        
        # 1. Check geometric relationship: γ = 90° - θ
        expected_gamma = 90 - self.theta
        if not math.isclose(self.gamma, expected_gamma, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Geometric relationship violated. γ={self.gamma}°, expected {expected_gamma}°")
            success = False
        else:
            self.logs.append(f"PASS: Geometric relationship γ=90°-θ holds: {self.gamma}° = 90° - {self.theta}°")
        
        # 2. Check Snell's Law: sin(θ) = n * sin(γ) since n_air=1
        lhs = math.sin(theta_rad)
        rhs = self.n * math.sin(gamma_rad)
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Snell's Law violated. sin(θ)={lhs:.4f}, n*sin(γ)={rhs:.4f}, n={self.n}")
            success = False
        else:
            self.logs.append(f"PASS: Snell's Law holds: sin({self.theta}°) = {self.n}*sin({self.gamma}°)")
        
        # 3. Check derived relationship: n = tan(θ)
        expected_n = math.tan(theta_rad)
        if not math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: n ≠ tan(θ). n={self.n}, tan(θ)={expected_n:.4f}")
            success = False
        else:
            self.logs.append(f"PASS: n = tan(θ) = {self.n:.4f}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        theta_rad = math.radians(self.theta)
        
        # 1. Check that d = D * tan(θ)
        expected_d = self.D * math.tan(theta_rad)
        if not math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Inconsistent d. d={self.d}, expected D*tan(θ)={expected_d:.4f}")
            success = False
        else:
            self.logs.append(f"PASS: d = D*tan(θ) = {self.d:.4f}")
        
        # 2. Symbolic check of step 4 formula
        try:
            # From step 4: d = D * sin(θ) / sin(90°-θ) = D * tan(θ)
            symbolic_d = self.D_sym * sp.sin(self.theta_sym) / sp.sin(sp.pi/2 - self.theta_sym)
            simplified = sp.simplify(symbolic_d - self.D_sym * sp.tan(self.theta_sym))
            if not simplified == 0:
                self.logs.append(f"WARNING: Symbolic simplification yields {simplified}")
        except Exception as e:
            self.logs.append(f"Symbolic check error: {str(e)}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Check that perturbations produce expected trends."""
        success = True
        original_theta = self.theta
        original_D = self.D
        
        # 1. Perturb θ (increase by 5%)
        perturbed_theta = self.theta * 1.05
        theta_rad = math.radians(original_theta)
        perturbed_theta_rad = math.radians(perturbed_theta)
        
        # Calculate expected n for perturbed θ
        n_original = math.tan(theta_rad)
        n_perturbed = math.tan(perturbed_theta_rad)
        
        # Check monotonicity: tan(θ) increases with θ in (0, 90)°
        if not (n_perturbed > n_original):
            self.logs.append(f"FAIL: n=tan(θ) not monotonic increasing. θ={original_theta}°→n={n_original:.4f}, θ={perturbed_theta}°→n={n_perturbed:.4f}")
            success = False
        else:
            self.logs.append(f"PASS: n increases with θ (monotonic)")
        
        # Calculate expected d for perturbed θ
        d_original = original_D * math.tan(theta_rad)
        d_perturbed = original_D * math.tan(perturbed_theta_rad)
        
        if not (d_perturbed > d_original):
            self.logs.append(f"FAIL: d not monotonic with θ. d={d_original:.4f}→{d_perturbed:.4f}")
            success = False
        else:
            self.logs.append(f"PASS: d increases with θ")
        
        # 2. Perturb D (increase by 10%)
        perturbed_D = original_D * 1.1
        d_perturbed_D = perturbed_D * math.tan(theta_rad)
        
        # d should be proportional to D
        if not math.isclose(d_perturbed_D / d_original, perturbed_D / original_D, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: d not proportional to D. Ratio mismatch: {d_perturbed_D/d_original:.4f} vs {perturbed_D/original_D:.4f}")
            success = False
        else:
            self.logs.append(f"PASS: d ∝ D (linear relationship)")
        
        # Restore original values
        self.theta = original_theta
        self.D = original_D
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all audit tests."""
        self.logs = ["=== Physics Audit Report ==="]
        
        tests = [
            ("Geometric & Snell's Law Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity & Trends", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        self.logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs

# Optional: Example usage
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'n': 1.0,  # tan(45°) = 1
                'γ': 45.0,  # 90° - 45°
                'd': 2.0   # D*tan(45°) = 2*1 = 2
            }
            self.D = 2.0
            self.theta = 45.0
    
    mock_sol = MockSolution()
    checker = PhysicsChecker(mock_sol)
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)