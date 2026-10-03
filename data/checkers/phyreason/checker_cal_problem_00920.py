import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all required values with safe defaults
        self.c = getattr(solution_instance, 'c', self.results.get('c', 3.0e8))
        self.lam = getattr(solution_instance, 'lambda', self.results.get('lambda', 2.0e-2))
        self.x1 = getattr(solution_instance, 'x_1', self.results.get('x_1', 0.0))
        self.x2 = getattr(solution_instance, 'x_2', self.results.get('x_2', 0.0))
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.delta_x = getattr(solution_instance, 'Deltax', self.results.get('Deltax', 0.0))
        
        # Problem constants from context
        self.t1 = 2.0e-4  # s
        self.t2 = 1.98e-4  # s
        self.dt_emission = 0.8  # s between emissions
        
        # Store logs
        self.logs = []
        
    def log(self, message: str):
        """Add message to logs"""
        self.logs.append(message)
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics laws"""
        success = True
        
        # 1. Check speed of light constant
        if not math.isclose(self.c, 3.0e8, rel_tol=1e-2, abs_tol=1e2):
            self.log(f"FAIL: Speed of light mismatch. Expected ~3e8, got {self.c}")
            success = False
        else:
            self.log(f"PASS: Speed of light correct: {self.c:.2e} m/s")
        
        # 2. Check distance calculations using c = distance/time
        # For first pulse: distance = c * t1 / 2
        x1_expected = self.c * self.t1 / 2
        if not math.isclose(self.x1, x1_expected, rel_tol=1e-2, abs_tol=100):
            self.log(f"FAIL: x1 calculation. Expected {x1_expected:.1f}, got {self.x1:.1f}")
            success = False
        else:
            self.log(f"PASS: x1 = c*t1/2 = {self.x1:.1f} m")
        
        # For second pulse: distance = c * t2 / 2
        x2_expected = self.c * self.t2 / 2
        if not math.isclose(self.x2, x2_expected, rel_tol=1e-2, abs_tol=100):
            self.log(f"FAIL: x2 calculation. Expected {x2_expected:.1f}, got {self.x2:.1f}")
            success = False
        else:
            self.log(f"PASS: x2 = c*t2/2 = {self.x2:.1f} m")
        
        # 3. Check displacement consistency
        delta_x_expected = x1_expected - x2_expected
        if not math.isclose(self.delta_x, delta_x_expected, rel_tol=1e-2, abs_tol=10):
            self.log(f"FAIL: Δx mismatch. Expected {delta_x_expected:.1f}, got {self.delta_x:.1f}")
            success = False
        else:
            self.log(f"PASS: Δx = x1 - x2 = {self.delta_x:.1f} m")
        
        # 4. Check velocity calculation
        # Time between reflections: Δt = 0.8 + t2/2 - t1/2
        dt_reflection = self.dt_emission + self.t2/2 - self.t1/2
        v_expected = delta_x_expected / dt_reflection
        
        if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=5):
            self.log(f"FAIL: Velocity calculation. Expected {v_expected:.1f}, got {self.v:.1f}")
            success = False
        else:
            self.log(f"PASS: v = Δx/Δt = {self.v:.1f} m/s")
        
        # 5. Check wavelength positivity (basic sanity)
        if self.lam <= 0:
            self.log(f"FAIL: Wavelength must be positive, got {self.lam}")
            success = False
        else:
            self.log(f"PASS: Wavelength positive: {self.lam:.2e} m")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check intermediate results match final answer"""
        success = True
        
        # Check if all computed distances are consistent
        if not math.isclose(self.delta_x, self.x1 - self.x2, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Δx ≠ x1 - x2. {self.delta_x} vs {self.x1 - self.x2}")
            success = False
        else:
            self.log("PASS: Δx = x1 - x2 (consistent)")
        
        # Check if velocity matches displacement over correct time
        dt_calc = self.dt_emission + self.t2/2 - self.t1/2
        v_from_delta = self.delta_x / dt_calc
        
        if not math.isclose(self.v, v_from_delta, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: v ≠ Δx/Δt. {self.v} vs {v_from_delta}")
            success = False
        else:
            self.log(f"PASS: v = Δx/Δt (consistent, Δt = {dt_calc:.4f} s)")
        
        # Check wavelength is reasonable for radar (typically cm to m range)
        if self.lam < 1e-3 or self.lam > 10:
            self.log(f"WARNING: Unusual radar wavelength: {self.lam:.2e} m")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb time measurements and check velocity trend.
        If t2 decreases (plane closer), velocity should be positive.
        """
        success = True
        
        # Store original values
        orig_t2 = self.t2
        
        # Perturb: make t2 slightly smaller (plane moving faster toward radar)
        perturb_t2 = self.t2 * 0.95  # 5% smaller
        perturb_x2 = self.c * perturb_t2 / 2
        perturb_delta_x = self.x1 - perturb_x2
        perturb_dt = self.dt_emission + perturb_t2/2 - self.t1/2
        perturb_v = perturb_delta_x / perturb_dt
        
        # Check monotonicity: smaller t2 → larger velocity
        if perturb_v <= self.v:
            self.log(f"FAIL: Monotonicity violated. t2↓ should give v↑, but {perturb_v:.1f} <= {self.v:.1f}")
            success = False
        else:
            self.log(f"PASS: Monotonicity: t2↓ ({perturb_t2:.2e}) → v↑ ({perturb_v:.1f} > {self.v:.1f})")
        
        # Restore original value
        self.t2 = orig_t2
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status with logs"""
        self.logs = []
        self.log("=== Physics Checker Started ===")
        
        # Run tests
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Check", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    all_passed = False
            except Exception as e:
                self.log(f"ERROR in {test_name}: {str(e)}")
                all_passed = False
        
        # Final summary
        self.log("\n=== Summary ===")
        if all_passed:
            self.log("✓ ALL TESTS PASSED")
        else:
            self.log("✗ SOME TESTS FAILED")
        
        return all_passed, self.logs

# Optional: Symbolic verification (not required for this problem)
def symbolic_verification():
    """Symbolic check of the relationships"""
    c, t1, t2, dt = sp.symbols('c t1 t2 dt', positive=True)
    x1 = c * t1 / 2
    x2 = c * t2 / 2
    delta_x = x1 - x2
    v = delta_x / (dt + t2/2 - t1/2)
    
    # Verify v is positive when t1 > t2 (approaching)
    cond = sp.simplify(v.subs({t1: 2e-4, t2: 1.98e-4, dt: 0.8, c: 3e8}))
    return cond

if __name__ == "__main__":
    # Example usage with dummy solution class
    class DummySolution:
        def __init__(self):
            self.results = {
                'c': 3.0e8,
                'lambda': 2.0e-2,
                'x_1': 30000.0,  # c*t1/2 = 3e8 * 2e-4 / 2 = 30000
                'x_2': 29700.0,  # c*t2/2 = 3e8 * 1.98e-4 / 2 = 29700
                'Deltax': 300.0,
                'v': 375.0
            }
    
    solution = DummySolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)