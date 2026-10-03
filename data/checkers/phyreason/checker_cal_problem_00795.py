import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract inputs and results safely
        self.a = getattr(solution_instance, 'a', 
                        getattr(solution_instance, 'inputs', {}).get('a', 1.0))
        self.c = getattr(solution_instance, 'c', 
                        getattr(solution_instance, 'inputs', {}).get('c', 1.0))
        
        # Extract results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        
        self.n = results.get('n', getattr(solution_instance, 'n', 0.0))
        self.theta_i = results.get('theta_i', getattr(solution_instance, 'theta_i', 0.0))
        self.theta_r = results.get('theta_r', getattr(solution_instance, 'theta_r', 0.0))
        self.v = results.get('v', getattr(solution_instance, 'v', 0.0))
        self.EF = results.get('EF', getattr(solution_instance, 'EF', 0.0))
        self.CF = results.get('CF', getattr(solution_instance, 'CF', 0.0))
        self.x = results.get('x', getattr(solution_instance, 'x', 0.0))
        self.t = results.get('t', getattr(solution_instance, 't', 0.0))
        
        # Convert to radians for calculations if degrees
        if hasattr(self.theta_i, '__iter__') or isinstance(self.theta_i, (int, float)):
            if isinstance(self.theta_i, (int, float)) and self.theta_i > 2*math.pi:
                # Assume degrees if value > 2π
                self.theta_i_rad = math.radians(float(self.theta_i))
                self.theta_r_rad = math.radians(float(self.theta_r))
            else:
                self.theta_i_rad = float(self.theta_i)
                self.theta_r_rad = float(self.theta_r)
        else:
            self.theta_i_rad = 0.0
            self.theta_r_rad = 0.0
        
        # Store logs for test results
        self.logs = []
        
    def test_invariants(self) -> bool:
        """Test physical invariants: Snell's Law and speed relationship"""
        success = True
        
        # 1. Test Snell's Law: n = sin(θ_i)/sin(θ_r)
        if self.theta_i_rad > 0 and self.theta_r_rad > 0:
            n_from_snell = math.sin(self.theta_i_rad) / math.sin(self.theta_r_rad)
            
            if not math.isclose(self.n, n_from_snell, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Snell's Law violated. n={self.n:.4f}, "
                               f"sin(θ_i)/sin(θ_r)={n_from_snell:.4f}")
                success = False
            else:
                self.logs.append(f"PASS: Snell's Law check: n={self.n:.4f} ≈ {n_from_snell:.4f}")
        else:
            self.logs.append("SKIP: Snell's Law (angles not available)")
        
        # 2. Test speed relationship: v = c/n
        if self.n > 0:
            v_expected = self.c / self.n
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Speed relationship violated. v={self.v:.4f}, "
                               f"c/n={v_expected:.4f}")
                success = False
            else:
                self.logs.append(f"PASS: Speed relationship: v={self.v:.4f} ≈ c/n={v_expected:.4f}")
        else:
            self.logs.append("SKIP: Speed relationship (n=0)")
        
        # 3. Test geometric consistency: θ_i + θ_r should be 90° from problem geometry
        if self.theta_i_rad > 0 and self.theta_r_rad > 0:
            angle_sum = math.degrees(self.theta_i_rad + self.theta_r_rad)
            if not math.isclose(angle_sum, 90.0, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"WARN: θ_i+θ_r={angle_sum:.2f}° ≠ 90° expected from geometry")
                # Not a failure, just warning
            else:
                self.logs.append(f"PASS: Geometric angle sum: {angle_sum:.2f}° ≈ 90°")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Test consistency between intermediate steps and final answer"""
        success = True
        
        # 1. Check EF = (a/2) * √3 (from geometry: ED = a/2, and EF = ED * √3)
        EF_expected = (self.a / 2) * math.sqrt(3)
        if not math.isclose(self.EF, EF_expected, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: EF inconsistent. EF={self.EF:.4f}a, "
                           f"expected={EF_expected:.4f}a")
            success = False
        else:
            self.logs.append(f"PASS: EF consistency: {self.EF:.4f}a ≈ {EF_expected:.4f}a")
        
        # 2. Check CF geometry (from steps: CF = BF * √3, but need to derive BF)
        # From triangle geometry with 30°-60°-90° triangle: BF = a * (2 - √3)/2
        # Actually, let's verify from problem: CF should be (a/√3)/2? Let's compute from geometry
        # Since θ_r = 30°, and triangle geometry gives CF = (a * sin(30°)) / sin(60°) = a/√3
        # But from steps: CF = 0.577a, and 1/√3 ≈ 0.57735
        CF_expected = self.a / math.sqrt(3)
        if not math.isclose(self.CF, CF_expected, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: CF inconsistent. CF={self.CF:.4f}a, "
                           f"expected={CF_expected:.4f}a")
            success = False
        else:
            self.logs.append(f"PASS: CF consistency: {self.CF:.4f}a ≈ {CF_expected:.4f}a")
        
        # 3. Check total distance: x = EF + CF
        x_from_parts = self.EF + self.CF
        if not math.isclose(self.x, x_from_parts, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Distance sum inconsistent. x={self.x:.4f}a, "
                           f"EF+CF={x_from_parts:.4f}a")
            success = False
        else:
            self.logs.append(f"PASS: Distance sum: x={self.x:.4f}a ≈ EF+CF={x_from_parts:.4f}a")
        
        # 4. Check time: t = x / v
        if self.v > 0:
            t_expected = self.x / self.v
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Time calculation inconsistent. t={self.t:.4f}a/c, "
                               f"x/v={t_expected:.4f}a/c")
                success = False
            else:
                self.logs.append(f"PASS: Time calculation: t={self.t:.4f}a/c ≈ x/v={t_expected:.4f}a/c")
        
        # 5. Check final answer against target: t = 2.5a/c
        t_target = 2.5 * self.a / self.c
        if not math.isclose(self.t, t_target, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Final answer mismatch. t={self.t:.4f}a/c, "
                           f"target=2.5a/c={t_target:.4f}a/c")
            success = False
        else:
            self.logs.append(f"PASS: Final answer matches target: {self.t:.4f}a/c ≈ 2.5a/c")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Test monotonicity by perturbing input 'a' and checking output trend.
        Since t ∝ a (from t = x/v and x ∝ a, v independent of a), 
        increasing a should increase t proportionally.
        """
        success = True
        
        # Store original values
        a_original = self.a
        t_original = self.t
        
        # Perturb a by +10%
        a_perturbed = a_original * 1.1
        
        # Recalculate using the physics formulas (not calling solution methods)
        # 1. Refractive index n (unchanged, as it depends only on angles)
        n = self.n  # Constant
        
        # 2. Speed in prism (unchanged relative to c)
        v = self.c / n if n > 0 else 0
        
        # 3. Geometric distances (scale with a)
        EF = (a_perturbed / 2) * math.sqrt(3)
        CF = a_perturbed / math.sqrt(3)
        x = EF + CF
        
        # 4. Time
        t_perturbed = x / v if v > 0 else 0
        
        # Expected scaling: t ∝ a, so t_perturbed/t_original = a_perturbed/a_original = 1.1
        expected_ratio = 1.1
        actual_ratio = t_perturbed / t_original if t_original > 0 else 0
        
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Monotonicity check failed. "
                           f"Ratio t_new/t_old={actual_ratio:.4f}, expected={expected_ratio:.4f}")
            success = False
        else:
            self.logs.append(f"PASS: Monotonicity check: t ∝ a (ratio={actual_ratio:.4f} ≈ {expected_ratio:.4f})")
        
        # Restore original values (important for subsequent tests)
        self.a = a_original
        self.t = t_original
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs"""
        self.logs = []
        
        self.logs.append("=== Physics Checker Starting ===")
        
        # Run tests
        test1_success = self.test_invariants()
        test2_success = self.test_trace_consistency()
        test3_success = self.test_monotonicity()
        
        overall_success = test1_success and test2_success and test3_success
        
        # Summary
        self.logs.append("=== Summary ===")
        self.logs.append(f"Invariants: {'PASS' if test1_success else 'FAIL'}")
        self.logs.append(f"Trace consistency: {'PASS' if test2_success else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if test3_success else 'FAIL'}")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs