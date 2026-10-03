import math
import sympy as sp
from typing import Dict, Tuple, List, Any, Optional

class PhysicsChecker:
    """
    Physics checker for hemispherical shell refraction problem.
    """
    
    def __init__(self, solution_instance):
        """
        Safe attribute extraction to prevent sandbox errors.
        """
        self.n = math.sqrt(2)  # Given refractive index
        self.R_prime = math.sqrt(2)  # R'/R ratio
        
        # Extract results with fallbacks
        results = getattr(solution_instance, 'results', {})
        
        self.d = getattr(solution_instance, 'd', results.get('d', 0.0))
        self.OB = getattr(solution_instance, 'OB', results.get('OB', 0.0))
        self.r_deg = getattr(solution_instance, 'r', results.get('r', 0.0))  # Refraction angle in degrees
        self.angle_Oprime_Oa = getattr(solution_instance, '∠O\'Oa', results.get('∠O\'Oa', results.get('∠O_Oa', 0.0)))
        self.i_deg = getattr(solution_instance, 'i', results.get('i', 0.0))  # Incidence angle in degrees
        self.OA = getattr(solution_instance, 'OA', results.get('OA', 0.0))
        self.C_deg = getattr(solution_instance, 'C', results.get('C', 0.0))  # Critical angle in degrees
        
        # Store for monotonicity test
        self.original_OB = self.OB
        self.original_d = self.d
        
    def _deg_to_rad(self, angle_deg: float) -> float:
        """Convert degrees to radians."""
        return math.radians(angle_deg)
    
    def _rad_to_deg(self, angle_rad: float) -> float:
        """Convert radians to degrees."""
        return math.degrees(angle_rad)
    
    def test_invariants(self) -> List[str]:
        """
        Verify physical invariants:
        1. Critical angle for total internal reflection
        2. Snell's law at outer surface
        3. Law of sines in triangle geometry
        """
        logs = []
        
        # 1. Check critical angle: sin(C) = 1/n
        C_rad = self._deg_to_rad(self.C_deg)
        sin_C = math.sin(C_rad)
        expected_sin_C = 1.0 / self.n
        
        if math.isclose(sin_C, expected_sin_C, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Critical angle check: sin({self.C_deg:.1f}°) = {sin_C:.3f} ≈ 1/√2 = {expected_sin_C:.3f}")
        else:
            logs.append(f"✗ Critical angle mismatch: sin({self.C_deg:.1f}°) = {sin_C:.3f}, expected {expected_sin_C:.3f}")
        
        # 2. Check Snell's law at outer surface: sin(i) = n * sin(r)
        i_rad = self._deg_to_rad(self.i_deg)
        r_rad = self._deg_to_rad(self.r_deg)
        sin_i = math.sin(i_rad)
        sin_r = math.sin(r_rad)
        lhs = sin_i
        rhs = self.n * sin_r
        
        if math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Snell's law: sin({self.i_deg:.1f}°) = {sin_i:.3f} ≈ √2 * sin({self.r_deg:.1f}°) = {rhs:.3f}")
        else:
            logs.append(f"✗ Snell's law mismatch: sin(i)={sin_i:.3f}, n*sin(r)={rhs:.3f}")
        
        # 3. Check law of sines: sin(180°-C)/R' = sin(r)/R
        # Using R' = √2 and OB = R = 1 (normalized), OA = √2
        sin_180_minus_C = math.sin(math.pi - C_rad)  # sin(180°-C) = sin(C)
        expected_sin_r = sin_180_minus_C / self.R_prime
        
        if math.isclose(sin_r, expected_sin_r, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Law of sines: sin(r) = {sin_r:.3f} ≈ sin({self.C_deg:.1f}°)/√2 = {expected_sin_r:.3f}")
        else:
            logs.append(f"✗ Law of sines mismatch: sin(r)={sin_r:.3f}, expected {expected_sin_r:.3f}")
        
        # 4. Check geometry: OA/OB = √2
        if not math.isclose(self.OB, 0.0, abs_tol=1e-9):
            ratio = self.OA / self.OB
            if math.isclose(ratio, self.R_prime, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ Geometry: OA/OB = {ratio:.3f} ≈ √2 = {self.R_prime:.3f}")
            else:
                logs.append(f"✗ Geometry mismatch: OA/OB = {ratio:.3f}, expected √2 = {self.R_prime:.3f}")
        else:
            logs.append("⚠ OB is zero, skipping geometry check")
        
        # 5. Check shield radius formula: d = R' * sin(i)
        expected_d = self.R_prime * self.OB * sin_i  # R' * R * sin(i) with R=OB
        if math.isclose(self.d, expected_d, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Shield radius: d = {self.d:.3f} ≈ √2 * R * sin({self.i_deg:.1f}°) = {expected_d:.3f}")
        else:
            logs.append(f"✗ Shield radius mismatch: d={self.d:.3f}, expected {expected_d:.3f}")
        
        # 6. Check angle O'Oa = 45° = i
        if math.isclose(self.angle_Oprime_Oa, self.i_deg, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Angle consistency: ∠O'Oa = {self.angle_Oprime_Oa:.1f}° ≈ i = {self.i_deg:.1f}°")
        else:
            logs.append(f"✗ Angle mismatch: ∠O'Oa={self.angle_Oprime_Oa:.1f}°, i={self.i_deg:.1f}°")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """
        Check consistency between intermediate results and final answer.
        """
        logs = []
        
        # 1. From critical angle and geometry, verify r = 30°
        expected_r = 30.0
        if math.isclose(self.r_deg, expected_r, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Refraction angle: r = {self.r_deg:.1f}° ≈ 30°")
        else:
            logs.append(f"✗ Refraction angle mismatch: r = {self.r_deg:.1f}°, expected 30°")
        
        # 2. From r and Snell's law, verify i = 45°
        expected_i = 45.0
        if math.isclose(self.i_deg, expected_i, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Incidence angle: i = {self.i_deg:.1f}° ≈ 45°")
        else:
            logs.append(f"✗ Incidence angle mismatch: i = {self.i_deg:.1f}°, expected 45°")
        
        # 3. Final check: d should equal OB (R)
        if math.isclose(self.d, self.OB, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Final answer: d = {self.d:.3f} ≈ R = {self.OB:.3f}")
        else:
            logs.append(f"✗ Final answer mismatch: d = {self.d:.3f}, R = {self.OB:.3f}")
        
        # 4. Symbolic verification of main result
        try:
            # Symbolic check: d = R' * sin(i) * R, with sin(i) = 1/√2, R' = √2R
            # So d = (√2R) * (1/√2) * R = R²? Wait, careful:
            # Actually: d = R' * sin(i) = (√2R) * (1/√2) = R
            # So d/R should be 1
            if not math.isclose(self.OB, 0.0, abs_tol=1e-9):
                ratio = self.d / self.OB
                if math.isclose(ratio, 1.0, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ Symbolic ratio: d/R = {ratio:.3f} ≈ 1")
                else:
                    logs.append(f"✗ Symbolic ratio mismatch: d/R = {ratio:.3f}, expected 1")
            else:
                logs.append("⚠ OB is zero, skipping symbolic ratio check")
        except Exception as e:
            logs.append(f"⚠ Symbolic check failed: {str(e)}")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """
        Test how shield radius changes with scale.
        Perturb OB (R) and verify d scales linearly.
        """
        logs = []
        
        try:
            # Save original values
            original_OB = self.OB
            original_d = self.d
            
            if math.isclose(original_OB, 0.0, abs_tol=1e-9):
                logs.append("⚠ OB is zero, skipping monotonicity test")
                return logs
            
            # Perturbation factor
            factor = 1.5
            
            # Scale R (OB) and recalculate expected d
            new_OB = original_OB * factor
            new_OA = self.R_prime * new_OB  # OA scales with R
            
            # d = R' * sin(i) * R, and sin(i) remains constant (45°)
            sin_i = math.sin(self._deg_to_rad(self.i_deg))
            expected_new_d = self.R_prime * new_OB * sin_i
            
            # Since d ∝ R, new_d should be factor * original_d
            expected_by_scaling = original_d * factor
            
            logs.append(f"Monotonicity test with scale factor {factor}:")
            logs.append(f"  Original: OB = {original_OB:.3f}, d = {original_d:.3f}")
            logs.append(f"  Scaled: OB = {new_OB:.3f}")
            logs.append(f"  Expected d by formula: {expected_new_d:.3f}")
            logs.append(f"  Expected d by scaling: {expected_by_scaling:.3f}")
            
            # Both should be close
            if (math.isclose(expected_new_d, expected_by_scaling, rel_tol=0.02, abs_tol=0.02) and
                math.isclose(expected_new_d/original_d, factor, rel_tol=0.02, abs_tol=0.02)):
                logs.append("  ✓ d scales linearly with R as expected")
            else:
                logs.append("  ✗ d does not scale linearly with R")
            
            # Restore original values
            self.OB = original_OB
            self.d = original_d
            
        except Exception as e:
            logs.append(f"⚠ Monotonicity test failed: {str(e)}")
            # Restore on error
            self.OB = self.original_OB
            self.d = self.original_d
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics checks and return overall success and logs.
        """
        all_logs = []
        success = True
        
        all_logs.append("=" * 50)
        all_logs.append("Physics Checker for Hemispherical Shell Problem")
        all_logs.append("=" * 50)
        
        # Test 1: Physical invariants
        all_logs.append("\n1. Physical Invariants Check:")
        all_logs.append("-" * 30)
        inv_logs = self.test_invariants()
        all_logs.extend(inv_logs)
        
        # Check for failures in invariants
        inv_failures = sum(1 for log in inv_logs if log.startswith("✗"))
        if inv_failures > 0:
            success = False
        
        # Test 2: Trace consistency
        all_logs.append("\n2. Trace Consistency Check:")
        all_logs.append("-" * 30)
        trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        
        trace_failures = sum(1 for log in trace_logs if log.startswith("✗"))
        if trace_failures > 0:
            success = False
        
        # Test 3: Monotonicity
        all_logs.append("\n3. Monotonicity Check:")
        all_logs.append("-" * 30)
        mono_logs = self.test_monotonicity()
        all_logs.extend(mono_logs)
        
        mono_failures = sum(1 for log in mono_logs if log.startswith("✗"))
        if mono_failures > 0:
            success = False
        
        # Summary
        all_logs.append("\n" + "=" * 50)
        all_logs.append(f"Overall Status: {'✓ PASS' if success else '✗ FAIL'}")
        all_logs.append(f"Invariant failures: {inv_failures}")
        all_logs.append(f"Trace failures: {trace_failures}")
        all_logs.append(f"Monotonicity failures: {mono_failures}")
        all_logs.append("=" * 50)
        
        return success, all_logs


# For direct testing
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'd': 1.0,
                'OB': 1.0,
                'r': 30.0,
                '∠O\'Oa': 45.0,
                'i': 45.0,
                'OA': math.sqrt(2),
                'C': 45.0
            }
    
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)
    
    print(f"\nFinal success: {success}")