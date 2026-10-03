import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Extract inputs with safe access
        self.d = getattr(solution_instance, 'd', getattr(solution_instance, 'results', {}).get('d', 1.0))
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'results', {}).get('v0', 1.0))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 1.0))
        self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'results', {}).get('q', 1.0))
        
        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        
        self.B = results.get('B', 0.0)
        self.E = results.get('E', 0.0)
        self.t = results.get('t', 0.0)
        self.t1 = results.get('t₁', results.get('t1', 0.0))
        self.T = results.get('T', 0.0)
        self.x = results.get('x', 0.0)
        self.r = results.get('r', 0.0)
        self.r1 = results.get('r₁', results.get('r1', 0.0))
        self.r2 = results.get('r₂', results.get('r2', 0.0))
        
        # Constants
        self.sin45 = math.sqrt(2)/2
        self.cos45 = math.sqrt(2)/2
        
        # Store logs
        self.logs = []

    def _log(self, msg: str, success: bool = None):
        """Add message to logs with optional success indicator."""
        if success is not None:
            prefix = "✓ " if success else "✗ "
            self.logs.append(prefix + msg)
        else:
            self.logs.append(msg)
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self) -> bool:
        """Verify fundamental physical laws and relationships."""
        success = True
        
        # 1. Check radius from geometry (step 1)
        expected_r = math.sqrt(2) * self.d
        if not self._is_close(self.r, expected_r):
            self._log(f"Geometry check: r={self.r}, expected √2*d={expected_r}", False)
            success = False
        else:
            self._log(f"Geometry check passed: r = √2*d", True)
        
        # 2. Check Lorentz force as centripetal force (step 2-3)
        # From q*v0*B = m*v0²/r => B = m*v0/(q*r)
        expected_B = self.m * self.v0 / (self.q * self.r)
        if not self._is_close(self.B, expected_B):
            self._log(f"Lorentz force check: B={self.B}, expected m*v0/(q*r)={expected_B}", False)
            success = False
        else:
            self._log(f"Lorentz force check passed: B = m*v0/(q*r)", True)
        
        # 3. Check r1 (step 4): r₁ = d (from B₁ = m*v0/(q*d) and r₁ = m*v0/(q*B₁))
        expected_r1 = self.d  # Correct from ground truth
        if not self._is_close(self.r1, expected_r1):
            self._log(f"r₁ check: r₁={self.r1}, expected d={expected_r1}", False)
            success = False
        else:
            self._log(f"r₁ check passed: r₁ = d", True)
        
        # 4. Check r2 (step 8): r₂ = 2√2*d
        expected_r2 = 2 * math.sqrt(2) * self.d
        if not self._is_close(self.r2, expected_r2):
            self._log(f"r₂ check: r₂={self.r2}, expected 2√2*d={expected_r2}", False)
            success = False
        else:
            self._log(f"r₂ check passed: r₂ = 2√2*d", True)
        
        # 5. Check x (step 9): x = 4d
        expected_x = 4 * self.d
        if not self._is_close(self.x, expected_x):
            self._log(f"Chord length check: x={self.x}, expected 4d={expected_x}", False)
            success = False
        else:
            self._log(f"Chord length check passed: x = 4d", True)
        
        # 6. Check period T (step 10): T = 2πr₂/v0
        expected_T = 2 * math.pi * self.r2 / self.v0
        if not self._is_close(self.T, expected_T):
            self._log(f"Period check: T={self.T}, expected 2πr₂/v0={expected_T}", False)
            success = False
        else:
            self._log(f"Period check passed: T = 2πr₂/v0", True)
        
        # 7. Check electric field E (step 5-7)
        # Correct equations from ground truth:
        # Horizontal distance: 2r₁ sin45° = v₀ cos45° * t
        # Vertical velocity change: 2v₀ sin45° = (qE/m) * t
        # Solving: t = (2r₁ sin45°)/(v₀ cos45°) = (2d * sin45°)/(v₀ cos45°)
        # Then E = (2v₀ sin45° * m)/(q * t)
        t_electric = (2 * self.d * self.sin45) / (self.v0 * self.cos45)
        if abs(t_electric) < 1e-12:
            self._log(f"Electric field check: t_electric too small ({t_electric})", False)
            success = False
        else:
            expected_E = (2 * self.v0 * self.sin45 * self.m) / (self.q * t_electric)
            if not self._is_close(self.E, expected_E):
                self._log(f"Electric field check: E={self.E}, expected from correct kinematics={expected_E}", False)
                success = False
            else:
                self._log(f"Electric field check passed", True)
        
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check t1 (step 11)
        # t₁ = 3*(3/4)T + 8√2*d/v0 + 1/4*T = (10/4)T + 8√2*d/v0
        expected_t1 = (10/4) * self.T + 8 * math.sqrt(2) * self.d / self.v0
        if not self._is_close(self.t1, expected_t1):
            self._log(f"t₁ consistency: t₁={self.t1}, expected (10/4)T+8√2d/v0={expected_t1}", False)
            success = False
        else:
            self._log(f"t₁ consistency check passed", True)
        
        # Check final time t (step 12): t = n*t₁ (n=1 for first return)
        expected_t = self.t1  # n=1 for first return
        if not self._is_close(self.t, expected_t):
            self._log(f"Final time t consistency: t={self.t}, expected t₁={expected_t} (n=1)", False)
            success = False
        else:
            self._log(f"Final time consistency check passed (n=1)", True)
        
        # Check that r2 = 2√2*d as in step 8
        expected_r2_from_d = 2 * math.sqrt(2) * self.d
        if not self._is_close(self.r2, expected_r2_from_d):
            self._log(f"r₂ from d: r₂={self.r2}, expected 2√2*d={expected_r2_from_d}", False)
            success = False
        else:
            self._log(f"r₂ = 2√2*d check passed", True)
        
        return success

    def test_monotonicity(self) -> bool:
        """Test how results change with perturbations of input parameters."""
        success = True
        
        # Save original values
        orig_d = self.d
        orig_v0 = self.v0
        orig_m = self.m
        orig_q = self.q
        orig_B = self.B
        orig_E = self.E
        orig_T = self.T
        
        # Test 1: Increase d by 10%, check B decreases (inverse relationship)
        self.d *= 1.1
        # B = m*v0/(q*√2*d) from Q1
        B_new = self.m * self.v0 / (self.q * (math.sqrt(2) * self.d))
        if not B_new < orig_B:
            self._log(f"Monotonicity d→B: B should decrease when d increases", False)
            success = False
        else:
            self._log(f"Monotonicity d→B: B ∝ 1/d confirmed", True)
        
        # Restore d
        self.d = orig_d
        
        # Test 2: Increase v0 by 10%, check B increases (direct relationship)
        self.v0 *= 1.1
        B_new = self.m * self.v0 / (self.q * (math.sqrt(2) * self.d))
        if not B_new > orig_B:
            self._log(f"Monotonicity v0→B: B should increase when v0 increases", False)
            success = False
        else:
            self._log(f"Monotonicity v0→B: B ∝ v0 confirmed", True)
        
        # Restore v0
        self.v0 = orig_v0
        
        # Test 3: Increase m by 10%, check B increases
        self.m *= 1.1
        B_new = self.m * self.v0 / (self.q * (math.sqrt(2) * self.d))
        if not B_new > orig_B:
            self._log(f"Monotonicity m→B: B should increase when m increases", False)
            success = False
        else:
            self._log(f"Monotonicity m→B: B ∝ m confirmed", True)
        
        # Restore m
        self.m = orig_m
        
        # Test 4: Increase q by 10%, check B decreases
        self.q *= 1.1
        B_new = self.m * self.v0 / (self.q * (math.sqrt(2) * self.d))
        if not B_new < orig_B:
            self._log(f"Monotonicity q→B: B should decrease when |q| increases", False)
            success = False
        else:
            self._log(f"Monotonicity q→B: B ∝ 1/|q| confirmed", True)
        
        # Restore q
        self.q = orig_q
        
        # Test 5: Check period T decreases with v0 (T ∝ 1/v from T = 2πr₂/v0, and r₂ constant with respect to v0? Actually r₂ = 2√2*d is independent of v0)
        self.v0 *= 1.1
        T_new = 2 * math.pi * (2 * math.sqrt(2) * self.d) / self.v0
        if not T_new < orig_T:
            self._log(f"Monotonicity v0→T: T should decrease when v0 increases (T ∝ 1/v)", False)
            success = False
        else:
            self._log(f"Monotonicity v0→T: T ∝ 1/v confirmed", True)
        
        # Restore v0
        self.v0 = orig_v0
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag with logs."""
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        # Run tests
        test_results = []
        
        self.logs.append("\n1. Testing Physical Invariants:")
        test_results.append(self.test_invariants())
        
        self.logs.append("\n2. Testing Trace Consistency:")
        test_results.append(self.test_trace_consistency())
        
        self.logs.append("\n3. Testing Monotonicity:")
        test_results.append(self.test_monotonicity())
        
        # Summary
        all_passed = all(test_results)
        self.logs.append(f"\n=== Summary: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'} ===")
        self.logs.append(f"Tests passed: {sum(test_results)}/{len(test_results)}")
        
        return all_passed, self.logs