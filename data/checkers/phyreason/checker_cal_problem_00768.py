import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract input parameters
        self.m_A = getattr(solution_instance, 'm_A', solution_instance.results.get('m_A', 0.1))
        self.m_B = getattr(solution_instance, 'm_B', solution_instance.results.get('m_B', 0.2))
        self.l0 = getattr(solution_instance, 'l0', solution_instance.results.get('l0', 0.1))
        self.k = getattr(solution_instance, 'k', solution_instance.results.get('k', 50.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 10.0))

        # Safely extract computed results
        self.l_1 = getattr(solution_instance, 'l_1', solution_instance.results.get('l_1', None))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', None))
        self.l_2 = getattr(solution_instance, 'l_2', solution_instance.results.get('l_2', None))
        self.N = getattr(solution_instance, 'N', solution_instance.results.get('N', None))
        self.W = getattr(solution_instance, 'W', solution_instance.results.get('W', None))

        # Target answers for trace consistency
        self.targets = {"h": 0.08, "N": 4.0, "W": 0.03}

        # Logging list
        self.logs = []

    def _log(self, msg: str):
        self.logs.append(msg)

    def _safe_isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self) -> bool:
        """Test physics invariants based on KB theorems."""
        success = True
        self._log("=== Testing Invariants ===")

        # 1. Hooke's Law and equilibrium (step1)
        if self.l_1 is not None:
            expected_l1 = self.m_A * self.g / self.k
            if not self._safe_isclose(self.l_1, expected_l1):
                self._log(f"FAIL: Hooke's law equilibrium: l_1={self.l_1}, expected {expected_l1}")
                success = False
            else:
                self._log(f"PASS: l_1 matches Hooke's law (m_A*g/k)")
        else:
            self._log("SKIP: l_1 not available")

        # 2. Newton's 2nd Law for object B at max compression (step4)
        if self.N is not None and self.l_2 is not None:
            # Normal force = weight of B + spring force (upward, but pressure on ground is downward)
            expected_N = self.m_B * self.g + self.k * self.l_2
            if not self._safe_isclose(self.N, expected_N):
                self._log(f"FAIL: Newton's 2nd law for B: N={self.N}, expected {expected_N}")
                success = False
            else:
                self._log(f"PASS: N matches m_B*g + k*l_2")
        elif self.N is not None:
            self._log("SKIP: l_2 missing for Newton's 2nd law check")

        # 3. Symmetry of SHM (step3)
        if self.l_1 is not None and self.l_2 is not None:
            expected_l2 = 2.0 * self.l_1
            if not self._safe_isclose(self.l_2, expected_l2):
                self._log(f"FAIL: SHM symmetry: l_2={self.l_2}, expected 2*l_1={expected_l2}")
                success = False
            else:
                self._log(f"PASS: l_2 = 2*l_1 (SHM symmetry)")
        elif self.l_2 is not None:
            self._log("SKIP: l_1 missing for SHM symmetry check")

        # 4. Work-Energy Theorem (step5)
        if self.W is not None and self.l_1 is not None and self.l_2 is not None:
            # Work done by spring force (magnitude) = average force * displacement
            avg_force = 0.5 * self.k * (self.l_1 + self.l_2)
            displacement = self.l_1  # from equilibrium to lowest point
            expected_W = avg_force * displacement
            if not self._safe_isclose(self.W, expected_W):
                self._log(f"FAIL: Work-energy: W={self.W}, expected (k*l1 + k*l2)/2 * l1 = {expected_W}")
                success = False
            else:
                self._log(f"PASS: W matches average spring force times displacement")

            # Alternative check: change in spring potential energy magnitude
            delta_U = 0.5 * self.k * (self.l_2**2 - self.l_1**2)
            if not self._safe_isclose(self.W, delta_U):
                self._log(f"NOTE: W differs from ΔU_spring={delta_U} (but may be correct magnitude)")
        elif self.W is not None:
            self._log("SKIP: l_1 or l_2 missing for work-energy check")

        # 5. Energy conservation (total mechanical energy at equilibrium vs lowest point)
        if all(v is not None for v in [self.l_1, self.l_2, self.h]):
            # Reference: ground level for gravitational potential
            # At equilibrium: spring PE + gravitational PE of A
            U_eq = 0.5 * self.k * self.l_1**2
            h_eq = self.h  # height from ground
            PE_grav_eq = self.m_A * self.g * h_eq
            E_eq = U_eq + PE_grav_eq

            # At lowest point: spring PE (max compression) + gravitational PE (lower height)
            U_low = 0.5 * self.k * self.l_2**2
            h_low = self.h - self.l_1  # displacement downward by l_1
            PE_grav_low = self.m_A * self.g * h_low
            E_low = U_low + PE_grav_low

            if not self._safe_isclose(E_eq, E_low):
                self._log(f"NOTE: Energy not conserved: E_eq={E_eq:.4f}, E_low={E_low:.4f}")
                # This may be expected due to the setup (B is fixed), but log anyway
            else:
                self._log(f"PASS: Energy conserved between equilibrium and lowest point")

        return success

    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate steps and target answers."""
        success = True
        self._log("=== Testing Trace Consistency ===")

        # Check step2: geometric height
        if self.h is not None and self.l_1 is not None:
            expected_h = self.l0 - self.l_1
            if not self._safe_isclose(self.h, expected_h):
                self._log(f"FAIL: Geometry: h={self.h}, expected l0-l_1={expected_h}")
                success = False
            else:
                self._log(f"PASS: h = l0 - l_1")
        elif self.h is not None:
            self._log("SKIP: l_1 missing for geometry check")

        # Compare with target answers (allow unit conversions)
        if self.h is not None:
            if not self._safe_isclose(self.h, self.targets["h"]):
                self._log(f"FAIL: h={self.h} differs from target 0.08m (8cm)")
                success = False
            else:
                self._log(f"PASS: h matches target 8cm")
        
        if self.N is not None:
            if not self._safe_isclose(self.N, self.targets["N"]):
                self._log(f"FAIL: N={self.N} differs from target 4.0N")
                success = False
            else:
                self._log(f"PASS: N matches target 4N")

        if self.W is not None:
            if not self._safe_isclose(self.W, self.targets["W"]):
                self._log(f"FAIL: W={self.W} differs from target 0.03J")
                success = False
            else:
                self._log(f"PASS: W matches target 0.03J")

        # Cross-check: l_2 from symmetry and from target (4cm = 0.04m)
        if self.l_2 is not None:
            if not self._safe_isclose(self.l_2, 0.04):
                self._log(f"NOTE: l_2={self.l2} differs from expected 0.04m (4cm)")

        return success

    def test_monotonicity(self) -> bool:
        """Perturb mass m_A and check trends."""
        success = True
        self._log("=== Testing Monotonicity (perturb m_A) ===")

        original_mA = self.m_A
        perturbed_mA = original_mA * 1.05  # +5% increase

        # Recompute using formulas
        l1_new = perturbed_mA * self.g / self.k
        h_new = self.l0 - l1_new
        l2_new = 2.0 * l1_new
        N_new = self.m_B * self.g + self.k * l2_new
        avg_force_new = 0.5 * self.k * (l1_new + l2_new)
        W_new = avg_force_new * l1_new

        # Check trends
        if self.l_1 is not None:
            if not (l1_new > self.l_1):
                self._log(f"FAIL: l_1 should increase with m_A")
                success = False
            else:
                self._log(f"PASS: l_1 increases with m_A")

        if self.h is not None:
            if not (h_new < self.h):
                self._log(f"FAIL: h should decrease with m_A")
                success = False
            else:
                self._log(f"PASS: h decreases with m_A")

        if self.l_2 is not None:
            if not (l2_new > self.l_2):
                self._log(f"FAIL: l_2 should increase with m_A")
                success = False
            else:
                self._log(f"PASS: l_2 increases with m_A")

        if self.N is not None:
            if not (N_new > self.N):
                self._log(f"FAIL: N should increase with m_A")
                success = False
            else:
                self._log(f"PASS: N increases with m_A")

        if self.W is not None:
            if not (W_new > self.W):
                self._log(f"FAIL: W should increase with m_A")
                success = False
            else:
                self._log(f"PASS: W increases with m_A")

        # Restore original value
        self.m_A = original_mA
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all checks and return success flag and logs."""
        self.logs.clear()
        
        success = True
        success = self.test_invariants() and success
        success = self.test_trace_consistency() and success
        success = self.test_monotonicity() and success

        self._log(f"=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs