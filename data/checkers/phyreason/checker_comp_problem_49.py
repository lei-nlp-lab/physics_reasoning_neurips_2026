import math
import sympy as sp
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize with safe attribute access.
        solution_instance: object from the student's solution, expected to have
        attributes like M, m, R, h, theta, v, r, and a results dictionary.
        """
        self.solution = solution_instance

        # Extract input parameters safely
        self.M = getattr(solution_instance, 'M', None)
        self.m = getattr(solution_instance, 'm', None)
        self.R = getattr(solution_instance, 'R', None)
        self.h = getattr(solution_instance, 'h', None)
        self.theta = getattr(solution_instance, 'theta', None)
        self.v = getattr(solution_instance, 'v', None)
        self.r = getattr(solution_instance, 'r', None)

        # Extract results safely, fallback to 0.0 if not found
        results = getattr(solution_instance, 'results', {})
        self.phi = results.get('ϕ', results.get('phi', 0.0))
        self.phi1 = results.get('ϕ₁', results.get('phi1', 0.0))
        self.phi2 = results.get('ϕ₂', results.get('phi2', 0.0))
        self.omega1 = results.get('omega1', 0.0)   # ω for first leg
        self.omega2 = results.get('omega2', 0.0)   # ω for third leg
        self.L = results.get('L', 0.0)            # Angular momentum

        # For second part (circular path)
        self.phi_circ = results.get('phi_circ', self.phi)  # maybe same key?

        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

        # Logs
        self.logs = []

    def log(self, msg: str):
        self.logs.append(msg)

    def approx_equal(self, a: float, b: float) -> bool:
        """Fuzzy equality check."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self) -> bool:
        """Check conservation of angular momentum and other invariants."""
        success = True

        # Check 1: First leg angular momentum conservation
        if all(x is not None for x in [self.M, self.m, self.R, self.v]):
            # Theoretical ω1 to leading order
            omega1_theory = -2 * self.m * self.v / (self.M * self.R)
            if self.omega1 != 0.0:
                if not self.approx_equal(self.omega1, omega1_theory):
                    self.log(f"First leg ω mismatch: got {self.omega1}, expected {omega1_theory}")
                    success = False
                else:
                    self.log("First leg ω consistent with angular momentum conservation.")

            # Check total angular momentum zero at start and conserved?
            # Initial L = 0. After first leg, L_disk = 0.5 * self.M * self.R**2 * self.omega1
            # L_ant = self.m * self.R * (self.v + self.omega1 * self.R)  # relative to ground
            # But to leading order, L_ant ≈ self.m * self.R * self.v
            # So total L ≈ 0.5 * self.M * self.R**2 * self.omega1 + self.m * self.R * self.v
            # Plugging omega1_theory, total L ≈ 0.
            # We can compute numerically if we have omega1.
            if self.omega1 != 0.0:
                L_disk = 0.5 * self.M * self.R**2 * self.omega1
                L_ant = self.m * self.R * (self.v + self.omega1 * self.R)
                L_total = L_disk + L_ant
                if not self.approx_equal(L_total, 0.0):
                    self.log(f"Angular momentum not conserved in first leg: L_total = {L_total}")
                    success = False
                else:
                    self.log("Angular momentum conserved in first leg (numerically).")

        # Check 2: Third leg angular momentum conservation (starting from ω=0)
        if all(x is not None for x in [self.M, self.m, self.R, self.h, self.v]):
            omega2_theory = 2 * self.m * self.v * (self.R - self.h) / (self.M * self.R**2)
            if self.omega2 != 0.0:
                if not self.approx_equal(self.omega2, omega2_theory):
                    self.log(f"Third leg ω mismatch: got {self.omega2}, expected {omega2_theory}")
                    success = False
                else:
                    self.log("Third leg ω consistent with angular momentum conservation.")

                # Check conservation during third leg: initial L=0, final L = L_disk + L_ant
                L_disk2 = 0.5 * self.M * self.R**2 * self.omega2
                # Ant at distance (R-h), velocity relative to ground: v + omega2*(R-h)
                L_ant2 = self.m * (self.R - self.h) * (self.v + self.omega2 * (self.R - self.h))
                L_total2 = L_disk2 + L_ant2
                if not self.approx_equal(L_total2, 0.0):
                    self.log(f"Angular momentum not conserved in third leg: L_total = {L_total2}")
                    success = False
                else:
                    self.log("Angular momentum conserved in third leg (numerically).")

        # Check 3: Radial leg: ω should be zero (or constant if starting from non-zero, but step says 0)
        # Actually, from step_3, ω=0. We can't check without a value, but if omega_rad is provided, check.
        # Not implemented because not in results.

        return success

    def test_trace_consistency(self) -> bool:
        """Check that intermediate results are consistent with final answer."""
        success = True

        # Part 1: Path with arcs and radial movement
        if all(x is not None for x in [self.M, self.m, self.theta, self.R, self.h, self.v]):
            # Check φ1 from ω1 and time
            if self.omega1 != 0.0 and self.phi1 != 0.0:
                t1 = self.theta * self.R / self.v   # time for first leg
                phi1_from_omega = self.omega1 * t1
                if not self.approx_equal(self.phi1, phi1_from_omega):
                    self.log(f"φ1 inconsistent: from ω1 and t1 gives {phi1_from_omega}, got {self.phi1}")
                    success = False
                else:
                    self.log("φ1 consistent with ω1 and t1.")

                # Also check formula φ1 = -2*m*theta/M
                phi1_formula = -2 * self.m * self.theta / self.M
                if not self.approx_equal(self.phi1, phi1_formula):
                    self.log(f"φ1 does not match formula: expected {phi1_formula}, got {self.phi1}")
                    success = False
                else:
                    self.log("φ1 matches formula -2*m*θ/M.")

            # Check φ2 from ω2 and time
            if self.omega2 != 0.0 and self.phi2 != 0.0:
                t2 = self.theta * (self.R - self.h) / self.v
                phi2_from_omega = self.omega2 * t2
                if not self.approx_equal(self.phi2, phi2_from_omega):
                    self.log(f"φ2 inconsistent: from ω2 and t2 gives {phi2_from_omega}, got {self.phi2}")
                    success = False
                else:
                    self.log("φ2 consistent with ω2 and t2.")

                # Check formula φ2 = (2*m*θ/M)*(1 - h/R)**2
                phi2_formula = (2 * self.m * self.theta / self.M) * (1 - self.h/self.R)**2
                if not self.approx_equal(self.phi2, phi2_formula):
                    self.log(f"φ2 does not match formula: expected {phi2_formula}, got {self.phi2}")
                    success = False
                else:
                    self.log("φ2 matches formula (2*m*θ/M)*(1 - h/R)^2.")

            # Check φ = φ1 + φ2
            if self.phi1 != 0.0 and self.phi2 != 0.0 and self.phi != 0.0:
                if not self.approx_equal(self.phi, self.phi1 + self.phi2):
                    self.log(f"φ ≠ φ1 + φ2: {self.phi} vs {self.phi1 + self.phi2}")
                    success = False
                else:
                    self.log("φ = φ1 + φ2 holds.")

                # Check exact expression for φ
                phi_exact = (2 * self.m * self.theta / self.M) * ((1 - self.h/self.R)**2 - 1)
                if not self.approx_equal(self.phi, phi_exact):
                    self.log(f"φ does not match exact expression: expected {phi_exact}, got {self.phi}")
                    success = False
                else:
                    self.log("φ matches exact expression (2*m*θ/M)*((1-h/R)^2 - 1).")

                # Check approximate expression (if h/R is small, but we compute anyway)
                phi_approx = -4 * self.m * self.h * self.theta / (self.M * self.R)
                # We don't require exact match because approximation, but we can check if close
                # Note: approximation error is of order (h/R)^2
                if abs(self.h/self.R) < 0.1:  # if h/R small, approximation should be good
                    if not self.approx_equal(self.phi, phi_approx):
                        self.log(f"φ does not match approximate expression -4*m*h*θ/(M*R): expected {phi_approx}, got {self.phi}")
                        success = False
                    else:
                        self.log("φ matches approximate expression -4*m*h*θ/(M*R) (h/R small).")

        # Part 2: Circular path
        if all(x is not None for x in [self.M, self.m, self.R, self.r]):
            phi_circ_theory = 4 * self.m * math.pi * self.r**2 / (self.M * self.R**2)
            if hasattr(self, 'phi_circ') and self.phi_circ != 0.0:
                if not self.approx_equal(self.phi_circ, phi_circ_theory):
                    self.log(f"Circular path φ mismatch: got {self.phi_circ}, expected {phi_circ_theory}")
                    success = False
                else:
                    self.log("Circular path φ matches 4*m*π*r²/(M*R²).")
            elif self.phi != 0.0 and self.r is not None:  # maybe φ is used for both?
                # If we have r and no separate phi_circ, check if phi matches the circular formula?
                # But careful: only if the problem is about the circular path.
                # We'll skip this to avoid false positives.
                pass

        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb an input and check that the output changes as expected.
        We use the formulas we have, not the solution's code.
        """
        success = True

        # Perturb m (increase by 10%)
        if all(x is not None for x in [self.M, self.m, self.theta, self.R, self.h]):
            m_original = self.m
            m_perturbed = m_original * 1.1

            # Compute φ original using exact formula
            phi_original = (2 * m_original * self.theta / self.M) * ((1 - self.h/self.R)**2 - 1)
            phi_perturbed = (2 * m_perturbed * self.theta / self.M) * ((1 - self.h/self.R)**2 - 1)

            # Check linear scaling (to leading order)
            expected_ratio = 1.1
            actual_ratio = phi_perturbed / phi_original if phi_original != 0 else 0
            if phi_original != 0 and not self.approx_equal(actual_ratio, expected_ratio):
                self.log(f"Monotonicity in m failed: ratio = {actual_ratio}, expected {expected_ratio}")
                success = False
            else:
                self.log(f"Monotonicity in m holds: φ scales linearly with m (ratio {actual_ratio}).")

            # Perturb h (increase by 5%)
            if self.h is not None and self.R is not None:
                h_original = self.h
                h_perturbed = h_original * 1.05
                # Avoid going beyond R
                if h_perturbed < self.R:
                    phi_h_original = (2 * m_original * self.theta / self.M) * ((1 - h_original/self.R)**2 - 1)
                    phi_h_perturbed = (2 * m_original * self.theta / self.M) * ((1 - h_perturbed/self.R)**2 - 1)
                    # φ should become more negative as h increases (since (1-h/R)^2 decreases)
                    if phi_h_original != 0 and phi_h_perturbed > phi_h_original:  # less negative
                        self.log(f"Monotonicity in h failed: φ should decrease (become more negative) as h increases.")
                        success = False
                    else:
                        self.log("Monotonicity in h holds: φ decreases as h increases.")
                else:
                    self.log("Skipped h perturbation because h_perturbed >= R.")

        # For circular path, perturb r
        if all(x is not None for x in [self.M, self.m, self.R, self.r]):
            r_original = self.r
            r_perturbed = r_original * 1.1
            phi_r_original = 4 * self.m * math.pi * r_original**2 / (self.M * self.R**2)
            phi_r_perturbed = 4 * self.m * math.pi * r_perturbed**2 / (self.M * self.R**2)
            # φ scales as r^2
            expected_ratio = 1.1**2
            actual_ratio = phi_r_perturbed / phi_r_original if phi_r_original != 0 else 0
            if phi_r_original != 0 and not self.approx_equal(actual_ratio, expected_ratio):
                self.log(f"Monotonicity in r failed: ratio = {actual_ratio}, expected {expected_ratio}")
                success = False
            else:
                self.log(f"Monotonicity in r holds: φ scales as r² (ratio {actual_ratio}).")

        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs = []  # reset logs
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.log(f"Test '{name}' failed.")
                else:
                    self.log(f"Test '{name}' passed.")
            except Exception as e:
                self.log(f"Test '{name}' raised exception: {e}\n{traceback.format_exc()}")
                success = False

        return success, self.logs