import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary inputs and results from the solution instance.
        """
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 1.0))

        # Safe attribute access for results (key mapping from KB)
        self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', 0.0))
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', 0.0))
        self.E_kB = getattr(solution_instance, 'E_kB', solution_instance.results.get('E_kB', 0.0))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 0.0))
        self.W = getattr(solution_instance, 'W', solution_instance.results.get('W', 0.0))
        self.E_kA = getattr(solution_instance, 'E_kA', solution_instance.results.get('E_kA', 0.0))

        # Trigonometric constants
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.sin53 = 0.8  # sin 53° = 0.8
        self.cos53 = 0.6  # cos 53° = 0.6
        self.tan53 = self.sin53 / self.cos53  # ≈ 1.333...

        # Derived quantities (computed for checking)
        self.omega_sq = None  # ω² from centripetal condition on B
        self.T1_rest = None   # Tension in OB at rest
        self.T2_rest = None   # Tension in AB at rest

    def compute_derived_quantities(self):
        """Compute derived quantities from the extracted results."""
        # ω² from centripetal condition on ball B (step 6)
        self.omega_sq = (self.g * self.tan53) / (4 * self.L / 5)  # = (5 * self.g * self.tan53) / (4 * self.L)

        # Tensions at rest from equilibrium on B (steps 2-3)
        # From vertical equilibrium: T1 cos37 + T2 cos37 = m g
        # From horizontal equilibrium: T1 sin37 = T2 sin37 → T1 = T2
        T = (self.m * self.g) / (2 * self.cos37)  # common tension
        self.T1_rest = T
        self.T2_rest = T

    def test_invariants(self):
        """
        Verify physical invariants:
        1. Static equilibrium at rest (forces on B and A).
        2. Centripetal force conditions during rotation.
        3. Energy conservation (work-energy theorem).
        """
        logs = []
        self.compute_derived_quantities()

        try:
            # 1. Check static equilibrium at rest
            # a) Vertical equilibrium on B: T1 cos37 + T2 cos37 = m g
            vertical_force = self.T1_rest * self.cos37 + self.T2_rest * self.cos37
            if not math.isclose(vertical_force, self.m * self.g, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical equilibrium on B at rest: {vertical_force} vs {self.m * self.g}")

            # b) Horizontal equilibrium on B: T1 sin37 = T2 sin37
            if not math.isclose(self.T1_rest * self.sin37, self.T2_rest * self.sin37, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("FAIL: Horizontal equilibrium on B at rest")

            # c) Horizontal equilibrium on A at rest: T2 sin37 = F (spring force)
            F_from_equilibrium = self.T2_rest * self.sin37
            if not math.isclose(F_from_equilibrium, self.F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Spring force from equilibrium on A: {F_from_equilibrium} vs {self.F}")

            # 2. Check centripetal force conditions during rotation
            # a) On ball B: m g tan53° = m ω² * (4L/5)  → ω² = (g tan53) / (4L/5)
            lhs_b = self.m * self.g * self.tan53
            rhs_b = self.m * self.omega_sq * (4 * self.L / 5)
            if not math.isclose(lhs_b, rhs_b, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Centripetal force on B: {lhs_b} vs {rhs_b}")

            # b) On ring A: F = M ω² * (8L/5)  (spring force magnitude unchanged)
            lhs_a = self.F
            rhs_a = self.M * self.omega_sq * (8 * self.L / 5)
            if not math.isclose(lhs_a, rhs_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Centripetal force on A: {lhs_a} vs {rhs_a}")

            # 3. Check work-energy theorem: W = (E_kA + E_kB) + m g h
            total_energy_change = self.E_kA + self.E_kB + self.m * self.g * self.h
            if not math.isclose(self.W, total_energy_change, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy theorem: W={self.W}, ΔE={total_energy_change}")

            if not logs:
                logs.append("PASS: All invariants satisfied.")
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")

        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        self.compute_derived_quantities()

        try:
            # 1. Check spring force F = 3/8 m g
            expected_F = (3 * self.m * self.g) / 8
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Spring force F: {self.F} vs expected {expected_F}")

            # 2. Check mass of ring M = 9/64 m
            expected_M = (9 / 64) * self.m
            if not math.isclose(self.M, expected_M, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Mass M: {self.M} vs expected {expected_M}")

            # 3. Check height h = L/5
            expected_h = self.L / 5
            if not math.isclose(self.h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Height h: {self.h} vs expected {expected_h}")

            # 4. Check kinetic energies from ω²
            expected_E_kA = 0.5 * self.M * (self.omega_sq * (8 * self.L / 5) ** 2)  # using ω² * r_A²
            if not math.isclose(self.E_kA, expected_E_kA, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_kA: {self.E_kA} vs expected {expected_E_kA}")

            expected_E_kB = 0.5 * self.m * (self.omega_sq * (4 * self.L / 5) ** 2)
            if not math.isclose(self.E_kB, expected_E_kB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_kB: {self.E_kB} vs expected {expected_E_kB}")

            # 5. Check total work W = 31/30 m g L
            expected_W = (31 / 30) * self.m * self.g * self.L
            if not math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work W: {self.W} vs expected {expected_W}")

            if not logs:
                logs.append("PASS: All trace consistency checks passed.")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")

        return logs

    def test_monotonicity(self):
        """
        Perturb input mass m by +1%, recalculate results using formulas, check trends, then restore.
        """
        logs = []
        original_m = self.m
        perturbed_m = original_m * 1.01  # +1% perturbation

        try:
            # Recalculate expected results for perturbed mass using derived formulas
            # 1. Spring force: F ∝ m
            expected_F_perturbed = (3 * perturbed_m * self.g) / 8
            actual_F_ratio = self.F / ((3 * original_m * self.g) / 8)
            perturbed_F_ratio = expected_F_perturbed / self.F
            if not math.isclose(actual_F_ratio, 1.0, rel_tol=1e-2) or not math.isclose(perturbed_F_ratio, 1.01, rel_tol=1e-2):
                logs.append(f"FAIL: F monotonicity: expected ∝ m, got ratios {actual_F_ratio} and {perturbed_F_ratio}")

            # 2. Mass of ring: M ∝ m
            expected_M_perturbed = (9 / 64) * perturbed_m
            actual_M_ratio = self.M / ((9 / 64) * original_m)
            perturbed_M_ratio = expected_M_perturbed / self.M
            if not math.isclose(actual_M_ratio, 1.0, rel_tol=1e-2) or not math.isclose(perturbed_M_ratio, 1.01, rel_tol=1e-2):
                logs.append(f"FAIL: M monotonicity: expected ∝ m, got ratios {actual_M_ratio} and {perturbed_M_ratio}")

            # 3. Work: W ∝ m (since E_kA, E_kB, and mgh all ∝ m)
            expected_W_perturbed = (31 / 30) * perturbed_m * self.g * self.L
            actual_W_ratio = self.W / ((31 / 30) * original_m * self.g * self.L)
            perturbed_W_ratio = expected_W_perturbed / self.W
            if not math.isclose(actual_W_ratio, 1.0, rel_tol=1e-2) or not math.isclose(perturbed_W_ratio, 1.01, rel_tol=1e-2):
                logs.append(f"FAIL: W monotonicity: expected ∝ m, got ratios {actual_W_ratio} and {perturbed_W_ratio}")

            # 4. Kinetic energies: E_kA ∝ m, E_kB ∝ m
            # Recompute ω² for perturbed mass? ω² is independent of m (from centripetal condition on B).
            # So E_kA ∝ M ∝ m, and E_kB ∝ m.
            expected_E_kA_perturbed = 0.5 * (9/64 * perturbed_m) * (self.omega_sq * (8 * self.L / 5) ** 2)
            actual_E_kA_ratio = self.E_kA / (0.5 * (9/64 * original_m) * (self.omega_sq * (8 * self.L / 5) ** 2))
            perturbed_E_kA_ratio = expected_E_kA_perturbed / self.E_kA
            if not math.isclose(actual_E_kA_ratio, 1.0, rel_tol=1e-2) or not math.isclose(perturbed_E_kA_ratio, 1.01, rel_tol=1e-2):
                logs.append(f"FAIL: E_kA monotonicity: expected ∝ m, got ratios {actual_E_kA_ratio} and {perturbed_E_kA_ratio}")

            expected_E_kB_perturbed = 0.5 * perturbed_m * (self.omega_sq * (4 * self.L / 5) ** 2)
            actual_E_kB_ratio = self.E_kB / (0.5 * original_m * (self.omega_sq * (4 * self.L / 5) ** 2))
            perturbed_E_kB_ratio = expected_E_kB_perturbed / self.E_kB
            if not math.isclose(actual_E_kB_ratio, 1.0, rel_tol=1e-2) or not math.isclose(perturbed_E_kB_ratio, 1.01, rel_tol=1e-2):
                logs.append(f"FAIL: E_kB monotonicity: expected ∝ m, got ratios {actual_E_kB_ratio} and {perturbed_E_kB_ratio}")

            if not logs:
                logs.append("PASS: All monotonicity checks passed.")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Restore original mass
            self.m = original_m
            self.compute_derived_quantities()  # recompute derived quantities with original m

        return logs

    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        """
        all_logs = []
        success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_method()
            if logs and logs[0].startswith(("FAIL", "ERROR")):
                success = False
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)

        return success, all_logs