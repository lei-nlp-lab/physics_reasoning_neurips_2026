import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Follows robust attribute access to avoid sandbox errors.
        """
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 8.0e-4)
        self.L = getattr(solution_instance, 'L', 0.1)
        self.d = getattr(solution_instance, 'd', 0.12)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.k = getattr(solution_instance, 'k', 9.0e9)

        # Extract results from results dictionary or attributes
        results = getattr(solution_instance, 'results', {})
        self.F = results.get('F', getattr(solution_instance, 'F', 0.0))
        self.q_A = results.get('q_A', getattr(solution_instance, 'q_A', 0.0))
        self.q_B = results.get('q_B', getattr(solution_instance, 'q_B', 0.0))
        self.E = results.get('E', getattr(solution_instance, 'E', 0.0))
        self.sinθ = results.get('sintheta', getattr(solution_instance, 'sinθ', 0.0))
        self.tanθ = results.get('tantheta', getattr(solution_instance, 'tanθ', 0.0))

        # Store the solution instance for possible reuse (e.g., monotonicity)
        self.solution = solution_instance

        # Expected values computed from inputs for cross-checking
        self.expected_sinθ = self.d / (2.0 * self.L)
        self.expected_tanθ = self.expected_sinθ / math.sqrt(1.0 - self.expected_sinθ**2)
        self.expected_F = self.m * self.g * self.expected_tanθ
        # Charge after contact: q_A = q_B = sqrt(F * d^2 / k)
        if self.expected_F > 0:
            self.expected_q = math.sqrt(self.expected_F * self.d**2 / self.k)
        else:
            self.expected_q = 0.0
        # Electric field at midpoint for two equal like charges is zero
        self.expected_E = 0.0

        # Tolerance for numerical comparisons (2% relative, 2% absolute)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental physical invariants:
        1. Geometric relationship: sinθ = d/(2L)
        2. Trigonometric identity: tanθ = sinθ/√(1-sin²θ)
        3. Force equilibrium: F = m * g * tanθ
        4. Coulomb's Law: F = k * q_A * q_B / d² and q_A == q_B
        5. Superposition principle: E = 0 at midpoint
        """
        logs = []
        success = True

        # 1. Geometric relationship
        if not math.isclose(self.sinθ, self.expected_sinθ, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Geometric invariant violated: sinθ = {self.sinθ}, expected {self.expected_sinθ}")
            success = False

        # 2. Trigonometric identity
        if not math.isclose(self.tanθ, self.expected_tanθ, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Trigonometric invariant violated: tanθ = {self.tanθ}, expected {self.expected_tanθ}")
            success = False

        # 3. Force equilibrium
        if not math.isclose(self.F, self.expected_F, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Force equilibrium violated: F = {self.F}, expected {self.expected_F}")
            success = False

        # 4. Coulomb's Law and charge equality
        if not math.isclose(self.q_A, self.q_B, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Charge equality violated: q_A = {self.q_A}, q_B = {self.q_B}")
            success = False
        
        # Coulomb force - FIXED: check should happen regardless of whether q_A and q_B are zero
        # If F is non-zero, q_A and q_B should not be zero
        if abs(self.F) > 1e-15:  # If F is significantly non-zero
            if abs(self.q_A) < 1e-15 or abs(self.q_B) < 1e-15:
                logs.append(f"Coulomb's Law violated: F = {self.F} is non-zero but q_A = {self.q_A}, q_B = {self.q_B} are zero")
                success = False
            else:
                F_coulomb = self.k * self.q_A * self.q_B / (self.d**2)
                if not math.isclose(self.F, F_coulomb, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    logs.append(f"Coulomb's Law violated: F = {self.F}, k*q_A*q_B/d² = {F_coulomb}")
                    success = False
        else:
            # If F is zero or very close to zero, q_A and q_B should also be zero (or charges opposite and equal)
            # For this problem, after contact charges are equal, so if F=0 then q_A=q_B=0
            if abs(self.q_A) > 1e-15 or abs(self.q_B) > 1e-15:
                logs.append(f"Coulomb's Law violated: F = {self.F} is zero but q_A = {self.q_A}, q_B = {self.q_B} are non-zero")
                success = False

        # 5. Electric field at midpoint
        if not math.isclose(self.E, self.expected_E, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Superposition principle violated: E = {self.E}, expected {self.expected_E}")
            success = False

        if success:
            logs.append("All invariants passed.")
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True

        # Step 1: sinθ and tanθ consistency
        if not math.isclose(self.sinθ, self.expected_sinθ, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Step 1 inconsistency: sinθ = {self.sinθ}, expected from geometry {self.expected_sinθ}")
            success = False
        if not math.isclose(self.tanθ, self.expected_tanθ, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Step 1 inconsistency: tanθ = {self.tanθ}, expected from trig {self.expected_tanθ}")
            success = False

        # Step 2: Force from equilibrium
        if not math.isclose(self.F, self.expected_F, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Step 2 inconsistency: F = {self.F}, expected from equilibrium {self.expected_F}")
            success = False

        # Step 3: Charge from Coulomb's Law
        if not math.isclose(self.q_A, self.expected_q, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Step 3 inconsistency: q_A = {self.q_A}, expected from Coulomb {self.expected_q}")
            success = False
        if not math.isclose(self.q_B, self.expected_q, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Step 3 inconsistency: q_B = {self.q_B}, expected from Coulomb {self.expected_q}")
            success = False

        # Step 4: Electric field at midpoint
        if not math.isclose(self.E, self.expected_E, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Step 4 inconsistency: E = {self.E}, expected from superposition {self.expected_E}")
            success = False

        if success:
            logs.append("Trace consistency passed.")
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb mass m by 1% and verify expected changes in F and q.
        """
        logs = []
        success = True

        # Perturbation factor
        factor = 1.01
        m_pert = self.m * factor

        # Recompute expected values with perturbed mass (geometry unchanged)
        sinθ_pert = self.d / (2.0 * self.L)  # unchanged
        tanθ_pert = sinθ_pert / math.sqrt(1.0 - sinθ_pert**2)  # unchanged
        F_pert = m_pert * self.g * tanθ_pert
        q_pert = math.sqrt(F_pert * self.d**2 / self.k)

        # Expected ratios
        expected_F_ratio = factor  # F ∝ m
        expected_q_ratio = math.sqrt(factor)  # q ∝ √m

        # Actual ratios (avoid division by zero)
        if abs(self.F) > 1e-15:
            F_ratio = F_pert / self.F
            if not math.isclose(F_ratio, expected_F_ratio, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Monotonicity failed for F: ratio = {F_ratio}, expected {expected_F_ratio}")
                success = False
        else:
            logs.append("Cannot test F monotonicity because original F is zero.")
            success = False

        if abs(self.q_A) > 1e-15:
            q_ratio = q_pert / self.q_A
            if not math.isclose(q_ratio, expected_q_ratio, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Monotonicity failed for q: ratio = {q_ratio}, expected {expected_q_ratio}")
                success = False
        else:
            logs.append("Cannot test q monotonicity because original q is zero.")
            success = False

        # Electric field should remain zero (charges remain equal)
        if not math.isclose(0.0, self.E, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Electric field should remain zero after perturbation, but original E = {self.E}")
            success = False

        if success:
            logs.append("Monotonicity test passed.")
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and collect logs.
        """
        all_logs = []
        overall_success = True

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
                all_logs.append(f"=== {test_name} ===")
                all_logs.append(f"Test failed with exception: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs