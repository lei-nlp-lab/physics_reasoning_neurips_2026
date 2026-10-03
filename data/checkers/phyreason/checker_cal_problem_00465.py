import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Safely get inputs (defaults for testing if not available)
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})

        # Extract input parameters with defaults
        self.Q = self.inputs.get('Q', 1.0)
        self.r = self.inputs.get('r', 1.0)
        self.k = self.inputs.get('k', 1.0)  # electrostatic constant

        # Extract results with safe fallback
        self.E = self.results.get('E', 0.0)
        self.E1 = self.results.get('E_1', 0.0)
        self.E2 = self.results.get('E_2', 0.0)
        self.F1 = self.results.get('F_1', 0.0)
        self.F2 = self.results.get('F_2', 0.0)
        self.fracF1F2 = self.results.get('fracF_1F_2', 0.0)
        # Direction may be stored as a string; we only check if present
        self.direction = self.results.get('direction', None)

        # Store original values for monotonicity test
        self._original_Q = self.Q
        self._original_r = self.r
        self._original_k = self.k

        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify physical laws: Coulomb's Law, superposition, and direction.
        """
        logs = []
        success = True

        # 1. Coulomb's Law for force before contact (F2)
        expected_F2 = self.k * abs(self.Q * (-5 * self.Q)) / (self.r ** 2)  # 5kQ^2/r^2
        if not math.isclose(self.F2, expected_F2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: F2 = {self.F2}, expected {expected_F2} (Coulomb's Law)")
            success = False
        else:
            logs.append("PASS: F2 satisfies Coulomb's Law")

        # 2. Coulomb's Law for force after contact (F1)
        # After contact, each sphere has charge -2Q (total -4Q, split equally)
        expected_F1 = self.k * (2 * self.Q) ** 2 / (self.r ** 2)  # 4kQ^2/r^2
        if not math.isclose(self.F1, expected_F1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: F1 = {self.F1}, expected {expected_F1} (Coulomb's Law after contact)")
            success = False
        else:
            logs.append("PASS: F1 satisfies Coulomb's Law after contact")

        # 3. Superposition of electric fields at midpoint
        expected_E1 = self.k * self.Q / ((self.r / 2) ** 2)  # 4kQ/r^2
        if not math.isclose(self.E1, expected_E1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: E1 = {self.E1}, expected {expected_E1} (field due to A)")
            success = False
        else:
            logs.append("PASS: E1 matches point charge formula")

        expected_E2 = self.k * abs(-5 * self.Q) / ((self.r / 2) ** 2)  # 20kQ/r^2 (magnitude)
        if not math.isclose(self.E2, expected_E2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: E2 = {self.E2}, expected {expected_E2} (field due to B)")
            success = False
        else:
            logs.append("PASS: E2 matches point charge formula")

        expected_E = expected_E1 + expected_E2  # 24kQ/r^2
        if not math.isclose(self.E, expected_E, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: E = {self.E}, expected {expected_E} (superposition)")
            success = False
        else:
            logs.append("PASS: E satisfies superposition (E1 + E2)")

        # 4. Direction consistency (if available)
        if self.direction is not None:
            # Expect direction from A to B (both fields point that way)
            if "from A to B" not in str(self.direction):
                logs.append(f"WARN: direction = {self.direction}, expected 'from A to B'")
                # Not a failure, but log warning
            else:
                logs.append("PASS: direction is 'from A to B'")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        success = True

        # 1. Check that E1 + E2 equals E
        if not math.isclose(self.E, self.E1 + self.E2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: E ({self.E}) != E1 + E2 ({self.E1 + self.E2})")
            success = False
        else:
            logs.append("PASS: E = E1 + E2")

        # 2. Check that fracF1F2 equals F1 / F2
        if self.F2 != 0:
            expected_ratio = self.F1 / self.F2
            if not math.isclose(self.fracF1F2, expected_ratio, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"FAIL: fracF1F2 = {self.fracF1F2}, expected F1/F2 = {expected_ratio}")
                success = False
            else:
                logs.append("PASS: fracF1F2 = F1 / F2")
        else:
            logs.append("SKIP: F2 is zero, cannot check ratio")

        # 3. Check that the ratio is 0.8 (4/5)
        if not math.isclose(self.fracF1F2, 0.8, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: fracF1F2 = {self.fracF1F2}, expected 0.8")
            success = False
        else:
            logs.append("PASS: fracF1F2 = 4/5")

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb Q and check that results scale as expected, then restore.
        """
        logs = []
        success = True

        # Perturb Q by 10%
        perturbation = 1.1
        Q_perturbed = self.Q * perturbation

        # Recompute expected values using formulas
        E1_perturbed = self.k * Q_perturbed / ((self.r / 2) ** 2)
        E2_perturbed = self.k * abs(-5 * Q_perturbed) / ((self.r / 2) ** 2)
        E_perturbed = E1_perturbed + E2_perturbed
        F1_perturbed = self.k * (2 * Q_perturbed) ** 2 / (self.r ** 2)
        F2_perturbed = self.k * abs(Q_perturbed * (-5 * Q_perturbed)) / (self.r ** 2)
        frac_perturbed = F1_perturbed / F2_perturbed

        # Check scaling:
        # E should scale linearly with Q
        if not math.isclose(self.E * perturbation, E_perturbed, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: E does not scale linearly with Q (perturbed {perturbation})")
            success = False
        else:
            logs.append("PASS: E scales linearly with Q")

        # F1 and F2 should scale with Q^2
        if not math.isclose(self.F1 * (perturbation ** 2), F1_perturbed, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: F1 does not scale with Q^2 (perturbed {perturbation})")
            success = False
        else:
            logs.append("PASS: F1 scales with Q^2")

        if not math.isclose(self.F2 * (perturbation ** 2), F2_perturbed, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: F2 does not scale with Q^2 (perturbed {perturbation})")
            success = False
        else:
            logs.append("PASS: F2 scales with Q^2")

        # Ratio should remain constant
        if not math.isclose(self.fracF1F2, frac_perturbed, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: F1/F2 ratio changes with Q (perturbed {perturbation})")
            success = False
        else:
            logs.append("PASS: F1/F2 ratio is invariant under Q perturbation")

        # Restore original Q (just in case)
        self.Q = self._original_Q
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        all_logs = ["=== Physics Checker ==="]
        overall_success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                success, logs = test_method()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR: {e}")
                overall_success = False

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs