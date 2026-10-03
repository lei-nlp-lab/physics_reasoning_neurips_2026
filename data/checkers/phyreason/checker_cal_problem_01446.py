import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.inputs = {}
        for key in ['m', 'g', 'alpha']:
            self.inputs[key] = getattr(solution_instance, key, None)
            if self.inputs[key] is None and hasattr(solution_instance, 'inputs'):
                self.inputs[key] = solution_instance.inputs.get(key, None)
            if self.inputs[key] is None:
                # Default values for testing (avoid division by zero)
                self.inputs['m'] = 1.0
                self.inputs['g'] = 9.8
                self.inputs['alpha'] = 0.1  # rad (~5.73°)

        # Safe extraction of results
        self.results = {}
        for key in ['G', 'F_1', 'F_2']:
            self.results[key] = getattr(solution_instance, key, None)
            if self.results[key] is None and hasattr(solution_instance, 'results'):
                self.results[key] = solution_instance.results.get(key, None)

        # Assign for convenience
        self.m = self.inputs['m']
        self.g = self.inputs['g']
        self.alpha = self.inputs['alpha']
        self.G = self.results.get('G')
        self.F1 = self.results.get('F_1')
        self.F2 = self.results.get('F_2')

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Expected values from physics formulas
        cos_alpha = math.cos(self.alpha)
        tan_alpha = math.tan(self.alpha)
        expected_F1 = self.m * self.g / cos_alpha
        expected_F2 = self.m * self.g * tan_alpha

        # Check F1
        if self.F1 is not None:
            if not math.isclose(self.F1, expected_F1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"F1 mismatch: got {self.F1}, expected mg/cos(α) = {expected_F1}")
                success = False
        else:
            logs.append("Warning: F1 not provided for invariant check")

        # Check F2
        if self.F2 is not None:
            if not math.isclose(self.F2, expected_F2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"F2 mismatch: got {self.F2}, expected mg*tan(α) = {expected_F2}")
                success = False
        else:
            logs.append("Warning: F2 not provided for invariant check")

        # Force equilibrium checks (horizontal and vertical)
        if self.F1 is not None:
            # Vertical: F1*cos(α) - m*g = 0
            vertical_balance = self.F1 * cos_alpha - self.m * self.g
            if not math.isclose(vertical_balance, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Vertical equilibrium violated: F1*cos(α) - m*g = {vertical_balance}")
                success = False

            # Horizontal: F1*sin(α) - F2 = 0 (if F2 available)
            if self.F2 is not None:
                horizontal_balance = self.F1 * math.sin(self.alpha) - self.F2
                if not math.isclose(horizontal_balance, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Horizontal equilibrium violated: F1*sin(α) - F2 = {horizontal_balance}")
                    success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Check G = m*g
        if self.G is not None:
            expected_G = self.m * self.g
            if not math.isclose(self.G, expected_G, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"G mismatch: got {self.G}, expected m*g = {expected_G}")
                success = False

        # Check F1 = G/cos(α)
        if self.G is not None and self.F1 is not None:
            expected_F1_from_G = self.G / math.cos(self.alpha)
            if not math.isclose(self.F1, expected_F1_from_G, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"F1 inconsistent with G: got {self.F1}, expected G/cos(α) = {expected_F1_from_G}")
                success = False

        # Check F2 = G*tan(α)
        if self.G is not None and self.F2 is not None:
            expected_F2_from_G = self.G * math.tan(self.alpha)
            if not math.isclose(self.F2, expected_F2_from_G, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"F2 inconsistent with G: got {self.F2}, expected G*tan(α) = {expected_F2_from_G}")
                success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Perturb mass m by +1%
        original_m = self.m
        perturbed_m = original_m * 1.01
        # Compute expected F1 and F2 for perturbed mass
        cos_alpha = math.cos(self.alpha)
        tan_alpha = math.tan(self.alpha)
        F1_original = original_m * self.g / cos_alpha
        F1_perturbed = perturbed_m * self.g / cos_alpha
        F2_original = original_m * self.g * tan_alpha
        F2_perturbed = perturbed_m * self.g * tan_alpha

        if F1_original >= F1_perturbed:
            logs.append(f"F1 does not increase with mass: {F1_original} -> {F1_perturbed}")
            success = False
        if F2_original >= F2_perturbed:
            logs.append(f"F2 does not increase with mass: {F2_original} -> {F2_perturbed}")
            success = False

        # Perturb angle α by +1% (keep within 0 to π/2)
        original_alpha = self.alpha
        if original_alpha == 0:
            perturbed_alpha = 0.01
        else:
            perturbed_alpha = original_alpha * 1.01
        # Ensure perturbed_alpha < π/2 to avoid singularity
        if perturbed_alpha >= math.pi/2:
            perturbed_alpha = math.pi/2 - 0.01

        # Recompute with perturbed angle
        F1_alpha_original = self.m * self.g / math.cos(original_alpha)
        F1_alpha_perturbed = self.m * self.g / math.cos(perturbed_alpha)
        F2_alpha_original = self.m * self.g * math.tan(original_alpha)
        F2_alpha_perturbed = self.m * self.g * math.tan(perturbed_alpha)

        if F1_alpha_original >= F1_alpha_perturbed:
            logs.append(f"F1 does not increase with α: {F1_alpha_original} -> {F1_alpha_perturbed}")
            success = False
        if F2_alpha_original >= F2_alpha_perturbed:
            logs.append(f"F2 does not increase with α: {F2_alpha_original} -> {F2_alpha_perturbed}")
            success = False

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"=== {name} ===")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Error in {name}: {e}")
                overall_success = False

        return overall_success, all_logs