import math
import sympy as sp
from typing import Tuple, List, Optional


class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from the solution instance."""

        # Constants from problem statement
        self.g = 10.0
        self.alpha_deg = 37.0

        # IMPORTANT:
        # Use the same approximation convention as the target answers:
        # sin(37°)=0.6, cos(37°)=0.8
        # This keeps invariants, trace checks, and final targets self-consistent.
        self.sin_alpha = 0.6
        self.cos_alpha = 0.8

        self.M = 3.0
        self.m = 1.0
        self.v0 = 3.0
        self.h = 0.8
        self.mu = 0.5

        # Results with safe access using getattr and .get() fallback
        self.results = {}
        result_keys = [
            'L_min', 'a_3', 'x', 'v_1', 'x_2', 't_2', 'Deltat', 't', 'v',
            'theta', 'a_2', 'x_1', 'a_1', 'v_y'
        ]

        # First try to get results from solution_instance.results dictionary
        results_dict = getattr(solution_instance, 'results', {})

        for key in result_keys:
            # Try direct attribute, then results dict, then default
            value = getattr(solution_instance, key, results_dict.get(key, None))
            if value is None:
                # Try LaTeX keys if present
                latex_key = {
                    'L_min': 'L_{min}',
                    'a_3': 'a_{3}',
                    'x': 'x',
                    'v_1': 'v_{1}',
                    'x_2': 'x_{2}',
                    't_2': 't_2',
                    'Deltat': '\\Delta t',
                    't': 't',
                    'v': 'v',
                    'theta': '\\theta',
                    'a_2': 'a_{2}',
                    'x_1': 'x_{1}',
                    'a_1': 'a_{1}',
                    'v_y': 'v_{y}'
                }.get(key)
                if latex_key:
                    value = getattr(solution_instance, latex_key, results_dict.get(latex_key, None))

            self.results[key] = float(value) if value is not None else None

        # Target answers for verification
        self.target_answers = {
            'v_mag': 5.0,   # Magnitude of velocity before collision
            'L_min': 0.06,  # Minimum board length
            'x_total': 0.43 # Total distance traveled by plank
        }

        # For monotonicity testing
        self.original_m = self.m

    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy comparison for physics values."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def _calculate_fall_time(self) -> float:
        """Calculate time to fall height h."""
        return math.sqrt(2 * self.h / self.g)

    def _calculate_vertical_velocity(self, t: float) -> float:
        """Calculate vertical velocity after falling time t."""
        return self.g * t

    def _calculate_velocity_magnitude(self, vx: float, vy: float) -> float:
        """Calculate magnitude of velocity from components."""
        return math.sqrt(vx**2 + vy**2)

    def _calculate_angle(self, vx: float, vy: float) -> float:
        """Calculate angle relative to vertical."""
        return math.degrees(math.atan(vx / vy)) if vy != 0 else 90.0

    def _calculate_a1(self) -> float:
        """Acceleration of plank before block lands."""
        return self.g * self.sin_alpha - self.mu * self.g * self.cos_alpha

    def _calculate_a2(self) -> float:
        """Acceleration of plank with block on it."""
        numerator = (
            self.M * self.g * self.sin_alpha
            - self.mu * (self.M + self.m) * self.g * self.cos_alpha
        )
        return numerator / self.M

    def _calculate_a3(self) -> float:
        """Acceleration of block sliding down incline."""
        return self.g * self.sin_alpha

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants."""
        logs = []
        success = True

        # 1. Time of fall should match free fall physics
        t_calc = self._calculate_fall_time()
        if self.results['t'] is not None:
            if not self._is_close(self.results['t'], t_calc):
                logs.append(f"FAIL: Fall time mismatch. Expected {t_calc:.3f}, got {self.results['t']:.3f}")
                success = False
            else:
                logs.append(f"PASS: Fall time {self.results['t']:.3f}s matches physics")

        # 2. Vertical velocity should be g*t
        if self.results['v_y'] is not None and self.results['t'] is not None:
            v_y_calc = self._calculate_vertical_velocity(self.results['t'])
            if not self._is_close(self.results['v_y'], v_y_calc):
                logs.append(f"FAIL: Vertical velocity mismatch. Expected {v_y_calc:.3f}, got {self.results['v_y']:.3f}")
                success = False
            else:
                logs.append(f"PASS: Vertical velocity {self.results['v_y']:.3f}m/s consistent")

        # 3. Velocity magnitude from components
        if self.results['v'] is not None and self.results['v_y'] is not None:
            v_mag_calc = self._calculate_velocity_magnitude(self.v0, self.results['v_y'])
            if not self._is_close(self.results['v'], v_mag_calc):
                logs.append(f"FAIL: Velocity magnitude mismatch. Expected {v_mag_calc:.3f}, got {self.results['v']:.3f}")
                success = False
            else:
                logs.append(f"PASS: Velocity magnitude {self.results['v']:.3f}m/s consistent")

        # 4. Angle calculation
        if self.results['theta'] is not None and self.results['v_y'] is not None:
            theta_calc = self._calculate_angle(self.v0, self.results['v_y'])
            if not self._is_close(self.results['theta'], theta_calc):
                logs.append(f"FAIL: Angle mismatch. Expected {theta_calc:.1f}°, got {self.results['theta']:.1f}°")
                success = False
            else:
                logs.append(f"PASS: Angle {self.results['theta']:.1f}° consistent")

        # 5. Accelerations should match Newton's laws
        a1_calc = self._calculate_a1()
        if self.results['a_1'] is not None:
            if not self._is_close(self.results['a_1'], a1_calc):
                logs.append(f"FAIL: a1 mismatch. Expected {a1_calc:.3f}, got {self.results['a_1']:.3f}")
                success = False
            else:
                logs.append(f"PASS: a1 = {self.results['a_1']:.3f}m/s² matches Newton's laws")

        a2_calc = self._calculate_a2()
        if self.results['a_2'] is not None:
            if not self._is_close(self.results['a_2'], a2_calc):
                logs.append(f"FAIL: a2 mismatch. Expected {a2_calc:.3f}, got {self.results['a_2']:.3f}")
                success = False
            else:
                logs.append(f"PASS: a2 = {self.results['a_2']:.3f}m/s² matches Newton's laws")

        a3_calc = self._calculate_a3()
        if self.results['a_3'] is not None:
            if not self._is_close(self.results['a_3'], a3_calc):
                logs.append(f"FAIL: a3 mismatch. Expected {a3_calc:.3f}, got {self.results['a_3']:.3f}")
                success = False
            else:
                logs.append(f"PASS: a3 = {self.results['a_3']:.3f}m/s² matches Newton's laws")

        # 6. Velocity of plank at collision: v1 = a1 * t
        if (
            self.results['v_1'] is not None
            and self.results['a_1'] is not None
            and self.results['t'] is not None
        ):
            v1_calc = self.results['a_1'] * self.results['t']
            if not self._is_close(self.results['v_1'], v1_calc):
                logs.append(f"FAIL: v1 mismatch. Expected {v1_calc:.3f}, got {self.results['v_1']:.3f}")
                success = False
            else:
                logs.append(f"PASS: v1 = {self.results['v_1']:.3f}m/s consistent")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate results."""
        logs = []
        success = True

        # 1. Check Δt calculation from kinematics
        if (
            self.results['Deltat'] is not None
            and self.results['v_1'] is not None
            and self.results['a_2'] is not None
            and self.results['a_3'] is not None
        ):
            # Equation: a3 * Δt = v1 + a2 * Δt  =>  Δt = v1 / (a3 - a2)
            if abs(self.results['a_3'] - self.results['a_2']) > 1e-9:
                Deltat_calc = self.results['v_1'] / (self.results['a_3'] - self.results['a_2'])
                if not self._is_close(self.results['Deltat'], Deltat_calc):
                    logs.append(f"FAIL: Δt mismatch. Expected {Deltat_calc:.3f}, got {self.results['Deltat']:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Δt = {self.results['Deltat']:.3f}s consistent with kinematics")

        # 2. Check L_min calculation
        if (
            self.results['L_min'] is not None
            and self.results['v_1'] is not None
            and self.results['Deltat'] is not None
            and self.results['a_2'] is not None
            and self.results['a_3'] is not None
        ):
            # L_min = v1*Δt + 0.5*a2*Δt² - 0.5*a3*Δt²
            L_min_calc = (
                self.results['v_1'] * self.results['Deltat']
                + 0.5 * self.results['a_2'] * self.results['Deltat']**2
                - 0.5 * self.results['a_3'] * self.results['Deltat']**2
            )
            if not self._is_close(self.results['L_min'], L_min_calc):
                logs.append(f"FAIL: L_min mismatch. Expected {L_min_calc:.3f}, got {self.results['L_min']:.3f}")
                success = False
            else:
                logs.append(f"PASS: L_min = {self.results['L_min']:.3f}m consistent")

        # 3. Check x1 calculation
        if (
            self.results['x_1'] is not None
            and self.results['v_1'] is not None
            and self.results['t'] is not None
        ):
            # x1 = 0.5 * v1 * t
            x1_calc = 0.5 * self.results['v_1'] * self.results['t']
            if not self._is_close(self.results['x_1'], x1_calc):
                logs.append(f"FAIL: x1 mismatch. Expected {x1_calc:.3f}, got {self.results['x_1']:.3f}")
                success = False
            else:
                logs.append(f"PASS: x1 = {self.results['x_1']:.3f}m consistent")

        # 4. Check t2 and x2 from simultaneous equations
        if (
            self.results['t_2'] is not None
            and self.results['x_2'] is not None
            and self.results['v_1'] is not None
            and self.results['a_2'] is not None
            and self.results['a_3'] is not None
        ):
            # Two equations:
            # 1) x2 = v1*t2 + 0.5*a2*t2²
            # 2) x2 = 0.5*a3*t2²
            x2_from_eq1 = (
                self.results['v_1'] * self.results['t_2']
                + 0.5 * self.results['a_2'] * self.results['t_2']**2
            )
            x2_from_eq2 = 0.5 * self.results['a_3'] * self.results['t_2']**2

            if not (
                self._is_close(self.results['x_2'], x2_from_eq1)
                and self._is_close(self.results['x_2'], x2_from_eq2)
            ):
                logs.append(
                    f"FAIL: x2 inconsistent. From eq1: {x2_from_eq1:.3f}, "
                    f"from eq2: {x2_from_eq2:.3f}, got {self.results['x_2']:.3f}"
                )
                success = False
            else:
                logs.append(f"PASS: x2 = {self.results['x_2']:.3f}m satisfies both equations")

            # Non-zero solution: t2 = 2*v1/(a3 - a2)
            if abs(self.results['a_3'] - self.results['a_2']) > 1e-9:
                t2_calc = 2 * self.results['v_1'] / (self.results['a_3'] - self.results['a_2'])
                if not self._is_close(self.results['t_2'], t2_calc):
                    logs.append(f"FAIL: t2 mismatch. Expected {t2_calc:.3f}, got {self.results['t_2']:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: t2 = {self.results['t_2']:.3f}s solves the quadratic")

        # 5. Check total distance
        if (
            self.results['x'] is not None
            and self.results['x_1'] is not None
            and self.results['x_2'] is not None
        ):
            x_total_calc = self.results['x_1'] + self.results['x_2']
            if not self._is_close(self.results['x'], x_total_calc):
                logs.append(f"FAIL: Total distance mismatch. Expected {x_total_calc:.3f}, got {self.results['x']:.3f}")
                success = False
            else:
                logs.append(f"PASS: Total distance x = {self.results['x']:.3f}m = x1 + x2")

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Test how quantities change when m increases, then restore original m.

        Expected trend:
        - a1 unchanged
        - a2 decreases
        - a3 unchanged
        - L_min decreases
        """
        logs = []
        success = True

        # Perturb m by +10%
        perturbation = 0.1
        self.m = self.original_m * (1 + perturbation)

        # Recalculate accelerations with perturbed m
        a1_pert = self._calculate_a1()
        a2_pert = self._calculate_a2()
        a3_pert = self._calculate_a3()

        # Get original accelerations
        self.m = self.original_m
        a1_orig = self._calculate_a1()
        a2_orig = self._calculate_a2()
        a3_orig = self._calculate_a3()

        # Restore perturbed m for further calculations
        self.m = self.original_m * (1 + perturbation)

        # a1 should remain the same
        if not self._is_close(a1_pert, a1_orig):
            logs.append("FAIL: a1 changed with m perturbation. Should be independent of m.")
            success = False
        else:
            logs.append("PASS: a1 unchanged with m perturbation (as expected)")

        # a2 should decrease
        if a2_pert >= a2_orig:
            logs.append(f"FAIL: a2 should decrease with increased m. Original: {a2_orig:.3f}, perturbed: {a2_pert:.3f}")
            success = False
        else:
            logs.append("PASS: a2 decreased with increased m (as expected)")

        # a3 unchanged
        if not self._is_close(a3_pert, a3_orig):
            logs.append("FAIL: a3 changed with m perturbation. Should be independent of m.")
            success = False
        else:
            logs.append("PASS: a3 unchanged with m perturbation (as expected)")

        # Calculate L_min with perturbed parameters
        t = self._calculate_fall_time()
        v1_pert = a1_orig * t  # a1 unchanged
        Deltat_pert = v1_pert / (a3_orig - a2_pert) if (a3_orig - a2_pert) > 0 else 0.0
        L_min_pert = (
            v1_pert * Deltat_pert
            + 0.5 * a2_pert * Deltat_pert**2
            - 0.5 * a3_orig * Deltat_pert**2
        )

        # Get original L_min from results
        self.m = self.original_m
        L_min_orig = self.results.get('L_min')

        if L_min_orig is not None:
            # Correct trend: L_min should decrease when m increases
            if L_min_pert >= L_min_orig:
                logs.append(f"FAIL: L_min should decrease with increased m. Original: {L_min_orig:.3f}, perturbed: {L_min_pert:.3f}")
                success = False
            else:
                logs.append("PASS: L_min decreased with increased m (as expected)")

        # Restore original m
        self.m = self.original_m
        logs.append("PASS: Monotonicity trends checked and original m restored")

        return success, logs

    def test_target_answers(self) -> Tuple[bool, List[str]]:
        """Check final answers against target values."""
        logs = []
        success = True

        # Check velocity magnitude
        if self.results['v'] is not None:
            if not self._is_close(self.results['v'], self.target_answers['v_mag']):
                logs.append(
                    f"FAIL: Velocity magnitude {self.results['v']:.3f}m/s "
                    f"≠ target {self.target_answers['v_mag']:.3f}m/s"
                )
                success = False
            else:
                logs.append("PASS: Velocity magnitude matches target")

        # Check minimum length
        if self.results['L_min'] is not None:
            if not self._is_close(self.results['L_min'], self.target_answers['L_min']):
                logs.append(
                    f"FAIL: L_min {self.results['L_min']:.3f}m "
                    f"≠ target {self.target_answers['L_min']:.3f}m"
                )
                success = False
            else:
                logs.append("PASS: L_min matches target")

        # Check total distance
        if self.results['x'] is not None:
            if not self._is_close(self.results['x'], self.target_answers['x_total']):
                logs.append(
                    f"FAIL: Total distance {self.results['x']:.3f}m "
                    f"≠ target {self.target_answers['x_total']:.3f}m"
                )
                success = False
            else:
                logs.append("PASS: Total distance matches target")

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True

        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("=== Testing Physics Invariants ===")
        all_logs.extend(logs)
        overall_success = overall_success and success

        # Test 2: Trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Testing Trace Consistency ===")
        all_logs.extend(logs)
        overall_success = overall_success and success

        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Testing Monotonicity ===")
        all_logs.extend(logs)
        overall_success = overall_success and success

        # Test 4: Target answers
        success, logs = self.test_target_answers()
        all_logs.append("\n=== Testing Target Answers ===")
        all_logs.extend(logs)
        overall_success = overall_success and success

        return overall_success, all_logs