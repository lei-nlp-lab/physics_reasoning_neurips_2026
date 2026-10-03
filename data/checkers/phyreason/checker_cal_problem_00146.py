import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all required inputs and results."""
        # Input parameters (problem constants)
        self.d = getattr(solution_instance, 'd', getattr(solution_instance, 'inputs', {}).get('d', 1.0))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'inputs', {}).get('q', 1.0))
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'inputs', {}).get('E', 1.0))
        
        # Results mapping with safe access
        results_dict = getattr(solution_instance, 'results', {})
        self.v = getattr(solution_instance, 'v', results_dict.get('v', 0.0))
        self.B = getattr(solution_instance, 'B', results_dict.get('B', 0.0))
        self.R = getattr(solution_instance, 'R', results_dict.get('R', 0.0))
        self.R1 = getattr(solution_instance, 'R_1', results_dict.get('R_1', 0.0))
        self.alpha = getattr(solution_instance, 'alpha', results_dict.get('alpha', 0.0))
        self.v1 = getattr(solution_instance, 'v_1', results_dict.get('v_1', 0.0))
        self.v0 = getattr(solution_instance, 'v_0', results_dict.get('v_0', 0.0))
        self.E_prime = getattr(solution_instance, "E'", results_dict.get("E'", 0.0))
        self.v2 = getattr(solution_instance, 'v_2', results_dict.get('v_2', 0.0))
        self.R2 = getattr(solution_instance, 'R_2', results_dict.get('R_2', 0.0))
        self.x_center = getattr(solution_instance, 'x', results_dict.get('x', 0.0))
        self.y_center = getattr(solution_instance, 'y', results_dict.get('y', 0.0))
        self.l = getattr(solution_instance, 'l', results_dict.get('l', 0.0))
        self.theta = getattr(solution_instance, 'theta', results_dict.get('theta', 0.0))
        
        # Store original values for monotonicity tests
        self._original_m = self.m
        self._original_q = self.q
        self._original_E = self.E
        self._original_d = self.d

    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify energy conservation, Lorentz force, and Newton's laws."""
        logs = []
        success = True
        
        # 1. Work-energy theorem for first acceleration (step 1)
        try:
            expected_v = math.sqrt(4 * self.q * self.E * self.d / self.m)
            if not self._is_close(self.v, expected_v):
                logs.append(f"Energy conservation violation: v = {self.v}, expected sqrt(4qEd/m) = {expected_v}")
                success = False
        except (ZeroDivisionError, ValueError) as e:
            logs.append(f"Work-energy check failed: {e}")
            success = False

        # 2. Lorentz force as centripetal force (step 2)
        try:
            # Check qvB = mv²/R
            lhs = self.q * self.v * self.B
            rhs = self.m * self.v**2 / self.R
            if not self._is_close(lhs, rhs):
                logs.append(f"Lorentz force mismatch: qvB = {lhs}, mv²/R = {rhs}")
                success = False
                
            # Check radius R = d/3 (step 3)
            if not self._is_close(self.R, self.d / 3):
                logs.append(f"Radius mismatch: R = {self.R}, expected d/3 = {self.d/3}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Lorentz force check failed: {e}")
            success = False

        # 3. Geometry check for R1 (step 5)
        try:
            expected_R1 = 5 * self.d / 2
            if not self._is_close(self.R1, expected_R1):
                logs.append(f"Geometry mismatch: R1 = {self.R1}, expected 5d/2 = {expected_R1}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Geometry check failed: {e}")
            success = False

        # 4. Lorentz force for second circular motion (step 7)
        try:
            lhs = self.q * self.v1 * self.B
            rhs = self.m * self.v1**2 / self.R1
            if not self._is_close(lhs, rhs):
                logs.append(f"Second Lorentz force mismatch: qv1B = {lhs}, mv1²/R1 = {rhs}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Second Lorentz force check failed: {e}")
            success = False

        # 5. Work-energy for second acceleration (step 8)
        try:
            delta_KE = 0.5 * self.m * (self.v1**2 - self.v0**2)
            work = self.q * self.E_prime * 2 * self.d
            if not self._is_close(delta_KE, work):
                logs.append(f"Second work-energy mismatch: ΔKE = {delta_KE}, qE'·2d = {work}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Second work-energy check failed: {e}")
            success = False

        # 6. Work-energy for third acceleration (step 13)
        try:
            delta_KE2 = 0.5 * self.m * (self.v2**2 - self.v1**2)
            work2 = self.q * self.E_prime * 2 * self.d
            if not self._is_close(delta_KE2, work2):
                logs.append(f"Third work-energy mismatch: ΔKE = {delta_KE2}, qE'·2d = {work2}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Third work-energy check failed: {e}")
            success = False

        # 7. Lorentz force for third circular motion (step 15)
        try:
            lhs = self.q * self.v2 * self.B
            rhs = self.m * self.v2**2 / self.R2
            if not self._is_close(lhs, rhs):
                logs.append(f"Third Lorentz force mismatch: qv2B = {lhs}, mv2²/R2 = {rhs}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Third Lorentz force check failed: {e}")
            success = False

        # 8. Geometry checks (steps 16-17)
        try:
            if not self._is_close(self.x_center, 5 * self.d / 2):
                logs.append(f"Center x mismatch: {self.x_center}, expected 5d/2 = {5*self.d/2}")
                success = False
            if not self._is_close(self.y_center, 4 * self.d):
                logs.append(f"Center y mismatch: {self.y_center}, expected 4d = {4*self.d}")
                success = False
            expected_l = math.sqrt(65) * self.d / 2
            if not self._is_close(self.l, expected_l):
                logs.append(f"Distance l mismatch: {self.l}, expected √65·d/2 = {expected_l}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Geometry checks failed: {e}")
            success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check that B from step 4 matches derived formula
        try:
            expected_B = 6 * math.sqrt(self.m * self.E / (self.q * self.d))
            if not self._is_close(self.B, expected_B):
                logs.append(f"B consistency: B = {self.B}, expected 6√(mE/qd) = {expected_B}")
                success = False
        except (ZeroDivisionError, ValueError) as e:
            logs.append(f"B consistency check failed: {e}")
            success = False

        # Check that v1, v0, E' from step 12 match expectations
        try:
            expected_v1 = 15 * math.sqrt(self.q * self.d * self.E / self.m)
            expected_v0 = 9 * math.sqrt(self.q * self.d * self.E / self.m)
            expected_E_prime = 36 * self.E
            
            if not self._is_close(self.v1, expected_v1):
                logs.append(f"v1 consistency: v1 = {self.v1}, expected 15√(qEd/m) = {expected_v1}")
                success = False
            if not self._is_close(self.v0, expected_v0):
                logs.append(f"v0 consistency: v0 = {self.v0}, expected 9√(qEd/m) = {expected_v0}")
                success = False
            if not self._is_close(self.E_prime, expected_E_prime):
                logs.append(f"E' consistency: E' = {self.E_prime}, expected 36E = {expected_E_prime}")
                success = False
        except (ZeroDivisionError, ValueError) as e:
            logs.append(f"Step 12 consistency check failed: {e}")
            success = False

        # Check v2 from step 14
        try:
            expected_v2 = 3 * math.sqrt(41 * self.q * self.E * self.d / self.m)
            if not self._is_close(self.v2, expected_v2):
                logs.append(f"v2 consistency: v2 = {self.v2}, expected 3√(41qEd/m) = {expected_v2}")
                success = False
        except (ZeroDivisionError, ValueError) as e:
            logs.append(f"v2 consistency check failed: {e}")
            success = False

        # Check R2 from step 15
        try:
            expected_R2 = math.sqrt(41) * self.d / 2
            if not self._is_close(self.R2, expected_R2):
                logs.append(f"R2 consistency: R2 = {self.R2}, expected √41·d/2 = {expected_R2}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"R2 consistency check failed: {e}")
            success = False

        # Check angle relationship (step 6)
        try:
            if not self._is_close(self.theta, 53):
                logs.append(f"Theta consistency: θ = {self.theta}, expected 53°")
                success = False
            if not self._is_close(self.alpha, 37):
                logs.append(f"Alpha consistency: α = {self.alpha}, expected 37°")
                success = False
            if not self._is_close(self.theta + self.alpha, 90):
                logs.append(f"Angle sum: θ + α = {self.theta + self.alpha}, expected 90°")
                success = False
        except Exception as e:
            logs.append(f"Angle consistency check failed: {e}")
            success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb inputs and check physical trends."""
        logs = []
        success = True
        
        # Test 1: Increase mass m, velocity should decrease
        try:
            original_m = self.m
            perturbed_m = original_m * 1.1  # +10%
            self.m = perturbed_m
            
            # Recompute v using formula from step 1
            v_perturbed = math.sqrt(4 * self.q * self.E * self.d / self.m)
            
            # Restore original mass
            self.m = original_m
            
            if v_perturbed > self.v:
                logs.append(f"Monotonicity violation: Increasing m should decrease v, but v increased from {self.v} to {v_perturbed}")
                success = False
        except (ZeroDivisionError, ValueError) as e:
            logs.append(f"Mass monotonicity check failed: {e}")
            success = False
            self.m = self._original_m  # Restore on error

        # Test 2: Increase charge q, velocity should increase
        try:
            original_q = self.q
            perturbed_q = original_q * 1.1  # +10%
            self.q = perturbed_q
            
            v_perturbed = math.sqrt(4 * self.q * self.E * self.d / self.m)
            
            self.q = original_q
            
            if v_perturbed < self.v:
                logs.append(f"Monotonicity violation: Increasing q should increase v, but v decreased from {self.v} to {v_perturbed}")
                success = False
        except (ZeroDivisionError, ValueError) as e:
            logs.append(f"Charge monotonicity check failed: {e}")
            success = False
            self.q = self._original_q

        # Test 3: Increase magnetic field B, radius should decrease (from Lorentz force)
        try:
            # Using R = mv/(qB) formula
            original_B = self.B
            perturbed_B = original_B * 1.1
            
            R_perturbed = self.m * self.v / (self.q * perturbed_B)
            
            if R_perturbed > self.R:
                logs.append(f"Monotonicity violation: Increasing B should decrease R, but R increased from {self.R} to {R_perturbed}")
                success = False
        except ZeroDivisionError as e:
            logs.append(f"Magnetic field monotonicity check failed: {e}")
            success = False

        # Test 4: Increase electric field E, velocity should increase
        try:
            original_E = self.E
            perturbed_E = original_E * 1.1
            self.E = perturbed_E
            
            v_perturbed = math.sqrt(4 * self.q * self.E * self.d / self.m)
            
            self.E = original_E
            
            if v_perturbed < self.v:
                logs.append(f"Monotonicity violation: Increasing E should increase v, but v decreased from {self.v} to {v_perturbed}")
                success = False
        except (ZeroDivisionError, ValueError) as e:
            logs.append(f"Electric field monotonicity check failed: {e}")
            success = False
            self.E = self._original_E

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success status and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
                overall_success = overall_success and success
            except Exception as e:
                all_logs.append(f"=== {test_name} ===")
                all_logs.append(f"ERROR: {str(e)}")
                all_logs.append("Result: FAIL")
                overall_success = False
        
        return overall_success, all_logs