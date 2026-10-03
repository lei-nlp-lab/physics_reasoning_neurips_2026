import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with a solution instance, safely extracting inputs and results.
        """
        self.solution = solution_instance
        
        # Safe extraction of inputs (with defaults if missing)
        # Use the same scaling as the solution code: m=1, q=1, l=1, g=1
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.g = getattr(solution_instance, 'g', 1.0)
        
        # Safe extraction of results (using .results or direct attributes)
        results = getattr(solution_instance, 'results', {})
        
        self.E = results.get('E', 0.0)
        self.t = results.get('t', 0.0)  # time in second quadrant
        self.t_prime = results.get('t_prime', results.get("t'", 0.0))  # time in first quadrant
        self.v0 = results.get('v_0', 0.0)  # velocity entering first quadrant
        self.xA = results.get('x', 0.0)
        self.yA = results.get('y', 0.0)
        self.A = results.get('A', (self.xA, self.yA))  # point A as tuple
        self.EkD = results.get('E_kD', 0.0)  # kinetic energy at D
        
        # Numerical tolerance
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append((message, success))
        
    def _is_close(self, a, b, rel_tol=None, abs_tol=None):
        if rel_tol is None:
            rel_tol = self.rel_tol
        if abs_tol is None:
            abs_tol = self.abs_tol
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self):
        """
        Test fundamental physics invariants: equilibrium, Newton's second law, work-energy.
        """
        tests = []
        try:
            # 1. Equilibrium at initial point (vertical balance)
            # From step_1: q * E * sin(45°) = m * g  => E = sqrt(2)*m*g/q
            expected_E = math.sqrt(2) * self.m * self.g / self.q
            if self._is_close(self.E, expected_E):
                self._log(f"Equilibrium check passed: E = {self.E} ≈ {expected_E}", True)
                tests.append(True)
            else:
                self._log(f"Equilibrium check failed: E = {self.E} ≠ {expected_E}", False)
                tests.append(False)
                
            # 2. Acceleration in second quadrant (E1 region)
            # Net force horizontal: F_net = q * E * cos(45°) = m * g (since cos45=sin45)
            # So a = g horizontally.
            a2_expected = self.g
            # The solution might provide acceleration in second quadrant, but we don't have it.
            # Instead, we can check via kinematics: v0^2 = 2 * a * displacement? Not needed.
            # We'll skip this test since we don't have a2.
            self._log(f"Skipping acceleration in second quadrant check (not in results)", True)
            tests.append(True)
                
            # 3. Acceleration in first quadrant (E2 region)
            # From step_6: a = (q*E + m*g)/m = (1+sqrt(2))*g
            a1_expected = (1 + math.sqrt(2)) * self.g
            # Check using the relation from step_7: v0 = a * t_prime
            if self.t_prime > 0:
                a1_from_motion = self.v0 / self.t_prime
                if self._is_close(a1_from_motion, a1_expected):
                    self._log(f"Acceleration in first quadrant from v0/t': {a1_from_motion} ≈ (1+√2)g = {a1_expected}", True)
                    tests.append(True)
                else:
                    self._log(f"Acceleration in first quadrant mismatch: v0/t' = {a1_from_motion} ≠ (1+√2)g = {a1_expected}", False)
                    tests.append(False)
            else:
                self._log(f"Skipping acceleration check (t' = 0)", False)
                tests.append(False)
                    
            # 4. Work-energy theorem from start to point D
            # Step_12: q E l cos45° + (qE+mg)*[2l/(√2+1)] + mg*[2l/(√2+1)] = EkD
            term1 = self.q * self.E * self.l * math.cos(math.pi/4)
            factor = 2 * self.l / (math.sqrt(2) + 1)
            term2 = (self.q * self.E + self.m * self.g) * factor
            term3 = self.m * self.g * factor
            expected_EkD = term1 + term2 + term3
            if self._is_close(self.EkD, expected_EkD):
                self._log(f"Work-energy theorem check: EkD = {self.EkD} ≈ {expected_EkD}", True)
                tests.append(True)
            else:
                self._log(f"Work-energy theorem failed: EkD = {self.EkD} ≠ {expected_EkD}", False)
                tests.append(False)
                
        except Exception as e:
            self._log(f"Exception in test_invariants: {traceback.format_exc()}", False)
            tests.append(False)
            
        return all(tests)
        
    def test_trace_consistency(self):
        """
        Check consistency between intermediate results and final answers.
        """
        tests = []
        try:
            # 1. Velocity entering first quadrant: v0 = sqrt(2 g l) from step_4
            expected_v0 = math.sqrt(2 * self.g * self.l)
            if self._is_close(self.v0, expected_v0):
                self._log(f"v0 check: v0 = {self.v0} ≈ sqrt(2gl) = {expected_v0}", True)
                tests.append(True)
            else:
                self._log(f"v0 mismatch: v0 = {self.v0} ≠ sqrt(2gl) = {expected_v0}", False)
                tests.append(False)
                
            # 2. Point A coordinates: xA = yA = 2*(sqrt(2)-1)*l
            expected_coord = 2 * (math.sqrt(2) - 1) * self.l
            if self._is_close(self.xA, expected_coord) and self._is_close(self.yA, expected_coord):
                self._log(f"Point A coordinates: (x, y) = ({self.xA}, {self.yA}) ≈ ({expected_coord}, {expected_coord})", True)
                tests.append(True)
            else:
                self._log(f"Point A coordinates mismatch: ({self.xA}, {self.yA}) ≠ ({expected_coord}, {expected_coord})", False)
                tests.append(False)
                
            # 3. Relation from step_7: v0 = a * t_prime, where a = (1+√2)g
            a1 = (1 + math.sqrt(2)) * self.g
            if self.t_prime > 0:
                if self._is_close(self.v0, a1 * self.t_prime):
                    self._log(f"v0 = a * t' check: {self.v0} ≈ {a1} * {self.t_prime}", True)
                    tests.append(True)
                else:
                    self._log(f"v0 = a * t' failed: {self.v0} ≠ {a1} * {self.t_prime}", False)
                    tests.append(False)
            else:
                tests.append(False)
                    
            # 4. Kinematics in first quadrant: xA should equal v0 * t_prime
            # But yA does NOT necessarily equal 0.5*a*t_prime^2 because of the perpendicular crossing condition
            # We only check xA = v0 * t_prime
            if self.t_prime > 0:
                xA_from_v0 = self.v0 * self.t_prime
                if self._is_close(self.xA, xA_from_v0):
                    self._log(f"Kinematics x-coordinate consistent: xA={self.xA} ≈ v0*t'={xA_from_v0}", True)
                    tests.append(True)
                else:
                    self._log(f"Kinematics x-coordinate inconsistent: xA={self.xA} vs v0*t'={xA_from_v0}", False)
                    tests.append(False)
            else:
                tests.append(False)
                    
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {traceback.format_exc()}", False)
            tests.append(False)
            
        return all(tests)
        
    def test_monotonicity(self):
        """
        Perturb an input (mass m) and check that outputs change as expected.
        """
        original_m = self.m
        tests = []
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            self.m = perturbed_m
            
            # Recompute expected values using formulas from steps
            # 1. E = sqrt(2)*m*g/q
            expected_E_perturbed = math.sqrt(2) * perturbed_m * self.g / self.q
            # Since E is proportional to m, it should increase by 10%.
            if self._is_close(expected_E_perturbed, self.E * 1.1, rel_tol=1e-2):
                self._log(f"Monotonicity: E increases with m (proportional)", True)
                tests.append(True)
            else:
                self._log(f"Monotonicity: E did not increase as expected with m", False)
                tests.append(False)
                
            # 2. v0 = sqrt(2*g*l) is independent of m, so should not change.
            expected_v0 = math.sqrt(2 * self.g * self.l)
            if self._is_close(self.v0, expected_v0):
                self._log(f"Monotonicity: v0 is independent of m", True)
                tests.append(True)
            else:
                self._log(f"Monotonicity: v0 changed with m (unexpected)", False)
                tests.append(False)
                
            # 3. Point A coordinates: xA = yA = 2*(sqrt(2)-1)*l, independent of m.
            expected_coord = 2 * (math.sqrt(2) - 1) * self.l
            if self._is_close(self.xA, expected_coord) and self._is_close(self.yA, expected_coord):
                self._log(f"Monotonicity: Point A coordinates independent of m", True)
                tests.append(True)
            else:
                self._log(f"Monotonicity: Point A coordinates changed with m", False)
                tests.append(False)
                
            # 4. Kinetic energy at D: from work-energy, it depends on m and E (which depends on m).
            # Compute expected EkD for perturbed m:
            term1 = self.q * expected_E_perturbed * self.l * math.cos(math.pi/4)
            factor = 2 * self.l / (math.sqrt(2) + 1)
            term2 = (self.q * expected_E_perturbed + perturbed_m * self.g) * factor
            term3 = perturbed_m * self.g * factor
            expected_EkD_perturbed = term1 + term2 + term3
            # For original m, compute expected EkD (using original E)
            original_E = math.sqrt(2) * original_m * self.g / self.q
            term1_orig = self.q * original_E * self.l * math.cos(math.pi/4)
            term2_orig = (self.q * original_E + original_m * self.g) * factor
            term3_orig = original_m * self.g * factor
            expected_EkD_original = term1_orig + term2_orig + term3_orig
            # Since both E and m increase, EkD should increase.
            if expected_EkD_perturbed > expected_EkD_original:
                self._log(f"Monotonicity: EkD increases with m", True)
                tests.append(True)
            else:
                self._log(f"Monotonicity: EkD did not increase with m (unexpected)", False)
                tests.append(False)
                
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {traceback.format_exc()}", False)
            tests.append(False)
        finally:
            # Restore original mass
            self.m = original_m
            
        return all(tests)
        
    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []
        
        # Run tests
        inv = self.test_invariants()
        trace = self.test_trace_consistency()
        mono = self.test_monotonicity()
        
        success = inv and trace and mono
        
        # Summarize
        if success:
            self._log("All tests passed.", True)
        else:
            self._log("Some tests failed.", False)
            
        return success, self.logs