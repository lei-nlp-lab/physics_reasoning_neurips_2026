import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract known quantities with safe defaults
        self.G_ground = getattr(solution_instance, 'G_ground', 
                                self.results.get('G_ground', 
                                self.results.get("G'", 0.0)))  # 16 N on ground
        self.F = getattr(solution_instance, 'F', self.results.get('F', 0.0))          # gravitational force at height h
        self.h = getattr(solution_instance, 'h', self.results.get('h', 0.0))          # height above ground
        
        # Constants from the problem
        self.g = 10.0                # m/s^2 (gravitational acceleration on ground)
        self.R = 6400.0e3            # m (Earth's radius, converted from 6400 km)
        self.N = 20.0                # N (normal force / pressure on rocket)
        self.a_rocket = 10.0         # m/s^2 (rocket's upward acceleration)
        
        # Convert h to meters if it appears to be in km (based on typical magnitude)
        if self.h < 1000:  # h likely given in km (since 6400 km is expected)
            self.h_m = self.h * 1000.0
        else:
            self.h_m = self.h
        
        # Compute derived quantities
        self.m = self.G_ground / self.g if self.g != 0 else 0.0  # mass of object
        
        # Logging
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((msg, success))

    def test_invariants(self):
        """
        Verify physical invariants using Newton's laws.
        """
        success = True
        
        # 1. Check mass is positive and reasonable (around 1.6 kg)
        if not math.isclose(self.m, 1.6, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Mass check failed: m = {self.m:.3f} kg, expected ~1.6 kg", False)
            success = False
        else:
            self._log(f"Mass check passed: m = {self.m:.3f} kg", True)
        
        # 2. Check Newton's second law in the rocket: N - F = m * a
        lhs = self.N - self.F
        rhs = self.m * self.a_rocket
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Newton's second law check failed: N - F = {lhs:.3f} N, m*a = {rhs:.3f} N", False)
            success = False
        else:
            self._log(f"Newton's second law check passed: N - F = m*a = {rhs:.3f} N", True)
        
        # 3. Check gravitational force ratio: F / G_ground = (R/(R + h))^2
        if self.G_ground == 0:
            self._log("G_ground is zero, cannot compute ratio", False)
            success = False
        else:
            ratio_force = self.F / self.G_ground
            ratio_distance = (self.R / (self.R + self.h_m)) ** 2
            if not math.isclose(ratio_force, ratio_distance, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Gravitational ratio check failed: F/G' = {ratio_force:.3f}, (R/(R+h))^2 = {ratio_distance:.3f}", False)
                success = False
            else:
                self._log(f"Gravitational ratio check passed: F/G' = (R/(R+h))^2 = {ratio_force:.3f}", True)
        
        return success

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Since we don't have explicit intermediate results, we verify that the
        steps leading to h and F are consistent with the extracted values.
        """
        success = True
        
        # Step consistency: From the steps, we expect F = 4 N and h = 6400 km.
        # Check that the extracted F is ~4 N.
        if not math.isclose(self.F, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"F value inconsistent with steps: F = {self.F:.3f} N, expected 4 N", False)
            success = False
        else:
            self._log(f"F value consistent with steps: F = {self.F:.3f} N", True)
        
        # Check that the extracted h is ~6400 km (converted to meters: 6.4e6 m)
        expected_h_m = 6400.0e3
        if not math.isclose(self.h_m, expected_h_m, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"h value inconsistent with steps: h = {self.h_m:.0f} m, expected {expected_h_m:.0f} m", False)
            success = False
        else:
            self._log(f"h value consistent with steps: h = {self.h_m:.0f} m", True)
        
        return success

    def test_monotonicity(self):
        """
        Perturb G_ground by +1% and check that F decreases and h increases,
        using the physical laws as a proxy for the solution's behavior.
        """
        success = True
        
        # Store original values
        G_original = self.G_ground
        F_original = self.F
        h_original_m = self.h_m
        
        # Perturb G_ground upward by 1%
        G_perturbed = G_original * 1.01
        
        # Recompute derived quantities
        m_perturbed = G_perturbed / self.g
        F_perturbed_expected = self.N - m_perturbed * self.a_rocket
        
        # Check that F decreases when G_ground increases
        if not F_perturbed_expected < F_original:
            self._log(f"Monotonicity (F): When G' increases, F does not decrease as expected. "
                      f"Original F = {F_original:.3f} N, perturbed expected F = {F_perturbed_expected:.3f} N", False)
            success = False
        else:
            self._log(f"Monotonicity (F) passed: F decreases when G' increases (from {F_original:.3f} N to {F_perturbed_expected:.3f} N)", True)
        
        # Compute h for the perturbed case using gravitational ratio
        # From: F_perturbed_expected / G_perturbed = (R/(R + h_perturbed))^2
        if G_perturbed == 0 or F_perturbed_expected <= 0:
            self._log("Cannot compute perturbed h because G_perturbed or F_perturbed_expected is non-positive", False)
            success = False
        else:
            ratio_perturbed = F_perturbed_expected / G_perturbed
            if ratio_perturbed <= 0:
                self._log(f"Ratio for perturbed h is non-positive: {ratio_perturbed}", False)
                success = False
            else:
                h_perturbed_m = self.R * (1.0 / math.sqrt(ratio_perturbed) - 1.0)
                # Check that h increases when G_ground increases
                if not h_perturbed_m > h_original_m:
                    self._log(f"Monotonicity (h): When G' increases, h does not increase as expected. "
                              f"Original h = {h_original_m:.0f} m, perturbed expected h = {h_perturbed_m:.0f} m", False)
                    success = False
                else:
                    self._log(f"Monotonicity (h) passed: h increases when G' increases (from {h_original_m:.0f} m to {h_perturbed_m:.0f} m)", True)
        
        return success

    def run_all_tests(self):
        """
        Run all tests and return success flag and logs.
        """
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{name}' raised an exception: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs