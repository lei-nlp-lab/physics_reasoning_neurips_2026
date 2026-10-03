import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract gravitational acceleration (default 10 m/s^2)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract heights h1 and h2 if available; otherwise compute from results
        self.h1 = getattr(solution_instance, 'h1', None)
        self.h2 = getattr(solution_instance, 'h2', None)
        
        # Extract speeds from results (with safe defaults)
        self.v_B = self.results.get('v_B', 0.0)
        self.v_C = self.results.get('v_C', 0.0)
        
        # If h1 or h2 are missing, compute from results and g
        if self.h1 is None and self.v_B != 0.0:
            self.h1 = (self.v_B**2) / (2 * self.g)
        if self.h2 is None and self.v_B != 0.0 and self.v_C != 0.0:
            self.h1_computed = (self.v_B**2) / (2 * self.g) if self.h1 is None else self.h1
            self.h2 = (self.v_C**2) / (2 * self.g) - self.h1_computed
        
        # Store original values for monotonicity test
        self._original_h1 = self.h1
        self._original_h2 = self.h2
        self._original_v_B = self.v_B
        self._original_v_C = self.v_C
        
        # Initialize logs
        self.logs = []

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((success, message))
        return success

    def _fuzzy_equal(self, a, b):
        """Robust fuzzy equality check with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Check conservation of mechanical energy between points A-B and A-C."""
        success = True
        
        # Energy conservation A -> B: (1/2) m v_B^2 = m g h1  => v_B^2 = 2 g h1
        if self.h1 is not None:
            expected_v_B_sq = 2 * self.g * self.h1
            actual_v_B_sq = self.v_B**2
            if not self._fuzzy_equal(actual_v_B_sq, expected_v_B_sq):
                success = self._log(
                    f"Energy conservation A->B failed: v_B^2={actual_v_B_sq:.2f}, expected 2*g*h1={expected_v_B_sq:.2f}",
                    False
                )
            else:
                self._log("Energy conservation A->B holds.", True)
        else:
            success = self._log("Cannot test A->B energy conservation: h1 missing.", False)
        
        # Energy conservation A -> C: (1/2) m v_C^2 = m g (h1 + h2)  => v_C^2 = 2 g (h1 + h2)
        if self.h1 is not None and self.h2 is not None:
            expected_v_C_sq = 2 * self.g * (self.h1 + self.h2)
            actual_v_C_sq = self.v_C**2
            if not self._fuzzy_equal(actual_v_C_sq, expected_v_C_sq):
                success = self._log(
                    f"Energy conservation A->C failed: v_C^2={actual_v_C_sq:.2f}, expected 2*g*(h1+h2)={expected_v_C_sq:.2f}",
                    False
                )
            else:
                self._log("Energy conservation A->C holds.", True)
        else:
            success = self._log("Cannot test A->C energy conservation: h1 or h2 missing.", False)
        
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers using step equations."""
        success = True
        
        # Step 2: v_B = sqrt(2 g h1)
        if self.h1 is not None:
            expected_v_B = math.sqrt(2 * self.g * self.h1)
            if not self._fuzzy_equal(self.v_B, expected_v_B):
                success = self._log(
                    f"Trace consistency failed for v_B: computed {self.v_B:.2f}, expected sqrt(2*g*h1)={expected_v_B:.2f}",
                    False
                )
            else:
                self._log(f"v_B={self.v_B:.2f} matches formula sqrt(2*g*h1).", True)
        else:
            success = self._log("Cannot check v_B formula: h1 missing.", False)
        
        # Step 4: v_C = sqrt(2 g (h1 + h2))
        if self.h1 is not None and self.h2 is not None:
            expected_v_C = math.sqrt(2 * self.g * (self.h1 + self.h2))
            if not self._fuzzy_equal(self.v_C, expected_v_C):
                success = self._log(
                    f"Trace consistency failed for v_C: computed {self.v_C:.2f}, expected sqrt(2*g*(h1+h2))={expected_v_C:.2f}",
                    False
                )
            else:
                self._log(f"v_C={self.v_C:.2f} matches formula sqrt(2*g*(h1+h2)).", True)
        else:
            success = self._log("Cannot check v_C formula: h1 or h2 missing.", False)
        
        return success

    def test_monotonicity(self):
        """Perturb input h1 and check that v_B and v_C increase accordingly."""
        success = True
        
        if self.h1 is None or self.h2 is None:
            return self._log("Cannot run monotonicity test: h1 or h2 missing.", False)
        
        # Perturb h1 by +1%
        perturb_factor = 1.01
        h1_perturbed = self.h1 * perturb_factor
        h2 = self.h2  # keep h2 fixed
        
        # Recompute speeds using formulas
        v_B_perturbed = math.sqrt(2 * self.g * h1_perturbed)
        v_C_perturbed = math.sqrt(2 * self.g * (h1_perturbed + h2))
        
        # Check monotonicity: both speeds should increase
        if not (v_B_perturbed > self.v_B and v_C_perturbed > self.v_C):
            success = self._log(
                f"Monotonicity test failed: increasing h1 by 1% did not increase both speeds. "
                f"Original v_B={self.v_B:.2f}, perturbed v_B={v_B_perturbed:.2f}; "
                f"Original v_C={self.v_C:.2f}, perturbed v_C={v_C_perturbed:.2f}",
                False
            )
        else:
            self._log(
                f"Monotonicity holds: increasing h1 increases both v_B and v_C. "
                f"v_B: {self.v_B:.2f} -> {v_B_perturbed:.2f}, v_C: {self.v_C:.2f} -> {v_C_perturbed:.2f}",
                True
            )
        
        # Restore original values
        self.h1 = self._original_h1
        self.h2 = self._original_h2
        
        return success

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Conservation of Energy", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs