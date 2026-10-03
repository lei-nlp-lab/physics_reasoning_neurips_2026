import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.L = getattr(solution_instance, 'L', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.R0 = getattr(solution_instance, 'R0', 0.8)
        self.E = getattr(solution_instance, 'E', 10.0)
        self.r = getattr(solution_instance, 'r', 0.2)
        self.R = getattr(solution_instance, 'R', 4.0)
        self.B = getattr(solution_instance, 'B', 3.0)
        self.alpha = getattr(solution_instance, 'alpha', 37.0)  # in degrees
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Convert alpha to radians for calculations
        self.alpha_rad = math.radians(self.alpha)
        self.sin_alpha = math.sin(self.alpha_rad)
        self.cos_alpha = math.cos(self.alpha_rad)

        # Extract results safely from solution_instance.results (if available)
        self.results = getattr(solution_instance, 'results', {})
        self.I = self.results.get('I', 0.0)
        self.F = self.results.get('F', 0.0)
        self.G_min = self.results.get('G_min', 0.0)
        self.G_max = self.results.get('G_max', 0.0)
        self.G_range = self.results.get('G_range', 0.0)

        # For internal use: computed expected values
        self.I_expected = self.E / (self.R + self.r + self.R0)
        self.F_expected = self.B * self.I_expected * self.L

        # Compute G_min and G_max from the equilibrium equations
        # IMPORTANT: The problem defines G_min as the smaller weight (1 N) and G_max as the larger weight (6.2 N)
        # Equations for smaller G (friction to the right):
        # F*sinα = T + μ*N, F*cosα + N = m*g, T = G_small
        N_small = (self.m * self.g - self.F_expected * self.cos_alpha)
        self.G_small_expected = self.F_expected * self.sin_alpha - self.mu * N_small  # ~1.0 N

        # Equations for larger G (friction to the left):
        # F*sinα + μ*N = T, F*cosα + N = m*g, T = G_large
        N_large = (self.m * self.g - self.F_expected * self.cos_alpha)
        self.G_large_expected = self.F_expected * self.sin_alpha + self.mu * N_large  # ~6.2 N

        # Now map to the required output variables: G_min should be the smaller, G_max the larger
        self.G_min_expected = self.G_small_expected
        self.G_max_expected = self.G_large_expected

        self.logs = []

    def log(self, msg):
        self.logs.append(msg)

    def test_invariants(self):
        """Check physical laws: Ohm's law, Ampere's force, and force equilibrium."""
        try:
            # Ohm's law
            if not math.isclose(self.I, self.I_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Ohm's law invariant failed: I={self.I}, expected {self.I_expected}")
                return False
            else:
                self.log(f"Ohm's law invariant passed: I={self.I} A")

            # Ampere's force law
            if not math.isclose(self.F, self.F_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Ampere's force invariant failed: F={self.F}, expected {self.F_expected}")
                return False
            else:
                self.log(f"Ampere's force invariant passed: F={self.F} N")

            # Force equilibrium for minimum G (should be the smaller value)
            if not math.isclose(self.G_min, self.G_min_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Force equilibrium (min G) failed: G_min={self.G_min}, expected {self.G_min_expected}")
                return False
            else:
                self.log(f"Force equilibrium (min G) passed: G_min={self.G_min} N")

            # Force equilibrium for maximum G (should be the larger value)
            if not math.isclose(self.G_max, self.G_max_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Force equilibrium (max G) failed: G_max={self.G_max}, expected {self.G_max_expected}")
                return False
            else:
                self.log(f"Force equilibrium (max G) passed: G_max={self.G_max} N")

            # Check that G_min < G_max (as per definition)
            if self.G_min_expected >= self.G_max_expected:
                self.log(f"Contradiction: G_min_expected ({self.G_min_expected}) >= G_max_expected ({self.G_max_expected})")
                return False

            # Check that the range is consistent with the problem's given values (1 N and 6.2 N)
            if not (math.isclose(self.G_min, 1.0, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.G_max, 6.2, rel_tol=1e-2, abs_tol=1e-2)):
                self.log(f"G range mismatch: expected [1, 6.2], got [{self.G_min}, {self.G_max}]")
                return False
            else:
                self.log(f"G range invariant passed: {self.G_min} N ≤ G ≤ {self.G_max} N")

            return True
        except Exception as e:
            self.log(f"test_invariants failed with error: {str(e)}")
            return False

    def test_trace_consistency(self):
        """Check that intermediate results (I, F) lead to consistent G range."""
        try:
            # Use the extracted I and F to compute G_min and G_max, and compare with stored G_min and G_max
            # This is a cross-check: if I and F are correct, then G_min and G_max should match.
            N_small = (self.m * self.g - self.F * self.cos_alpha)
            G_min_from_trace = self.F * self.sin_alpha - self.mu * N_small  # smaller value

            N_large = (self.m * self.g - self.F * self.cos_alpha)
            G_max_from_trace = self.F * self.sin_alpha + self.mu * N_large  # larger value

            if not (math.isclose(self.G_min, G_min_from_trace, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.G_max, G_max_from_trace, rel_tol=1e-2, abs_tol=1e-2)):
                self.log(f"Trace consistency failed: G_min={self.G_min}, computed={G_min_from_trace}; G_max={self.G_max}, computed={G_max_from_trace}")
                return False
            else:
                self.log("Trace consistency passed: intermediate I and F yield correct G range.")
            return True
        except Exception as e:
            self.log(f"test_trace_consistency failed with error: {str(e)}")
            return False

    def test_monotonicity(self):
        """Perturb an input (e.g., mass m) and check the trend of G_min and G_max."""
        original_m = self.m
        try:
            # Increase mass by 10%
            perturbed_m = original_m * 1.1
            # Recompute I and F (I does not depend on m, F does not depend on m)
            I_perturbed = self.E / (self.R + self.r + self.R0)
            F_perturbed = self.B * I_perturbed * self.L
            # Compute new G_min and G_max
            N_small_perturbed = (perturbed_m * self.g - F_perturbed * self.cos_alpha)
            G_min_perturbed = F_perturbed * self.sin_alpha - self.mu * N_small_perturbed
            N_large_perturbed = (perturbed_m * self.g - F_perturbed * self.cos_alpha)
            G_max_perturbed = F_perturbed * self.sin_alpha + self.mu * N_large_perturbed

            # Check monotonicity: as mass increases, G_min (smaller) should decrease? Let's derive:
            # G_min = F sinα - μ (m g - F cosα) = F sinα + μ F cosα - μ m g
            # Coefficient of m is -μ g < 0, so G_min decreases with increasing m.
            # G_max = F sinα + μ (m g - F cosα) = F sinα - μ F cosα + μ m g
            # Coefficient of m is +μ g > 0, so G_max increases with increasing m.
            if not (G_min_perturbed < self.G_min_expected and G_max_perturbed > self.G_max_expected):
                self.log(f"Monotonicity test failed for mass increase: G_min changed from {self.G_min_expected} to {G_min_perturbed}, G_max from {self.G_max_expected} to {G_max_perturbed}")
                return False
            else:
                self.log(f"Monotonicity passed for mass increase: G_min decreases, G_max increases.")

            # Restore original mass
            self.m = original_m
            return True
        except Exception as e:
            self.log(f"test_monotonicity failed with error: {str(e)}")
            self.m = original_m
            return False

    def run_all_tests(self):
        """Run all tests and return success and logs."""
        self.logs = []  # Reset logs
        success = True

        try:
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self.log(f"Unexpected error in run_all_tests: {str(e)}")
            success = False

        return success, self.logs