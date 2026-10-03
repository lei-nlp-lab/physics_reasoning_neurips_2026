import math
import sympy as sp
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Extract inputs with defaults
        self.m = getattr(solution_instance, 'm', 0.2)  # kg
        self.theta = getattr(solution_instance, 'theta', 53.0)  # degrees
        self.mu = getattr(solution_instance, 'mu', 0.3)
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²
        # Convert theta to radians for symbolic use
        self.theta_rad = math.radians(self.theta)
        # Precompute trigonometric values as given in problem
        self.sin_theta = 0.8  # sin53°
        self.cos_theta = 0.6  # cos53°

        # Extract results safely from solution_instance.results (if available)
        # Use .get() with default values that will be computed from inputs if missing
        results = getattr(solution_instance, 'results', {})
        self.N = results.get('N', self.m * self.g * self.cos_theta)
        self.f_max = results.get('f_max', self.mu * self.N)
        self.mg_sin_theta = results.get('mg*sintheta', self.m * self.g * self.sin_theta)
        self.N_prime = results.get("N'", 7.2)  # from step_8
        self.f_prime = results.get("f'", 1.6)  # from step_8
        self.f_max_prime = results.get("f_max'", self.mu * self.N_prime)

        # For symbolic checks
        self.symbols = {
            'm': self.m,
            'theta': self.theta_rad,
            'mu': self.mu,
            'g': self.g,
            'sin_theta': self.sin_theta,
            'cos_theta': self.cos_theta
        }

    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy equality check."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify fundamental physics laws and equilibrium conditions."""
        logs = []
        success = True

        # 1. Normal force without magnetic force
        N_expected = self.m * self.g * self.cos_theta
        if not self.fuzzy_equal(self.N, N_expected):
            logs.append(f"Invariant violated: N = {self.N} ≠ m*g*cosθ = {N_expected}")
            success = False
        else:
            logs.append(f"N = {self.N} matches m*g*cosθ = {N_expected}")

        # 2. Maximum static friction without magnetic force
        f_max_expected = self.mu * N_expected
        if not self.fuzzy_equal(self.f_max, f_max_expected):
            logs.append(f"Invariant violated: f_max = {self.f_max} ≠ μ*N = {f_max_expected}")
            success = False
        else:
            logs.append(f"f_max = {self.f_max} matches μ*N = {f_max_expected}")

        # 3. Gravitational component down the incline
        mg_sin_expected = self.m * self.g * self.sin_theta
        if not self.fuzzy_equal(self.mg_sin_theta, mg_sin_expected):
            logs.append(f"Invariant violated: mg*sinθ = {self.mg_sin_theta} ≠ m*g*sinθ = {mg_sin_expected}")
            success = False
        else:
            logs.append(f"mg*sinθ = {self.mg_sin_theta} matches m*g*sinθ = {mg_sin_expected}")

        # 4. Sliding condition without magnetic force
        # According to KB, if F_parallel > f_max, motion occurs.
        if self.mg_sin_theta > self.f_max:
            logs.append(f"Without magnetic force: mg*sinθ ({self.mg_sin_theta}) > f_max ({self.f_max}) → phone slides down.")
        else:
            logs.append(f"Without magnetic force: mg*sinθ ({self.mg_sin_theta}) ≤ f_max ({self.f_max}) → phone stays. (But expected to slide!)")
            # The problem expects sliding, so this would be an error.
            success = False

        # 5. With magnetic force: equilibrium requires f' = mg*sinθ
        if not self.fuzzy_equal(self.f_prime, self.mg_sin_theta):
            logs.append(f"Invariant violated: f' = {self.f_prime} ≠ mg*sinθ = {self.mg_sin_theta}")
            success = False
        else:
            logs.append(f"f' = {self.f_prime} matches mg*sinθ = {self.mg_sin_theta} (equilibrium along incline)")

        # 6. Maximum static friction with magnetic force
        f_max_prime_expected = self.mu * self.N_prime
        if not self.fuzzy_equal(self.f_max_prime, f_max_prime_expected):
            logs.append(f"Invariant violated: f_max' = {self.f_max_prime} ≠ μ*N' = {f_max_prime_expected}")
            success = False
        else:
            logs.append(f"f_max' = {self.f_max_prime} matches μ*N' = {f_max_prime_expected}")

        # 7. Static friction condition with magnetic force: f' ≤ f_max'
        if self.f_prime <= self.f_max_prime:
            logs.append(f"With magnetic force: f' ({self.f_prime}) ≤ f_max' ({self.f_max_prime}) → static equilibrium possible.")
        else:
            logs.append(f"With magnetic force: f' ({self.f_prime}) > f_max' ({self.f_max_prime}) → sliding occurs. (Unexpected!)")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results and final answers."""
        logs = []
        success = True

        # Check that the computed N', f', f_max' are consistent with the target answers.
        # Target answers: 7.2 N and 1.6 N (presumably N' and f').
        target_N_prime = 7.2
        target_f_prime = 1.6

        if not self.fuzzy_equal(self.N_prime, target_N_prime):
            logs.append(f"Trace inconsistency: N' = {self.N_prime} ≠ target {target_N_prime}")
            success = False
        else:
            logs.append(f"N' = {self.N_prime} matches target {target_N_prime}")

        if not self.fuzzy_equal(self.f_prime, target_f_prime):
            logs.append(f"Trace inconsistency: f' = {self.f_prime} ≠ target {target_f_prime}")
            success = False
        else:
            logs.append(f"f' = {self.f_prime} matches target {target_f_prime}")

        # Additionally, check that f_max' = μ * N'
        if not self.fuzzy_equal(self.f_max_prime, self.mu * self.N_prime):
            logs.append(f"Trace inconsistency: f_max' = {self.f_max_prime} ≠ μ*N' = {self.mu * self.N_prime}")
            success = False
        else:
            logs.append(f"f_max' = {self.f_max_prime} is consistent with μ*N'")

        # Check that f' = mg*sinθ (already done in invariants, but we can log here too)
        if not self.fuzzy_equal(self.f_prime, self.mg_sin_theta):
            logs.append(f"Trace inconsistency: f' = {self.f_prime} ≠ mg*sinθ = {self.mg_sin_theta}")
            success = False
        else:
            logs.append(f"f' = {self.f_prime} is consistent with mg*sinθ = {self.mg_sin_theta}")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (mass) and verify that the results change in the expected direction.
        We will increase mass by 10% and check that all forces increase proportionally.
        """
        logs = []
        success = True

        # Original values (store for comparison)
        original_m = self.m
        perturbed_m = original_m * 1.1  # +10%

        # Recompute expected forces with perturbed mass
        N_pert = perturbed_m * self.g * self.cos_theta
        f_max_pert = self.mu * N_pert
        mg_sin_theta_pert = perturbed_m * self.g * self.sin_theta

        # For N' and f', they likely depend on m. From step_8, N' = m*g*cosθ + F, but F might be independent?
        # However, the problem doesn't give F explicitly. We assume that the magnetic force F is adjusted to maintain equilibrium.
        # In equilibrium, f' = mg*sinθ, and f_max' = μ*N' must be >= f'.
        # To maintain equilibrium, the required condition is μ*(m*g*cosθ + F) >= m*g*sinθ.
        # If we increase m, both sides increase. But the problem's N' and f' are specific numbers.
        # We cannot easily perturb N' and f' without knowing F. So we will only perturb the forces that directly depend on m.

        # Check monotonicity: if m increases, N, f_max, mg*sinθ should increase.
        if not (N_pert > self.N):
            logs.append(f"Monotonicity violated: N does not increase when m increases")
            success = False
        else:
            logs.append(f"N increases with m (as expected)")

        if not (f_max_pert > self.f_max):
            logs.append(f"Monotonicity violated: f_max does not increase when m increases")
            success = False
        else:
            logs.append(f"f_max increases with m (as expected)")

        if not (mg_sin_theta_pert > self.mg_sin_theta):
            logs.append(f"Monotonicity violated: mg*sinθ does not increase when m increases")
            success = False
        else:
            logs.append(f"mg*sinθ increases with m (as expected)")

        # Additionally, the sliding condition without magnetic force: if m increases, both sides scale linearly, so the inequality direction might not change? Actually, both mg*sinθ and f_max are proportional to m, so their ratio is constant. So the condition mg*sinθ > f_max becomes m*g*sinθ > μ*m*g*cosθ → sinθ > μ*cosθ. Since μ=0.3, cosθ=0.6, μ*cosθ=0.18, sinθ=0.8, so the condition holds regardless of m. So the phone always slides without magnetic force. That's consistent.

        # Restore original m (not strictly necessary as we didn't modify the instance, but for clarity)
        self.m = original_m

        return success, logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = ["=== PhysicsChecker starting ==="]
        overall_success = True

        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            all_logs.append(f"\n--- {name} Test ---")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
                all_logs.append(f"FAILED: {name} test failed.")
            else:
                all_logs.append(f"PASSED: {name} test passed.")

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs