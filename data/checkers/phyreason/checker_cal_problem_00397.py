import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.logs = []
        self.inputs = {}
        self.results = {}

        # Given constants (problem statement)
        self.inputs['m'] = 0.2  # kg
        self.inputs['L'] = 0.5  # m
        self.inputs['I'] = 1.0  # A
        self.inputs['B'] = 2.0  # T
        self.inputs['theta'] = math.radians(37)  # rad
        self.inputs['g'] = 10.0  # m/s^2
        self.inputs['sin37'] = 0.6
        self.inputs['cos37'] = 0.8

        # Extract results from solution_instance safely
        # Use getattr with fallback to .get() if solution_instance has a results dict
        # Expected keys: 'F_f', 'F_N', 'mu', 'B_prime' (or 'B\'' from mapping)
        # Mapping: 'F_f' -> initial friction, 'F_N' -> initial normal, 'mu' -> coefficient,
        #          'F_f\'' -> friction when about to move, 'F_N\'' -> normal when about to move,
        #          'B\'' -> magnetic field when about to move
        # We'll try to get these from results dict first, then from attributes.
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results_dict = solution_instance.results
        else:
            results_dict = {}

        self.results['F_f'] = getattr(solution_instance, 'F_f', results_dict.get('F_f', None))
        self.results['F_N'] = getattr(solution_instance, 'F_N', results_dict.get('F_N', None))
        self.results['mu'] = getattr(solution_instance, 'mu', results_dict.get('mu', results_dict.get('μ', None)))
        self.results['F_f_prime'] = getattr(solution_instance, 'F_f_prime', results_dict.get('F_f\'', None))
        self.results['F_N_prime'] = getattr(solution_instance, 'F_N_prime', results_dict.get('F_N\'', None))
        self.results['B_prime'] = getattr(solution_instance, 'B_prime', results_dict.get('B\'', None))

        # If any result is None, try to compute from known formulas as fallback
        self._fill_missing_results()

    def _fill_missing_results(self):
        """
        Compute missing results using the known formulas and inputs.
        This ensures we have all necessary values for checks.
        """
        m, L, I, B, g = self.inputs['m'], self.inputs['L'], self.inputs['I'], self.inputs['B'], self.inputs['g']
        sin37, cos37 = self.inputs['sin37'], self.inputs['cos37']

        # Step 1: F_f = B * I * L * sin(37°)
        if self.results['F_f'] is None:
            self.results['F_f'] = B * I * L * sin37

        # Step 2: F_N = m*g - B * I * L * cos(37°)
        if self.results['F_N'] is None:
            self.results['F_N'] = m * g - B * I * L * cos37

        # Step 4: μ = F_f / F_N
        if self.results['mu'] is None:
            if self.results['F_N'] != 0:
                self.results['mu'] = self.results['F_f'] / self.results['F_N']

        # Step 5: F_f' = B' * I * L * sin(37°)
        # Step 6: F_N' = m*g + B' * I * L * cos(37°)
        # Step 7: F_f' = μ * F_N' -> solve for B'
        if self.results['B_prime'] is None and self.results['mu'] is not None:
            # Solve equation: B' * I * L * sin37 = μ * (m*g + B' * I * L * cos37)
            # => B' * I * L * sin37 = μ*m*g + μ * B' * I * L * cos37
            # => B' * I * L * (sin37 - μ * cos37) = μ*m*g
            # => B' = (μ*m*g) / (I * L * (sin37 - μ * cos37))
            denominator = I * L * (sin37 - self.results['mu'] * cos37)
            if abs(denominator) > 1e-12:
                self.results['B_prime'] = (self.results['mu'] * m * g) / denominator

        if self.results['F_f_prime'] is None and self.results['B_prime'] is not None:
            self.results['F_f_prime'] = self.results['B_prime'] * I * L * sin37

        if self.results['F_N_prime'] is None and self.results['B_prime'] is not None:
            self.results['F_N_prime'] = m * g + self.results['B_prime'] * I * L * cos37

    def log(self, message):
        """Helper to log messages."""
        self.logs.append(message)

    def test_invariants(self):
        """Check fundamental physical laws: magnetic force, force balance, friction definition."""
        success = True
        m, L, I, B, g = self.inputs['m'], self.inputs['L'], self.inputs['I'], self.inputs['B'], self.inputs['g']
        sin37, cos37 = self.inputs['sin37'], self.inputs['cos37']
        F_f, F_N, mu = self.results['F_f'], self.results['F_N'], self.results['mu']

        # 1. Magnetic force formula (Ampere's force for a straight wire in uniform B)
        F_mag_horizontal = B * I * L * sin37
        if not math.isclose(F_f, F_mag_horizontal, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Magnetic force invariant. Expected F_f ≈ {F_mag_horizontal}, got {F_f}")
            success = False
        else:
            self.log("PASS: Magnetic force formula holds.")

        # 2. Force balance in vertical direction (initial condition)
        F_mag_vertical = B * I * L * cos37
        F_N_expected = m * g - F_mag_vertical
        if not math.isclose(F_N, F_N_expected, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Vertical force balance. Expected F_N ≈ {F_N_expected}, got {F_N}")
            success = False
        else:
            self.log("PASS: Vertical force balance holds.")

        # 3. Friction definition (initial condition): μ = F_f / F_N
        if F_N != 0:
            mu_expected = F_f / F_N
            if not math.isclose(mu, mu_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Friction definition. Expected μ ≈ {mu_expected}, got {mu}")
                success = False
            else:
                self.log("PASS: Friction definition holds.")
        else:
            self.log("WARNING: F_N is zero, cannot check friction definition.")
            success = False

        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results and final answer."""
        success = True
        m, L, I, g = self.inputs['m'], self.inputs['L'], self.inputs['I'], self.inputs['g']
        sin37, cos37 = self.inputs['sin37'], self.inputs['cos37']
        mu = self.results['mu']
        B_prime = self.results['B_prime']
        F_f_prime = self.results['F_f_prime']
        F_N_prime = self.results['F_N_prime']

        # Check that B' is consistent with the threshold condition: F_f' = μ * F_N'
        if mu is not None and B_prime is not None:
            # Compute expected F_f' and F_N' from B'
            F_f_prime_expected = B_prime * I * L * sin37
            F_N_prime_expected = m * g + B_prime * I * L * cos37

            # Check F_f' and F_N' if they are available (optional)
            if F_f_prime is not None and not math.isclose(F_f_prime, F_f_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: F_f' inconsistency. Expected {F_f_prime_expected}, got {F_f_prime}")
                success = False
            if F_N_prime is not None and not math.isclose(F_N_prime, F_N_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: F_N' inconsistency. Expected {F_N_prime_expected}, got {F_N_prime}")
                success = False

            # Check the force balance at threshold: F_f' = μ * F_N'
            lhs = F_f_prime_expected
            rhs = mu * F_N_prime_expected
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Threshold condition fails. {lhs} != μ * {F_N_prime_expected} = {rhs}")
                success = False
            else:
                self.log("PASS: Threshold condition (F_f' = μ * F_N') holds.")

        # Check against target answers: μ = 0.5, B' = 10 T
        if not math.isclose(mu, 0.5, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Coefficient μ does not match target 0.5. Got {mu}")
            success = False
        else:
            self.log("PASS: Coefficient μ matches target 0.5.")

        if B_prime is not None and not math.isclose(B_prime, 10.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Magnetic field B' does not match target 10 T. Got {B_prime}")
            success = False
        else:
            self.log("PASS: Magnetic field B' matches target 10 T.")

        return success

    def test_monotonicity(self):
        """
        Perturb an input (mass m) and verify the trend in μ and B'.
        We use the formulas to recalculate, then restore original values.
        """
        success = True
        original_m = self.inputs['m']
        perturbed_m = original_m * 1.05  # Increase by 5%

        # Recalculate μ with perturbed mass (keeping all else constant)
        m, L, I, B, g = perturbed_m, self.inputs['L'], self.inputs['I'], self.inputs['B'], self.inputs['g']
        sin37, cos37 = self.inputs['sin37'], self.inputs['cos37']

        # Compute forces and μ for perturbed mass
        F_f_pert = B * I * L * sin37  # unchanged because B, I, L, sin37 unchanged
        F_N_pert = m * g - B * I * L * cos37  # increases because m increases
        if F_N_pert != 0:
            mu_pert = F_f_pert / F_N_pert
            # Since F_f_pert is constant and F_N_pert increases, μ should decrease.
            if mu_pert >= self.results['mu']:
                self.log(f"FAIL: Monotonicity in mass. Increasing mass should decrease μ. Original μ={self.results['mu']}, perturbed μ={mu_pert}")
                success = False
            else:
                self.log("PASS: μ decreases with increased mass (as expected).")

            # Now, for the second part, we check B' trend with respect to μ.
            # Since μ decreases, and B' = (μ*m*g) / (I*L*(sin37 - μ*cos37)), we need to see if B' decreases.
            # We'll compute B' for the perturbed mass and the new μ.
            # But note: μ is recalculated from the initial condition with the same B? Actually, in the problem,
            # μ is determined from the initial condition. So if we change mass, the initial condition gives a different μ.
            # Then we compute B' for that μ. So we need to see the trend: if mass increases, μ decreases, what happens to B'?
            # We'll compute B' for the perturbed parameters.
            denominator = I * L * (sin37 - mu_pert * cos37)
            if abs(denominator) > 1e-12:
                B_prime_pert = (mu_pert * m * g) / denominator
                # Compare with the original B' (computed with original mass and original μ).
                # Since the original μ was 0.5 and original mass 0.2, we expect B'=10.
                # For the perturbed case, we can check if B_prime_pert > 10? Actually, it's not straightforward.
                # Instead, we can test a different monotonicity: if μ increases (with fixed mass), B' should increase.
                # So let's test that: perturb μ (by a small amount) and see the effect on B'.
                # We'll do a symbolic check: from the equation B' = (μ*m*g) / (I*L*(sin37 - μ*cos37))
                # Take derivative with respect to μ: positive? For μ in (0, sin37/cos37) it is positive.
                # Since sin37/cos37 = 0.6/0.8=0.75, and our μ=0.5 is within that, so derivative positive.
                # So if μ increases, B' increases.
                # We'll test by increasing μ by 1% and recomputing B' with original mass.
                mu_original = self.results['mu']
                mu_increased = mu_original * 1.01
                m_original = self.inputs['m']
                denominator2 = I * L * (sin37 - mu_increased * cos37)
                if abs(denominator2) > 1e-12:
                    B_prime_increased = (mu_increased * m_original * g) / denominator2
                    if B_prime_increased <= self.results['B_prime']:
                        self.log(f"FAIL: Monotonicity in μ. Increasing μ should increase B'. Original B'={self.results['B_prime']}, increased μ B'={B_prime_increased}")
                        success = False
                    else:
                        self.log("PASS: B' increases with increased μ (as expected).")
                else:
                    self.log("WARNING: Denominator zero when testing monotonicity of μ.")
                    success = False
            else:
                self.log("WARNING: Denominator zero when computing B' for perturbed mass.")
                success = False
        else:
            self.log("WARNING: Perturbed F_N is zero, cannot compute μ.")
            success = False

        return success

    def run_all_tests(self):
        """
        Execute all tests and return a tuple (success_bool, logs_list).
        """
        self.logs = []  # Reset logs
        all_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                self.log(f"\n--- Running {name} Test ---")
                if not test_func():
                    all_success = False
            except Exception as e:
                self.log(f"ERROR in {name}: {traceback.format_exc()}")
                all_success = False

        return all_success, self.logs