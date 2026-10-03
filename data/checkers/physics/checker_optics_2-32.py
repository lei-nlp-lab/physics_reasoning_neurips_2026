import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Expected result keys
        self.L = self.results.get('L', None)
        self.ratio = self.results.get('ratio', None)

        # Givens (use None if missing)
        self.a = self.givens.get('a', None)
        self.lambda_ = self.givens.get('lambda', None)  # prefer 'lambda'

    # ---- Interface Tests ----
    def test_interface_L_exists(self):
        if self.L is None:
            return False, "L missing from results"
        return True, "L exists in results"

    def test_interface_ratio_exists(self):
        if self.ratio is None:
            return False, "ratio missing from results"
        return True, "ratio exists in results"

    def test_interface_L_numeric(self):
        if self.L is None:
            return False, "SKIP: L not available"
        if not isinstance(self.L, (int, float)):
            return False, f"L is not numeric, got {type(self.L)}"
        return True, "L is numeric"

    def test_interface_ratio_numeric(self):
        if self.ratio is None:
            return False, "SKIP: ratio not available"
        if not isinstance(self.ratio, (int, float)):
            return False, f"ratio is not numeric, got {type(self.ratio)}"
        return True, "ratio is numeric"

    def test_interface_values_positive(self):
        if self.L is not None and self.L <= 0:
            return False, f"L = {self.L} is not positive"
        if self.ratio is not None and self.ratio <= 0:
            return False, f"ratio = {self.ratio} is not positive"
        return True, "All available values positive"

    # ---- Invariant Tests (theorem-based physical sanity) ----
    def test_invariant_fresnel_number_odd(self):
        if self.L is None or self.a is None or self.lambda_ is None:
            return False, "SKIP: missing a, lambda, or L"
        try:
            N = (self.a ** 2) / (self.lambda_ * self.L)
        except ZeroDivisionError:
            return False, "L or lambda is zero"
        # Fresnel number should be an odd integer (here N=1)
        if math.isclose(N, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"N ≈ 1 indeed (N={N:.5f})"
        else:
            return False, f"Fresnel number N={N:.5f} not close to 1 (first bright spot)"

    def test_invariant_ratio_from_N(self):
        # Relative on-axis intensity I_rel = 4 sin^2(pi N/2)
        if self.L is None or self.a is None or self.lambda_ is None or self.ratio is None:
            return False, "SKIP: missing data to compute N"
        try:
            N = (self.a ** 2) / (self.lambda_ * self.L)
        except ZeroDivisionError:
            return False, "L or lambda is zero"
        expected_ratio = 4 * (math.sin(math.pi * N / 2.0) ** 2)
        # Compare with reported ratio
        if math.isclose(self.ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ratio {self.ratio} matches formula value {expected_ratio:.5f}"
        else:
            return False, f"ratio {self.ratio} does not match formula value {expected_ratio:.5f} (N={N:.5f})"

    # ---- Modeling/Trace Tests (quality of derived values) ----
    def test_modeling_L_derivation(self):
        # L = a^2 / lambda (since N=1)
        if self.a is None or self.lambda_ is None or self.L is None:
            return False, "SKIP: missing a, lambda, or L"
        expected_L = (self.a ** 2) / self.lambda_
        if math.isclose(self.L, expected_L, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"L = {self.L} matches a^2/lambda = {expected_L:.6f}"
        else:
            return False, f"L = {self.L} differs from a^2/lambda = {expected_L:.6f}"

    def test_modeling_ratio_nonnegative(self):
        if self.ratio is None:
            return False, "SKIP: ratio not available"
        if self.ratio >= 0:
            return True, f"ratio is nonnegative ({self.ratio})"
        else:
            return False, f"ratio negative ({self.ratio})"

    # ---- Monotonicity Tests (trend checks) ----
    def test_monotonicity_increasing_a(self):
        # If aperture radius increases, L should increase (L ~ a^2)
        if self.a is None or self.lambda_ is None or self.L is None:
            return False, "SKIP: missing data for monotonicity check"
        # Original L
        L_base = (self.a ** 2) / self.lambda_
        # Perturb a by +5%
        a_pert = self.a * 1.05
        L_pert = (a_pert ** 2) / self.lambda_
        if L_pert > L_base:
            return True, f"L increases when a increases by 5% (L_base={L_base:.6f}, L_pert={L_pert:.6f})"
        else:
            return False, f"L does not increase when a increases (L_base={L_base:.6f}, L_pert={L_pert:.6f})"

    def test_monotonicity_increasing_lambda(self):
        # If wavelength increases, L should decrease (L ~ 1/lambda)
        if self.a is None or self.lambda_ is None or self.L is None:
            return False, "SKIP: missing data for monotonicity check"
        L_base = (self.a ** 2) / self.lambda_
        # Perturb lambda by +5%
        lam_pert = self.lambda_ * 1.05
        L_pert = (self.a ** 2) / lam_pert
        if L_pert < L_base:
            return True, f"L decreases when lambda increases (L_base={L_base:.6f}, L_pert={L_pert:.6f})"
        else:
            return False, f"L does not decrease when lambda increases (L_base={L_base:.6f}, L_pert={L_pert:.6f})"

    # ---- Answer Sanity (weak, only one) ----
    def test_answer_sanity_L_magnitude(self):
        # Only check broad range: L should be in [0.1 m, 10 m]? Actually expected 0.5 m.
        if self.L is None:
            return False, "SKIP: L not available"
        # Use weak sanity: L between 0.1 and 5 m (no exact comparison)
        if 0.1 <= self.L <= 5.0:
            return True, f"L = {self.L} m is within plausible range [0.1, 5.0]"
        else:
            return False, f"L = {self.L} m outside plausible range [0.1, 5.0]"

    # ---- Runner ----
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)