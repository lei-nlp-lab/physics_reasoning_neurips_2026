import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.delta_nu = results.get('delta_nu', None)
        self.delta_nu_line = results.get('delta_nu_line', None)
        self.N_modes = results.get('N_modes', None)

        # Extract givens from various possible locations
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        self.c = getattr(solution_instance, 'c', None)
        self.L = getattr(solution_instance, 'L', None)
        self.lambda_ = getattr(solution_instance, 'lambda', None)  # avoid reserved word
        self.delta_lambda = getattr(solution_instance, 'delta_lambda', None)

        # Fallback to results or givens dict
        if self.c is None:
            self.c = self.givens.get('c', None)
        if self.L is None:
            self.L = self.givens.get('L', None)
        if self.lambda_ is None:
            self.lambda_ = self.givens.get('lambda', None) or self.givens.get('lambda_', None)
        if self.delta_lambda is None:
            self.delta_lambda = self.givens.get('delta_lambda', None)

    # ----------------------------------------
    # Interface checks – verify required keys exist and are numeric/positive
    # ----------------------------------------
    def test_interface_delta_nu_exists(self):
        if self.delta_nu is None:
            return (False, "SKIP: delta_nu not found in results")
        if not isinstance(self.delta_nu, (int, float)):
            return (False, "FAIL: delta_nu is not numeric")
        if self.delta_nu <= 0:
            return (False, "FAIL: delta_nu must be positive")
        return (True, "delta_nu exists and is positive")

    def test_interface_delta_nu_line_exists(self):
        if self.delta_nu_line is None:
            return (False, "SKIP: delta_nu_line not found in results")
        if not isinstance(self.delta_nu_line, (int, float)):
            return (False, "FAIL: delta_nu_line is not numeric")
        if self.delta_nu_line <= 0:
            return (False, "FAIL: delta_nu_line must be positive")
        return (True, "delta_nu_line exists and is positive")

    def test_interface_N_modes_exists(self):
        if self.N_modes is None:
            return (False, "SKIP: N_modes not found in results")
        if not isinstance(self.N_modes, (int, float)):
            return (False, "FAIL: N_modes is not numeric")
        if self.N_modes <= 0:
            return (False, "FAIL: N_modes must be positive")
        return (True, "N_modes exists and is positive")

    # ----------------------------------------
    # Invariant checks – theorem-based physical sanity (use math.isclose)
    # ----------------------------------------
    def test_invariant_delta_nu_formula(self):
        """Δν = c/(2L)"""
        if self.c is None or self.L is None:
            return (False, "SKIP: c or L not available")
        expected = self.c / (2.0 * self.L)
        if self.delta_nu is None:
            return (False, "SKIP: delta_nu not available")
        if not math.isclose(self.delta_nu, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: delta_nu {self.delta_nu} does not match expected {expected} (c/(2L))")
        return (True, "delta_nu matches c/(2L) within tolerance")

    def test_invariant_delta_nu_line_formula(self):
        """Δν_line = (c/λ²) * Δλ"""
        if self.c is None or self.lambda_ is None or self.delta_lambda is None:
            return (False, "SKIP: c, lambda, or delta_lambda not available")
        if self.lambda_ <= 0:
            return (False, "FAIL: lambda must be positive")
        if self.delta_lambda <= 0:
            return (False, "FAIL: delta_lambda must be positive")
        expected = (self.c / (self.lambda_**2)) * self.delta_lambda
        if self.delta_nu_line is None:
            return (False, "SKIP: delta_nu_line not available")
        if not math.isclose(self.delta_nu_line, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: delta_nu_line {self.delta_nu_line} does not match expected {expected} (c/λ² * Δλ)")
        return (True, "delta_nu_line matches (c/λ²)*Δλ within tolerance")

    def test_invariant_N_modes_relation(self):
        """N_modes ≈ Δν_line / Δν"""
        if self.delta_nu is None or self.delta_nu_line is None:
            return (False, "SKIP: delta_nu or delta_nu_line not available")
        if self.delta_nu <= 0:
            return (False, "FAIL: delta_nu must be positive for division")
        expected_ratio = self.delta_nu_line / self.delta_nu
        if self.N_modes is None:
            return (False, "SKIP: N_modes not available")
        if not math.isclose(self.N_modes, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: N_modes {self.N_modes} does not match ratio {expected_ratio}")
        return (True, "N_modes matches Δν_line/Δν within tolerance")

    # ----------------------------------------
    # Modeling / trace checks – consistency with explanation steps
    # ----------------------------------------
    def test_modeling_N_modes_integer_approximant(self):
        """Check N_modes is close to an integer (since it's a count)"""
        if self.N_modes is None:
            return (False, "SKIP: N_modes not available")
        nearest_int = round(self.N_modes)
        if abs(self.N_modes - nearest_int) > 0.1:
            return (False, f"FAIL: N_modes {self.N_modes} is not close to an integer (threshold 0.1)")
        return (True, f"N_modes is close to integer {nearest_int}")

    def test_modeling_delta_nu_line_from_wavelength(self):
        """Check delta_nu_line is derived from given delta_lambda and lambda (overall shape)"""
        if self.lambda_ is None or self.delta_lambda is None or self.delta_nu_line is None:
            return (False, "SKIP: lambda, delta_lambda, or delta_nu_line missing")
        # The linewidth in frequency should be much larger than mode spacing because N_modes > 1
        if self.delta_nu is not None and self.delta_nu_line <= self.delta_nu:
            return (False, f"FAIL: delta_nu_line {self.delta_nu_line} should be > delta_nu {self.delta_nu} for multiple modes")
        return (True, "delta_nu_line is greater than delta_nu (consistent with multiple modes)")

    # ----------------------------------------
    # Monotonicity checks – perturb inputs and verify trend
    # ----------------------------------------
    def test_monotonicity_delta_nu_decreases_with_L(self):
        """If L increases by 10%, delta_nu should decrease"""
        if self.c is None or self.L is None:
            return (False, "SKIP: c or L not available")
        if self.delta_nu is None:
            return (False, "SKIP: delta_nu not available")
        # Perturb L by +10%
        L_perturbed = self.L * 1.1
        delta_nu_perturbed = self.c / (2.0 * L_perturbed)
        if delta_nu_perturbed > self.delta_nu:
            return (False, f"FAIL: increasing L should decrease delta_nu, but perturbed {delta_nu_perturbed} >= original {self.delta_nu}")
        return (True, "Monotonicity: delta_nu decreases with increasing L")

    def test_monotonicity_N_modes_increases_with_delta_lambda(self):
        """If delta_lambda increases, delta_nu_line increases, so N_modes should increase"""
        if self.lambda_ is None or self.delta_lambda is None or self.c is None:
            return (False, "SKIP: lambda, delta_lambda, or c not available")
        if self.delta_nu is None or self.N_modes is None:
            return (False, "SKIP: delta_nu or N_modes not available")
        # Perturb delta_lambda by +10%
        delta_lambda_perturbed = self.delta_lambda * 1.1
        delta_nu_line_perturbed = (self.c / (self.lambda_**2)) * delta_lambda_perturbed
        N_modes_perturbed = delta_nu_line_perturbed / self.delta_nu
        if N_modes_perturbed <= self.N_modes:
            return (False, f"FAIL: increasing delta_lambda should increase N_modes, but perturbed {N_modes_perturbed} <= original {self.N_modes}")
        return (True, "Monotonicity: N_modes increases with increasing delta_lambda")

    # ----------------------------------------
    # Weak answer sanity (only one such method, no re-derivation)
    # ----------------------------------------
    def test_answer_sanity_N_modes_multiple_modes(self):
        """Weak sanity: N_modes should be > 1 (laser operates at several axial frequencies)"""
        if self.N_modes is None:
            return (False, "SKIP: N_modes not available")
        if self.N_modes <= 1:
            return (False, f"FAIL: N_modes {self.N_modes} suggests single mode operation, but expected multiple modes")
        return (True, "N_modes > 1, consistent with multiple axial frequencies")

    # ----------------------------------------
    # Master test runner
    # ----------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test methods
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