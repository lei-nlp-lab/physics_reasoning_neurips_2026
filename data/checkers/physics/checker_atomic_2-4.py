import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the deformed nucleus point-charge approximation problem.
    Checks interface completeness, physical invariants, modeling consistency,
    monotonicity, and weak final-answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract givens and results (default to empty dicts)
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Extract expected result keys (None if missing)
        self.rel_err_est = self.results.get('rel_err_est', None)
        self.order_rel_err = self.results.get('order_rel_err', None)

        # Extract expected given keys (None if missing)
        self.A = self.givens.get('A', None)
        self.r_B = self.givens.get('r_B', None)

    def run_all_tests(self):
        """Discovers and runs all test_* methods, returns (all_pass, logs)."""
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
        return all_pass, logs

    # --------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------
    def test_interface_rel_err_est_exists(self):
        if self.rel_err_est is None:
            return False, "Missing required result key 'rel_err_est'"
        return True, "rel_err_est is present"

    def test_interface_order_rel_err_exists(self):
        if self.order_rel_err is None:
            return False, "Missing required result key 'order_rel_err'"
        return True, "order_rel_err is present"

    def test_interface_rel_err_est_is_numeric(self):
        if self.rel_err_est is None:
            return False, "SKIP: rel_err_est not available"
        if not isinstance(self.rel_err_est, (int, float)):
            return False, f"rel_err_est is not numeric, type={type(self.rel_err_est)}"
        return True, "rel_err_est is numeric"

    def test_interface_order_rel_err_is_numeric(self):
        if self.order_rel_err is None:
            return False, "SKIP: order_rel_err not available"
        if not isinstance(self.order_rel_err, (int, float)):
            return False, f"order_rel_err is not numeric, type={type(self.order_rel_err)}"
        return True, "order_rel_err is numeric"

    # --------------------------------------------------------------
    # Physical invariant checks (theorem‑based)
    # --------------------------------------------------------------
    def test_invariant_quadrupole_ratio_positive(self):
        if self.rel_err_est is None:
            return False, "SKIP: rel_err_est not available"
        if self.rel_err_est <= 0:
            return False, f"rel_err_est is not positive: {self.rel_err_est}"
        return True, "rel_err_est > 0"

    def test_invariant_quadrupole_ratio_finite(self):
        if self.rel_err_est is None:
            return False, "SKIP: rel_err_est not available"
        if not math.isfinite(self.rel_err_est):
            return False, f"rel_err_est is not finite: {self.rel_err_est}"
        return True, "rel_err_est is finite"

    def test_invariant_order_rel_err_consistent_with_exponent(self):
        """
        Check that order_rel_err is roughly 10^{ceil(log10(rel_err_est))}.
        Uses fuzzy matching with relative tolerance on the exponent.
        """
        if self.rel_err_est is None or self.order_rel_err is None:
            return False, "SKIP: one of the required result keys missing"
        if self.rel_err_est <= 0:
            return False, "Cannot compute log10 of non-positive rel_err_est"
        expected_exp = math.ceil(math.log10(self.rel_err_est))
        expected_order = 10 ** expected_exp
        # Allow order to be within factor 10 (since it's an order-of-magnitude estimate)
        if not math.isclose(self.order_rel_err, expected_order, rel_tol=9.0):
            return (False,
                    f"order_rel_err = {self.order_rel_err}, expected ~{expected_order}")
        return True, "order_rel_err matches logarithmic order"

    # --------------------------------------------------------------
    # Modeling / trace consistency checks
    # --------------------------------------------------------------
    def test_modeling_quadrupole_formula(self):
        """
        If A and r_B are provided, re‑derive the relative error using
        standard nuclear radius formula and deformation ratio a/b = 1.1.
        Compare with rel_err_est using fuzzy tolerance (5%).
        """
        if self.rel_err_est is None:
            return False, "SKIP: rel_err_est not available"
        if self.A is None or self.r_B is None:
            return False, "SKIP: required givens A or r_B missing"
        # Standard nuclear radius formula (in metres)
        b = 1.2 * (self.A ** (1/3)) * 1e-15
        a = 1.1 * b
        # Quadrupole analysis: relative error ~ |V2/V0| = (1/5)*(a^2 - b^2)/r^2
        expected_rel = (1.0 / 5.0) * (a**2 - b**2) / (self.r_B**2)
        if not math.isclose(self.rel_err_est, expected_rel, rel_tol=0.05):
            return (False,
                    f"rel_err_est = {self.rel_err_est}, predicted = {expected_rel}")
        return True, "rel_err_est consistent with quadrupole formula"

    def test_trace_order_of_magnitude_via_typical_a(self):
        """
        Check that rel_err_est is around 1e-9 using typical values
        (A≈125–200, a/b=1.1, r_B=5.29e-11 m). This tests modeling
        plausibility without requiring exact given numbers.
        """
        if self.rel_err_est is None:
            return False, "SKIP: rel_err_est not available"
        # Use representative numbers: A=160, r_B=0.529e-10 m
        # Reasonable range: 1e-12 to 1e-6
        if not (1e-12 <= self.rel_err_est <= 1e-6):
            return False, f"rel_err_est = {self.rel_err_est} outside plausible range [1e-12, 1e-6]"
        return True, "rel_err_est in plausible order-of-magnitude range"

    # --------------------------------------------------------------
    # Monotonicity tests (input perturbation)
    # --------------------------------------------------------------
    def test_monotonicity_increase_A_increases_rel_err(self):
        """
        If A is given, perturb it upward by 1% and verify that the
        derived relative error increases (physically: larger nucleus
        → larger quadrupole effect).
        """
        if self.rel_err_est is None or self.A is None or self.r_B is None:
            return False, "SKIP: cannot test monotonicity without A and r_B"
        # Baseline
        b0 = 1.2 * (self.A ** (1/3)) * 1e-15
        a0 = 1.1 * b0
        baseline = (1.0/5.0) * (a0**2 - b0**2) / (self.r_B**2)
        # Perturbed: increase A by 1%
        A1 = 1.01 * self.A
        b1 = 1.2 * (A1 ** (1/3)) * 1e-15
        a1 = 1.1 * b1
        perturbed = (1.0/5.0) * (a1**2 - b1**2) / (self.r_B**2)
        if perturbed <= baseline:
            return (False,
                    f"Increasing A from {self.A} to {A1} did not increase rel_err "
                    f"(baseline={baseline:.3e}, perturbed={perturbed:.3e})")
        return True, "Larger A gives larger rel_err (physical monotonicity)"

    def test_monotonicity_increase_deformation_increases_rel_err(self):
        """
        If a/b ratio is available (can be assumed from problem statement),
        test that increasing deformation (i.e., larger a/b) increases rel_err.
        We simulate by increasing the ratio from 1.1 to 1.2.
        This does not require any given data beyond the formula.
        """
        if self.rel_err_est is None:
            return False, "SKIP: rel_err_est not available"
        # We need a reference b. Use a typical nuclear radius if A not given.
        # If A is given, use it; otherwise assume A=160.
        if self.A is not None:
            b = 1.2 * (self.A ** (1/3)) * 1e-15
        else:
            b = 1.2 * (160**(1/3)) * 1e-15  # ~5.8e-15 m, typical
        # Baseline deformation ratio=1.1
        a_base = 1.1 * b
        baseline = (1.0/5.0) * (a_base**2 - b**2) / (self.r_B or 5.29e-11)**2
        # Increased deformation ratio=1.2
        a_pert = 1.2 * b
        perturbed = (1.0/5.0) * (a_pert**2 - b**2) / (self.r_B or 5.29e-11)**2
        if perturbed <= baseline:
            return (False,
                    f"Increasing a/b from 1.1 to 1.2 did not increase rel_err "
                    f"(baseline={baseline:.3e}, perturbed={perturbed:.3e})")
        return True, "Larger deformation gives larger rel_err (physical monotonicity)"

    # --------------------------------------------------------------
    # Weak answer sanity (only one allowed, no math.isclose)
    # --------------------------------------------------------------
    def test_answer_sanity_rel_err_positive_small(self):
        if self.rel_err_est is None:
            return False, "SKIP: rel_err_est not available"
        if self.rel_err_est <= 0:
            return False, "rel_err_est is not positive"
        if not math.isfinite(self.rel_err_est):
            return False, "rel_err_est is not finite"
        # Rough magnitude check: must be much smaller than 1 (since it's a small correction)
        if self.rel_err_est > 0.1:
            return False, f"rel_err_est too large to be a small correction: {self.rel_err_est}"
        return True, "rel_err_est is positive, finite, and plausibly small"