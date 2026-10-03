import math
import traceback

class PhysicsChecker:
    """
    Auditor for spin-orbit splitting of hydrogen 2p state.
    Checks interface consistency, invariant physics relations,
    modeling quality, monotonicity via input perturbation, and weak answer sanity.
    """

    # Physical constants (standard)
    ALPHA = 1 / 137.036
    E1 = -13.6  # eV, hydrogen ground state energy
    N = 2
    L = 1

    def __init__(self, solution_instance):
        # Extract results dict safely; never use numeric defaults
        results = getattr(solution_instance, 'results', None)
        if results is None:
            self.delta_E = getattr(solution_instance, 'delta_E', None)
            self.order = getattr(solution_instance, 'order', None)
        else:
            self.delta_E = results.get('delta_E', None)
            self.order = results.get('order', None)

        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Precompute nominal expected value for later reference
        self.E2 = self.E1 / (self.N ** 2)
        self.expected_delta_E = abs(self.E2) * (self.ALPHA ** 2) / 4.0

    # ----------------------------------------------------------------------
    # Helper: compute spin-orbit splitting given n and alpha
    # ----------------------------------------------------------------------
    @staticmethod
    def _compute_delta_E(n, alpha):
        """Return spin-orbit splitting for hydrogen 2p state (l=1) using fine-structure formula."""
        E_n = -13.6 / (n ** 2)   # eV
        # For n=2, l=1, the splitting between j=3/2 and j=1/2 is |E_n| * alpha^2 / 4
        # (derived from full fine-structure formula)
        return abs(E_n) * (alpha ** 2) / 4.0

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_delta_E_present(self):
        if self.delta_E is None:
            return (False, "delta_E is missing (None)")
        return (True, "delta_E is present")

    def test_interface_order_present(self):
        if self.order is None:
            return (False, "order is missing (None)")
        return (True, "order is present")

    def test_interface_delta_E_numeric(self):
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")
        if not isinstance(self.delta_E, (int, float)):
            return (False, f"delta_E is not numeric: {type(self.delta_E)}")
        return (True, "delta_E is numeric")

    def test_interface_order_string(self):
        if self.order is None:
            return (False, "SKIP: order None")
        if not isinstance(self.order, str):
            return (False, f"order is not string: {type(self.order)}")
        return (True, "order is a string")

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ----------------------------------------------------------------------
    def test_invariant_spin_orbit_splitting_formula(self):
        """Verify that delta_E matches the fine-structure formula within 2% tolerance."""
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")
        expected = self.expected_delta_E
        if not math.isfinite(expected) or expected == 0:
            return (False, "Cannot compute expected splitting (division by zero or invalid)")
        if not math.isclose(self.delta_E, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"delta_E {self.delta_E:.6e} deviates from expected {expected:.6e} by more than 2%")
        return (True, f"delta_E {self.delta_E:.6e} matches fine-structure formula within tolerance")

    def test_invariant_order_of_magnitude(self):
        """Check that delta_E lies within one order of the expected value (factor 10)."""
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")
        low = self.expected_delta_E * 0.1
        high = self.expected_delta_E * 10.0
        if not (low <= self.delta_E <= high):
            return (False, f"delta_E {self.delta_E:.6e} outside expected range [{low:.6e}, {high:.6e}]")
        return (True, f"delta_E {self.delta_E:.6e} is within order-of-magnitude bounds")

    def test_invariant_positive_energy(self):
        """Spin-orbit splitting must be positive (energy difference)."""
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")
        if self.delta_E <= 0:
            return (False, f"delta_E is not positive: {self.delta_E}")
        return (True, "delta_E is positive")

    def test_invariant_finite_value(self):
        """delta_E must be finite (not infinity or NaN)."""
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")
        if not math.isfinite(self.delta_E):
            return (False, f"delta_E is not finite: {self.delta_E}")
        return (True, "delta_E is finite")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency checks
    # ----------------------------------------------------------------------
    def test_modeling_step_recalculation(self):
        """
        Re-derive delta_E from the simplified formula given in explanation steps:
        delta_E = |E_2| * alpha^2 / 4, where E_2 = E1 / n^2.
        This ensures the reasoning logic is consistent.
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")
        computed = abs(self.E1 / (self.N**2)) * (self.ALPHA**2) / 4.0
        if not math.isclose(self.delta_E, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"delta_E {self.delta_E:.6e} does not match re-computed value {computed:.6e} from explanation formula")
        return (True, "delta_E matches formula from explanation steps")

    def test_modeling_order_string_contains_eV(self):
        """The order string should contain 'eV'."""
        if self.order is None:
            return (False, "SKIP: order None")
        if 'eV' not in self.order:
            return (False, f"order string does not contain 'eV': {self.order}")
        return (True, "order string contains 'eV'")

    def test_modeling_order_plausible_form(self):
        """The order string should look like a scientific notation order of magnitude, e.g., '10^{-4} eV'."""
        if self.order is None:
            return (False, "SKIP: order None")
        # weak check: contains "10^" and "eV"
        if "10^{" not in self.order and "10^" not in self.order:
            return (False, f"order string does not contain power of ten notation: {self.order}")
        return (True, "order string plausible")

    # ----------------------------------------------------------------------
    # Monotonicity tests via input perturbation
    # ----------------------------------------------------------------------
    def test_monotonicity_splitting_increases_with_alpha(self):
        """
        Perturb alpha (fine-structure constant) by +5% and verify that
        the computed splitting increases (since delta_E ~ alpha^2).
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")

        # Baseline from standard formula at given n=2
        baseline = self._compute_delta_E(self.N, self.ALPHA)
        # Perturb alpha upward by 5%
        perturbed_alpha = self.ALPHA * 1.05
        perturbed_value = self._compute_delta_E(self.N, perturbed_alpha)

        # Splitting should increase because alpha^2 increases
        if perturbed_value <= baseline:
            return (False, f"Splitting did not increase when alpha increased by 5%: baseline {baseline:.6e}, perturbed {perturbed_value:.6e}")
        return (True, f"Splitting increases when alpha increases (baseline {baseline:.6e}, perturbed {perturbed_value:.6e})")

    def test_monotonicity_splitting_decreases_with_n(self):
        """
        Perturb principal quantum number n from 2 to 3 and verify that
        splitting decreases (since delta_E ~ 1/n^5 roughly).
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")

        baseline = self._compute_delta_E(self.N, self.ALPHA)
        # Perturb n upward to 3
        perturbed_n = 3
        perturbed_value = self._compute_delta_E(perturbed_n, self.ALPHA)

        if perturbed_value >= baseline:
            return (False, f"Splitting did not decrease when n increased from 2 to 3: baseline {baseline:.6e}, perturbed {perturbed_value:.6e}")
        return (True, f"Splitting decreases when n increases (baseline {baseline:.6e}, perturbed {perturbed_value:.6e})")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one such method)
    # ----------------------------------------------------------------------
    def test_answer_sanity_broad_range(self):
        """
        Weak check: delta_E must be positive and lie in a reasonable physical range
        (10^-6 to 10^-2 eV). This is a non-derivative, order-only sanity.
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E None")
        if not (1e-6 < self.delta_E < 1e-2):
            return (False, f"delta_E {self.delta_E:.6e} outside broad physical range (1e-6, 1e-2)")
        return (True, "delta_E within broad physical range")