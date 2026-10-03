import math
import sympy
import traceback

class PhysicsChecker:
    """
    Audits the thermal conductivity solution for granite.
    Checks include interface presence, physical invariants,
    modeling consistency with Fourier's law and damping depth,
    monotonicity of the relation, and a weak order-of-magnitude sanity.
    """

    def __init__(self, solution_instance):
        # Extract results (mandatory)
        results = getattr(solution_instance, 'results', {})
        self.k = results.get('k', None)

        # Extract givens (optional, may come as attributes or dict)
        self.delta = getattr(solution_instance, 'delta', None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.c = getattr(solution_instance, 'c', None)
        self.T = getattr(solution_instance, 'T', None)   # period of daily variation

        # Fallback to givens dict if not set as attributes
        givens = getattr(solution_instance, 'givens', {})
        if self.delta is None:
            self.delta = givens.get('delta', None)
        if self.rho is None:
            self.rho = givens.get('rho', None)
        if self.c is None:
            self.c = givens.get('c', None)
        if self.T is None:
            self.T = givens.get('T', None)

        # Derived angular frequency
        self.omega = None
        if self.T is not None and self.T > 0:
            self.omega = 2 * math.pi / self.T

        # Constants from the problem's Fourier estimate (used for modeling only)
        self.Q_approx = 3.3e-2          # cal/s·cm²  (solar intensity)
        self.dx_approx = 10.0           # cm
        self.dT_approx = 10.0           # °C
        self.upper_bound_k = self.Q_approx * self.dx_approx / self.dT_approx  # 3.3e-2

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_k_exists(self):
        if self.k is None:
            return (False, "Key 'k' missing from results.")
        return (True, "k is present.")

    def test_interface_k_type(self):
        if self.k is None:
            return (False, "SKIP: k not available.")
        if not isinstance(self.k, (int, float)):
            return (False, f"k has type {type(self.k).__name__}, expected numeric.")
        return (True, "k is numeric.")

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) tests
    # ----------------------------------------------------------------------
    def test_invariant_k_positive(self):
        if self.k is None:
            return (False, "SKIP: k not available.")
        if self.k <= 0:
            return (False, f"k = {self.k} is not positive.")
        return (True, "k > 0.")

    def test_invariant_k_finite(self):
        if self.k is None:
            return (False, "SKIP: k not available.")
        if not math.isfinite(self.k):
            return (False, f"k = {self.k} is not finite.")
        return (True, "k is finite.")

    def test_invariant_k_order(self):
        """Thermal conductivity of solids (in these units) is typically 10^{-3}–10^{-1}.
           Here we check that k lies in a plausible range for granite."""
        if self.k is None:
            return (False, "SKIP: k not available.")
        # Using isclose for tolerance in invariant test
        if not (1e-4 < self.k < 0.1):
            return (False, f"k = {self.k} is outside expected order of magnitude (1e-4 to 0.1).")
        return (True, "k in plausible magnitude range (1e-4–0.1).")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ----------------------------------------------------------------------
    def test_modeling_fourier_upper_bound(self):
        """The Fourier estimate with full absorption gives an upper bound.
           The true conduction must be smaller because part of the radiation is reflected."""
        if self.k is None:
            return (False, "SKIP: k not available.")
        if self.k > self.upper_bound_k:
            return (False, f"k = {self.k} exceeds the Fourier upper bound {self.upper_bound_k}.")
        return (True, f"k <= Fourier upper bound ({self.upper_bound_k}).")

    def test_modeling_damping_depth_consistency(self):
        """If the givens (delta, rho, c, T) are provided, recompute k from the
           damping depth formula and compare with the result using isclose."""
        if self.k is None:
            return (False, "SKIP: k not available.")
        if any(x is None for x in [self.delta, self.rho, self.c, self.omega]):
            return (False, "SKIP: missing givens for damping depth check (delta, rho, c, T).")
        expected_k = self.rho * self.c * self.omega * (self.delta ** 2) / 2.0
        if not math.isclose(self.k, expected_k, rel_tol=5e-2, abs_tol=1e-4):
            return (False, f"k computed ({self.k}) differs from expected ({expected_k}) by more than 5%.")
        return (True, f"k agrees with damping depth formula (expected {expected_k:.2e}).")

    def test_modeling_explanation_reflection_factor(self):
        """The explanation states that the value shall be smaller than the rough Fourier estimate.
           Check that k is significantly smaller (by a factor of at least 2)."""
        if self.k is None:
            return (False, "SKIP: k not available.")
        if self.k < self.upper_bound_k / 2:
            return (True, f"k = {self.k} is less than half the Fourier bound, consistent with reflection.")
        return (False, f"k = {self.k} is not significantly below the Fourier bound; reflection not accounted for.")

    # ----------------------------------------------------------------------
    # Monotonicity test
    # ----------------------------------------------------------------------
    def test_monotonicity_k_with_density(self):
        """
        If the density rho is increased (while all other parameters remain constant),
        the thermal conductivity k should increase (monotonic positive relationship).
        We recompute k for a 10% higher density using the damping depth formula,
        and check that the perturbed value is greater than the original formula value.
        """
        if any(x is None for x in [self.delta, self.rho, self.c, self.omega]):
            return (False, "SKIP: missing givens (delta, rho, c, omega) for monotonicity test.")
        # Baseline k from the formula (the solution's result should match this)
        k_base = self.rho * self.c * self.omega * (self.delta ** 2) / 2.0
        # Perturb density by +10%
        rho_pert = self.rho * 1.1
        k_pert = rho_pert * self.c * self.omega * (self.delta ** 2) / 2.0
        if k_pert <= k_base:
            return (False, f"Predicted k for +10% density ({k_pert}) is not > baseline ({k_base}).")
        return (True, f"Monotonicity holds: higher density yields higher k (predicted {k_pert} > {k_base}).")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one allowed: order-of-magnitude)
    # ----------------------------------------------------------------------
    def test_answer_sanity_order(self):
        """Very weak check: the answer must be around 10^{-3} cal/(s·cm·°C)."""
        if self.k is None:
            return (False, "SKIP: k not available.")
        # Use a broad range that includes 5e-3 but also allows slight variation.
        if not (1e-4 < self.k < 1e-2):
            return (False, f"k = {self.k} is not in the 10^{-3} range (1e-4 to 1e-2).")
        return (True, "k is within 10^{-3} order of magnitude.")

    # ----------------------------------------------------------------------
    # Runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)