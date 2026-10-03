import math
import traceback

class PhysicsChecker:
    """Robust checker for relativistic Doppler shift problem."""

    def __init__(self, solution_instance):
        # Extract givens and results safely, default to None
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}

        # Givens (from problem statement)
        self.c = self.givens.get('c', None)
        self.omega0 = self.givens.get('omega0', None)
        self.v = self.givens.get('v', None)

        # Derived quantities (None if missing)
        self.beta = None
        if self.c is not None and self.v is not None and self.c != 0:
            self.beta = self.v / self.c
        self.gamma = None
        if self.beta is not None and abs(self.beta) < 1:
            self.gamma = 1.0 / math.sqrt(1.0 - self.beta ** 2)

        # Results expected from the solution
        self.justification = self.results.get('justification', None)
        self.omega_away = self.results.get('omega_away', None)
        self.omega_towards = self.results.get('omega_towards', None)

    # ------------------------------------------------------------------
    # Interface tests – required keys exist and have valid types
    # ------------------------------------------------------------------

    def test_interface_justification(self):
        if self.justification is None:
            return (False, "Missing or None: 'justification'")
        if not isinstance(self.justification, str) or len(self.justification) < 10:
            return (False, "'justification' is not a meaningful string")
        return (True, "'justification' present and adequate")

    def test_interface_omega_away(self):
        if self.omega_away is None:
            return (False, "Missing or None: 'omega_away'")
        if not isinstance(self.omega_away, (int, float)):
            return (False, "'omega_away' is not numeric")
        return (True, "'omega_away' present and numeric")

    def test_interface_omega_towards(self):
        if self.omega_towards is None:
            return (False, "Missing or None: 'omega_towards'")
        if not isinstance(self.omega_towards, (int, float)):
            return (False, "'omega_towards' is not numeric")
        return (True, "'omega_towards' present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests – fundamental physical constraints
    # ------------------------------------------------------------------

    def test_invariant_product(self):
        """Check ω_away * ω_towards = ω0² (relativistic invariant)."""
        if self.omega_away is None or self.omega_towards is None or self.omega0 is None:
            return (False, "SKIP: missing omega_away, omega_towards, or omega0")
        product = self.omega_away * self.omega_towards
        expected = self.omega0 ** 2
        if math.isclose(product, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Product = {product} ≈ ω0² = {expected}")
        else:
            return (False, f"Product {product} != ω0² {expected}")

    def test_invariant_ordering(self):
        """For a source moving toward the observer (v>0), ω_towards > ω0 > ω_away."""
        if self.omega_away is None or self.omega_towards is None or self.omega0 is None:
            return (False, "SKIP: missing frequencies")
        if not (self.omega_towards > self.omega0 > self.omega_away):
            return (False, f"Ordering wrong: ω_towards={self.omega_towards}, "
                           f"ω0={self.omega0}, ω_away={self.omega_away}")
        return (True, "Correct ordering: ω_towards > ω0 > ω_away")

    # ------------------------------------------------------------------
    # Modeling / trace tests – verify the Lorentz transformation and
    # the simplified Doppler formulas are correctly used.
    # ------------------------------------------------------------------

    def test_modeling_transformation_away(self):
        """Verify ω_away = γ (ω0 - v ω0/c) from the Lorentz boost."""
        if self.gamma is None or self.omega0 is None or self.v is None or self.c is None:
            return (False, "SKIP: missing inputs for transformation check")
        if self.omega_away is None:
            return (False, "SKIP: omega_away missing")
        expected = self.gamma * self.omega0 * (1 - self.beta)
        if math.isclose(self.omega_away, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ω_away = {self.omega_away} ≈ Lorentz boost result {expected}")
        else:
            return (False, f"ω_away {self.omega_away} != {expected} from Lorentz boost")

    def test_modeling_transformation_towards(self):
        """Verify ω_towards = γ (ω0 + v ω0/c)."""
        if self.gamma is None or self.omega0 is None or self.v is None or self.c is None:
            return (False, "SKIP: missing inputs for transformation check")
        if self.omega_towards is None:
            return (False, "SKIP: omega_towards missing")
        expected = self.gamma * self.omega0 * (1 + self.beta)
        if math.isclose(self.omega_towards, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ω_towards = {self.omega_towards} ≈ Lorentz boost result {expected}")
        else:
            return (False, f"ω_towards {self.omega_towards} != {expected} from Lorentz boost")

    def test_modeling_simplified_form_away(self):
        """Verify ω_away = ω0 √((1-β)/(1+β))."""
        if self.omega0 is None or self.beta is None or self.omega_away is None:
            return (False, "SKIP: missing parameters")
        expected = self.omega0 * math.sqrt((1 - self.beta) / (1 + self.beta))
        if math.isclose(self.omega_away, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "ω_away matches simplified Doppler formula")
        else:
            return (False, f"ω_away {self.omega_away} != {expected}")

    def test_modeling_simplified_form_towards(self):
        """Verify ω_towards = ω0 √((1+β)/(1-β))."""
        if self.omega0 is None or self.beta is None or self.omega_towards is None:
            return (False, "SKIP: missing parameters")
        expected = self.omega0 * math.sqrt((1 + self.beta) / (1 - self.beta))
        if math.isclose(self.omega_towards, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "ω_towards matches simplified Doppler formula")
        else:
            return (False, f"ω_towards {self.omega_towards} != {expected}")

    # ------------------------------------------------------------------
    # Monotonicity tests – explicitly perturb an input and compare output
    # ------------------------------------------------------------------

    def _compute_towards(self, v):
        """Compute ω_towards for a given v (uses self.omega0, self.c)."""
        if self.omega0 is None or self.c is None:
            return None
        beta = v / self.c
        if abs(beta) >= 1:
            return None
        return self.omega0 * math.sqrt((1 + beta) / (1 - beta))

    def _compute_away(self, v):
        """Compute ω_away for a given v."""
        if self.omega0 is None or self.c is None:
            return None
        beta = v / self.c
        if abs(beta) >= 1:
            return None
        return self.omega0 * math.sqrt((1 - beta) / (1 + beta))

    def test_monotonicity_towards_increasing(self):
        """
        Perturb v upward and check that ω_towards increases.
        Use a 10% increase, but ensure v + delta < c.
        """
        if self.omega0 is None or self.v is None or self.c is None or self.omega_towards is None:
            return (False, "SKIP: missing givens/results for monotonicity test")

        # Choose a small positive delta (e.g., 10% of v, but at least 1 m/s, and ensure stays below c)
        delta = min(0.1 * self.v, (self.c - self.v) * 0.5)
        if delta <= 0:
            delta = self.c * 1e-3  # tiny safety

        v_new = self.v + delta
        if v_new >= self.c:
            return (False, "Cannot perturb: v already too close to c")

        omega_new = self._compute_towards(v_new)
        if omega_new is None:
            return (False, "Failed to compute perturbed frequency")

        if omega_new > self.omega_towards:
            return (True, f"Perturbed ω_towards {omega_new} > baseline {self.omega_towards} (v increased)")
        else:
            return (False, f"Perturbed ω_towards {omega_new} <= baseline {self.omega_towards}, "
                           f"expected increase when v increased")

    def test_monotonicity_away_decreasing(self):
        """
        Perturb v upward and check that ω_away decreases.
        """
        if self.omega0 is None or self.v is None or self.c is None or self.omega_away is None:
            return (False, "SKIP: missing givens/results for monotonicity test")

        delta = min(0.1 * self.v, (self.c - self.v) * 0.5)
        if delta <= 0:
            delta = self.c * 1e-3

        v_new = self.v + delta
        if v_new >= self.c:
            return (False, "Cannot perturb: v already too close to c")

        omega_new = self._compute_away(v_new)
        if omega_new is None:
            return (False, "Failed to compute perturbed frequency")

        if omega_new < self.omega_away:
            return (True, f"Perturbed ω_away {omega_new} < baseline {self.omega_away} (v increased)")
        else:
            return (False, f"Perturbed ω_away {omega_new} >= baseline {self.omega_away}, "
                           f"expected decrease when v increased")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one such method)
    # ------------------------------------------------------------------

    def test_answer_sanity_frequencies_positive(self):
        """Both derived frequencies must be positive and finite."""
        if self.omega_away is None or self.omega_towards is None:
            return (False, "SKIP: results missing")
        if self.omega_away <= 0 or self.omega_towards <= 0:
            return (False, "One frequency is non‑positive")
        if math.isnan(self.omega_away) or math.isinf(self.omega_away):
            return (False, "omega_away is NaN or Inf")
        if math.isnan(self.omega_towards) or math.isinf(self.omega_towards):
            return (False, "omega_towards is NaN or Inf")
        return (True, "Both frequencies positive and finite")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)