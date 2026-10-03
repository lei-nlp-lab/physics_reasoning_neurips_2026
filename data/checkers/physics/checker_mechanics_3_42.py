import math
import traceback

class PhysicsChecker:
    """Auditor for the photon-proton collision threshold problem."""

    def __init__(self, solution_instance):
        # Safe extraction of all required data (no numeric defaults)
        self.results = getattr(solution_instance, 'results', None) or {}
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Extract result quantities with None fallback
        self.E0 = self.results.get('E_0', None)              # in units of M_p c^2
        self.v_over_c = self.results.get('v_over_c', None)    # unitless ratio

    # ----------------------------------------------------------------------
    # Helpers: physics in units where M_p*c^2 = 1
    def _s_initial(self, E):
        """Invariant s = (E+1)^2 - E^2 = 2E + 1."""
        return 2 * E + 1

    def _gamma(self, v_over_c):
        if v_over_c is None or v_over_c <= 0 or v_over_c >= 1:
            return None
        return 1.0 / math.sqrt(1.0 - v_over_c * v_over_c)

    def _expected_v_over_c(self, E):
        """v/c = E/(E+1) for threshold scenario."""
        return E / (E + 1.0)

    # ----------------------------------------------------------------------
    # Interface checks
    def test_interface_required_results_exist(self):
        """Check that the essential result keys are present."""
        if self.E0 is None:
            return (False, "Missing required key 'E_0' in results.")
        if self.v_over_c is None:
            return (False, "Missing required key 'v_over_c' in results.")
        return (True, "All required result keys present.")

    # ----------------------------------------------------------------------
    # Invariant / theorem-based checks (using math.isclose)
    def test_invariant_threshold_energy(self):
        """Check that E0 satisfies s_initial = (5)^2 = 25 (threshold invariant)."""
        if self.E0 is None:
            return (False, "SKIP: E_0 not available.")
        s = self._s_initial(self.E0)
        if not math.isclose(s, 25.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Invariant s = {s}, expected ~25 (threshold).")
        return (True, f"s = {s} matches threshold invariant 25.")

    def test_invariant_positive_energy(self):
        """Photon energy must be positive."""
        if self.E0 is None:
            return (False, "SKIP: E_0 not available.")
        if self.E0 <= 0:
            return (False, f"E_0 = {self.E0} is not positive.")
        return (True, "E_0 > 0.")

    def test_invariant_velocity_range(self):
        """Velocity must be between 0 and 1 (non-relativistic limit < c)."""
        if self.v_over_c is None:
            return (False, "SKIP: v_over_c not available.")
        if not (0 < self.v_over_c < 1):
            return (False, f"v_over_c = {self.v_over_c} not in (0,1).")
        return (True, "v_over_c is in (0,1).")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (using math.isclose)
    def test_modeling_velocity_from_energy(self):
        """At threshold, v/c = E_0/(E_0 + M_p c^2) = E_0/(E_0+1)."""
        if self.E0 is None or self.v_over_c is None:
            return (False, "SKIP: E_0 or v_over_c missing.")
        expected_v = self._expected_v_over_c(self.E0)
        if not math.isclose(self.v_over_c, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"v/c = {self.v_over_c}, expected ~{expected_v} from E_0.")
        return (True, f"v/c = {self.v_over_c} matches formula E_0/(E_0+1).")

    def test_modeling_energy_balance_at_threshold(self):
        """Check that total initial energy equals total final energy using gamma from v/c."""
        if self.E0 is None or self.v_over_c is None:
            return (False, "SKIP: E_0 or v_over_c missing.")
        gamma = self._gamma(self.v_over_c)
        if gamma is None:
            return (False, "Could not compute gamma from v_over_c.")
        total_initial = self.E0 + 1.0
        # Final state: proton (mass 1) + psi' (mass 4), both with gamma factor
        total_final = 5.0 * gamma
        if not math.isclose(total_initial, total_final, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Initial energy {total_initial} ≠ final energy {total_final} (gamma={gamma})")
        return (True, f"Energy balance holds: {total_initial} ≈ {total_final}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (no math.isclose, no re-derivation)
    def test_answer_sanity_broad_range(self):
        """Final answers should be within reasonable physical ranges."""
        passes = True
        msgs = []
        if self.E0 is not None:
            if not (10 <= self.E0 <= 14):
                passes = False
                msgs.append(f"E_0 = {self.E0} not in [10,14]")
        if self.v_over_c is not None:
            if not (0.9 <= self.v_over_c <= 1.0):
                passes = False
                msgs.append(f"v/c = {self.v_over_c} not in [0.9,1.0]")
        if msgs:
            return (False, "; ".join(msgs))
        return (True, "Final answers within plausible ranges.")

    # ----------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation)
    def test_monotonicity_invariant_increases_with_energy(self):
        """Check that s increases with E by perturbing E0 by -10%."""
        if self.E0 is None:
            return (False, "SKIP: E_0 not available.")
        baseline_s = self._s_initial(self.E0)
        perturbed_E = 0.9 * self.E0
        perturbed_s = self._s_initial(perturbed_E)
        if perturbed_s >= baseline_s - 1e-3:
            return (False, f"s({perturbed_E}) = {perturbed_s} is not less than s(E0) = {baseline_s}")
        return (True, f"s increases with E: s({perturbed_E})={perturbed_s} < s(E0)={baseline_s}.")

    def test_monotonicity_velocity_increases_with_energy(self):
        """Check that v/c increases with E by perturbing E0 by -10%."""
        if self.E0 is None or self.v_over_c is None:
            return (False, "SKIP: E_0 or v_over_c missing.")
        baseline_v = self.v_over_c
        perturbed_E = 0.9 * self.E0
        perturbed_v = self._expected_v_over_c(perturbed_E)
        if perturbed_v >= baseline_v - 1e-3:
            return (False, f"v/c({perturbed_E}) = {perturbed_v} is not less than v/c(E0) = {baseline_v}")
        return (True, f"v/c increases with E: v/c({perturbed_E})={perturbed_v} < v/c(E0)={baseline_v}.")

    # ----------------------------------------------------------------------
    # Run all tests
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