import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Auditor for WKB scattering problem with two turning points.
    Checks interface invariants, modeling relations, monotonicity, and answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.gamma = None
        self.T = None
        self.A_F_relation = None

        # Try to get gamma from various sources
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.gamma = solution_instance.givens.get('gamma', None)
        if self.gamma is None and hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.gamma = solution_instance.results.get('gamma', None)

        # Extract T (should be numeric) and A_F_relation (string) from results
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.T = solution_instance.results.get('T', None)
            self.A_F_relation = solution_instance.results.get('A_F_relation', None)

        # No numeric defaults – let tests fail explicitly

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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_results_keys_present(self):
        """Check that required keys exist in results dictionary."""
        if self.T is None:
            return (False, "SKIP: results['T'] not found (None)")
        if self.A_F_relation is None:
            return (False, "SKIP: results['A_F_relation'] not found (None)")
        return (True, "Both key results ('T' and 'A_F_relation') are present")

    def test_interface_t_is_numeric(self):
        """Verify that T is a numeric value (float/int)."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if not isinstance(self.T, (int, float)):
            return (False, f"T is not numeric, got {type(self.T).__name__}")
        if math.isnan(self.T) or math.isinf(self.T):
            return (False, "T is NaN or Inf")
        return (True, "T is a finite numeric value")

    def test_interface_a_f_relation_is_string(self):
        """Verify that A_F_relation is a non-empty string."""
        if self.A_F_relation is None:
            return (False, "SKIP: A_F_relation not available")
        if not isinstance(self.A_F_relation, str):
            return (False, f"A_F_relation is not a string, got {type(self.A_F_relation).__name__}")
        if len(self.A_F_relation.strip()) == 0:
            return (False, "A_F_relation is an empty string")
        return (True, "A_F_relation is a non-empty string")

    def test_interface_gamma_present(self):
        """Check that gamma (barrier factor) is provided in givens or results."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available; many physics checks cannot run")
        return (True, "gamma is available")

    # ------------------------------------------------------------------
    # Invariant tests (using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_t_range(self):
        """T must be between 0 and 1 (probability)."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if not math.isclose(self.T, 0.0, abs_tol=1e-12) and self.T < -1e-12:
            return (False, f"T is negative: {self.T:.6e}")
        if not math.isclose(self.T, 1.0, abs_tol=1e-12) and self.T > 1.0 + 1e-12:
            return (False, f"T exceeds 1: {self.T:.6e}")
        return (True, f"T = {self.T:.6e} lies in [0,1] (within tolerance)")

    def test_invariant_t_gamma_relation(self):
        """Theoretical relation: T = exp(-2*gamma) / (1 - (exp(-gamma)/2)^2)^2."""
        if self.gamma is None or self.T is None:
            return (False, "SKIP: gamma or T missing")
        if self.gamma < 0:
            return (False, f"gamma should be non-negative, got {self.gamma:.6e}")
        try:
            T_theory = math.exp(-2 * self.gamma) / (1 - (math.exp(-self.gamma) / 2)**2)**2
        except OverflowError:
            return (False, "Overflow in theoretical T computation (gamma too large?)")
        if not math.isclose(self.T, T_theory, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T = {self.T:.6e} does not match theoretical value {T_theory:.6e} (rel_tol=1e-2)")
        return (True, f"T matches theoretical value within 1% tolerance")

    def test_invariant_t_positive(self):
        """Transmission coefficient must be positive (finite)."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if self.T <= 0:
            return (False, f"T is non-positive: {self.T:.6e}")
        return (True, f"T = {self.T:.6e} > 0")

    # ------------------------------------------------------------------
    # Modeling tests based on explanation steps / relations
    # ------------------------------------------------------------------
    def test_modeling_a_f_relation_structure(self):
        """A_F_relation string should resemble 'A = i(1/4 e^{-gamma} - e^{gamma})F'."""
        if self.A_F_relation is None:
            return (False, "SKIP: A_F_relation not available")
        text = self.A_F_relation.replace(" ", "")
        checks = [
            "A" in text,
            "i" in text,
            "e^{-\\gamma}" in text or "exp(-gamma)" in text,
            "e^{\\gamma}" in text or "exp(gamma)" in text,
            "1/4" in text,
            "F" in text
        ]
        if not all(checks):
            return (False, f"A_F_relation string lacks required components: {self.A_F_relation}")
        return (True, "A_F_relation contains expected mathematical structure")

    def test_modeling_gamma_nonnegative(self):
        """Gamma (barrier factor) must be nonnegative."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        if self.gamma < 0:
            return (False, f"gamma is negative: {self.gamma:.6e}")
        return (True, f"gamma = {self.gamma:.6e} >= 0")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation, compare to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_t_decreases_small_gamma_increase(self):
        """If gamma increases by 10%, T should decrease (or stay same)."""
        if self.gamma is None or self.T is None:
            return (False, "SKIP: gamma or T missing")
        base_T = self.T
        perturbed_gamma = self.gamma * 1.1
        try:
            T_pert = math.exp(-2 * perturbed_gamma) / (1 - (math.exp(-perturbed_gamma) / 2)**2)**2
        except OverflowError:
            return (False, "Overflow in perturbed T computation")
        if T_pert > base_T + 1e-12:
            return (False, f"T did not decrease when gamma increased from {self.gamma:.6e} to {perturbed_gamma:.6e}: base_T={base_T:.6e}, T_pert={T_pert:.6e}")
        return (True, f"T decreases as gamma increases (base={base_T:.6e}, perturbed={T_pert:.6e})")

    def test_monotonicity_t_decreases_large_gamma_increase(self):
        """If gamma is increased tenfold, T should be much smaller than original T."""
        if self.gamma is None or self.T is None:
            return (False, "SKIP: gamma or T missing")
        if self.gamma == 0:
            return (False, "SKIP: gamma is zero, cannot perturb with factor")
        base_T = self.T
        perturbed_gamma = self.gamma * 10.0
        try:
            T_pert = math.exp(-2 * perturbed_gamma) / (1 - (math.exp(-perturbed_gamma) / 2)**2)**2
        except OverflowError:
            return (False, "Overflow in perturbed T computation")
        if T_pert >= base_T - 1e-15:
            return (False, f"T did not decrease when gamma increased tenfold: base_T={base_T:.6e}, T_pert={T_pert:.6e}")
        return (True, f"T decreased from {base_T:.6e} to {T_pert:.6e} when gamma increased tenfold")

    # ------------------------------------------------------------------
    # Final answer sanity (weak, single method)
    # ------------------------------------------------------------------
    def test_answer_sanity_t_basic(self):
        """Basic sanity: T must be a finite number between 0 and 1 (inclusive)."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if math.isinf(self.T) or math.isnan(self.T):
            return (False, f"T is inf or nan: {self.T}")
        if self.T < 0.0 or self.T > 1.0:
            return (False, f"T out of physical range [0,1]: {self.T:.6e}")
        return (True, f"T = {self.T:.6e} is finite and within [0,1]")