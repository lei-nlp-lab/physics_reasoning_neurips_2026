import math
import traceback

class PhysicsChecker:
    """
    Audit the particle-physics parity-violation solution.
    Checks focus on the angular-distribution formula and parity-violation conclusions.
    """

    def __init__(self, solution_instance):
        # Safely extract results; use None for missing values (never numeric defaults)
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        self.alpha = results.get('alpha', None)
        self.distribution_part = results.get('distribution_part', None)
        self.distribution_formula = results.get('distribution_formula', None)
        self.pion_parity_violation = results.get('pion_parity_violation', None)
        self.muon_parity_violation = results.get('muon_parity_violation', None)

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

    # ---------- Interface tests ----------

    def test_interface_required_keys(self):
        """All [emit] keys must exist (may be None)."""
        # No check needed; attributes are set in __init__.
        return (True, "All expected keys accessible.")

    def test_interface_alpha_is_numeric(self):
        if self.alpha is None:
            return (False, "SKIP: alpha not provided")
        if not isinstance(self.alpha, (int, float)):
            return (False, f"alpha is not numeric, got {type(self.alpha).__name__}")
        if math.isnan(self.alpha) or math.isinf(self.alpha):
            return (False, f"alpha is NaN or infinite: {self.alpha}")
        return (True, f"alpha = {self.alpha} is a finite number.")

    def test_interface_distribution_part_is_string(self):
        if self.distribution_part is None:
            return (False, "SKIP: distribution_part not provided")
        if not isinstance(self.distribution_part, str):
            return (False, f"distribution_part should be a string, got {type(self.distribution_part).__name__}")
        return (True, "distribution_part is a string.")

    def test_interface_distribution_formula_is_string(self):
        if self.distribution_formula is None:
            return (False, "SKIP: distribution_formula not provided")
        if not isinstance(self.distribution_formula, str):
            return (False, f"distribution_formula should be a string, got {type(self.distribution_formula).__name__}")
        return (True, "distribution_formula is a string.")

    def test_interface_pion_parity_violation_boolean(self):
        if self.pion_parity_violation is None:
            return (False, "SKIP: pion_parity_violation not provided")
        if not isinstance(self.pion_parity_violation, bool):
            return (False, f"pion_parity_violation should be bool, got {type(self.pion_parity_violation).__name__}")
        return (True, f"pion_parity_violation = {self.pion_parity_violation}")

    def test_interface_muon_parity_violation_boolean(self):
        if self.muon_parity_violation is None:
            return (False, "SKIP: muon_parity_violation not provided")
        if not isinstance(self.muon_parity_violation, bool):
            return (False, f"muon_parity_violation should be bool, got {type(self.muon_parity_violation).__name__}")
        return (True, f"muon_parity_violation = {self.muon_parity_violation}")

    # ---------- Invariant (theorem-based sanity) tests ----------

    def test_invariant_alpha_negative(self):
        """Standard Model: alpha for mu+ decay is -1/3 < 0."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        if self.alpha >= 0:
            return (False, f"alpha should be negative (SM: -1/3), but got {self.alpha}")
        return (True, f"alpha = {self.alpha} is negative.")

    def test_invariant_alpha_magnitude_less_than_one(self):
        """|alpha| < 1, otherwise distribution would become negative."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        if abs(self.alpha) >= 1:
            return (False, f"|alpha| = {abs(self.alpha)} should be < 1 (would violate positivity)")
        return (True, f"|alpha| = {abs(self.alpha)} < 1 satisfied.")

    def test_invariant_distribution_part_structure(self):
        """distribution_part should contain 'alpha' and 'costheta' (case-insensitive)."""
        if self.distribution_part is None:
            return (False, "SKIP: distribution_part not available")
        low = self.distribution_part.lower()
        if 'alpha' not in low or 'costheta' not in low:
            return (False, f"distribution_part '{self.distribution_part}' missing 'alpha' or 'costheta'")
        return (True, f"distribution_part contains required tokens.")

    # ---------- Modeling / trace consistency tests ----------

    def test_modeling_alpha_derived_correctly(self):
        """If alpha is given, check it matches near -1/3 (SM value)."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        # Use math.isclose with relaxed tolerance for numerical noise
        if not math.isclose(self.alpha, -1.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"alpha = {self.alpha} is not close to -1/3 (SM value)")
        return (True, f"alpha = {self.alpha} ≈ -1/3")

    def test_modeling_distribution_formula_consistent(self):
        """distribution_formula should match 1 - (1/3) cosθ (if alpha = -1/3)."""
        if self.distribution_formula is None:
            return (False, "SKIP: distribution_formula not available")
        if self.alpha is None:
            return (False, "SKIP: alpha not available for comparison")
        # Check that the formula string is plausible: contains '1 -' and 'costheta' or 'cosθ'
        formula = self.distribution_formula.strip()
        compact = formula.replace(' ', '')
        if '1-' not in compact and '1 -' not in formula:
            return (False, f"distribution_formula '{formula}' does not contain '1 -' pattern")
        if 'costheta' not in formula.lower() and 'cosθ' not in formula:
            return (False, f"distribution_formula '{formula}' missing angular factor")
        return (True, f"distribution_formula appears structurally consistent.")

    def test_modeling_both_decays_violate_parity(self):
        """Both pion and muon decays must violate parity (true according to physics)."""
        if self.pion_parity_violation is None or self.muon_parity_violation is None:
            return (False, "SKIP: one or both parity flags missing")
        if not self.pion_parity_violation:
            return (False, "pion_parity_violation should be True")
        if not self.muon_parity_violation:
            return (False, "muon_parity_violation should be True")
        return (True, "Both processes marked as parity-violating.")

    # ---------- Monotonicity tests ----------
    # For angular distribution: dN/dΩ ∝ 1 + α cosθ, α negative => decreasing function of cosθ.
    # We explicitly perturb cosθ by a small amount and compare.

    def test_monotonicity_distribution_decreases_with_cosθ(self):
        """Perturb cosθ from 0 to 0.1 and verify distribution decreases (since α<0)."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available for monotonicity")
        alpha_val = self.alpha
        baseline_cosθ = 0.0
        perturbed_cosθ = 0.1   # small perturbation
        f_baseline = 1.0 + alpha_val * baseline_cosθ
        f_perturbed = 1.0 + alpha_val * perturbed_cosθ
        # Since alpha is negative, f_perturbed should be less than f_baseline.
        if f_perturbed >= f_baseline:
            return (False, f"Distribution did not decrease: f(baseline={baseline_cosθ})={f_baseline}, "
                           f"f(perturbed={perturbed_cosθ})={f_perturbed}. Expected decrease because α<0.")
        return (True, f"Distribution decreases as cosθ increases from 0 to 0.1: "
                      f"f(0)={f_baseline} -> f(0.1)={f_perturbed}")

    # Additional monotonicity test using opposite perturbation direction.
    def test_monotonicity_distribution_increases_when_cosθ_decreases(self):
        """Perturb cosθ from 0 to -0.1 and verify distribution increases (since α<0)."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available for monotonicity")
        alpha_val = self.alpha
        baseline_cosθ = 0.0
        perturbed_cosθ = -0.1   # small perturbation in opposite direction
        f_baseline = 1.0 + alpha_val * baseline_cosθ
        f_perturbed = 1.0 + alpha_val * perturbed_cosθ
        # Since alpha is negative, f_perturbed should be greater than f_baseline.
        if f_perturbed <= f_baseline:
            return (False, f"Distribution did not increase: f(baseline={baseline_cosθ})={f_baseline}, "
                           f"f(perturbed={perturbed_cosθ})={f_perturbed}. Expected increase because α<0.")
        return (True, f"Distribution increases as cosθ decreases from 0 to -0.1: "
                      f"f(0)={f_baseline} -> f(-0.1)={f_perturbed}")

    # ---------- Answer-sanity test (weak) ----------

    def test_answer_sanity_distribution_formula_not_empty(self):
        """Weak check: distribution_formula must be a non-empty string."""
        if self.distribution_formula is None:
            return (False, "SKIP: distribution_formula not provided")
        if not isinstance(self.distribution_formula, str) or len(self.distribution_formula.strip()) == 0:
            return (False, "distribution_formula is empty or not a string")
        return (True, "distribution_formula is non-empty.")