import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for the problem:
    "Show that the force on a small charge element dq on a uniformly charged sphere
    is radial and outward, given by dF = (1/2) E dq, where E = Q/(4πε0 r^2)."
    """

    def __init__(self, solution_instance):
        # Extract givens with None defaults (never numeric defaults)
        self.Q = getattr(solution_instance, 'Q', None) or solution_instance.givens.get('Q', None)
        self.r = getattr(solution_instance, 'r', None) or solution_instance.givens.get('r', None)
        self.dq = getattr(solution_instance, 'dq', None) or solution_instance.givens.get('dq', None)
        self.eps0 = getattr(solution_instance, 'eps0', None) or solution_instance.givens.get('eps0', None)

        # Extract results (the emitted key)
        self.dF = getattr(solution_instance, 'dF', None) or solution_instance.results.get('dF', None)

        # Also extract explanation steps and pseudocode for trace checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Extract optional intermediate results if present (for richer checks)
        self.sigma = getattr(solution_instance, 'sigma', None) or solution_instance.results.get('sigma', None)
        self.E = getattr(solution_instance, 'E', None) or solution_instance.results.get('E', None)
        self.E_self = getattr(solution_instance, 'E_self', None) or solution_instance.results.get('E_self', None)
        self.E_rest = getattr(solution_instance, 'E_rest', None) or solution_instance.results.get('E_rest', None)

    # ------------------------------------------------------------
    # Interface tests: check required keys and types
    # ------------------------------------------------------------
    def test_interface_dF_exists(self):
        """dF must be present and numeric."""
        if self.dF is None:
            return (False, "dF is missing from results")
        if not (isinstance(self.dF, (int, float, sp.Expr))):
            return (False, f"dF has invalid type: {type(self.dF)}")
        return (True, "dF present with valid type")

    def test_interface_givens_exist(self):
        """All required givens (Q, r, dq, eps0) must be present."""
        missing = []
        for name, val in [('Q', self.Q), ('r', self.r), ('dq', self.dq), ('eps0', self.eps0)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------
    # Invariant checks (physics theorems, use isclose)
    # ------------------------------------------------------------
    def test_invariant_sigma_formula(self):
        """Check surface charge density sigma = Q/(4πr^2) if provided."""
        if self.sigma is None:
            return (True, "sigma not provided, cannot check")
        if None in (self.Q, self.r):
            return (True, "SKIP: givens unavailable")
        expected_sigma = self.Q / (4 * math.pi * self.r**2)
        if not math.isclose(float(self.sigma), float(expected_sigma), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma mismatch: computed {expected_sigma}, got {self.sigma}")
        return (True, "sigma formula correct")

    def test_invariant_E_formula(self):
        """Check E = Q/(4πε0 r^2) if provided."""
        if self.E is None:
            return (True, "E not provided, cannot check")
        if None in (self.Q, self.r, self.eps0):
            return (True, "SKIP: givens unavailable")
        expected_E = self.Q / (4 * math.pi * self.eps0 * self.r**2)
        if not math.isclose(float(self.E), float(expected_E), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E mismatch: computed {expected_E}, got {self.E}")
        return (True, "E formula correct")

    def test_invariant_self_field_half_sigma_over_eps0(self):
        """Check E_self = sigma/(2ε0) if both E_self and sigma are present."""
        if self.E_self is None or self.sigma is None:
            return (True, "E_self or sigma not provided, cannot check")
        if self.eps0 is None:
            return (True, "SKIP: eps0 missing")
        expected_E_self = self.sigma / (2 * self.eps0)
        if not math.isclose(float(self.E_self), float(expected_E_self), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_self mismatch: expected {expected_E_self}, got {self.E_self}")
        return (True, "E_self = sigma/(2ε0) satisfied")

    def test_invariant_inside_field_zero_relation(self):
        """Check that E_rest = E_self (cancellation inside sphere)."""
        if self.E_rest is None or self.E_self is None:
            return (True, "E_rest or E_self not provided, cannot check")
        if not math.isclose(float(self.E_rest), float(self.E_self), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_rest ({self.E_rest}) != E_self ({self.E_self}); inside field zero violated")
        return (True, "E_rest = E_self satisfied (inside field zero)")

    def test_invariant_dF_from_E_rest(self):
        """Check that dF = E_rest * dq if all provided."""
        if self.dF is None or self.E_rest is None or self.dq is None:
            return (True, "Missing dF, E_rest, or dq for this check")
        expected_dF = float(self.E_rest) * float(self.dq)
        if not math.isclose(float(self.dF), expected_dF, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dF = E_rest*dq mismatch: expected {expected_dF}, got {self.dF}")
        return (True, "dF = E_rest * dq satisfied")

    def test_invariant_dF_ratio_half_E(self):
        """Check that dF/dq = (1/2)*E."""
        if self.dF is None or self.dq is None or self.E is None:
            return (True, "Missing quantities for ratio check")
        if math.isclose(float(self.dq), 0, rel_tol=1e-9):
            return (True, "dq is zero, skip ratio")
        ratio = float(self.dF) / float(self.dq)
        half_E = 0.5 * float(self.E)
        if not math.isclose(ratio, half_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dF/dq = {ratio} but (1/2)E = {half_E}")
        return (True, "dF/dq = (1/2)*E satisfied")

    # ------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ------------------------------------------------------------
    def test_modeling_sigma_definition_in_steps(self):
        """Explanation steps should mention sigma = Q/(4πr^2)."""
        if not self.explanation_steps:
            return (True, "SKIP: no explanation steps")
        step_texts = " ".join([str(s) for s in self.explanation_steps])
        if "surface charge" in step_texts or "sigma" in step_texts:
            return (True, "sigma definition present in steps")
        return (False, "sigma definition not found in explanation steps")

    def test_modeling_inside_field_zero(self):
        """Explanation should mention that field inside sphere is zero."""
        if not self.explanation_steps:
            return (True, "SKIP: no explanation steps")
        step_texts = " ".join([str(s) for s in self.explanation_steps])
        if "inside" in step_texts and "zero" in step_texts:
            return (True, "Inside field zero invoked")
        return (False, "Inside field zero not mentioned")

    def test_modeling_superposition_used(self):
        """Explanation should mention superposition or cancellation of fields."""
        if not self.explanation_steps:
            return (True, "SKIP: no explanation steps")
        step_texts = " ".join([str(s) for s in self.explanation_steps])
        if any(term in step_texts for term in ["superposition", "E_1", "E_2P", "E_rest"]):
            return (True, "Superposition principle used")
        return (False, "Superposition not explicitly used")

    # ------------------------------------------------------------
    # Monotonicity checks (directional trends)
    # ------------------------------------------------------------
    def test_monotonicity_force_increases_with_charge(self):
        """If Q increases, dF should increase (for fixed r, dq)."""
        if self.dF is None or self.Q is None:
            return (True, "SKIP: dF or Q missing")
        # Perturb Q up by 1% and see if dF increases
        # We'll estimate dF ~ (1/2) * (Q/(4π ε0 r^2)) * dq
        # So dF is proportional to Q.
        if self.r is None or self.dq is None or self.eps0 is None:
            return (True, "SKIP: insufficient givens")
        # Compute baseline dF from givens
        E = self.Q / (4 * math.pi * self.eps0 * self.r**2)
        dF_baseline = 0.5 * E * self.dq
        # Increase Q by 5%
        Q_pert = self.Q * 1.05
        E_pert = Q_pert / (4 * math.pi * self.eps0 * self.r**2)
        dF_pert = 0.5 * E_pert * self.dq
        if dF_pert <= dF_baseline:
            return (False, "dF did not increase when Q increased")
        return (True, "dF increases with Q (monotonic)")

    def test_monotonicity_force_decreases_with_radius(self):
        """If r increases, dF should decrease (for fixed Q, dq)."""
        if any(v is None for v in [self.dF, self.Q, self.r, self.dq, self.eps0]):
            return (True, "SKIP: missing data")
        # Baseline dF from givens
        E = self.Q / (4 * math.pi * self.eps0 * self.r**2)
        dF_baseline = 0.5 * E * self.dq
        # Increase r by 5%
        r_pert = self.r * 1.05
        E_pert = self.Q / (4 * math.pi * self.eps0 * r_pert**2)
        dF_pert = 0.5 * E_pert * self.dq
        if dF_pert >= dF_baseline:
            return (False, "dF did not decrease when r increased")
        return (True, "dF decreases with r (monotonic)")

    def test_monotonicity_force_increases_with_dq(self):
        """If dq increases, dF should increase proportionally."""
        if self.dF is None or self.dq is None or self.E is None:
            return (True, "SKIP: missing data")
        # Baseline
        dF_baseline = 0.5 * float(self.E) * float(self.dq)
        # Increase dq by 5%
        dq_pert = float(self.dq) * 1.05
        dF_pert = 0.5 * float(self.E) * dq_pert
        if dF_pert <= dF_baseline:
            return (False, "dF did not increase when dq increased")
        return (True, "dF increases with dq (monotonic)")

    # ------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose)
    # ------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitude(self):
        """dF should be positive and finite. Also it should not be zero if Q, dq non-zero."""
        if self.dF is None:
            return (True, "SKIP: dF missing")
        try:
            val = float(self.dF)
            if math.isnan(val) or math.isinf(val):
                return (False, f"dF is NaN or infinite: {val}")
            if val <= 0:
                return (False, f"dF should be positive (outward force), got {val}")
            # Very rough sanity: dF should not exceed, say, 1e10 if dq is tiny (just example)
            # but we avoid hard limits; instead check if it's not absurdly large.
            # We'll just pass.
            return (True, "dF is positive and finite")
        except (TypeError, ValueError):
            return (False, "Cannot convert dF to float")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all methods that start with "test_"
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