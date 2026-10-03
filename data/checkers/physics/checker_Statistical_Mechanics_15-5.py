import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the problem:
    (a) Prove ⟨(E-⟨E⟩)²⟩ = kT² C_V using the partition function.
    (b) Show that energy of a macroscopic system can be considered constant in equilibrium.
    """

    def __init__(self, solution_instance):
        # Safely extract the results dictionary
        results = getattr(solution_instance, 'results', {}) or {}
        self.deltaE2 = results.get('deltaE2', None)
        self.relation = results.get('relation', None)
        self.rel_fluct_sq = results.get('rel_fluct_sq', None)
        self.scaling = results.get('scaling', None)

        # Optionally extract explanation_steps (list of strings)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Fallback if it is nested inside results (unlikely but handle)
        if self.explanation_steps is None:
            self.explanation_steps = results.get('explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface tests: check that expected keys exist and have correct types
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four required keys are present (not None)."""
        if self.deltaE2 is None:
            return (False, "SKIP: 'deltaE2' is missing (None)")
        if self.relation is None:
            return (False, "SKIP: 'relation' is missing (None)")
        if self.rel_fluct_sq is None:
            return (False, "SKIP: 'rel_fluct_sq' is missing (None)")
        if self.scaling is None:
            return (False, "SKIP: 'scaling' is missing (None)")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Check that deltaE2 is numeric, relation is string, rel_fluct_sq is numeric, scaling is string."""
        if self.deltaE2 is None:
            return (False, "SKIP: deltaE2 not available")
        if self.relation is None:
            return (False, "SKIP: relation not available")
        if self.rel_fluct_sq is None:
            return (False, "SKIP: rel_fluct_sq not available")
        if self.scaling is None:
            return (False, "SKIP: scaling not available")

        # deltaE2 should be int or float (or sympy numeric, but we allow float/int)
        if not isinstance(self.deltaE2, (int, float)):
            return (False, f"deltaE2 should be numeric, got {type(self.deltaE2).__name__}")
        if not isinstance(self.relation, str):
            return (False, f"relation should be a string, got {type(self.relation).__name__}")
        if not isinstance(self.rel_fluct_sq, (int, float)):
            return (False, f"rel_fluct_sq should be numeric, got {type(self.rel_fluct_sq).__name__}")
        if not isinstance(self.scaling, str):
            return (False, f"scaling should be a string, got {type(self.scaling).__name__}")
        return (True, "All types correct")

    # ----------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_deltaE2_nonnegative(self):
        """Variance is non‑negative: deltaE2 >= 0 (with small tolerance for rounding)."""
        if self.deltaE2 is None:
            return (False, "SKIP: deltaE2 not available")
        # Allow a tiny negative due to numerical noise
        if self.deltaE2 < -1e-12:
            return (False, f"deltaE2 is negative ({self.deltaE2})")
        return (True, "deltaE2 is non‑negative")

    def test_invariant_deltaE2_finite(self):
        """Variance must be finite (not inf, not NaN)."""
        if self.deltaE2 is None:
            return (False, "SKIP: deltaE2 not available")
        if math.isfinite(self.deltaE2):
            return (True, "deltaE2 is finite")
        else:
            return (False, f"deltaE2 is not finite ({self.deltaE2})")

    def test_invariant_rel_fluct_sq_nonnegative(self):
        """Relative squared fluctuation is non‑negative."""
        if self.rel_fluct_sq is None:
            return (False, "SKIP: rel_fluct_sq not available")
        if self.rel_fluct_sq < -1e-12:
            return (False, f"rel_fluct_sq is negative ({self.rel_fluct_sq})")
        return (True, "rel_fluct_sq is non‑negative")

    # ----------------------------------------------------------------------
    # Modeling/trace tests: check consistency of relation and scaling strings
    # ----------------------------------------------------------------------
    def test_modeling_relation_string_pattern(self):
        """The relation string should contain 'kT^2' and 'C_V' (or similar)."""
        if self.relation is None:
            return (False, "SKIP: relation not available")
        rel = self.relation.strip()
        # Use simple substring checks; allow variations like 'kT^2', 'k T^2', 'C_V', 'C_V'
        has_kT2 = ('kT^2' in rel) or ('kT^2' in rel.replace(' ', ''))
        has_CV = ('C_V' in rel) or ('C_V' in rel.replace(' ', ''))
        if has_kT2 and has_CV:
            return (True, "Relation string contains required terms")
        else:
            return (False, f"Relation string missing expected terms: got '{rel}'")

    def test_modeling_scaling_string_correct(self):
        """Scaling string should indicate 1/N behaviour."""
        if self.scaling is None:
            return (False, "SKIP: scaling not available")
        s = self.scaling.strip().replace(' ', '')
        # Allow variations like '1/N', '~1/N', 'O(1/N)' etc.
        if '1/N' in s:
            return (True, f"Scaling string matches '1/N': '{self.scaling}'")
        else:
            return (False, f"Scaling string should contain '1/N', got '{self.scaling}'")

    def test_modeling_explanation_steps_present(self):
        """Check that explanation_steps is a non-empty list."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        if not isinstance(self.explanation_steps, list):
            return (False, f"explanation_steps should be a list, got {type(self.explanation_steps).__name__}")
        if len(self.explanation_steps) == 0:
            return (False, "explanation_steps is empty")
        return (True, f"explanation_steps has {len(self.explanation_steps)} step(s)")

    def test_modeling_explanation_steps_mention_derivative(self):
        """At least one step should mention derivative or partition function."""
        if self.explanation_steps is None or not isinstance(self.explanation_steps, list):
            return (False, "SKIP: explanation_steps not available or not a list")
        keywords = ['derivative', '∂', 'partial', 'ln Z', 'partition function']
        for step in self.explanation_steps:
            text = str(step).lower()
            if any(kw.lower() in text for kw in keywords):
                return (True, "Found relevant keyword in explanation steps")
        return (False, "No step mentions derivative, ∂, ln Z, or partition function")

    # ----------------------------------------------------------------------
    # Monotonicity test: verify that deltaE2 increases with temperature
    # ----------------------------------------------------------------------
    def test_monotonicity_deltaE2_increases_with_T(self):
        """
        Based on the derived formula deltaE2 = k T^2 C_V,
        verify that increasing temperature leads to an increase in deltaE2.
        We use arbitrary positive values (T=300, k=1.38e-23, C_V=1) to perform
        a small perturbation (+1%).
        """
        # Use standard values; no reliance on solution's numeric results.
        k = 1.380649e-23  # Boltzmann constant (J/K)
        T0 = 300.0        # Kelvin
        C_V = 1.0         # arbitrary positive (J/K)
        deltaE2_0 = k * T0**2 * C_V

        # Perturb temperature by +1%
        T1 = T0 * 1.01
        deltaE2_1 = k * T1**2 * C_V

        # Check that deltaE2 increased (strictly)
        if deltaE2_1 > deltaE2_0:
            return (True, f"deltaE2 increased from {deltaE2_0:.2e} to {deltaE2_1:.2e} when T increased by 1%")
        else:
            return (False, f"deltaE2 did not increase: baseline {deltaE2_0:.2e}, perturbed {deltaE2_1:.2e}")

    # ----------------------------------------------------------------------
    # Very weak final-answer sanity (only one, as per rules)
    # ----------------------------------------------------------------------
    def test_answer_sanity_deltaE2_not_NAN(self):
        """deltaE2 must not be NaN."""
        if self.deltaE2 is None:
            return (False, "SKIP: deltaE2 not available")
        if math.isnan(self.deltaE2):
            return (False, "deltaE2 is NaN")
        return (True, "deltaE2 is not NaN")

    # ----------------------------------------------------------------------
    # Run all tests
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