import math
import traceback

class PhysicsChecker:
    """
    Checker for the Carnot cycle with phase change problem.
    Verifies efficiency and Clausius-Clapeyron relation.
    """

    def __init__(self, solution_instance):
        """
        Extract all needed quantities from the solution instance.
        Use None defaults for missing values.
        """
        # Results emitted by the solver
        self.results = getattr(solution_instance, 'results', {})
        self.eta = getattr(solution_instance, 'eta', self.results.get('eta', None))
        self.dpdT = getattr(solution_instance, 'dpdT', self.results.get('dpdT', None))
        self.dpdT2 = getattr(solution_instance, 'dpdT2', self.results.get('dpdT2', None))

        # Explanation steps (expected to be a list of step dicts)
        self.explanation = getattr(solution_instance, 'explanation_steps', []) or \
                           getattr(solution_instance, 'explanation', [])

        # Try to extract givens (may be attributes or in results)
        self.v_g = getattr(solution_instance, 'v_g', self.results.get('v_g', None))
        self.v_l = getattr(solution_instance, 'v_l', self.results.get('v_l', None))
        self.L_v = getattr(solution_instance, 'L_v', self.results.get('L_v', None))
        self.n = getattr(solution_instance, 'n', self.results.get('n', None))
        self.T = getattr(solution_instance, 'T', self.results.get('T', None))
        self.delta_T = getattr(solution_instance, 'delta_T', self.results.get('delta_T', None))
        self.delta_p = getattr(solution_instance, 'delta_p', self.results.get('delta_p', None))
        self.s_g = getattr(solution_instance, 's_g', self.results.get('s_g', None))
        self.s_l = getattr(solution_instance, 's_l', self.results.get('s_l', None))

        # Derived quantities that we compute from givens if possible
        self._derived_eta_formula = None
        self._derived_dpdT_clausius = None
        self._derived_dpdT_gibbs = None
        if self.v_g is not None and self.v_l is not None and self.delta_p is not None and self.L_v is not None:
            self._derived_eta_formula = (self.v_g - self.v_l) * self.delta_p / self.L_v
        if self.L_v is not None and self.T is not None and self.v_g is not None and self.v_l is not None:
            self._derived_dpdT_clausius = self.L_v / (self.T * (self.v_g - self.v_l))
        if self.s_g is not None and self.s_l is not None and self.v_g is not None and self.v_l is not None:
            self._derived_dpdT_gibbs = (self.s_g - self.s_l) / (self.v_g - self.v_l)

    def run_all_tests(self):
        """Discover all test_* methods and run them."""
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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        """Check that all three required result keys are present and non-None."""
        missing = []
        if self.eta is None:
            missing.append("eta")
        if self.dpdT is None:
            missing.append("dpdT")
        if self.dpdT2 is None:
            missing.append("dpdT2")
        if missing:
            return (False, f"Missing required result(s): {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_givens_available_for_checks(self):
        """Check that enough givens are present to run derived checks (optional)."""
        # Not a hard fail, but warn if missing critical ones
        missing = []
        for key in ['v_g', 'v_l', 'L_v', 'T', 'delta_T', 'delta_p']:
            if getattr(self, key, None) is None:
                missing.append(key)
        if missing:
            return (True, f"Skipping some checks due to missing givens: {missing}")
        return (True, "All givens present")

    # ------------------------------------------------------------------
    # Invariant / physics correctness tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_efficiency_formula(self):
        """Check eta = (v_g - v_l)*delta_p / L_v"""
        if self.eta is None:
            return (False, "SKIP: eta not available")
        if self._derived_eta_formula is None:
            return (False, "SKIP: insufficient givens to compute expected eta")
        if math.isclose(self.eta, self._derived_eta_formula, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"eta matches formula: {self.eta} vs {self._derived_eta_formula}")
        else:
            return (False, f"eta {self.eta} deviates from formula {self._derived_eta_formula}")

    def test_invariant_carnot_efficiency(self):
        """Check eta ≈ delta_T / T (Carnot efficiency equality)"""
        if self.eta is None or self.delta_T is None or self.T is None:
            return (False, "SKIP: missing eta, delta_T, or T")
        carnot_eta = self.delta_T / self.T
        if math.isclose(self.eta, carnot_eta, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"eta {self.eta} matches Carnot efficiency {carnot_eta}")
        else:
            return (False, f"eta {self.eta} vs Carnot {carnot_eta}")

    def test_invariant_clausius_clapeyron(self):
        """Check dpdT = L_v / (T*(v_g - v_l))"""
        if self.dpdT is None:
            return (False, "SKIP: dpdT not available")
        if self._derived_dpdT_clausius is None:
            return (False, "SKIP: insufficient givens")
        if math.isclose(self.dpdT, self._derived_dpdT_clausius, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dpdT {self.dpdT} matches Clausius-Clapeyron {self._derived_dpdT_clausius}")
        else:
            return (False, f"dpdT {self.dpdT} vs expected {self._derived_dpdT_clausius}")

    def test_invariant_gibbs_clapeyron(self):
        """Check dpdT2 = (s_g - s_l)/(v_g - v_l) (Gibbs equality)"""
        if self.dpdT2 is None:
            return (False, "SKIP: dpdT2 not available")
        if self._derived_dpdT_gibbs is None:
            return (False, "SKIP: insufficient givens for Gibbs check")
        if math.isclose(self.dpdT2, self._derived_dpdT_gibbs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dpdT2 {self.dpdT2} matches Gibbs form {self._derived_dpdT_gibbs}")
        else:
            return (False, f"dpdT2 {self.dpdT2} vs expected {self._derived_dpdT_gibbs}")

    def test_invariant_dpdT_and_dpdT2_consistent(self):
        """If both dpdT formulas are available, they should be consistent.
           Check that dpdT ≈ L_v/(T(v_g-v_l)) and dpdT2 ≈ (s_g-s_l)/(v_g-v_l) are equal.
        """
        if self.dpdT is None or self.dpdT2 is None:
            return (False, "SKIP: one of dpdT/dpdT2 missing")
        if math.isclose(self.dpdT, self.dpdT2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dpdT and dpdT2 consistent: {self.dpdT} vs {self.dpdT2}")
        else:
            return (False, f"dpdT {self.dpdT} != dpdT2 {self.dpdT2}")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_efficiency_in_explanation(self):
        """Check that explanation steps contain the efficiency formula (21.3.1)."""
        if not self.explanation:
            return (False, "SKIP: no explanation_steps")
        for step in self.explanation:
            text = str(step)
            if "eta = (v_g - v_l) * delta_p / L_v" in text or \
               "\\eta = \\frac{(v_g - v_\\ell) \\Delta p}{L_v}" in text:
                return (True, "Efficiency formula found in explanation")
        return (False, "Efficiency formula not found in explanation")

    def test_modeling_clausius_clapeyron_in_explanation(self):
        """Check that explanation contains the Clausius-Clapeyron relation (21.3.3)."""
        if not self.explanation:
            return (False, "SKIP: no explanation_steps")
        for step in self.explanation:
            text = str(step)
            if "dp_v/dT = L_v/(T(v_g - v_l))" in text or \
               "\\frac{dp_v}{dT} = \\frac{L_v}{T(v_g - v_\\ell)}" in text:
                return (True, "Clausius-Clapeyron relation found")
        return (False, "Clausius-Clapeyron relation not found in explanation")

    def test_modeling_gibbs_derivation_in_explanation(self):
        """Check that explanation includes the Gibbs free energy derivation (21.3.4)."""
        if not self.explanation:
            return (False, "SKIP: no explanation_steps")
        for step in self.explanation:
            text = str(step)
            if "dp_v/dT = (s_g - s_l)/(v_g - v_l)" in text or \
               "\\frac{dp_v}{dT} = \\frac{s_g - s_\\ell}{v_g - v_\\ell}" in text:
                return (True, "Gibbs form found in explanation")
        return (False, "Gibbs form not found in explanation")

    # ------------------------------------------------------------------
    # Monotonicity tests (if possible)
    # ------------------------------------------------------------------
    def test_monotonicity_efficiency_with_delta_p(self):
        """If we perturb delta_p upward, eta should increase (all else same)."""
        if self.eta is None or self.v_g is None or self.v_l is None or self.L_v is None:
            return (False, "SKIP: missing variables to test monotonicity")
        # Baseline efficiency formula: (v_g-v_l)*delta_p/L_v
        # Perturb delta_p by +10% and check that new eta > baseline
        baseline_eta = (self.v_g - self.v_l) * self.delta_p / self.L_v
        new_delta_p = self.delta_p * 1.10
        new_eta = (self.v_g - self.v_l) * new_delta_p / self.L_v
        if new_eta > baseline_eta:
            return (True, f"eta increases with delta_p: {baseline_eta} -> {new_eta}")
        else:
            return (False, f"Monotonicity violated: eta {baseline_eta} not < {new_eta}")

    def test_monotonicity_clausius_clapeyron_with_T(self):
        """If T increases, dpdT should decrease (provided L_v, v_g-v_l constant)."""
        if self.dpdT is None or self.L_v is None or self.v_g is None or self.v_l is None or self.T is None:
            return (False, "SKIP: missing variables")
        baseline = self.L_v / (self.T * (self.v_g - self.v_l))
        new_T = self.T * 1.10
        new_dpdT = self.L_v / (new_T * (self.v_g - self.v_l))
        if new_dpdT < baseline:
            return (True, f"dpdT decreases with T: {baseline} -> {new_dpdT}")
        else:
            return (False, f"Monotonicity violated: dpdT {baseline} not > {new_dpdT}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, very simple)
    # ------------------------------------------------------------------
    def test_answer_sanity_eta_positive_finite(self):
        """Check that eta is positive and not NaN/inf."""
        if self.eta is None:
            return (False, "SKIP: eta not available")
        if isinstance(self.eta, (int, float)) and math.isfinite(self.eta) and self.eta > 0:
            return (True, f"eta = {self.eta} positive and finite")
        else:
            return (False, f"eta = {self.eta} not positive/finite")