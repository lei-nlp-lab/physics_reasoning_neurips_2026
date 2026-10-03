import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or self.results.get('explanation_steps', [])

        # Problem-specific givens
        self.Q = self.givens.get('Q', None)
        self.L = self.givens.get('L', None)
        self.R = self.givens.get('R', None)
        self.T_R = self.givens.get('T_R', None)
        self.V = self.givens.get('Vdot', self.givens.get('V', None))
        self.p0 = self.givens.get('p0', None)
        self.T0 = self.givens.get('T0', None)

        # Results (emitted by solution)
        self.p_p = self.results.get('p_p', None)           # symbolic or numeric
        self.T_He_formula = self.results.get('T_He_formula', None)
        self.p_p_num = self.results.get('p_p_num', None)
        self.T_He_num = self.results.get('T_He_num', None)

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
        return all_pass, logs

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_required_results_keys(self):
        """Check that all required result keys exist and are not None."""
        required = ['p_p', 'T_He_formula', 'p_p_num', 'T_He_num']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return False, f"Missing or None results: {missing}"
        return True, "All required result keys present."

    def test_interface_givens_availability(self):
        """Warn if critical givens are missing (do not fail)."""
        required = ['Q', 'L', 'R', 'T_R', 'V', 'p0', 'T0']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return True, f"Warning: missing givens: {missing}. Some tests may be skipped."
        return True, "All critical givens available."

    # ----------------------------------------------------------------
    # Invariant / physics sanity checks (use math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_heat_ideal_gas_relation(self):
        """Check that p_p_num = Q R T_R / (L V) within tolerance."""
        if any(x is None for x in [self.Q, self.L, self.R, self.T_R, self.V, self.p_p_num]):
            return False, "SKIP: missing givens or p_p_num"
        expected_p_p = (self.Q * self.R * self.T_R) / (self.L * self.V)
        if not math.isclose(self.p_p_num, expected_p_p, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"p_p_num {self.p_p_num} differs from expected {expected_p_p}"
        return True, f"p_p_num matches Q R T_R / (L V): {self.p_p_num} ≈ {expected_p_p}"

    def test_invariant_clausius_clapeyron_integrated(self):
        """Check that ln(p_p_num/p0) = (L/R) * (1/T0 - 1/T_He_num)."""
        if any(x is None for x in [self.p0, self.T0, self.L, self.R, self.p_p_num, self.T_He_num]):
            return False, "SKIP: missing givens or numerical results"
        LHS = math.log(self.p_p_num / self.p0)
        RHS = (self.L / self.R) * (1 / self.T0 - 1 / self.T_He_num)
        if not math.isclose(LHS, RHS, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"CC integrated mismatch: LHS={LHS}, RHS={RHS}"
        return True, "Clausius-Clapeyron integrated relation holds."

    def test_invariant_pressure_less_than_atmospheric(self):
        """p_p_num must be less than p0 (physical)."""
        if self.p_p_num is None or self.p0 is None:
            return False, "SKIP: missing p_p_num or p0"
        if self.p_p_num >= self.p0:
            return False, f"p_p_num ({self.p_p_num}) is not less than p0 ({self.p0})"
        return True, f"p_p_num < p0: {self.p_p_num} < {self.p0}"

    def test_invariant_temperature_less_than_boiling(self):
        """T_He_num must be less than T0."""
        if self.T_He_num is None or self.T0 is None:
            return False, "SKIP: missing T_He_num or T0"
        if self.T_He_num >= self.T0:
            return False, f"T_He_num ({self.T_He_num}) is not less than T0 ({self.T0})"
        return True, f"T_He_num < T0: {self.T_He_num} < {self.T0}"

    # ----------------------------------------------------------------
    # Modeling / trace quality tests
    # ----------------------------------------------------------------
    def test_modeling_explanation_steps_contain_key_concepts(self):
        """Check that explanation steps mention at least two of the core physics relations."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps provided"
        key_concepts = [
            'heat balance', 'latent heat', 'ideal gas', 'Clausius-Clapeyron',
            'Q = L * n_dot', 'p_p V = n_dot R T_R', 'evaporation rate'
        ]
        found = [c for c in key_concepts if any(c.lower() in step.lower() for step in self.explanation_steps)]
        if len(found) < 2:
            return False, f"Less than two key concepts found. Found: {found}"
        return True, f"Modeling covers {len(found)} key concepts: {found}"

    def test_modeling_trace_clausius_clapeyron_equation(self):
        """Check that the explanation steps contain the Clausius-Clapeyron equation in any form."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps provided"
        cc_terms = ['Clausius', 'Clapeyron', 'dp/dT', 'dP/dT', 'L p / (R T^2)']
        found = any(term.lower() in ' '.join(self.explanation_steps).lower() for term in cc_terms)
        if not found:
            return False, "Explanation steps do not mention Clausius-Clapeyron equation."
        return True, "Clausius-Clapeyron equation appears in explanation steps."

    def test_modeling_trace_ideal_gas_mention(self):
        """Check that explanation steps mention the ideal gas law."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps provided"
        found = any('ideal gas' in step.lower() for step in self.explanation_steps)
        if not found:
            return False, "Explanation steps do not mention ideal gas law."
        return True, "Ideal gas law mentioned in explanation steps."

    # ----------------------------------------------------------------
    # Monotonicity tests (perturb input, compare output to baseline)
    # ----------------------------------------------------------------
    def test_monotonicity_T_He_decreases_with_increasing_Q(self):
        """If Q increases, T_He should decrease (latent heat removal faster -> lower pressure -> lower T)."""
        if any(x is None for x in [self.Q, self.L, self.R, self.T_R, self.V, self.p0, self.T0, self.T_He_num]):
            return False, "SKIP: missing required data"
        # Baseline is the solution's numerical T_He
        baseline_T_He = self.T_He_num
        # Perturb Q by +10%
        Q_new = self.Q * 1.1
        # Compute new p_p and new T_He from the same physical relations
        p_p_new = (Q_new * self.R * self.T_R) / (self.L * self.V)
        # Use Clausius-Clapeyron integrated formula (with sign of ln correct)
        term = (self.R * self.T0 / self.L) * math.log(self.p0 / p_p_new)
        T_He_new = self.T0 / (1 + term)
        if T_He_new >= baseline_T_He:
            return False, f"Increasing Q did not decrease T_He: baseline={baseline_T_He:.4f}, new={T_He_new:.4f}"
        return True, f"Increasing Q from {self.Q} to {Q_new} decreases T_He from {baseline_T_He:.3f} to {T_He_new:.3f}"

    def test_monotonicity_T_He_increases_with_increasing_V(self):
        """If volume flow rate V increases, T_He should increase (higher flow -> lower pressure drop? Actually higher V -> lower p_p -> lower T_He. Wait, reversing direction? Let's reason: Higher V at constant Q means smaller p_p (since p_p = QRT_R/(LV)), lower p_p -> lower T_He, so T_He should decrease. But the problem asks for "T_He increases with increasing V"? Check: p_p = Q R T_R / (L V), so larger V => smaller p_p. Then ln(p0/p_p) is larger, so T_He = T0/(1 + (R T0/L) ln(p0/p_p)) => denominator larger => T_He smaller. So T_He decreases with increasing V. But the test name says 'increases' - that would be physically wrong. Let's correct direction: T_He decreases when V increases. We'll adjust the test name and logic to match physics."""
        if any(x is None for x in [self.Q, self.L, self.R, self.T_R, self.V, self.p0, self.T0, self.T_He_num]):
            return False, "SKIP: missing required data"
        baseline_T_He = self.T_He_num
        V_new = self.V * 1.1
        p_p_new = (self.Q * self.R * self.T_R) / (self.L * V_new)
        term = (self.R * self.T0 / self.L) * math.log(self.p0 / p_p_new)
        T_He_new = self.T0 / (1 + term)
        # Physics: larger V => smaller p_p => smaller T_He
        if T_He_new >= baseline_T_He:
            return False, f"Increasing V did not decrease T_He as expected: baseline={baseline_T_He:.4f}, new={T_He_new:.4f}"
        return True, f"Increasing V from {self.V} to {V_new} decreases T_He from {baseline_T_He:.3f} to {T_He_new:.3f}"

    # ----------------------------------------------------------------
    # Weak final-answer sanity (no math.isclose, no re-derivation)
    # ----------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """Very basic sanity: p_p_num and T_He_num are positive, finite, and within plausible ranges."""
        if self.p_p_num is None or self.T_He_num is None:
            return False, "SKIP: numerical results missing"
        # Check positivity and finiteness
        if self.p_p_num <= 0 or not math.isfinite(self.p_p_num):
            return False, f"p_p_num is not positive and finite: {self.p_p_num}"
        if self.T_He_num <= 0 or not math.isfinite(self.T_He_num):
            return False, f"T_He_num is not positive and finite: {self.T_He_num}"
        # Broad physical ranges
        if self.p0 is not None and self.p_p_num > self.p0 / 10:
            return False, f"p_p_num {self.p_p_num} seems too close to p0 {self.p0}"
        if self.T0 is not None and (self.T_He_num > self.T0 or self.T_He_num < 0.5):
            return False, f"T_He_num {self.T_He_num} out of plausible range [0.5, {self.T0}]"
        return True, f"p_p_num={self.p_p_num}, T_He_num={self.T_He_num} pass basic sanity."