import sympy as sp
import math

class PhysicsChecker:
    """
    Checker for solution to parallel-plate capacitor with time-varying voltage:
    - Uses Ampere-Maxwell to find magnetic field inside, then Faraday to find induced emf.
    - Focuses on symbolic and numeric consistency of derived expressions.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        # Expected result keys from pseudocode emit contract
        self.B_mag = self.results.get('B_mag', None)
        self.emf_peak = self.results.get('emf_peak', None)

        # Physical parameters (may be symbolic or numeric)
        self.R = self.givens.get('R', None)
        self.d = self.givens.get('d', None)
        self.V0 = self.givens.get('V0', None)
        self.omega = self.givens.get('omega', None)
        self.r = self.givens.get('r', None)
        self.mu0 = self.givens.get('mu0', None)
        self.epsilon0 = self.givens.get('epsilon0', None)
        self.loop_area = self.givens.get('A', None)

        # Define symbolic variables for constructing expected expressions
        self._sym_mu0 = sp.Symbol('mu0', positive=True)
        self._sym_eps0 = sp.Symbol('epsilon0', positive=True)
        self._sym_V0 = sp.Symbol('V0', positive=True)
        self._sym_omega = sp.Symbol('omega', positive=True)
        self._sym_r = sp.Symbol('r', positive=True)
        self._sym_d = sp.Symbol('d', positive=True)
        self._sym_t = sp.Symbol('t', real=True)
        self._sym_A = sp.Symbol('A', positive=True)

    # ----------------------------------------------------------------------
    # Helper: expected symbolic expressions based on Ampere‑Maxwell + Faraday
    # ----------------------------------------------------------------------
    def _expected_B_mag(self):
        """Expected symbolic expression for B magnitude (with cos(omega*t))."""
        term = (self._sym_mu0 * self._sym_eps0 * self._sym_V0 *
                self._sym_omega * self._sym_r) / (2 * self._sym_d)
        return term * sp.cos(self._sym_omega * self._sym_t)

    def _expected_emf_peak(self):
        """Expected symbolic expression for peak induced emf."""
        term = (self._sym_mu0 * self._sym_eps0 * self._sym_V0 *
                self._sym_omega**2 * self._sym_r * self._sym_A) / (2 * self._sym_d)
        return term

    # ----------------------------------------------------------------------
    # Interface tests – check required keys exist
    # ----------------------------------------------------------------------
    def test_interface_B_mag_exists(self):
        if self.B_mag is None:
            return (False, "Missing results['B_mag']")
        return (True, "results['B_mag'] present")

    def test_interface_emf_peak_exists(self):
        if self.emf_peak is None:
            return (False, "Missing results['emf_peak']")
        return (True, "results['emf_peak'] present")

    # ----------------------------------------------------------------------
    # Invariant tests – theorem‑based symbolic consistency
    # ----------------------------------------------------------------------
    def test_invariant_ampere_maxwell(self):
        """Check that B_mag satisfies the Ampere‑Maxwell derived expression."""
        if self.B_mag is None:
            return (False, "SKIP: B_mag not available")
        try:
            expected = self._expected_B_mag()
            diff = sp.simplify(self.B_mag - expected)
            if diff == 0:
                return (True, "B_mag matches Ampere‑Maxwell expression")
            else:
                return (False, f"B_mag deviates by {diff}")
        except Exception as e:
            return (False, f"ERROR: {e}")

    def test_invariant_faraday(self):
        """Check that emf_peak satisfies the Faraday‑derived expression."""
        if self.emf_peak is None:
            return (False, "SKIP: emf_peak not available")
        try:
            expected = self._expected_emf_peak()
            diff = sp.simplify(self.emf_peak - expected)
            if diff == 0:
                return (True, "emf_peak matches Faraday expression")
            else:
                return (False, f"emf_peak deviates by {diff}")
        except Exception as e:
            return (False, f"ERROR: {e}")

    def test_invariant_relation_emf_and_B(self):
        """
        Check the intermediate relation |emf| = ω * B_amplitude * A.
        Because emf_peak = ω * B_amplitude * A (with B_amplitude = (mu0 eps0 V0 omega r)/(2d)).
        We test that emf_peak equals ω * (B_amplitude) * A.
        """
        if self.emf_peak is None or self.B_mag is None:
            return (False, "SKIP: B_mag or emf_peak missing")
        try:
            # Extract B_amplitude: from B_mag, factor out cos(ωt)
            cos_factor = sp.cos(self._sym_omega * self._sym_t)
            B_amp = sp.simplify(self.B_mag / cos_factor)
            # Now check: emf_peak == ω * B_amp * A
            expected_emf = self._sym_omega * B_amp * self._sym_A
            diff = sp.simplify(self.emf_peak - expected_emf)
            if diff == 0:
                return (True, "emf_peak = ω * B_amplitude * A holds")
            else:
                return (False, f"Relation emf = ω B A deviates by {diff}")
        except Exception as e:
            return (False, f"ERROR: {e}")

    # ----------------------------------------------------------------------
    # Monotonicity tests – explicitly perturb input and compare outputs
    # ----------------------------------------------------------------------
    def test_monotonicity_B_increasing_with_r(self):
        """
        Perturb r upward by 10% and confirm B increases.
        Baseline: r_base; perturbed: r_pert = 1.1 * r_base.
        """
        if self.B_mag is None:
            return (False, "SKIP: B_mag not available")
        try:
            # Build substitution dictionary with plausible values (use givens if available)
            subs = {}
            # Use None-checked values; if missing, use reasonable SI defaults
            subs[self._sym_mu0] = float(self.mu0) if self.mu0 is not None else 4*math.pi*1e-7
            subs[self._sym_eps0] = float(self.epsilon0) if self.epsilon0 is not None else 8.854187817e-12
            subs[self._sym_V0] = float(self.V0) if self.V0 is not None else 1.0
            subs[self._sym_omega] = float(self.omega) if self.omega is not None else 2*math.pi*60.0
            subs[self._sym_d] = float(self.d) if self.d is not None else 1e-3
            subs[self._sym_t] = 0.0   # cos(0)=1, peak field

            # Baseline r
            r_base = float(self.r) if self.r is not None else 0.01
            r_pert = r_base * 1.1

            subs_base = {**subs, self._sym_r: r_base}
            subs_pert = {**subs, self._sym_r: r_pert}

            B_base = float(self.B_mag.subs(subs_base))
            B_pert = float(self.B_mag.subs(subs_pert))

            if B_pert > B_base:
                return (True, f"B increases with r: B({r_base})={B_base:.6e}, B({r_pert})={B_pert:.6e}")
            else:
                return (False, f"Monotonicity violated: B({r_base})={B_base:.6e}, B({r_pert})={B_pert:.6e}")
        except Exception as e:
            return (False, f"ERROR: {e}")

    def test_monotonicity_B_increasing_with_omega(self):
        """
        Perturb ω upward by 10% and confirm B increases.
        Baseline: omega_base; perturbed: omega_pert = 1.1 * omega_base.
        """
        if self.B_mag is None:
            return (False, "SKIP: B_mag not available")
        try:
            subs = {}
            subs[self._sym_mu0] = float(self.mu0) if self.mu0 is not None else 4*math.pi*1e-7
            subs[self._sym_eps0] = float(self.epsilon0) if self.epsilon0 is not None else 8.854187817e-12
            subs[self._sym_V0] = float(self.V0) if self.V0 is not None else 1.0
            subs[self._sym_d] = float(self.d) if self.d is not None else 1e-3
            subs[self._sym_r] = float(self.r) if self.r is not None else 0.01
            subs[self._sym_t] = 0.0

            omega_base = float(self.omega) if self.omega is not None else 2*math.pi*60.0
            omega_pert = omega_base * 1.1

            subs_base = {**subs, self._sym_omega: omega_base}
            subs_pert = {**subs, self._sym_omega: omega_pert}

            B_base = float(self.B_mag.subs(subs_base))
            B_pert = float(self.B_mag.subs(subs_pert))

            if B_pert > B_base:
                return (True, f"B increases with ω: B(ω={omega_base:.2f})={B_base:.6e}, B(ω={omega_pert:.2f})={B_pert:.6e}")
            else:
                return (False, f"Monotonicity violated: B(ω={omega_base:.2f})={B_base:.6e}, B(ω={omega_pert:.2f})={B_pert:.6e}")
        except Exception as e:
            return (False, f"ERROR: {e}")

    # ----------------------------------------------------------------------
    # Modeling/trace tests – check specific physical relations
    # ----------------------------------------------------------------------
    def test_modeling_B_linear_in_r(self):
        """
        The magnetic field should be proportional to r.  Check that ∂B/∂r
        (symbolic derivative) is independent of r.
        """
        if self.B_mag is None:
            return (False, "SKIP: B_mag not available")
        try:
            dBr = sp.simplify(sp.diff(self.B_mag, self._sym_r))
            if dBr.has(self._sym_r):
                return (False, f"∂B/∂r still depends on r: {dBr}")
            # Also check it is positive (but monotonicity already covers direction)
            return (True, f"∂B/∂r = {dBr}, independent of r – linear dependence confirmed")
        except Exception as e:
            return (False, f"ERROR: {e}")

    def test_modeling_temporal_dependence(self):
        """
        The magnetic field should have the form cos(ωt).  Check that the
        expression contains a cos(ωt) factor.
        """
        if self.B_mag is None:
            return (False, "SKIP: B_mag not available")
        try:
            factor = sp.factor(self.B_mag)
            if not factor.has(sp.cos(self._sym_omega * self._sym_t)):
                return (False, f"B_mag does not contain cos(ω t) factor: {factor}")
            return (True, "B_mag temporal dependence is cos(ω t)")
        except Exception as e:
            return (False, f"ERROR: {e}")

    # ----------------------------------------------------------------------
    # Weak answer-sanity check – only sign and finiteness
    # ----------------------------------------------------------------------
    def test_answer_sanity_positive(self):
        """
        For a typical set of positive parameters (no current, no charge),
        both B and emf should be positive at t=0 (cos=1).  Weak check:
        just verify positivity and finiteness.
        """
        if self.B_mag is None or self.emf_peak is None:
            return (False, "SKIP: need both B_mag and emf_peak")
        try:
            subs = {}
            subs[self._sym_mu0] = float(self.mu0) if self.mu0 is not None else 4*math.pi*1e-7
            subs[self._sym_eps0] = float(self.epsilon0) if self.epsilon0 is not None else 8.854187817e-12
            subs[self._sym_V0] = float(self.V0) if self.V0 is not None else 1.0
            subs[self._sym_omega] = float(self.omega) if self.omega is not None else 2*math.pi*60.0
            subs[self._sym_d] = float(self.d) if self.d is not None else 1e-3
            subs[self._sym_r] = float(self.r) if self.r is not None else 0.01
            subs[self._sym_t] = 0.0
            A_val = float(self.loop_area) if self.loop_area is not None else 1e-4
            subs[self._sym_A] = A_val

            B_val = float(self.B_mag.subs(subs))
            emf_val = float(self.emf_peak.subs(subs))
            if B_val > 0 and emf_val > 0:
                return (True, f"Both B and emf are positive: B={B_val:.6e}, emf={emf_val:.6e}")
            else:
                return (False, f"Non-positive values: B={B_val:.6e}, emf={emf_val:.6e}")
        except Exception as e:
            return (False, f"ERROR: {e}")

    # ----------------------------------------------------------------------
    # Main runner
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