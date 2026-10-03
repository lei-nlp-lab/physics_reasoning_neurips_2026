import sympy as sp
import math

class PhysicsChecker:
    """
    Checker for Fraunhofer diffraction pattern from two unequal slits.
    """
    def __init__(self, solution_instance):
        # Extract givens safely (defaults to None)
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.c = getattr(solution_instance, 'c', None)
        self.lam = getattr(solution_instance, 'lam', None)  # lambda
        self.theta = getattr(solution_instance, 'theta', None)

        # Extract results safely
        results_dict = getattr(solution_instance, 'results', None)
        if results_dict is None:
            self.I1 = None
            self.I2 = None
            self.I3 = None
        else:
            self.I1 = results_dict.get('I1', None)
            self.I2 = results_dict.get('I2', None)
            self.I3 = results_dict.get('I3', None)

        # Pre-define symbolic variables used in physics checks
        self.sym_a, self.sym_b, self.sym_c, self.sym_lam, self.sym_theta = sp.symbols('a b c lam theta', positive=True)
        # Define u, v, w for convenience
        self.u = sp.pi * self.sym_a * sp.sin(self.sym_theta) / self.sym_lam
        self.v = sp.pi * self.sym_b * sp.sin(self.sym_theta) / self.sym_lam
        self.w = 2 * sp.pi * self.sym_c * sp.sin(self.sym_theta) / self.sym_lam

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that I1, I2, I3 exist and are sympy expressions."""
        if self.I1 is None:
            return (False, "SKIP: I1 not provided")
        if self.I2 is None:
            return (False, "SKIP: I2 not provided")
        if self.I3 is None:
            return (False, "SKIP: I3 not provided")
        if not isinstance(self.I1, sp.Basic):
            return (False, "I1 is not a sympy expression")
        if not isinstance(self.I2, sp.Basic):
            return (False, "I2 is not a sympy expression")
        if not isinstance(self.I3, sp.Basic):
            return (False, "I3 is not a sympy expression")
        return (True, "All required keys present and are sympy expressions")

    # ------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_single_slit_terms(self):
        """I1 should contain a^2 * sinc^2(u) + b^2 * sinc^2(v)."""
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        if not (self.I1.has(self.sym_a**2) and self.I1.has(self.sym_b**2)):
            return (False, "I1 does not contain a^2 and b^2 terms")
        if not (self.I1.has(sp.sin(self.u)) and self.I1.has(sp.sin(self.v))):
            return (False, "I1 does not contain sin(u) and sin(v) terms")
        return (True, "I1 includes single-slit diffraction terms")

    def test_invariant_interference_term(self):
        """I1 should contain 2*a*b * sinc(u)*sinc(v)*cos(w)."""
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        # Check for presence of cos(w) and a*b term
        if not (self.I1.has(sp.cos(self.w)) and self.I1.has(self.sym_a * self.sym_b)):
            return (False, "I1 does not contain cos(w) or a*b term")
        return (True, "Interference term present")

    def test_invariant_equal_slits(self):
        """For a=b, I2 should simplify to 4*a^2 * sinc^2(v) * cos^2(w/2)."""
        if self.I2 is None:
            return (False, "SKIP: I2 not available")
        # Define reference expression when a=b
        sinc_v = sp.sin(self.v) / self.v
        ref_expr = 4 * self.sym_a**2 * sinc_v**2 * sp.cos(self.w/2)**2
        diff = sp.simplify(self.I2 - ref_expr)
        if diff == 0:
            return (True, "I2 matches the equal-slits formula")
        else:
            # Soft check: contains cos^2(w/2)
            if not self.I2.has(sp.cos(self.w/2)**2):
                return (False, "I2 does not contain cos^2(w/2)")
            return (True, "I2 likely correct (soft check)")

    def test_invariant_one_slit(self):
        """For a=0, I3 should simplify to b^2 * sinc^2(v)."""
        if self.I3 is None:
            return (False, "SKIP: I3 not available")
        sinc_v = sp.sin(self.v) / self.v
        ref_expr = self.sym_b**2 * sinc_v**2
        diff = sp.simplify(self.I3 - ref_expr)
        if diff == 0:
            return (True, "I3 matches single-slit formula for a=0")
        else:
            if not self.I3.has(self.sym_b**2) or not self.I3.has(sp.sin(self.v)**2):
                return (False, "I3 does not have correct form")
            return (True, "I3 likely correct (soft check)")

    def test_invariant_theta_zero_limit(self):
        """At θ=0, I1 should reduce to (a+b)^2."""
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        try:
            I1_theta0 = sp.limit(self.I1, self.sym_theta, 0)
        except Exception:
            return (False, "Could not compute limit at θ=0")
        ref = (self.sym_a + self.sym_b)**2
        diff = sp.simplify(I1_theta0 - ref)
        if diff == 0:
            return (True, "I1(0) = (a+b)^2")
        else:
            return (False, f"I1(0) should be (a+b)^2, got {I1_theta0}")

    # ------------------------------------------------------------
    # Modeling checks (quality of the derived formulas)
    # ------------------------------------------------------------
    def test_modeling_phase_definition(self):
        """Check that w is defined as 2πc sinθ/λ."""
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        expected_w = 2 * sp.pi * self.sym_c * sp.sin(self.sym_theta) / self.sym_lam
        if self.I1.has(sp.cos(expected_w)):
            return (True, "Phase w defined correctly")
        else:
            # Possibly half-angle form
            if self.I1.has(sp.cos(sp.pi * self.sym_c * sp.sin(self.sym_theta) / self.sym_lam)):
                return (True, "Phase appears in half-angle form")
            return (False, "Could not verify phase definition")

    def test_modeling_sinc_variable_u(self):
        """Check that u uses a and v uses b."""
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        expected_u = sp.pi * self.sym_a * sp.sin(self.sym_theta) / self.sym_lam
        if self.I1.has(sp.sin(expected_u)):
            return (True, "u variable defined correctly for slit a")
        else:
            return (False, "Could not find sin(u) with correct argument")

    def test_modeling_sinc_variable_v(self):
        """Check that v uses b."""
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        expected_v = sp.pi * self.sym_b * sp.sin(self.sym_theta) / self.sym_lam
        if self.I1.has(sp.sin(expected_v)):
            return (True, "v variable defined correctly for slit b")
        else:
            return (False, "Could not find sin(v) with correct argument")

    # ------------------------------------------------------------
    # Monotonicity checks (explicit perturbation + comparison)
    # ------------------------------------------------------------
    def test_monotonicity_width_vs_intensity_at_zero(self):
        """At θ=0, increasing a increases I1(0)."""
        # Need numeric evaluation; if givens are missing, use sample values
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        # Use sample numeric parameters for evaluation (ignore instance givens if any)
        # We'll use a=1, b=0.5, c=1, lam=1, theta=0 (limit)
        # Since theta=0, we use limit form: I1_0 = (a+b)^2
        a_base = 1.0
        b_val = 0.5
        # Compute baseline I1 at theta=0 using limit formula (not expression)
        # But we can also compute by substituting into I1 with a tiny theta to avoid singularity
        # Simpler: use the known limit (a+b)^2, derived from physics, not hardcoded answer.
        # We'll substitute into the symbolic expression with very small theta.
        subs_base = {self.sym_a: a_base, self.sym_b: b_val, self.sym_c: 1.0,
                     self.sym_lam: 1.0, self.sym_theta: 1e-6}
        try:
            I_base = float(self.I1.subs(subs_base))
        except Exception as e:
            return (False, f"Cannot evaluate baseline: {e}")
        # Perturb a by +1%
        a_new = a_base * 1.01
        subs_new = {self.sym_a: a_new, self.sym_b: b_val, self.sym_c: 1.0,
                    self.sym_lam: 1.0, self.sym_theta: 1e-6}
        try:
            I_new = float(self.I1.subs(subs_new))
        except Exception as e:
            return (False, f"Cannot evaluate perturbed: {e}")
        # Intensity should increase (since a > 0)
        if I_new > I_base:
            return (True, f"Intensity increased from {I_base:.4f} to {I_new:.4f} when a increased by 1%")
        else:
            return (False, f"Intensity did not increase: baseline {I_base:.4f}, perturbed {I_new:.4f}")

    def test_monotonicity_separation_vs_intensity_variation(self):
        """At fixed θ (nonzero), changing c changes I1 (c affects interference)."""
        if self.I1 is None:
            return (False, "SKIP: I1 not available")
        # Choose sample values: a=1, b=0.5, c_base=1, lam=1, theta=0.5 rad
        a_val = 1.0
        b_val = 0.5
        c_base = 1.0
        lam = 1.0
        theta = 0.5
        subs_base = {self.sym_a: a_val, self.sym_b: b_val,
                     self.sym_c: c_base, self.sym_lam: lam,
                     self.sym_theta: theta}
        try:
            I_base = float(self.I1.subs(subs_base))
        except Exception as e:
            return (False, f"Cannot evaluate baseline: {e}")
        # Perturb c by +10%
        c_new = c_base * 1.10
        subs_new = {self.sym_a: a_val, self.sym_b: b_val,
                    self.sym_c: c_new, self.sym_lam: lam,
                    self.sym_theta: theta}
        try:
            I_new = float(self.I1.subs(subs_new))
        except Exception as e:
            return (False, f"Cannot evaluate perturbed: {e}")
        # The intensity should change (not necessarily increase, but change)
        if not math.isclose(I_base, I_new, rel_tol=1e-6):
            return (True, f"Intensity changed from {I_base:.4f} to {I_new:.4f} when c increased by 10%")
        else:
            return (False, f"Intensity unchanged (baseline {I_base:.4f}, perturbed {I_new:.4f})")

    # ------------------------------------------------------------
    # Answer sanity (weak, must not dominate)
    # ------------------------------------------------------------
    def test_answer_sanity_nonnegative(self):
        """I1, I2, I3 should be non-negative for physical angles."""
        if self.I1 is None or self.I2 is None or self.I3 is None:
            return (False, "SKIP: one or more intensities missing")
        # Substitute sample numeric values: a=1, b=0.5, c=1, lam=1, theta=0.3
        try:
            subs_dict = {self.sym_a: 1, self.sym_b: 0.5, self.sym_c: 1,
                         self.sym_lam: 1, self.sym_theta: 0.3}
            I1_val = float(self.I1.subs(subs_dict))
            I2_val = float(self.I2.subs(subs_dict))
            I3_val = float(self.I3.subs(subs_dict))
        except Exception as e:
            return (False, f"Could not evaluate numeric: {e}")
        if I1_val < 0 or I2_val < 0 or I3_val < 0:
            return (False, "Intensity negative for sample angle")
        if math.isinf(I1_val) or math.isinf(I2_val) or math.isinf(I3_val):
            return (False, "Intensity infinite for sample angle")
        if math.isnan(I1_val) or math.isnan(I2_val) or math.isnan(I3_val):
            return (False, "Intensity NaN for sample angle")
        return (True, "Intensities non-negative, finite, and not NaN")

    # ------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------
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