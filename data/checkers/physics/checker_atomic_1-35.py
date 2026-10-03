import sympy as sp
import math

class PhysicsChecker:
    """
    Audits a solution for the hydrogen atom energy derivation and sodium
    doublet explanation.  Checks physics invariants, modelling quality and
    interface requirements.
    """

    def __init__(self, solution_instance):
        # ---------- extract results and explanation ----------
        results = getattr(solution_instance, 'results', {})
        self.E_n = results.get('E_n', None)               # expected: string expression
        self.doublet_cause = results.get('doublet_cause', None)  # expected: string
        self.explanation_steps = getattr(solution_instance,
                                         'explanation_steps', None)

        # ---------- symbolic constants for internal checks ----------
        self.m, self.e, self.eps0, self.hbar, self.n = sp.symbols(
            'm e eps0 hbar n', positive=True)
        self.k = 1 / (4 * sp.pi * self.eps0)

        # standard Bohr energy (used for exact matching)
        self.bohr_energy = -sp.Rational(1, 2) * self.m * self.e**4 / \
            ((4 * sp.pi * self.eps0)**2 * self.hbar**2 * self.n**2)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_E_n_exists(self):
        if self.E_n is None:
            return (False, "Key 'E_n' missing or None")
        if not isinstance(self.E_n, str):
            return (False, "'E_n' should be a string")
        return (True, "E_n present and is string")

    def test_interface_doublet_cause_exists(self):
        if self.doublet_cause is None:
            return (False, "Key 'doublet_cause' missing or None")
        if not isinstance(self.doublet_cause, str):
            return (False, "'doublet_cause' should be a string")
        return (True, "doublet_cause present and is string")

    # ------------------------------------------------------------------
    # Physics invariants (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_energy_negative(self):
        """Bound state energy must be negative for any positive n."""
        if self.E_n is None:
            return (False, "SKIP: E_n not available")
        try:
            expr = sp.sympify(self.E_n)
        except Exception as e:
            return (False, f"Cannot parse E_n: {e}")

        # substitute positive numeric dummy values for all constants
        free = expr.free_symbols - {self.n}
        sub = {s: 1.0 for s in free}
        sub[self.n] = 2          # any n>0
        try:
            val = expr.subs(sub).evalf()
        except Exception:
            return (False, "Cannot evaluate expression with numeric substitutions")
        if val >= 0:
            return (False, f"Energy must be negative for bound states, got {val}")
        return (True, "Energy negative for typical inputs")

    def test_invariant_energy_scales_as_1_over_n2(self):
        """E(n) ~ constant / n^2  →  ratio E(1) / E(2) = 4."""
        if self.E_n is None:
            return (False, "SKIP: E_n not available")
        try:
            expr = sp.sympify(self.E_n)
        except Exception as e:
            return (False, f"Cannot parse E_n: {e}")

        try:
            r = sp.simplify(expr.subs({self.n: 1}) / expr.subs({self.n: 2}))
        except Exception as e:
            return (False, f"Ratio computation failed: {e}")
        if not r.equals(sp.Integer(4)):
            return (False, f"Energy ratio E(1)/E(2) should be 4, got {r}")
        return (True, "Energy scales as 1/n² (ratio = 4)")

    def test_invariant_energy_contains_required_constants(self):
        """Check that the expression contains e⁴, ħ⁻², ε₀⁻² and m."""
        if self.E_n is None:
            return (False, "SKIP: E_n not available")
        try:
            expr = sp.sympify(self.E_n)
        except Exception as e:
            return (False, f"Cannot parse E_n: {e}")

        # first try exact match with the Bohr formula
        diff = sp.simplify(expr - self.bohr_energy)
        if diff == 0:
            return (True, "Expression matches the standard Bohr energy exactly")

        # fallback: check essential factors are present
        if not expr.has(self.e**4):
            return (False, "Expression should contain e⁴ factor")
        if not expr.has(1 / self.hbar**2) and not expr.has(self.hbar**(-2)):
            return (False, "Expression should contain ħ⁻² factor")
        if not expr.has(1 / self.eps0**2) and not expr.has(self.eps0**(-2)):
            return (False, "Expression should contain ε₀⁻² factor")
        if not expr.has(self.m):
            return (False, "Expression should contain electron mass m")
        return (True, "Expression contains required constants")

    # ------------------------------------------------------------------
    # Modelling / explanation quality checks
    # ------------------------------------------------------------------
    def test_modeling_doublet_cause_contains_spin_orbit(self):
        """The explanation must mention the coupling of orbital and spin."""
        if self.doublet_cause is None:
            return (False, "SKIP: doublet_cause not available")
        cause = self.doublet_cause.lower()
        if 'spin' in cause and 'orbit' in cause:
            return (True, "doublet_cause mentions 'spin' and 'orbit'")
        if 'spin-orbit' in cause or 'spin orbit' in cause:
            return (True, "doublet_cause mentions spin-orbit coupling")
        return (False, f"doublet_cause does not contain required terms: "
                       f"{self.doublet_cause[:100]}...")

    def test_modeling_derivation_mentions_coulomb_centripetal(self):
        """
        The explanation_steps should mention both Coulomb's law and
        centripetal force (or force balance) for the derivation of E_n.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation_steps to check")
        text = ' '.join(self.explanation_steps).lower()
        has_coulomb = ('coulomb' in text or 'coulomb\'s' in text or
                       'force' in text and 'charge' in text)
        has_centripetal = ('centripetal' in text or 'force balance' in text or
                           'm v^2 / r' in text or 'circular orbit' in text)
        if has_coulomb and has_centripetal:
            return (True, "Explanation mentions both Coulomb force and centripetal condition")
        else:
            missing = []
            if not has_coulomb:
                missing.append("Coulomb's law")
            if not has_centripetal:
                missing.append("centripetal/force balance")
            return (False, f"Missing: {', '.join(missing)}")

    def test_invariant_doublet_splitting_requires_l_gt_0(self):
        """The derivation should state that l>0 (or l=0 gives no splitting)."""
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation_steps to check")
        text = ' '.join(self.explanation_steps).lower()
        # look for explicit l>0 or mention that s-states do not split
        if ('l>0' in text or 'l > 0' in text or
            'l greater than 0' in text or 'non-zero orbital' in text):
            return (True, "Explanation explicitly mentions l>0 condition")
        if ('s-state' in text and 'no splitting' in text):
            return (True, "Explanation implies l=0 (s-state) does not split")
        return (False, "No mention of condition l>0 (or s-state single) in "
                       "explanation_steps")

    # ------------------------------------------------------------------
    # Monotonicity (trend) test
    # ------------------------------------------------------------------
    def test_monotonicity_energy_increases_with_n(self):
        """
        Bound state energy increases (becomes less negative) as n grows.
        Perturb n by +1: compare E(2) and E(3).
        """
        if self.E_n is None:
            return (False, "SKIP: E_n not available")
        try:
            expr = sp.sympify(self.E_n)
        except Exception as e:
            return (False, f"Cannot parse E_n: {e}")

        # baseline n=2, perturbed n=3
        baseline = expr.subs({self.n: 2}).evalf(subs={})  # symbolic, will need numeric constants
        perturbed = expr.subs({self.n: 3}).evalf(subs={})
        free = baseline.free_symbols
        sub_vals = {s: 1.0 for s in free}
        try:
            base_val = baseline.subs(sub_vals).evalf()
            pert_val = perturbed.subs(sub_vals).evalf()
        except Exception as e:
            return (False, f"Cannot evaluate: {e}")
        # Bound state energy increases (less negative) with n
        if pert_val <= base_val:
            return (False, f"Expected E(3) > E(2), got E(2)={base_val}, E(3)={pert_val}")
        return (True, "Energy increases (less negative) when n is increased by 1")

    # ------------------------------------------------------------------
    # Orchestrator
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)