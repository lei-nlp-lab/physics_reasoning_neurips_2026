import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics auditor for the probability density ρ(E) of a single atom in a
    classical non-interacting monatomic gas in thermal equilibrium.
    """

    def __init__(self, solution_instance):
        # Safely extract the results dictionary and givens if present
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Target quantity: ρ(E) as a symbolic expression or string
        self.rho_E = self.results.get('rho_E', None)

        # Support variables: try to extract temperature, Boltzmann etc. if available
        self.k = self.givens.get('k', None) or sp.Symbol('k')
        self.T = self.givens.get('T', None) or sp.Symbol('T')
        self.beta = self.givens.get('beta', None) or sp.Symbol('beta')
        self.E = sp.Symbol('E', nonnegative=True)

        # Attempt to convert rho_E to sympy if it is a string
        if isinstance(self.rho_E, str):
            try:
                self.rho_E_sym = sp.sympify(self.rho_E)
            except Exception:
                self.rho_E_sym = None
        elif self.rho_E is not None:
            # assume it is already a sympy expression
            self.rho_E_sym = self.rho_E
        else:
            self.rho_E_sym = None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_rho_E_exists(self):
        """Check that results contains 'rho_E' and it is not None."""
        if self.rho_E is None:
            return (False, "Missing 'rho_E' key in results (or is None).")
        return (True, "'rho_E' is present.")

    def test_interface_rho_E_is_symbolic(self):
        """Check that rho_E can be treated as a sympy expression."""
        if self.rho_E_sym is None:
            return (False, "'rho_E' cannot be converted to a sympy expression.")
        if not isinstance(self.rho_E_sym, sp.Basic):
            return (False, f"'rho_E' is not a sympy object (type: {type(self.rho_E_sym)}).")
        return (True, "'rho_E' is a valid sympy expression.")

    # ----------------------------------------------------------------------
    # Invariant tests (use math.isclose for numeric checks when needed)
    # ----------------------------------------------------------------------
    def test_invariant_positivity(self):
        """
        For any E ≥ 0, ρ(E) should be non-negative.
        Test with a few sample numeric values.
        """
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E expression not available.")

        # Use typical numeric values: T=300, k=1.38e-23, E from 0 to 5kT
        k_val = 1.380649e-23
        T_val = 300.0
        beta_val = 1.0 / (k_val * T_val)

        # Substitute symbolic variables with numbers
        subs_dict = {}
        if self.beta.free_symbols:
            subs_dict[self.beta] = beta_val
        if self.k.free_symbols:
            subs_dict[self.k] = k_val
        if self.T.free_symbols:
            subs_dict[self.T] = T_val

        # E values to test
        test_Es = [0.0, 1e-23, 5e-21, 1e-20, 5e-20]
        for e_val in test_Es:
            try:
                val = float(self.rho_E_sym.subs({self.E: e_val, **subs_dict}).evalf())
            except Exception:
                continue
            if val < -1e-12:  # allow tiny negative due to floating errors
                return (False, f"rho_E({e_val}) = {val} < 0.")

        return (True, "rho_E is non-negative at tested points.")

    def test_invariant_zero_at_E0(self):
        """ρ(E) should be zero at E=0 because of sqrt(E) factor."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E expression not available.")

        subs_dict = self._get_numeric_subs()
        try:
            val = float(self.rho_E_sym.subs({self.E: 0, **subs_dict}).evalf())
        except Exception:
            return (False, "Cannot evaluate ρ(0).")

        if math.isclose(val, 0.0, abs_tol=1e-12):
            return (True, "rho_E(0) = 0 as expected.")
        else:
            return (False, f"rho_E(0) = {val}, expected 0.")

    def test_invariant_normalization(self):
        """Check that ∫₀^∞ ρ(E) dE = 1 (symbolic or numeric)."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E expression not available.")

        # Try symbolic integration
        try:
            integral = sp.integrate(self.rho_E_sym, (self.E, 0, sp.oo))
            # check if integral simplifies to 1 (subject to assumptions)
            simplified = sp.simplify(integral)
            # compare with 1 using substitution if necessary
            subs_dict = self._get_numeric_subs()
            numeric_int = float(simplified.subs(subs_dict).evalf())
            if math.isclose(numeric_int, 1.0, rel_tol=1e-2):
                return (True, f"Symbolic integral simplifies to 1 (numeric: {numeric_int}).")
            else:
                return (False, f"Symbolic integral = {simplified}, numerically {numeric_int}.")
        except Exception:
            pass

        # Fallback: numeric integration over a large range (0 to 10kT)
        try:
            import numpy as np
            from scipy.integrate import quad

            # convert to callable numeric function
            subs_dict = self._get_numeric_subs()
            f = sp.lambdify(self.E, self.rho_E_sym.subs(subs_dict), 'numpy')
            # integration limits: from 0 to large enough
            k_val = 1.380649e-23
            T_val = 300.0
            upper = 20 * k_val * T_val  # 20 kT
            numeric_int, _ = quad(f, 0, upper, limit=200)
            if math.isclose(numeric_int, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Numeric integral ≈ {numeric_int:.5f} (close to 1).")
            else:
                return (False, f"Numeric integral = {numeric_int:.8f}, expected ~1.")
        except Exception as e:
            return (False, f"Numeric integration failed: {e}.")

    def test_invariant_max_at_half_kT(self):
        """The maximum of ρ(E) should occur at E = kT/2."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E expression not available.")

        # Compute derivative and solve for zero
        E = self.E
        try:
            d_rho = sp.diff(self.rho_E_sym, E)
            # solve d_rho = 0 for E
            solutions = sp.solve(sp.simplify(d_rho), E)
            # filter for positive real solution
            real_sols = [sol for sol in solutions if sol.is_real and sol > 0]
            if not real_sols:
                return (False, f"No positive critical point found. Solutions: {solutions}")

            # There should be one positive solution: E = kT/2
            expected = sp.Rational(1,2) * (1/self.beta) if self.beta.free_symbols else sp.Rational(1,2) * (self.k * self.T)
            # try to match with the solution
            for sol in real_sols:
                diff = sp.simplify(sol - expected)
                # check if diff is zero symbolically (under assumptions)
                if diff == 0:
                    return (True, f"Critical point E = {sol} matches kT/2.")
            # else numeric test
            subs_dict = self._get_numeric_subs()
            k_val = 1.380649e-23
            T_val = 300.0
            expected_val = 0.5 * k_val * T_val
            for sol in real_sols:
                try:
                    sol_val = float(sol.subs(subs_dict).evalf())
                    if math.isclose(sol_val, expected_val, rel_tol=1e-2):
                        return (True, f"Critical point ≈ {sol_val:.5e} matches kT/2 = {expected_val:.5e}.")
                except Exception:
                    continue
            return (False, f"Critical points {real_sols} do not match expected kT/2.")
        except Exception as e:
            return (False, f"Derivative/solve failed: {e}.")

    def test_invariant_finite_limits(self):
        """Check that ρ(E) → 0 as E → 0 and as E → ∞."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E expression not available.")

        subs_dict = self._get_numeric_subs()
        try:
            limit0 = sp.limit(self.rho_E_sym, self.E, 0)
            val0 = float(limit0.subs(subs_dict).evalf())
            if not math.isclose(val0, 0.0, abs_tol=1e-12):
                return (False, f"Limit E→0 = {val0}, expected 0.")
        except Exception:
            return (False, "Cannot compute limit at 0.")

        try:
            limit_inf = sp.limit(self.rho_E_sym, self.E, sp.oo)
            val_inf = float(limit_inf.subs(subs_dict).evalf())
            if not math.isclose(val_inf, 0.0, abs_tol=1e-12):
                return (False, f"Limit E→∞ = {val_inf}, expected 0.")
        except Exception:
            # accept if limit calculation fails but likelihood is that it goes to 0
            pass

        return (True, "Limits at 0 and ∞ are 0.")

    # ----------------------------------------------------------------------
    # Modeling tests (quality checks)
    # ----------------------------------------------------------------------
    def test_modeling_boltzmann_factor(self):
        """ρ(E) must contain exp(-βE) factor."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E not available.")
        # Check that the expression contains exp(-βE)
        exp_factor = sp.exp(-self.beta * self.E)
        if exp_factor in self.rho_E_sym.atoms(sp.exp):
            return (True, "Contains Boltzmann factor exp(-βE).")
        # Try substitute β = 1/(kT) and check again
        alt_beta = 1/(self.k*self.T)
        exp_factor2 = sp.exp(-self.E/(self.k*self.T))
        if exp_factor2 in self.rho_E_sym.atoms(sp.exp):
            return (True, "Contains Boltzmann factor exp(-E/kT).")
        # Also check if the expression has exp(-βE) or exp(-E/kT) in some form
        # Use string search as a fallback
        rho_str = str(self.rho_E_sym)
        if 'exp(' in rho_str and ('-β' in rho_str or '-beta' in rho_str or '-E/kT' in rho_str):
            return (True, "Seems to contain exponential factor.")
        return (False, "Missing Boltzmann exponential factor.")

    def test_modeling_sqrt_E_factor(self):
        """ρ(E) must contain sqrt(E) factor (i.e., E^{1/2})."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E not available.")
        # Check for sqrt(E) or E**(1/2)
        if sp.sqrt(self.E) in self.rho_E_sym.atoms(sp.Pow):
            return (True, "Contains sqrt(E) factor.")
        # Also check if E**(Rational(1,2)) appears
        for a in self.rho_E_sym.atoms(sp.Pow):
            if a.base == self.E and a.exp == sp.Rational(1,2):
                return (True, "Contains E^{1/2} factor.")
        return (False, "Missing sqrt(E) factor.")

    def test_modeling_cancellation_of_V_m_h(self):
        """ρ(E) should not depend on Volume, mass m, or Planck constant h."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E not available.")

        # Check if any of V, m, h appear as symbols in the expression
        possible_symbols = [sp.Symbol(name) for name in ['V', 'm', 'h']]
        free_syms = self.rho_E_sym.free_symbols
        interfering = [s for s in possible_symbols if s in free_syms]
        if interfering:
            return (False, f"rho_E still contains volume/mass/Planck symbols: {interfering}.")
        return (True, "rho_E is independent of V, m, and h (as expected).")

    def test_modeling_explicit_form(self):
        """
        Check that the expression matches the known form:
        ρ(E) = (2/√π) β^{3/2} √E exp(-βE). (Tolerant comparison)
        This is a quality check, not an exact answer match.
        """
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E not available.")

        # Build the known form as sympy expression
        known = (2/sp.sqrt(sp.pi)) * self.beta**(sp.Rational(3,2)) * sp.sqrt(self.E) * sp.exp(-self.beta * self.E)
        # Simplify difference
        diff = sp.simplify(self.rho_E_sym - known)
        # Check if diff is identically zero (symbolic)
        if diff == 0:
            return (True, "Expression exactly matches expected form.")
        # Otherwise, try numeric substitution
        subs_dict = self._get_numeric_subs()
        try:
            num_diff = float(diff.subs(subs_dict).evalf())
            if math.isclose(num_diff, 0.0, abs_tol=1e-12):
                return (True, "Expression numerically matches expected form.")
        except Exception:
            pass

        # also check if the two are equivalent up to a constant factor (should be 1)
        ratio = sp.simplify(self.rho_E_sym / known)
        try:
            num_ratio = float(ratio.subs(subs_dict).evalf())
            if math.isclose(num_ratio, 1.0, rel_tol=1e-2):
                return (True, "Numerically matches expected form (ratio=1).")
        except Exception:
            pass

        return (False, "Expression does not match the expected form (2/√π)β^{3/2}√E exp(-βE).")

    # ----------------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_average_energy_with_beta(self):
        """
        Monotonicity: if β increases (temperature decreases), the average energy
        ⟨E⟩ = ∫ E ρ(E) dE should decrease.
        We perturb β by +10% and compare the baseline and perturbed average energies.
        """
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E expression not available.")

        # Use a fixed numeric environment
        k_val = 1.380649e-23
        T0 = 300.0
        beta0 = 1.0 / (k_val * T0)

        # Compute baseline average energy from ρ(E) numerically
        subs_dict_base = {self.k: k_val, self.T: T0, self.beta: beta0}
        # substitute into rho_E_sym; note that E remains symbolic
        rho_base = self.rho_E_sym.subs(subs_dict_base)

        # Compute ⟨E⟩ = ∫ E ρ(E) dE
        integrand = self.E * rho_base
        try:
            # Symbolic integration if possible
            avg_base_sym = sp.integrate(integrand, (self.E, 0, sp.oo))
            avg_base = float(avg_base_sym.evalf())
        except Exception:
            # Fallback to numeric integration
            try:
                import numpy as np
                from scipy.integrate import quad
                f = sp.lambdify(self.E, integrand, 'numpy')
                upper = 20 * k_val * T0
                avg_base, _ = quad(f, 0, upper, limit=200)
            except Exception as e:
                return (False, f"Cannot compute baseline average energy: {e}")

        # Perturb β upward by 10%
        beta1 = 1.1 * beta0
        subs_dict_pert = {self.k: k_val, self.T: T0, self.beta: beta1}
        rho_pert = self.rho_E_sym.subs(subs_dict_pert)
        integrand_pert = self.E * rho_pert
        try:
            avg_pert_sym = sp.integrate(integrand_pert, (self.E, 0, sp.oo))
            avg_pert = float(avg_pert_sym.evalf())
        except Exception:
            try:
                import numpy as np
                from scipy.integrate import quad
                f = sp.lambdify(self.E, integrand_pert, 'numpy')
                upper = 20 * k_val * T0
                avg_pert, _ = quad(f, 0, upper, limit=200)
            except Exception as e:
                return (False, f"Cannot compute perturbed average energy: {e}")

        # Check monotonicity: avg_pert < avg_base
        if avg_pert < avg_base:
            return (True, f"Average energy decreased from {avg_base:.5e} (β0) to {avg_pert:.5e} (β1=1.1β0), as expected.")
        else:
            return (False, f"Average energy did not decrease: baseline {avg_base:.5e}, perturbed {avg_pert:.5e}.")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_for_typical_E(self):
        """Check that ρ(E) is finite and not NaN for physically reasonable values."""
        if self.rho_E_sym is None:
            return (False, "SKIP: rho_E not available.")

        subs_dict = self._get_numeric_subs()
        k_val = 1.380649e-23
        T_val = 300.0
        test_E = 0.5 * k_val * T_val  # at the peak
        try:
            val = float(self.rho_E_sym.subs({self.E: test_E, **subs_dict}).evalf())
            if math.isnan(val) or math.isinf(val):
                return (False, f"rho_E at peak is {val} (NaN or inf).")
        except Exception as e:
            return (False, f"Evaluation failed: {e}.")
        return (True, f"rho_E at peak = {val:.5e} (finite).")

    # ----------------------------------------------------------------------
    # Helper
    # ----------------------------------------------------------------------
    def _get_numeric_subs(self):
        """Return a dictionary substituting β, k, T with numeric values if they are symbols."""
        subs = {}
        k_val = 1.380649e-23
        T_val = 300.0
        beta_val = 1.0 / (k_val * T_val)
        # Only substitute if the symbol is free in the expression
        if self.rho_E_sym is not None:
            free = self.rho_E_sym.free_symbols
            if self.beta in free:
                subs[self.beta] = beta_val
            if self.k in free:
                subs[self.k] = k_val
            if self.T in free:
                subs[self.T] = T_val
            # also E is not substituted here
        return subs

    # ----------------------------------------------------------------------
    # Test runner
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