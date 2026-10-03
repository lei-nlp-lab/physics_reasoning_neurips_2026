import sympy as sp
import math
import re
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dict (most important)
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.mu = None
            self.N_scaling = None
            self.rho_omega = None
            self.E_scaling = None
        else:
            self.mu = self.results.get('mu', None)
            self.N_scaling = self.results.get('N_scaling', None)
            self.rho_omega = self.results.get('rho_omega', None)
            self.E_scaling = self.results.get('E_scaling', None)

        # Try to get givens if available (optional, for advanced tests)
        self.V = getattr(solution_instance, 'V', None)
        self.T = getattr(solution_instance, 'T', None)
        # Constants: we define symbols for symbolic checks
        self.hbar_sym = sp.Symbol('hbar', positive=True)
        self.c_sym = sp.Symbol('c', positive=True)
        self.k_sym = sp.Symbol('k', positive=True)
        self.T_sym = sp.Symbol('T', positive=True)
        self.omega_sym = sp.Symbol('omega', positive=True)

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
    # HELPER: parse rho_omega string into sympy expression (or None)
    # ------------------------------------------------------------------
    def _parse_rho_omega(self):
        """Return sympy expression for rho_omega or None if parsing fails."""
        if self.rho_omega is None:
            return None
        try:
            parsed_str = self.rho_omega.replace("e^{", "exp(").replace("}", ")")
            parsed_str = parsed_str.replace("\\omega", "omega")
            local_dict = {
                'hbar': self.hbar_sym,
                'c': self.c_sym,
                'k': self.k_sym,
                'T': self.T_sym,
                'omega': self.omega_sym,
                'pi': sp.pi,
                'exp': sp.exp
            }
            return sp.sympify(parsed_str, locals=local_dict)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        missing = []
        if self.mu is None:
            missing.append('mu')
        if self.N_scaling is None:
            missing.append('N_scaling')
        if self.rho_omega is None:
            missing.append('rho_omega')
        if self.E_scaling is None:
            missing.append('E_scaling')
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All four required keys are present.")

    def test_interface_mu_is_not_none(self):
        if self.mu is None:
            return (False, "mu is missing.")
        return (True, "mu is present.")

    def test_interface_N_scaling_is_string(self):
        if not isinstance(self.N_scaling, str):
            return (False, "N_scaling should be a string.")
        return (True, "N_scaling is a string.")

    def test_interface_rho_omega_is_string(self):
        if not isinstance(self.rho_omega, str):
            return (False, "rho_omega should be a string.")
        return (True, "rho_omega is a string.")

    def test_interface_E_scaling_is_string(self):
        if not isinstance(self.E_scaling, str):
            return (False, "E_scaling should be a string.")
        return (True, "E_scaling is a string.")

    # ------------------------------------------------------------------
    # INVARIANT TESTS (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_mu_is_zero(self):
        if self.mu is None:
            return (False, "mu not provided.")
        try:
            mu_val = float(self.mu)
        except (ValueError, TypeError):
            return (False, "mu cannot be converted to number.")
        if math.isclose(mu_val, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return (True, "mu = 0 as required.")
        else:
            return (False, f"mu = {mu_val}, expected 0.")

    def test_invariant_N_scaling_contains_T3(self):
        if self.N_scaling is None:
            return (False, "N_scaling not provided.")
        s = self.N_scaling.replace(" ", "").lower()
        if "t^3" in s or "t**3" in s or "t^{3}" in s:
            return (True, "N_scaling includes T^3.")
        return (False, f"'{self.N_scaling}' does not contain T^3.")

    def test_invariant_E_scaling_contains_T4(self):
        if self.E_scaling is None:
            return (False, "E_scaling not provided.")
        s = self.E_scaling.replace(" ", "").lower()
        if "t^4" in s or "t**4" in s or "t^{4}" in s:
            return (True, "E_scaling includes T^4.")
        return (False, f"'{self.E_scaling}' does not contain T^4.")

    def test_invariant_rho_omega_functional_form(self):
        if self.rho_omega is None:
            return (False, "rho_omega not provided.")
        expr = self.rho_omega.replace(" ", "")
        checks = [
            ("hbar" in expr, "missing hbar"),
            ("omega^3" in expr or "ω^3" in expr, "missing ω^3"),
            ("e^{" in expr or "exp(" in expr, "missing exponential denominator"),
            ("1/(" in expr or "/(e" in expr or "/(exp" in expr, "missing 1/denominator pattern"),
        ]
        failures = [msg for ok, msg in checks if not ok]
        if failures:
            return (False, f"rho_omega: {'; '.join(failures)}")
        return (True, "rho_omega has Planck distribution form.")

    # ------------------------------------------------------------------
    # MODELING / TRACE TESTS (consistency with explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_bose_einstein_energy_integrand(self):
        """Check that rho_omega matches  ϵ g(ϵ)⟨n(ϵ)⟩/V  in terms of ω."""
        if self.rho_omega is None:
            return (False, "rho_omega not provided.")
        omega = self.omega_sym
        hbar = self.hbar_sym
        c = self.c_sym
        k = self.k_sym
        T = self.T_sym
        expected = (hbar * omega**3) / (sp.pi**2 * c**3) * 1/(sp.exp(hbar*omega/(k*T)) - 1)
        user_expr = self._parse_rho_omega()
        if user_expr is None:
            return (False, "Cannot parse rho_omega string.")
        diff = sp.simplify(user_expr - expected)
        if diff == 0:
            return (True, "rho_omega matches theoretical expression symbolically.")
        # fallback numeric check
        subs = {hbar: 1.0545718e-34, c: 3e8, k: 1.380649e-23, T: 300, omega: 1e14}
        try:
            num_user = float(user_expr.subs(subs).evalf())
            num_exp = float(expected.subs(subs).evalf())
            if math.isclose(num_user, num_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "rho_omega matches numerically (within 1%).")
        except Exception:
            pass
        return (False, "rho_omega does not match the expected Bose-Einstein integrand.")

    def test_modeling_density_of_states_connection(self):
        """Check that rho_omega = (ħω) * (g(ω)/V) * n(ω)."""
        if self.rho_omega is None:
            return (False, "rho_omega not provided.")
        omega = self.omega_sym
        hbar = self.hbar_sym
        c = self.c_sym
        k = self.k_sym
        T = self.T_sym
        g_over_V = omega**2 / (sp.pi**2 * c**3)
        n = 1/(sp.exp(hbar*omega/(k*T)) - 1)
        expected = hbar * omega * g_over_V * n
        user_expr = self._parse_rho_omega()
        if user_expr is None:
            return (False, "Cannot parse rho_omega string.")
        diff = sp.simplify(user_expr - expected)
        if diff == 0:
            return (True, "rho_omega is correctly g(ω)/V * ħω * n(ω).")
        subs = {hbar: 1.0545718e-34, c: 3e8, k: 1.380649e-23, T: 500, omega: 5e13}
        try:
            num_user = float(user_expr.subs(subs).evalf())
            num_exp = float(expected.subs(subs).evalf())
            if math.isclose(num_user, num_exp, rel_tol=1e-2):
                return (True, "rho_omega matches density of states relation numerically.")
        except Exception:
            pass
        return (False, "rho_omega not consistent with density of states relation.")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS (explicit perturbation of input)
    # ------------------------------------------------------------------
    def test_monotonicity_rho_increases_with_temperature(self):
        """
        Explicit perturbation: evaluate rho(ω=1e14, T) at T=300 K and T=330 K.
        Check that rho increases.
        """
        if self.rho_omega is None:
            return (False, "SKIP: rho_omega not provided.")
        user_expr = self._parse_rho_omega()
        if user_expr is None:
            return (False, "SKIP: cannot parse rho_omega.")
        # Pick a fixed ω and constants
        hbar_val = 1.0545718e-34
        c_val = 3e8
        k_val = 1.380649e-23
        omega_val = 1e14
        T_low = 300.0
        T_high = 330.0
        subs_low = {self.hbar_sym: hbar_val, self.c_sym: c_val,
                    self.k_sym: k_val, self.omega_sym: omega_val,
                    self.T_sym: T_low}
        subs_high = {self.hbar_sym: hbar_val, self.c_sym: c_val,
                     self.k_sym: k_val, self.omega_sym: omega_val,
                     self.T_sym: T_high}
        try:
            rho_low = float(user_expr.subs(subs_low).evalf())
            rho_high = float(user_expr.subs(subs_high).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate rho_omega numerically: {e}")
        if math.isclose(rho_low, 0.0, abs_tol=1e-30):
            return (False, "rho at low temperature is zero or negligible, cannot test increase.")
        if rho_high > rho_low:
            return (True, f"rho(330 K)={rho_high:.3e} > rho(300 K)={rho_low:.3e}. Monotonic increase confirmed.")
        else:
            return (False, f"rho(330 K)={rho_high:.3e} is not greater than rho(300 K)={rho_low:.3e}.")

    def test_monotonicity_N_scaling_with_temperature(self):
        """
        Use the scaling law N ∝ T^exponent. Perturb T and compare relative values.
        Extract exponent from N_scaling string, assume f(T)=T^exponent.
        """
        if self.N_scaling is None:
            return (False, "SKIP: N_scaling not provided.")
        # Extract exponent
        s = self.N_scaling.replace(" ", "")
        exponent = None
        # Try different patterns
        for pat in [r'T\^([0-9.]+)', r'T\*\*([0-9.]+)', r'T\^{([0-9.]+)}']:
            m = re.search(pat, s)
            if m:
                exponent = float(m.group(1))
                break
        if exponent is None:
            return (False, "Cannot extract numerical exponent from N_scaling.")
        # Define f(T) = T^exponent
        T_low = 200.0
        T_high = 300.0
        f_low = T_low ** exponent
        f_high = T_high ** exponent
        if f_high > f_low:
            return (True, f"f(300K)={f_high:.2e} > f(200K)={f_low:.2e}. N_scaling implies N increases with T.")
        else:
            return (False, f"N_scaling does not give monotonic increase: f(300K)={f_high:.2e}, f(200K)={f_low:.2e}.")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY CHECK (only one allowed, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak_check(self):
        if self.mu is None or self.N_scaling is None or self.rho_omega is None or self.E_scaling is None:
            return (False, "One or more results missing.")
        # mu must be a number
        try:
            float(self.mu)
        except (ValueError, TypeError):
            return (False, "mu is not a number.")
        # N_scaling should contain T
        if 'T' not in str(self.N_scaling):
            return (False, "N_scaling does not contain T.")
        # E_scaling should contain T
        if 'T' not in str(self.E_scaling):
            return (False, "E_scaling does not contain T.")
        # rho_omega should contain some form of omega
        if 'omega' not in self.rho_omega and 'ω' not in self.rho_omega:
            return (False, "rho_omega missing omega symbol.")
        return (True, "All results pass basic sanity.")

# Example self-check
if __name__ == "__main__":
    class Dummy:
        def __init__(self):
            self.results = {
                'mu': 0,
                'N_scaling': 'T^3',
                'rho_omega': 'hbar * omega^3 / (pi^2 * c^3) * 1/(exp(hbar*omega/(k*T)) - 1)',
                'E_scaling': 'T^4'
            }
    checker = PhysicsChecker(Dummy())
    ok, logs = checker.run_all_tests()
    for l in logs:
        print(l)
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")