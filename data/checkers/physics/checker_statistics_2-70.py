import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dictionary
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.u_omega_formula = results.get('u_omega_formula', None)
        self.temp_dependence = results.get('temp_dependence', None)

        # Optionally extract given parameters (not required for tests)
        self.L = getattr(solution_instance, 'L', None)
        self.T = getattr(solution_instance, 'T', None)

    # --------------------- Interface Tests ---------------------
    def test_interface_required_keys(self):
        """Check that both required result keys are present."""
        required = ['u_omega_formula', 'temp_dependence']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required keys are present")

    # --------------------- Invariant / Theorem Tests ---------------------
    def test_invariant_planck_law_structure(self):
        """Verify the formula matches Planck's law for cavity radiation."""
        if self.u_omega_formula is None:
            return (False, "SKIP: u_omega_formula not available")
        if not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "u_omega_formula is not a sympy expression")
        try:
            omega, T, hbar, c, kB = sp.symbols('omega T hbar c kB', positive=True)
            expected = (hbar/(sp.pi**2 * c**3)) * omega**3 / (sp.exp(hbar*omega/(kB*T)) - 1)

            # Exact symbolic match
            diff = sp.simplify(self.u_omega_formula - expected)
            if diff == 0:
                return (True, "Formula exactly matches Planck's law")

            # Numerical fallback
            subs = {
                omega: 1e15,
                T: 3000,
                hbar: 1.0545718e-34,
                c: 3e8,
                kB: 1.380649e-23
            }
            val_expr = self.u_omega_formula.subs(subs).evalf()
            val_expected = expected.subs(subs).evalf()
            if sp.Abs(val_expr - val_expected) / val_expected < 1e-5:
                return (True, "Formula matches Planck's law numerically")
            else:
                return (False, f"Numerical mismatch: ratio {float(val_expr/val_expected):.7f}")
        except Exception as e:
            return (False, f"Structure test exception: {e}")

    def test_invariant_rayleigh_jeans_limit(self):
        """Low-frequency (or high-temperature) limit: u ~ (ω^2 k_B T)/(π^2 c^3)."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            omega, T, hbar, c, kB = sp.symbols('omega T hbar c kB', positive=True)
            # Expand around ω=0
            series = sp.series(self.u_omega_formula, omega, 0, 3).removeO()
            expected_low = (kB * T / (sp.pi**2 * c**3)) * omega**2

            subs = {
                omega: 1e10,      # very low frequency
                T: 3000,
                hbar: 1.0545718e-34,
                c: 3e8,
                kB: 1.380649e-23
            }
            val_series = series.subs(subs).evalf()
            val_expected = expected_low.subs(subs).evalf()
            if sp.Abs(val_series - val_expected) / val_expected < 1e-2:
                return (True, "Low-frequency limit matches Rayleigh-Jeans")
            else:
                return (False, f"Ratio deviation: {float(val_series/val_expected):.4f}")
        except Exception as e:
            return (False, f"Rayleigh-Jeans test exception: {e}")

    def test_invariant_wien_limit(self):
        """High-frequency limit: u ~ (ħ ω^3)/(π^2 c^3) exp(-ħ ω/(k_B T))."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            omega, T, hbar, c, kB = sp.symbols('omega T hbar c kB', positive=True)
            expected_high = (hbar * omega**3) / (sp.pi**2 * c**3) * sp.exp(-hbar*omega/(kB*T))

            subs = {
                omega: 1e18,      # very high frequency
                T: 3000,
                hbar: 1.0545718e-34,
                c: 3e8,
                kB: 1.380649e-23
            }
            val_expr = self.u_omega_formula.subs(subs).evalf()
            val_expected = expected_high.subs(subs).evalf()
            if sp.Abs(val_expr - val_expected) / val_expected < 1e-2:
                return (True, "High-frequency limit matches Wien approximation")
            else:
                return (False, f"Ratio deviation: {float(val_expr/val_expected):.4f}")
        except Exception as e:
            return (False, f"Wien test exception: {e}")

    def test_invariant_total_energy_scaling(self):
        """Check that total energy density is proportional to T^4."""
        if self.temp_dependence is None:
            return (False, "SKIP: temp_dependence not available")
        dep = str(self.temp_dependence).strip()
        if 'T^4' in dep or 'T**4' in dep or 'T⁴' in dep:
            return (True, "Temperature dependence is T^4")
        # If it's a sympy expression, check for T**4
        if isinstance(self.temp_dependence, sp.Basic):
            if self.temp_dependence.has(sp.Pow) and self.temp_dependence.has(sp.Symbol('T')):
                try:
                    poly = sp.Poly(self.temp_dependence, sp.Symbol('T'))
                    if poly.degree() == 4:
                        return (True, "Temperature dependence is T^4")
                except Exception:
                    pass
        return (False, f"Temperature dependence not recognized as T^4: {dep}")

    # --------------------- Modeling / Trace Tests (at least two) ---------------------
    def test_modeling_positivity(self):
        """Check that the energy density is positive for physical parameters."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            omega, T, hbar, c, kB = sp.symbols('omega T hbar c kB', positive=True)
            subs = {omega: 1e15, T: 3000, hbar: 1.0545718e-34, c: 3e8, kB: 1.380649e-23}
            val = self.u_omega_formula.subs(subs).evalf()
            if float(val) > 0:
                return (True, "Energy density is positive")
            else:
                return (False, f"Non-positive value: {val}")
        except Exception as e:
            return (False, f"Positivity test exception: {e}")

    def test_modeling_contains_hbar_and_kB(self):
        """Verify that the formula contains the correct physical constants: ħ and k_B."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            symbols_in_expr = {s.name for s in self.u_omega_formula.free_symbols}
            required = {'hbar', 'kB'}
            missing = required - symbols_in_expr
            if missing:
                return (False, f"Missing constant symbols: {missing}")
            return (True, "Formula contains hbar and kB")
        except Exception as e:
            return (False, f"Constants check exception: {e}")

    def test_modeling_denominator_form(self):
        """Check that the denominator is of the form exp(...) - 1."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            expr = sp.simplify(self.u_omega_formula)
            # Find denominators that contain exponentials
            denom = None
            for term in sp.preorder_traversal(expr):
                if term.is_Pow and term.exp == -1:
                    potential = term.base
                    if potential.has(sp.exp):
                        denom = potential
                        break
            if denom is None:
                frac = sp.together(expr)
                if isinstance(frac, sp.Mul):
                    for factor in frac.args:
                        if factor.is_Pow and factor.exp == -1:
                            denom = factor.base
                            break
                elif isinstance(frac, sp.Pow) and frac.exp == -1:
                    denom = frac.base
            if denom is not None:
                if isinstance(denom, sp.Add) and denom.args[-1] == -1 and denom.args[0].has(sp.exp):
                    return (True, "Denominator has form exp(...) - 1")
                else:
                    return (False, f"Denominator not of form exp(...)-1: {denom}")
            else:
                return (False, "Could not identify a denominator with exponential")
        except Exception as e:
            return (False, f"Denominator check exception: {e}")

    # --------------------- Monotonicity Tests (each must perturb input and compare) ---------------------
    def test_monotonicity_energy_increases_with_T(self):
        """For a fixed frequency, energy density increases when temperature is increased."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            omega, T, hbar, c, kB = sp.symbols('omega T hbar c kB', positive=True)
            subs = {omega: 1e15, hbar: 1.0545718e-34, c: 3e8, kB: 1.380649e-23}

            # Baseline at T0
            T0 = 3000.0
            baseline_val = float(self.u_omega_formula.subs({**subs, T: T0}).evalf())

            # Perturbation: increase T by 10 K
            T_perturbed = T0 + 10.0
            perturbed_val = float(self.u_omega_formula.subs({**subs, T: T_perturbed}).evalf())

            if perturbed_val > baseline_val:
                return (True, f"Energy density increased with T: baseline={baseline_val:.6e}, perturbed={perturbed_val:.6e}")
            else:
                return (False, f"Energy density did not increase: baseline={baseline_val:.6e}, perturbed={perturbed_val:.6e}")
        except Exception as e:
            return (False, f"Monotonicity test exception: {e}")

    def test_monotonicity_energy_decreases_with_frequency_in_high_region(self):
        """For fixed T, in the high-frequency tail (above the peak), energy density decreases when frequency is increased."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            omega, T, hbar, c, kB = sp.symbols('omega T hbar c kB', positive=True)
            subs = {T: 3000, hbar: 1.0545718e-34, c: 3e8, kB: 1.380649e-23}

            # Choose a frequency well above the peak (≈ 2e15 rad/s at T=3000K)
            omega0 = 5e15
            baseline_val = float(self.u_omega_formula.subs({**subs, omega: omega0}).evalf())

            # Perturbation: increase frequency by 10^14
            omega_perturbed = omega0 + 1e14
            perturbed_val = float(self.u_omega_formula.subs({**subs, omega: omega_perturbed}).evalf())

            if perturbed_val < baseline_val:
                return (True, f"Energy density decreased with frequency increase: baseline={baseline_val:.6e}, perturbed={perturbed_val:.6e}")
            else:
                return (False, f"Energy density did not decrease: baseline={baseline_val:.6e}, perturbed={perturbed_val:.6e}")
        except Exception as e:
            return (False, f"Monotonicity test exception: {e}")

    # --------------------- Answer Sanity (weak, minimal) ---------------------
    def test_answer_sanity_sign_and_finite(self):
        """Check that the formula yields a positive finite number for typical parameters."""
        if self.u_omega_formula is None or not isinstance(self.u_omega_formula, sp.Basic):
            return (False, "SKIP: u_omega_formula not available or not symbolic")
        try:
            omega, T, hbar, c, kB = sp.symbols('omega T hbar c kB', positive=True)
            subs = {omega: 1e15, T: 3000, hbar: 1.0545718e-34, c: 3e8, kB: 1.380649e-23}
            val = float(self.u_omega_formula.subs(subs).evalf())
            if val > 0 and math.isfinite(val):
                return (True, "Value is positive and finite")
            else:
                return (False, f"Value not positive and finite: {val}")
        except Exception as e:
            return (False, f"Sanity test exception: {e}")

    # --------------------- Test Runner ---------------------
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