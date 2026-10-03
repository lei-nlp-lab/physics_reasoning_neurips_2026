import math
import sympy as sym
import traceback

class PhysicsChecker:
    """
    Physics checker for molecular scattering (Rayleigh, fluctuations, Tyndall effect).
    """

    def __init__(self, solution_instance):
        # Extract results from the solution instance; default to None
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.lambda_dependence = results.get('lambda_dependence', None)
        self.angular_dependence = results.get('angular_dependence', None)
        self.I_no_fluct = results.get('I_no_fluct', None)
        self.I_total_fluct = results.get('I_total_fluct', None)
        self.enhanced = results.get('enhanced', None)

        # Store the solution instance for potential future use
        self.solution_instance = solution_instance

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_lambda_dependence_exists(self):
        if self.lambda_dependence is None:
            return (False, "Missing 'lambda_dependence' in results.")
        if not isinstance(self.lambda_dependence, str):
            return (False, f"'lambda_dependence' should be a string, got {type(self.lambda_dependence)}")
        return (True, "lambda_dependence is present and is a string.")

    def test_interface_angular_dependence_exists(self):
        if self.angular_dependence is None:
            return (False, "Missing 'angular_dependence' in results.")
        if not isinstance(self.angular_dependence, str):
            return (False, f"'angular_dependence' should be a string, got {type(self.angular_dependence)}")
        return (True, "angular_dependence is present and is a string.")

    def test_interface_I_no_fluct_exists(self):
        if self.I_no_fluct is None:
            return (False, "Missing 'I_no_fluct' in results.")
        # numeric or string? Expect numeric 0
        try:
            float(self.I_no_fluct)
        except (TypeError, ValueError):
            return (False, f"'I_no_fluct' should be numeric, got {type(self.I_no_fluct)}")
        return (True, "I_no_fluct is present and numeric.")

    def test_interface_I_total_fluct_exists(self):
        if self.I_total_fluct is None:
            return (False, "Missing 'I_total_fluct' in results.")
        if not isinstance(self.I_total_fluct, str):
            return (False, f"'I_total_fluct' should be a string, got {type(self.I_total_fluct)}")
        return (True, "I_total_fluct is present and is a string.")

    def test_interface_enhanced_exists(self):
        if self.enhanced is None:
            return (False, "Missing 'enhanced' in results.")
        if not isinstance(self.enhanced, str):
            return (False, f"'enhanced' should be a string, got {type(self.enhanced)}")
        return (True, "enhanced is present and is a string.")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_lambda_dependence_inverse_fourth(self):
        if self.lambda_dependence is None:
            return (False, "SKIP: lambda_dependence not available")
        text = self.lambda_dependence.replace(' ', '').lower()
        # Accept various forms: '1/lambda^4', 'λ⁻⁴', 'lambda^{-4}', 'propto 1/lambda^4'
        contains_inv_fourth = ('1/lambda^4' in text or 'λ⁻⁴' in text or
                               'lambda^{-4}' in text or 'propto1/lambda^4' in text)
        if not contains_inv_fourth:
            return (False, f"lambda_dependence '{self.lambda_dependence}' does not indicate inverse fourth power.")
        return (True, "lambda_dependence correctly shows I ∝ 1/λ⁴.")

    def test_invariant_angular_dependence_cos_squared(self):
        if self.angular_dependence is None:
            return (False, "SKIP: angular_dependence not available")
        text = self.angular_dependence.replace(' ', '').lower()
        # Must contain 'cos^2' or 'cos²' or 'cos^2theta'
        has_cos2 = ('cos^2' in text or 'cos²' in text)
        if not has_cos2:
            return (False, f"angular_dependence '{self.angular_dependence}' does not contain cos^2 term.")
        return (True, "angular_dependence includes cos^2 term as expected.")

    def test_invariant_no_fluct_zero(self):
        """Without fluctuations, net scattering should be zero (excluding forward direction)."""
        if self.I_no_fluct is None:
            return (False, "SKIP: I_no_fluct not available")
        try:
            val = float(self.I_no_fluct)
        except (TypeError, ValueError):
            return (False, f"I_no_fluct is not numeric: {self.I_no_fluct}")
        if math.isclose(val, 0.0, abs_tol=1e-9):
            return (True, f"I_no_fluct = {val} ≈ 0 as expected.")
        else:
            return (False, f"I_no_fluct = {val}, expected 0 (or very close).")

    def test_invariant_total_fluct_proportional_to_N0V(self):
        """Fluctuation-induced scattering scales with N0 V."""
        if self.I_total_fluct is None:
            return (False, "SKIP: I_total_fluct not available")
        text = self.I_total_fluct.replace(' ', '').lower()
        # Check for "N0 V" or "N0V" or "n0 V"
        if not ('n0v' in text or 'n0 v' in text or 'n_0 v' in text):
            return (False, f"I_total_fluct '{self.I_total_fluct}' does not explicitly include N0 V factor.")
        return (True, "I_total_fluct includes N0 V factor (scales with total number).")

    def test_invariant_enhanced_tyndall(self):
        """High humidity leads to Tyndall effect."""
        if self.enhanced is None:
            return (False, "SKIP: enhanced not available")
        text = self.enhanced.lower()
        if 'tyndall' not in text and 'mie' not in text:
            return (False, f"enhanced explanation '{self.enhanced}' does not mention Tyndall or Mie scattering.")
        return (True, "enhanced correctly attributes to Tyndall/Mie effect.")

    # ----------------------------------------------------------------------
    # Modeling/trace tests
    # ----------------------------------------------------------------------
    def test_modeling_fluct_contains_classical_electron_radius(self):
        """Check that the expression for I_total_fluct includes the classical electron radius squared factor."""
        if self.I_total_fluct is None:
            return (False, "SKIP: I_total_fluct not available")
        # Look for e^2/(4πε₀ m c^2) or equivalent
        text = self.I_total_fluct.replace(' ', '')
        # Accept various forms
        has_classical = ('e^2/(4πϵ₀mc^2)' in text or
                         'e^2/(4piepsilon0 m c^2)' in text or
                         '(e^2/(4πε0 m c^2))' in text or
                         'r_e' in text or
                         'clasicalradius' in text)
        if not has_classical:
            return (False, f"I_total_fluct '{self.I_total_fluct}' does not contain the classical electron radius squared factor.")
        return (True, "I_total_fluct includes the classical electron radius squared factor.")

    def test_modeling_fluct_contains_omega_to_fourth(self):
        if self.I_total_fluct is None:
            return (False, "SKIP: I_total_fluct not available")
        text = self.I_total_fluct.replace(' ', '')
        # Look for omega^4 or ω⁴
        has_omega4 = ('omega^4' in text or 'ω^4' in text or 'ω⁴' in text or 'omega^{4}' in text)
        if not has_omega4:
            return (False, f"I_total_fluct '{self.I_total_fluct}' does not contain ω⁴ factor.")
        return (True, "I_total_fluct correctly includes ω⁴ dependence.")

    def test_modeling_fluct_contains_sintheta_over_r(self):
        if self.I_total_fluct is None:
            return (False, "SKIP: I_total_fluct not available")
        text = self.I_total_fluct.replace(' ', '')
        # Look for sinθ/r or sintheta/r
        has_sintheta_r = ('sintheta/r' in text or 'sinθ/r' in text or 'sin(theta)/r' in text)
        if not has_sintheta_r:
            return (False, f"I_total_fluct '{self.I_total_fluct}' does not contain sinθ/r factor.")
        return (True, "I_total_fluct includes sinθ/r factor for dipole radiation pattern.")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_intensity_decreases_with_wavelength(self):
        """
        Rayleigh scattering: I ∝ 1/λ^4.
        If we increase λ by 10%, the intensity should decrease.
        """
        if self.lambda_dependence is None:
            return (False, "SKIP: lambda_dependence not available")
        # Check that expression indicates inverse fourth power
        text = self.lambda_dependence.replace(' ', '').lower()
        if not ('1/lambda^4' in text or 'λ⁻⁴' in text or 'lambda^{-4}' in text):
            return (False, "Cannot verify monotonicity: lambda_dependence does not indicate 1/λ⁴ proportionality.")
        # Use symbolic re-derivation: I ∝ 1/λ⁴
        lam = sym.symbols('lam', positive=True)
        I_expr = 1 / lam**4
        baseline = I_expr.subs(lam, 1)  # arbitrary unit
        perturbed = I_expr.subs(lam, 1.1)  # increase by 10%
        ratio = perturbed / baseline
        # Expected ratio = 1/1.1^4 ≈ 0.683
        expected_ratio = (1/1.1)**4
        if ratio < 1 and math.isclose(float(ratio), expected_ratio, rel_tol=1e-2):
            return (True, f"I(1.1λ)/I(λ) = {float(ratio):.4f} < 1, consistent with 1/λ⁴.")
        else:
            return (False, f"Ratio {float(ratio):.4f} not as expected (expected ≈0.683).")

    def test_monotonicity_angular_decrease_from_forward(self):
        """
        For unpolarized incident light, I(θ) ∝ 1+cos²θ.
        This function is strictly decreasing for small positive θ from 0.
        Perturb angle by +0.1 rad from 0 and verify intensity decreases.
        """
        if self.angular_dependence is None:
            return (False, "SKIP: angular_dependence not available")
        text = self.angular_dependence.replace(' ', '').lower()
        if 'cos^2' not in text and 'cos²' not in text:
            return (False, "Cannot verify monotonicity: angular_dependence does not contain cos^2 term.")
        # Re-derive expression: I ∝ 1+cos²θ
        θ = sym.symbols('θ', real=True)
        I_θ = 1 + sym.cos(θ)**2
        baseline_θ = 0.0
        perturbed_θ = 0.1  # rad, small increase
        I_baseline = I_θ.subs(θ, baseline_θ)
        I_perturbed = I_θ.subs(θ, perturbed_θ)
        # Since derivative dI/dθ = -2 cosθ sinθ = - sin(2θ), negative for θ in (0, π/2)
        if float(I_perturbed) < float(I_baseline):
            return (True, f"I({perturbed_θ}) = {float(I_perturbed):.4f} < I({baseline_θ}) = {float(I_baseline):.4f}, consistent with decreasing.")
        else:
            return (False, f"I increased after increasing angle by {perturbed_θ} rad; not monotonic decreasing.")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one allowed, must be very minimal)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Very light check: all required results exist and are finite/non-NaN.
        """
        fails = []
        # lambda_dependence: string, not empty
        if not isinstance(self.lambda_dependence, str) or len(self.lambda_dependence) < 1:
            fails.append("lambda_dependence missing or empty")
        # angular_dependence: string
        if not isinstance(self.angular_dependence, str) or len(self.angular_dependence) < 1:
            fails.append("angular_dependence missing or empty")
        # I_no_fluct: numeric and finite
        try:
            v = float(self.I_no_fluct)
            if math.isnan(v):
                fails.append("I_no_fluct is NaN")
        except Exception:
            fails.append("I_no_fluct not numeric")
        # I_total_fluct: string, non-empty
        if not isinstance(self.I_total_fluct, str) or len(self.I_total_fluct) < 1:
            fails.append("I_total_fluct missing or empty")
        # enhanced: string, non-empty
        if not isinstance(self.enhanced, str) or len(self.enhanced) < 1:
            fails.append("enhanced missing or empty")
        if fails:
            return (False, "; ".join(fails))
        return (True, "All required results present and basic sanity checks passed.")

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