import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Checks the solution of a quantum scattering problem (Born approximation)
    described in the prompt.
    """
    def __init__(self, solution_instance):
        # Safely extract all needed data, default to None if missing
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Required output keys (from [emit] contract)
        self.q = self.results.get('q', None)
        self.f = self.results.get('f', None)
        self.dsigma_dOmega = self.results.get('dsigma_dOmega', None)

        # Input parameters (may be symbolic or numeric)
        self.V0 = self.givens.get('V0', None)
        self.R = self.givens.get('R', None)
        self.m = self.givens.get('m', None)
        self.hbar = self.givens.get('hbar', None)
        self.k = self.givens.get('k', None)
        self.theta = self.givens.get('theta', None)

        # Helper: determine if we are dealing with symbolic expressions
        self.symbolic_mode = False
        for attr in ['q', 'f', 'dsigma_dOmega', 'V0', 'R', 'm', 'hbar', 'k', 'theta']:
            val = getattr(self, attr, None)
            if val is not None and hasattr(val, 'free_symbols'):
                self.symbolic_mode = True
                break

    def _to_float(self, expr):
        """Try to convert expression to a float, handling sympy objects."""
        if expr is None:
            return None
        if isinstance(expr, (int, float)):
            return float(expr)
        try:
            return float(expr.evalf())
        except:
            return None

    def _extract_theta(self):
        """If q is symbolic and contains theta, return the theta symbol."""
        if self.q is not None and hasattr(self.q, 'free_symbols'):
            for sym in self.q.free_symbols:
                if str(sym).lower().startswith('theta') or str(sym) == 't':
                    return sym
        return None

    def _compute_dsigma_numeric(self, V0, R, m, hbar, k, theta):
        """Re-derive dsigma/dΩ from fundamental Born expression using numeric values."""
        q = 2.0 * k * math.sin(theta / 2.0)
        sin_qR = math.sin(q * R)
        cos_qR = math.cos(q * R)
        f = (2.0 * m * V0 / (hbar**2)) * (1.0 / (q**3)) * (sin_qR - q * R * cos_qR)
        return f * f

    # ---------------------------------------------------------------
    # Interface tests
    # ---------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist."""
        missing = []
        if self.q is None:
            missing.append('q')
        if self.f is None:
            missing.append('f')
        if self.dsigma_dOmega is None:
            missing.append('dsigma_dOmega')
        if missing:
            return (False, f"Missing required results: {missing}")
        return (True, "All required keys present")

    def test_interface_f_type(self):
        """Check that f is numeric (or symbolic) – not a string or None."""
        if self.f is None:
            return (False, "f is None")
        if isinstance(self.f, (int, float)):
            return (True, "f is numeric")
        if hasattr(self.f, 'free_symbols'):
            return (True, "f is symbolic")
        return (False, f"f has unexpected type: {type(self.f)}")

    def test_interface_dsigma_dOmega_type(self):
        """Check that dsigma_dOmega is numeric or symbolic."""
        if self.dsigma_dOmega is None:
            return (False, "dsigma_dOmega is None")
        if isinstance(self.dsigma_dOmega, (int, float)):
            return (True, "dsigma_dOmega is numeric")
        if hasattr(self.dsigma_dOmega, 'free_symbols'):
            return (True, "dsigma_dOmega is symbolic")
        return (False, f"dsigma_dOmega has unexpected type: {type(self.dsigma_dOmega)}")

    # ---------------------------------------------------------------
    # Invariant / theorem-based sanity (use math.isclose for numeric)
    # ---------------------------------------------------------------
    def test_invariant_dsigma_nonnegative(self):
        """Differential cross section must be non-negative."""
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega not available")
        if self.symbolic_mode:
            return (True, "Symbolic expression accepted (cannot automatically determine sign)")
        val = self._to_float(self.dsigma_dOmega)
        if val is None:
            return (False, "Cannot convert dsigma_dOmega to float")
        if val < -1e-12:
            return (False, f"dsigma_dOmega is negative: {val}")
        return (True, "dsigma_dOmega is non-negative")

    def test_invariant_f_real(self):
        """Scattering amplitude f should be real (for real potential)."""
        if self.f is None:
            return (False, "SKIP: f not available")
        if self.symbolic_mode:
            # For symbolic expressions, we attempt to check imaginary part is zero
            if hasattr(self.f, 'as_real_imag'):
                real_part, imag_part = self.f.as_real_imag()
                imag_simpl = sp.simplify(imag_part)
                if imag_simpl == 0:
                    return (True, "Symbolic f is real")
                else:
                    return (False, f"Symbolic f has nonzero imaginary part: {imag_simpl}")
            # Assume real if no complex structure
            return (True, "Symbolic expression accepted as real (assumed)")
        # Numeric path
        try:
            f_complex = complex(self.f)
            if abs(f_complex.imag) < 1e-12:
                return (True, "f is real")
            else:
                return (False, f"f has nonzero imaginary part: {f_complex.imag}")
        except:
            return (False, "Cannot convert f to complex number")

    def test_invariant_q_from_k_theta(self):
        """Check that q = 2k sin(theta/2) if theta can be extracted."""
        if self.q is None or self.k is None:
            return (False, "SKIP: q or k not available")
        # Try to extract theta from givens or from q symbolic structure
        if self.symbolic_mode:
            theta_sym = self._extract_theta()
            if theta_sym is None:
                return (False, "SKIP: theta symbol not found in q")
            # Build expected expression: 2*k*sin(theta/2)
            expected = 2 * self.k * sp.sin(theta_sym / 2)
            diff = sp.simplify(self.q - expected)
            if diff == 0:
                return (True, "Symbolic q = 2k sin(theta/2)")
            else:
                return (False, f"Symbolic q mismatch, difference: {diff}")
        # Numeric path
        theta_val = self._to_float(self.theta) if self.theta is not None else None
        if theta_val is None:
            return (False, "SKIP: numeric theta not available")
        q_numeric = self._to_float(self.q)
        k_val = self._to_float(self.k)
        if q_numeric is None or k_val is None:
            return (False, "q or k not convertible to float")
        expected_q = 2.0 * k_val * math.sin(theta_val / 2.0)
        if math.isclose(q_numeric, expected_q, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"q matches 2*k*sin(θ/2): {q_numeric} ~ {expected_q}")
        else:
            return (False, f"q mismatch: got {q_numeric}, expected {expected_q}")

    # ---------------------------------------------------------------
    # Modeling / trace tests (use math.isclose for numeric)
    # ---------------------------------------------------------------
    def test_modeling_dsigma_from_f(self):
        """Check dsigma_dOmega = |f|^2."""
        if self.f is None or self.dsigma_dOmega is None:
            return (False, "SKIP: f or dsigma_dOmega missing")
        if self.symbolic_mode:
            f_sq = sp.simplify(self.f * sp.conjugate(self.f))
            diff = sp.simplify(self.dsigma_dOmega - f_sq)
            if diff == 0:
                return (True, "Symbolic dsigma_dOmega = |f|^2")
            else:
                return (False, f"Symbolic difference nonzero: {diff}")
        # Numeric path
        f_val = self._to_float(self.f)
        dsigma_val = self._to_float(self.dsigma_dOmega)
        if f_val is None or dsigma_val is None:
            return (False, "Cannot convert f or dsigma_dOmega to float")
        # f might be complex; use absolute value
        if isinstance(self.f, complex):
            f_abs = abs(self.f)
        else:
            f_abs = abs(f_val)
        expected = f_abs ** 2
        if math.isclose(dsigma_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dsigma_dOmega = |f|^2: {dsigma_val} ~ {expected}")
        else:
            return (False, f"Mismatch: got {dsigma_val}, expected |f|^2 = {expected}")

    def test_modeling_f_dependence_on_V0(self):
        """Check that f ~ V0 (linear in V0 in lowest order)."""
        if self.f is None or self.V0 is None:
            return (False, "SKIP: f or V0 missing")
        if self.symbolic_mode:
            try:
                # Check that derivative w.r.t. V0 is constant (doesn't contain V0)
                deriv = sp.diff(self.f, self.V0)
                if not deriv.has(self.V0):
                    return (True, "f is linear in V0 (symbolic derivative check)")
                else:
                    return (False, "f is not linear in V0 (derivative contains V0)")
            except:
                return (False, "Could not perform symbolic linearity check")
        # Numeric path: skip because we cannot easily vary V0 here.
        return (False, "SKIP: numeric linearity check requires perturbation, not implemented")

    # ---------------------------------------------------------------
    # Monotonicity tests (must explicitly perturb an input)
    # ---------------------------------------------------------------
    def test_monotonicity_dsigma_with_V0(self):
        """
        Perturb V0 by +10% and check that dsigma_dΩ increases.
        Uses re-derived formula to avoid dependency on solution's output.
        """
        # Check that all needed numeric inputs are present
        if self.V0 is None or self.R is None or self.m is None or self.hbar is None or self.k is None:
            return (False, "SKIP: missing numeric givens (V0, R, m, hbar, k)")
        if self.theta is None:
            # Try to extract theta from the solution's q? Not reliable. Skip.
            return (False, "SKIP: numeric theta not available in givens")

        V0 = self._to_float(self.V0)
        R = self._to_float(self.R)
        m = self._to_float(self.m)
        hbar = self._to_float(self.hbar)
        k = self._to_float(self.k)
        theta = self._to_float(self.theta)
        if None in (V0, R, m, hbar, k, theta):
            return (False, "SKIP: some givens are not convertible to float")

        # Baseline cross section with original V0
        dsigma_baseline = self._compute_dsigma_numeric(V0, R, m, hbar, k, theta)

        # Perturb V0 by +10%
        V0_perturbed = V0 * 1.1
        dsigma_perturbed = self._compute_dsigma_numeric(V0_perturbed, R, m, hbar, k, theta)

        # Check monotonic increase: perturbed > baseline
        if dsigma_perturbed > dsigma_baseline + 1e-12:
            return (True, f"dsigma increases with V0: baseline={dsigma_baseline}, perturbed={dsigma_perturbed}")
        else:
            return (False, f"Non-monotonic: baseline={dsigma_baseline}, perturbed={dsigma_perturbed}")

    # ---------------------------------------------------------------
    # Weak answer sanity (only one allowed, NO math.isclose, NO rederivation)
    # ---------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """Very weak sanity: dsigma_dΩ must be finite and non-negative."""
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega not available")
        if self.symbolic_mode:
            return (True, "Symbolic expression accepted (no obvious issues)")
        val = self._to_float(self.dsigma_dOmega)
        if val is None:
            return (False, "Cannot convert dsigma_dOmega to float")
        if math.isnan(val) or math.isinf(val):
            return (False, f"dsigma_dΩ is NaN or inf: {val}")
        if val < 0:
            return (False, f"dsigma_dΩ is negative: {val}")
        return (True, "dsigma_dΩ is finite and non-negative")

    # ---------------------------------------------------------------
    # Main runner
    # ---------------------------------------------------------------
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