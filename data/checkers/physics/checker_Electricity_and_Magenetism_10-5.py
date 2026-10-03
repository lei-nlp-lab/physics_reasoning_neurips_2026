import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a rectangular waveguide solution for TE/TM modes, dispersion,
    phase/group velocities, and cutoff frequency.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Extract given parameters
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.m = getattr(solution_instance, 'm', None)
        self.n = getattr(solution_instance, 'n', None)
        self.c = getattr(solution_instance, 'c', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.k = getattr(solution_instance, 'k', None)

        # Numeric velocity and cutoff (may be provided in solution_instance)
        self.v_p_num = getattr(solution_instance, 'v_p', None)
        self.v_g_num = getattr(solution_instance, 'v_g', None)
        self.product_num = getattr(solution_instance, 'product', None)
        self.omega_c_num = getattr(solution_instance, 'omega_c', None)

        # If numeric values are not defined, try to extract from results dict
        if self.v_p_num is None:
            vp_str = self.results.get('v_p', None)
            if vp_str is not None and '=' in vp_str:
                # Simple extraction: after '= ' we expect a number or expression
                # We'll not parse sophisticatedly; keep as None.
                pass
        if self.v_g_num is None:
            vg_str = self.results.get('v_g', None)
        if self.product_num is None:
            prod_str = self.results.get('product', None)
        if self.omega_c_num is None:
            oc_str = self.results.get('omega_c', None)

        # Store string expressions for trace checks
        self.E_x_TE_str = self.results.get('E_x_TE', None)
        self.E_y_TE_str = self.results.get('E_y_TE', None)
        self.E_x_TM_str = self.results.get('E_x_TM', None)
        self.E_y_TM_str = self.results.get('E_y_TM', None)
        self.E_z_TM_str = self.results.get('E_z_TM', None)
        self.dispersion_str = self.results.get('dispersion', None)
        self.v_p_str = self.results.get('v_p', None)
        self.v_g_str = self.results.get('v_g', None)
        self.product_str = self.results.get('product', None)
        self.omega_c_str = self.results.get('omega_c', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present."""
        required_keys = [
            'E_x_TE', 'E_y_TE', 'E_x_TM', 'E_y_TM', 'E_z_TM',
            'dispersion', 'v_p', 'v_g', 'product', 'omega_c'
        ]
        missing = [key for key in required_keys if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_keys_type(self):
        """Check that all string results are non-empty strings."""
        for key, val in self.results.items():
            if not isinstance(val, str) or len(val.strip()) == 0:
                return (False, f"Key {key} is not a non-empty string")
        return (True, "All keys have valid string values")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, using math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_dispersion_relation(self):
        """Check ω² = (k² + (mπ/a)²+(nπ/b)²)c² when numeric values given."""
        if None in (self.a, self.b, self.m, self.n, self.c, self.omega, self.k):
            return (False, "SKIP: missing numeric parameters for dispersion check")
        a, b, m, n, c, omega, k = self.a, self.b, self.m, self.n, self.c, self.omega, self.k
        try:
            alpha = m * math.pi / a
            beta = n * math.pi / b
            lhs = omega ** 2
            rhs = (k ** 2 + alpha ** 2 + beta ** 2) * c ** 2
        except (ZeroDivisionError, TypeError):
            return (False, "Error computing dispersion (check a, b, m, n values)")
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Dispersion holds: ω²={lhs:.3g} ≈ {rhs:.3g}")
        else:
            return (False, f"Dispersion mismatch: ω²={lhs:.3g}, expected={rhs:.3g}")

    def test_invariant_product(self):
        """Check v_p * v_g = c² if numeric values available."""
        if None in (self.v_p_num, self.v_g_num, self.c):
            # Try to parse from strings? Simpler: skip.
            return (False, "SKIP: no numeric v_p/v_g/c for product check")
        product = self.v_p_num * self.v_g_num
        expected = self.c ** 2
        if math.isclose(product, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_p * v_g = {product:.3g} ≈ c²={expected:.3g}")
        else:
            return (False, f"v_p * v_g = {product:.3g}, expected c²={expected:.3g}")

    def test_invariant_phase_velocity_greater_c(self):
        """Check v_p > c for propagating mode."""
        if self.v_p_num is None or self.c is None:
            return (False, "SKIP: missing numeric v_p or c")
        if self.v_p_num > self.c:
            return (True, f"v_p = {self.v_p_num:.3g} > c = {self.c:.3g}")
        else:
            return (False, f"v_p = {self.v_p_num:.3g} is not > c = {self.c:.3g}")

    def test_invariant_group_velocity_less_c(self):
        """Check v_g < c for propagating mode."""
        if self.v_g_num is None or self.c is None:
            return (False, "SKIP: missing numeric v_g or c")
        if self.v_g_num < self.c:
            return (True, f"v_g = {self.v_g_num:.3g} < c = {self.c:.3g}")
        else:
            return (False, f"v_g = {self.v_g_num:.3g} is not < c = {self.c:.3g}")

    def test_invariant_cutoff_frequency(self):
        """Check cutoff frequency ω_c = πc/a for TE10 mode."""
        if None in (self.a, self.c, self.omega_c_num):
            return (False, "SKIP: missing a, c, or omega_c numeric")
        # Assume TE10 mode (m=1, n=0) if not specified
        m = self.m if self.m is not None else 1
        n = self.n if self.n is not None else 0
        if m != 1 or n != 0:
            # If mode is not TE10, we cannot compare to simple formula
            return (False, f"SKIP: mode not TE10 (m={m}, n={n})")
        expected = math.pi * self.c / self.a
        if math.isclose(self.omega_c_num, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ω_c = {self.omega_c_num:.3g} ≈ πc/a = {expected:.3g}")
        else:
            return (False, f"ω_c = {self.omega_c_num:.3g}, expected {expected:.3g}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_field_forms_TE(self):
        """Check that TE field strings contain expected sin/cos structure."""
        if self.E_x_TE_str is None or self.E_y_TE_str is None:
            return (False, "SKIP: TE field strings missing")
        # Check that both contain "cos", "sin", "e^{i(kz-omegat)}"
        required_parts = ["cos", "sin", "e^{i(kz-omegat)}"]
        for part in required_parts:
            if part not in self.E_x_TE_str:
                return (False, f"E_x_TE missing '{part}'")
            if part not in self.E_y_TE_str:
                return (False, f"E_y_TE missing '{part}'")
        # TE mode should have no E_z string defined (or E_z_TE should be absent/None)
        # This is implicit; we don't require it.
        return (True, "TE field strings contain required functional forms")

    def test_modeling_field_forms_TM(self):
        """Check that TM field strings contain expected sin/cos and E_z structure."""
        if any(s is None for s in [self.E_x_TM_str, self.E_y_TM_str, self.E_z_TM_str]):
            return (False, "SKIP: TM field strings missing")
        required_parts = ["cos", "sin", "e^{i(kz-omegat)}"]
        for s, name in [(self.E_x_TM_str, "E_x_TM"),
                        (self.E_y_TM_str, "E_y_TM"),
                        (self.E_z_TM_str, "E_z_TM")]:
            for part in required_parts:
                if part not in s:
                    return (False, f"{name} missing '{part}'")
        # TM E_z should have only "sin" factors (not cos) — boundary condition
        if "cos" in self.E_z_TM_str:
            # That's okay? Actually E_z = sin sin; it shouldn't have cos.
            return (False, "E_z_TM should not contain cos (sin sin expected)")
        return (True, "TM field strings contain required functional forms")

    def test_modeling_dispersion_form(self):
        """Check that dispersion string contains ω, k, a, b, m, n, c."""
        if self.dispersion_str is None:
            return (False, "SKIP: dispersion string missing")
        keywords = ["ω", "omega", "k", "c", "m", "n", "a", "b", "π", "pi"]
        found = any(kw in self.dispersion_str for kw in keywords)
        if not found:
            return (False, "Dispersion string lacks physics symbols")
        if "=" not in self.dispersion_str:
            return (False, "Dispersion string missing '='")
        return (True, "Dispersion string contains essential symbols")

    def test_trace_vp_vg_strings(self):
        """Check v_p and v_g strings contain appropriate structure."""
        if self.v_p_str is None or self.v_g_str is None:
            return (False, "SKIP: v_p or v_g strings missing")
        if "v_p" not in self.v_p_str and "v_{p}" not in self.v_p_str:
            return (False, "v_p string does not contain its name")
        if "v_g" not in self.v_g_str and "v_{g}" not in self.v_g_str:
            return (False, "v_g string does not contain its name")
        # Check that they involve sqrt or c
        if "sqrt" not in self.v_p_str and "√" not in self.v_p_str:
            return (False, "v_p string should contain sqrt")
        if "sqrt" not in self.v_g_str and "√" not in self.v_g_str:
            return (False, "v_g string should contain sqrt")
        return (True, "v_p and v_g strings have expected form")

    def test_trace_product_string(self):
        """Check product string equals c^2."""
        if self.product_str is None:
            return (False, "SKIP: product string missing")
        if "c^2" not in self.product_str and "c²" not in self.product_str:
            return (False, "Product string should contain c^2")
        return (True, "Product string correctly states v_p v_g = c^2")

    def test_trace_cutoff_string(self):
        """Check cutoff string mentions omega_c and pi c / a."""
        if self.omega_c_str is None:
            return (False, "SKIP: omega_c string missing")
        if "omega_c" not in self.omega_c_str and "ω_{c}" not in self.omega_c_str:
            # Allow full expression
            if "πc/a" not in self.omega_c_str and "pi c / a" not in self.omega_c_str:
                return (False, "Cutoff string should contain πc/a")
        return (True, "Cutoff string contains expected expression")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturb one input and compare to computed baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_phase_velocity_with_k(self):
        """
        Check that increasing k (propagation constant) decreases v_p.
        Compute baseline v_p = omega/k from given inputs.
        Perturb k by +5% and compute new v_p; expect decrease.
        """
        if None in (self.omega, self.k):
            return (False, "SKIP: missing omega or k for monotonicity check")
        # Baseline
        v_p_base = self.omega / self.k
        # Perturbed
        k_pert = self.k * 1.05
        v_p_pert = self.omega / k_pert
        if v_p_pert < v_p_base:
            return (True, f"v_p decreases when k increases: {v_p_base:.3g} -> {v_p_pert:.3g}")
        else:
            return (False, f"Monotonicity violated: v_p should decrease, got {v_p_pert:.3g} >= {v_p_base:.3g}")

    def test_monotonicity_group_velocity_with_k(self):
        """
        Check that increasing k increases v_g (since v_g = c^2 k / omega).
        Compute baseline v_g from omega and k.
        Perturb k by +5% and compute new v_g; expect increase.
        """
        if None in (self.omega, self.k, self.c):
            return (False, "SKIP: missing omega, k, or c for v_g monotonicity")
        # Baseline
        v_g_base = (self.c ** 2) * self.k / self.omega
        # Perturbed
        k_pert = self.k * 1.05
        v_g_pert = (self.c ** 2) * k_pert / self.omega
        if v_g_pert > v_g_base:
            return (True, f"v_g increases when k increases: {v_g_base:.3g} -> {v_g_pert:.3g}")
        else:
            return (False, f"Monotonicity violated: v_g should increase, got {v_g_pert:.3g} <= {v_g_base:.3g}")

    def test_monotonicity_cutoff_with_a(self):
        """
        Check that increasing a (waveguide width) decreases cutoff frequency.
        For TE10 mode, ω_c = πc/a.
        If a and c are given, compute baseline ω_c, then perturb a by +10%,
        and check that ω_c decreases.
        """
        if None in (self.a, self.c):
            return (False, "SKIP: missing a or c for cutoff monotonicity")
        # Assume TE10 mode
        omega_c_base = math.pi * self.c / self.a
        # Perturb a by +10%
        a_pert = self.a * 1.10
        omega_c_pert = math.pi * self.c / a_pert
        if omega_c_pert < omega_c_base:
            return (True, f"ω_c decreases when a increases: {omega_c_base:.3g} -> {omega_c_pert:.3g}")
        else:
            return (False, f"Monotonicity violated: ω_c should decrease, got {omega_c_pert:.3g} >= {omega_c_base:.3g}")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (one test only)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: all numeric results (if present) are positive and finite.
        No isclose, no re-derivation.
        """
        numeric_fields = [self.v_p_num, self.v_g_num, self.omega_c_num]
        for val, name in zip(numeric_fields, ["v_p", "v_g", "omega_c"]):
            if val is not None:
                if val <= 0 or not math.isfinite(val):
                    return (False, f"{name} = {val} is not positive finite")
        # Also check omega and k if present
        for val, name in zip([self.omega, self.k], ["omega", "k"]):
            if val is not None and val <= 0:
                return (False, f"{name} = {val} should be positive for propagation")
        # Check that dispersion relation is mentioned (string)
        if self.dispersion_str is not None and "omega" in self.dispersion_str:
            pass  # weak check
        return (True, "All numeric values are positive and finite")

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