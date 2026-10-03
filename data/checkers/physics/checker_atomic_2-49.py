import math
import traceback

class PhysicsChecker:
    """Audits a solution for the deuteron photodisintegration problem."""

    def __init__(self, solution_instance):
        # ------------------------------------------------------------
        # Safe extraction of givens (atomic attributes)
        # ------------------------------------------------------------
        self.M = getattr(solution_instance, 'M', None)
        self.m_n = getattr(solution_instance, 'm_n', None)
        self.m_p = getattr(solution_instance, 'm_p', None)
        self.c = getattr(solution_instance, 'c', None)

        # ------------------------------------------------------------
        # Safely extract optional B (if provided by solution)
        # ------------------------------------------------------------
        self.B_given = getattr(solution_instance, 'B', None)

        # ------------------------------------------------------------
        # Safe extraction of results (expected key 'Delta_E')
        # ------------------------------------------------------------
        self.results = getattr(solution_instance, 'results', None)
        if self.results is not None and isinstance(self.results, dict):
            self.Delta_E = self.results.get('Delta_E', None)
        else:
            self.Delta_E = None

        # ------------------------------------------------------------
        # Derived binding energy B (only if all givens are present)
        # ------------------------------------------------------------
        if all(v is not None for v in [self.M, self.m_n, self.m_p, self.c]):
            self.B_derived = (self.m_n + self.m_p - self.M) * (self.c ** 2)
        else:
            self.B_derived = None

        # Use derived B as the reference for all physics tests
        self.B = self.B_derived

        # -----------------------------------------------------------------
        # Flag to skip numerical checks if any critical quantity is missing
        # -----------------------------------------------------------------
        self._numerical_possible = all(v is not None for v in
                                       [self.M, self.m_n, self.m_p, self.c, self.Delta_E])

    # -----------------------------------------------------------------
    # Helper to safely convert possibly symbolic values to float
    # -----------------------------------------------------------------
    @staticmethod
    def _to_float(value):
        """Convert to float if possible; otherwise return None."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # =================================================================
    # Interface tests
    # =================================================================
    def test_interface_givens_present(self):
        """Check that all required input parameters are provided."""
        missing = []
        for name in ['M', 'm_n', 'm_p', 'c']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present.")

    def test_interface_result_present(self):
        """Check that the result key 'Delta_E' exists and is numeric."""
        if self.Delta_E is None:
            return (False, "Missing result 'Delta_E'")
        val = self._to_float(self.Delta_E)
        if val is None:
            return (False, "'Delta_E' is not a numeric value")
        return (True, "'Delta_E' present and numeric.")

    def test_interface_binding_energy_positivity(self):
        """Binding energy B must be positive (deuteron is bound)."""
        if self.B is None:
            return (False, "SKIP: B not computable (missing givens)")
        B_val = self._to_float(self.B)
        if B_val is None:
            return (False, "B is not numeric")
        if B_val <= 0.0:
            return (False, f"B = {B_val} is not positive (unphysical)")
        return (True, f"B = {B_val} > 0")

    # =================================================================
    # Invariant tests (theorem‑based physical sanity)
    # =================================================================
    def test_invariant_lorentz_squared_energy_momentum(self):
        """
        Check the Lorentz‑invariant condition:
        (E_gamma + M c^2)^2 - E_gamma^2 = (m_n + m_p)^2 c^4
        where E_gamma = Delta_E + B.
        """
        if not self._numerical_possible or self.B is None:
            return (False, "SKIP: missing data for invariant check")
        M_val = self._to_float(self.M)
        m_n_val = self._to_float(self.m_n)
        m_p_val = self._to_float(self.m_p)
        c_val = self._to_float(self.c)
        delta_val = self._to_float(self.Delta_E)
        B_val = self._to_float(self.B)
        if any(v is None for v in [M_val, m_n_val, m_p_val, c_val, delta_val, B_val]):
            return (False, "SKIP: non‑numeric values for invariant check")

        E_gamma = delta_val + B_val
        left_side = (E_gamma + M_val * c_val ** 2) ** 2 - E_gamma ** 2
        right_side = (m_n_val + m_p_val) ** 2 * c_val ** 4

        if not math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Lorentz invariant violated: LHS={left_side}, RHS={right_side}")
        return (True, f"Lorentz invariant holds: values agree within 1%")

    def test_invariant_threshold_energy_consistency(self):
        """
        Alternative check using the derived relation:
        2 E_gamma M c^2 = [(m_n + m_p)^2 - M^2] c^4.
        """
        if not self._numerical_possible or self.B is None:
            return (False, "SKIP: missing data for threshold check")
        M_val = self._to_float(self.M)
        m_n_val = self._to_float(self.m_n)
        m_p_val = self._to_float(self.m_p)
        c_val = self._to_float(self.c)
        delta_val = self._to_float(self.Delta_E)
        B_val = self._to_float(self.B)
        if any(v is None for v in [M_val, m_n_val, m_p_val, c_val, delta_val, B_val]):
            return (False, "SKIP: non‑numeric values")

        E_gamma = delta_val + B_val
        left_side = 2.0 * E_gamma * M_val * c_val ** 2
        mass_diff = (m_n_val + m_p_val) ** 2 - M_val ** 2
        right_side = mass_diff * c_val ** 4

        if not math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Threshold relation violated: LHS={left_side}, RHS={right_side}")
        return (True, "Threshold energy relation satisfied")

    # =================================================================
    # Modeling / trace tests
    # =================================================================
    def test_modeling_delta_e_expression(self):
        """
        Verify that Delta_E matches the theoretically expected expression:
        Delta_E = B^2 / (2 M c^2)
        """
        if not self._numerical_possible or self.B is None:
            return (False, "SKIP: missing data for expression check")
        M_val = self._to_float(self.M)
        c_val = self._to_float(self.c)
        B_val = self._to_float(self.B)
        delta_val = self._to_float(self.Delta_E)
        if any(v is None for v in [M_val, c_val, B_val, delta_val]):
            return (False, "SKIP: non‑numeric values")

        expected_delta = (B_val ** 2) / (2.0 * M_val * c_val ** 2)
        if not math.isclose(delta_val, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Delta_E = {delta_val}, expected {expected_delta} (tolerance 1%)")
        return (True, f"Delta_E agrees with theoretical expression: {expected_delta}")

    def test_modeling_binding_energy_consistency(self):
        """
        If the solution provides its own B, check that it matches the
        derived B = (m_n + m_p - M) c^2.
        """
        if self.B_given is None:
            return (False, "SKIP: no B provided by solution")
        if self.B_derived is None:
            return (False, "SKIP: cannot compute derived B (missing givens)")
        B_given_val = self._to_float(self.B_given)
        B_derived_val = self._to_float(self.B_derived)
        if B_given_val is None or B_derived_val is None:
            return (False, "SKIP: non‑numeric B values")
        if not math.isclose(B_given_val, B_derived_val, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B solution = {B_given_val}, derived = {B_derived_val}")
        return (True, f"Solution B matches derived value: {B_derived_val}")

    def test_modeling_mass_squared_difference(self):
        """
        Trace check: Verify the identity used in the derivation:
        (m_n + m_p)^2 - M^2 = (B/c^2) * (2M + B/c^2)
        """
        if not self._numerical_possible or self.B is None:
            return (False, "SKIP: missing data for mass squared check")
        M_val = self._to_float(self.M)
        m_n_val = self._to_float(self.m_n)
        m_p_val = self._to_float(self.m_p)
        c_val = self._to_float(self.c)
        B_val = self._to_float(self.B)
        if any(v is None for v in [M_val, m_n_val, m_p_val, c_val, B_val]):
            return (False, "SKIP: non‑numeric values")

        left_side = (m_n_val + m_p_val) ** 2 - M_val ** 2
        right_side = (B_val / c_val ** 2) * (2.0 * M_val + B_val / c_val ** 2)

        if not math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Mass squared identity violated: LHS={left_side}, RHS={right_side}")
        return (True, "Mass squared identity holds")

    # =================================================================
    # Monotonicity tests (explicit input perturbation)
    # =================================================================
    def test_monotonicity_delta_e_decreases_with_deuteron_mass(self):
        """
        Test monotonicity: If the deuteron mass M increases, Delta_E should decrease.
        We perturb M by +1% and compare the theoretical Delta_E computed from the
        unperturbed and perturbed masses.
        """
        if not self._numerical_possible or self.B is None:
            return (False, "SKIP: missing data for monotonicity check")
        M_val = self._to_float(self.M)
        m_n_val = self._to_float(self.m_n)
        m_p_val = self._to_float(self.m_p)
        c_val = self._to_float(self.c)
        if any(v is None for v in [M_val, m_n_val, m_p_val, c_val]):
            return (False, "SKIP: non‑numeric values")

        # Baseline theoretical Delta_E (using the expression derived from physics)
        B_base = (m_n_val + m_p_val - M_val) * c_val ** 2
        Delta_E_base = B_base ** 2 / (2.0 * M_val * c_val ** 2)

        # Perturb M upward by 1%
        M_pert = M_val * 1.01
        B_pert = (m_n_val + m_p_val - M_pert) * c_val ** 2
        Delta_E_pert = B_pert ** 2 / (2.0 * M_pert * c_val ** 2)

        # Baseline should be larger than perturbed (since Delta_E decreases with M)
        if not (Delta_E_base > Delta_E_pert):
            return (False, f"Monotonicity violated: baseline Delta_E={Delta_E_base} "
                           f"not greater than perturbed (M+1%) Delta_E={Delta_E_pert}")
        return (True, f"Delta_E decreases with M: baseline={Delta_E_base:.6e} > perturbed={Delta_E_pert:.6e}")

    def test_monotonicity_delta_e_increases_with_binding_energy(self):
        """
        Test monotonicity: If the binding energy B increases, Delta_E should increase.
        We perturb M downward by 1% (so B increases) and compare theoretical Delta_E.
        """
        if not self._numerical_possible or self.B is None:
            return (False, "SKIP: missing data for monotonicity check")
        M_val = self._to_float(self.M)
        m_n_val = self._to_float(self.m_n)
        m_p_val = self._to_float(self.m_p)
        c_val = self._to_float(self.c)
        if any(v is None for v in [M_val, m_n_val, m_p_val, c_val]):
            return (False, "SKIP: non‑numeric values")

        # Baseline theoretical values
        B_base = (m_n_val + m_p_val - M_val) * c_val ** 2
        Delta_E_base = B_base ** 2 / (2.0 * M_val * c_val ** 2)

        # Perturb M downward by 1% (so B increases)
        M_pert = M_val * 0.99
        B_pert = (m_n_val + m_p_val - M_pert) * c_val ** 2
        Delta_E_pert = B_pert ** 2 / (2.0 * M_pert * c_val ** 2)

        # Baseline should be less than perturbed (since Delta_E increases with B)
        if not (Delta_E_base < Delta_E_pert):
            return (False, f"Monotonicity violated: baseline Delta_E={Delta_E_base} "
                           f"not less than perturbed (B increased) Delta_E={Delta_E_pert}")
        return (True, f"Delta_E increases with B: baseline={Delta_E_base:.6e} < perturbed={Delta_E_pert:.6e}")

    # =================================================================
    # Weak answer sanity (only one allowed)
    # =================================================================
    def test_answer_sanity_broad_range(self):
        """
        Weak sanity: Delta_E should be positive and smaller than B
        (since B << M c^2 implies B^2/(2Mc^2) << B).
        """
        if not self._numerical_possible or self.B is None:
            return (False, "SKIP: missing data for range check")
        B_val = self._to_float(self.B)
        delta_val = self._to_float(self.Delta_E)
        if delta_val is None or B_val is None:
            return (False, "SKIP: non‑numeric values")
        if delta_val <= 0.0:
            return (False, f"Delta_E = {delta_val} is not positive")
        if delta_val >= B_val:
            return (False, f"Delta_E = {delta_val} is not less than B = {B_val}")
        return (True, f"Delta_E = {delta_val} lies in (0, B)")

    # =================================================================
    # Main runner
    # =================================================================
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