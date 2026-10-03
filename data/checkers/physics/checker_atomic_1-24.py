import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Extract givens - try common locations
        self.hbar = getattr(solution_instance, 'hbar', None)
        if self.hbar is None:
            # try inside givens dict
            givens = getattr(solution_instance, 'givens', {})
            self.hbar = givens.get('hbar', None)

        self.m = getattr(solution_instance, 'm', None)
        if self.m is None:
            givens = getattr(solution_instance, 'givens', {})
            self.m = givens.get('m', None)

        self.R = getattr(solution_instance, 'R', None)
        if self.R is None:
            givens = getattr(solution_instance, 'givens', {})
            self.R = givens.get('R', None)

        # Extract computed pressure
        self.P = self.results.get('P', None)

        # Additional result variables often present in the solving pseudocode
        self.k = self.results.get('k', None)
        self.E = self.results.get('E', None)
        self.V = self.results.get('V', None)
        self.dE_dR = self.results.get('dE_dR', None)
        self.dV_dR = self.results.get('dV_dR', None)

    # ---------------------------------------------------------------------
    # Interface tests
    # ---------------------------------------------------------------------
    def test_interface_required_key_P(self):
        """Check that the result dict contains key 'P'."""
        if 'P' not in self.results:
            return (False, "SKIP: results['P'] missing")
        return (True, "results['P'] exists")

    def test_interface_P_is_numeric(self):
        """Check that P is a finite number."""
        if self.P is None:
            return (False, "SKIP: P is None")
        try:
            val = float(self.P)
            if math.isinf(val) or math.isnan(val):
                return (False, f"P is inf or nan: {val}")
            return (True, "P is finite numeric")
        except (TypeError, ValueError):
            return (False, f"P cannot be converted to float: {self.P}")

    def test_interface_givens_present(self):
        """Check that hbar, m, R are provided."""
        missing = []
        if self.hbar is None:
            missing.append("hbar")
        if self.m is None:
            missing.append("m")
        if self.R is None:
            missing.append("R")
        if missing:
            return (False, f"SKIP: missing givens: {', '.join(missing)}")
        return (True, "All givens hbar, m, R present")

    # ---------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ---------------------------------------------------------------------
    def test_invariant_pressure_formula(self):
        """Check that P = ħ²π / (4 m R⁵)."""
        if self.P is None:
            return (False, "SKIP: P is None")
        if self.hbar is None or self.m is None or self.R is None:
            return (False, "SKIP: missing givens for formula")
        try:
            hbar_val = float(self.hbar)
            m_val = float(self.m)
            R_val = float(self.R)
        except (TypeError, ValueError) as e:
            return (False, f"Cannot convert givens to float: {e}")
        expected_P = (hbar_val**2 * math.pi) / (4.0 * m_val * R_val**5)
        actual_P = float(self.P)
        if math.isclose(actual_P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P matches expected value (rel diff {abs(actual_P-expected_P)/expected_P:.4f})")
        else:
            return (False, f"P={actual_P:.6e} differs from expected {expected_P:.6e}")

    def test_invariant_positive_pressure(self):
        """Pressure must be positive."""
        if self.P is None:
            return (False, "SKIP: P is None")
        try:
            val = float(self.P)
        except (TypeError, ValueError):
            return (False, "P cannot be converted to float")
        if val > 0:
            return (True, f"P = {val:.4e} > 0")
        else:
            return (False, f"P = {val:.4e} is not positive")

    def test_invariant_energy_volume_relation(self):
        """
        Check the relation P = 2E/(3V) if energy is also provided.
        If energy is not in results, we skip.
        """
        E = self.results.get('E', None)
        if E is None or self.P is None:
            return (False, "SKIP: E or P not available for this check")
        V = self.results.get('V', None)
        if V is None and (self.R is not None):
            # compute V from R
            try:
                R_val = float(self.R)
                V = (4.0/3.0) * math.pi * R_val**3
            except:
                V = None
        if V is None:
            return (False, "SKIP: cannot get volume V")
        try:
            E_val = float(E)
            V_val = float(V)
            P_val = float(self.P)
        except (TypeError, ValueError):
            return (False, "Cannot convert E, V, or P to float")
        expected_P = 2.0 * E_val / (3.0 * V_val)
        if math.isclose(P_val, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "P = 2E/(3V) holds")
        else:
            return (False, f"P = {P_val:.6e} but 2E/(3V) = {expected_P:.6e}")

    # ---------------------------------------------------------------------
    # Modeling / trace tests (at least two required)
    # ---------------------------------------------------------------------
    def test_modeling_derivative_relation(self):
        """
        If dE_dR and dV_dR are available in results, check P = - (dE/dR) / (dV/dR).
        """
        dE_dR = self.results.get('dE_dR', None)
        dV_dR = self.results.get('dV_dR', None)
        if dE_dR is None or dV_dR is None or self.P is None:
            return (False, "SKIP: dE_dR, dV_dR, or P not available")
        try:
            dE_val = float(dE_dR)
            dV_val = float(dV_dR)
            P_val = float(self.P)
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float")
        expected_P = - dE_val / dV_val
        if math.isclose(P_val, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "P = - (dE/dR) / (dV/dR) holds")
        else:
            return (False, f"P = {P_val:.6e} but -dE/dR/dV/dR = {expected_P:.6e}")

    def test_modeling_k_relation(self):
        """
        If k is present, check that k = pi / R.
        """
        if self.k is None or self.R is None:
            return (False, "SKIP: k or R not available")
        try:
            k_val = float(self.k)
            R_val = float(self.R)
        except (TypeError, ValueError):
            return (False, "Cannot convert k or R to float")
        expected_k = math.pi / R_val
        if math.isclose(k_val, expected_k, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"k = π/R holds (k={k_val:.6e}, π/R={expected_k:.6e})")
        else:
            return (False, f"k={k_val:.6e} differs from π/R={expected_k:.6e}")

    def test_modeling_energy_k_relation(self):
        """
        If E and k are present, check that E = ħ² k² / (2m).
        """
        if self.E is None or self.k is None or self.hbar is None or self.m is None:
            return (False, "SKIP: E, k, hbar, m not available")
        try:
            E_val = float(self.E)
            k_val = float(self.k)
            hbar_val = float(self.hbar)
            m_val = float(self.m)
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float")
        expected_E = (hbar_val**2 * k_val**2) / (2.0 * m_val)
        if math.isclose(E_val, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E = ħ²k²/(2m) holds (E={E_val:.6e}, expected={expected_E:.6e})")
        else:
            return (False, f"E={E_val:.6e} differs from ħ²k²/(2m)={expected_E:.6e}")

    # ---------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare)
    # ---------------------------------------------------------------------
    def test_monotonicity_pressure_decreases_with_R(self):
        """
        Perturb R upward by 10% and check that expected P decreases
        (since P ∝ 1/R⁵).
        """
        if self.P is None or self.R is None or self.hbar is None or self.m is None:
            return (False, "SKIP: need P, R, hbar, m for monotonicity")
        try:
            P_orig = float(self.P)
            R_orig = float(self.R)
            hbar_val = float(self.hbar)
            m_val = float(self.m)
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float")

        # Perturb R upward
        R_new = R_orig * 1.1
        P_expected_new = (hbar_val**2 * math.pi) / (4.0 * m_val * R_new**5)
        # Since R increased, P must be strictly smaller
        if P_expected_new < P_orig:
            return (True, f"P decreases when R increases (P_orig={P_orig:.6e}, P_new={P_expected_new:.6e})")
        else:
            return (False, f"Monotonicity violated: P_orig={P_orig:.6e}, P_expected_new={P_expected_new:.6e}")

    def test_monotonicity_pressure_increases_with_hbar(self):
        """
        Perturb hbar upward by 10% and check that expected P increases
        (since P ∝ ħ²).
        """
        if self.P is None or self.R is None or self.hbar is None or self.m is None:
            return (False, "SKIP: need P, R, hbar, m for monotonicity")
        try:
            P_orig = float(self.P)
            R_val = float(self.R)
            hbar_orig = float(self.hbar)
            m_val = float(self.m)
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float")

        hbar_new = hbar_orig * 1.1
        P_expected_new = (hbar_new**2 * math.pi) / (4.0 * m_val * R_val**5)
        if P_expected_new > P_orig:
            return (True, f"P increases when hbar increases (P_orig={P_orig:.6e}, P_new={P_expected_new:.6e})")
        else:
            return (False, f"Monotonicity violated: P_orig={P_orig:.6e}, P_expected_new={P_expected_new:.6e}")

    # ---------------------------------------------------------------------
    # Weak answer sanity (at most one such method, no derived expected values)
    # ---------------------------------------------------------------------
    def test_answer_sanity_finite_and_positive(self):
        """
        Weak sanity: P is finite and positive.
        No expected/target values are computed.
        """
        if self.P is None:
            return (False, "SKIP: P is None")
        try:
            P_val = float(self.P)
        except (TypeError, ValueError):
            return (False, "P not numeric")
        if math.isinf(P_val) or math.isnan(P_val):
            return (False, "P is inf or nan")
        if P_val <= 0:
            return (False, "P ≤ 0")
        return (True, "P is finite and positive")

    # ---------------------------------------------------------------------
    # Main runner
    # ---------------------------------------------------------------------
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)