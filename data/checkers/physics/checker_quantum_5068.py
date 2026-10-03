import math
import sympy as sp

class PhysicsChecker:
    """
    PhysicsChecker for estimating magnetic susceptibility of He atom (ground state).
    Checks based on Langevin diamagnetism and physical constraints.
    """

    # Physical constants in cgs units (used in pseudocode)
    E_CGS = 4.8032047e-10       # elementary charge in esu
    M_E_CGS = 9.10938356e-28    # electron mass in g
    C_CGS = 2.99792458e10       # speed of light in cm/s
    A0_CGS = 5.29177210903e-9   # Bohr radius in cm

    def __init__(self, solution_instance):
        # Extract results key 'chi' safely
        results = getattr(solution_instance, 'results', {})
        self.chi = results.get('chi', None)

        # Extract givens if provided (optional; we can use default constants)
        givens = getattr(solution_instance, 'givens', {})
        self.e = givens.get('e', None)
        self.m_e = givens.get('m_e', None)
        self.c = givens.get('c', None)
        self.a0 = givens.get('a0', None)

        # Store explanation steps (not used directly, but available for context)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Helper: ensure value is numeric (int, float, sympy number)
    # ------------------------------------------------------------------
    def _is_numeric(self, x):
        if isinstance(x, (int, float)):
            return True
        if hasattr(x, 'is_Number') and x.is_Number:
            return True
        return False

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_chi_exists(self):
        if self.chi is None:
            return (False, "SKIP: 'chi' not found in results")
        return (True, "chi key is present")

    def test_interface_chi_numeric(self):
        if self.chi is None:
            return (False, "SKIP: chi is None")
        if not self._is_numeric(self.chi):
            return (False, f"chi has non-numeric type: {type(self.chi)}")
        return (True, "chi is numeric")

    def test_interface_chi_finite(self):
        if self.chi is None:
            return (False, "SKIP: chi is None")
        chi_val = float(self.chi)
        if math.isnan(chi_val) or math.isinf(chi_val):
            return (False, f"chi is not finite: {chi_val}")
        return (True, "chi is finite")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_diamagnetic_sign(self):
        # Ground state of He has L=0, no unpaired spins => diamagnetic (chi < 0)
        if self.chi is None:
            return (False, "SKIP: chi is None")
        try:
            chi_val = float(self.chi)
        except Exception:
            return (False, f"Cannot convert chi to float: {self.chi}")
        if chi_val >= 0:
            return (False, f"chi = {chi_val} is not negative; expected diamagnetic (<0)")
        return (True, f"chi = {chi_val} is negative => diamagnetic")

    def test_invariant_reasonable_magnitude(self):
        # He diamagnetic susceptibility per atom is extremely small (~10^-6 cgs? Actually ~10^-29 cm^3).
        # Check it's not larger than 1 in absolute value (unphysical)
        if self.chi is None:
            return (False, "SKIP: chi is None")
        try:
            chi_val = float(self.chi)
        except Exception:
            return (False, f"Cannot convert to float: {self.chi}")
        if abs(chi_val) > 1.0:
            return (False, f"chi magnitude too large: |{chi_val}| > 1, not physical")
        return (True, f"chi magnitude is reasonable: {chi_val}")

    # ------------------------------------------------------------------
    # Modeling/trace tests (quality of the derivation)
    # ------------------------------------------------------------------
    def test_modeling_langevin_formula(self):
        # Re-derive expected chi from Langevin formula using provided constants or defaults.
        # χ = - (e^2 * a0^2) / (4 * m_e * c^2)  (cgs units)
        if self.chi is None:
            return (False, "SKIP: chi is None")
        e = self.e if self.e is not None else self.E_CGS
        m_e = self.m_e if self.m_e is not None else self.M_E_CGS
        c = self.c if self.c is not None else self.C_CGS
        a0 = self.a0 if self.a0 is not None else self.A0_CGS
        expected_chi = - (e**2 * a0**2) / (4 * m_e * c**2)
        try:
            chi_val = float(self.chi)
            expected = float(expected_chi)
        except Exception as e:
            return (False, f"Cannot convert to float: {e}")
        if not math.isclose(chi_val, expected, rel_tol=1e-2, abs_tol=1e-30):
            return (False, f"chi = {chi_val} does not match expected {expected} (rel_tol=1e-2)")
        return (True, f"chi matches Langevin formula within tolerance")

    def test_modeling_derived_mean_square_radius(self):
        # From the pseudocode, the total mean square radius for two electrons is 3 a0^2 / 2.
        # Check that chi is consistent with that value.
        if self.chi is None:
            return (False, "SKIP: chi is None")
        e = self.e if self.e is not None else self.E_CGS
        m_e = self.m_e if self.m_e is not None else self.M_E_CGS
        c = self.c if self.c is not None else self.C_CGS
        a0 = self.a0 if self.a0 is not None else self.A0_CGS
        try:
            chi_val = float(self.chi)
        except Exception:
            return (False, f"Cannot convert chi to float: {self.chi}")
        # From Langevin: chi = - (e^2 / (6 m_e c^2)) * total_r2  => total_r2 = -6 m_e c^2 / e^2 * chi
        total_r2_from_chi = -6 * m_e * c**2 / e**2 * chi_val
        expected_total_r2 = 1.5 * a0**2   # 3 a0^2 / 2
        if not math.isclose(total_r2_from_chi, float(expected_total_r2), rel_tol=0.03, abs_tol=1e-40):
            return (False, f"Derived total <r^2> = {total_r2_from_chi} does not match expected {float(expected_total_r2)} (rel_tol=0.03)")
        return (True, f"Derived total <r^2> = {total_r2_from_chi} is close to {float(expected_total_r2)}")

    # Optional: test trace of explanation steps (not required but can add)
    # We skip this to keep focus on modeling.

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_chi_with_a0(self):
        # Perturb a0 upward by 1% and verify chi becomes more negative (smaller)
        if self.chi is None:
            return (False, "SKIP: chi is None")
        e = self.e if self.e is not None else self.E_CGS
        m_e = self.m_e if self.m_e is not None else self.M_E_CGS
        c = self.c if self.c is not None else self.C_CGS
        a0 = self.a0 if self.a0 is not None else self.A0_CGS
        try:
            chi_orig = float(self.chi)
            expected_orig = float(- (e**2 * a0**2) / (4 * m_e * c**2))
        except Exception as e:
            return (False, f"Cannot compute values: {e}")
        # Perturb a0 by +1%
        a0_pert = a0 * 1.01
        expected_pert = float(- (e**2 * a0_pert**2) / (4 * m_e * c**2))
        # Since a0 increases, magnitude increases => expected_pert should be smaller (more negative)
        if expected_pert >= expected_orig:  # more negative means smaller in value
            return (False, f"Monotonicity violation: perturbed chi = {expected_pert} is not more negative than original {expected_orig}")
        return (True, f"Monotonicity holds: 1% increase in a0 makes chi more negative (from {expected_orig:.3e} to {expected_pert:.3e})")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_negative(self):
        # Only check that chi is negative (no hardcoded target value)
        if self.chi is None:
            return (False, "SKIP: chi is None")
        try:
            chi_val = float(self.chi)
        except Exception:
            return (False, f"Cannot convert chi to float")
        if chi_val >= 0:
            return (False, f"chi = {chi_val} is not negative")
        return (True, f"chi is negative (diamagnetic)")

    # ------------------------------------------------------------------
    # Run all tests (discover and execute)
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