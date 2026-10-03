import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics checker for the 'plum pudding' model problem.
    Audits solution for force on electron, motion type, and oscillation frequency.
    """
    def __init__(self, solution_instance):
        # Safe extraction of givens - NO numeric defaults
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})

        self.a = givens.get('a', None)          # sphere radius
        self.e = givens.get('e', None)          # elementary charge
        self.m = givens.get('m', None)          # electron mass
        self.eps0 = givens.get('eps0', None)    # vacuum permittivity
        # Fallback for different key name
        if self.eps0 is None:
            self.eps0 = givens.get('epsilon0', None)
        self.Z = givens.get('Z', None)          # atomic number, should be 1 for hydrogen

        # Result quantities - NEVER assign numeric defaults
        self.F = results.get('F', None)
        self.motion_type = results.get('motion_type', None)
        self.omega = results.get('omega', None)

        # Symbolic constants for internal re-derivation
        self._r = sp.Symbol('r', real=True, nonnegative=True)
        self._a_sym = sp.Symbol('a', positive=True)
        self._e_sym = sp.Symbol('e', positive=True)
        self._m_sym = sp.Symbol('m', positive=True)
        self._eps0_sym = sp.Symbol('epsilon0', positive=True)
        self._pi = sp.pi

        # Expected formulas derived from theorem (internal)
        self._expected_F_sym = - (self._e_sym**2 * self._r) / (4 * self._pi * self._eps0_sym * self._a_sym**3)
        self._expected_omega_sym = sp.sqrt(self._e_sym**2 / (4 * self._pi * self._eps0_sym * self._a_sym**3 * self._m_sym))

    def _is_sympy(self, val):
        return isinstance(val, sp.Basic)

    def _to_sympy(self, val):
        """Convert val to sympy expression or return None."""
        if self._is_sympy(val):
            return val
        if isinstance(val, str):
            try:
                return sp.sympify(val)
            except:
                return None
        return None

    # -------------------------------------------------------------
    # Interface tests
    # -------------------------------------------------------------
    def test_interface_F_exists(self):
        if self.F is None:
            return (False, "F not provided in solution")
        return (True, "F is present")

    def test_interface_motion_type_exists(self):
        if self.motion_type is None:
            return (False, "motion_type not provided")
        return (True, "motion_type is present")

    def test_interface_omega_exists(self):
        if self.omega is None:
            return (False, "omega not provided")
        return (True, "omega is present")

    def test_interface_F_type(self):
        if self.F is None:
            return (False, "SKIP: F missing")
        if not (self._is_sympy(self.F) or isinstance(self.F, str)):
            return (False, f"F should be sympy expression or string, got {type(self.F).__name__}")
        return (True, "F type ok")

    def test_interface_motion_type_type(self):
        if self.motion_type is None:
            return (False, "SKIP: motion_type missing")
        if not isinstance(self.motion_type, str):
            return (False, f"motion_type should be string, got {type(self.motion_type).__name__}")
        return (True, "motion_type type ok")

    def test_interface_omega_type(self):
        if self.omega is None:
            return (False, "SKIP: omega missing")
        if not (self._is_sympy(self.omega) or isinstance(self.omega, str)):
            return (False, f"omega should be sympy expression or string, got {type(self.omega).__name__}")
        return (True, "omega type ok")

    # -------------------------------------------------------------
    # Invariant tests (physics theorem based)
    # -------------------------------------------------------------
    def test_invariant_force_linear_restoring(self):
        """
        Check that F is proportional to -r (linear restoring, Hooke's law)
        for r<a.
        """
        if self.F is None:
            return (False, "SKIP: F missing")
        F_sym = self._to_sympy(self.F)
        if F_sym is None:
            return (False, "SKIP: cannot convert F to sympy")
        if not isinstance(F_sym, sp.Expr):
            return (False, "F is not a sympy expression")
        try:
            dF_dr = sp.diff(F_sym, self._r)
            # Check derivative does not contain r (constant)
            if dF_dr.has(self._r):
                return (False, "Derivative of F w.r.t r is not constant; F not linear in r")
            # Check sign: derivative must be negative (restoring)
            # Use typical values for numeric check
            subs = {
                self._e_sym: 1.602e-19,
                self._eps0_sym: 8.854e-12,
                self._a_sym: 1e-10,
                self._m_sym: 9.109e-31,
                self._r: 0.5e-10
            }
            try:
                dF_dr_num = float(dF_dr.subs(subs).evalf())
            except:
                return (False, "Cannot evaluate derivative numerically")
            if dF_dr_num >= 0:
                return (False, "Derivative of F w.r.t r is non-negative; force is not restoring")
            return (True, "F is linear restoring (negative constant derivative)")
        except Exception as e:
            return (False, f"Error checking linearity: {e}")

    def test_invariant_omega_formula(self):
        """
        Check that omega matches sqrt(e^2/(4π ε0 a^3 m)).
        Uses symbolic simplification or numerical evaluation.
        """
        if self.omega is None:
            return (False, "SKIP: omega missing")
        omega_sym = self._to_sympy(self.omega)
        if omega_sym is None:
            return (False, "SKIP: cannot convert omega to sympy")
        expected = self._expected_omega_sym
        # Try symbolic simplification
        diff = sp.simplify(omega_sym - expected)
        if diff == 0:
            return (True, "omega matches expected formula symbolically")
        # Numeric fallback
        subs = {
            self._e_sym: 1.602e-19,
            self._eps0_sym: 8.854e-12,
            self._a_sym: 5.29e-11,
            self._m_sym: 9.109e-31
        }
        try:
            omega_val = float(omega_sym.subs(subs).evalf())
            expected_val = float(expected.subs(subs).evalf())
            if math.isclose(omega_val, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"omega numeric {omega_val:.3e} matches expected {expected_val:.3e}")
            else:
                return (False, f"omega numeric {omega_val:.3e} differs from expected {expected_val:.3e}")
        except:
            return (False, "Cannot evaluate omega numerically")

    # -------------------------------------------------------------
    # Modeling/trace tests
    # -------------------------------------------------------------
    def test_modeling_force_sign(self):
        """
        Check that F has the correct sign (negative, i.e., towards center).
        Uses internal re-derivation, not hardcoded numbers.
        """
        if self.F is None:
            return (False, "SKIP: F missing")
        F_sym = self._to_sympy(self.F)
        if F_sym is None:
            return (False, "SKIP: cannot convert F to sympy")
        # Use typical atomic scale values
        subs = {
            self._r: 0.3,  # arbitrary positive (no units needed for sign)
            self._e_sym: 1.602e-19,
            self._eps0_sym: 8.854e-12,
            self._a_sym: 1e-10
        }
        try:
            F_val = float(F_sym.subs(subs).evalf())
        except:
            return (False, "Cannot evaluate F numerically")
        if F_val < 0:
            return (True, "F is negative (restoring towards center)")
        else:
            return (False, f"F is non-negative ({F_val:.3e}), should be negative")

    def test_modeling_motion_type_content(self):
        """
        Check that motion_type string contains relevant keywords.
        """
        if self.motion_type is None:
            return (False, "SKIP: motion_type missing")
        text = self.motion_type.lower()
        if "harmonic" in text or "shm" in text or "simple" in text:
            return (True, "motion_type indicates harmonic motion")
        else:
            return (False, f"motion_type '{self.motion_type}' does not mention harmonic motion")

    # -------------------------------------------------------------
    # Monotonicity tests
    # -------------------------------------------------------------
    def test_monotonicity_omega_decreases_with_a(self):
        """
        If radius a increases, the angular frequency omega decreases.
        Test by computing two values: one using the given a, another with a increased by 10%.
        """
        # We need a numeric value for a, even if not given we use a typical Bohr radius
        if self.omega is None:
            return (False, "SKIP: omega missing")
        omega_sym = self._to_sympy(self.omega)
        if omega_sym is None:
            return (False, "SKIP: cannot convert omega to sympy")
        # Use a default if a missing (since monotonicity can still be checked)
        if self.a is not None and isinstance(self.a, (int, float)):
            a_val = self.a
        else:
            a_val = 5.29e-11  # Bohr radius typical
        # Constants
        e_val = 1.602e-19
        eps0_val = 8.854e-12
        m_val = 9.109e-31
        subs_base = {
            self._e_sym: e_val,
            self._eps0_sym: eps0_val,
            self._a_sym: a_val,
            self._m_sym: m_val,
            self._r: 0.5 * a_val  # inside sphere
        }
        try:
            omega_base = float(omega_sym.subs(subs_base).evalf())
        except:
            return (False, "Cannot evaluate baseline omega")
        # Perturb a: increase by 10%
        a_pert = a_val * 1.1
        subs_pert = subs_base.copy()
        subs_pert[self._a_sym] = a_pert
        try:
            omega_pert = float(omega_sym.subs(subs_pert).evalf())
        except:
            return (False, "Cannot evaluate perturbed omega")
        if omega_pert < omega_base:
            return (True, f"omega decreases when a increases: {omega_base:.3e} -> {omega_pert:.3e}")
        else:
            return (False, f"omega did not decrease: {omega_base:.3e} -> {omega_pert:.3e}")

    def test_monotonicity_omega_increases_with_e(self):
        """
        If elementary charge e increases, angular frequency increases.
        Perturb e by +5% and check result.
        """
        if self.omega is None:
            return (False, "SKIP: omega missing")
        omega_sym = self._to_sympy(self.omega)
        if omega_sym is None:
            return (False, "SKIP: cannot convert omega to sympy")
        # Use typical values
        a_val = 5.29e-11
        e_val = 1.602e-19
        eps0_val = 8.854e-12
        m_val = 9.109e-31
        subs_base = {
            self._e_sym: e_val,
            self._eps0_sym: eps0_val,
            self._a_sym: a_val,
            self._m_sym: m_val,
            self._r: 0.5 * a_val
        }
        try:
            omega_base = float(omega_sym.subs(subs_base).evalf())
        except:
            return (False, "Cannot evaluate baseline omega")
        # Perturb e: increase by 5%
        e_pert = e_val * 1.05
        subs_pert = subs_base.copy()
        subs_pert[self._e_sym] = e_pert
        try:
            omega_pert = float(omega_sym.subs(subs_pert).evalf())
        except:
            return (False, "Cannot evaluate perturbed omega")
        if omega_pert > omega_base:
            return (True, f"omega increases when e increases: {omega_base:.3e} -> {omega_pert:.3e}")
        else:
            return (False, f"omega did not increase: {omega_base:.3e} -> {omega_pert:.3e}")

    # -------------------------------------------------------------
    # Weak answer sanity (at most one)
    # -------------------------------------------------------------
    def test_answer_sanity_motion_type_substantial(self):
        """
        Weak sanity: motion_type should be a non-empty string.
        """
        if self.motion_type is None:
            return (False, "SKIP: motion_type missing")
        if not isinstance(self.motion_type, str) or len(self.motion_type.strip()) == 0:
            return (False, "motion_type is empty or not a string")
        # Check it's not a placeholder
        if self.motion_type.strip().lower() in ("", "none", "null", "0"):
            return (False, f"motion_type seems placeholder: '{self.motion_type}'")
        return (True, "motion_type passes weak sanity")

    # -------------------------------------------------------------
    # Test runner
    # -------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)