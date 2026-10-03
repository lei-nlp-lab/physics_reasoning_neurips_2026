import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for Problem: Electric dipole near a conducting sphere.
    Audits boundary conditions, induced dipole, interaction energy, and force.
    """

    def __init__(self, solution_instance):
        # Safely extract givens and results
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        # Givens (with None defaults)
        self.P = self.givens.get('P', None)          # dipole moment
        self.a = self.givens.get('a', None)          # sphere radius
        self.r = self.givens.get('r', None)          # distance from sphere center to dipole
        self.epsilon_0 = self.givens.get('epsilon_0', None)  # permittivity

        # Results (with None defaults)
        self.E_t = self.results.get('E_t', None)
        self.E_n = self.results.get('E_n', None)
        self.alpha = self.results.get('alpha', None)
        self.E_ext = self.results.get('E_ext', None)
        self.p_i = self.results.get('p_i', None)
        self.U = self.results.get('U', None)
        self.F = self.results.get('F', None)

        # Pre-compute expected expressions for modeling tests (numeric fallback)
        self._expected_alpha = None
        self._expected_E_ext = None
        self._expected_p_i = None
        self._expected_U = None
        self._expected_F_mag = None

        self._compute_expected()

    def _compute_expected(self):
        """Numerically compute expected values from givens if numeric."""
        if all(x is not None and self._is_numeric(x) for x in [self.P, self.a, self.r, self.epsilon_0]):
            P, a, r, eps = map(float, [self.P, self.a, self.r, self.epsilon_0])
            pi = math.pi
            self._expected_alpha = 4 * pi * eps * a**3
            self._expected_E_ext = P / (2 * pi * eps * r**3)
            self._expected_p_i = 2 * P * a**3 / r**3
            self._expected_U = - P**2 * a**3 / (pi * eps * r**6)
            self._expected_F_mag = 6 * P**2 * a**3 / (pi * eps * r**7)

    def _is_numeric(self, val):
        """Check if val is numeric (int, float, or sympy number)."""
        if isinstance(val, (int, float)):
            return True
        if hasattr(val, 'is_Number') and val.is_Number:
            return True
        return False

    def _to_float(self, val):
        """Convert sympy numeric to float, else return as float."""
        if isinstance(val, (int, float)):
            return float(val)
        if hasattr(val, 'evalf'):
            return float(val.evalf())
        return float(val)

    def _check_approx(self, actual, expected, rel_tol=1e-2, abs_tol=1e-2):
        """Check if actual ≈ expected using math.isclose."""
        try:
            a = self._to_float(actual)
            e = self._to_float(expected)
            return math.isclose(a, e, rel_tol=rel_tol, abs_tol=abs_tol)
        except Exception:
            return False

    def run_all_tests(self):
        """Discover and run all test methods, return (success, logs)."""
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---- Interface tests ----
    def test_interface_required_keys(self):
        required_keys = ['E_t', 'E_n', 'alpha', 'E_ext', 'p_i', 'U', 'F']
        missing = [k for k in required_keys if k not in self.results or self.results[k] is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_type_numeric(self):
        keys = ['E_t', 'E_n', 'alpha', 'E_ext', 'p_i', 'U', 'F']
        bad = []
        for k in keys:
            val = self.results.get(k)
            if val is None:
                bad.append(f"{k} is None")
            elif not self._is_numeric(val):
                bad.append(f"{k} is not numeric (type {type(val).__name__})")
        if bad:
            return (False, "; ".join(bad))
        return (True, "All result values are numeric")

    # ---- Invariant tests (theorem-based) ----
    def test_invariant_E_t_zero(self):
        """Tangential electric field on a conductor surface must be zero."""
        if self.E_t is None:
            return (False, "SKIP: E_t not available")
        if not self._is_numeric(self.E_t):
            return (False, "E_t is not numeric")
        try:
            e_t = self._to_float(self.E_t)
            if math.isclose(e_t, 0.0, abs_tol=1e-12):
                return (True, "E_t ≈ 0 (conductor surface tangential field zero)")
            else:
                return (False, f"E_t = {e_t}, expected 0")
        except Exception as e:
            return (False, f"Error checking E_t: {e}")

    def test_invariant_induced_dipole_relation(self):
        """Induced dipole p_i must equal α * E_ext (polarization relation)."""
        if any(x is None for x in [self.p_i, self.alpha, self.E_ext]):
            return (False, "SKIP: p_i, alpha, or E_ext not available")
        if not all(self._is_numeric(x) for x in [self.p_i, self.alpha, self.E_ext]):
            return (False, "Non-numeric values for invariant check")
        try:
            # Compute product α * E_ext
            prod = self._to_float(self.alpha) * self._to_float(self.E_ext)
            actual = self._to_float(self.p_i)
            if math.isclose(actual, prod, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"p_i ≈ α * E_ext: actual={actual:.4e}, product={prod:.4e}")
            else:
                return (False, f"p_i = {actual}, but α*E_ext = {prod}, mismatch")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_invariant_energy_force_relation(self):
        """Force F should equal -dU/dr (numerical finite difference)."""
        if any(x is None for x in [self.U, self.F, self.r]):
            return (False, "SKIP: U, F, or r not available")
        if not all(self._is_numeric(x) for x in [self.U, self.F, self.r]):
            return (False, "Non-numeric values for invariant check")
        try:
            U = self._to_float(self.U)
            r = self._to_float(self.r)
            F = self._to_float(self.F)
            # Small perturbation Δr = 1% of r
            dr = 0.01 * r
            r_up = r + dr
            # Estimate U at r_up using physics formula (assumes givens available)
            if not all(self._is_numeric(x) for x in [self.P, self.a, self.epsilon_0]):
                return (False, "SKIP: numeric givens needed for energy derivative")
            P = self._to_float(self.P)
            a = self._to_float(self.a)
            eps = self._to_float(self.epsilon_0)
            U_up = - P**2 * a**3 / (math.pi * eps * r_up**6)
            dU_dr_num = (U_up - U) / dr
            # expect F ≈ - dU/dr
            if math.isclose(-dU_dr_num, F, rel_tol=1e-1, abs_tol=1e-1):
                return (True, f"F ≈ -dU/dr: numerical derivative gives {-dU_dr_num:.4e}, F={F:.4e}")
            else:
                return (False, f"F = {F}, but -dU/dr ≈ {-dU_dr_num:.4e}")
        except Exception as e:
            return (False, f"Error: {e}")

    # ---- Modeling tests (check key physics relationships) ----
    def test_modeling_alpha(self):
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        if self._expected_alpha is None:
            return (False, "SKIP: cannot compute expected alpha (non-numeric givens)")
        if self._check_approx(self.alpha, self._expected_alpha):
            return (True, f"alpha ≈ {self._expected_alpha:.4e}")
        else:
            return (False, f"alpha = {self.alpha}, expected ~{self._expected_alpha:.4e}")

    def test_modeling_E_ext(self):
        if self.E_ext is None:
            return (False, "SKIP: E_ext not available")
        if self._expected_E_ext is None:
            return (False, "SKIP: cannot compute expected E_ext (non-numeric givens)")
        if self._check_approx(self.E_ext, self._expected_E_ext):
            return (True, f"E_ext ≈ {self._expected_E_ext:.4e}")
        else:
            return (False, f"E_ext = {self.E_ext}, expected ~{self._expected_E_ext:.4e}")

    def test_modeling_p_i(self):
        if self.p_i is None:
            return (False, "SKIP: p_i not available")
        if self._expected_p_i is None:
            return (False, "SKIP: cannot compute expected p_i (non-numeric givens)")
        if self._check_approx(self.p_i, self._expected_p_i):
            return (True, f"p_i ≈ {self._expected_p_i:.4e}")
        else:
            return (False, f"p_i = {self.p_i}, expected ~{self._expected_p_i:.4e}")

    def test_modeling_U(self):
        if self.U is None:
            return (False, "SKIP: U not available")
        if self._expected_U is None:
            return (False, "SKIP: cannot compute expected U (non-numeric givens)")
        if self._check_approx(self.U, self._expected_U):
            return (True, f"U ≈ {self._expected_U:.4e}")
        else:
            return (False, f"U = {self.U}, expected ~{self._expected_U:.4e}")

    def test_modeling_F_magnitude(self):
        if self.F is None:
            return (False, "SKIP: F not available")
        if self._expected_F_mag is None:
            return (False, "SKIP: cannot compute expected |F| (non-numeric givens)")
        try:
            f_mag = abs(self._to_float(self.F))
            if self._check_approx(f_mag, self._expected_F_mag):
                return (True, f"|F| ≈ {self._expected_F_mag:.4e}")
            else:
                return (False, f"|F| = {f_mag}, expected ~{self._expected_F_mag:.4e}")
        except Exception as e:
            return (False, f"Error computing |F|: {e}")

    # ---- Monotonicity tests (explicit perturbation, multiple steps) ----
    def test_monotonicity_U_increases_with_r(self):
        """U becomes less negative (increases) as r increases. Check at 2% and 5% perturbations."""
        if any(x is None for x in [self.P, self.a, self.r, self.epsilon_0, self.U]):
            return (False, "SKIP: missing givens or U for monotonicity test")
        if not all(self._is_numeric(x) for x in [self.P, self.a, self.r, self.epsilon_0, self.U]):
            return (False, "Non-numeric values for monotonicity")
        try:
            P, a, r, eps = map(self._to_float, [self.P, self.a, self.r, self.epsilon_0])
            U_base = self._to_float(self.U)
            # Two perturbations: +2% and +5%
            for percent in [2, 5]:
                r_pert = r * (1 + percent/100.0)
                expected_U_pert = - P**2 * a**3 / (math.pi * eps * r_pert**6)
                # Check that perturbed U is larger (less negative)
                if not (expected_U_pert > U_base):
                    return (False, f"Monotonicity failed at {percent}% increase: U_base={U_base:.4e}, U_pert={expected_U_pert:.4e}")
            return (True, f"U increases with r: base at r={r}, +2% and +5% both give higher U")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_monotonicity_F_magnitude_increases_with_P(self):
        """|F| increases as dipole moment P increases. Check at 5% and 10% perturbations."""
        if any(x is None for x in [self.P, self.a, self.r, self.epsilon_0, self.F]):
            return (False, "SKIP: missing givens or F for monotonicity test")
        if not all(self._is_numeric(x) for x in [self.P, self.a, self.r, self.epsilon_0, self.F]):
            return (False, "Non-numeric values for monotonicity")
        try:
            a, r, eps = map(self._to_float, [self.a, self.r, self.epsilon_0])
            P_base = self._to_float(self.P)
            F_mag_base = abs(self._to_float(self.F))
            for percent in [5, 10]:
                P_pert = P_base * (1 + percent/100.0)
                expected_F_mag_pert = 6 * P_pert**2 * a**3 / (math.pi * eps * r**7)
                if not (expected_F_mag_pert > F_mag_base):
                    return (False, f"Monotonicity failed at {percent}% increase: base|F|={F_mag_base:.4e}, perturbed|F|={expected_F_mag_pert:.4e}")
            return (True, f"|F| increases with P: base at P={P_base}, +5% and +10% both give larger |F|")
        except Exception as e:
            return (False, f"Error: {e}")

    # ---- Answer sanity (weak) ----
    def test_answer_sanity_U_negative(self):
        """U should be negative for aligned dipoles in this configuration."""
        if self.U is None:
            return (False, "SKIP: U not available")
        if not self._is_numeric(self.U):
            return (False, "U is not numeric")
        try:
            u = self._to_float(self.U)
            if u < 0:
                return (True, f"U = {u} < 0 (negative)")
            else:
                return (False, f"U = {u}, expected negative")
        except Exception as e:
            return (False, f"Error: {e}")