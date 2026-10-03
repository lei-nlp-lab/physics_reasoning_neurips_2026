import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics checker for the relativistic beam problem.
    Validates interface, physical invariants, modeling quality, monotonicity,
    and a weak answer sanity check.
    """

    def __init__(self, solution_instance):
        # Safe extraction of results
        results = getattr(solution_instance, 'results', {}) or {}
        givens = getattr(solution_instance, 'givens', {}) or {}

        # Givens (inputs) – all defaults must be None
        self.r = givens.get('r', None)
        self.a = givens.get('a', None)
        self.rho0 = givens.get('rho0', None)          # rest frame density
        self.rho = givens.get('rho', None)            # lab frame density
        self.v = givens.get('v', None)
        self.c = givens.get('c', None)                # speed of light – no default
        self.epsilon0 = givens.get('epsilon0', None)  # permittivity – no default
        self.e_mag = givens.get('e', None)            # magnitude of electron charge – no default
        self.n = givens.get('n', None)                # number density in cm^-3

        # Derived gamma if v and c available
        if self.v is not None and self.c is not None and self.c != 0:
            try:
                self.gamma = 1.0 / math.sqrt(1.0 - (float(self.v) / float(self.c))**2)
            except (ValueError, ZeroDivisionError):
                self.gamma = None
        else:
            self.gamma = None

        # Results (as emitted by pseudocode) – defaults must be None
        self.F_a = results.get('F_a', None)
        self.F_lab = results.get('F_lab', None)
        self.F_rest = results.get('F_rest', None)
        self.g_final = results.get('g_final', None)

    # ------------------------------------------------------------------
    # Helper methods (not tests)
    # ------------------------------------------------------------------
    def _is_valid_number(self, x):
        """Check if x is a finite number (int, float, or SymPy numeric)."""
        if x is None:
            return False
        if isinstance(x, (int, float)):
            return math.isfinite(x)
        if isinstance(x, sp.Expr):
            try:
                float_x = float(x)
                return math.isfinite(float_x)
            except:
                return False
        return False

    def _sign_of(self, x):
        """Return sign: 1, -1, or 0. Works for numeric and symbolic."""
        try:
            val = float(x)
            if val > 0:
                return 1
            elif val < 0:
                return -1
            else:
                return 0
        except:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_results_keys(self):
        """Ensure all required result keys exist."""
        if not hasattr(self, 'F_a') or self.F_a is None:
            return (False, "Missing result key 'F_a'")
        if not hasattr(self, 'F_lab') or self.F_lab is None:
            return (False, "Missing result key 'F_lab'")
        if not hasattr(self, 'F_rest') or self.F_rest is None:
            return (False, "Missing result key 'F_rest'")
        if not hasattr(self, 'g_final') or self.g_final is None:
            return (False, "Missing result key 'g_final'")
        return (True, "All required result keys present")

    def test_interface_result_types(self):
        """Check that result values are numeric (int, float, or SymPy numbers)."""
        try:
            for name, val in [('F_a', self.F_a), ('F_lab', self.F_lab),
                              ('F_rest', self.F_rest), ('g_final', self.g_final)]:
                if not self._is_valid_number(val):
                    return (False, f"Result '{name}' is not a valid finite number")
            return (True, "All results are finite numbers")
        except Exception as e:
            return (False, f"Type check crashed: {e}")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_force_sign_F_a(self):
        """Check that F_a is negative (repulsive outward for electron)."""
        if not self._is_valid_number(self.F_a):
            return (False, "SKIP: F_a not available")
        if self.F_a > 0:
            return (False, "F_a should be negative (repulsive outward for electron)")
        return (True, "F_a has correct sign (negative)")

    def test_invariant_force_relation_lab_vs_rest(self):
        """Check F_rest = gamma * F_lab (from relativistic transformation)."""
        if not all(self._is_valid_number(x) for x in [self.F_lab, self.F_rest, self.gamma]):
            return (False, "SKIP: Missing F_lab, F_rest, or gamma")
        expected = self.gamma * float(self.F_lab)
        if not math.isclose(float(self.F_rest), expected, rel_tol=1e-2, abs_tol=1e-12):
            return (False, f"F_rest = {self.F_rest}, but gamma*F_lab = {expected}")
        return (True, "F_rest ≈ gamma * F_lab")

    def test_invariant_force_magnitude_ordering(self):
        """Check |F_rest| > |F_lab| for gamma > 1."""
        if not all(self._is_valid_number(x) for x in [self.F_lab, self.F_rest]):
            return (False, "SKIP: Missing F_lab or F_rest")
        if self.gamma is None or self.gamma <= 1:
            return (False, "SKIP: gamma not >1")
        if abs(float(self.F_rest)) <= abs(float(self.F_lab)):
            return (False, "|F_rest| should be > |F_lab| for gamma>1")
        return (True, "|F_rest| > |F_lab| as expected")

    def test_invariant_gradient_dependence(self):
        """Check that g_final ∝ n / (ε0 γ² v) at fixed constants."""
        if not self._is_valid_number(self.g_final):
            return (False, "SKIP: g_final not available")
        if any(x is None for x in [self.e_mag, self.n, self.epsilon0, self.gamma, self.v]):
            return (False, "SKIP: missing parameters to recompute g_final")
        try:
            # Re-derive expected gradient (in Gs/cm) from givens
            expected = (self.e_mag * self.n) / (2 * self.epsilon0 * self.gamma**2 * self.v) * 100
            if not math.isclose(float(self.g_final), float(expected), rel_tol=1e-2, abs_tol=1e-6):
                return (False, f"g_final = {self.g_final}, but expected {expected}")
        except Exception as e:
            return (False, f"Recompute g_final crashed: {e}")
        return (True, "g_final matches analytic expression")

    # ------------------------------------------------------------------
    # Modeling/trace tests (quality checks based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_force_lab_expression(self):
        """Check that F_lab = -e * rho * r / (2 * epsilon0 * gamma^2)."""
        if not all(self._is_valid_number(x) for x in [self.F_lab, self.r, self.rho, self.epsilon0, self.gamma, self.e_mag]):
            return (False, "SKIP: missing variables for F_lab check")
        try:
            expected = -self.e_mag * self.rho * self.r / (2 * self.epsilon0 * self.gamma**2)
            if not math.isclose(float(self.F_lab), float(expected), rel_tol=1e-2, abs_tol=1e-12):
                return (False, f"F_lab = {self.F_lab}, but expected {expected}")
        except Exception as e:
            return (False, f"F_lab calculation error: {e}")
        return (True, "F_lab expression matches theoretical derivation")

    def test_modeling_force_rest_expression(self):
        """Check that F_rest = -e * rho0 * r / (2 * epsilon0) (or equivalently -e*rho*r/(2*epsilon0*gamma))."""
        if not all(self._is_valid_number(x) for x in [self.F_rest, self.r, self.rho, self.rho0, self.epsilon0, self.e_mag]):
            return (False, "SKIP: missing variables for F_rest check")
        try:
            # Using rho0 = rho / gamma
            expected = -self.e_mag * self.rho0 * self.r / (2 * self.epsilon0)
            if not math.isclose(float(self.F_rest), float(expected), rel_tol=1e-2, abs_tol=1e-12):
                return (False, f"F_rest = {self.F_rest}, but expected {expected}")
        except Exception as e:
            return (False, f"F_rest calculation error: {e}")
        return (True, "F_rest expression matches theoretical derivation")

    def test_modeling_gradient_expression(self):
        """Check that g_final = (e * n) / (2 * epsilon0 * gamma^2 * v) * 100."""
        if not self._is_valid_number(self.g_final):
            return (False, "SKIP: g_final not available")
        if any(x is None for x in [self.e_mag, self.n, self.epsilon0, self.gamma, self.v]):
            return (False, "SKIP: missing parameters")
        try:
            expected = (self.e_mag * self.n) / (2 * self.epsilon0 * self.gamma**2 * self.v) * 100
            if not math.isclose(float(self.g_final), float(expected), rel_tol=1e-2, abs_tol=1e-6):
                return (False, f"g_final = {self.g_final}, but expected {expected}")
        except Exception as e:
            return (False, f"g_final expression error: {e}")
        return (True, "g_final expression matches derivation")

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks via perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_force_increases_with_r(self):
        """Check that increasing r leads to larger |F_lab| (or |F_a|)."""
        # Use F_lab, but if not available fallback to F_a
        force = self.F_lab if self._is_valid_number(self.F_lab) else self.F_a
        if not self._is_valid_number(force):
            return (False, "SKIP: no valid force available")
        r_base = self.r
        if r_base is None or not self._is_valid_number(r_base):
            return (False, "SKIP: r not available")
        # Perturb r by +10%
        r_pert = float(r_base) * 1.1
        # Recompute force for perturbed r using the analytic form (if possible)
        if self.rho is not None and self.epsilon0 is not None and self.gamma is not None and self.e_mag is not None:
            try:
                force_base_abs = abs(float(force))
                force_pert = -self.e_mag * self.rho * r_pert / (2 * self.epsilon0 * self.gamma**2)
                force_pert_abs = abs(float(force_pert))
            except (ValueError, TypeError):
                return (False, "Cannot compute perturbed force")
            if force_pert_abs <= force_base_abs:
                return (False, "Increasing r did not increase |F_lab|")
            return (True, "|F_lab| increases with r as expected")
        else:
            return (False, "SKIP: missing parameters for monotonicity check")

    def test_monotonicity_gradient_increases_with_n(self):
        """Check that increasing n (density) increases |g_final|."""
        if not self._is_valid_number(self.g_final):
            return (False, "SKIP: g_final not available")
        n_base = self.n
        if n_base is None:
            return (False, "SKIP: n not available")
        if any(x is None for x in [self.e_mag, self.epsilon0, self.gamma, self.v]):
            return (False, "SKIP: missing parameters")
        try:
            g_base = float(self.g_final)
            n_pert = float(n_base) * 1.1
            g_pert = (self.e_mag * n_pert) / (2 * self.epsilon0 * self.gamma**2 * self.v) * 100
            if abs(g_pert) <= abs(g_base):
                return (False, "Increasing n did not increase |g_final|")
        except Exception as e:
            return (False, f"Perturbation error: {e}")
        return (True, "|g_final| increases with n as expected")

    # ------------------------------------------------------------------
    # Weak answer sanity check (only sign/range)
    # ------------------------------------------------------------------
    def test_answer_sanity_gradient_magnitude(self):
        """Broad magnitude check: g_final should be around 1 Gs/cm for typical numbers."""
        if not self._is_valid_number(self.g_final):
            return (False, "SKIP: g_final not available")
        try:
            g = float(self.g_final)
            if g <= 0:
                return (False, "g_final should be positive")
            if g < 0.1 or g > 10:
                # For typical numbers (n=2e10/cm^3, v=0.99c), it's ~1.21
                return (False, f"g_final = {g} outside expected range [0.1, 10] Gs/cm")
        except Exception as e:
            return (False, f"g_final conversion error: {e}")
        return (True, "g_final magnitude in plausible range")

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)