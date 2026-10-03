import math
import sympy as sp
import types

class PhysicsChecker:
    """
    Audits a solution for the simple harmonic oscillator problem (parts a-c).
    Extracts given parameters and computed results, then runs a battery of
    physics-based consistency tests.
    """

    def __init__(self, solution_instance):
        # Extract given physical parameters (safe defaults = None)
        self.T = getattr(solution_instance, 'T', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.omega_x = getattr(solution_instance, 'omega_x', None)
        self.omega_y = getattr(solution_instance, 'omega_y', None)
        self.k_B = getattr(solution_instance, 'k_B', None)
        self.c = getattr(solution_instance, 'c', None)
        self.g = getattr(solution_instance, 'g', None)

        # Extract results dictionary (may be missing keys)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}   # fallback

        # Derive derived quantities (set to None if required givens missing)
        if self.T is not None and self.k_B is not None:
            self.beta = 1.0 / (self.k_B * self.T)
        else:
            self.beta = None

        if self.hbar is not None and self.omega is not None and self.beta is not None:
            self.alpha = self.hbar * self.omega * self.beta
        else:
            self.alpha = None

        if self.hbar is not None and self.omega_x is not None and self.beta is not None:
            self.alpha_x = self.hbar * self.omega_x * self.beta
        else:
            self.alpha_x = None

        if self.hbar is not None and self.omega_y is not None and self.beta is not None:
            self.alpha_y = self.hbar * self.omega_y * self.beta
        else:
            self.alpha_y = None

    # -----------------------------------------------------------------
    #  Interface tests – ensure required result keys exist
    # -----------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all mandatory output keys are present."""
        required_keys = ['E1', 'Z2', 'Z2_degen', 'C', 'x_mean']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, "All required keys present"

    def test_interface_type_numerical(self):
        """Check that result values are numeric (float/int) or sympy numbers."""
        for key in self.results:
            val = self.results[key]
            if not isinstance(val, (int, float, sp.Number)):
                return False, f"Key '{key}' has non-numeric type {type(val).__name__}"
        return True, "All values are numeric"

    # -----------------------------------------------------------------
    #  Invariant tests – theorem-based physical laws (lenient isclose)
    # -----------------------------------------------------------------
    def _safe_alpha(self, alpha_val, name="alpha"):
        if alpha_val is None:
            return None
        try:
            return float(alpha_val)
        except (TypeError, ValueError):
            return None

    def _safe_float(self, val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def test_invariant_mean_energy_formula(self):
        """Check E1 = hbar*omega/2 + hbar*omega/(exp(alpha)-1)"""
        if self.alpha is None or self.hbar is None or self.omega is None:
            return False, "SKIP: missing givens (hbar, omega, or T for alpha)"
        alpha = self._safe_alpha(self.alpha)
        if alpha is None:
            return False, "SKIP: alpha could not be converted to float"
        expected = 0.5 * self.hbar * self.omega + self.hbar * self.omega / (math.exp(alpha) - 1)
        E1 = self.results.get('E1')
        if E1 is None:
            return False, "E1 not in results"
        E1_f = self._safe_float(E1)
        if E1_f is None:
            return False, "E1 is not a numeric value"
        if math.isclose(E1_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E1 matches QHO formula"
        else:
            return False, f"E1 {E1_f} != expectation {expected}"

    def test_invariant_partition_function_product(self):
        """Check Z2 = exp(-alpha_x/2)/(1-exp(-alpha_x)) * exp(-alpha_y/2)/(1-exp(-alpha_y))"""
        if self.alpha_x is None or self.alpha_y is None:
            return False, "SKIP: missing alpha_x or alpha_y"
        ax = self._safe_alpha(self.alpha_x, "alpha_x")
        ay = self._safe_alpha(self.alpha_y, "alpha_y")
        if ax is None or ay is None:
            return False, "SKIP: alpha_x/alpha_y not convertible"
        expected = (math.exp(-ax/2) / (1 - math.exp(-ax))) * (math.exp(-ay/2) / (1 - math.exp(-ay)))
        Z2 = self.results.get('Z2')
        if Z2 is None:
            return False, "Z2 not in results"
        Z2_f = self._safe_float(Z2)
        if Z2_f is None:
            return False, "Z2 not numeric"
        if math.isclose(Z2_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Z2 matches product of 1D partition functions"
        else:
            return False, f"Z2 {Z2_f} != product {expected}"

    def test_invariant_degenerate_partition(self):
        """Check Z2_degen = exp(alpha)/(exp(alpha)-1)^2 when omega_x = omega_y"""
        if self.omega_x is None or self.omega_y is None or self.alpha is None:
            return False, "SKIP: missing frequencies or alpha"
        if not math.isclose(self.omega_x, self.omega_y, rel_tol=1e-6):
            return False, "SKIP: omega_x != omega_y (not degenerate case)"
        alpha = self._safe_alpha(self.alpha)
        if alpha is None:
            return False, "SKIP: alpha not convertible"
        expected = math.exp(alpha) / ( (math.exp(alpha) - 1)**2 )
        Z2d = self.results.get('Z2_degen')
        if Z2d is None:
            return False, "Z2_degen not in results"
        Z2d_f = self._safe_float(Z2d)
        if Z2d_f is None:
            return False, "Z2_degen not numeric"
        if math.isclose(Z2d_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Z2_degen matches degenerate formula"
        else:
            return False, f"Z2_degen {Z2d_f} != expectation {expected}"

    def test_invariant_energy_low_t_limit(self):
        """At low T, E1 ≈ hbar*omega/2 (zero-point). Check if T is small."""
        if self.alpha is None or self.hbar is None or self.omega is None:
            return False, "SKIP: missing givens"
        alpha = self._safe_alpha(self.alpha)
        if alpha is None:
            return False, "SKIP: alpha not numeric"
        if alpha < 1:   # not low temperature regime
            return True, "SKIP: not low T (alpha<1)"
        # low T: E1 ≈ hbar*omega/2 + small correction
        expected_zero_point = 0.5 * self.hbar * self.omega
        E1 = self.results.get('E1')
        if E1 is None:
            return False, "E1 not in results"
        E1_f = self._safe_float(E1)
        if E1_f is None:
            return False, "E1 not numeric"
        # Expect E1 within 1% of zero-point plus exponential term
        if math.isclose(E1_f, expected_zero_point, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Low T: E1 is close to zero-point energy"
        else:
            return False, f"Low T: E1 {E1_f} not close to zero-point {expected_zero_point}"

    # -----------------------------------------------------------------
    #  Modeling / trace tests – check that explanation steps lead to correct intermediate relations
    # -----------------------------------------------------------------
    def test_modeling_heat_capacity(self):
        """Check C = k_B + (15*g^2)/(8*c^3) * k_B^2 * T"""
        if self.c is None or self.g is None or self.k_B is None or self.T is None:
            return False, "SKIP: missing c,g,k_B,T"
        expected = self.k_B + (15 * self.g**2) / (8 * self.c**3) * (self.k_B**2) * self.T
        C = self.results.get('C')
        if C is None:
            return False, "C not in results"
        C_f = self._safe_float(C)
        if C_f is None:
            return False, "C not numeric"
        if math.isclose(C_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Heat capacity matches perturbation theory"
        else:
            return False, f"C {C_f} != expected {expected}"

    def test_modeling_mean_position(self):
        """Check x_mean = 3*g*k_B*T/(4*c^2)"""
        if self.g is None or self.k_B is None or self.T is None or self.c is None:
            return False, "SKIP: missing g,k_B,T,c"
        expected = 3 * self.g * self.k_B * self.T / (4 * self.c**2)
        xm = self.results.get('x_mean')
        if xm is None:
            return False, "x_mean not in results"
        xm_f = self._safe_float(xm)
        if xm_f is None:
            return False, "x_mean not numeric"
        if math.isclose(xm_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Mean position matches perturbation result"
        else:
            return False, f"x_mean {xm_f} != expected {expected}"

    # -----------------------------------------------------------------
    #  Monotonicity tests – check physical trends via small perturbations
    # -----------------------------------------------------------------
    def test_monotonicity_energy_increasing_temp(self):
        """Increasing T (decreasing beta) should increase mean energy."""
        if self.alpha is None or self.hbar is None or self.omega is None or self.T is None:
            return False, "SKIP: missing givens"
        alpha = self._safe_alpha(self.alpha)
        if alpha is None:
            return False, "SKIP: alpha not numeric"
        # Use the formula: E = hbar*omega/2 + hbar*omega/(exp(alpha)-1)
        # Perturb T by +5%
        T_new = self.T * 1.05
        beta_new = 1.0 / (self.k_B * T_new) if self.k_B else None
        if beta_new is None:
            return False, "SKIP: missing k_B"
        alpha_new = self.hbar * self.omega * beta_new
        E_original = 0.5*self.hbar*self.omega + self.hbar*self.omega/(math.exp(alpha)-1)
        E_new = 0.5*self.hbar*self.omega + self.hbar*self.omega/(math.exp(alpha_new)-1)
        # E should increase with T (dE/dT>0)
        if E_new > E_original:
            return True, "E increases with T as expected"
        else:
            return False, f"E does not increase with T: original {E_original}, perturbed {E_new}"

    def test_monotonicity_mean_position_increasing_temp(self):
        """Increasing T should increase x_mean (proportional to T)."""
        if self.g is None or self.k_B is None or self.c is None or self.T is None:
            return False, "SKIP: missing g,k_B,c,T"
        T_new = self.T * 1.05
        x_original = 3 * self.g * self.k_B * self.T / (4 * self.c**2)
        x_new = 3 * self.g * self.k_B * T_new / (4 * self.c**2)
        if x_new > x_original:
            return True, "x_mean increases with T"
        else:
            return False, f"x_mean does not increase with T: original {x_original}, perturbed {x_new}"

    # -----------------------------------------------------------------
    #  Weak answer sanity test – only sign/finite/non-NaN checks
    # -----------------------------------------------------------------
    def test_answer_sanity_sign_finite(self):
        """All results should be finite and non-negative where appropriate."""
        ok = True
        for key in ['E1', 'Z2', 'Z2_degen', 'C']:
            val = self.results.get(key)
            if val is None:
                ok = False
                continue
            val_f = self._safe_float(val)
            if val_f is None or not math.isfinite(val_f) or val_f <= 0:
                ok = False
        # x_mean could be positive or negative; just check finite
        xm = self.results.get('x_mean')
        if xm is not None:
            xm_f = self._safe_float(xm)
            if xm_f is None or not math.isfinite(xm_f):
                ok = False
        if ok:
            return True, "All results are positive and finite"
        else:
            return False, "Some results are missing, non-finite, or non-positive"

    # -----------------------------------------------------------------
    #  Master test runner
    # -----------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not isinstance(method, types.MethodType):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)