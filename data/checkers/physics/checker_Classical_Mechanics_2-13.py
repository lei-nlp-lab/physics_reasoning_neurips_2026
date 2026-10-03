import math
import re
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        # Extract givens – use None for missing values (no numeric defaults)
        self.G = getattr(solution_instance, 'G', None)
        self.M = getattr(solution_instance, 'M', None)
        self.m = getattr(solution_instance, 'm', None)
        self.b = getattr(solution_instance, 'b', None)
        self.L = getattr(solution_instance, 'L', None)
        # Extract results – use None for missing keys
        self.R = getattr(solution_instance, 'R', None)
        self.T = getattr(solution_instance, 'T', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.delta_t = getattr(solution_instance, 'delta_t', None)   # string
        self.phi_t = getattr(solution_instance, 'phi_t', None)       # string

    # ---------- Helper methods ----------
    def _compute_R_from_L(self, L, G, M, m, b):
        """
        Compute the larger (stable) circular radius from the quadratic:
        GM m^2 R^2 - L^2 R + 3 b m = 0
        """
        A = G * M * m**2
        B = -L**2
        C = 3 * b * m
        discriminant = B**2 - 4 * A * C   # L^4 - 12 G M m^3 b
        if discriminant <= 0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        R_plus = (-B + sqrt_disc) / (2 * A)
        return R_plus

    def _compute_Veff_prime(self, r):
        # dV_eff/dr = GMm/r^2 + 3b/r^4 - L^2/(m r^3)
        return (self.G * self.M * self.m) / (r**2) + (3 * self.b) / (r**4) - (self.L**2) / (self.m * r**3)

    def _compute_Veff_double(self, r):
        # second derivative: -2GMm/r^3 - 12b/r^5 + 3L^2/(m r^4)
        return (-2 * self.G * self.M * self.m) / (r**3) - (12 * self.b) / (r**5) + (3 * self.L**2) / (self.m * r**4)

    # ---------- Interface Tests ----------
    def test_interface_R_exists(self):
        if self.R is None:
            return False, "R is missing (None)"
        if not isinstance(self.R, (int, float)):
            return False, f"R is not numeric: {type(self.R)}"
        return True, f"R = {self.R} present and numeric"

    def test_interface_T_exists(self):
        if self.T is None:
            return False, "T is missing (None)"
        if not isinstance(self.T, (int, float)):
            return False, f"T is not numeric: {type(self.T)}"
        return True, f"T = {self.T} present and numeric"

    def test_interface_omega_exists(self):
        if self.omega is None:
            return False, "omega is missing (None)"
        if not isinstance(self.omega, (int, float)):
            return False, f"omega is not numeric: {type(self.omega)}"
        return True, f"omega = {self.omega} present and numeric"

    def test_interface_delta_t_exists(self):
        if self.delta_t is None:
            return False, "delta_t is missing (None)"
        if not isinstance(self.delta_t, str):
            return False, f"delta_t is not a string: {type(self.delta_t)}"
        return True, "delta_t expression present and is string"

    def test_interface_phi_t_exists(self):
        if self.phi_t is None:
            return False, "phi_t is missing (None)"
        if not isinstance(self.phi_t, str):
            return False, f"phi_t is not a string: {type(self.phi_t)}"
        return True, "phi_t expression present and is string"

    def test_interface_givens_present(self):
        missing = []
        for name in ('G', 'M', 'm', 'b', 'L'):
           if getattr(self, name) is None:
               missing.append(name)
        if missing:
            return False, f"Missing givens: {missing}"
        return True, "All required givens present"

    # ---------- Invariant Tests (use math.isclose) ----------
    def test_invariant_circular_orbit_condition(self):
        # dV_eff/dr at R must be zero
        if any(v is None for v in (self.G, self.M, self.m, self.b, self.L, self.R)):
            return True, "SKIP: missing data for circular orbit condition"
        val = self._compute_Veff_prime(self.R)
        if not math.isclose(val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"dV_eff/dr(R) = {val:.4e}, expected 0"
        return True, f"dV_eff/dr(R) ≈ 0 (value {val:.4e})"

    def test_invariant_stability_positive(self):
        # d²V_eff/dr² at R must be > 0
        if any(v is None for v in (self.G, self.M, self.m, self.b, self.L, self.R)):
            return True, "SKIP: missing data for stability check"
        val = self._compute_Veff_double(self.R)
        if val <= 0:
            return False, f"V''_eff(R) = {val:.4e}, should be > 0 for stable orbit"
        return True, f"V''_eff(R) = {val:.4e} > 0 (stable)"

    def test_invariant_omega_squared(self):
        # omega^2 ≈ V''_eff(R) / m
        if any(v is None for v in (self.m, self.R, self.omega)):
            return True, "SKIP: missing data for omega check"
        V_double = self._compute_Veff_double(self.R)
        expected_omega2 = V_double / self.m
        actual_omega2 = self.omega**2
        if not math.isclose(actual_omega2, expected_omega2, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"omega² = {actual_omega2:.4e}, expected {expected_omega2:.4e} from V''/m"
        return True, f"omega² ≈ V''/m (difference {abs(actual_omega2 - expected_omega2):.4e})"

    def test_invariant_period(self):
        # T = 2π m R^2 / L
        if any(v is None for v in (self.m, self.R, self.L, self.T)):
            return True, "SKIP: missing data for period check"
        expected_T = 2 * math.pi * self.m * self.R**2 / self.L
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"T = {self.T:.4e}, expected {expected_T:.4e} from formula"
        return True, f"T ≈ expected period (difference {abs(self.T - expected_T):.4e})"

    def test_invariant_R_formula(self):
        # R (stable) matches derived expression from L
        if any(v is None for v in (self.G, self.M, self.m, self.b, self.L, self.R)):
            return True, "SKIP: missing data for R formula check"
        R_calc = self._compute_R_from_L(self.L, self.G, self.M, self.m, self.b)
        if R_calc is None:
            return False, "Cannot compute R from L (discriminant not positive)"
        if not math.isclose(self.R, R_calc, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"R = {self.R:.4e}, formula gives {R_calc:.4e}"
        return True, f"R = {self.R:.4e} matches formula {R_calc:.4e}"

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_delta_form(self):
        # delta_t should be a sine function containing omega and t
        if self.delta_t is None:
            return False, "delta_t is missing"
        text = self.delta_t.replace(" ", "")
        if "sin" not in text.lower():
            return False, "delta_t expression does not contain 'sin'"
        if "omega" not in text.lower() and "ω" not in text:
            return False, "delta_t expression does not reference omega"
        if "t" not in text:
            return False, "delta_t expression does not contain 't'"
        return True, "delta_t contains sin(omega*t + constant)"

    def test_modeling_phi_form(self):
        # phi_t should be of form const + (L/(mR^2))*t + (cos term)
        if self.phi_t is None:
            return False, "phi_t is missing"
        text = self.phi_t.replace(" ", "")
        if "cos" not in text.lower():
            return False, "phi_t does not contain 'cos'"
        if "omega" not in text.lower() and "ω" not in text:
            return False, "phi_t does not reference omega"
        if "t" not in text:
            return False, "phi_t does not contain 't'"
        if "L" not in text:
            return False, "phi_t does not contain 'L'"
        if "m" not in text:
            return False, "phi_t does not contain 'm'"
        if "R" not in text:
            return False, "phi_t does not contain 'R'"
        return True, "phi_t contains expected terms: L, m, R, omega, cos"

    def test_modeling_phi_cos_sign(self):
        # The cos term should be added (positive sign) as per expected formula.
        # Look for "+ ... cos" or " + ... cos" (allow spaces). If only "-cos" then fail.
        if self.phi_t is None:
            return False, "phi_t is missing"
        # Remove all spaces to simplify pattern
        text = self.phi_t.replace(" ", "")
        # Find the cos term: look for a "+" before "cos" (after the constant part)
        # Expected pattern ends with + ... cos(...)
        # We'll check that "+cos" or "++" not present, but there is a "+" before the cos.
        # Alternatively, we can check that the sign before cos is positive.
        # Simple approach: find 'cos' and look at one character before.
        idx = text.lower().find("cos")
        if idx == -1:
            return False, "No cos term found"
        if idx == 0:
            return False, "cos term at start without sign"
        char_before = text[idx-1]
        if char_before == '-':
            return False, "cos term has negative sign, expected positive"
        # If char_before is '+', acceptable; also could be a number or ')' but usually '+'.
        return True, "cos term appears with positive sign"

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_R_increases_with_L(self):
        # Perturb L upward by 1% and compute new stable R; new R should be larger.
        if any(v is None for v in (self.G, self.M, self.m, self.b, self.L, self.R)):
            return True, "SKIP: missing data for R monotonicity"
        original_R = self.R
        L_pert = self.L * 1.01
        new_R = self._compute_R_from_L(L_pert, self.G, self.M, self.m, self.b)
        if new_R is None:
            return True, "SKIP: cannot compute R for perturbed L (discriminant too small)"
        if new_R <= original_R:
            return False, f"R decreased when L increased: original R={original_R:.4e}, new R={new_R:.4e}"
        return True, f"R increases with L (original {original_R:.4e} -> perturbed {new_R:.4e})"

    def test_monotonicity_T_increases_with_L(self):
        # T = 2π m R^2 / L. Perturb L by +1%, recompute R (and thus T), expect T larger.
        if any(v is None for v in (self.G, self.M, self.m, self.b, self.L, self.m, self.R, self.T)):
            return True, "SKIP: missing data for T monotonicity"
        m = self.m
        L_orig = self.L
        R_orig = self.R
        T_orig = self.T
        L_pert = L_orig * 1.01
        R_pert = self._compute_R_from_L(L_pert, self.G, self.M, m, self.b)
        if R_pert is None:
            return True, "SKIP: cannot compute R for perturbed L"
        T_pert = 2 * math.pi * m * R_pert**2 / L_pert
        if T_pert <= T_orig:
            return False, f"T decreased when L increased: original T={T_orig:.4e}, new T={T_pert:.4e}"
        return True, f"T increases with L (original {T_orig:.4e} -> perturbed {T_pert:.4e})"

    # ---------- Answer Sanity (weak, only one) ----------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: final answers should be physically plausible.
        (Only one such test allowed; no math.isclose, no re-derivation.)
        """
        violations = []
        if self.R is not None and self.R <= 0:
            violations.append("R is non-positive")
        if self.T is not None and self.T <= 0:
            violations.append("T is non-positive")
        if self.omega is not None and self.omega <= 0:
            violations.append("omega is non-positive")
        if self.delta_t is not None and not isinstance(self.delta_t, str):
            violations.append("delta_t is not a string")
        if self.phi_t is not None and not isinstance(self.phi_t, str):
            violations.append("phi_t is not a string")
        if violations:
            return False, "; ".join(violations)
        return True, "All final answers are physically plausible (positive and correct types)"

    # ---------- Runner ----------
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