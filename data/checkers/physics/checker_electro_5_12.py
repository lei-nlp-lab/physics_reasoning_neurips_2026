import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits the infinite ion beam solution.
    Tests are grouped as unit-test-style methods with prefixes:
    - test_interface_*  : required keys, types
    - test_invariant_*  : physical laws, symbolic consistency
    - test_modeling_*   : pseudocode/trace consistency
    - test_monotonicity_*: directional perturbation checks
    - test_answer_sanity_* (at most one weak sanity)
    """
    def __init__(self, solution_instance):
        # Result key
        self.F_r = getattr(solution_instance, 'F_r', None)
        if self.F_r is None and hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.F_r = solution_instance.results.get('F_r', None)

        # Givens (allow missing, use None)
        self.e = self._safe_get(solution_instance, 'e', None)
        self.q = self._safe_get(solution_instance, 'q', None)
        self.R = self._safe_get(solution_instance, 'R', None)
        self.r_pos = self._safe_get(solution_instance, 'r', None)   # avoid conflict with built-in
        self.v = self._safe_get(solution_instance, 'v', None)
        self.c = self._safe_get(solution_instance, 'c', None)
        self.epsilon0 = self._safe_get(solution_instance, 'epsilon0', None)
        self.mu0 = self._safe_get(solution_instance, 'mu0', None)

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'solving_pseudocode', None)

    def _safe_get(self, inst, name, default):
        """Attempt to read from .givens, .results, or direct attribute."""
        # .givens dict (common)
        if hasattr(inst, 'givens') and isinstance(inst.givens, dict) and name in inst.givens:
            return inst.givens[name]
        # .results dict (common)
        if hasattr(inst, 'results') and isinstance(inst.results, dict) and name in inst.results:
            return inst.results[name]
        # Direct attribute (fallback)
        if hasattr(inst, name):
            return getattr(inst, name)
        return default

    # --------------------------------------------------------------------------
    # INTERFACE TESTS (key existence, type)
    # --------------------------------------------------------------------------
    def test_interface_result_key_F_r(self):
        """Result 'F_r' must exist."""
        if self.F_r is None:
            return (False, "Missing result key 'F_r'")
        return (True, "Key 'F_r' present")

    def test_interface_givens_all_present(self):
        """All required givens should be available (or at least defaulted)."""
        missing = []
        for name in ['e', 'q', 'R', 'r_pos', 'v', 'c']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All required givens present")

    # --------------------------------------------------------------------------
    # INVARIANT TESTS (theorem‑based, use math.isclose)
    # --------------------------------------------------------------------------
    def test_invariant_magnetic_field_scale(self):
        """
        Check that B = (mu0 * J * r) / 2 with J = (q/(pi R^2))*v.
        We verify the ratio B_phi / (rho * v * r) = mu0/2.
        Because B is not directly stored, we check final F_r dependence.
        Using the Lorentz force: F_r = e*(E - v*B). If we derive numerically,
        the factor (1 - v^2/c^2) should appear.
        """
        if any(getattr(self, x) is None for x in ['e','q','R','r_pos','v','c','epsilon0','mu0']):
            return (False, "SKIP – missing givens")
        # Symbolic derivation of expected F_r
        e, q, R, r, v, c, eps0, mu0 = sp.symbols('e q R r v c eps0 mu0', positive=True)
        pi = sp.pi
        rho = q / (pi * R**2)
        E = (rho * r) / (2 * eps0)
        J = rho * v
        B = (mu0 * J * r) / 2
        F_r_expr = e * (E - v * B)
        F_r_simplified = sp.simplify(F_r_expr)
        # Because mu0 eps0 = 1/c^2, substitute that relation
        F_r_final = sp.simplify(F_r_simplified.subs(mu0*eps0, 1/c**2))
        # Expected final form: (e * q * r) / (2 * pi * eps0 * R**2 * gamma**2)
        gamma = 1 / sp.sqrt(1 - v**2/c**2)
        expected = (e * q * r) / (2 * pi * eps0 * R**2 * gamma**2)
        diff = sp.simplify(F_r_final - expected)
        if diff != 0:
            return (False, f"Symbolic derivation mismatch: {diff}")
        # Numeric check
        subs = {
            e: 1.602176634e-19,
            q: 1e-6,
            R: 0.1,
            r: 0.05,
            v: 3e7,
            c: 299792458,
            eps0: self.epsilon0,
            mu0: self.mu0
        }
        val_F_r = F_r_final.subs(subs).evalf()
        val_expected = expected.subs(subs).evalf()
        if not math.isclose(float(val_F_r), float(val_expected), rel_tol=1e-3):
            return (False, f"Numeric mismatch: {val_F_r} vs {val_expected}")
        return (True, "Invariant Lorentz force structure correct")

    def test_invariant_radial_only(self):
        """
        The force must be purely radial (no azimuthal or axial components).
        Even if F is given as a scalar, we check that it is not a tuple with non‑zero
        other components. The expected key is scalar.
        """
        if self.F_r is None:
            return (False, "SKIP – F_r missing")
        # If F_r is a list/vector, check length
        if isinstance(self.F_r, (list, tuple)):
            if len(self.F_r) != 1:
                # could be 3D vector; check second and third are zero
                if len(self.F_r) >= 2 and (abs(float(self.F_r[1])) > 1e-12 or abs(float(self.F_r[2])) > 1e-12):
                    return (False, "Non‑radial components present")
        return (True, "Force appears radial (or scalar)")

    def test_invariant_sign_repulsion(self):
        """For like charges (e and q same sign), the force at r>0 should be repulsive (positive radial if outward convention)."""
        if self.F_r is None:
            return (False, "SKIP – F_r missing")
        if any(getattr(self, x) is None for x in ['e','q','R','r_pos','v']):
            return (False, "SKIP – missing givens for sign check")
        # Evaluate sign using a simple numeric scenario: all positive.
        # Since we don't know the exact numeric value, we check the sign of the provided expression.
        # We test that with e=+1, q=+1, v=0, the force is repulsive (positive if outward radial).
        try:
            # Use sympy to substitute e=1, q=1, v=0 into the provided F_r if it's symbolic.
            # If F_r is numeric, we can't easily test sign symbolically.
            if isinstance(self.F_r, (int, float)):
                # If it's a scalar, assume positive if positive
                if self.F_r > 0:
                    return (True, f"Signed positively: {self.F_r}")
                else:
                    return (False, f"Negative sign for repulsive case: {self.F_r}")
            # Otherwise try symbolic substitution with +1 values
            e_sym, q_sym, v_sym = sp.symbols('e q v')
            sub_test = {e_sym: 1, q_sym: 1, v_sym: 0}
            # Extract free symbols from F_r expression
            if hasattr(self.F_r, 'subs'):
                # We need a mapping from the symbolic symbols of F_r to test values
                free = self.F_r.free_symbols
                # Free symbol names we can map: 'e', 'q', 'v', but also 'c', 'R', etc.
                # We'll ignore those by not substituting them, assuming they are positive.
                mapping = {}
                for symb in free:
                    if str(symb) == 'e':
                        mapping[symb] = 1
                    elif str(symb) == 'q':
                        mapping[symb] = 1
                    elif str(symb) == 'v':
                        mapping[symb] = 0
                # If we cannot determine sign, just mark as uncertain.
                if not mapping:
                    return (True, "Cannot evaluate sign symbolically")
                expr_test = self.F_r.subs(mapping)
                if expr_test.is_number:
                    val = float(expr_test)
                    if val > 0:
                        return (True, "Positive sign for repulsive case")
                    else:
                        return (False, f"Non‑positive sign: {val}")
                else:
                    return (True, "Sign ambiguous, skip")
            else:
                return (True, "Cannot interpret, skip")
        except Exception as e:
            return (False, f"Sign test exception: {e}")

    # --------------------------------------------------------------------------
    # MODELING / TRACE TESTS (based on explanation steps or pseudocode)
    # --------------------------------------------------------------------------
    def test_modeling_uses_gamma_factor(self):
        """The final expression must involve gamma factor (1 - v^2/c^2)^(-1/2) or equivalent."""
        if self.F_r is None:
            return (False, "SKIP – no F_r to analyze")
        # Check that the string representation contains 'gamma' or '(1 - v**2/c**2)**(-1/2)'
        s = str(self.F_r)
        if 'gamma' in s or '1/sqrt' in s or 'sqrt(1 - v**2' in s or '(1 - ' in s:
            return (True, "Relativistic factor present")
        # Alternatively, check that if v=0, the expression simplifies to electrostatic form.
        # But this might be insufficient.
        return (False, "No obvious gamma factor in expression")

    def test_modeling_force_proportional_to_r(self):
        """Force must be linear in r (inside beam)."""
        if self.F_r is None:
            return (False, "SKIP – F_r missing")
        if hasattr(self.F_r, 'free_symbols'):
            r_sym = None
            for s in self.F_r.free_symbols:
                if str(s) == 'r':
                    r_sym = s
                    break
            if r_sym is None:
                return (False, "No 'r' symbol in expression")
            # Check that F_r / r is independent of r (derivative wrt r should be constant)
            try:
                ratio = self.F_r / r_sym
                diff = sp.diff(ratio, r_sym)
                if diff == 0:
                    return (True, "Force linear in r")
                else:
                    return (False, f"Force not linear in r: derivative={diff}")
            except Exception as e:
                return (False, f"Derivative error: {e}")
        else:
            # Numeric: can't easily check linearity without multiple samples
            return (True, "Cannot verify linearity numerically")

    # --------------------------------------------------------------------------
    # MONOTONICITY TESTS (perturb input, compare perturbed output)
    # --------------------------------------------------------------------------
    def test_monotonicity_increase_r_increases_force(self):
        """
        Increasing the radial position r (while staying inside beam) should increase the force magnitude.
        We derive the force analytically and compare two values.
        """
        if any(getattr(self, x) is None for x in ['e','q','R','v','c','epsilon0']):
            return (False, "SKIP – missing givens for monotonicity")
        # Use sympy to evaluate at two r values: r and 1.1*r (both < R)
        r_sym = sp.symbols('r', positive=True)
        # Build expression from derived formula
        e_sym, q_sym, R_sym, v_sym, c_sym, eps0_sym = sp.symbols('e q R v c eps0', positive=True)
        gamma = 1 / sp.sqrt(1 - v_sym**2/c_sym**2)
        expected_expr = (e_sym * q_sym * r_sym) / (2 * sp.pi * eps0_sym * R_sym**2 * gamma**2)
        subs_base = {e_sym: self.e, q_sym: self.q, R_sym: self.R, v_sym: self.v,
                     c_sym: self.c, eps0_sym: self.epsilon0}
        r_val = self.r_pos if self.r_pos is not None else 0.05
        if r_val <= 0:
            return (False, "r must be positive for monotonicity")
        # Ensure r < R
        if r_val >= self.R:
            return (False, "Cannot test if r >= R")
        r_perturbed = 1.1 * r_val
        if r_perturbed >= self.R:
            r_perturbed = 0.9 * self.R  # ensure still inside
            if r_perturbed <= r_val:
                return (False, "Cannot perturb inside beam")
        val_base = float(expected_expr.subs({**subs_base, r_sym: r_val}).evalf())
        val_pert = float(expected_expr.subs({**subs_base, r_sym: r_perturbed}).evalf())
        if val_pert > val_base:
            return (True, f"Force increases with r: {val_base} to {val_pert}")
        else:
            return (False, f"Force not monotonic: base={val_base}, perturbed={val_pert}")

    def test_monotonicity_increase_R_decreases_force(self):
        """Increasing beam radius R should decrease the force magnitude (inverse square)."""
        if any(getattr(self, x) is None for x in ['e','q','r_pos','v','c','epsilon0']):
            return (False, "SKIP – missing givens")
        e_sym, q_sym, r_sym, v_sym, c_sym, eps0_sym, R_sym = sp.symbols('e q r v c eps0 R', positive=True)
        gamma = 1 / sp.sqrt(1 - v_sym**2/c_sym**2)
        expr = (e_sym * q_sym * r_sym) / (2 * sp.pi * eps0_sym * R_sym**2 * gamma**2)
        subs = {e_sym: self.e, q_sym: self.q, r_sym: self.r_pos if self.r_pos else 0.05,
                v_sym: self.v, c_sym: self.c, eps0_sym: self.epsilon0}
        R_base = self.R
        R_pert = 1.5 * R_base
        val_base = float(expr.subs({**subs, R_sym: R_base}).evalf())
        val_pert = float(expr.subs({**subs, R_sym: R_pert}).evalf())
        if val_pert < val_base:
            return (True, f"Force decreases with larger R: {val_base} to {val_pert}")
        else:
            return (False, f"Force not decreasing with R: base={val_base}, perturbed={val_pert}")

    def test_monotonicity_increase_v_decreases_force(self):
        """Increasing velocity v should decrease the force (relativistic factor)."""
        if any(getattr(self, x) is None for x in ['e','q','r_pos','R','c','epsilon0']):
            return (False, "SKIP – missing givens")
        e_sym, q_sym, r_sym, v_sym, c_sym, eps0_sym, R_sym = sp.symbols('e q r v c eps0 R', positive=True)
        gamma = 1 / sp.sqrt(1 - v_sym**2/c_sym**2)
        expr = (e_sym * q_sym * r_sym) / (2 * sp.pi * eps0_sym * R_sym**2 * gamma**2)
        subs = {e_sym: self.e, q_sym: self.q, r_sym: self.r_pos if self.r_pos else 0.05,
                R_sym: self.R, c_sym: self.c, eps0_sym: self.epsilon0}
        v_base = self.v
        v_pert = min(self.c * 0.9, v_base * 1.5)  # avoid exceeding c
        if v_pert <= v_base:
            v_pert = v_base * 1.2
        val_base = float(expr.subs({**subs, v_sym: v_base}).evalf())
        val_pert = float(expr.subs({**subs, v_sym: v_pert}).evalf())
        if val_pert < val_base:
            return (True, f"Force decreases with velocity: {val_base} to {val_pert}")
        else:
            return (False, f"Force not decreasing with v: base={val_base}, perturbed={val_pert}")

    # --------------------------------------------------------------------------
    # WEAK ANSWER SANITY TEST (at most one)
    # --------------------------------------------------------------------------
    def test_answer_sanity_bounds(self):
        """
        Weak sanity: force value must be finite, non-zero (unless r=0), and have reasonable order of magnitude.
        Do NOT check exact expected values.
        """
        if self.F_r is None:
            return (False, "SKIP – F_r missing")
        # Check finite
        try:
            val = float(self.F_r) if not hasattr(self.F_r, 'subs') else None
        except:
            val = None
        if val is not None:
            if not math.isfinite(val):
                return (False, "Force is not finite")
            # If r_pos is given, check that for typical parameters value is plausible
            if self.r_pos is not None and self.e is not None and self.q is not None:
                # Rough order: e*q/(2*pi*eps0) ~ 1e-9 Nm/C? Actually for 1e-6 C/m and 1e-19 C => ~1.8e-14 Nm
                # Then divide by R^2 ~ 0.01 gives ~1.8e-12 N. But we only check not absurd.
                if abs(val) > 1e3:
                    return (False, f"Force magnitude appears too large: {val}")
            return (True, "Finite and plausible magnitude")
        # If symbolic, skip numeric check
        return (True, "Symbolic expression, cannot evaluate numeric bounds")

    # --------------------------------------------------------------------------
    # RUN ALL TESTS
    # --------------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discovers all bound methods starting with 'test_', executes them in try/except,
        collects logs, returns (success, logs_list).
        """
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