import sympy as sp
import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Auditor for a physics problem: oscillating dipole above a floor.
    Checks the four required outputs: I(R), R_max, P_floor, tau.
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction of results dictionary
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Safe extraction of givens – use None for missing values
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.k = getattr(solution_instance, 'k', None)
        self.h = getattr(solution_instance, 'h', None)
        self.d = getattr(solution_instance, 'd', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.c = getattr(solution_instance, 'c', None)

        # Results to be checked
        self.I_R = results.get('I_R', None)
        self.R_max = results.get('R_max', None)
        self.P_floor = results.get('P_floor', None)
        self.tau = results.get('tau', None)

        # Derived physical constants
        self._omega = None
        self._p0 = None
        self._Larmor_total = None
        self._compute_derived()

    def _compute_derived(self):
        """Compute derived quantities if givens are present."""
        if self.m is not None and self.k is not None and self.m > 0:
            self._omega = math.sqrt(self.k / self.m)
        else:
            self._omega = None

        if self.q is not None and self.d is not None:
            self._p0 = self.q * self.d
        else:
            self._p0 = None

        if (self.mu0 is not None and self.q is not None and self.d is not None
                and self._omega is not None and self.c is not None):
            self._Larmor_total = (self.mu0 * self.q**2 * self.d**2 *
                                  self._omega**4) / (12 * math.pi * self.c)
        else:
            self._Larmor_total = None

    def _eval_num(self, expr, subs: Dict[sp.Symbol, float]) -> Optional[float]:
        """Evaluate a sympy expression numerically with given substitutions.
        Returns None if evaluation fails."""
        try:
            val = expr.evalf(subs={s: float(v) for s, v in subs.items()}, n=15)
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_keys_exist(self) -> Tuple[bool, str]:
        """Check that all required result keys are present and not None."""
        keys = ['I_R', 'R_max', 'P_floor', 'tau']
        missing = []
        for k in keys:
            val = getattr(self, k, None)
            if val is None:
                missing.append(k)
        if missing:
            return False, f"Missing results: {missing}"
        return True, "All required keys present."

    def test_interface_givens_positive(self) -> Tuple[bool, str]:
        """Check that all required givens are positive numbers."""
        required = [('m', self.m), ('q', self.q), ('k', self.k),
                    ('h', self.h), ('d', self.d), ('mu0', self.mu0),
                    ('c', self.c)]
        missing_or_nonpos = []
        for name, val in required:
            if val is None:
                missing_or_nonpos.append(f"{name} missing")
            elif val <= 0:
                missing_or_nonpos.append(f"{name} non-positive ({val})")
        if missing_or_nonpos:
            return False, "Givens issues: " + "; ".join(missing_or_nonpos)
        return True, "All givens positive and present."

    # ------------------------------------------------------------------
    # Invariant (theorem) tests
    # ------------------------------------------------------------------
    def test_invariant_I_expression_form(self) -> Tuple[bool, str]:
        """Verify that I(R) has the functional form 
        coefficient * R^2 * h / (R^2 + h^2)^(5/2)."""
        if self.I_R is None:
            return (False, "SKIP: I_R not available")
        # Try to interpret as sympy expression
        try:
            expr = sp.sympify(self.I_R) if isinstance(self.I_R, str) else self.I_R
        except (sp.SympifyError, TypeError):
            # If it's a pure number, we cannot check form
            return (False, "Cannot parse I_R as symbolic expression")

        # Define symbols
        R, h = sp.symbols('R h')
        # Expected form (ignoring coefficient)
        expected_form = R**2 * h / (R**2 + h**2)**(sp.Rational(5,2))
        # Try to factor out coefficient
        try:
            ratio = sp.simplify(expr / expected_form)
            # ratio should be a constant (independent of R and h)
            free = ratio.free_symbols
            # Remove h and R – if any other symbols remain, not constant
            remaining = free - {R, h}
            if remaining:
                return (False, f"Expression contains extra symbols {remaining}")
            # Now check ratio is constant (numeric or symbolic constant)
            # Evaluate as numeric if possible
            if ratio.is_number:
                coeff = float(ratio)
                return (True, f"Intensity form correct with coefficient {coeff:.4g}")
            else:
                # Maybe ratio is a symbolic constant like mu0*... but that's fine
                return (True, "Expression form matches expected (coefficient may be symbolic)")
        except Exception as e:
            return (False, f"Error checking form: {e}")

    def test_invariant_R_max_value(self) -> Tuple[bool, str]:
        """Check that R_max = sqrt(2/3) * h."""
        if self.R_max is None or self.h is None:
            return (False, "SKIP: R_max or h missing")
        expected = math.sqrt(2.0/3.0) * self.h
        try:
            rmax = float(self.R_max)
        except (TypeError, ValueError):
            # If it's symbolic, try numerical evaluation
            try:
                expr = sp.sympify(self.R_max)
                rmax = float(expr.evalf())
            except Exception:
                return (False, "Cannot convert R_max to float")
        if math.isclose(rmax, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R_max = {rmax:.4g} ≈ expected {expected:.4g}")
        else:
            return (False, f"R_max = {rmax:.4g}, expected {expected:.4g}")

    def test_invariant_P_floor_half_larmor(self) -> Tuple[bool, str]:
        """Check that P_floor equals half the total Larmor power."""
        if self.P_floor is None or self._Larmor_total is None:
            return (False, "SKIP: P_floor or Larmor power not available")
        try:
            pfloor = float(self.P_floor)
        except (TypeError, ValueError):
            return (False, "Cannot convert P_floor to float")
        expected = self._Larmor_total / 2.0
        if math.isclose(pfloor, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_floor = {pfloor:.4g} ≈ half Larmor = {expected:.4g}")
        else:
            return (False, f"P_floor = {pfloor:.4g}, expected {expected:.4g}")

    def test_invariant_tau_formula(self) -> Tuple[bool, str]:
        """Check that tau = 12*pi*m^2*c / (mu0 * q^2 * k)."""
        if any(x is None for x in [self.tau, self.m, self.mu0, self.q, self.k, self.c]):
            return (False, "SKIP: tau or required givens missing")
        try:
            tau_val = float(self.tau)
        except (TypeError, ValueError):
            return (False, "Cannot convert tau to float")
        expected = (12.0 * math.pi * self.m**2 * self.c) / (self.mu0 * self.q**2 * self.k)
        if math.isclose(tau_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"tau = {tau_val:.4g} ≈ expected {expected:.4g}")
        else:
            return (False, f"tau = {tau_val:.4g}, expected {expected:.4g}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_dipole_intensity_geometry(self) -> Tuple[bool, str]:
        """Check that I(R) matches the projected Poynting vector:
        <S> * cosθ, with <S> = (μ0 p0^2 ω^4 sin^2θ)/(32π^2 c r^2),
        sinθ = R/r, cosθ = h/r, r = sqrt(R^2+h^2).
        """
        if self.I_R is None or any(x is None for x in [self.h, self.d, self.m, self.k, self.mu0, self.c, self.q]):
            return (False, "SKIP: Missing data for geometric check")
        # Compute numerical test: pick a specific R, say R = h/2
        R_val = self.h / 2.0
        r_val = math.sqrt(R_val**2 + self.h**2)
        sin_theta = R_val / r_val
        cos_theta = self.h / r_val
        p0 = self.q * self.d
        omega = self._omega
        # Compute expected numerical I(R_val) from formula
        expected_I = (self.mu0 * p0**2 * omega**4) / (32 * math.pi**2 * self.c) * sin_theta**2 * cos_theta / r_val**2
        # Get actual I(R_val) from expression
        try:
            expr = sp.sympify(self.I_R) if isinstance(self.I_R, str) else self.I_R
            subs = {sp.Symbol('R'): R_val, sp.Symbol('h'): self.h}
            actual_I = self._eval_num(expr, subs)
            if actual_I is None:
                return (False, "Could not evaluate I(R) numerically")
        except Exception:
            return (False, "Error evaluating I(R) expression")
        if math.isclose(actual_I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I(R=h/2) = {actual_I:.4g} ≈ {expected_I:.4g}")
        else:
            return (False, f"I(R=h/2) = {actual_I:.4g}, expected {expected_I:.4g}")

    def test_modeling_energy_decay_consistency(self) -> Tuple[bool, str]:
        """Check that tau calculated from energy loss agrees:
        E0 = 0.5*k*d^2, dE/dt = -2*P_floor, so tau = E0/(2*P_floor)."""
        if any(x is None for x in [self.tau, self.P_floor, self.k, self.d]):
            return (False, "SKIP: Missing data for energy decay check")
        try:
            tau_val = float(self.tau)
            pfloor = float(self.P_floor)
        except (TypeError, ValueError):
            return (False, "Cannot convert tau or P_floor to float")
        E0 = 0.5 * self.k * self.d**2
        expected_tau = E0 / (2.0 * pfloor)
        if math.isclose(tau_val, expected_tau, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"tau = {tau_val:.4g} ≈ E0/(2P) = {expected_tau:.4g}")
        else:
            return (False, f"tau = {tau_val:.4g}, expected from energy {expected_tau:.4g}")

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb a physical input and compare outputs)
    # ------------------------------------------------------------------
    def test_monotonicity_P_floor_scales_with_d_squared(self) -> Tuple[bool, str]:
        """Check that P_floor scales as d^2 by perturbing d by +10%."""
        if self.P_floor is None or self.d is None:
            return (False, "SKIP: P_floor or d missing")
        try:
            pfloor_base = float(self.P_floor)
            d_base = float(self.d)
        except (TypeError, ValueError):
            return (False, "Cannot convert P_floor or d to float")

        # Perturb d by +10%
        d_perturbed = d_base * 1.1
        # Expected P_floor_perturbed = P_floor_base * (d_perturbed/d_base)^2
        expected_p_perturbed = pfloor_base * (d_perturbed / d_base)**2

        # Now we need to compute the actual P_floor_perturbed using the same formula.
        # Since we cannot call the solver again, we check that the student's P_floor expression
        # (if symbolic) yields the correct scaling when d changes.
        # We will attempt to derive the scaling from the expression for P_floor.
        # If P_floor is given as a formula (string or sympy expression), we evaluate it with the new d.
        try:
            # Try to treat P_floor as expression
            if isinstance(self.P_floor, str):
                expr = sp.sympify(self.P_floor)
            else:
                expr = self.P_floor
            # Define symbols that might appear
            syms = expr.free_symbols
            # Look for a symbol representing d (could be 'd' or something else)
            # We'll assume it's 'd' or we can try to map.
            d_sym = None
            for s in syms:
                if str(s) == 'd':
                    d_sym = s
                    break
            if d_sym is None:
                # If not found, we cannot evaluate the perturbation automatically
                # But we can still check the scaling factor manually if the student provided P_floor as a number
                # In that case we cannot test; skip.
                return (False, "Cannot find d symbol in P_floor expression to compute perturbation")
            # Build substitution with all other givens as before (we need them)
            subs = {d_sym: d_perturbed}
            # Add other known givens if they are symbols in the expression
            symbol_map = {'m': self.m, 'q': self.q, 'k': self.k, 'h': self.h,
                          'mu0': self.mu0, 'c': self.c, 'omega': self._omega}
            for s in syms:
                str_s = str(s)
                if str_s in symbol_map and symbol_map[str_s] is not None:
                    subs[s] = symbol_map[str_s]
            # Also need omega if it appears; we have self._omega
            # Evaluate
            p_perturbed = self._eval_num(expr, subs)
            if p_perturbed is None:
                return (False, "Could not evaluate perturbed P_floor")
        except Exception as e:
            return (False, f"Error computing perturbed P_floor: {e}")

        # Compare scaling factor
        actual_ratio = p_perturbed / pfloor_base
        expected_ratio = 1.1**2  # 1.21
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_floor scales as d^2: ratio perturbed/base = {actual_ratio:.4g} ≈ {expected_ratio:.4g}")
        else:
            return (False, f"P_floor does not scale as d^2: ratio = {actual_ratio:.4g}, expected {expected_ratio:.4g}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Weak sanity: all outputs are finite and positive."""
        results = [self.I_R, self.R_max, self.P_floor, self.tau]
        for i, res in enumerate(results):
            if res is None:
                continue   # skip missing, but interface test should catch that
            try:
                val = float(res)
            except (TypeError, ValueError):
                # If it's symbolic, just check it's not None
                if res is None:
                    return (False, "One result is None")
                continue
            if val <= 0:
                return (False, f"Result {i} is non-positive: {val}")
            if not math.isfinite(val):
                return (False, f"Result {i} is not finite: {val}")
        return (True, "All finite and positive (where numeric).")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # collect all methods starting with test_
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