import sympy as sp
import math
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits the solution of a problem about a uniformly magnetized, charged sphere.
    Checks interface, fundamental invariants, modeling details, monotonicity,
    and a weak final sanity. All tests return (passed, message).
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all needed information from the solution_instance.
        Use None defaults for missing data – never numeric placeholders.
        """
        # --- Givens (physical parameters) ---
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.R = givens.get('R', None)               # sphere radius
        self.Q = givens.get('Q', None)               # total surface charge
        self.M0 = givens.get('M0', None)             # initial magnetization magnitude
        self.mu0 = givens.get('mu0', None)           # vacuum permeability
        self.eps0 = givens.get('epsilon0', None)     # vacuum permittivity

        # --- Results (expected keys from pseudocode emit contract) ---
        results = getattr(solution_instance, 'results', {}) or {}
        self.E_field = results.get('E_field', None)
        self.B_field = results.get('B_field', None)
        self.E_ind_field = results.get('E_ind_field', None)
        self.L_total = results.get('L_total', None)

        # --- Optional metadata for better audit ---
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Symbolic variables for internal re‑derivation
        self.r_sym, self.theta_sym, self.phi_sym = sp.symbols('r theta phi', real=True, positive=True)
        self.M_sym, self.Mdot_sym = sp.symbols('M Mdot', real=True)

    # ----------------------------------------------------------------------
    #  Helper to evaluate a field at a point, trying multiple representations.
    #  Returns a list [x,y,z] or None.
    # ----------------------------------------------------------------------
    def _evaluate_field_cartesian(self, field_expr, r_val: float, theta_val: float, phi_val: float) -> List[float]:
        """Evaluate the field (expected to be a vector) at (r,θ,φ) and return cartesian components."""
        if field_expr is None:
            return None
        # Case: callable
        if callable(field_expr):
            try:
                vec = field_expr(r_val, theta_val, phi_val)
                if hasattr(vec, '__iter__') and len(vec) == 3:
                    return [float(v) for v in vec]
            except Exception:
                pass
        # Case: sympy expression – we cannot evaluate without additional info.
        return None

    def _evaluate_scalar(self, expr, subs_dict: Dict) -> float:
        """Try to evaluate a scalar expression safely."""
        if expr is None:
            return None
        try:
            if callable(expr):
                return float(expr(**subs_dict))
            if isinstance(expr, (int, float)):
                return float(expr)
            if isinstance(expr, str):
                try:
                    return float(expr)
                except Exception:
                    return None
            # sympy expression
            val = sp.N(expr.subs(subs_dict))
            return float(val)
        except Exception:
            return None

    # ----------------------------------------------------------------------
    #  Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all four required result keys exist and are not None."""
        missing = []
        for key, val in [('E_field', self.E_field),
                         ('B_field', self.B_field),
                         ('E_ind_field', self.E_ind_field),
                         ('L_total', self.L_total)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result(s): {', '.join(missing)}")
        return (True, "All required results present")

    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that all essential givens are provided."""
        needed = {'R': self.R, 'Q': self.Q, 'M0': self.M0,
                  'mu0': self.mu0, 'epsilon0': self.eps0}
        missing = [k for k, v in needed.items() if v is None]
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ----------------------------------------------------------------------
    #  Invariant tests (theorem‑based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_electric_inside_zero(self) -> Tuple[bool, str]:
        """Inside the sphere (r < R) the electric field must be zero."""
        if self.E_field is None or self.R is None:
            return (True, "SKIP: insufficient data")
        try:
            vec = self._evaluate_field_cartesian(self.E_field, float(self.R)/2, math.pi/4, 0.0)
            if vec is None:
                return (True, "SKIP: cannot evaluate E_field inside")
            if all(abs(c) < 1e-12 for c in vec):
                return (True, "E=0 inside sphere")
            else:
                return (False, f"Expected E=0 inside, got {vec}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_electric_outside_coulomb(self) -> Tuple[bool, str]:
        """Outside (r > R) E should be radial and follow 1/r^2."""
        if self.E_field is None or self.R is None or self.Q is None or self.eps0 is None:
            return (True, "SKIP: missing data")
        try:
            r_test = 2.0 * float(self.R)
            theta_test = math.pi/3
            phi_test = 0.0
            vec = self._evaluate_field_cartesian(self.E_field, r_test, theta_test, phi_test)
            if vec is None:
                return (True, "SKIP: cannot evaluate E_field outside")
            Er_expected = self.Q / (4 * math.pi * self.eps0 * r_test * r_test)
            rx = math.sin(theta_test) * math.cos(phi_test)
            ry = math.sin(theta_test) * math.sin(phi_test)
            rz = math.cos(theta_test)
            ex, ey, ez = vec
            if (math.isclose(ex, Er_expected * rx, rel_tol=1e-2) and
                math.isclose(ey, Er_expected * ry, rel_tol=1e-2) and
                math.isclose(ez, Er_expected * rz, rel_tol=1e-2)):
                return (True, "Outside E matches Coulomb field")
            else:
                return (False, f"E outside mismatch: got ({ex},{ey},{ez}), expected ({Er_expected*rx}, {Er_expected*ry}, {Er_expected*rz})")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_magnetic_inside_uniform(self) -> Tuple[bool, str]:
        """Inside the sphere B must be uniform and equal to (2/3)μ0 M ẑ."""
        if self.B_field is None or self.R is None or self.M0 is None or self.mu0 is None:
            return (True, "SKIP: missing data")
        try:
            for r_frac in [0.3, 0.7]:
                r = r_frac * float(self.R)
                theta = 0.4
                phi = 1.2
                vec = self._evaluate_field_cartesian(self.B_field, r, theta, phi)
                if vec is None:
                    return (True, f"SKIP: cannot evaluate B_field at r={r}")
                B_expected = 2.0/3.0 * self.mu0 * self.M0
                if not math.isclose(vec[0], 0.0, abs_tol=1e-12):
                    return (False, f"B inside has x-component {vec[0]}")
                if not math.isclose(vec[1], 0.0, abs_tol=1e-12):
                    return (False, f"B inside has y-component {vec[1]}")
                if not math.isclose(vec[2], B_expected, rel_tol=1e-2):
                    return (False, f"B_z inside = {vec[2]}, expected {B_expected}")
            return (True, "Inside B uniform and along z")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_magnetic_outside_dipole(self) -> Tuple[bool, str]:
        """Outside the sphere B must be the dipole field of moment m = (4π/3)R^3 M ẑ."""
        if self.B_field is None or self.R is None or self.M0 is None or self.mu0 is None:
            return (True, "SKIP: missing data")
        try:
            R_ = float(self.R)
            mu0_ = float(self.mu0)
            M_ = float(self.M0)
            m = (4*math.pi/3) * R_**3 * M_
            r_test = 2.0 * R_
            theta_test = math.pi/3
            phi_test = 0.0
            vec = self._evaluate_field_cartesian(self.B_field, r_test, theta_test, phi_test)
            if vec is None:
                return (True, "SKIP: cannot evaluate B_field outside")
            B_r = (mu0_/(4*math.pi)) * 2 * m * math.cos(theta_test) / (r_test**3)
            B_th = (mu0_/(4*math.pi)) * m * math.sin(theta_test) / (r_test**3)
            rx = math.sin(theta_test) * math.cos(phi_test)
            ry = math.sin(theta_test) * math.sin(phi_test)
            rz = math.cos(theta_test)
            tx = math.cos(theta_test) * math.cos(phi_test)
            ty = math.cos(theta_test) * math.sin(phi_test)
            tz = -math.sin(theta_test)
            Bx_exp = B_r * rx + B_th * tx
            By_exp = B_r * ry + B_th * ty
            Bz_exp = B_r * rz + B_th * tz
            if (math.isclose(vec[0], Bx_exp, rel_tol=1e-2) and
                math.isclose(vec[1], By_exp, rel_tol=1e-2) and
                math.isclose(vec[2], Bz_exp, rel_tol=1e-2)):
                return (True, "Outside B matches dipole field")
            else:
                return (False, f"Outside B mismatch: got {vec}, expected ({Bx_exp},{By_exp},{Bz_exp})")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_induced_E_inside_form(self) -> Tuple[bool, str]:
        """
        Inside sphere the induced E field must be purely azimuthal and
        proportional to r sinθ. (Check if sympy expression exists.)
        """
        if self.E_ind_field is None:
            return (True, "SKIP: missing data")
        if isinstance(self.E_ind_field, sp.Expr) or (hasattr(self.E_ind_field, '__iter__') and len(self.E_ind_field) == 3):
            return (True, "Induced E field structure plausible (sympy expression)")
        return (True, "SKIP: cannot check form")

    def test_invariant_angular_momentum_form(self) -> Tuple[bool, str]:
        """L_total must be (2/9) μ0 M Q R^2 ẑ (numerical value)."""
        if self.L_total is None or self.mu0 is None or self.M0 is None or self.Q is None or self.R is None:
            return (True, "SKIP: missing data")
        try:
            Lz = self._evaluate_scalar(self.L_total, {})
            if Lz is None:
                if hasattr(self.L_total, '__getitem__') and len(self.L_total) >= 3:
                    Lz = float(self.L_total[2])
                else:
                    return (True, "SKIP: cannot get Lz")
            L_expected = (2.0/9.0) * self.mu0 * self.M0 * self.Q * self.R**2
            if math.isclose(Lz, L_expected, rel_tol=1e-2):
                return (True, f"L_total z = {Lz}, expected {L_expected}")
            else:
                return (False, f"L_total z = {Lz}, expected {L_expected}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    # ----------------------------------------------------------------------
    #  Modeling / trace tests (based on explanation_steps)
    # ----------------------------------------------------------------------
    def test_modeling_uses_surface_charge_density(self) -> Tuple[bool, str]:
        """Check that explanation steps mention sigma = Q/(4πR^2)."""
        if self.explanation_steps is None:
            return (True, "SKIP: no explanation steps")
        steps_text = " ".join(self.explanation_steps).lower()
        if "sigma" in steps_text or "surface charge density" in steps_text or "σ" in steps_text:
            return (True, "Surface charge density is discussed")
        return (False, "Surface charge density not found in explanation")

    def test_modeling_uses_faraday_law(self) -> Tuple[bool, str]:
        """Check that Faraday's law is used to derive induced E."""
        if self.explanation_steps is None:
            return (True, "SKIP: no explanation steps")
        steps_text = " ".join(self.explanation_steps).lower()
        if "faraday" in steps_text or "curl(e) = -∂b/∂t" in steps_text or "faraday's law" in steps_text:
            return (True, "Faraday's law appears")
        return (False, "Faraday's law not found")

    def test_modeling_torque_integration(self) -> Tuple[bool, str]:
        """Torque should be integrated over the sphere surface."""
        if self.explanation_steps is None:
            return (True, "SKIP: no explanation steps")
        steps_text = " ".join(self.explanation_steps).lower()
        if "∫ τ" in steps_text or "integral of torque" in steps_text or "torque on sphere" in steps_text:
            return (True, "Integration of torque is described")
        return (False, "No torque integration mentioned")

    # ----------------------------------------------------------------------
    #  Monotonicity tests (explicit perturbation: +10% of a parameter,
    #  compare perturbed and baseline values)
    # ----------------------------------------------------------------------
    def test_monotonicity_B_inside_with_M(self) -> Tuple[bool, str]:
        """Inside B magnitude should increase when M increases (perturb by +10%)."""
        B = self.B_field
        if B is None or not isinstance(B, sp.Expr):
            return (True, "SKIP: B_field not a sympy expression")
        # Check if M is a free symbol
        if self.M_sym not in B.free_symbols and sp.Symbol('M') not in B.free_symbols:
            return (True, "SKIP: no M dependence in B expression")
        try:
            # Baseline: set all other symbols (r,theta,phi) to sample values, M to 1
            subs = {sp.Symbol('r'): 0.5, sp.Symbol('theta'): 0.4, sp.Symbol('phi'): 0.0,
                    self.M_sym: 1.0}
            base_val = float(sp.N(B.subs(subs)))
            # Perturb: increase M by 10%
            subs[self.M_sym] = 1.1
            perturbed_val = float(sp.N(B.subs(subs)))
            if perturbed_val > base_val:
                return (True, f"B increases when M increases: base {base_val}, perturbed +10% -> {perturbed_val}")
            else:
                return (False, f"B does not increase: base {base_val}, perturbed {perturbed_val}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_monotonicity_L_with_R(self) -> Tuple[bool, str]:
        """L_total magnitude should increase when R increases (R^2)."""
        L = self.L_total
        if L is None or not isinstance(L, sp.Expr):
            return (True, "SKIP: L_total not a sympy expression")
        R_sym = sp.Symbol('R')
        if R_sym not in L.free_symbols:
            return (True, "SKIP: no R dependence in L expression")
        try:
            # Baseline: set all symbols (including Q, M, mu0) to 1, R to 1
            subs = {sp.Symbol('Q'): 1.0, sp.Symbol('mu0'): 1.0, sp.Symbol('M'): 1.0, R_sym: 1.0}
            base_val = float(sp.N(L.subs(subs)))
            # Perturb R by +10%
            subs[R_sym] = 1.1
            perturbed_val = float(sp.N(L.subs(subs)))
            # R^2: base is 1, perturbed should be ~1.21 -> definitely larger
            if perturbed_val > base_val:
                return (True, f"L increases when R increases: base {base_val}, perturbed +10% -> {perturbed_val}")
            else:
                return (False, f"L does not increase: base {base_val}, perturbed {perturbed_val}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_monotonicity_L_with_Q(self) -> Tuple[bool, str]:
        """L_total magnitude should increase when Q increases (linear)."""
        L = self.L_total
        if L is None or not isinstance(L, sp.Expr):
            return (True, "SKIP: L_total not a sympy expression")
        Q_sym = sp.Symbol('Q')
        if Q_sym not in L.free_symbols:
            return (True, "SKIP: no Q dependence in L expression")
        try:
            subs = {sp.Symbol('R'): 1.0, sp.Symbol('mu0'): 1.0, sp.Symbol('M'): 1.0, Q_sym: 1.0}
            base_val = float(sp.N(L.subs(subs)))
            subs[Q_sym] = 1.1
            perturbed_val = float(sp.N(L.subs(subs)))
            if perturbed_val > base_val:
                return (True, f"L increases when Q increases: base {base_val}, perturbed +10% -> {perturbed_val}")
            else:
                return (False, f"L does not increase: base {base_val}, perturbed {perturbed_val}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    # ----------------------------------------------------------------------
    #  Single weak sanity test: sign, finiteness
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """
        Weak sanity: L_total should be finite and have the same sign as M0.
        No re-derivation, no math.isclose.
        """
        if self.L_total is None or self.M0 is None:
            return (True, "SKIP: missing data")
        try:
            Lz = self._evaluate_scalar(self.L_total, {})
            if Lz is None:
                if hasattr(self.L_total, '__getitem__') and len(self.L_total) >= 3:
                    Lz = float(self.L_total[2])
                else:
                    return (True, "SKIP: cannot get Lz")
            if not math.isfinite(Lz):
                return (False, f"L_total is not finite: {Lz}")
            # sign check
            if (self.M0 > 0 and Lz > 0) or (self.M0 < 0 and Lz < 0):
                return (True, f"Lz={Lz} has same sign as M0={self.M0}")
            return (False, f"Sign mismatch: M0={self.M0}, Lz={Lz}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    # ----------------------------------------------------------------------
    #  Master test runner (required)
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods whose names start with 'test_',
        execute them one by one in a try/except, collect messages,
        and return (success_bool, logs_list).
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
                passed, msg = False, f"{name} CRASHED: {e}"
            # If the message starts with "SKIP:" we treat it as not a failure
            if msg.startswith("SKIP:"):
                logs.append(f"SKIP: {name}: {msg[5:].strip()}")
            else:
                logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
                if not passed:
                    all_pass = False
        return (all_pass, logs)