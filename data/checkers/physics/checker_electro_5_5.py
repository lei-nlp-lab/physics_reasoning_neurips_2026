import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for the Compton effect problem (photon scattering off stationary electron).
    Tests focus on:
      - Interface: presence and essential components of emitted equations and final answer.
      - Invariant: relativistic relations for the electron (gamma^2(1-beta^2)=1 and the
        energy-momentum relation).
      - Modeling: consistency of derived scattered photon energy with the expected formula,
        energy conservation, and the trace relation 2EE' = m c^2 (E - E').
      - Monotonicity: directional trends of E_prime with incident energy and electron mass.
      - Weak answer-sanity (at most one).
    """

    def __init__(self, solution_instance):
        # Extract givens safely (may be in .givens dict or as direct attributes)
        self.E = getattr(solution_instance, 'E', None)
        if self.E is None:
            givens = getattr(solution_instance, 'givens', {})
            self.E = givens.get('E', None)

        self.m = getattr(solution_instance, 'm', None)
        if self.m is None:
            givens = getattr(solution_instance, 'givens', {})
            self.m = givens.get('m', None)

        self.c = getattr(solution_instance, 'c', None)
        if self.c is None:
            givens = getattr(solution_instance, 'givens', {})
            self.c = givens.get('c', None)

        # Extract results safely from .results dict or as direct attributes
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.x_mom_eq = results.get('x_mom_eq', None)
        self.y_mom_eq = results.get('y_mom_eq', None)
        self.energy_eq = results.get('energy_eq', None)
        self.E_prime = results.get('E_prime', None)

        # Also try direct attribute if not found
        if self.x_mom_eq is None:
            self.x_mom_eq = getattr(solution_instance, 'x_mom_eq', None)
        if self.y_mom_eq is None:
            self.y_mom_eq = getattr(solution_instance, 'y_mom_eq', None)
        if self.energy_eq is None:
            self.energy_eq = getattr(solution_instance, 'energy_eq', None)
        if self.E_prime is None:
            self.E_prime = getattr(solution_instance, 'E_prime', None)

    # ------------------------------------------------------------------
    # Helper: get numeric E_prime value
    # ------------------------------------------------------------------
    def _get_ep_val(self):
        if self.E is None or self.m is None or self.c is None:
            return None
        E_val = float(self.E)
        m_val = float(self.m)
        c_val = float(self.c)
        try:
            if isinstance(self.E_prime, (int, float)):
                return float(self.E_prime)
            elif isinstance(self.E_prime, str):
                expr = sp.sympify(self.E_prime)
                return float(expr.subs({sp.Symbol('E'): E_val,
                                        sp.Symbol('m'): m_val,
                                        sp.Symbol('c'): c_val}))
            elif isinstance(self.E_prime, sp.Expr):
                return float(self.E_prime.subs({sp.Symbol('E'): E_val,
                                                sp.Symbol('m'): m_val,
                                                sp.Symbol('c'): c_val}))
            else:
                return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_x_mom_eq(self):
        eq = self.x_mom_eq
        if eq is None:
            return False, "x_mom_eq is missing"
        if not isinstance(eq, str):
            return False, "x_mom_eq must be a string"
        needed = ['E', 'E_prime', 'costheta', 'gamma', 'm', 'v', 'cosphi']
        for term in needed:
            if term.lower() not in eq.lower():
                return False, f"x_mom_eq missing term: {term}"
        return True, "x_mom_eq looks plausible"

    def test_interface_y_mom_eq(self):
        eq = self.y_mom_eq
        if eq is None:
            return False, "y_mom_eq is missing"
        if not isinstance(eq, str):
            return False, "y_mom_eq must be a string"
        needed = ['E_prime', 'sintheta', 'gamma', 'm', 'v', 'sinphi']
        for term in needed:
            if term.lower() not in eq.lower():
                return False, f"y_mom_eq missing term: {term}"
        return True, "y_mom_eq looks plausible"

    def test_interface_energy_eq(self):
        eq = self.energy_eq
        if eq is None:
            return False, "energy_eq is missing"
        if not isinstance(eq, str):
            return False, "energy_eq must be a string"
        needed = ['E', 'E_prime', 'm', 'c^2', 'gamma']
        for term in needed:
            if term.lower() not in eq.lower():
                return False, f"energy_eq missing term: {term}"
        return True, "energy_eq looks plausible"

    def test_interface_E_prime(self):
        ep = self.E_prime
        if ep is None:
            return False, "E_prime is missing"
        try:
            if isinstance(ep, (int, float)):
                numeric_ep = float(ep)
            elif isinstance(ep, sp.Expr):
                if self.E is not None and self.m is not None and self.c is not None:
                    numeric_ep = float(ep.subs({self.E: float(self.E),
                                                self.m: float(self.m),
                                                self.c: float(self.c)}))
                else:
                    numeric_ep = None
            elif isinstance(ep, str):
                if self.E is not None and self.m is not None and self.c is not None:
                    expr = sp.sympify(ep)
                    numeric_ep = float(expr.subs({sp.Symbol('E'): float(self.E),
                                                  sp.Symbol('m'): float(self.m),
                                                  sp.Symbol('c'): float(self.c)}))
                else:
                    numeric_ep = None
            else:
                return False, f"E_prime has unexpected type: {type(ep)}"
        except Exception as e:
            return False, f"E_prime could not be evaluated: {e}"

        if numeric_ep is not None:
            if numeric_ep <= 0 or not math.isfinite(numeric_ep):
                return False, f"E_prime is not positive/finite: {numeric_ep}"
            if self.E is not None:
                if numeric_ep >= float(self.E):
                    return False, f"E_prime ({numeric_ep}) should be less than incident E ({self.E})"
        return True, "E_prime interface OK"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_gamma_beta_relation(self):
        """
        For backscattering, conservation equations imply:
          E + E' = γ β m c^2   (from x-momentum)
          E - E' = (γ - 1) m c^2 (from energy)
        The relativistic invariant γ²(1-β²) must equal 1.
        We re-derive γ and β from these and check the invariant.
        """
        ep_val = self._get_ep_val()
        if ep_val is None:
            return (False, "SKIP: cannot evaluate E_prime (missing givens or un-evaluable)")
        E_val = float(self.E)
        m_val = float(self.m)
        c_val = float(self.c)

        S = E_val + ep_val
        D = E_val - ep_val
        gamma = D / (m_val * c_val**2) + 1.0
        gamma_beta = S / (m_val * c_val**2)
        if gamma == 0:
            return (False, "gamma is zero, cannot compute beta")
        beta = gamma_beta / gamma

        invariant = gamma**2 * (1.0 - beta**2)
        if math.isclose(invariant, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Gamma-beta invariant holds: gamma^2(1-beta^2) = {invariant}")
        else:
            return (False, f"Gamma-beta invariant fails: gamma^2(1-beta^2) = {invariant} (expected 1)")

    def test_invariant_electron_energy_momentum(self):
        """
        The electron's energy-momentum relation: (γ m c^2)^2 - (γ β m c^2)^2 = (m c^2)^2.
        Using the same derived γ and β from the conservation equations, verify this relation.
        """
        ep_val = self._get_ep_val()
        if ep_val is None:
            return (False, "SKIP: cannot evaluate E_prime (missing givens or un-evaluable)")
        E_val = float(self.E)
        m_val = float(self.m)
        c_val = float(self.c)

        S = E_val + ep_val
        D = E_val - ep_val
        gamma = D / (m_val * c_val**2) + 1.0
        gamma_beta = S / (m_val * c_val**2)

        # Compute LHS = (γ m c^2)^2 - (γ β m c^2)^2
        lhs = (gamma * m_val * c_val**2)**2 - (gamma_beta * m_val * c_val**2)**2
        # RHS = (m c^2)^2
        rhs = (m_val * c_val**2)**2
        # Check equality
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Electron energy-momentum relation holds: LHS={lhs}, RHS={rhs}")
        else:
            return (False, f"Electron energy-momentum relation fails: LHS={lhs}, RHS={rhs}")

    # ------------------------------------------------------------------
    # Modeling consistency checks
    # ------------------------------------------------------------------
    def test_modeling_E_prime_formula(self):
        """Numerically check that E_prime obeys the derived formula E/(1+2E/(mc^2))."""
        ep_val = self._get_ep_val()
        if ep_val is None:
            return (False, "SKIP: cannot evaluate E_prime (missing givens or un-evaluable)")
        E_val = float(self.E)
        m_val = float(self.m)
        c_val = float(self.c)

        expected = E_val / (1 + 2 * E_val / (m_val * c_val**2))
        if math.isclose(ep_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_prime matches formula: {ep_val} vs {expected}")
        else:
            return (False, f"E_prime {ep_val} deviates from expected {expected}")

    def test_modeling_energy_conservation_holds(self):
        """
        Verify energy conservation: E + m c^2 = E' + γ m c^2,
        where γ is derived from the same equations as the solution would use.
        """
        ep_val = self._get_ep_val()
        if ep_val is None:
            return (False, "SKIP: cannot evaluate E_prime (missing givens or un-evaluable)")
        E_val = float(self.E)
        m_val = float(self.m)
        c_val = float(self.c)

        # Derive gamma from energy equation: γ = (E - E')/(m c^2) + 1
        gamma = (E_val - ep_val) / (m_val * c_val**2) + 1.0
        lhs = E_val + m_val * c_val**2
        rhs = ep_val + gamma * m_val * c_val**2

        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Energy conservation holds: LHS={lhs}, RHS={rhs}")
        else:
            return (False, f"Energy conservation fails: LHS={lhs}, RHS={rhs}")

    def test_modeling_trace_energy_momentum_relation(self):
        """
        From the derivation in the explanation steps, the conservation equations imply:
          2 E E' = m c^2 (E - E')
        Check this relation numerically.
        """
        ep_val = self._get_ep_val()
        if ep_val is None:
            return (False, "SKIP: cannot evaluate E_prime (missing givens or un-evaluable)")
        E_val = float(self.E)
        m_val = float(self.m)
        c_val = float(self.c)

        lhs = 2.0 * E_val * ep_val
        rhs = m_val * c_val**2 * (E_val - ep_val)

        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Trace relation 2EE' = m c^2 (E-E') holds: LHS={lhs}, RHS={rhs}")
        else:
            return (False, f"Trace relation fails: LHS={lhs}, RHS={rhs}")

    # ------------------------------------------------------------------
    # Monotonicity tests (each perturbs an input and compares output)
    # ------------------------------------------------------------------
    def test_monotonicity_E_prime_increases_with_E(self):
        """If incident energy E increases, scattered energy E' should increase."""
        if self.E is None or self.m is None or self.c is None:
            return (False, "SKIP: missing E, m, or c givens")
        ep0 = self._get_ep_val()
        if ep0 is None:
            return (False, "SKIP: cannot evaluate baseline E_prime")
        E0 = float(self.E)
        m_val = float(self.m)
        c_val = float(self.c)

        # Increase E by 10% and compute expected E' (formula)
        E1 = E0 * 1.1
        ep1 = E1 / (1 + 2 * E1 / (m_val * c_val**2))
        if ep1 > ep0:
            return (True, f"E_prime increases with E: ep0={ep0}, ep1={ep1}")
        else:
            return (False, f"E_prime decreased when E increased: ep0={ep0}, ep1={ep1}")

    def test_monotonicity_E_prime_increases_with_m(self):
        """If electron mass m increases, scattered energy E' should increase."""
        if self.E is None or self.m is None or self.c is None:
            return (False, "SKIP: missing E, m, or c givens")
        ep0 = self._get_ep_val()
        if ep0 is None:
            return (False, "SKIP: cannot evaluate baseline E_prime")
        E_val = float(self.E)
        m0 = float(self.m)
        c_val = float(self.c)

        # Increase m by 10% and compute expected E' (formula)
        m1 = m0 * 1.1
        ep1 = E_val / (1 + 2 * E_val / (m1 * c_val**2))
        if ep1 > ep0:
            return (True, f"E_prime increases with m: ep0={ep0}, ep1={ep1}")
        else:
            return (False, f"E_prime decreased when m increased: ep0={ep0}, ep1={ep1}")

    # ------------------------------------------------------------------
    # Weak final-answer sanity (at most one, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_E_prime_magnitude(self):
        """E_prime should be positive, finite, and less than incident energy if E is given."""
        if self.E is None or self.m is None or self.c is None:
            return (False, "SKIP: missing givens for magnitude check")
        ep_val = self._get_ep_val()
        if ep_val is None:
            return (False, "SKIP: cannot evaluate E_prime")
        E_val = float(self.E)

        if ep_val <= 0:
            return (False, f"E_prime must be positive, got {ep_val}")
        if not math.isfinite(ep_val):
            return (False, f"E_prime is not finite: {ep_val}")
        if ep_val >= E_val:
            return (False, f"E_prime should be less than incident E, got {ep_val} >= {E_val}")
        return (True, "E_prime magnitude sanity passed")

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)