import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract given masses
        self.m_p = getattr(solution_instance, 'm_p', None)
        self.m_D = getattr(solution_instance, 'm_D', None)
        self.m_pi = getattr(solution_instance, 'm_pi', None)

        # Safely extract results dict
        self.results = getattr(solution_instance, 'results', {})
        self.T = self.results.get('T', None)
        self.dP_dOmega = self.results.get('dP_dOmega', None)

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that both result keys 'T' and 'dP_dOmega' exist."""
        if 'T' not in self.results:
            return (False, "Missing required key 'T' in results")
        if 'dP_dOmega' not in self.results:
            return (False, "Missing required key 'dP_dOmega' in results")
        return (True, "All required keys present")

    # ------------------------------------------------------------
    # Invariant checks (theorem-based)
    # ------------------------------------------------------------
    def test_invariant_squared_mass_threshold(self):
        """Verify that invariant mass squared equals (m_D + m_pi)^2 at threshold."""
        if None in (self.m_p, self.m_D, self.m_pi, self.T):
            return (False, "SKIP: missing m_p, m_D, m_pi, or T")
        E_p = self.m_p + self.T
        p_p = math.sqrt(E_p**2 - self.m_p**2)
        s = (E_p + self.m_p)**2 - p_p**2
        s_th = (self.m_D + self.m_pi)**2
        if math.isclose(s, s_th, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Invariant mass s={s:.1f} matches threshold s_th={s_th:.1f}")
        else:
            return (False, f"Invariant mass s={s:.1f} != threshold s_th={s_th:.1f}")

    def test_invariant_lab_energy_formula(self):
        """Check that m_p + T equals the threshold lab energy derived from invariant mass."""
        if None in (self.m_p, self.m_D, self.m_pi, self.T):
            return (False, "SKIP: missing masses or T")
        E_th_formula = ((self.m_D + self.m_pi)**2 - 2*self.m_p**2) / (2*self.m_p)
        actual_E = self.m_p + self.T
        if math.isclose(actual_E, E_th_formula, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Lab energy E={actual_E:.1f} matches threshold formula E_th={E_th_formula:.1f}")
        else:
            return (False, f"Lab energy E={actual_E:.1f} != threshold formula E_th={E_th_formula:.1f}")

    def test_invariant_kinetic_energy_formula(self):
        """Check that T matches the standard threshold formula."""
        if None in (self.m_p, self.m_D, self.m_pi, self.T):
            return (False, "SKIP: missing masses or T")
        T_formula = ((self.m_D + self.m_pi)**2 - 4*self.m_p**2) / (2*self.m_p)
        if math.isclose(self.T, T_formula, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T={self.T:.1f} matches formula T_formula={T_formula:.1f}")
        else:
            return (False, f"T={self.T:.1f} != formula T_formula={T_formula:.1f}")

    # ------------------------------------------------------------
    # Modeling checks (quality of the angular distribution)
    # ------------------------------------------------------------
    def test_modeling_dP_dOmega_expressions(self):
        """Check that dP_dOmega is a non-trivial expression containing angular dependence."""
        if self.dP_dOmega is None:
            return (False, "SKIP: dP_dOmega not provided")
        expr_str = str(self.dP_dOmega)
        if 'cos' in expr_str or 'theta' in expr_str:
            return (True, "dP_dOmega expression includes angular dependence")
        else:
            return (False, "dP_dOmega expression does not contain 'cos' or 'theta'; may be a constant")

    def test_modeling_dP_dOmega_denominator_structure(self):
        """Check that the expression has a denominator of the form (1 - A*cosθ)."""
        if self.dP_dOmega is None:
            return (False, "SKIP: dP_dOmega not provided")
        try:
            # Try to parse as symbolic
            if isinstance(self.dP_dOmega, str):
                expr = sp.sympify(self.dP_dOmega)
            else:
                expr = sp.sympify(self.dP_dOmega)
            denom = sp.denom(expr)
            if denom == sp.Integer(1):
                return (False, "dP_dOmega does not have a denominator; expected 1/(1 - A cosθ) structure")
            # Check that the denominator contains cosθ symbol
            if denom.has(sp.cos(sp.Symbol('theta'))) or denom.has(sp.cos(sp.Symbol('theta_lab'))):
                return (True, "Denominator contains cosθ as expected")
            else:
                return (False, f"Denominator {denom} does not contain cosθ")
        except Exception as e:
            return (False, f"Could not parse dP_dOmega as symbolic: {e}")

    # ------------------------------------------------------------
    # Monotonicity check (explicit perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_T_increases_with_m_pi(self):
        """Check that a 10% increase in m_pi leads to an increase in T."""
        if None in (self.m_p, self.m_D, self.m_pi, self.T):
            return (False, "SKIP: missing masses or T")
        # Compute baseline T using formula
        T_base = ((self.m_D + self.m_pi)**2 - 4*self.m_p**2) / (2*self.m_p)
        # Perturb m_pi upward by 10%
        m_pi_pert = self.m_pi * 1.10
        T_pert = ((self.m_D + m_pi_pert)**2 - 4*self.m_p**2) / (2*self.m_p)
        if T_pert > T_base:
            return (True, f"T increases from {T_base:.1f} to {T_pert:.1f} when m_pi increases by 10%")
        else:
            return (False, f"T did not increase: baseline {T_base:.1f}, perturbed {T_pert:.1f}")

    # ------------------------------------------------------------
    # Weak answer sanity (only one method allowed)
    # ------------------------------------------------------------
    def test_answer_sanity(self):
        """Weak sanity: T positive and bounded; dP_dOmega not empty."""
        issues = []
        if self.T is not None:
            if not (0 < self.T < 10000):
                issues.append(f"T={self.T} out of reasonable range (0, 10000)")
            if not math.isfinite(self.T):
                issues.append("T is not finite")
        else:
            issues.append("T not provided for sanity check")
        if self.dP_dOmega is None:
            issues.append("dP_dOmega not provided")
        else:
            # Check it's a non-empty string/expression
            if not str(self.dP_dOmega).strip():
                issues.append("dP_dOmega is empty")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All sanity checks passed")

    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
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