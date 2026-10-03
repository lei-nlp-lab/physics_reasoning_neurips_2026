import math
import sympy as sp
from typing import Any, Dict, Tuple, List, Optional, Callable

class PhysicsChecker:
    """
    Audits solutions to the problem:
    "Derive ρ_- = -γ²ρ_+ using Lorentz transformation of charge densities
     and neutrality condition in electron rest frame."
    """
    
    def __init__(self, solution_instance: Any):
        # --- Safe extraction of givens ---
        # Try results dict first, then direct attributes
        self.results = getattr(solution_instance, 'results', {})
        
        # Givens: rho_plus, v, c, gamma
        self.rho_plus = getattr(solution_instance, 'rho_plus', None)
        if self.rho_plus is None:
            self.rho_plus = self.results.get('rho_plus', None)
            
        self.v = getattr(solution_instance, 'v', None)
        if self.v is None:
            self.v = self.results.get('v', None)
            
        self.c = getattr(solution_instance, 'c', None)
        if self.c is None:
            self.c = self.results.get('c', None)
            
        self.gamma = getattr(solution_instance, 'gamma', None)
        if self.gamma is None:
            self.gamma = self.results.get('gamma', None)
        
        # Result key: rho_minus
        self.rho_minus = self.results.get('rho_minus', None)
        
        # Explanation steps (optional, used for modeling checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = solution_instance.results.get('explanation_steps', None) \
                if hasattr(solution_instance, 'results') else None
        
        # Internal constants
        self.mu0 = sp.Symbol('mu0', positive=True)  # not needed numerically
        self.eps0 = sp.Symbol('eps0', positive=True)
        
    def _is_numeric(self, x: Any) -> bool:
        """Check if value is a finite number (int or float)."""
        return isinstance(x, (int, float)) and math.isfinite(x)
    
    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that all required givens are present and numeric."""
        missing = []
        for name, val in [('rho_plus', self.rho_plus), ('v', self.v),
                          ('c', self.c), ('gamma', self.gamma)]:
            if not self._is_numeric(val):
                missing.append(name)
        if missing:
            return (False, f"Missing or non-numeric givens: {', '.join(missing)}")
        # Additional sanity: v/c should be less than 1
        if self.v is not None and self.c is not None and self.c != 0:
            if abs(self.v / self.c) >= 1:
                return (False, f"v/c = {self.v/self.c} >= 1, not physical")
        return (True, "All givens present and numeric")
    
    def test_interface_results(self) -> Tuple[bool, str]:
        """Check that rho_minus exists in results and is numeric."""
        if self.rho_minus is None:
            return (False, "Result 'rho_minus' is missing")
        if not self._is_numeric(self.rho_minus):
            return (False, f"rho_minus is not numeric: {self.rho_minus}")
        return (True, "rho_minus present and numeric")
    
    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use isclose)
    # ------------------------------------------------------------------
    def test_invariant_gamma_definition(self) -> Tuple[bool, str]:
        """Verify gamma = 1/sqrt(1 - v²/c²)"""
        if any(x is None for x in [self.gamma, self.v, self.c]):
            return (False, "SKIP: gamma, v, or c not available")
        if self.c == 0:
            return (False, "c = 0, cannot compute")
        beta = self.v / self.c
        if abs(beta) >= 1:
            return (False, f"v/c = {beta} >= 1, unphysical")
        expected_gamma = 1 / math.sqrt(1 - beta**2)
        if math.isclose(self.gamma, expected_gamma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Gamma matches 1/sqrt(1 - v²/c²)")
        else:
            return (False, f"Gamma = {self.gamma}, expected {expected_gamma}")
    
    def test_invariant_gamma_ge_one(self) -> Tuple[bool, str]:
        """Gamma must be >= 1 for real v/c < 1."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        if self.gamma >= 1:
            return (True, f"gamma = {self.gamma} >= 1")
        else:
            return (False, f"gamma = {self.gamma} < 1, unphysical")
    
    def test_invariant_lorentz_transformation_relation(self) -> Tuple[bool, str]:
        """
        Check that in the electron rest frame, the positive charge density
        transforms as rho_plus' = gamma * rho_plus, and the proper electron density
        is rho0_minus = rho_minus / gamma.
        Then neutrality in that frame gives gamma*rho_plus + rho_minus/gamma = 0.
        """
        if any(x is None for x in [self.rho_plus, self.rho_minus, self.gamma]):
            return (False, "SKIP: rho_plus, rho_minus, or gamma missing")
        if self.gamma == 0:
            return (False, "gamma = 0, cannot divide")
        lhs = self.gamma * self.rho_plus + self.rho_minus / self.gamma
        if math.isclose(lhs, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Neutrality in electron rest frame holds: gamma*rho_plus + rho_minus/gamma ≈ 0")
        else:
            return (False, f"Neutrality violation: gamma*rho_plus + rho_minus/gamma = {lhs} (should be 0)")
    
    # ------------------------------------------------------------------
    # Modeling tests (check the derived formula, use isclose)
    # ------------------------------------------------------------------
    def test_modeling_rho_minus_formula(self) -> Tuple[bool, str]:
        """Verify rho_minus = -gamma² * rho_plus"""
        if any(x is None for x in [self.rho_minus, self.gamma, self.rho_plus]):
            return (False, "SKIP: rho_minus, gamma, or rho_plus missing")
        expected = -self.gamma**2 * self.rho_plus
        if math.isclose(self.rho_minus, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"rho_minus = -gamma² * rho_plus holds")
        else:
            return (False, f"rho_minus = {self.rho_minus}, expected {expected}")
    
    def test_modeling_explanation_step_lorentz(self) -> Tuple[bool, str]:
        """
        Check if explanation_steps mention Lorentz transformation or gamma.
        This is a weak modeling quality check: the explanation should reference
        the transformation of charge densities.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        if not isinstance(self.explanation_steps, str):
            return (False, "SKIP: explanation_steps not a string")
        keywords = ['Lorentz', 'gamma', 'transformation', 'neutral']
        found = [kw for kw in keywords if kw.lower() in self.explanation_steps.lower()]
        if len(found) >= 2:
            return (True, f"Found relevant keywords in explanation: {found}")
        else:
            return (False, f"Explanation lacks Lorentz/transformation content; found: {found}")
    
    def test_modeling_explanation_density_uniformity(self) -> Tuple[bool, str]:
        """
        Check that the explanation acknowledges uniform densities.
        The problem statement explicitly states 'none of these depends on the distance from the axis'.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        if not isinstance(self.explanation_steps, str):
            return (False, "SKIP: explanation_steps not a string")
        if 'uniform' in self.explanation_steps.lower() or 'constant' in self.explanation_steps.lower():
            return (True, "Explanation mentions uniform/constant densities")
        else:
            return (False, "Explanation does not mention uniformity of densities")
    
    # ------------------------------------------------------------------
    # Monotonicity test (perturb v, check rho_minus trend)
    # ------------------------------------------------------------------
    def test_monotonicity_rho_minus_increases_with_v(self) -> Tuple[bool, str]:
        """
        Check that increasing v (making gamma larger) makes rho_minus more negative
        in magnitude (i.e., decreases algebraically).
        We perturb v by +1%, recompute expected rho_minus using rho_plus and gamma,
        and compare signs of change.
        """
        if any(x is None for x in [self.rho_plus, self.v, self.c, self.gamma, self.rho_minus]):
            return (False, "SKIP: missing variables for monotonicity test")
        if self.c == 0 or self.v is None:
            return (False, "SKIP: c=0 or v None")
        beta = self.v / self.c
        if abs(beta) >= 1:
            return (False, "v/c >= 1, cannot perturb")
        
        perturb_factor = 1.01  # +1%
        v_new = self.v * perturb_factor
        if v_new / self.c >= 1:
            return (False, "Perturbed v exceeds c, cannot test monotonicity")
        gamma_new = 1 / math.sqrt(1 - (v_new / self.c)**2)
        expected_new = -gamma_new**2 * self.rho_plus
        
        # Since rho_plus > 0 (given), rho_minus is negative; increasing gamma makes it more negative.
        # Check that delta_rho_minus has the correct sign: should algebraically decrease (more negative)
        delta_expected = expected_new - self.rho_minus
        # For positive rho_plus, expected delta < 0
        if self.rho_plus > 0:
            if delta_expected < 0:
                return (True, f"Monotonicity holds: v increased by 1%, rho_minus decreases by {abs(delta_expected):.2e}")
            else:
                return (False, f"Monotonicity fails: v increased, but rho_minus increased algebraically (delta={delta_expected:.2e})")
        else:
            # If rho_plus negative, sign flips; just check absolute magnitude increases
            if abs(expected_new) > abs(self.rho_minus):
                return (True, f"Monotonicity holds: |rho_minus| increases with v")
            else:
                return (False, f"|rho_minus| does not increase with v")
    
    # ------------------------------------------------------------------
    # Answer sanity tests (weak, single method, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity: rho_minus should be finite and have opposite sign to rho_plus
        (since rho_plus > 0 in typical problem, the derived formula gives rho_minus negative).
        This is a simple sign check, not a numerical equality.
        """
        if any(x is None for x in [self.rho_plus, self.rho_minus]):
            return (False, "SKIP: rho_plus or rho_minus missing")
        if not math.isfinite(self.rho_minus):
            return (False, "rho_minus is not finite")
        # Opposite signs check
        if self.rho_plus * self.rho_minus < 0:
            return (True, "rho_plus and rho_minus have opposite signs (consistent with -γ² rho_plus)")
        else:
            return (False, "rho_plus and rho_minus have same sign (should be opposite)")
    
    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods, collect results.
        Returns (all_pass, log_messages).
        """
        logs: List[str] = []
        all_pass = True
        
        # Get all bound methods starting with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED: {type(e).__name__}: {e}"
            
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        
        return (all_pass, logs)