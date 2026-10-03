import math
import sympy as sp
import traceback
from typing import Tuple, List

class PhysicsChecker:
    """
    Audits solution instances for the rod-in-tube problem.
    Checks interface completeness, theorem-based invariants,
    modeling correctness, monotonicity, and weak answer sanity.
    """
    
    def __init__(self, solution_instance):
        # Extract results dictionary – support both attribute and method
        if hasattr(solution_instance, 'results') and callable(solution_instance.results):
            self.results = solution_instance.results()
        elif hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        else:
            self.results = {}
        
        # Extract givens – assume stored in a 'givens' attribute/dict
        if hasattr(solution_instance, 'givens') and callable(solution_instance.givens):
            givens = solution_instance.givens()
        elif hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
        else:
            givens = {}
        self.M = givens.get('M', None)
        self.l = givens.get('l', None)
        self.omega0 = givens.get('omega0', None)
        
        # Result keys (from pseudocode contract)
        self.I = self.results.get('I', None)
        self.radial_eq = self.results.get('radial_eq', None)
        self.L_const = self.results.get('L_const', None)
        self.theta_dot_inf = self.results.get('theta_dot_inf', None)
        self.r_dot_inf = self.results.get('r_dot_inf', None)
        
        # Derived quantity – only used when M and l are present
        self.I_calc = None
        if self.M is not None and self.l is not None:
            self.I_calc = self.M * self.l**2 / 12.0
        
    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all five required result keys exist."""
        required = ['I', 'radial_eq', 'L_const', 'theta_dot_inf', 'r_dot_inf']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys are present")
    
    def test_interface_given_quantities(self) -> Tuple[bool, str]:
        """Check that the givens M, l, omega0 are provided."""
        if self.M is None or self.l is None or self.omega0 is None:
            return (False, "Missing one or more givens: M, l, omega0")
        return (True, "All givens (M, l, omega0) are present")
    
    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_moment_of_inertia_formula(self) -> Tuple[bool, str]:
        """Check I = M l^2 / 12 with 1% tolerance."""
        if self.I is None or self.M is None or self.l is None:
            return (False, "SKIP: I, M or l missing")
        expected = self.M * self.l**2 / 12.0
        if not isinstance(self.I, (int, float)):
            return (False, "I is not numeric")
        if math.isclose(self.I, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I = {self.I} matches expected {expected:.4f}")
        return (False, f"I = {self.I} differs from {expected:.4f} by >1%")
    
    def test_invariant_energy_conservation_asymptotic(self) -> Tuple[bool, str]:
        """
        Check that for t → ∞ the kinetic energy equals the initial energy.
        Uses: E_initial = 0.5 * I * omega0^2, E_final ≈ 0.5 * M * r_dot_inf^2
        (the rotational part vanishes as r → ∞).
        """
        if self.I is None or self.omega0 is None or self.M is None or self.r_dot_inf is None:
            return (False, "SKIP: missing I, omega0, M or r_dot_inf")
        E_initial = 0.5 * self.I * self.omega0**2
        E_final   = 0.5 * self.M * self.r_dot_inf**2
        if math.isclose(E_initial, E_final, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Energy conserved: initial {E_initial:.4f} ≈ final {E_final:.4f}")
        return (False, f"Energy mismatch: initial {E_initial:.4f}, final {E_final:.4f}")
    
    def test_invariant_angular_momentum_formula(self) -> Tuple[bool, str]:
        """Check that L_const string contains the correct conserved form."""
        if self.L_const is None:
            return (False, "SKIP: L_const not provided")
        # Expected pattern: (M*r^2 + I)*theta_dot = I*omega0
        # We use substring search – not perfect but adequate for modeling audit.
        if 'theta_dot' in self.L_const and 'I*omega0' in self.L_const:
            return (True, "Angular momentum conservation form is recognised")
        return (False, "L_const string does not match expected pattern")
    
    # ------------------------------------------------------------------
    # Modelling/trace tests (check key equations from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_radial_equation(self) -> Tuple[bool, str]:
        """Check that the radial equation string matches the expected one."""
        if self.radial_eq is None:
            return (False, "SKIP: radial_eq not provided")
        expected = r'\ddot{r} = r\dot{\theta}^2'
        # Strip whitespace and compare ignoring LaTeX formatting variations
        eq_clean = self.radial_eq.replace(' ', '')
        expected_clean = expected.replace(' ', '')
        if eq_clean == expected_clean:
            return (True, "Radial equation matches expected form")
        return (False, f"Radial equation is: {self.radial_eq}, expected: {expected}")
    
    def test_modeling_lagrange_equation_theta(self) -> Tuple[bool, str]:
        """Check that L_const indicates d/dt( (M r^2+I) theta_dot ) = 0."""
        if self.L_const is None:
            return (False, "SKIP: L_const missing")
        # Look for the conserved quantity form
        if 'M*r^2 + I' in self.L_const:
            return (True, "Theta equation conservation is present")
        return (False, "Theta equation not correctly represented in L_const")
    
    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation of input, compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_theta_dot_decreases_with_r(self) -> Tuple[bool, str]:
        """
        Explicitly perturb r (distance from pivot) and check that
        theta_dot decreases. Baseline: r0 = 0.3*l, perturb to r1 = 1.1*r0.
        Theta_dot computed from conservation law: td = I*omega0/(M*r^2+I).
        """
        if self.M is None or self.l is None or self.omega0 is None:
            return (False, "SKIP: givens missing")
        I_val = self.M * self.l**2 / 12.0
        r0 = 0.3 * self.l
        td_baseline = I_val * self.omega0 / (self.M * r0**2 + I_val)
        # Perturb r by +10%
        r1 = 1.1 * r0
        td_perturbed = I_val * self.omega0 / (self.M * r1**2 + I_val)
        if td_perturbed < td_baseline:
            return (True, f"theta_dot decreases with r: baseline={td_baseline:.4f}, perturbed={td_perturbed:.4f}")
        return (False, f"Monotonicity violated: baseline={td_baseline:.4f}, perturbed={td_perturbed:.4f}")
    
    # ------------------------------------------------------------------
    # Weak answer sanity (one test, no isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_asymptotic_values(self) -> Tuple[bool, str]:
        """Weak check: theta_dot_inf >= 0, r_dot_inf > 0 and finite."""
        errors = []
        if self.theta_dot_inf is not None:
            if not isinstance(self.theta_dot_inf, (int, float)):
                errors.append("theta_dot_inf not numeric")
            else:
                if self.theta_dot_inf < 0:
                    errors.append("theta_dot_inf negative")
                if self.theta_dot_inf > self.omega0:   # should be less than initial
                    errors.append("theta_dot_inf larger than omega0?")
        else:
            errors.append("theta_dot_inf missing")
        
        if self.r_dot_inf is not None:
            if not isinstance(self.r_dot_inf, (int, float)):
                errors.append("r_dot_inf not numeric")
            else:
                if self.r_dot_inf <= 0:
                    errors.append("r_dot_inf non‑positive")
        else:
            errors.append("r_dot_inf missing")
        
        if errors:
            return (False, "; ".join(errors))
        return (True, "Asymptotic values seem physically plausible")
    
    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, run them inside try/except,
        collect messages and overall pass/fail.
        """
        logs = []
        all_pass = True
        
        # Get all methods starting with 'test_'
        for attr_name in dir(self):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{attr_name} CRASHED: {traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        
        return (all_pass, logs)