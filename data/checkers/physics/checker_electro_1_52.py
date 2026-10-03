import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the dielectric force direction problem.
    Audits whether the solution correctly determines that the dielectric
    is pulled into the capacitor.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely using None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        
        # Expected result keys
        self.force_direction = self.results.get('force_direction', None)
        
        # Givens (used for numerical checks if available)
        self.epsilon0 = self.givens.get('epsilon0', None)
        self.w = self.givens.get('w', None)
        self.d = self.givens.get('d', None)
        self.L = self.givens.get('L', None)
        self.epsilon_r = self.givens.get('epsilon_r', None)
        self.Q = self.givens.get('Q', None)
        self.x = self.givens.get('x', None)
        
        # Additional results that might be present
        self.C = self.results.get('C', None)
        self.U = self.results.get('U', None)
        self.F = self.results.get('F', None)

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_force_direction_exists(self):
        if self.force_direction is None:
            return (False, "SKIP: 'force_direction' not provided")
        return (True, "'force_direction' key present")

    def test_interface_force_direction_type(self):
        if self.force_direction is None:
            return (False, "SKIP: 'force_direction' not available")
        if not isinstance(self.force_direction, str):
            return (False, f"Expected string, got {type(self.force_direction).__name__}")
        return (True, "force_direction is a string")

    # -----------------------------------------------------------------
    # Theorem-based invariant tests (use math.isclose if numerical)
    # -----------------------------------------------------------------
    def test_invariant_force_sign_from_energy(self):
        # Derivative of capacitance with respect to insertion length
        if self.epsilon_r is None or self.w is None or self.d is None or self.epsilon0 is None:
            return (False, "SKIP: missing parameters to compute dC/dx")
        # dC/dx = eps0 * w * (eps_r - 1) / d
        dC_dx = self.epsilon0 * self.w * (self.epsilon_r - 1) / self.d
        if dC_dx <= 0:
            return (False, f"dC/dx = {dC_dx:.6e} <= 0, violates physics (must be >0 for eps_r>1)")
        return (True, f"dC/dx = {dC_dx:.6e} > 0, energy decreases with x")
    
    def test_invariant_force_positive(self):
        # From energy gradient: F = (Q^2/(2*C^2)) * dC/dx > 0
        # If we have Q and C, we can check sign
        if self.Q is None or self.C is None or self.epsilon_r is None or self.epsilon0 is None or self.w is None or self.d is None:
            return (False, "SKIP: missing data for full force sign check")
        # C depends on x; we need C(x). If C is given as value, assume it is at some x.
        # But we can still check positivity of dC/dx and of Q^2/C^2.
        dC_dx = self.epsilon0 * self.w * (self.epsilon_r - 1) / self.d
        if dC_dx <= 0:
            return (False, "dC/dx not positive, force would be negative")
        if self.Q == 0:
            return (False, "Q=0 gives zero force, cannot test sign")
        # Force sign determined by dC/dx alone when Q constant
        return (True, "Force positive (pulling in) by energy principle")

    def test_invariant_dielectric_constant_above_one(self):
        if self.epsilon_r is None:
            return (False, "SKIP: epsilon_r not given")
        if self.epsilon_r <= 1:
            return (False, f"epsilon_r = {self.epsilon_r} <= 1, material not polarizable")
        return (True, "epsilon_r > 1, valid dielectric")

    # -----------------------------------------------------------------
    # Modeling / trace consistency checks (one local relation per test)
    # -----------------------------------------------------------------
    def test_modeling_capacitance_expression(self):
        # Check if results contain a capacitance expression that matches expected form
        if self.C is None:
            return (False, "SKIP: C not provided")
        # If C is a symbolic expression, we can compare structurally
        if hasattr(self.C, 'free_symbols'):
            # attempt to verify dependence on x and parameters
            x_sym = sp.Symbol('x')
            eps0_sym = sp.Symbol('epsilon0')
            w_sym = sp.Symbol('w')
            d_sym = sp.Symbol('d')
            L_sym = sp.Symbol('L')
            eps_r_sym = sp.Symbol('epsilon_r')
            expected = eps0_sym * w_sym * (L_sym + (eps_r_sym - 1)*x_sym) / d_sym
            if sp.simplify(self.C - expected) == 0:
                return (True, "Capacitance expression matches theoretical form")
            else:
                return (False, "Capacitance expression does not match expected linear form")
        else:
            # Fallback: if C is numeric, we cannot check form; skip
            return (False, "SKIP: C is numeric, cannot verify expression form")

    def test_trace_derivative_positive(self):
        # Check that the derivative of C w.r.t. x is computed as positive.
        # This test may use pseudocode steps: expect a variable like 'dC_dx'
        dC_dx = self.results.get('dC_dx', None)
        if dC_dx is None:
            return (False, "SKIP: dC_dx not found in results")
        if dC_dx <= 0:
            return (False, f"dC_dx = {dC_dx} is not positive")
        return (True, "dC_dx > 0 as required")

    # -----------------------------------------------------------------
    # Monotonicity tests (perturbation-based)
    # -----------------------------------------------------------------
    def test_monotonicity_dielectric_constant_increases_force(self):
        # Increase epsilon_r by 1% and see if force magnitude increases.
        # We need a numerical scenario. If parameters are missing, skip.
        if any(p is None for p in [self.epsilon0, self.w, self.d, self.L, self.epsilon_r, self.Q, self.x]):
            return (False, "SKIP: insufficient parameters for monotonicity test")
        # Compute baseline F at current epsilon_r
        C_base = self.epsilon0 * self.w * (self.L + (self.epsilon_r - 1)*self.x) / self.d
        F_base = (self.Q**2 / (2 * C_base**2)) * (self.epsilon0 * self.w * (self.epsilon_r - 1) / self.d)
        # Perturb epsilon_r upward by 1%
        eps_r_new = self.epsilon_r * 1.01
        C_new = self.epsilon0 * self.w * (self.L + (eps_r_new - 1)*self.x) / self.d
        F_new = (self.Q**2 / (2 * C_new**2)) * (self.epsilon0 * self.w * (eps_r_new - 1) / self.d)
        if F_new <= F_base:
            return (False, f"Force did not increase when epsilon_r increased: F_base={F_base:.3e}, F_new={F_new:.3e}")
        return (True, f"Force increases with epsilon_r: {F_base:.3e} -> {F_new:.3e}")

    # -----------------------------------------------------------------
    # Weak answer sanity (max one test, no isclose, no rederivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_force_direction_matches_expected(self):
        accepted = ['pulling in', 'pulled in', 'into', 'inward']
        if self.force_direction is None:
            return (False, "SKIP: No force direction to check")
        if self.force_direction.lower() in [a.lower() for a in accepted]:
            return (True, f"Direction '{self.force_direction}' is plausible")
        else:
            return (False, f"Direction '{self.force_direction}' not among expected: {accepted}")

    # -----------------------------------------------------------------
    # Main test runner
    # -----------------------------------------------------------------
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
                passed, msg = (False, f"{name} CRASHED: {e}\n{traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)