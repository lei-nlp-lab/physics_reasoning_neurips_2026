import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for the heat shield problem (radiative heat transfer
    between infinite parallel black planes with an inserted shield).
    """
    def __init__(self, solution_instance):
        # Safely extract givens and results using None as default
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', []) or []

        # Givens
        self.T_u = self.givens.get('T_u', None)
        if self.T_u is None:
            self.T_u = getattr(solution_instance, 'T_u', None)
        self.T_l = self.givens.get('T_l', None)
        if self.T_l is None:
            self.T_l = getattr(solution_instance, 'T_l', None)
        self.sigma = self.givens.get('sigma', None)
        if self.sigma is None:
            self.sigma = getattr(solution_instance, 'sigma', None)
        self.r = self.givens.get('r', None)
        if self.r is None:
            self.r = getattr(solution_instance, 'r', None)

        # Results
        self.T_m = self.results.get('T_m', None)
        if self.T_m is None:
            self.T_m = getattr(solution_instance, 'T_m', None)
        self.J_U = self.results.get('J_U', None)
        if self.J_U is None:
            self.J_U = getattr(solution_instance, 'J_U', None)
        self.J_U_prime = self.results.get('J_U_prime', None)
        if self.J_U_prime is None:
            self.J_U_prime = getattr(solution_instance, 'J_U_prime', None)

    def _isclose(self, a, b):
        """Fuzzy comparison for invariant/modeling checks (1% tolerance)."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_results_exist(self):
        """Check that required result keys are present (not None)."""
        if self.T_m is None:
            return (False, "SKIP: T_m not found in results")
        if self.J_U is None:
            return (False, "SKIP: J_U not found in results")
        if self.J_U_prime is None:
            return (False, "SKIP: J_U_prime not found in results")
        return (True, "All required result keys present")

    def test_interface_givens_exist(self):
        """Check that essential givens are present (not None)."""
        missing = []
        if self.T_u is None:
            missing.append('T_u')
        if self.T_l is None:
            missing.append('T_l')
        if self.sigma is None:
            missing.append('sigma')
        # r is optional (can be 0 if not provided)
        if missing:
            return (False, f"SKIP: missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_Tm_fourth_power(self):
        """T_m^4 must equal (T_u^4 + T_l^4)/2."""
        if None in (self.T_u, self.T_l, self.T_m):
            return (False, "SKIP: T_u, T_l or T_m is None")
        lhs = self.T_m**4
        rhs = (self.T_u**4 + self.T_l**4) / 2.0
        if self._isclose(lhs, rhs):
            return (True, "T_m^4 = (T_u^4+T_l^4)/2 satisfied")
        else:
            return (False, f"T_m^4={lhs:.6e} vs (T_u^4+T_l^4)/2={rhs:.6e}")

    def test_invariant_J_U_from_Tm(self):
        """J_U must equal sigma*(T_u^4 - T_m^4)."""
        if None in (self.J_U, self.sigma, self.T_u, self.T_m):
            return (False, "SKIP: one or more required values missing")
        expected = self.sigma * (self.T_u**4 - self.T_m**4)
        if self._isclose(self.J_U, expected):
            return (True, "J_U = sigma*(T_u^4 - T_m^4) satisfied")
        else:
            return (False, f"J_U={self.J_U:.6e} vs sigma*(T_u^4-T_m^4)={expected:.6e}")

    def test_invariant_J_U_direct(self):
        """J_U must equal (sigma/2)*(T_u^4 - T_l^4)."""
        if None in (self.J_U, self.sigma, self.T_u, self.T_l):
            return (False, "SKIP: one or more required values missing")
        expected = 0.5 * self.sigma * (self.T_u**4 - self.T_l**4)
        if self._isclose(self.J_U, expected):
            return (True, "J_U = (sigma/2)*(T_u^4 - T_l^4) satisfied")
        else:
            return (False, f"J_U={self.J_U:.6e} vs (sigma/2)*(T_u^4-T_l^4)={expected:.6e}")

    def test_invariant_J_U_prime_scaling(self):
        """J_U_prime must equal (1 - r) * J_U."""
        if None in (self.J_U_prime, self.r, self.J_U):
            return (False, "SKIP: J_U_prime, r or J_U missing")
        expected = (1.0 - self.r) * self.J_U
        if self._isclose(self.J_U_prime, expected):
            return (True, "J_U_prime = (1-r)*J_U satisfied")
        else:
            return (False, f"J_U_prime={self.J_U_prime:.6e} vs (1-r)*J_U={expected:.6e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (quality of explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_steady_state_flux_balance(self):
        """For r=0: sigma*(T_u^4 - T_m^4) must equal sigma*(T_m^4 - T_l^4)."""
        if None in (self.sigma, self.T_u, self.T_l, self.T_m):
            return (False, "SKIP: sigma, T_u, T_l or T_m missing")
        lhs = self.sigma * (self.T_u**4 - self.T_m**4)
        rhs = self.sigma * (self.T_m**4 - self.T_l**4)
        if self._isclose(lhs, rhs):
            return (True, "Steady state flux balance holds (sigma*(T_u^4 - T_m^4) = sigma*(T_m^4 - T_l^4))")
        else:
            return (False, f"Leftward flux mismatch: {lhs:.6e} vs {rhs:.6e}")

    def test_modeling_reflecting_shield_flux(self):
        """J_U_prime must equal (1-r)*sigma*(T_u^4 - T_m^4)."""
        if None in (self.J_U_prime, self.sigma, self.T_u, self.T_m, self.r):
            return (False, "SKIP: required values missing")
        expected = (1.0 - self.r) * self.sigma * (self.T_u**4 - self.T_m**4)
        if self._isclose(self.J_U_prime, expected):
            return (True, "Reflecting shield flux relation holds")
        else:
            return (False, f"J_U_prime={self.J_U_prime:.6e} vs (1-r)*sigma*(T_u^4-T_m^4)={expected:.6e}")

    def test_modeling_Tm_range(self):
        """T_m should be between T_l and T_u when T_u > T_l."""
        if None in (self.T_u, self.T_l, self.T_m):
            return (False, "SKIP: T_u, T_l or T_m missing")
        if self.T_u <= self.T_l:
            return (True, "T_u <= T_l, range check not applicable")
        if self.T_l < self.T_m < self.T_u:
            return (True, "T_m lies between T_l and T_u")
        else:
            return (False, f"T_m={self.T_m:.4f} not in ({self.T_l:.4f}, {self.T_u:.4f})")

    def test_modeling_explanation_uses_flux_balance(self):
        """Check that explanation_steps mention flux balance."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        if "flux" in combined and "balance" in combined:
            return (True, "Explanation uses flux balance concept")
        else:
            return (False, "Explanation does not mention 'flux balance'")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_Tm_increases_with_Tu(self):
        """
        Perturb T_u upward by 1% and verify that the computed Tm 
        (using the relation Tm^4 = (Tu^4+Tl^4)/2) increases.
        """
        if None in (self.T_u, self.T_l, self.T_m):
            return (False, "SKIP: T_u, T_l or T_m missing")
        # Derive Tm from given temperatures (re-derive inside checker)
        Tm_derived = ((self.T_u**4 + self.T_l**4) / 2.0) ** 0.25
        # Perturb T_u by +1%
        Tu_perturbed = self.T_u * 1.01
        Tm_perturbed = ((Tu_perturbed**4 + self.T_l**4) / 2.0) ** 0.25
        if Tm_perturbed > Tm_derived:
            return (True, f"Tm increases when Tu increases by 1%: {Tm_derived:.6f} -> {Tm_perturbed:.6f}")
        else:
            return (False, f"Tm did not increase: baseline {Tm_derived:.6f}, perturbed {Tm_perturbed:.6f}")

    def test_monotonicity_Tm_increases_with_Tl(self):
        """
        Perturb T_l upward by 1% and verify that Tm increases.
        """
        if None in (self.T_u, self.T_l, self.T_m):
            return (False, "SKIP: T_u, T_l or T_m missing")
        Tm_derived = ((self.T_u**4 + self.T_l**4) / 2.0) ** 0.25
        Tl_perturbed = self.T_l * 1.01
        Tm_perturbed = ((self.T_u**4 + Tl_perturbed**4) / 2.0) ** 0.25
        if Tm_perturbed > Tm_derived:
            return (True, f"Tm increases when Tl increases by 1%: {Tm_derived:.6f} -> {Tm_perturbed:.6f}")
        else:
            return (False, f"Tm did not increase: baseline {Tm_derived:.6f}, perturbed {Tm_perturbed:.6f}")

    def test_monotonicity_J_U_prime_decreases_with_r(self):
        """
        Perturb r upward by 10% (or to a nonzero if r=0) and verify that 
        J_U_prime decreases (since J_U'>0 for Tu>Tl).
        """
        if None in (self.J_U_prime, self.r, self.J_U):
            return (False, "SKIP: J_U_prime, r or J_U missing")
        if self.J_U <= 0:
            return (False, "J_U must be positive for monotonicity check")
        # Baseline J_U_prime from results. Compute perturbed using formula J_U_prime = (1-r)*J_U
        # Perturb r by +0.1 (or at least 0.1 if r=0)
        if self.r == 0:
            r_perturbed = 0.1
        else:
            r_perturbed = self.r * 1.1
        J_U_prime_perturbed = (1.0 - r_perturbed) * self.J_U
        if J_U_prime_perturbed < self.J_U_prime:
            return (True, f"J_U_prime decreases when r increases: baseline {self.J_U_prime:.6e}, perturbed {J_U_prime_perturbed:.6e}")
        else:
            return (False, f"J_U_prime did not decrease: baseline {self.J_U_prime:.6e}, perturbed {J_U_prime_perturbed:.6e}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, non-dominant)
    # ------------------------------------------------------------------
    def test_answer_sanity_Tm_positive(self):
        """T_m must be positive."""
        if self.T_m is None:
            return (False, "SKIP: T_m not available")
        if self.T_m > 0:
            return (True, "T_m > 0")
        else:
            return (False, f"T_m = {self.T_m} is not positive")

    # ------------------------------------------------------------------
    # Runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# Example usage:
# from your_solution_module import HeatShieldSolution  # hypothetical
# sol = HeatShieldSolution(...)
# checker = PhysicsChecker(sol)
# success, log = checker.run_all_tests()