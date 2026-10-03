import math
import re
import traceback

class PhysicsChecker:
    """
    Audits solutions for the inverse-cube central force problem.
    Checks interface, invariant (theorem-based), modeling (consistency), 
    monotonicity, and weak answer-sanity.
    """
    def __init__(self, solution_instance):
        # Robustly extract givens and results
        self.givens = getattr(solution_instance, 'givens', 
                              getattr(solution_instance, 'input', {}))
        if not isinstance(self.givens, dict):
            self.givens = {}
        
        self.results = getattr(solution_instance, 'results', 
                               getattr(solution_instance, 'output', {}))
        if not isinstance(self.results, dict):
            self.results = {}

        # Provided base parameters (all optional)
        self.m = self.givens.get('m', None)
        self.k = self.givens.get('k', None)
        self.l = self.givens.get('l', None)          # angular momentum
        self.E = self.givens.get('E', None)          # total energy
        self.r0 = self.givens.get('r0', None)        # initial radius
        self.theta0 = self.givens.get('theta0', None)

        # Emitted results
        self.beta = self.results.get('beta', None)
        self.gamma = self.results.get('gamma', None)
        self.orbit_eq_bound = self.results.get('orbit_eq_bound', None)
        self.orbit_eq_unbound = self.results.get('orbit_eq_unbound', None)

    # ----------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four expected keys are present and have valid types."""
        missing = []
        for key in ['beta', 'gamma', 'orbit_eq_bound', 'orbit_eq_unbound']:
            if key not in self.results or self.results[key] is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {missing}")

        # Type checks
        if not isinstance(self.beta, (int, float)):
            return (False, f"beta should be numeric, got {type(self.beta)}")
        if not isinstance(self.gamma, (int, float)):
            return (False, f"gamma should be numeric, got {type(self.gamma)}")
        if not isinstance(self.orbit_eq_bound, str) or not self.orbit_eq_bound.strip():
            return (False, "orbit_eq_bound must be a non-empty string")
        if not isinstance(self.orbit_eq_unbound, str) or not self.orbit_eq_unbound.strip():
            return (False, "orbit_eq_unbound must be a non-empty string")
        return (True, "All required keys present with correct types")

    # ----------------------------------------------------------------
    # Invariant / theorem checks
    # ----------------------------------------------------------------
    def test_invariant_beta_gamma_mutual_exclusivity(self):
        """beta and gamma cannot both be positive reals (they belong to different cases)."""
        beta_ok = self.beta is not None and self.beta > 0
        gamma_ok = self.gamma is not None and self.gamma > 0
        if beta_ok and gamma_ok:
            return (False, f"Both beta={self.beta:.4f} and gamma={self.gamma:.4f} are positive – impossible, they correspond to exclusive cases")
        return (True, "Mutual exclusivity holds (at most one of beta/gamma is positive)")

    def test_invariant_beta_definition(self):
        """If m, k, l are given, verify beta^2 = 1 - (k*l^2?)*... Actually beta^2 = 1 - mk/(l^2)."""
        if self.m is None or self.k is None or self.l is None:
            return (False, "SKIP: m, k, l not all given")
        if self.beta is None:
            return (False, "SKIP: beta not provided")
        try:
            expected_beta_sq = 1 - (self.m * self.k) / (self.l ** 2)
            if expected_beta_sq < 0:
                return (False, f"Expected beta^2 negative ({expected_beta_sq:.4f}), but beta is real – inconsistent")
            expected_beta = math.sqrt(expected_beta_sq)
            if not math.isclose(self.beta, expected_beta, rel_tol=1e-2):
                return (False, f"beta={self.beta:.4f} does not match expected {expected_beta:.4f}")
        except ZeroDivisionError:
            return (False, "l is zero – angular momentum cannot be zero for non‑radial motion")
        return (True, "beta matches theoretical definition")

    def test_invariant_gamma_definition(self):
        """If m, k, l are given, verify gamma^2 = (mk/l^2) - 1."""
        if self.m is None or self.k is None or self.l is None:
            return (False, "SKIP: m, k, l not all given")
        if self.gamma is None:
            return (False, "SKIP: gamma not provided")
        try:
            expected_gamma_sq = (self.m * self.k) / (self.l ** 2) - 1
            if expected_gamma_sq < 0:
                return (False, f"Expected gamma^2 negative ({expected_gamma_sq:.4f}), but gamma is real – inconsistent")
            expected_gamma = math.sqrt(expected_gamma_sq)
            if not math.isclose(self.gamma, expected_gamma, rel_tol=1e-2):
                return (False, f"gamma={self.gamma:.4f} does not match expected {expected_gamma:.4f}")
        except ZeroDivisionError:
            return (False, "l is zero – angular momentum cannot be zero for non‑radial motion")
        return (True, "gamma matches theoretical definition")

    # ----------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------
    def test_modeling_orbit_bound_format(self):
        """orbit_eq_bound must contain 'cos' and 'beta' (or 'cos(beta(theta-theta0))')."""
        if not self.orbit_eq_bound:
            return (False, "SKIP: no bound orbit equation")
        eq = self.orbit_eq_bound.strip()
        if 'cos' not in eq:
            return (False, f"bound equation does not contain 'cos': {eq}")
        if 'beta' not in eq:
            return (False, f"bound equation does not contain 'beta': {eq}")
        # Also check for r and r0 and theta
        if not all(var in eq for var in ['r', 'theta', 'r0']):
            return (False, f"bound equation missing expected variables (r, theta, r0): {eq}")
        return (True, "Bound orbit equation has correct form")

    def test_modeling_orbit_unbound_format(self):
        """orbit_eq_unbound must contain 'cosh' and 'gamma'."""
        if not self.orbit_eq_unbound:
            return (False, "SKIP: no unbound orbit equation")
        eq = self.orbit_eq_unbound.strip()
        if 'cosh' not in eq:
            return (False, f"unbound equation does not contain 'cosh': {eq}")
        if 'gamma' not in eq:
            return (False, f"unbound equation does not contain 'gamma': {eq}")
        if not all(var in eq for var in ['r', 'theta', 'r0']):
            return (False, f"unbound equation missing expected variables (r, theta, r0): {eq}")
        return (True, "Unbound orbit equation has correct form")

    def test_modeling_angular_momentum_constant(self):
        """Check that angular momentum conservation is implicitly encoded: l = m r^2 dtheta/dt.
           Since we don't have r and dtheta, we can only check that if l given,
           the emitted beta/gamma respect the dimensionless parameter."""
        if self.l is None or self.m is None or self.k is None:
            return (False, "SKIP: m, k, l not all given")
        # Compute a = k/(m h^2) where h = l/m → h^2 = l^2/m^2 → a = k*m/l^2
        a = (self.k * self.m) / (self.l ** 2)
        if a < 1 and self.beta is not None:
            expected_beta = math.sqrt(1 - a)
            if not math.isclose(self.beta, expected_beta, rel_tol=1e-2):
                return (False, f"For a<1, beta should be sqrt(1-{a:.4f})={expected_beta:.4f}, got {self.beta:.4f}")
        elif a > 1 and self.gamma is not None:
            expected_gamma = math.sqrt(a - 1)
            if not math.isclose(self.gamma, expected_gamma, rel_tol=1e-2):
                return (False, f"For a>1, gamma should be sqrt({a:.4f}-1)={expected_gamma:.4f}, got {self.gamma:.4f}")
        return (True, "Angular momentum consistent with beta/gamma definitions")

    # ----------------------------------------------------------------
    # Monotonicity checks
    # ----------------------------------------------------------------
    def test_monotonicity_beta_decreases_with_k(self):
        """For the bound case, increasing k should decrease beta (if l fixed)."""
        if self.m is None or self.k is None or self.l is None:
            return (False, "SKIP: m, k, l not all given")
        if self.beta is None or self.beta <= 0:
            return (False, "SKIP: bound case not active (beta not positive)")

        k_base = self.k
        try:
            # Perturb k up by 10%
            k_pert = k_base * 1.1
            new_a = (self.m * k_pert) / (self.l ** 2)
            if new_a >= 1:
                return (False, "Perturbation pushes into unbound regime – cannot test monotonicity")
            new_beta = math.sqrt(1 - new_a)
            if new_beta >= self.beta:
                return (False, f"Increasing k from {k_base:.4f} to {k_pert:.4f} did not decrease beta "
                              f"(old={self.beta:.4f}, new={new_beta:.4f})")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")
        return (True, "beta decreases when k increases (bound case)")

    def test_monotonicity_gamma_increases_with_k(self):
        """For the unbound case, increasing k should increase gamma."""
        if self.m is None or self.k is None or self.l is None:
            return (False, "SKIP: m, k, l not all given")
        if self.gamma is None or self.gamma <= 0:
            return (False, "SKIP: unbound case not active (gamma not positive)")

        k_base = self.k
        try:
            k_pert = k_base * 1.1
            new_a = (self.m * k_pert) / (self.l ** 2)
            if new_a <= 1:
                return (False, "Perturbation pushes into bound regime – cannot test monotonicity")
            new_gamma = math.sqrt(new_a - 1)
            if new_gamma <= self.gamma:
                return (False, f"Increasing k from {k_base:.4f} to {k_pert:.4f} did not increase gamma "
                              f"(old={self.gamma:.4f}, new={new_gamma:.4f})")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")
        return (True, "gamma increases when k increases (unbound case)")

    # ----------------------------------------------------------------
    # Weak answer sanity (one method only, no isclose)
    # ----------------------------------------------------------------
    def test_answer_sanity_orbit_equations_non_trivial(self):
        """Check that orbit equations are non‑empty and contain the standard variables."""
        # Both should be present (even if only one is physically relevant)
        msgs = []
        if self.orbit_eq_bound:
            if not isinstance(self.orbit_eq_bound, str):
                msgs.append("orbit_eq_bound not a string")
            elif len(self.orbit_eq_bound.strip()) < 5:
                msgs.append("orbit_eq_bound too short")
        if self.orbit_eq_unbound:
            if not isinstance(self.orbit_eq_unbound, str):
                msgs.append("orbit_eq_unbound not a string")
            elif len(self.orbit_eq_unbound.strip()) < 5:
                msgs.append("orbit_eq_unbound too short")
        if msgs:
            return (False, " | ".join(msgs))
        return (True, "Orbit equations appear non‑trivial")

    # ----------------------------------------------------------------
    # Test dispatcher
    # ----------------------------------------------------------------
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)