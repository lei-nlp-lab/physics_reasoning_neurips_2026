import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics checker for inclined plane problem with two masses and friction.
    Checks interface completeness, invariant relations, modeling consistency,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens (use None as default, no numeric fallbacks)
        self.m1 = getattr(solution_instance, 'm1', None)
        self.m2 = getattr(solution_instance, 'm2', None)
        self.M = getattr(solution_instance, 'M', None)
        self.g = getattr(solution_instance, 'g', None)
        self.theta = getattr(solution_instance, 'theta', None)

        # Extract results dict
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.a = results.get('a', None)
        self.T = results.get('T', None)
        self.mu_min = results.get('mu_min', None)

        # Compute intermediate quantities for checks (only if all needed givens exist)
        if all(v is not None for v in [self.m1, self.m2, self.M, self.g, self.theta]):
            self.theta_val = float(self.theta)
            self.sin_theta = math.sin(self.theta_val)
            self.cos_theta = math.cos(self.theta_val)

            # Derived formulas for invariant checks (not from results)
            self.expected_a_num = (self.m1 - self.m2 * self.sin_theta) * self.g / (self.m1 + self.m2)
            self.expected_T_num = self.m1 * self.m2 * (1 + self.sin_theta) * self.g / (self.m1 + self.m2)
            self.expected_mu_min_num = (self.m2 * self.cos_theta * (self.m1 - self.m2 * self.sin_theta)) / \
                                       (self.M * (self.m1 + self.m2) + 
                                        self.m1 * self.m2 * (1 + self.sin_theta)**2 + 
                                        (self.m1 + self.m2) * self.m2 * self.cos_theta**2)
        else:
            self.sin_theta = None
            self.cos_theta = None
            self.expected_a_num = None
            self.expected_T_num = None
            self.expected_mu_min_num = None

    # -------------------------------------------------------------------------
    # Interface checks
    # -------------------------------------------------------------------------
    def test_interface_results_keys(self):
        """Check that required result keys exist."""
        required = ['a', 'T', 'mu_min']
        if not hasattr(self, 'results') or not isinstance(self.results, dict):
            return (False, "results dict missing or not a dict")
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_givens_not_none(self):
        """Check that required given parameters are not None."""
        missing = []
        for attr in ['m1', 'm2', 'M', 'g', 'theta']:
            if getattr(self, attr, None) is None:
                missing.append(attr)
        if missing:
            return (False, f"Missing or None givens: {missing}")
        return (True, "All givens present")

    # -------------------------------------------------------------------------
    # Invariant checks (use math.isclose with tolerance)
    # -------------------------------------------------------------------------
    def test_invariant_a_from_equations(self):
        """Check that acceleration satisfies both equations of motion."""
        if None in (self.a, self.m1, self.m2, self.g, self.sin_theta):
            return (False, "SKIP: missing data for a invariant")
        # From m1: m1*g - T = m1*a  => a = g - T/m1
        # From m2: T - m2*g*sinθ = m2*a => a = T/m2 - g*sinθ
        # Both should give same a when T is consistent
        # But here we just check that a is consistent with m1 eq and m2 eq using the provided T
        if self.T is None:
            return (False, "SKIP: T not available")
        a_from_m1 = (self.m1 * self.g - self.T) / self.m1
        a_from_m2 = (self.T - self.m2 * self.g * self.sin_theta) / self.m2
        if not math.isclose(a_from_m1, a_from_m2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"a from m1 eq ({a_from_m1}) and m2 eq ({a_from_m2}) differ")
        if not math.isclose(self.a, a_from_m1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Provided a ({self.a}) not consistent with m1 eq ({a_from_m1})")
        return (True, "a invariant passes")

    def test_invariant_tension_consistent_with_a(self):
        """Check that tension derived from a matches provided T."""
        if None in (self.a, self.T, self.m1, self.g):
            return (False, "SKIP: missing data for T invariant")
        T_from_a = self.m1 * self.g - self.m1 * self.a
        if not math.isclose(self.T, T_from_a, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Provided T ({self.T}) differs from T = m1*g - m1*a ({T_from_a})")
        return (True, "T invariant passes")

    def test_invariant_mu_min_from_forces(self):
        """Check that mu_min equals H/N_floor using computed tension and normal."""
        if any(v is None for v in [self.mu_min, self.T, self.m2, self.g, self.sin_theta, self.cos_theta, self.M, self.theta]):
            return (False, "SKIP: missing data for mu_min invariant")
        # Compute H and N_floor as per formulas
        N = self.m2 * self.g * self.cos_theta
        H = N * self.sin_theta - self.T * self.cos_theta
        N_floor = self.M * self.g + self.T * (1 + self.sin_theta) + N * self.cos_theta
        if N_floor == 0:
            return (False, "N_floor is zero")
        mu_min_calc = abs(H) / N_floor
        if not math.isclose(self.mu_min, mu_min_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Provided mu_min ({self.mu_min}) differs from |H|/N_floor ({mu_min_calc})")
        return (True, "mu_min invariant passes")

    # -------------------------------------------------------------------------
    # Modeling / trace checks
    # -------------------------------------------------------------------------
    def test_modeling_a_sign(self):
        """Check that a is positive when m1 > m2*sinθ, negative otherwise."""
        if None in (self.a, self.m1, self.m2, self.sin_theta):
            return (False, "SKIP: missing data for a sign check")
        expected_positive = (self.m1 > self.m2 * self.sin_theta)
        actual_positive = (self.a > 0)
        if expected_positive != actual_positive:
            return (False, f"a sign mismatch: expected positive={expected_positive}, a={self.a}")
        return (True, "a sign consistent with condition")

    def test_modeling_tension_bounds(self):
        """Check that tension lies between 0 and m1*g (if a>0) or between 0 and m2*g*sinθ (if a<0)."""
        if None in (self.T, self.m1, self.m2, self.g, self.sin_theta, self.a):
            return (False, "SKIP: missing data for T bounds")
        if self.a >= 0:
            max_T = self.m1 * self.g
            min_T = 0
        else:
            max_T = self.m2 * self.g * self.sin_theta
            min_T = 0
        if self.T < min_T or self.T > max_T:
            return (False, f"T ({self.T}) outside plausible bounds [{min_T}, {max_T}]")
        return (True, "T within plausible bounds")

    def test_modeling_mu_min_nonnegative(self):
        """Check that mu_min is nonnegative."""
        if self.mu_min is None:
            return (False, "SKIP: mu_min not available")
        if self.mu_min < 0:
            return (False, f"mu_min negative ({self.mu_min})")
        return (True, "mu_min nonnegative")

    # -------------------------------------------------------------------------
    # Monotonicity checks (perturb one input, recompute expected behavior)
    # -------------------------------------------------------------------------
    def test_monotonicity_a_with_m1(self):
        """Check that increasing m1 increases a."""
        if any(v is None for v in [self.a, self.m1, self.m2, self.g, self.sin_theta]):
            return (False, "SKIP: missing data for monotonicity a(m1)")
        # Perturb m1 by +10%
        m1_pert = self.m1 * 1.1
        a_pert = (m1_pert - self.m2 * self.sin_theta) * self.g / (m1_pert + self.m2)
        if a_pert <= self.a:
            return (False, f"a did not increase when m1 increased: baseline {self.a}, perturbed {a_pert}")
        # Also check perturbation in opposite direction
        m1_pert_down = self.m1 * 0.9
        a_pert_down = (m1_pert_down - self.m2 * self.sin_theta) * self.g / (m1_pert_down + self.m2)
        if a_pert_down >= self.a:
            return (False, f"a did not decrease when m1 decreased: baseline {self.a}, perturbed {a_pert_down}")
        return (True, "a monotonic with m1 (as expected)")

    def test_monotonicity_mu_min_with_M(self):
        """Check that increasing M decreases mu_min."""
        if any(v is None for v in [self.mu_min, self.M, self.m1, self.m2, self.g, self.sin_theta, self.cos_theta, self.theta]):
            return (False, "SKIP: missing data for monotonicity mu_min(M)")
        # Compute mu_min with M perturbed +10%
        M_pert = self.M * 1.1
        mu_pert = (self.m2 * self.cos_theta * (self.m1 - self.m2 * self.sin_theta)) / \
                  (M_pert * (self.m1 + self.m2) + 
                   self.m1 * self.m2 * (1 + self.sin_theta)**2 + 
                   (self.m1 + self.m2) * self.m2 * self.cos_theta**2)
        if mu_pert >= self.mu_min:
            return (False, f"mu_min did not decrease when M increased: baseline {self.mu_min}, perturbed {mu_pert}")
        # Check opposite perturbation
        M_pert_down = self.M * 0.9
        mu_pert_down = (self.m2 * self.cos_theta * (self.m1 - self.m2 * self.sin_theta)) / \
                       (M_pert_down * (self.m1 + self.m2) + 
                        self.m1 * self.m2 * (1 + self.sin_theta)**2 + 
                        (self.m1 + self.m2) * self.m2 * self.cos_theta**2)
        if mu_pert_down <= self.mu_min:
            return (False, f"mu_min did not increase when M decreased: baseline {self.mu_min}, perturbed {mu_pert_down}")
        return (True, "mu_min monotonic with M (as expected)")

    # -------------------------------------------------------------------------
    # Weak answer sanity (only one, no re-derivation)
    # -------------------------------------------------------------------------
    def test_answer_sanity_bounds(self):
        """Check that a, T, mu_min are finite and within broad reasonable range."""
        if self.a is None or self.T is None or self.mu_min is None:
            return (False, "SKIP: one or more results missing")
        # a should be between -g and g (since max acceleration is g)
        if not ( -self.g <= self.a <= self.g ):
            return (False, f"a ({self.a}) outside [-g, g]")
        # T should be between 0 and (m1+m2)*g
        max_T_approx = (self.m1 + self.m2) * self.g
        if not (0 <= self.T <= max_T_approx):
            return (False, f"T ({self.T}) outside plausible range [0, {max_T_approx}]")
        # mu_min should be between 0 and 1 (since it's a coefficient of friction)
        if not (0 <= self.mu_min <= 1):
            return (False, f"mu_min ({self.mu_min}) outside [0,1]")
        return (True, "All results are finite and within broad plausible ranges")

    # -------------------------------------------------------------------------
    # Main runner
    # -------------------------------------------------------------------------
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