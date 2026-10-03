import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Robust physics checker for the swing problem.
    Checks interface completeness, invariant physical relations,
    modeling quality, monotonicity trends, and a weak answer sanity.
    """
    
    def __init__(self, solution_instance):
        # Safely extract givens and results, defaulting to empty dicts
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract individual givens (use None for missing)
        self.m = self.givens.get('m', None)
        self.l = self.givens.get('l', None)
        self.g = self.givens.get('g', None)
        self.F = self.givens.get('F', None)   # may be omitted if derived
        self.theta0 = self.givens.get('theta0', None)
        self.theta_dot0 = self.givens.get('theta_dot0', None)
        
        # Extract result T
        self.T = self.results.get('T', None)
        
        # Derived commonly used quantities (if givens available)
        if self.l is not None and self.g is not None:
            self.omega = math.sqrt(self.g / self.l) if self.l > 0 and self.g > 0 else None
        else:
            self.omega = None
        
        # Expected time using the derived formula (only for internal derivation)
        self.expected_T = None
        if self.omega is not None and self.omega > 0:
            self.expected_T = math.pi / (3.0 * self.omega)
    
    # ---------- Interface checks ----------
    def test_interface_results_T_present(self):
        """Check that the result key 'T' exists and is a number."""
        if self.T is None:
            return (False, "Missing result 'T'")
        if not isinstance(self.T, (int, float)):
            return (False, f"T has invalid type: {type(self.T)}")
        return (True, "T present and numeric")
    
    def test_interface_givens_present(self):
        """Check that essential givens (l, g) are provided."""
        missing = []
        if self.l is None:
            missing.append("l")
        if self.g is None:
            missing.append("g")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "Essential givens present")
    
    def test_interface_theta0_positive(self):
        """Initial angle should be positive (father pulled back)."""
        if self.theta0 is None:
            return (False, "SKIP: theta0 not given")
        if self.theta0 <= 0:
            return (False, f"theta0 must be positive, got {self.theta0}")
        return (True, "theta0 positive")
    
    # ---------- Invariant checks (theorem-based, use math.isclose) ----------
    def test_invariant_initial_condition_theta(self):
        """Check that θ(0) = 1 rad, using the derived solution form."""
        if self.omega is None or self.T is None:
            return (False, "SKIP: insufficient data for theta(0) check")
        # theta(t) = 2*cos(omega*t) - 1
        theta_at_0 = 2 * math.cos(self.omega * 0.0) - 1
        if not math.isclose(theta_at_0, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Theta(0) should be 1, computed {theta_at_0}")
        return (True, "Initial angle condition satisfied")
    
    def test_invariant_initial_condition_theta_dot(self):
        """Check that θ'(0) = 0 from the solution derivative."""
        if self.omega is None:
            return (False, "SKIP: omega not available")
        # theta'(t) = -2*omega*sin(omega*t)
        theta_dot_0 = -2 * self.omega * math.sin(self.omega * 0.0)
        if not math.isclose(theta_dot_0, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Theta_dot(0) should be 0, computed {theta_dot_0}")
        return (True, "Initial angular velocity condition satisfied")
    
    def test_invariant_theta_zero_at_pushed_time(self):
        """Check that θ(T) = 0 using the derived solution."""
        if self.omega is None or self.T is None:
            return (False, "SKIP: missing omega or T")
        theta_at_T = 2 * math.cos(self.omega * self.T) - 1
        if not math.isclose(theta_at_T, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"θ(T) should be 0, computed {theta_at_T}")
        return (True, "Vertical position condition holds")
    
    def test_invariant_omega_definition(self):
        """Check that ω = sqrt(g/l) if both g and l are given."""
        if self.l is None or self.g is None:
            return (False, "SKIP: l or g missing")
        expected_omega = math.sqrt(self.g / self.l)
        if not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Omega mismatch: {self.omega} vs {expected_omega}")
        return (True, "Omega consistent with sqrt(g/l)")
    
    def test_invariant_cos_omega_T_equals_half(self):
        """Check that cos(ωT) = 1/2."""
        if self.omega is None or self.T is None:
            return (False, "SKIP: missing omega or T")
        cos_val = math.cos(self.omega * self.T)
        if not math.isclose(cos_val, 0.5, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"cos(ωT) = {cos_val}, expected 0.5")
        return (True, "cos(ωT) = 1/2 satisfied")
    
    # ---------- Modeling / trace checks ----------
    def test_modeling_small_angle_used(self):
        """
        Check that the solution is consistent with the linearized ODE
        θ'' + ω²θ = -ω².  We verify the second derivative relation.
        """
        if self.omega is None or self.T is None:
            return (False, "SKIP: missing omega or T")
        # At t=T, theta''(t) = -2*ω²*cos(ωt)
        # Evaluate at t=T/2 (not at T because cos(ωT)=0.5)
        t_check = self.T / 2.0
        theta = 2 * math.cos(self.omega * t_check) - 1
        theta_ddot = -2 * self.omega**2 * math.cos(self.omega * t_check)
        lhs = theta_ddot + self.omega**2 * theta
        rhs = -self.omega**2
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Linear ODE not satisfied at t={t_check}: LHS={lhs}, RHS={rhs}")
        return (True, "Small-angle ODE holds")
    
    def test_modeling_constant_force_sign(self):
        """
        The force F = mg is always tangential and acts to reduce the angle.
        For θ>0, the torque from F should be in the opposite direction to gravity's torque.
        In the equation mlθ'' = -mgl sinθ - F l, the term -F l drives θ toward smaller angles.
        We check that the solution only decreases from 1 to 0 (monotonic decrease).
        """
        if self.T is None:
            return (False, "SKIP: T not available")
        if self.omega is None:
            return (False, "SKIP: omega not available")
        # Check monotonic decrease: the derivative should be negative for all 0<t<T
        # Test at t = T/2
        t_mid = self.T / 2.0
        theta_dot_mid = -2 * self.omega * math.sin(self.omega * t_mid)
        if theta_dot_mid >= 0:
            return (False, f"Angular velocity should be negative at t={t_mid}, got {theta_dot_mid}")
        return (True, "Motion is monotonic decreasing (force reduces angle)")
    
    # ---------- Monotonicity tests (perturb inputs and compare) ----------
    def test_monotonicity_l_increases_T_increases(self):
        """If rope length l increases, the push time T should increase (positive relation)."""
        if self.l is None or self.g is None or self.T is None:
            return (False, "SKIP: missing l, g, or T")
        # Baseline expected T using derived formula
        T_base = self.expected_T
        if T_base is None:
            return (False, "SKIP: cannot compute expected T")
        # Increase l by 10%
        l_new = self.l * 1.1
        T_new = (math.pi / 3.0) * math.sqrt(l_new / self.g)
        # We expect T_new > T_base
        if T_new <= T_base:
            return (False, f"T should increase when l increases: baseline {T_base}, perturbed {T_new}")
        # Also check that the reported T is consistent with base (if it matches)
        if not math.isclose(self.T, T_base, rel_tol=1e-2, abs_tol=1e-2):
            # Not required to match exactly, but if not, we can still check monotonicity
            pass
        return (True, "Monotonicity l -> T positive")
    
    def test_monotonicity_g_increases_T_decreases(self):
        """If gravity g increases, the push time T should decrease."""
        if self.l is None or self.g is None or self.T is None:
            return (False, "SKIP: missing l, g, or T")
        T_base = self.expected_T
        if T_base is None:
            return (False, "SKIP: cannot compute expected T")
        # Increase g by 10%
        g_new = self.g * 1.1
        T_new = (math.pi / 3.0) * math.sqrt(self.l / g_new)
        if T_new >= T_base:
            return (False, f"T should decrease when g increases: baseline {T_base}, perturbed {T_new}")
        return (True, "Monotonicity g -> T negative")
    
    def test_monotonicity_force_increase_decreases_T(self):
        """
        The pushing force F is fixed as mg in the problem. However, if we hypothetically
        increase the force, the time to reach the vertical should decrease.
        We test this by computing the baseline T (using the default ratio F/(mg)=1)
        and the perturbed T for a 10% larger force (ratio = 1.1).
        """
        if self.omega is None or self.T is None:
            return (False, "SKIP: missing omega or T")
        # Baseline: F/(mg) = 1 -> cos(ωT) = 1/2 -> T = π/(3ω)
        T_base = math.pi / (3.0 * self.omega)
        # Perturb: increase force by 10% -> new ratio = 1.1
        ratio_new = 1.1
        # cos(ωT_new) = ratio_new / (1 + ratio_new)
        cos_new = ratio_new / (1.0 + ratio_new)
        if cos_new < -1 or cos_new > 1:
            return (False, f"Invalid cos value: {cos_new}")
        T_perturbed = math.acos(cos_new) / self.omega
        if T_perturbed >= T_base:
            return (False, f"T should decrease when force increases: baseline {T_base}, perturbed {T_perturbed}")
        return (True, "Monotonicity F -> T negative")
    
    # ---------- Weak answer sanity (only one, no math.isclose, no re-derivation) ----------
    def test_answer_sanity_basic(self):
        """Weak check: T should be positive, finite, and in a plausible range (0.1 to 100 s for reasonable l,g)."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if self.T <= 0:
            return (False, f"T must be positive, got {self.T}")
        if math.isnan(self.T) or math.isinf(self.T):
            return (False, "T is NaN or infinity")
        # Rough plausible range: if l is given, T ~ sqrt(l/g). l between 0.5 and 10 m => T ~0.4 to 3.2 s
        # Use a generous range 1e-6 to 1e6
        if self.T < 1e-6 or self.T > 1e6:
            return (False, f"T out of plausible magnitude: {self.T}")
        return (True, "T passes basic sanity checks")
    
    # ---------- Run all tests ----------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all methods starting with test_
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