import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract inputs and results safely."""
        # Constants from problem context
        self.L = getattr(solution_instance, 'L', 0.8)
        self.R = getattr(solution_instance, 'R', 0.4)
        self.S = getattr(solution_instance, 'S', 1.25)
        self.mu = getattr(solution_instance, 'mu', 0.25)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.m_ratio = getattr(solution_instance, 'm_ratio', 3.0)  # m_a = 3 * m_b

        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        self.v0 = self.results.get('v_0', None)
        self.v0min = self.results.get('v_0min', None)
        self.v0max = self.results.get('v_0max', None)
        self.vb1 = self.results.get('v_b1', None)
        self.va1 = self.results.get('v_a1', None)
        self.va1max = self.results.get('v_a1max', None)
        self.vb2 = self.results.get('v_b2', None)
        self.v0_prime = self.results.get("v_0'", None)

        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # --------------------------------------------------------------
    # Individual test methods
    # --------------------------------------------------------------
    def test_elastic_collision(self):
        """Check that v_a1 and v_b1 follow from elastic collision formulas for consistent v_0."""
        # This test should check that if we have a specific v_0, then v_a1 = v_0/2 and v_b1 = 1.5*v_0
        # Since we have multiple v_0 values (v_0min, v_0_prime, v_0), we need to be careful
        # Instead, check the relationship holds for the stored v_0 (lower bound)
        if self.v0 is not None and self.va1 is not None and self.vb1 is not None:
            # Check that va1 = v0/2 and vb1 = 1.5*v0
            if not (self.isclose(self.va1, 0.5 * self.v0) and self.isclose(self.vb1, 1.5 * self.v0)):
                self.log(f"Elastic collision failed: v0={self.v0}, va1={self.va1} (expected {0.5*self.v0}), vb1={self.vb1} (expected {1.5*self.v0})")
                return False
            self.log("Elastic collision passed for stored v_0.")
            return True
        else:
            self.log("Missing v0, va1, or vb1, skipping elastic collision test.")
            return True

    def test_ball_b_energy(self):
        """Verify energy conservation for ball b from bottom to top at minimum condition."""
        if self.vb1 is None or self.vb2 is None:
            self.log("Missing v_b1 or v_b2, skipping ball b energy test.")
            return True

        # Energy conservation: 0.5 * v_b1^2 = 0.5 * v_b2^2 + g*(2R + L)
        # This should hold for the MINIMUM v_b1 required. If v_b1 is larger, left side > right side.
        # So we compute the minimum v_b1 and check that actual v_b1 >= minimum.
        v_b2_min = math.sqrt(self.g * self.R)
        if not self.isclose(self.vb2, v_b2_min):
            self.log(f"v_b2 should be sqrt(gR) = {v_b2_min}, got {self.vb2}")
            return False
        
        # Compute minimum v_b1 from energy conservation
        v_b1_min_required = math.sqrt(v_b2_min**2 + 2 * self.g * (2 * self.R + self.L))
        
        # Check that v_b1 >= v_b1_min_required (with tolerance)
        if self.vb1 < v_b1_min_required - 1e-2:
            self.log(f"Ball b energy condition failed: v_b1 = {self.vb1} must be >= {v_b1_min_required}")
            return False
        
        # Also verify that energy conservation holds for the minimum case
        energy_bottom_min = 0.5 * v_b1_min_required**2
        energy_top = 0.5 * v_b2_min**2 + self.g * (2 * self.R + self.L)
        if not self.isclose(energy_bottom_min, energy_top):
            self.log(f"Energy conservation for minimum v_b1 failed: {energy_bottom_min} != {energy_top}")
            return False
            
        self.log("Ball b energy condition passed.")
        return True

    def test_ball_b_circular_motion(self):
        """Check that v_b2 satisfies minimum centripetal force at top."""
        if self.vb2 is None:
            self.log("Missing v_b2, skipping circular motion test.")
            return True
        # At top: m*g <= m*v_b2^2 / R  -> v_b2 >= sqrt(g*R)
        v_b2_min = math.sqrt(self.g * self.R)
        if self.vb2 < v_b2_min - 1e-2:  # Allow small tolerance for threshold
            self.log(f"Ball b circular motion condition failed: v_b2 = {self.vb2} < sqrt(gR) = {v_b2_min}")
            return False
        self.log(f"Ball b circular motion passed: v_b2 = {self.vb2} >= sqrt(gR) = {v_b2_min}")
        return True

    def test_ball_a_energy(self):
        """Check v_a1max = sqrt(2g(L+R))."""
        if self.va1max is None:
            self.log("Missing v_a1max, skipping ball a energy test.")
            return True
        expected = math.sqrt(2 * self.g * (self.L + self.R))
        if not self.isclose(self.va1max, expected):
            self.log(f"Ball a max velocity failed: expected {expected}, got {self.va1max}")
            return False
        self.log(f"Ball a max velocity passed: {self.va1max} ≈ {expected}")
        return True

    def test_friction_condition(self):
        """Verify v_a1 >= sqrt(2 μ g S) for ball a to slide out."""
        if self.va1 is None:
            self.log("Missing v_a1, skipping friction test.")
            return True
        threshold = math.sqrt(2 * self.mu * self.g * self.S)
        # Allow equality since it's minimum condition
        if self.va1 < threshold - 1e-2:
            self.log(f"Friction condition failed: v_a1 = {self.va1} must be >= {threshold}")
            return False
        self.log(f"Friction condition passed: v_a1 = {self.va1} >= {threshold}")
        return True

    def test_range_consistency(self):
        """Check that v0min and v0max are computed correctly."""
        # Compute expected v0min from energy condition for ball b
        # v_b1_min = sqrt(gR + 2g(2R+L)) = sqrt(gR + 4gR + 2gL) = sqrt(5gR + 2gL)
        # v_0_min_from_b = (2/3)*v_b1_min = (2/3)*sqrt(5gR + 2gL)
        # Square both sides: v_0_min_from_b^2 = (4/9)*(5gR + 2gL) = (20gR + 8gL)/9
        expected_v0min_b = math.sqrt((20 * self.g * self.R + 8 * self.g * self.L) / 9)
        
        # Compute expected v0min from friction
        expected_v0min_friction = 2 * math.sqrt(2 * self.mu * self.g * self.S)
        
        # The actual v0min should be the MAXIMUM of these two conditions
        expected_v0min_correct = max(expected_v0min_b, expected_v0min_friction)
        
        # Compute expected v0max: v_a1max = sqrt(2g(L+R)), v_0max = 2*v_a1max
        expected_v0max = 2 * math.sqrt(2 * self.g * (self.L + self.R))

        if self.v0min is not None and not self.isclose(self.v0min, expected_v0min_correct):
            self.log(f"v0min mismatch: expected {expected_v0min_correct}, got {self.v0min}")
            return False
        if self.v0max is not None and not self.isclose(self.v0max, expected_v0max):
            self.log(f"v0max mismatch: expected {expected_v0max}, got {self.v0max}")
            return False
        self.log(f"Range consistency passed: v0min ≈ {expected_v0min_correct}, v0max ≈ {expected_v0max}")
        return True

    def test_invariants(self):
        """Overall energy and momentum invariants."""
        # Check momentum conservation for the stored v_0 (lower bound)
        if self.v0 is not None and self.va1 is not None and self.vb1 is not None:
            # m_a * v0 = m_a * va1 + m_b * vb1
            lhs = self.m_ratio * self.v0
            rhs = self.m_ratio * self.va1 + 1.0 * self.vb1  # m_b = 1 (unit mass)
            if not self.isclose(lhs, rhs):
                self.log(f"Momentum conservation failed: {lhs} != {rhs}")
                return False
            self.log("Momentum conservation passed.")
            
            # Elastic collision: KE conserved
            ke_before = 0.5 * self.m_ratio * self.v0**2
            ke_after = 0.5 * self.m_ratio * self.va1**2 + 0.5 * 1.0 * self.vb1**2
            if not self.isclose(ke_before, ke_after):
                self.log(f"Kinetic energy conservation in collision failed: {ke_before} != {ke_after}")
                return False
            self.log("Kinetic energy conservation in collision passed.")
        return True

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        # If v0 is given, check that va1 and vb1 follow
        if self.v0 is not None and self.va1 is not None and self.vb1 is not None:
            if not self.isclose(self.va1, 0.5 * self.v0):
                self.log(f"Trace consistency for v_a1 failed: expected {0.5*self.v0}, got {self.va1}")
                return False
            if not self.isclose(self.vb1, 1.5 * self.v0):
                self.log(f"Trace consistency for v_b1 failed: expected {1.5*self.v0}, got {self.vb1}")
                return False
            self.log("Trace consistency passed.")
        else:
            self.log("Missing v0, va1, or vb1, skipping trace consistency.")
        return True

    def test_monotonicity(self):
        """Perturb an input and check trend."""
        original_mu = self.mu
        # Increase friction, expected v0min (from friction part) should increase
        self.mu = original_mu * 1.1
        new_threshold = 2 * math.sqrt(2 * self.mu * self.g * self.S)
        self.mu = original_mu
        original_threshold = 2 * math.sqrt(2 * self.mu * self.g * self.S)

        if new_threshold > original_threshold:
            self.log(f"Monotonicity passed: increasing μ increases friction threshold ({original_threshold} -> {new_threshold})")
            return True
        else:
            self.log(f"Monotonicity failed: μ increase did not raise threshold")
            return False

    def run_all_tests(self):
        """Execute all checks and return success and logs."""
        self.logs = []
        success = True
        tests = [
            self.test_elastic_collision,
            self.test_ball_b_energy,
            self.test_ball_b_circular_motion,
            self.test_ball_a_energy,
            self.test_friction_condition,
            self.test_range_consistency,
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity,
        ]
        for test in tests:
            try:
                if not test():
                    success = False
            except Exception as e:
                self.log(f"Test {test.__name__} crashed: {e}\n{traceback.format_exc()}")
                success = False
        return success, self.logs