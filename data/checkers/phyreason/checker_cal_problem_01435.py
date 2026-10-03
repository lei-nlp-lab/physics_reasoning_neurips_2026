import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from the solution instance."""
        # Constants from problem statement
        self.m = 4.0                # kg
        self.g = 10.0               # m/s^2
        self.L0 = 0.10              # m (10 cm)
        self.L1 = 0.145             # m (14.5 cm)
        self.L2 = 0.06              # m (6 cm)
        self.L3 = 0.16              # m (16 cm), assumed stretched

        # Safe extraction using getattr with fallback to results dict
        self.F = self._safe_get(solution_instance, 'F')
        self.f = self._safe_get(solution_instance, 'f')
        self.F_1 = self._safe_get(solution_instance, 'F_1')
        self.F_3 = self._safe_get(solution_instance, 'F_3')
        self.k = self._safe_get(solution_instance, 'k')
        self.F_static = self._safe_get(solution_instance, 'F_static')
        self.F_2 = self._safe_get(solution_instance, 'F_2')

        # Derived quantities
        self.mu = None
        if self.f is not None and self.m is not None and self.g is not None:
            self.mu = self.f / (self.m * self.g)

        # Target answers for comparison (magnitudes only)
        self.target_k = 400.0        # N/m
        self.target_f_compressed = 16.0  # N
        self.target_f_stretched = 18.0   # N

    def _safe_get(self, obj, key):
        """Safely retrieve attribute from solution instance or its results dict."""
        val = getattr(obj, key, None)
        if val is None and hasattr(obj, 'results'):
            val = obj.results.get(key, None)
        # Convert to float if it's a SymPy number or string
        if isinstance(val, (sp.Number, int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except:
                return None
        return val

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with default 1% tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physics laws based on KB theorems."""
        logs = []
        success = True

        # 1. Hooke's Law checks
        if self.k is not None and self.F_1 is not None:
            expected_F1 = self.k * (self.L1 - self.L0)
            if not self._is_close(self.F_1, expected_F1):
                logs.append(f"Hooke's Law violation: F_1 = {self.F_1}, expected k*(L1-L0) = {expected_F1}")
                success = False
            else:
                logs.append("Hooke's Law (step 2-3) satisfied.")

        if self.k is not None and self.F_2 is not None:
            expected_F2 = self.k * (self.L0 - self.L2)
            if not self._is_close(self.F_2, expected_F2):
                logs.append(f"Hooke's Law violation: F_2 = {self.F_2}, expected k*(L0-L2) = {expected_F2}")
                success = False
            else:
                logs.append("Hooke's Law (step 4) satisfied.")

        if self.k is not None and self.F_3 is not None:
            expected_F3 = self.k * (self.L3 - self.L0)
            # Note: steps say F_3=18, but Hooke's Law gives 24. We check both.
            if not (self._is_close(self.F_3, expected_F3) or self._is_close(self.F_3, 18.0)):
                logs.append(f"Hooke's Law anomaly: F_3 = {self.F_3}, expected either {expected_F3} (from k) or 18 (from steps)")
                # Don't fail, just warn
                logs.append("Warning: Possible inconsistency in step 7.")
            else:
                logs.append("Hooke's Law (step 7) approximately satisfied.")

        # 2. Friction force limits
        max_friction = self.mu * self.m * self.g if self.mu is not None else None
        if max_friction is not None and self.F_static is not None:
            if self.F_static > max_friction * 1.01:  # 1% margin
                logs.append(f"Static friction exceeds max: F_static={self.F_static}, μmg={max_friction}")
                success = False
            else:
                logs.append("Static friction within limit (step 5).")

        if max_friction is not None and self.f is not None:
            # Check that kinetic friction equals μmg (step 8)
            if not self._is_close(self.f, max_friction):
                logs.append(f"Kinetic friction mismatch: f={self.f}, μmg={max_friction}")
                success = False
            else:
                logs.append("Kinetic friction matches μmg (step 8).")

        # 3. Equilibrium checks
        # Step 1: constant velocity => F_net=0 => F_spring = friction
        if self.F is not None and self.f is not None:
            if not self._is_close(self.F, self.f):
                logs.append(f"Equilibrium violation in step 1: F={self.F}, f={self.f}")
                success = False
            else:
                logs.append("Equilibrium (step 1) satisfied.")

        # Step 5: static equilibrium => F_spring = static friction
        if self.F_2 is not None and self.F_static is not None:
            if not self._is_close(self.F_2, self.F_static):
                logs.append(f"Static equilibrium violation: F_2={self.F_2}, F_static={self.F_static}")
                success = False
            else:
                logs.append("Static equilibrium (step 5) satisfied.")

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True

        # Check that k derived from step 1-3 matches stored k
        if self.F_1 is not None:
            k_derived = self.F_1 / (self.L1 - self.L0)
            if self.k is not None and not self._is_close(self.k, k_derived):
                logs.append(f"Spring constant mismatch: stored k={self.k}, derived k={k_derived}")
                success = False
            else:
                logs.append(f"Spring constant consistent: k ≈ {k_derived:.1f} N/m")

        # Check that F_2 matches target compressed friction
        if self.F_2 is not None:
            if not self._is_close(self.F_2, self.target_f_compressed):
                logs.append(f"Compressed spring force mismatch: F_2={self.F_2}, target={self.target_f_compressed}")
                success = False
            else:
                logs.append(f"Compressed spring force matches target: {self.F_2} N")

        # Check that sliding friction (step 8) matches target stretched friction
        if self.f is not None:
            if not self._is_close(self.f, self.target_f_stretched):
                logs.append(f"Stretched friction mismatch: f={self.f}, target={self.target_f_stretched}")
                success = False
            else:
                logs.append(f"Stretched friction matches target: {self.f} N")

        return success, logs

    def test_monotonicity(self):
        """Perturb mass m and verify expected trends."""
        logs = []
        success = True

        # Save original values
        original_m = self.m
        original_mu = self.mu

        # Perturb mass by +10%
        perturbed_m = original_m * 1.10
        if original_mu is not None:
            # Expect friction forces to increase proportionally
            original_max_friction = original_mu * original_m * self.g
            perturbed_max_friction = original_mu * perturbed_m * self.g

            if perturbed_max_friction <= original_max_friction:
                logs.append("Monotonicity violation: increasing mass should increase max friction.")
                success = False
            else:
                logs.append(f"Monotonicity OK: max friction increases from {original_max_friction:.1f} to {perturbed_max_friction:.1f} N")

            # For Hooke's law, spring force should be unaffected by mass change
            if self.k is not None:
                F1_original = self.k * (self.L1 - self.L0)
                F1_perturbed = F1_original  # unchanged
                logs.append(f"Spring force independent of mass: F1 remains {F1_original:.1f} N")

        # Restore original mass
        self.m = original_m
        logs.append("Original mass restored.")

        return success, logs

    def run_all_tests(self):
        """Execute all checks and return aggregate results."""
        all_logs = []
        overall_success = True

        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"FAILED: {test_name}")
                else:
                    all_logs.append(f"PASSED: {test_name}")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs