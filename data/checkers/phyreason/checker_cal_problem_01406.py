import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract ALL values from the results dictionary ONLY
        # The solution_instance should have a 'results' dictionary
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        else:
            results = {}
        
        # Extract required variables with defaults
        self.m = results.get('m', 30 * math.sqrt(3) * 1e-3)  # kg
        self.g = results.get('g', 10.0)                     # m/s²
        # theta might be stored in degrees or radians; assume degrees
        theta_deg = results.get('theta', 30.0)
        self.theta = math.radians(theta_deg)                # convert to radians
        self.h = results.get('h', 0.1)                      # m (10 cm)
        self.k = results.get('k', 9e9)                      # N·m²/C²
        
        # Extract results
        self.Q = results.get('Q', 0.0)                      # C
        self.F = results.get('F', 0.0)                      # N
        self.L = results.get('L', 0.0)                      # m

    # ------------------ Individual test methods ------------------
    def test_geometry(self) -> bool:
        """Check L = h / tan(theta)."""
        expected_L = self.h / math.tan(self.theta)
        return math.isclose(self.L, expected_L, rel_tol=1e-2, abs_tol=1e-2)

    def test_equilibrium(self) -> bool:
        """Check F = m * g * tan(theta)."""
        expected_F = self.m * self.g * math.tan(self.theta)
        return math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2)

    def test_coulomb(self) -> bool:
        """Check F = k * Q² / L²."""
        if self.L == 0:
            return False
        expected_F = self.k * self.Q**2 / self.L**2
        return math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2)

    def test_Q_value(self) -> bool:
        """Check Q ≈ 1.0e-6 C."""
        return math.isclose(self.Q, 1e-6, rel_tol=1e-2, abs_tol=1e-2)

    # ------------------ Composite test methods ------------------
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify fundamental physical invariants."""
        tests = [
            ("Geometry L = h / tan(θ)", self.test_geometry()),
            ("Equilibrium F = m g tan(θ)", self.test_equilibrium()),
            ("Coulomb's law F = k Q² / L²", self.test_coulomb()),
            ("Q ≈ 1.0e-6 C", self.test_Q_value())
        ]
        logs = []
        all_ok = True
        for name, ok in tests:
            if ok:
                logs.append(f"PASS: {name}")
            else:
                logs.append(f"FAIL: {name}")
                all_ok = False
        return all_ok, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer."""
        if self.F <= 0 or self.L <= 0:
            return False, ["Trace consistency: F or L is non-positive."]
        Q_calc = math.sqrt(self.F * self.L**2 / self.k)
        ok = math.isclose(self.Q, Q_calc, rel_tol=1e-2, abs_tol=1e-2)
        logs = [
            f"Trace consistency: Q from F and L = {Q_calc:.2e}, "
            f"given Q = {self.Q:.2e}, match = {ok}"
        ]
        return ok, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb mass m and verify expected trend in Q."""
        original_m = self.m
        perturbed_m = original_m * 1.1  # +10%
        # Recalculate F and Q using physical formulas
        F_new = perturbed_m * self.g * math.tan(self.theta)
        Q_new = math.sqrt(F_new * self.L**2 / self.k)
        # Check monotonic increase
        ok = Q_new > self.Q
        logs = [
            f"Monotonicity: m {original_m:.4f} -> {perturbed_m:.4f} kg, "
            f"Q {self.Q:.2e} -> {Q_new:.2e} C, increasing = {ok}"
        ]
        return ok, logs

    # ------------------ Main test runner ------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return aggregate result."""
        all_ok = True
        logs = []

        # Invariants
        ok, inv_logs = self.test_invariants()
        all_ok = all_ok and ok
        logs.extend(inv_logs)

        # Trace consistency
        ok, trace_logs = self.test_trace_consistency()
        all_ok = all_ok and ok
        logs.extend(trace_logs)

        # Monotonicity
        ok, mono_logs = self.test_monotonicity()
        all_ok = all_ok and ok
        logs.extend(mono_logs)

        return all_ok, logs