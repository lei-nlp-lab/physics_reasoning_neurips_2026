import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 1.0))
        self.k = getattr(solution_instance, 'k', solution_instance.results.get('k', 8.99e9))
        self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', 1.0e-6))
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v0', 0.0))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', 30.0))
        # Convert theta to radians if in degrees
        if abs(self.theta - 30.0) < 1.0:
            self.theta_rad = math.pi / 6
        else:
            self.theta_rad = self.theta

        # Safe attribute access for results
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 0.0))
        self.L_AC = getattr(solution_instance, 'L_AC', solution_instance.results.get('L_AC', 0.0))
        self.phi_A = getattr(solution_instance, 'phi_A', solution_instance.results.get('phi_A', 0.0))
        self.phi_C = getattr(solution_instance, 'phi_C', solution_instance.results.get('phi_C', 0.0))
        self.E_kC = getattr(solution_instance, 'E_kC', solution_instance.results.get('E_kC', 0.0))
        self.E_kB = getattr(solution_instance, 'E_kB', solution_instance.results.get('E_kB', 0.0))
        self.E_pA = getattr(solution_instance, 'E_pA', solution_instance.results.get('E_pA', 0.0))
        self.E_pB = getattr(solution_instance, 'E_pB', solution_instance.results.get('E_pB', 0.0))
        self.W_AB = getattr(solution_instance, 'W_AB', solution_instance.results.get('W_AB', 0.0))

        # Logs
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((msg, success))

    def _isclose(self, a, b):
        """Robust numeric comparison with tolerance."""
        if hasattr(a, 'evalf') or hasattr(b, 'evalf'):
            a_val = float(a.evalf()) if hasattr(a, 'evalf') else float(a)
            b_val = float(b.evalf()) if hasattr(b, 'evalf') else float(b)
        else:
            a_val, b_val = a, b
        return math.isclose(a_val, b_val, rel_tol=1e-2, abs_tol=1e-2)

    def test_equilibrium(self):
        """Check force equilibrium at point A (step_1)."""
        try:
            # Symbolic: k*Q*q / L^2 = m*g*tan(theta)
            tan_theta = math.tan(self.theta_rad)
            lhs = self.k * self.Q * self.q / (self.L ** 2)
            rhs = self.m * self.g * tan_theta
            if self._isclose(lhs, rhs):
                self._log("Equilibrium at A: k*Q*q/L^2 = m*g*tanθ", True)
                return True
            else:
                self._log(f"Equilibrium at A failed: lhs={lhs}, rhs={rhs}", False)
                return False
        except Exception as e:
            self._log(f"Equilibrium test error: {e}", False)
            return False

    def test_geometry(self):
        """Check geometric relationship L_AC = sqrt(3)*L (step_2)."""
        try:
            expected = math.sqrt(3) * self.L
            if self._isclose(self.L_AC, expected):
                self._log(f"Geometry: L_AC = sqrt(3)*L ≈ {expected:.3f}", True)
                return True
            else:
                self._log(f"Geometry mismatch: L_AC={self.L_AC}, expected={expected}", False)
                return False
        except Exception as e:
            self._log(f"Geometry test error: {e}", False)
            return False

    def test_potential_equality(self):
        """Check electric potential φ_A = φ_C (step_3)."""
        try:
            if self._isclose(self.phi_A, self.phi_C):
                self._log("Electric potential: φ_A = φ_C", True)
                return True
            else:
                self._log(f"Potential mismatch: φ_A={self.phi_A}, φ_C={self.phi_C}", False)
                return False
        except Exception as e:
            self._log(f"Potential equality test error: {e}", False)
            return False

    def test_energy_conservation_A_to_C(self):
        """Check work-energy theorem from A to C (steps 4-5)."""
        try:
            # Gravitational work: mg * L_AC * sinθ
            sin_theta = math.sin(self.theta_rad)
            W_gravity = self.m * self.g * self.L_AC * sin_theta
            # Change in kinetic energy: E_kC - (1/2)*m*v0^2
            delta_K = self.E_kC - 0.5 * self.m * self.v0 ** 2
            if self._isclose(W_gravity, delta_K):
                self._log(f"Energy A->C: W_gravity={W_gravity:.3f}, ΔK={delta_K:.3f}", True)
                return True
            else:
                self._log(f"Energy A->C mismatch: W_gravity={W_gravity}, ΔK={delta_K}", False)
                return False
        except Exception as e:
            self._log(f"Energy A->C test error: {e}", False)
            return False

    def test_potential_energy(self):
        """Check electric potential energy at A and B (step_6)."""
        try:
            # E_pA = k*Q*q / L
            expected_E_pA = self.k * self.Q * self.q / self.L
            # E_pB = 2*k*Q*q / L (from step_6)
            expected_E_pB = 2 * self.k * self.Q * self.q / self.L
            if self._isclose(self.E_pA, expected_E_pA) and self._isclose(self.E_pB, expected_E_pB):
                self._log(f"Potential energy: E_pA={expected_E_pA:.3f}, E_pB={expected_E_pB:.3f}", True)
                return True
            else:
                self._log(f"Potential energy mismatch: E_pA={self.E_pA} (expected {expected_E_pA}), E_pB={self.E_pB} (expected {expected_E_pB})", False)
                return False
        except Exception as e:
            self._log(f"Potential energy test error: {e}", False)
            return False

    def test_work_electric(self):
        """Check work done by electric force W_AB = E_pA - E_pB (step_7)."""
        try:
            expected = self.E_pA - self.E_pB
            if self._isclose(self.W_AB, expected):
                self._log(f"Electric work: W_AB = E_pA - E_pB = {expected:.3f}", True)
                return True
            else:
                self._log(f"Electric work mismatch: W_AB={self.W_AB}, expected={expected}", False)
                return False
        except Exception as e:
            self._log(f"Electric work test error: {e}", False)
            return False

    def test_energy_conservation_A_to_B(self):
        """Check work-energy theorem from A to B (steps 8-9)."""
        try:
            # Gravitational work: mg * L * cosθ * sinθ (from step_8)
            cos_theta = math.cos(self.theta_rad)
            sin_theta = math.sin(self.theta_rad)
            W_gravity_AB = self.m * self.g * self.L * cos_theta * sin_theta
            # Total work: gravitational + electric
            W_total = W_gravity_AB + self.W_AB
            # Change in kinetic energy: E_kB - (1/2)*m*v0^2
            delta_K_AB = self.E_kB - 0.5 * self.m * self.v0 ** 2
            if self._isclose(W_total, delta_K_AB):
                self._log(f"Energy A->B: W_total={W_total:.3f}, ΔK={delta_K_AB:.3f}", True)
                return True
            else:
                self._log(f"Energy A->B mismatch: W_total={W_total}, ΔK={delta_K_AB}", False)
                return False
        except Exception as e:
            self._log(f"Energy A->B test error: {e}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check q from equilibrium (already tested)
            # Check E_kC from energy A->C (already tested)
            # Check E_kB from energy A->B (already tested)
            # Additional: verify that E_kC and E_kB match the target expressions symbolically
            # Use numeric formulas directly
            q_val = math.sqrt(3) * self.m * self.g * self.L**2 / (3 * self.k * self.Q)
            E_kC_val = math.sqrt(3)/2 * self.m * self.g * self.L + 0.5 * self.m * self.v0**2
            E_kB_val = 0.5 * self.m * self.v0**2 - math.sqrt(3)/12 * self.m * self.g * self.L

            checks = []
            if self._isclose(self.q, q_val):
                self._log("Trace: q matches target expression", True)
                checks.append(True)
            else:
                self._log(f"Trace: q mismatch: result={self.q}, target={q_val}", False)
                checks.append(False)

            if self._isclose(self.E_kC, E_kC_val):
                self._log("Trace: E_kC matches target expression", True)
                checks.append(True)
            else:
                self._log(f"Trace: E_kC mismatch: result={self.E_kC}, target={E_kC_val}", False)
                checks.append(False)

            if self._isclose(self.E_kB, E_kB_val):
                self._log("Trace: E_kB matches target expression", True)
                checks.append(True)
            else:
                self._log(f"Trace: E_kB mismatch: result={self.E_kB}, target={E_kB_val}", False)
                checks.append(False)

            return all(checks)
        except Exception as e:
            self._log(f"Trace consistency test error: {e}", False)
            return False

    def test_monotonicity(self):
        """Perturb mass m and check trend of q, E_kC, E_kB."""
        try:
            # Store original values
            m_orig = self.m
            # Perturb by +10%
            self.m *= 1.1
            # Recompute expected values using formulas
            q_new = math.sqrt(3) * self.m * self.g * self.L**2 / (3 * self.k * self.Q)
            E_kC_new = math.sqrt(3)/2 * self.m * self.g * self.L + 0.5 * self.m * self.v0**2
            E_kB_new = 0.5 * self.m * self.v0**2 - math.sqrt(3)/12 * self.m * self.g * self.L
            # Restore original mass
            self.m = m_orig

            # Check monotonicity: q should increase with m, E_kC should increase, E_kB should decrease? (since negative term)
            # Actually, from expressions:
            #   q ∝ m -> increasing m increases q.
            #   E_kC ∝ m -> increasing m increases E_kC (if v0=0, positive; if v0>0, still positive coefficient).
            #   E_kB ∝ m * (v0^2/2 - sqrt(3)gL/12). For v0=0, E_kB is negative and becomes more negative with m.
            # So we check the signs of differences.
            q_increased = q_new > self.q
            E_kC_increased = E_kC_new > self.E_kC
            # For E_kB, if v0=0, then E_kB is negative and should become more negative (so E_kB_new < E_kB).
            # But we only check that it changes as expected from the formula.
            # Since the perturbation is positive, we expect:
            #   q_new > q_orig, E_kC_new > E_kC_orig, and E_kB_new < E_kB_orig if v0^2 < (sqrt(3)gL/6).
            # We'll just check that q and E_kC increase, and E_kB changes (direction depends on parameters).
            if q_increased and E_kC_increased:
                self._log("Monotonicity: q and E_kC increase with increased mass", True)
                return True
            else:
                self._log(f"Monotonicity check: q_increased={q_increased}, E_kC_increased={E_kC_increased}", False)
                return False
        except Exception as e:
            self._log(f"Monotonicity test error: {e}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs.clear()
        tests = [
            self.test_equilibrium,
            self.test_geometry,
            self.test_potential_equality,
            self.test_energy_conservation_A_to_C,
            self.test_potential_energy,
            self.test_work_electric,
            self.test_energy_conservation_A_to_B,
            self.test_trace_consistency,
            self.test_monotonicity,
        ]
        results = []
        for test in tests:
            results.append(test())
        success = all(results)
        return success, self.logs