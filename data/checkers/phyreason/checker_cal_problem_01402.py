import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults
        self.d = getattr(solution_instance, 'd', 0.4)  # 40 cm -> 0.4 m
        self.E = getattr(solution_instance, 'E', 24.0)
        self.r = getattr(solution_instance, 'r', 1.0)
        self.R = getattr(solution_instance, 'R', 15.0)
        self.v0 = getattr(solution_instance, 'v0', 4.0)
        self.q = getattr(solution_instance, 'q', 0.01)  # 1e-2 C
        self.m = getattr(solution_instance, 'm', 0.02)  # 2e-2 kg
        self.g = getattr(solution_instance, 'g', 10.0)

        # Safely extract results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.U_AB = results.get('U_AB', 0.0)
        self.I = results.get('I', 0.0)
        self.R_slip = results.get('R_slip', 0.0)
        self.P_out = results.get('P_out', 0.0)

        # For symbolic checks
        self.symbols = {
            'd': self.d, 'E': self.E, 'r': self.r, 'R': self.R,
            'v0': self.v0, 'q': self.q, 'm': self.m, 'g': self.g,
            'U_AB': self.U_AB, 'I': self.I, 'R_slip': self.R_slip,
            'P_out': self.P_out
        }
        self.logs = []

    def _log(self, msg, success=None):
        if success is not None:
            msg = f"{'PASS' if success else 'FAIL'}: {msg}"
        self.logs.append(msg)

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify Work-Energy Theorem and Ohm's Law symbolically and numerically."""
        try:
            # Work-Energy Theorem: -m*g*d - q*U_AB = 0 - (1/2)*m*v0**2
            # Rearranged: U_AB = (0.5*m*v0**2 - m*g*d) / q
            lhs = -self.m * self.g * self.d - self.q * self.U_AB
            rhs = 0 - 0.5 * self.m * self.v0**2
            if self._is_close(lhs, rhs):
                self._log(f"Work-energy theorem holds: lhs={lhs:.3f}, rhs={rhs:.3f}", True)
            else:
                self._log(f"Work-energy theorem mismatch: lhs={lhs:.3f}, rhs={rhs:.3f}", False)

            # Ohm's Law for closed circuit: I = (E - U_AB) / (R + r)
            i_calc = (self.E - self.U_AB) / (self.R + self.r)
            if self._is_close(self.I, i_calc):
                self._log(f"Ohm's law (closed circuit) holds: I={self.I:.3f}, calculated={i_calc:.3f}", True)
            else:
                self._log(f"Ohm's law (closed circuit) mismatch: I={self.I:.3f}, calculated={i_calc:.3f}", False)

            # Ohm's Law for rheostat: R_slip = U_AB / I (if I != 0)
            if abs(self.I) > 1e-12:
                r_calc = self.U_AB / self.I
                if self._is_close(self.R_slip, r_calc):
                    self._log(f"Ohm's law (rheostat) holds: R_slip={self.R_slip:.3f}, calculated={r_calc:.3f}", True)
                else:
                    self._log(f"Ohm's law (rheostat) mismatch: R_slip={self.R_slip:.3f}, calculated={r_calc:.3f}", False)
            else:
                self._log("Skipping rheostat Ohm's law because I is zero.", None)

            # Power formula: P_out = I**2 * (R + R_slip)
            p_calc = self.I**2 * (self.R + self.R_slip)
            if self._is_close(self.P_out, p_calc):
                self._log(f"Power formula holds: P_out={self.P_out:.3f}, calculated={p_calc:.3f}", True)
            else:
                self._log(f"Power formula mismatch: P_out={self.P_out:.3f}, calculated={p_calc:.3f}", False)

        except Exception as e:
            self._log(f"Exception in test_invariants: {e}\n{traceback.format_exc()}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate results against final answers."""
        try:
            # Step 2: U_AB from work-energy theorem
            u_calc = (0.5 * self.m * self.v0**2 - self.m * self.g * self.d) / self.q
            if self._is_close(self.U_AB, u_calc):
                self._log(f"U_AB consistent with work-energy: {self.U_AB:.3f} vs {u_calc:.3f}", True)
            else:
                self._log(f"U_AB inconsistent: {self.U_AB:.3f} vs {u_calc:.3f}", False)

            # Step 3: I from closed-circuit Ohm's law
            i_calc = (self.E - self.U_AB) / (self.R + self.r)
            if self._is_close(self.I, i_calc):
                self._log(f"I consistent with closed-circuit Ohm's law: {self.I:.3f} vs {i_calc:.3f}", True)
            else:
                self._log(f"I inconsistent: {self.I:.3f} vs {i_calc:.3f}", False)

            # Step 4: R_slip from Ohm's law
            if abs(self.I) > 1e-12:
                r_calc = self.U_AB / self.I
                if self._is_close(self.R_slip, r_calc):
                    self._log(f"R_slip consistent with Ohm's law: {self.R_slip:.3f} vs {r_calc:.3f}", True)
                else:
                    self._log(f"R_slip inconsistent: {self.R_slip:.3f} vs {r_calc:.3f}", False)
            else:
                self._log("Skipping R_slip consistency (I=0)", None)

            # Step 5: P_out from power formula
            p_calc = self.I**2 * (self.R + self.R_slip)
            if self._is_close(self.P_out, p_calc):
                self._log(f"P_out consistent with power formula: {self.P_out:.3f} vs {p_calc:.3f}", True)
            else:
                self._log(f"P_out inconsistent: {self.P_out:.3f} vs {p_calc:.3f}", False)

        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {e}\n{traceback.format_exc()}", False)

    def test_monotonicity(self):
        """Perturb initial velocity v0 and check trend of U_AB, I, R_slip, P_out."""
        original_v0 = self.v0
        try:
            # Perturb v0 by +10%
            self.v0 *= 1.1

            # Recalculate expected values using formulas
            u_new = (0.5 * self.m * self.v0**2 - self.m * self.g * self.d) / self.q
            i_new = (self.E - u_new) / (self.R + self.r)
            r_new = u_new / i_new if abs(i_new) > 1e-12 else 0.0
            p_new = i_new**2 * (self.R + r_new)

            # Check monotonic trends:
            # As v0 increases, initial KE increases, so U_AB must increase to stop the ball.
            if u_new > self.U_AB:
                self._log(f"U_AB increases with increased v0 (expected): {self.U_AB:.3f} -> {u_new:.3f}", True)
            else:
                self._log(f"U_AB does not increase with increased v0 (unexpected): {self.U_AB:.3f} -> {u_new:.3f}", False)

            # I = (E - U_AB)/(R+r): if U_AB increases, I should decrease.
            if i_new < self.I:
                self._log(f"I decreases with increased v0 (expected): {self.I:.3f} -> {i_new:.3f}", True)
            else:
                self._log(f"I does not decrease with increased v0 (unexpected): {self.I:.3f} -> {i_new:.3f}", False)

            # R_slip = U_AB/I: U_AB increases, I decreases => R_slip should increase.
            if r_new > self.R_slip:
                self._log(f"R_slip increases with increased v0 (expected): {self.R_slip:.3f} -> {r_new:.3f}", True)
            else:
                self._log(f"R_slip does not increase with increased v0 (unexpected): {self.R_slip:.3f} -> {r_new:.3f}", False)

            # P_out = I^2*(R+R_slip): I decreases, R_slip increases -> sign of change is ambiguous.
            # We only check that the recalculated value is positive and finite.
            if p_new > 0 and not math.isinf(p_new):
                self._log(f"P_out remains positive and finite after perturbation: {self.P_out:.3f} -> {p_new:.3f}", True)
            else:
                self._log(f"P_out invalid after perturbation: {p_new:.3f}", False)

        except Exception as e:
            self._log(f"Exception in test_monotonicity: {e}\n{traceback.format_exc()}", False)
        finally:
            # Restore original v0
            self.v0 = original_v0

    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        self.logs.clear()
        self.logs.append("Starting PhysicsChecker tests...")

        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()

        # Determine overall success: all checks must pass (no FAIL logs)
        success = not any(log.startswith("FAIL") for log in self.logs)
        self.logs.insert(0, f"Overall test result: {'PASS' if success else 'FAIL'}")

        return success, self.logs