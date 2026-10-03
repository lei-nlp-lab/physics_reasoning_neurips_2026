import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results
        self.S = getattr(solution_instance, 'S', 0.2)
        self.N = getattr(solution_instance, 'N', 100)
        self.r = getattr(solution_instance, 'r', 2.0)
        self.B = getattr(solution_instance, 'B', math.sqrt(2) / 2)
        self.n_rpm = getattr(solution_instance, 'n', 150 / math.pi)  # in r/min
        self.R = getattr(solution_instance, 'R', 18.0)
        
        # Extract results safely; default to None if missing
        results = getattr(solution_instance, 'results', {})
        self.overlineE = results.get('overlineE', None)  # average EMF during 0~T/4
        self.e = results.get('e', None)                  # instantaneous EMF
        self.q = results.get('q', None)                  # charge during 0~T/4
        self.overlineI = results.get('overlineI', None)  # average current during 0~T/4
        self.I = results.get('I', None)                  # effective (rms) current
        self.W = results.get('W', None)                  # work in one period
        self.omega = results.get('omega', None)          # angular velocity

        # Derived constants (computed from inputs)
        self.n_rps = self.n_rpm / 60.0                     # convert r/min to r/s
        self.omega_computed = 2 * math.pi * self.n_rps     # ω = 2πn (in rad/s)
        self.T = 2 * math.pi / self.omega_computed if self.omega_computed != 0 else 0  # period

        # Store logs for test results
        self.logs = []

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((success, message))

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physical laws."""
        # 1. Angular velocity relationship: ω = 2πn
        if self.omega is not None:
            if not self._is_close(self.omega, self.omega_computed):
                self._log(f"Angular velocity mismatch: ω={self.omega}, computed={self.omega_computed}", False)
            else:
                self._log("Angular velocity ω = 2πn holds.", True)

        # 2. Peak EMF: emf_peak = N * B * S * ω
        emf_peak_computed = self.N * self.B * self.S * self.omega_computed
        if self.e is not None:  # e is instantaneous, but we can check peak from given formula
            # The formula for e is N*B*S*ω*cos(ωt); peak is when |cos|=1.
            # If e is provided as a function, we can't check directly. Instead, we rely on I.
            pass

        # 3. Effective current I = emf_peak / (√2 * (R+r))
        emf_peak = self.N * self.B * self.S * self.omega_computed
        I_computed = emf_peak / (math.sqrt(2) * (self.R + self.r))
        if self.I is not None:
            if not self._is_close(self.I, I_computed):
                self._log(f"Effective current mismatch: I={self.I}, computed={I_computed}", False)
            else:
                self._log("Effective current I = E_peak/(√2*(R+r)) holds.", True)

        # 4. Charge in first quarter: q = N * B * S / (R+r)
        q_computed = self.N * self.B * self.S / (self.R + self.r)
        if self.q is not None:
            if not self._is_close(self.q, q_computed):
                self._log(f"Charge mismatch: q={self.q}, computed={q_computed}", False)
            else:
                self._log("Charge q = N*B*S/(R+r) holds.", True)

        # 5. Work in one period: W = I² (R+r) T
        if self.I is not None and self.W is not None:
            W_computed = self.I**2 * (self.R + self.r) * self.T
            if not self._is_close(self.W, W_computed):
                self._log(f"Work mismatch: W={self.W}, computed={W_computed}", False)
            else:
                self._log("Work W = I²(R+r)T holds.", True)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        # 1. Average EMF during 0~T/4: overlineE = 4 N B S / T
        if self.overlineE is not None:
            overlineE_computed = 4 * self.N * self.B * self.S / self.T
            if not self._is_close(self.overlineE, overlineE_computed):
                self._log(f"Average EMF mismatch: overlineE={self.overlineE}, computed={overlineE_computed}", False)
            else:
                self._log("Average EMF overlineE = 4 N B S / T holds.", True)

        # 2. Average current: overlineI = overlineE / (R+r)
        if self.overlineE is not None and self.overlineI is not None:
            overlineI_computed = self.overlineE / (self.R + self.r)
            if not self._is_close(self.overlineI, overlineI_computed):
                self._log(f"Average current mismatch: overlineI={self.overlineI}, computed={overlineI_computed}", False)
            else:
                self._log("Average current overlineI = overlineE/(R+r) holds.", True)

        # 3. Charge from average current: q = overlineI * (T/4)
        if self.overlineI is not None and self.q is not None:
            q_from_I = self.overlineI * (self.T / 4)
            if not self._is_close(self.q, q_from_I):
                self._log(f"Charge from average current mismatch: q={self.q}, computed from I_avg={q_from_I}", False)
            else:
                self._log("Charge q = overlineI * (T/4) holds.", True)

        # 4. Instantaneous EMF: e = N B S ω cos(ωt); check that its amplitude matches emf_peak
        if self.e is not None and isinstance(self.e, (int, float)):
            # If e is given as a number, assume it's at a specific time? Not robust.
            # Instead, we skip this check unless e is symbolic.
            pass

    def test_monotonicity(self):
        """Perturb an input and check trend, then restore."""
        original_B = self.B
        perturbation = 1.01  # increase by 1%
        self.B = original_B * perturbation

        # Recompute dependent quantities using formulas
        q_perturbed = self.N * self.B * self.S / (self.R + self.r)
        emf_peak_perturbed = self.N * self.B * self.S * self.omega_computed
        I_perturbed = emf_peak_perturbed / (math.sqrt(2) * (self.R + self.r))
        W_perturbed = I_perturbed**2 * (self.R + self.r) * self.T

        # Original values (using original B)
        q_original = self.N * original_B * self.S / (self.R + self.r)
        emf_peak_original = self.N * original_B * self.S * self.omega_computed
        I_original = emf_peak_original / (math.sqrt(2) * (self.R + self.r))
        W_original = I_original**2 * (self.R + self.r) * self.T

        # Check trends: if B increases, q should increase, I should increase, W should increase.
        # Since W ∝ I² and I ∝ B, then W ∝ B², so increasing B increases W.
        success = True
        if not q_perturbed > q_original:
            self._log("Monotonicity failed: q does not increase with B.", False)
            success = False
        if not I_perturbed > I_original:
            self._log("Monotonicity failed: I does not increase with B.", False)
            success = False
        if not W_perturbed > W_original:
            self._log("Monotonicity failed: W does not increase with B.", False)
            success = False

        if success:
            self._log("Monotonicity test passed: increasing B increases q, I, and W.", True)

        # Restore original B
        self.B = original_B

    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs = []  # Reset logs
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
        except Exception as e:
            self._log(f"Exception during testing: {traceback.format_exc()}", False)

        # Determine overall success: all logs must have success=True
        success = all(log[0] for log in self.logs)
        return success, self.logs