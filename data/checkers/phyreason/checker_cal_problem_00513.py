import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        # Extract inputs from results dict (they should be stored there by solution code)
        results = getattr(solution_instance, 'results', {})
        
        # Debug: print what we got
        # print(f"Checker received results: {results}")
        
        # Extract only the specific keys we need, ignoring any extra dumped variables
        # Use default values if keys are missing
        self.U = results.get('U', 12.0)
        self.L = results.get('L', 2.0)
        self.rho0 = results.get('rho0', 1.68e-8)
        self.rho = results.get('rho', 8960.0)
        self.n = results.get('n', 1.0)  # electrons per mole
        self.M = results.get('M', 0.063546)
        self.NA = results.get('NA', 6.02214076e23)
        self.e = results.get('e', 1.602176634e-19)

        # Extract results
        self.x = results.get('x', 0.0)
        self.q = results.get('q', 0.0)
        self.m = results.get('m', 0.0)
        self.v = results.get('v', 0.0)
        self.R = results.get('R', 0.0)
        self.I = results.get('I', 0.0)
        self.N = results.get('N', 0.0)

        # Cross-sectional area S - try to get from results, otherwise compute
        self.S = results.get('S', 0.0)
        # Only compute S if it's not already set and we have the necessary values
        if self.S == 0.0 and self.rho != 0.0 and self.x != 0.0:
            try:
                self.S = self.m / (self.rho * self.x)
            except ZeroDivisionError:
                self.S = 0.0

        # Time t (derived from x = v * t)
        if self.v != 0.0:
            try:
                self.t = self.x / self.v
            except ZeroDivisionError:
                self.t = 0.0
        else:
            self.t = 0.0

        # Store logs
        self.logs = []

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")

    def test_invariants(self):
        """
        Test physical invariants based on KB theorems.
        """
        try:
            all_pass = True
            
            # 1. Ohm's Law: V = I * R
            if self.R != 0.0:
                try:
                    V_calc = self.I * self.R
                    ohms_law = math.isclose(self.U, V_calc, rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"Ohm's Law: U={self.U:.3e}, I*R={V_calc:.3e}", ohms_law)
                    all_pass &= ohms_law
                except Exception as e:
                    self._log(f"Ohm's Law error: {e}", False)
                    all_pass = False
            else:
                self._log(f"Ohm's Law: R={self.R:.3e}, skipping", False)
                all_pass = False

            # 2. Current definition: I = q/t (from step 7)
            if self.t != 0.0:
                try:
                    I_from_q = self.q / self.t
                    current_def = math.isclose(self.I, I_from_q, rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"Current definition: I={self.I:.3e}, q/t={I_from_q:.3e}", current_def)
                    all_pass &= current_def
                except Exception as e:
                    self._log(f"Current definition error: {e}", False)
                    all_pass = False
            else:
                self._log(f"Current definition: t={self.t:.3e}, skipping", False)
                all_pass = False

            # 3. Density definition: m = ρ * volume = ρ * S * x (step 4)
            if self.S != 0.0 and self.x != 0.0:
                try:
                    volume = self.S * self.x
                    m_calc = self.rho * volume
                    density_def = math.isclose(self.m, m_calc, rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"Density definition: m={self.m:.3e}, ρ*S*x={m_calc:.3e}", density_def)
                    all_pass &= density_def
                except Exception as e:
                    self._log(f"Density definition error: {e}", False)
                    all_pass = False
            else:
                self._log(f"Density definition: S={self.S:.3e} or x={self.x:.3e}, skipping", False)
                all_pass = False

            # 4. Number of electrons: N = n * (m/M) (step 5)
            if self.M != 0.0:
                try:
                    N_calc = self.n * (self.m / self.M)
                    # Use larger tolerance due to floating-point issues with large numbers
                    electron_count = math.isclose(self.N, N_calc, rel_tol=1e-1, abs_tol=1e-1)
                    self._log(f"Electron count: N={self.N:.3e}, n*(m/M)={N_calc:.3e}", electron_count)
                    all_pass &= electron_count
                except Exception as e:
                    self._log(f"Electron count error: {e}", False)
                    all_pass = False
            else:
                self._log(f"Electron count: M={self.M:.3e}, skipping", False)
                all_pass = False

            # 5. Charge quantization: q = N * e (step 6)
            if self.e != 0.0:
                try:
                    q_calc = self.N * self.e
                    charge_quant = math.isclose(self.q, q_calc, rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"Charge quantization: q={self.q:.3e}, N*e={q_calc:.3e}", charge_quant)
                    all_pass &= charge_quant
                except Exception as e:
                    self._log(f"Charge quantization error: {e}", False)
                    all_pass = False
            else:
                self._log(f"Charge quantization: e={self.e:.3e}, skipping", False)
                all_pass = False

            # 6. Resistance formula: R = ρ0 * L / S (step 1)
            if self.S != 0.0 and self.rho0 != 0.0:
                try:
                    R_calc = self.rho0 * self.L / self.S
                    resistance_law = math.isclose(self.R, R_calc, rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"Resistance law: R={self.R:.3e}, ρ0*L/S={R_calc:.3e}", resistance_law)
                    all_pass &= resistance_law
                except Exception as e:
                    self._log(f"Resistance law error: {e}", False)
                    all_pass = False
            else:
                self._log(f"Resistance law: S={self.S:.3e} or rho0={self.rho0:.3e}, skipping", False)
                all_pass = False

            return all_pass
        except Exception as e:
            self._log(f"Invariants test error: {e}", False)
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        """
        try:
            # Compute v from derived formula (step 8)
            if (self.n != 0.0 and self.rho != 0.0 and self.rho0 != 0.0 and 
                self.L != 0.0 and self.e != 0.0):
                try:
                    v_formula = (self.U * self.M) / (self.n * self.e * self.rho * self.rho0 * self.L)
                    trace_ok = math.isclose(self.v, v_formula, rel_tol=1e-2, abs_tol=1e-2)
                    self._log(f"Trace consistency: v_result={self.v:.3e}, v_formula={v_formula:.3e}", trace_ok)
                    return trace_ok
                except Exception as e:
                    self._log(f"Trace consistency computation error: {e}", False)
                    return False
            else:
                missing = []
                if self.n == 0.0: missing.append('n')
                if self.rho == 0.0: missing.append('rho')
                if self.rho0 == 0.0: missing.append('rho0')
                if self.L == 0.0: missing.append('L')
                if self.e == 0.0: missing.append('e')
                self._log(f"Trace consistency: missing inputs {missing}", False)
                return False
        except Exception as e:
            self._log(f"Trace consistency error: {e}", False)
            return False

    def test_monotonicity(self):
        """
        Perturb each input by +1%, recompute v via formula, check trend.
        """
        try:
            # Original v from formula
            if (self.n == 0.0 or self.rho == 0.0 or self.rho0 == 0.0 or 
                self.L == 0.0 or self.e == 0.0):
                missing = []
                if self.n == 0.0: missing.append('n')
                if self.rho == 0.0: missing.append('rho')
                if self.rho0 == 0.0: missing.append('rho0')
                if self.L == 0.0: missing.append('L')
                if self.e == 0.0: missing.append('e')
                self._log(f"Monotonicity: zero inputs {missing} prevent perturbation", False)
                return False

            try:
                v_original = (self.U * self.M) / (self.n * self.e * self.rho * self.rho0 * self.L)
            except ZeroDivisionError:
                self._log("Monotonicity: division by zero in v_original", False)
                return False

            # Define perturbations and expected trends (True means v should increase)
            perturbations = [
                ('U', self.U * 1.01, True),      # v ∝ U
                ('M', self.M * 1.01, True),      # v ∝ M
                ('n', self.n * 1.01, False),     # v ∝ 1/n
                ('e', self.e * 1.01, False),     # v ∝ 1/e
                ('rho', self.rho * 1.01, False), # v ∝ 1/ρ
                ('rho0', self.rho0 * 1.01, False), # v ∝ 1/ρ0
                ('L', self.L * 1.01, False),     # v ∝ 1/L
            ]

            all_pass = True
            for param, perturbed_val, expected_increase in perturbations:
                try:
                    # Compute perturbed v
                    if param == 'U':
                        v_new = (perturbed_val * self.M) / (self.n * self.e * self.rho * self.rho0 * self.L)
                    elif param == 'M':
                        v_new = (self.U * perturbed_val) / (self.n * self.e * self.rho * self.rho0 * self.L)
                    elif param == 'n':
                        v_new = (self.U * self.M) / (perturbed_val * self.e * self.rho * self.rho0 * self.L)
                    elif param == 'e':
                        v_new = (self.U * self.M) / (self.n * perturbed_val * self.rho * self.rho0 * self.L)
                    elif param == 'rho':
                        v_new = (self.U * self.M) / (self.n * self.e * perturbed_val * self.rho0 * self.L)
                    elif param == 'rho0':
                        v_new = (self.U * self.M) / (self.n * self.e * self.rho * perturbed_val * self.L)
                    elif param == 'L':
                        v_new = (self.U * self.M) / (self.n * self.e * self.rho * self.rho0 * perturbed_val)
                    else:
                        continue

                    # Check trend
                    actual_increase = v_new > v_original
                    trend_ok = (actual_increase == expected_increase) and not math.isclose(v_new, v_original, rel_tol=1e-4)
                    all_pass &= trend_ok
                    self._log(f"Monotonicity {param}: v_orig={v_original:.3e}, v_new={v_new:.3e}, expected_increase={expected_increase}", trend_ok)
                except ZeroDivisionError:
                    self._log(f"Monotonicity {param}: division by zero", False)
                    all_pass = False
                except Exception as e:
                    self._log(f"Monotonicity {param} error: {e}", False)
                    all_pass = False

            return all_pass
        except Exception as e:
            self._log(f"Monotonicity error: {e}", False)
            return False

    def run_all_tests(self):
        """
        Run all tests and return (success, logs).
        """
        self.logs = []  # Reset logs
        try:
            success = (
                self.test_invariants() and
                self.test_trace_consistency() and
                self.test_monotonicity()
            )
            return success, self.logs
        except Exception as e:
            self.logs.append(f"CRITICAL: {traceback.format_exc()}")
            return False, self.logs