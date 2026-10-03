import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Safely extract all expected parameters and results
        self.R1 = self._get_value('R1')
        self.R2 = self._get_value('R2')
        self.alpha = self._get_value('alpha')
        self.Tc = self._get_value('Tc')
        self.T0 = self._get_value('T0')
        self.L = self._get_value('L')
        self.V = self._get_value('V')
        self.I1 = self._get_value('I1') or self._get_value('I_1')
        self.I2 = self._get_value('I2') or self._get_value('I_2')
        self.T_max = self._get_value('T_max')
        self.T_min = self._get_value('T_min')
        self.period = self._get_value('t_1+t_2') or self._get_value('period')
        self.V1 = self._get_value('V_1')
        self.V2 = self._get_value('V_2')
        self.ratio = self._get_value('fracT_textmax-T_0T_textmin-T_0')

    def _get_value(self, key):
        # Safe attribute access: try solution attribute, then results dict
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, None)
        return val

    def _is_close(self, a, b):
        # Fuzzy numerical matching with 1% tolerance
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_critical_voltages(self):
        logs = []
        success = True

        # Check V1 = sqrt(R1 * alpha * (Tc - T0))
        if None not in (self.R1, self.alpha, self.Tc, self.T0, self.V1):
            expected_V1 = math.sqrt(self.R1 * self.alpha * (self.Tc - self.T0))
            if self._is_close(self.V1, expected_V1):
                logs.append(f"V1 correct: {self.V1} ≈ {expected_V1}")
            else:
                logs.append(f"V1 mismatch: expected {expected_V1}, got {self.V1}")
                success = False
        else:
            logs.append("Skipped V1 check (missing parameters or result)")

        # Check V2 = sqrt(R2 * alpha * (Tc - T0))
        if None not in (self.R2, self.alpha, self.Tc, self.T0, self.V2):
            expected_V2 = math.sqrt(self.R2 * self.alpha * (self.Tc - self.T0))
            if self._is_close(self.V2, expected_V2):
                logs.append(f"V2 correct: {self.V2} ≈ {expected_V2}")
            else:
                logs.append(f"V2 mismatch: expected {expected_V2}, got {self.V2}")
                success = False
        else:
            logs.append("Skipped V2 check (missing parameters or result)")

        return success, logs

    def test_temperature_ratio(self):
        logs = []
        success = True

        # Check (T_max - T0) / (T_min - T0) = (R2 / R1)^2
        if None not in (self.T_max, self.T_min, self.T0, self.R1, self.R2):
            numerator = self.T_max - self.T0
            denominator = self.T_min - self.T0
            if abs(denominator) > 1e-12:
                actual_ratio = numerator / denominator
                expected_ratio = (self.R2 / self.R1) ** 2
                if self._is_close(actual_ratio, expected_ratio):
                    logs.append(f"Temperature ratio correct: {actual_ratio} ≈ {expected_ratio}")
                else:
                    logs.append(f"Temperature ratio mismatch: expected {expected_ratio}, got {actual_ratio}")
                    success = False
            else:
                logs.append("Denominator (T_min - T0) too close to zero, skipping ratio check")
                success = False
        else:
            logs.append("Skipped temperature ratio check (missing parameters)")

        return success, logs

    def test_oscillation_period(self):
        logs = []
        success = True

        # Check exact period formula (if all parameters available)
        if None not in (self.I1, self.I2, self.V, self.R1, self.R2, self.L, self.period):
            # Compute term1 = (L/R1) * ln((I2 - V/R1)/(I1 - V/R1))
            term1_arg_num = self.I2 - self.V / self.R1
            term1_arg_den = self.I1 - self.V / self.R1
            if abs(term1_arg_den) > 1e-12 and term1_arg_num / term1_arg_den > 0:
                term1 = (self.L / self.R1) * math.log(term1_arg_num / term1_arg_den)
            else:
                logs.append("Term1 argument invalid, skipping exact period check")
                return False, logs

            # Compute term2 = (L/R2) * ln((I1 - V/R2)/(I2 - V/R2))
            term2_arg_num = self.I1 - self.V / self.R2
            term2_arg_den = self.I2 - self.V / self.R2
            if abs(term2_arg_den) > 1e-12 and term2_arg_num / term2_arg_den > 0:
                term2 = (self.L / self.R2) * math.log(term2_arg_num / term2_arg_den)
            else:
                logs.append("Term2 argument invalid, skipping exact period check")
                return False, logs

            expected_period = term1 + term2
            if self._is_close(self.period, expected_period):
                logs.append(f"Exact period correct: {self.period} ≈ {expected_period}")
            else:
                logs.append(f"Exact period mismatch: expected {expected_period}, got {self.period}")
                success = False
        else:
            logs.append("Skipped exact period check (missing parameters)")

        # Check approximate period: 0.68 * L / R1
        if None not in (self.L, self.R1, self.period):
            approx_period = 0.68 * self.L / self.R1
            if self._is_close(self.period, approx_period):
                logs.append(f"Approximate period correct: {self.period} ≈ {approx_period}")
            else:
                logs.append(f"Approximate period mismatch: expected {approx_period}, got {self.period}")
                # This is an approximation, so don't fail the test entirely
        else:
            logs.append("Skipped approximate period check (missing parameters)")

        return success, logs

    def test_currents_relation(self):
        logs = []
        success = True

        # Check I2 = I1/4 when R2 = 16 R1
        if None not in (self.R1, self.R2, self.I1, self.I2):
            if self._is_close(self.R2, 16 * self.R1):
                expected_I2 = self.I1 / 4
                if self._is_close(self.I2, expected_I2):
                    logs.append(f"Currents relation correct: I2 = {self.I2} ≈ I1/4 = {expected_I2}")
                else:
                    logs.append(f"Currents relation mismatch: expected I2 = {expected_I2}, got {self.I2}")
                    success = False
            else:
                logs.append("R2 ≠ 16 R1, skipping currents relation check")
        else:
            logs.append("Skipped currents relation check (missing parameters)")

        return success, logs

    def test_thermal_equilibrium(self):
        logs = []
        success = True

        # At critical temperature, electrical power = heat flux
        # For R1: V1^2 / R1 = alpha * (Tc - T0)
        if None not in (self.V1, self.R1, self.alpha, self.Tc, self.T0):
            power_el = self.V1 ** 2 / self.R1
            heat_flux = self.alpha * (self.Tc - self.T0)
            if self._is_close(power_el, heat_flux):
                logs.append(f"Thermal equilibrium at R1: electrical power {power_el} ≈ heat flux {heat_flux}")
            else:
                logs.append(f"Thermal equilibrium mismatch at R1: power {power_el}, heat flux {heat_flux}")
                success = False
        else:
            logs.append("Skipped thermal equilibrium at R1 (missing parameters)")

        # For R2: V2^2 / R2 = alpha * (Tc - T0)
        if None not in (self.V2, self.R2, self.alpha, self.Tc, self.T0):
            power_el = self.V2 ** 2 / self.R2
            heat_flux = self.alpha * (self.Tc - self.T0)
            if self._is_close(power_el, heat_flux):
                logs.append(f"Thermal equilibrium at R2: electrical power {power_el} ≈ heat flux {heat_flux}")
            else:
                logs.append(f"Thermal equilibrium mismatch at R2: power {power_el}, heat flux {heat_flux}")
                success = False
        else:
            logs.append("Skipped thermal equilibrium at R2 (missing parameters)")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Check I1 = sqrt(alpha*(Tc-T0)/R1)
        if None not in (self.I1, self.alpha, self.Tc, self.T0, self.R1):
            expected_I1 = math.sqrt(self.alpha * (self.Tc - self.T0) / self.R1)
            if self._is_close(self.I1, expected_I1):
                logs.append(f"I1 consistent with formula: {self.I1} ≈ sqrt(alpha*(Tc-T0)/R1) = {expected_I1}")
            else:
                logs.append(f"I1 inconsistency: expected {expected_I1}, got {self.I1}")
                success = False
        else:
            logs.append("Skipped I1 consistency check (missing parameters)")

        # Check I2 = sqrt(alpha*(Tc-T0)/R2)
        if None not in (self.I2, self.alpha, self.Tc, self.T0, self.R2):
            expected_I2 = math.sqrt(self.alpha * (self.Tc - self.T0) / self.R2)
            if self._is_close(self.I2, expected_I2):
                logs.append(f"I2 consistent with formula: {self.I2} ≈ sqrt(alpha*(Tc-T0)/R2) = {expected_I2}")
            else:
                logs.append(f"I2 inconsistency: expected {expected_I2}, got {self.I2}")
                success = False
        else:
            logs.append("Skipped I2 consistency check (missing parameters)")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb R1 and observe effect on period (should decrease as R1 increases)
        if None not in (self.R1, self.L, self.V, self.alpha, self.Tc, self.T0, self.period):
            # Store original values
            R1_original = self.R1
            # Perturb R1 by +10%
            R1_perturbed = R1_original * 1.10
            # Keep R2 = 16 R1 (as per problem condition)
            R2_perturbed = 16 * R1_perturbed

            # Recompute I1 and I2 for perturbed values
            I1_perturbed = math.sqrt(self.alpha * (self.Tc - self.T0) / R1_perturbed)
            I2_perturbed = I1_perturbed / 4  # since R2 = 16 R1

            # Compute perturbed period using exact formula
            term1_arg_num = I2_perturbed - self.V / R1_perturbed
            term1_arg_den = I1_perturbed - self.V / R1_perturbed
            term2_arg_num = I1_perturbed - self.V / R2_perturbed
            term2_arg_den = I2_perturbed - self.V / R2_perturbed

            if (abs(term1_arg_den) > 1e-12 and term1_arg_num / term1_arg_den > 0 and
                abs(term2_arg_den) > 1e-12 and term2_arg_num / term2_arg_den > 0):
                term1 = (self.L / R1_perturbed) * math.log(term1_arg_num / term1_arg_den)
                term2 = (self.L / R2_perturbed) * math.log(term2_arg_num / term2_arg_den)
                period_perturbed = term1 + term2

                # Check monotonicity: period should decrease when R1 increases
                if period_perturbed < self.period:
                    logs.append(f"Monotonicity holds: period decreases from {self.period} to {period_perturbed} when R1 increases from {R1_original} to {R1_perturbed}")
                else:
                    logs.append(f"Monotonicity violation: period did not decrease when R1 increased (original {self.period}, perturbed {period_perturbed})")
                    success = False
            else:
                logs.append("Invalid arguments in perturbed period calculation, skipping monotonicity check")
        else:
            logs.append("Skipped monotonicity test (missing parameters)")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            self.test_critical_voltages,
            self.test_temperature_ratio,
            self.test_oscillation_period,
            self.test_currents_relation,
            self.test_thermal_equilibrium,
            self.test_trace_consistency,
            self.test_monotonicity
        ]

        for test in tests:
            try:
                test_success, test_logs = test()
                all_logs.append(f"=== {test.__name__} ===")
                all_logs.extend(test_logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Test {test.__name__} failed with exception: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs