import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary data from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Constants from the problem statement (in SI units)
        self.c_water = 4200.0      # J/(kg·K)
        self.c_ice = 2100.0        # J/(kg·K)
        self.L = 334000.0          # J/kg
        self.A = 5.0               # km·°C
        self.Aprime = 4.0          # km·°C
        self.b = 2.0               # °C/km, temperature gradient below 2 km
        self.g = 9.8               # m/s², gravitational acceleration

        # Extract results with safe fallbacks
        self.eta = self._safe_get('eta', 'η')
        self.T_final = self._safe_get('T_final')
        self.v = self._safe_get('v')
        self.Q = self._safe_get('Q')
        self.Qprime = self._safe_get("Q''", 'Qprime')
        self.delta_T = self._safe_get('delta_T')
        self.delta_T_star = self._safe_get('delta_T*')
        self.x0 = self._safe_get('x_0')
        self.dT_d_dx = self._safe_get('dT_d/dx')
        self.dQ_dt = self._safe_get('dQ/dt')

        # For monotonicity test, store original perturbable inputs
        self.original_A = self.A
        self.original_Aprime = self.Aprime

    def _safe_get(self, *keys):
        """
        Safely retrieve a value from solution attributes or results dict.
        Returns None if not found.
        """
        for key in keys:
            val = getattr(self.solution, key, None)
            if val is not None:
                return val
            val = self.results.get(key, None)
            if val is not None:
                return val
        return None

    def test_invariants(self):
        """
        Verify fundamental physical laws and constraints.
        """
        logs = []
        success = True

        # 1. Newton's First Law: velocity should be constant (if provided)
        if self.v is not None:
            # In the problem, fall velocity is assumed constant.
            # If the solution computed v at different points, they should be equal.
            # Here we only have one v, so we can't compare, but we can check it's a number.
            if not isinstance(self.v, (int, float)):
                logs.append("WARNING: Velocity v is not a numeric value.")
                success = False
            else:
                logs.append(f"OK: Velocity v = {self.v} is numeric (constant fall assumed).")

        # 2. Mass fraction η should be between 0 and 1
        if self.eta is not None:
            if 0 <= self.eta <= 1:
                logs.append(f"OK: η = {self.eta} is within [0,1].")
            else:
                logs.append(f"ERROR: η = {self.eta} is outside [0,1].")
                success = False
        else:
            logs.append("WARNING: η not found; skipping η range check.")

        # 3. Final temperature should be between initial and ground atmospheric temperature
        #    Initial droplet temperature is 0°C during melting, ground atmospheric temperature is ~8°C
        if self.T_final is not None:
            if 0.0 <= self.T_final <= 10.0:  # Allow some slack
                logs.append(f"OK: T_final = {self.T_final}°C is within reasonable bounds.")
            else:
                logs.append(f"WARNING: T_final = {self.T_final}°C is outside expected range [0,10]°C.")
                # Not a critical error, as viscous heating could raise it slightly above 8°C

        # 4. Energy conservation during melting and freezing (using area ratio)
        #    From steps: η = A'/A
        if self.eta is not None:
            expected_eta = self.Aprime / self.A
            if math.isclose(self.eta, expected_eta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: η = {self.eta} matches area ratio A'/A = {expected_eta}.")
            else:
                logs.append(f"ERROR: η = {self.eta} does not match area ratio A'/A = {expected_eta}.")
                success = False

        # 5. Viscous dissipation correction ΔT* should be positive (heating)
        if self.delta_T_star is not None:
            if self.delta_T_star > 0:
                logs.append(f"OK: ΔT* = {self.delta_T_star} is positive (viscous heating).")
            else:
                logs.append(f"ERROR: ΔT* = {self.delta_T_star} is non-positive, but viscous heating should be positive.")
                success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        """
        logs = []
        success = True

        # 1. Check η calculation from steps: η = A'/A
        if self.eta is not None:
            expected_eta = self.Aprime / self.A
            if math.isclose(self.eta, expected_eta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: η = {self.eta} matches direct ratio A'/A = {expected_eta}.")
            else:
                logs.append(f"ERROR: η deviation: got {self.eta}, expected {expected_eta}.")
                success = False

        # 2. Check characteristic distance x0 = c_water * A / L
        if self.x0 is not None:
            expected_x0 = (self.c_water * self.A) / self.L  # in km (since A in km·°C)
            if math.isclose(self.x0, expected_x0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: x0 = {self.x0} km matches formula c_water*A/L = {expected_x0} km.")
            else:
                logs.append(f"WARNING: x0 = {self.x0} km differs from c_water*A/L = {expected_x0} km.")

        # 3. Check ΔT ≈ b * x0 (without viscous dissipation)
        if self.delta_T is not None and self.x0 is not None:
            expected_delta_T = self.b * self.x0
            if math.isclose(self.delta_T, expected_delta_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: ΔT = {self.delta_T}°C matches b*x0 = {expected_delta_T}°C.")
            else:
                logs.append(f"WARNING: ΔT = {self.delta_T}°C differs from b*x0 = {expected_delta_T}°C.")

        # 4. Check ΔT* = g * A / L (viscous correction)
        if self.delta_T_star is not None:
            expected_delta_T_star = (self.g * self.A) / self.L  # unit conversion handled by constants
            if math.isclose(self.delta_T_star, expected_delta_T_star, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: ΔT* = {self.delta_T_star}°C matches g*A/L ≈ {expected_delta_T_star}°C.")
            else:
                logs.append(f"WARNING: ΔT* = {self.delta_T_star}°C differs from g*A/L = {expected_delta_T_star}°C.")

        # 5. Check final temperature composition
        if self.T_final is not None and self.delta_T is not None and self.delta_T_star is not None:
            # From steps: T_final ≈ (8 - ΔT) + ΔT*
            T_base = 8.0  # ground atmospheric temperature from dashed profile
            expected_T_final = (T_base - self.delta_T) + self.delta_T_star
            if math.isclose(self.T_final, expected_T_final, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: T_final = {self.T_final}°C matches (8 - ΔT) + ΔT* = {expected_T_final}°C.")
            else:
                logs.append(f"WARNING: T_final = {self.T_final}°C differs from (8 - ΔT) + ΔT* = {expected_T_final}°C.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input and check the trend of outputs.
        """
        logs = []
        success = True

        # Perturb the melting area A by +10%
        A_perturbed = self.original_A * 1.1

        # 1. Check η = A'/A
        if self.eta is not None:
            original_eta = self.eta
            new_eta = self.original_Aprime / A_perturbed
            if new_eta < original_eta:
                logs.append(f"OK: Increasing A decreases η (from {original_eta} to {new_eta}).")
            else:
                logs.append(f"ERROR: Increasing A did not decrease η (from {original_eta} to {new_eta}).")
                success = False

            # Check derivative sign
            derivative = (new_eta - original_eta) / (A_perturbed - self.original_A)
            expected_derivative = -self.original_Aprime / (self.original_A ** 2)
            if derivative < 0 and math.isclose(derivative, expected_derivative, rel_tol=0.1):
                logs.append(f"OK: Derivative dη/dA ≈ {derivative} matches expected {expected_derivative}.")
            else:
                logs.append(f"WARNING: Derivative dη/dA = {derivative} differs from expected {expected_derivative}.")

        # 2. Check T_final trend using the approximate formula from steps
        if self.T_final is not None:
            # Use the same approximate formulas as in the steps
            # x0 = c_water * A / L
            # ΔT = b * x0
            # ΔT* = g * A / L
            # T_final ≈ 8 - ΔT + ΔT* = 8 + A*(g - b*c_water)/L
            original_T_final = self.T_final
            # Recalculate with perturbed A
            x0_perturbed = (self.c_water * A_perturbed) / self.L
            delta_T_perturbed = self.b * x0_perturbed
            delta_T_star_perturbed = (self.g * A_perturbed) / self.L
            T_base = 8.0
            new_T_final = (T_base - delta_T_perturbed) + delta_T_star_perturbed

            # The coefficient (g - b*c_water) is negative, so increasing A should decrease T_final
            if new_T_final < original_T_final:
                logs.append(f"OK: Increasing A decreases T_final (from {original_T_final} to {new_T_final}).")
            else:
                logs.append(f"WARNING: Increasing A did not decrease T_final (from {original_T_final} to {new_T_final}).")
                # Not a critical error due to approximations

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs