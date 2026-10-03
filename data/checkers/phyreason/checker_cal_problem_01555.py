import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        """
        try:
            # Input parameters (safe access)
            self.m = getattr(solution_instance, 'm', 2e3)  # mass in kg
            self.P = getattr(solution_instance, 'P', 80e3)  # power in W
            self.g = getattr(solution_instance, 'g', 10.0)  # gravity m/s²
            self.mu = getattr(solution_instance, 'mu', 0.1)  # resistance coefficient
            
            # Results from solution (safe access via results dictionary)
            results = getattr(solution_instance, 'results', {})
            self.F = results.get('F', 0.0)   # traction force at v=10 m/s
            self.S = results.get('S', 0.0)   # displacement to reach v_max
            self.a = results.get('a', 0.0)   # acceleration at v=10 m/s
            self.F_f = results.get('F_f', 0.0)  # resistance force
            self.v_m = results.get('v_m', 0.0)  # maximum speed
            
            # Derived intermediate values (for cross-checks)
            self.v_given = 10.0  # speed at which traction force is evaluated
            self.t = None  # time to reach v_max (not directly given)
            
        except Exception as e:
            self.error_msg = f"Initialization error: {e}"
            raise

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Newton's 2nd Law at v=10 m/s.
        2. Resistance force proportional to weight.
        3. Maximum speed condition (net force zero).
        4. Work-energy theorem for displacement S.
        """
        logs = []
        try:
            # 1. Newton's 2nd Law: F_net = m*a
            F_net_calc = self.F - self.F_f
            F_net_expected = self.m * self.a
            if not math.isclose(F_net_calc, F_net_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton 2nd Law violation: F_net = {F_net_calc} vs m*a = {F_net_expected}")
            
            # 2. Resistance force: F_f = μ * m * g
            F_f_expected = self.mu * self.m * self.g
            if not math.isclose(self.F_f, F_f_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Resistance force mismatch: F_f = {self.F_f} vs μmg = {F_f_expected}")
            
            # 3. Maximum speed condition: Traction force = Resistance force => v_m = P / F_f
            v_m_expected = self.P / self.F_f
            if not math.isclose(self.v_m, v_m_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Max speed mismatch: v_m = {self.v_m} vs P/F_f = {v_m_expected}")
            
            # 4. Work-energy theorem: P*t - F_f*S = ΔKE = 0.5*m*v_m^2 (assuming v0=0)
            # But time t is not given. Use kinematic relation with average velocity approximation.
            # From steps analysis: t = S / v_avg, where v_avg = (0 + v_m)/2 = v_m/2
            if self.v_m > 0:
                v_avg = self.v_m / 2
                t_est = self.S / v_avg
                left_side = self.P * t_est - self.F_f * self.S
                delta_KE = 0.5 * self.m * self.v_m**2
                if not math.isclose(left_side, delta_KE, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Work-energy mismatch: P*t - F_f*S = {left_side} vs ΔKE = {delta_KE}")
            else:
                logs.append("Max speed is zero, cannot check work-energy.")
            
            return len(logs) == 0, logs
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
            return False, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        try:
            # Check traction force at v=10 m/s: F = P / v
            F_expected = self.P / self.v_given
            if not math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Traction force mismatch: F = {self.F} vs P/v = {F_expected}")
            
            # Check acceleration from Newton's 2nd: a = (F - F_f) / m
            a_expected = (self.F - self.F_f) / self.m
            if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Acceleration mismatch: a = {self.a} vs (F-F_f)/m = {a_expected}")
            
            # Check displacement S using work-energy with average velocity approximation
            if self.v_m > 0 and self.F_f > 0:
                v_avg = self.v_m / 2
                t_est = self.S / v_avg
                S_from_energy = (0.5 * self.m * self.v_m**2) / (self.P / v_avg - self.F_f)
                if not math.isclose(self.S, S_from_energy, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Displacement mismatch: S = {self.S} vs work-energy estimate = {S_from_energy}")
            
            return len(logs) == 0, logs
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False, logs

    def test_monotonicity(self):
        """
        Perturb mass m and check expected trends (then restore original).
        """
        logs = []
        original_m = self.m
        try:
            # Increase mass by 10%
            self.m *= 1.1
            new_F_f = self.mu * self.m * self.g
            new_v_m = self.P / new_F_f
            new_a = (self.F - new_F_f) / self.m  # assuming same traction force F
            
            # Check trends
            if not (new_F_f > self.F_f):
                logs.append(f"Monotonicity fail: F_f should increase with mass, but {new_F_f} <= {self.F_f}")
            if not (new_v_m < self.v_m):
                logs.append(f"Monotonicity fail: v_m should decrease with mass, but {new_v_m} >= {self.v_m}")
            if not (new_a < self.a):
                logs.append(f"Monotonicity fail: a should decrease with mass, but {new_a} >= {self.a}")
            
            # Restore original mass
            self.m = original_m
            return len(logs) == 0, logs
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            self.m = original_m  # restore on error
            return False, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                if not success:
                    overall_success = False
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
            except Exception as e:
                overall_success = False
                all_logs.append(f"=== {test_name} ===")
                all_logs.append(f"Test crashed: {traceback.format_exc()}")
        
        return overall_success, all_logs