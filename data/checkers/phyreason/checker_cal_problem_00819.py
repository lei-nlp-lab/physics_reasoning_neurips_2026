import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution instance.
        Use getattr with fallbacks to avoid AttributeError in sandbox.
        """
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters from problem context
        self.x1 = 0.0  # m
        self.x2 = 14.0  # m
        self.xP = 4.0  # m
        self.A1 = 3.0  # cm (amplitude of S1)
        self.A2 = 5.0  # cm (amplitude of S2)
        self.omega = 5 * math.pi  # rad/s (angular frequency)
        
        # Extract results with safe fallbacks
        self.s_P = self.results.get('s_P', 0.0)  # total distance (cm)
        self.A_P = self.results.get('A_P', 0.0)  # amplitude during interference (cm)
        self.delta_t_1 = self.results.get('delta_t_1', 0.0)  # s
        self.t_2 = self.results.get('t_2', 0.0)  # s
        self.T = self.results.get('T', 0.0)  # period (s)
        self.s_1 = self.results.get('s_1', 0.0)  # cm
        self.s_2 = self.results.get('s_2', 0.0)  # cm
        self.delta_t_2 = self.results.get('delta_t_2', 0.0)  # s
        self.v = self.results.get('v', 0.0)  # wave speed (m/s)
        self.lam = self.results.get('λ', 0.0)  # wavelength (m)
        
        # Target answers for verification
        self.target_wavelength = 4.0  # m
        self.target_s_P = 28.0  # cm
        
        # Tolerance for numerical comparisons (2% relative)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for tracking test results
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "✓" if success else "✗"
            self.logs.append(f"{status} {message}")
        else:
            self.logs.append(f"  {message}")
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants from KB theorems.
        """
        try:
            # 1. Period from angular frequency: T = 2π/ω
            T_calc = 2 * math.pi / self.omega
            T_ok = math.isclose(self.T, T_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"Period check: T = {self.T:.3f}, expected {T_calc:.3f}", T_ok)
            
            # 2. Wave speed from distance/time: v = xP / 0.4 (since t=0.4s when wave from S1 reaches P)
            v_calc = self.xP / 0.4
            v_ok = math.isclose(self.v, v_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"Wave speed: v = {self.v:.3f} m/s, expected {v_calc:.3f} m/s", v_ok)
            
            # 3. Wavelength from wave speed and period: λ = v * T
            lam_calc = self.v * self.T
            lam_ok = math.isclose(self.lam, lam_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"Wavelength: λ = {self.lam:.3f} m, expected {lam_calc:.3f} m", lam_ok)
            
            # 4. Time for wave from S2 to reach P: t_2 = (x2 - xP) / v
            t2_calc = (self.x2 - self.xP) / self.v
            t2_ok = math.isclose(self.t_2, t2_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"Time t_2: {self.t_2:.3f} s, expected {t2_calc:.3f} s", t2_ok)
            
            # 5. Amplitude during interference: A_P = |A1 - A2|
            A_P_calc = abs(self.A1 - self.A2)
            A_P_ok = math.isclose(self.A_P, A_P_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"Amplitude A_P: {self.A_P:.3f} cm, expected {A_P_calc:.3f} cm", A_P_ok)
            
            # 6. Check destructive interference condition (path difference = λ/2 * odd integer)
            path_diff = abs((self.xP - self.x1) - (self.x2 - self.xP))  # = 10 - 4 = 6 m
            # For destructive interference: path_diff = (n + 1/2) * λ
            # Check if path_diff is approximately an odd multiple of λ/2
            n_half = (2 * path_diff) / self.lam - 1  # Solve for n in path_diff = (n + 0.5)*λ
            n_half_ok = math.isclose(n_half % 2, 1.0, rel_tol=0.1) or math.isclose(n_half % 2, 0.0, rel_tol=0.1)
            self._log(f"Destructive interference: path_diff={path_diff:.3f} m, λ={self.lam:.3f} m, n≈{n_half:.1f}", n_half_ok)
            
            return all([T_ok, v_ok, lam_ok, t2_ok, A_P_ok])
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        """
        try:
            # 1. Check Δt_1 = t_2 - 0.4
            delta_t_1_calc = self.t_2 - 0.4
            delta_t_1_ok = math.isclose(self.delta_t_1, delta_t_1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"Δt_1: {self.delta_t_1:.3f} s, expected {delta_t_1_calc:.3f} s", delta_t_1_ok)
            
            # 2. Check s_1 = 6 * A1
            s_1_calc = 6 * self.A1
            s_1_ok = math.isclose(self.s_1, s_1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"s_1: {self.s_1:.3f} cm, expected {s_1_calc:.3f} cm", s_1_ok)
            
            # 3. Check Δt_2 = 0.5 s (from problem context)
            delta_t_2_ok = math.isclose(self.delta_t_2, 0.5, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"Δt_2: {self.delta_t_2:.3f} s, expected 0.500 s", delta_t_2_ok)
            
            # 4. Check s_2 = 5 * A_P
            s_2_calc = 5 * self.A_P
            s_2_ok = math.isclose(self.s_2, s_2_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"s_2: {self.s_2:.3f} cm, expected {s_2_calc:.3f} cm", s_2_ok)
            
            # 5. Check total distance: s_P = s_1 + s_2
            s_P_calc = self.s_1 + self.s_2
            s_P_ok = math.isclose(self.s_P, s_P_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"s_P: {self.s_P:.3f} cm, expected {s_P_calc:.3f} cm", s_P_ok)
            
            # 6. Check against target answers
            target_lam_ok = math.isclose(self.lam, self.target_wavelength, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"λ matches target: {self.lam:.3f} m ≈ {self.target_wavelength} m", target_lam_ok)
            
            target_s_P_ok = math.isclose(self.s_P, self.target_s_P, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            self._log(f"s_P matches target: {self.s_P:.3f} cm ≈ {self.target_s_P} cm", target_s_P_ok)
            
            return all([delta_t_1_ok, s_1_ok, delta_t_2_ok, s_2_ok, s_P_ok, target_lam_ok, target_s_P_ok])
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and verify expected trend in output.
        """
        try:
            # Perturb amplitude A1 from 3 cm to 4 cm
            A1_original = self.A1
            self.A1 = 4.0
            
            # Recalculate expected distances using same formulas
            A_P_new = abs(self.A1 - self.A2)  # |4 - 5| = 1 cm
            s_1_new = 6 * self.A1  # 6 * 4 = 24 cm
            s_2_new = 5 * A_P_new  # 5 * 1 = 5 cm
            s_P_new = s_1_new + s_2_new  # 24 + 5 = 29 cm
            
            # Restore original value
            self.A1 = A1_original
            
            # Check monotonicity: increasing A1 should increase s_P (since A1 < A2)
            if A1_original < self.A2:
                # When A1 < A2, s_P = A1 + 5*A2 = A1 + 25, so derivative is +1
                monotonic_ok = (s_P_new > self.s_P)
                self._log(f"Monotonicity: A1↑ from 3 to 4 → s_P↑ from {self.s_P:.1f} to {s_P_new:.1f} cm", monotonic_ok)
            else:
                # When A1 > A2, s_P = 11*A1 - 5*A2, derivative is +11
                monotonic_ok = True  # trivially true
                self._log(f"Monotonicity: A1 > A2 case, derivative positive", True)
            
            return monotonic_ok
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """
        Run all tests and return success flag with logs.
        """
        self.logs = []
        self.logs.append("=== PhysicsChecker started ===")
        
        # Run tests
        invariants_ok = self.test_invariants()
        self.logs.append("")
        
        trace_ok = self.test_trace_consistency()
        self.logs.append("")
        
        monotonic_ok = self.test_monotonicity()
        
        # Summary
        all_ok = invariants_ok and trace_ok and monotonic_ok
        self.logs.append(f"\n=== Summary: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ===")
        
        return all_ok, self.logs

# Example usage (commented out):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 's_P': 28.0,
#                 'A_P': 2.0,
#                 'delta_t_1': 0.6,
#                 't_2': 1.0,
#                 'T': 0.4,
#                 's_1': 18.0,
#                 's_2': 10.0,
#                 'delta_t_2': 0.5,
#                 'v': 10.0,
#                 'λ': 4.0
#             }
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)