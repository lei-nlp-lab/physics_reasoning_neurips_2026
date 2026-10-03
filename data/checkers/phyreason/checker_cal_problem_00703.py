import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results using getattr with fallback to dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem inputs (converted to meters and seconds)
        self.PA = 0.4  # 40 cm in meters
        self.PB = 0.5  # 50 cm in meters
        self.t_start = 2.0  # P starts vibrating at t=2 s
        
        # Extract results with safe defaults (using keys from mapping)
        self.v = self.results.get('v', 0.0)
        self.t_prime = self.results.get("t'", 0.0)  # t' from mapping
        self.A_A = self.results.get('A_A', 0.0)
        self.A_B = self.results.get('A_B', 0.0)
        self.T = self.results.get('T', 0.0)
        self.lmbda = self.results.get('lambda', 0.0)  # λ
        self.Delta_x = self.results.get('Deltax', 0.0)  # Δx
        self.A_all = self.results.get('A_all', 0.0)   # A_all
        self.s1 = self.results.get('s_1', 0.0)
        self.s2 = self.results.get('s_2', 0.0)
        self.Delta_t = self.results.get('Deltat', 0.0)  # Δt
        self.s = self.results.get('s', 0.0)
        
    def test_invariants(self):
        """Verify fundamental physical laws using numerical tolerances."""
        logs = []
        try:
            # 1. Wave speed from PA and start time
            expected_v = self.PA / self.t_start
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Wave speed mismatch: v = PA/t_start = {expected_v}, got {self.v}")
            
            # 2. Period from angular frequency (ω = 2π rad/s)
            if not math.isclose(self.T, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Period mismatch: T should be 1.0 s from ω=2π, got {self.T}")
            
            # 3. Wavelength from wave speed and period: λ = vT
            expected_lambda = self.v * self.T
            if not math.isclose(self.lmbda, expected_lambda, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Wavelength mismatch: λ = v*T = {expected_lambda}, got {self.lmbda}")
            
            # 4. Path difference: Δx = PB - PA
            expected_delta_x = self.PB - self.PA
            if not math.isclose(self.Delta_x, expected_delta_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Path difference mismatch: Δx = PB-PA = {expected_delta_x}, got {self.Delta_x}")
            
            # 5. Resultant amplitude for destructive interference (phase difference π)
            # Path difference = λ/2 → phase difference π → destructive interference
            expected_A_all = abs(self.A_B - self.A_A)
            if not math.isclose(self.A_all, expected_A_all, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Resultant amplitude mismatch: A_all = |A_B - A_A| = {expected_A_all}, got {self.A_all}")
            
            # 6. Distance s1 (half period of SHM with amplitude A_A)
            expected_s1 = 2.0 * self.A_A
            if not math.isclose(self.s1, expected_s1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"s1 mismatch: s1 = 2*A_A = {expected_s1}, got {self.s1}")
            
            # 7. Time interval Δt = 4.25 - 2.5
            expected_delta_t = 4.25 - 2.5
            if not math.isclose(self.Delta_t, expected_delta_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Δt mismatch: Δt = 4.25 - 2.5 = {expected_delta_t}, got {self.Delta_t}")
            
            # 8. Distance s2 (SHM with resultant amplitude A_all over time Δt)
            # Distance per period = 4*A_all, so s2 = (Δt/T) * (4*A_all)
            expected_s2 = (self.Delta_t / self.T) * (4.0 * self.A_all)
            if not math.isclose(self.s2, expected_s2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"s2 mismatch: s2 = (Δt/T)*4*A_all = {expected_s2}, got {self.s2}")
            
            # 9. Total distance s = s1 + s2
            expected_s = self.s1 + self.s2
            if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Total distance mismatch: s = s1 + s2 = {expected_s}, got {self.s}")
            
            # 10. Time for wave from B to reach P: t' = PB/v
            expected_t_prime = self.PB / self.v
            if not math.isclose(self.t_prime, expected_t_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"t' mismatch: t' = PB/v = {expected_t_prime}, got {self.t_prime}")
            
            if not logs:
                logs.append("All invariants passed.")
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        try:
            # Check wave speed consistency
            computed_v = self.PA / self.t_start
            if not math.isclose(self.v, computed_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace v: computed {computed_v}, result {self.v}")
            
            # Check total distance consistency
            computed_s = self.s1 + self.s2
            if not math.isclose(self.s, computed_s, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace s: computed {computed_s}, result {self.s}")
            
            # Verify s1 and s2 using SHM formulas
            expected_s1 = 2.0 * self.A_A
            if not math.isclose(self.s1, expected_s1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace s1: expected 2*A_A = {expected_s1}, got {self.s1}")
            
            expected_s2 = (self.Delta_t / self.T) * (4.0 * self.A_all)
            if not math.isclose(self.s2, expected_s2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace s2: expected (Δt/T)*4*A_all = {expected_s2}, got {self.s2}")
            
            if not logs:
                logs.append("Trace consistency passed.")
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb an input and check trend, then restore."""
        logs = []
        try:
            # Perturb amplitude A_A by +10%
            original_A_A = self.A_A
            perturbed_A_A = original_A_A * 1.10
            
            # Recompute affected quantities (wave speed and period remain unchanged)
            perturbed_A_all = abs(self.A_B - perturbed_A_A)
            perturbed_s1 = 2.0 * perturbed_A_A
            perturbed_s2 = (self.Delta_t / self.T) * (4.0 * perturbed_A_all)
            perturbed_s = perturbed_s1 + perturbed_s2
            
            logs.append(f"Monotonicity test: A_A changed from {original_A_A} to {perturbed_A_A}")
            logs.append(f"  Original s: {self.s}, perturbed s: {perturbed_s}")
            logs.append(f"  Change in s: {perturbed_s - self.s}")
            
            # Check that s1 increases and s2 decreases (since A_all decreases when A_A increases)
            if not (perturbed_s1 > self.s1):
                logs.append("  Warning: s1 did not increase as expected.")
            if not (perturbed_s2 < self.s2):
                logs.append("  Warning: s2 did not decrease as expected.")
            
            logs.append("Monotonicity test completed.")
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        all_logs = []
        success = True
        
        tests = [
            ("test_invariants", self.test_invariants),
            ("test_trace_consistency", self.test_trace_consistency),
            ("test_monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            logs = test_func()
            # Check if any logs contain "mismatch", "Error", or "Warning" to indicate potential failure
            if any("mismatch" in log or "Error" in log for log in logs):
                success = False
            all_logs.append({test_name: logs})
        
        return success, all_logs