import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Inputs (with defaults if not present)
        self.m = getattr(solution_instance, 'm', 0.6)          # kg
        self.R = getattr(solution_instance, 'R', 0.3)          # m
        self.theta = getattr(solution_instance, 'theta', 60)   # degrees
        self.v_A = getattr(solution_instance, 'v_A', 4.0)      # m/s
        self.g = getattr(solution_instance, 'g', 10.0)         # m/s^2
        
        # Results (use .results dictionary if available, otherwise attributes)
        results = getattr(solution_instance, 'results', {})
        
        # Map expected keys to internal variables with safe extraction
        self.theta_A = self._safe_get(results, 'theta_A', self.theta)  # degrees
        self.v_x = self._safe_get(results, 'v_x', None)                # m/s
        self.v_y = self._safe_get(results, 'v_y', None)                # m/s (magnitude)
        self.t = self._safe_get(results, 't', None)                    # s
        self.x = self._safe_get(results, 'x', None)                    # m
        self.y = self._safe_get(results, 'y', None)                    # m
        self.v_C = self._safe_get(results, 'v_C', None)                # m/s (speed at C)
        self.x_prime = self._safe_get(results, 'x\'', None)            # m (if available)
        self.y_prime = self._safe_get(results, 'y\'', None)            # m (if available)
        self.t_prime = self._safe_get(results, 't\'', None)            # s (if available)
        self.v_C_prime = self._safe_get(results, 'v_C\'', None)        # m/s (required horizontal velocity)
        
        # Convert to float if possible (handle symbolic strings like "sqrt(7)")
        self._convert_to_float()
        
        # Store logs for reporting
        self.logs = []
    
    def _safe_get(self, results_dict, key, default):
        """Safely get a value from results dict or attribute."""
        val = results_dict.get(key, getattr(self.solution, key, default))
        if val is None:
            return default
        return val
    
    def _convert_to_float(self):
        """Convert symbolic strings to float if necessary."""
        # Helper to evaluate simple expressions
        def eval_str(s):
            if isinstance(s, (int, float)):
                return float(s)
            if isinstance(s, str):
                try:
                    # Handle common symbols
                    s = s.replace('sqrt', 'sp.sqrt')
                    # Use sympy to evaluate
                    return float(sp.sympify(s).evalf())
                except:
                    return None
            return None
        
        attrs = ['v_x', 'v_y', 't', 'x', 'y', 'v_C', 'x_prime', 'y_prime', 't_prime', 'v_C_prime']
        for attr in attrs:
            val = getattr(self, attr)
            if val is not None:
                num = eval_str(val)
                if num is not None:
                    setattr(self, attr, num)
                else:
                    setattr(self, attr, None)
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        logs = []
        try:
            # 1. Energy conservation from A to C
            h_AC = 1.5 * self.R  # height difference from A to C
            # Kinetic energy at A and C
            KE_A = 0.5 * self.m * self.v_A**2
            KE_C = 0.5 * self.m * self.v_C**2
            PE_diff = self.m * self.g * h_AC
            # Work-energy: ΔKE = -ΔPE  (since gravity does negative work)
            expected_KE_C = KE_A - PE_diff
            if not self._isclose(KE_C, expected_KE_C):
                logs.append(f"Energy conservation violation: KE_C = {KE_C}, expected {expected_KE_C}")
            else:
                logs.append("Energy conservation from A to C holds.")
            
            # 2. Newton's second law at C (if we had forces, but we can check centripetal condition)
            # At the top of the loop, the net force toward center is mg + N = m v_C^2 / R.
            # Since track is smooth, N >= 0 for contact. So v_C^2 >= g R.
            if self.v_C**2 < self.g * self.R - 1e-2:  # tolerance
                logs.append(f"Ball might lose contact at C: v_C^2 = {self.v_C**2}, gR = {self.g * self.R}")
            else:
                logs.append(f"Ball maintains contact at C (v_C^2 >= gR).")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Verify intermediate steps are consistent."""
        logs = []
        try:
            # Step 2: v_x = v_A * cos(60°)
            expected_vx = self.v_A * math.cos(math.radians(self.theta_A))
            if not self._isclose(self.v_x, expected_vx):
                logs.append(f"v_x mismatch: got {self.v_x}, expected {expected_vx}")
            else:
                logs.append(f"v_x correct: {self.v_x}")
            
            # Step 3: v_y magnitude = v_A * sin(60°)
            expected_vy_mag = self.v_A * math.sin(math.radians(self.theta_A))
            if not self._isclose(abs(self.v_y), expected_vy_mag):
                logs.append(f"v_y magnitude mismatch: got {self.v_y}, expected magnitude {expected_vy_mag}")
            else:
                logs.append(f"v_y magnitude correct: {self.v_y}")
            
            # Step 4: t = v_y / g (using magnitude, downward positive)
            if self.v_y is not None and self.v_y > 0:
                expected_t = self.v_y / self.g
                if not self._isclose(self.t, expected_t):
                    logs.append(f"Time t mismatch: got {self.t}, expected {expected_t}")
                else:
                    logs.append(f"Time t correct: {self.t}")
            
            # Step 5: x = v_x * t
            if self.v_x is not None and self.t is not None:
                expected_x = self.v_x * self.t
                if not self._isclose(self.x, expected_x):
                    logs.append(f"Horizontal displacement x mismatch: got {self.x}, expected {expected_x}")
                else:
                    logs.append(f"Horizontal displacement x correct: {self.x}")
            
            # Step 6: y = 1/2 * g * t^2
            if self.t is not None:
                expected_y = 0.5 * self.g * self.t**2
                if not self._isclose(self.y, expected_y):
                    logs.append(f"Vertical displacement y mismatch: got {self.y}, expected {expected_y}")
                else:
                    logs.append(f"Vertical displacement y correct: {self.y}")
            
            # Step 7: v_C from energy: v_C^2 = v_A^2 - 3 g R
            expected_vC_sq = self.v_A**2 - 3 * self.g * self.R
            if not self._isclose(self.v_C**2, expected_vC_sq):
                logs.append(f"v_C mismatch: got v_C^2 = {self.v_C**2}, expected {expected_vC_sq}")
            else:
                logs.append(f"v_C correct: {self.v_C}")
            
            # Steps 8-10: Projectile from C to A (if data available)
            if self.x_prime is not None and self.y_prime is not None:
                # x' = R sin(60°)
                expected_x_prime = self.R * math.sin(math.radians(self.theta_A))
                if not self._isclose(self.x_prime, expected_x_prime):
                    logs.append(f"x' mismatch: got {self.x_prime}, expected {expected_x_prime}")
                else:
                    logs.append(f"x' correct: {self.x_prime}")
                
                # y' = R + R cos(60°) = 1.5 R
                expected_y_prime = 1.5 * self.R
                if not self._isclose(self.y_prime, expected_y_prime):
                    logs.append(f"y' mismatch: got {self.y_prime}, expected {expected_y_prime}")
                else:
                    logs.append(f"y' correct: {self.y_prime}")
                
                # t' from free fall: y' = 1/2 g t'^2
                if self.y_prime is not None:
                    expected_t_prime = math.sqrt(2 * self.y_prime / self.g)
                    if self.t_prime is not None and not self._isclose(self.t_prime, expected_t_prime):
                        logs.append(f"t' mismatch: got {self.t_prime}, expected {expected_t_prime}")
                    elif self.t_prime is not None:
                        logs.append(f"t' correct: {self.t_prime}")
                
                # v_C' = x' / t'
                if self.x_prime is not None and self.t_prime is not None:
                    expected_vC_prime = self.x_prime / self.t_prime
                    if self.v_C_prime is not None and not self._isclose(self.v_C_prime, expected_vC_prime):
                        logs.append(f"v_C' mismatch: got {self.v_C_prime}, expected {expected_vC_prime}")
                    elif self.v_C_prime is not None:
                        logs.append(f"v_C' correct: {self.v_C_prime}")
                    
                    # Compare v_C (actual) with v_C' (required)
                    if self.v_C is not None and self.v_C_prime is not None:
                        if self._isclose(self.v_C, self.v_C_prime):
                            logs.append("v_C equals v_C' -> ball could hit A (but problem says unable).")
                        else:
                            logs.append(f"v_C ({self.v_C}) ≠ v_C' ({self.v_C_prime}) -> ball cannot hit A.")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb an input and check trend."""
        logs = []
        try:
            # Perturb v_A by +10%
            v_A_original = self.v_A
            self.v_A *= 1.10
            # Recalculate v_C using energy formula (independent of solution_instance)
            v_C_new = math.sqrt(max(0, self.v_A**2 - 3 * self.g * self.R))
            # Check that v_C increases with v_A (since derivative positive)
            if v_C_new > self.v_C:
                logs.append(f"Monotonicity: increasing v_A increases v_C (v_C went from {self.v_C:.3f} to {v_C_new:.3f}).")
            else:
                logs.append(f"Monotonicity warning: v_C did not increase as expected.")
            # Restore original v_A
            self.v_A = v_A_original
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        self.logs.append("=== PhysicsChecker started ===")
        
        # Run individual tests
        self.logs.append("\n--- test_invariants ---")
        self.logs.extend(self.test_invariants())
        
        self.logs.append("\n--- test_trace_consistency ---")
        self.logs.extend(self.test_trace_consistency())
        
        self.logs.append("\n--- test_monotonicity ---")
        self.logs.extend(self.test_monotonicity())
        
        # Determine overall success: no errors in logs? We'll define success as no critical errors.
        # For now, if any test produced an error message containing "Error" or "violation", consider failure.
        critical_errors = [log for log in self.logs if any(word in log.lower() for word in ['error', 'violation', 'mismatch', 'warning'])]
        success = len(critical_errors) == 0
        
        self.logs.append(f"\n=== PhysicsChecker finished: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)