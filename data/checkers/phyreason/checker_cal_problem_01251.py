import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        self.solution = solution_instance
        # Safe extraction of inputs with defaults
        self.m = getattr(solution_instance, 'm', 2.0)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.F = getattr(solution_instance, 'F', 20.0)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safe extraction of results with defaults
        results = getattr(solution_instance, 'results', {})
        self.a = results.get('a', 0.0)
        self.s1 = results.get('s_1', 0.0)
        self.v1 = results.get('v_1', 0.0)
        self.s2 = results.get('s_2', 0.0)
        self.s = results.get('s', 0.0)
        
        # Trigonometric values (sin37=0.6, cos37=0.8)
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        # Store logs for all tests
        self.logs = []
    
    def _log(self, message, is_error=False):
        """Helper to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
        if is_error:
            print(f"ERROR: {message}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical invariants using KB theorems."""
        errors = []
        
        # Convert to float for safety
        m = float(self.m)
        mu = float(self.mu)
        F = float(self.F)
        g = float(self.g)
        a = float(self.a)
        s1 = float(self.s1)
        v1 = float(self.v1)
        s2 = float(self.s2)
        s = float(self.s)
        
        # 1. Force balance in vertical direction during force application
        N_applied = m * g - F * self.sin37  # Normal force when force is applied
        if not self._is_close(N_applied, max(0, N_applied)):
            errors.append(f"Normal force during application should be non-negative, got {N_applied}")
        
        # 2. Newton's 2nd law during force application
        F_horizontal = F * self.cos37
        F_friction_applied = mu * N_applied
        F_net_applied = F_horizontal - F_friction_applied
        a_calculated = F_net_applied / m if m > 0 else 0
        
        if not self._is_close(a, a_calculated):
            errors.append(f"Acceleration mismatch: expected {a_calculated}, got {a}")
        
        # 3. Kinematics consistency for first phase
        t1 = 4.0  # Given in problem
        s1_calculated = 0.5 * a * t1 ** 2
        v1_calculated = a * t1
        
        if not self._is_close(s1, s1_calculated):
            errors.append(f"s1 mismatch: expected {s1_calculated}, got {s1}")
        
        if not self._is_close(v1, v1_calculated):
            errors.append(f"v1 mismatch: expected {v1_calculated}, got {v1}")
        
        # 4. Newton's 2nd law after force removal
        N_removed = m * g  # Normal force when force is removed
        F_friction_removed = mu * N_removed
        a_decel = -F_friction_removed / m if m > 0 else 0
        
        # 5. Kinematics consistency for second phase
        s2_calculated = (v1 ** 2) / (2 * abs(a_decel)) if abs(a_decel) > 1e-9 else 0
        
        if not self._is_close(s2, s2_calculated):
            errors.append(f"s2 mismatch: expected {s2_calculated}, got {s2}")
        
        # 6. Energy/work considerations
        # Work done by applied force: W_F = F * cos37 * s1
        # Work done by friction in phase 1: W_f1 = -mu * N_applied * s1
        # Work done by friction in phase 2: W_f2 = -mu * N_removed * s2
        # Initial KE = 0, final KE = 0, so net work should be 0
        W_F = F * self.cos37 * s1
        W_f1 = -mu * N_applied * s1
        W_f2 = -mu * N_removed * s2
        net_work = W_F + W_f1 + W_f2
        
        if not self._is_close(net_work, 0, abs_tol=1.0):  # Allow 1J tolerance
            errors.append(f"Energy not conserved: net work = {net_work}")
        
        # 7. Total distance consistency
        s_calculated = s1 + s2
        if not self._is_close(s, s_calculated):
            errors.append(f"Total distance mismatch: expected {s_calculated}, got {s}")
        
        if errors:
            for err in errors:
                self._log(err, is_error=True)
            return False
        else:
            self._log("All invariants passed")
            return True
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        errors = []
        
        a = float(self.a)
        s1 = float(self.s1)
        v1 = float(self.v1)
        s2 = float(self.s2)
        s = float(self.s)
        
        # Check step-by-step consistency from the solution trace
        
        # Step 4: acceleration should be 6 m/s²
        if not self._is_close(a, 6.0):
            errors.append(f"Step 4 acceleration mismatch: expected 6, got {a}")
        
        # Step 5: s1 = 48 m (from kinematics with a=6, t=4)
        if not self._is_close(s1, 48.0):
            errors.append(f"Step 5 distance mismatch: expected 48, got {s1}")
        
        # Step 6: v1 = 24 m/s (from v = a*t with a=6, t=4)
        if not self._is_close(v1, 24.0):
            errors.append(f"Step 6 velocity mismatch: expected 24, got {v1}")
        
        # Step 8: s2 = 57.6 m (from kinematics with v1=24, a_decel=-5)
        if not self._is_close(s2, 57.6):
            errors.append(f"Step 8 distance mismatch: expected 57.6, got {s2}")
        
        # Step 9: s = 105.6 m (sum of s1 and s2)
        if not self._is_close(s, 105.6):
            errors.append(f"Step 9 total distance mismatch: expected 105.6, got {s}")
        
        if errors:
            for err in errors:
                self._log(err, is_error=True)
            return False
        else:
            self._log("Trace consistency passed")
            return True
    
    def test_monotonicity(self):
        """Perturb mass and check if total distance changes as expected."""
        original_m = float(self.m)
        perturbed_m = original_m * 1.05  # Increase by 5%
        
        # Recalculate with perturbed mass using formulas
        # Phase 1: Force applied
        N_applied_pert = perturbed_m * self.g - self.F * self.sin37
        N_applied_pert = max(0, N_applied_pert)  # Ensure non-negative
        
        F_horizontal = self.F * self.cos37
        F_friction_applied_pert = self.mu * N_applied_pert
        F_net_applied_pert = F_horizontal - F_friction_applied_pert
        a_pert = F_net_applied_pert / perturbed_m if perturbed_m > 0 else 0
        
        t1 = 4.0
        s1_pert = 0.5 * a_pert * t1 ** 2
        v1_pert = a_pert * t1
        
        # Phase 2: Force removed
        N_removed_pert = perturbed_m * self.g
        F_friction_removed_pert = self.mu * N_removed_pert
        a_decel_pert = -F_friction_removed_pert / perturbed_m if perturbed_m > 0 else 0
        
        s2_pert = (v1_pert ** 2) / (2 * abs(a_decel_pert)) if abs(a_decel_pert) > 1e-9 else 0
        
        s_pert = s1_pert + s2_pert
        
        # Original total distance
        s_original = float(self.s)
        
        # Check monotonicity: increasing mass should decrease total distance
        # Because friction increases with mass, reducing net acceleration
        if s_pert < s_original:
            self._log(f"Monotonicity passed: increasing mass from {original_m} to {perturbed_m} "
                     f"decreased distance from {s_original:.2f} to {s_pert:.2f}")
            return True
        else:
            self._log(f"Monotonicity failed: increasing mass should decrease distance, but "
                     f"distance changed from {s_original:.2f} to {s_pert:.2f}", is_error=True)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []  # Reset logs
        success = True
        
        try:
            if not self.test_invariants():
                success = False
        except Exception as e:
            self._log(f"test_invariants failed with exception: {traceback.format_exc()}", is_error=True)
            success = False
        
        try:
            if not self.test_trace_consistency():
                success = False
        except Exception as e:
            self._log(f"test_trace_consistency failed with exception: {traceback.format_exc()}", is_error=True)
            success = False
        
        try:
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self._log(f"test_monotonicity failed with exception: {traceback.format_exc()}", is_error=True)
            success = False
        
        return success, self.logs