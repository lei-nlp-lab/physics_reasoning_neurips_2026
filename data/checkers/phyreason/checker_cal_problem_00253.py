import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Follows robust access patterns to avoid sandbox errors.
        """
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters from problem statement
        self.v_kick = 30.0  # m/s, velocity after penalty kick
        self.dt_kick = 0.1  # s, duration of foot-ball interaction
        self.t_air = 1.0    # s, time in air before blocking (not directly used)
        self.dt_catch = 0.2 # s, duration of goalkeeper contact
        self.v_rebound = 10.0  # m/s, velocity after rebound (opposite direction)
        
        # Extract calculated results with safe fallbacks
        self.a_kick = self._safe_get('a', 0.0)  # Acceleration during kick
        self.dv_catch = self._safe_get("Deltav'", 0.0)  # Change in velocity during catch
        self.a_catch = self._safe_get("a'", 0.0)  # Acceleration during catch
        
        # Expected values from steps analysis
        self.expected_a_kick = 300.0
        self.expected_dv_catch = -40.0
        self.expected_a_catch = -200.0
        
        # For monotonicity tests
        self.original_inputs = {
            'v_kick': self.v_kick,
            'dt_kick': self.dt_kick,
            'dt_catch': self.dt_catch,
            'v_rebound': self.v_rebound
        }
    
    def _safe_get(self, key, default):
        """Safely retrieve value from results dict or attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.results.get(key, default)
    
    def _is_close(self, a, b):
        """Robust fuzzy comparison for physics values."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def _check_definitions(self):
        """Verify definitions of acceleration and velocity change."""
        logs = []
        
        # Check kick acceleration definition
        dv_kick = self.v_kick - 0  # From rest to v_kick
        a_kick_calc = dv_kick / self.dt_kick
        if not self._is_close(self.a_kick, a_kick_calc):
            logs.append(f"Kick acceleration mismatch: expected {a_kick_calc}, got {self.a_kick}")
        
        # Check catch velocity change definition
        dv_catch_calc = -self.v_rebound - self.v_kick  # Opposite direction: final = -10, initial = 30
        if not self._is_close(self.dv_catch, dv_catch_calc):
            logs.append(f"Catch Δv mismatch: expected {dv_catch_calc}, got {self.dv_catch}")
        
        # Check catch acceleration definition
        if abs(self.dt_catch) > 1e-9:  # Avoid division by zero
            a_catch_calc = self.dv_catch / self.dt_catch
            if not self._is_close(self.a_catch, a_catch_calc):
                logs.append(f"Catch acceleration mismatch: expected {a_catch_calc}, got {self.a_catch}")
        
        return logs
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Definition consistency (acceleration = Δv/Δt)
        2. Sign conventions (direction consistency)
        """
        logs = []
        
        # 1. Check definitions
        logs.extend(self._check_definitions())
        
        # 2. Check sign conventions
        # During kick: acceleration should be positive (speeding up forward)
        if self.a_kick < 0:
            logs.append(f"Kick acceleration negative ({self.a_kick}), expected positive")
        
        # During catch: acceleration should be negative (slowing down forward motion, then reversing)
        if self.a_catch > 0:
            logs.append(f"Catch acceleration positive ({self.a_catch}), expected negative")
        
        # 3. Check magnitude relationships
        # |Δv_catch| = |v_initial| + |v_final| since opposite directions
        expected_mag_dv = abs(self.v_kick) + abs(self.v_rebound)
        if not self._is_close(abs(self.dv_catch), expected_mag_dv):
            logs.append(f"|Δv_catch| mismatch: expected {expected_mag_dv}, got {abs(self.dv_catch)}")
        
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify that step-by-step calculations are internally consistent.
        """
        logs = []
        
        # Check against expected values from steps analysis
        if not self._is_close(self.a_kick, self.expected_a_kick):
            logs.append(f"Kick acceleration: expected {self.expected_a_kick}, got {self.a_kick}")
        
        if not self._is_close(self.dv_catch, self.expected_dv_catch):
            logs.append(f"Catch Δv: expected {self.expected_dv_catch}, got {self.dv_catch}")
        
        if not self._is_close(self.a_catch, self.expected_a_catch):
            logs.append(f"Catch acceleration: expected {self.expected_a_catch}, got {self.a_catch}")
        
        # Check that a_catch = dv_catch / dt_catch (already in invariants, but double-check)
        if abs(self.dt_catch) > 1e-9:
            calc_a_catch = self.dv_catch / self.dt_catch
            if not self._is_close(self.a_catch, calc_a_catch):
                logs.append(f"Trace inconsistency: a' should be Δv'/Δt' = {calc_a_catch}, got {self.a_catch}")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb input parameters and verify expected trends.
        Restore original values after testing.
        """
        logs = []
        
        # Test 1: Increase kick velocity -> increase kick acceleration
        try:
            self.v_kick += 5.0  # Perturb
            dv_kick = self.v_kick - 0
            new_a_kick = dv_kick / self.dt_kick
            
            if new_a_kick <= self.a_kick:
                logs.append(f"Monotonicity fail: increasing v_kick didn't increase a_kick")
        finally:
            self.v_kick = self.original_inputs['v_kick']  # Restore
        
        # Test 2: Increase kick duration -> decrease kick acceleration
        try:
            self.dt_kick += 0.02  # Perturb
            dv_kick = self.v_kick - 0
            new_a_kick = dv_kick / self.dt_kick
            
            if new_a_kick >= self.a_kick:
                logs.append(f"Monotonicity fail: increasing dt_kick didn't decrease a_kick")
        finally:
            self.dt_kick = self.original_inputs['dt_kick']  # Restore
        
        # Test 3: Increase rebound speed magnitude -> increase |Δv_catch|
        try:
            self.v_rebound += 2.0  # More negative (opposite direction)
            new_dv_catch = -self.v_rebound - self.v_kick
            
            if abs(new_dv_catch) <= abs(self.dv_catch):
                logs.append(f"Monotonicity fail: increasing |v_rebound| didn't increase |Δv_catch|")
        finally:
            self.v_rebound = self.original_inputs['v_rebound']  # Restore
        
        # Test 4: Increase catch duration -> decrease |a_catch|
        try:
            self.dt_catch += 0.05  # Perturb
            new_a_catch = self.dv_catch / self.dt_catch
            
            if abs(new_a_catch) >= abs(self.a_catch):
                logs.append(f"Monotonicity fail: increasing dt_catch didn't decrease |a_catch|")
        finally:
            self.dt_catch = self.original_inputs['dt_catch']  # Restore
        
        return logs
    
    def run_all_tests(self):
        """
        Execute all physics audits and return success flag with logs.
        """
        all_logs = []
        success = True
        
        try:
            # Run invariant checks
            logs = self.test_invariants()
            if logs:
                all_logs.append("=== Invariant Test Failures ===")
                all_logs.extend(logs)
                success = False
            
            # Run trace consistency checks
            logs = self.test_trace_consistency()
            if logs:
                all_logs.append("=== Trace Consistency Failures ===")
                all_logs.extend(logs)
                success = False
            
            # Run monotonicity checks
            logs = self.test_monotonicity()
            if logs:
                all_logs.append("=== Monotonicity Test Failures ===")
                all_logs.extend(logs)
                # Note: monotonicity failures don't necessarily mean wrong answer
                # They indicate potential issues in reasoning
            
        except Exception as e:
            all_logs.append(f"=== Runtime Error ===")
            all_logs.append(traceback.format_exc())
            success = False
        
        # Final check: magnitudes match target answers
        target1 = 300.0  # m/s²
        target2 = 200.0  # m/s² (magnitude)
        
        if not self._is_close(self.a_kick, target1):
            all_logs.append(f"Target answer mismatch: a_kick should be {target1}, got {self.a_kick}")
            success = False
        
        if not self._is_close(abs(self.a_catch), target2):
            all_logs.append(f"Target answer mismatch: |a_catch| should be {target2}, got {abs(self.a_catch)}")
            success = False
        
        return success, all_logs

# Optional: Symbolic verification for completeness
def symbolic_verification():
    """Symbolic physics laws for reference (not called by default)."""
    v0, v1, dt = sp.symbols('v0 v1 dt')
    delta_v = v1 - v0
    a_avg = delta_v / dt
    
    # Return symbolic expressions for documentation
    return {
        'delta_v': delta_v,
        'a_avg': a_avg
    }