import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract input parameters from results dict
        self.m = solution_instance.results.get('m', 1.0)
        self.q = solution_instance.results.get('q', 1.0)
        self.E = solution_instance.results.get('E', 1.0)
        self.L = solution_instance.results.get('L', 1.0)
        
        # Safely extract results
        self.v = solution_instance.results.get('v', None)
        self.DeltaE = solution_instance.results.get('DeltaE', None)
        self.v1 = solution_instance.results.get('v_1', None)
        self.I = solution_instance.results.get('I', None)
        
        # Store original inputs for restoration in monotonicity test
        self._original_inputs = {'m': self.m, 'q': self.q, 'E': self.E, 'L': self.L}
        
        # We are in numeric mode (all values should be floats)
        self.numeric_mode = True
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Compare two numeric values."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        logs = []
        try:
            # 1. Work-energy theorem for ball A before collision
            work = self.q * self.E * self.L
            ke_change = 0.5 * (3 * self.m) * self.v**2
            if not self._is_close(work, ke_change):
                logs.append(f"FAIL: Work-energy theorem violation. Work={work}, ΔKE={ke_change}")
            else:
                logs.append("PASS: Work-energy theorem holds for ball A.")
            
            # 2. Conservation of momentum during collision
            p_initial = 3 * self.m * self.v
            p_final = 4 * self.m * self.v1
            if not self._is_close(p_initial, p_final):
                logs.append(f"FAIL: Momentum not conserved. p_i={p_initial}, p_f={p_final}")
            else:
                logs.append("PASS: Momentum conserved during collision.")
            
            # 3. Energy loss consistency
            ke_i = 0.5 * (3 * self.m) * self.v**2
            ke_f = 0.5 * (4 * self.m) * self.v1**2
            delta_expected = ke_i - ke_f
            if not self._is_close(self.DeltaE, delta_expected):
                logs.append(f"FAIL: Energy loss mismatch. ΔE={self.DeltaE}, expected={delta_expected}")
            else:
                logs.append("PASS: Energy loss matches KE difference.")
            
            # 4. Impulse on ball B
            impulse_expected = self.m * self.v1
            if not self._is_close(self.I, impulse_expected):
                logs.append(f"FAIL: Impulse mismatch. I={self.I}, expected={impulse_expected}")
            else:
                logs.append("PASS: Impulse on B equals its momentum change.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Check that v1 derived from v via momentum conservation matches stored v1
            v1_from_v = (3 * self.v) / 4
            if not self._is_close(self.v1, v1_from_v):
                logs.append(f"FAIL: v1 inconsistent. Stored v1={self.v1}, from v: {v1_from_v}")
            else:
                logs.append("PASS: v1 consistent with v via momentum conservation.")
            
            # Check that DeltaE computed from v and v1 matches stored DeltaE
            delta_from_v_v1 = 0.5 * (3 * self.m) * self.v**2 - 0.5 * (4 * self.m) * self.v1**2
            if not self._is_close(self.DeltaE, delta_from_v_v1):
                logs.append(f"FAIL: ΔE inconsistent. Stored ΔE={self.DeltaE}, computed: {delta_from_v_v1}")
            else:
                logs.append("PASS: ΔE consistent with v and v1.")
            
            # Check that I computed from v1 matches stored I
            I_from_v1 = self.m * self.v1
            if not self._is_close(self.I, I_from_v1):
                logs.append(f"FAIL: I inconsistent. Stored I={self.I}, from v1: {I_from_v1}")
            else:
                logs.append("PASS: I consistent with v1.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        logs = []
        try:
            # Perturb q by +10% and check trends
            q_orig = self.q
            self.q = q_orig * 1.1  # Increase by 10%
            
            # Recompute derived quantities using formulas
            v_new = math.sqrt(2 * self.E * self.q * self.L / (3 * self.m))
            DeltaE_new = 0.25 * self.E * self.q * self.L
            I_new = math.sqrt(6 * self.E * self.q * self.L * self.m) / 4
            
            # Check monotonicity: all should increase
            if not (v_new > float(self.v) and DeltaE_new > float(self.DeltaE) and I_new > float(self.I)):
                logs.append(f"FAIL: Monotonicity violated. v: {self.v} -> {v_new}, ΔE: {self.DeltaE} -> {DeltaE_new}, I: {self.I} -> {I_new}")
            else:
                logs.append("PASS: Monotonicity holds (v, ΔE, I increase with q).")
            
            # Restore original q
            self.q = q_orig
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            # Ensure restoration on error
            self.q = self._original_inputs['q']
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_method()
            # Check for any FAIL or ERROR entries
            if any(log.startswith(("FAIL", "ERROR")) for log in logs):
                success = False
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
        
        return success, all_logs