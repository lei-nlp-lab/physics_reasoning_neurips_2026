import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallbacks
        self.v_2 = self.results.get('v_2', 0.0)
        self.t_2 = self.results.get('t_2', 0.0)
        self.t = self.results.get('t', 0.0)
        self.v = self.results.get('v', 0.0)
        self.t_1 = self.results.get('t_1', 0.0)
        self.v_1 = self.results.get('v_1', 0.0)
        self.L_result = self.results.get('L', 0.0)
        
        # Use L_result for all calculations
        self.L = self.L_result
        
        # Use h=1.0 and g=9.8 as in the solution
        self.h = 1.0
        self.g = 9.8
        
    def _compare(self, val1, val2):
        """Fuzzy comparison with physics tolerance"""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2)
        return False
    
    def test_step1(self):
        """Check midpoint time calculation"""
        logs = []
        try:
            # Expected: t = sqrt(3h/g)
            t_expected = math.sqrt(3 * self.h / self.g)
            if self._compare(self.t, t_expected):
                logs.append("✓ Step 1: Midpoint time correct")
                return True, logs
            else:
                logs.append(f"✗ Step 1: t={self.t:.4f}, expected={t_expected:.4f}")
                return False, logs
        except Exception as e:
            logs.append(f"✗ Step 1 error: {str(e)}")
            return False, logs
    
    def test_step2(self):
        """Check point B calculations"""
        logs = []
        try:
            # t1 = sqrt(4h/g)
            t1_expected = math.sqrt(4 * self.h / self.g)
            # v1 = L * sqrt(g/(4h))
            v1_expected = self.L * math.sqrt(self.g / (4 * self.h))
            
            t1_ok = self._compare(self.t_1, t1_expected)
            v1_ok = self._compare(self.v_1, v1_expected)
            
            if t1_ok and v1_ok:
                logs.append("✓ Step 2: Point B calculations correct")
                return True, logs
            else:
                if not t1_ok:
                    logs.append(f"✗ Step 2 t1: {self.t_1:.4f}, expected={t1_expected:.4f}")
                if not v1_ok:
                    logs.append(f"✗ Step 2 v1: {self.v_1:.4f}, expected={v1_expected:.4f}")
                return False, logs
        except Exception as e:
            logs.append(f"✗ Step 2 error: {str(e)}")
            return False, logs
    
    def test_step3(self):
        """Check point A calculations"""
        logs = []
        try:
            # t2 = sqrt(2h/g)
            t2_expected = math.sqrt(2 * self.h / self.g)
            # v2 = L * sqrt(g/(2h))
            v2_expected = self.L * math.sqrt(self.g / (2 * self.h))
            
            t2_ok = self._compare(self.t_2, t2_expected)
            v2_ok = self._compare(self.v_2, v2_expected)
            
            if t2_ok and v2_ok:
                logs.append("✓ Step 3: Point A calculations correct")
                return True, logs
            else:
                if not t2_ok:
                    logs.append(f"✗ Step 3 t2: {self.t_2:.4f}, expected={t2_expected:.4f}")
                if not v2_ok:
                    logs.append(f"✗ Step 3 v2: {self.v_2:.4f}, expected={v2_expected:.4f}")
                return False, logs
        except Exception as e:
            logs.append(f"✗ Step 3 error: {str(e)}")
            return False, logs
    
    def test_step4(self):
        """Check velocity range"""
        logs = []
        try:
            v_min = self.L * math.sqrt(self.g / (4 * self.h))
            v_max = self.L * math.sqrt(self.g / (2 * self.h))
            
            # Check if v is within range
            if v_min <= self.v <= v_max:
                logs.append("✓ Step 4: Velocity within valid range")
                return True, logs
            else:
                logs.append(f"✗ Step 4: v={self.v:.4f} not in [{v_min:.4f},{v_max:.4f}]")
                return False, logs
        except Exception as e:
            logs.append(f"✗ Step 4 error: {str(e)}")
            return False, logs
    
    def test_step5_energy(self):
        """Check energy conservation leads to L=2√2 h"""
        logs = []
        try:
            # Numerical check
            v1_num = self.L * math.sqrt(self.g / (4 * self.h))
            v2_num = self.L * math.sqrt(self.g / (2 * self.h))
            
            energy_A_num = 0.5 * v2_num**2 + self.g * self.h  # m cancels
            energy_B_num = 0.5 * v1_num**2 + self.g * (2 * self.h)
            
            if self._compare(energy_A_num, energy_B_num):
                logs.append("✓ Step 5 (numerical): Energies match")
                
                # Also check L result
                L_expected_num = 2 * math.sqrt(2) * self.h
                if self._compare(self.L_result, L_expected_num):
                    logs.append("✓ Step 5: L result matches 2√2 h")
                    return True, logs
                else:
                    logs.append(f"✗ Step 5 L: {self.L_result:.4f}, expected={L_expected_num:.4f}")
                    return False, logs
            else:
                logs.append(f"✗ Step 5 energies: A={energy_A_num:.4f}, B={energy_B_num:.4f}")
                return False, logs
        except Exception as e:
            logs.append(f"✗ Step 5 error: {str(e)}")
            return False, logs
    
    def test_invariants(self):
        """Verify energy conservation at different points"""
        logs = []
        success = True
        
        try:
            # Check energy conservation between A and B (from step 5)
            test, step_logs = self.test_step5_energy()
            logs.extend(step_logs)
            success = success and test
            
            logs.append("Note: Energy conservation check uses P as reference (U=0 at P)")
            
        except Exception as e:
            logs.append(f"✗ Energy invariant error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results"""
        logs = []
        success = True
        
        try:
            # Check t, t1, t2 ordering
            if self.t_2 > 0 and self.t > 0 and self.t_1 > 0:
                # t2 (to A) < t (to midpoint) < t1 (to B)
                if self.t_2 < self.t < self.t_1:
                    logs.append("✓ Time ordering correct: t_A < t_mid < t_B")
                else:
                    logs.append(f"✗ Time ordering wrong: t2={self.t_2:.4f}, t={self.t:.4f}, t1={self.t_1:.4f}")
                    success = False
            
            # Check v1, v2 ordering
            if self.v_1 > 0 and self.v_2 > 0:
                # v1 (to B) < v2 (to A) since B is farther down
                if self.v_1 < self.v_2:
                    logs.append("✓ Velocity ordering correct: v_B < v_A")
                else:
                    logs.append(f"✗ Velocity ordering wrong: v1={self.v_1:.4f}, v2={self.v_2:.4f}")
                    success = False
            
            # Check that L/t gives velocities
            v1_from_L = self.L / self.t_1 if self.t_1 > 0 else 0
            v2_from_L = self.L / self.t_2 if self.t_2 > 0 else 0
            
            if self._compare(v1_from_L, self.v_1):
                logs.append("✓ v1 = L/t1 consistent")
            else:
                logs.append(f"✗ v1 inconsistency: {v1_from_L:.4f} vs {self.v_1:.4f}")
                success = False
                
            if self._compare(v2_from_L, self.v_2):
                logs.append("✓ v2 = L/t2 consistent")
            else:
                logs.append(f"✗ v2 inconsistency: {v2_from_L:.4f} vs {self.v_2:.4f}")
                success = False
                
        except Exception as e:
            logs.append(f"✗ Trace consistency error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Check that velocity bounds are constant with h when L = 2√2 h"""
        logs = []
        success = True
        
        try:
            # With L = 2√2 h, velocity bounds should be independent of h
            # v_min = √(2g), v_max = 2√g
            
            # Compute theoretical constant values
            v_min_constant = math.sqrt(2 * self.g)
            v_max_constant = 2 * math.sqrt(self.g)
            
            # Compute actual values from the solution
            v_min_actual = self.L * math.sqrt(self.g / (4 * self.h))
            v_max_actual = self.L * math.sqrt(self.g / (2 * self.h))
            
            # Check if they match the theoretical constants
            min_ok = self._compare(v_min_actual, v_min_constant)
            max_ok = self._compare(v_max_actual, v_max_constant)
            
            if min_ok and max_ok:
                logs.append("✓ Monotonicity: Velocity bounds are constant with h (as expected from L = 2√2 h)")
                logs.append(f"  v_min = √(2g) = {v_min_constant:.4f} m/s")
                logs.append(f"  v_max = 2√g = {v_max_constant:.4f} m/s")
                return True, logs
            else:
                if not min_ok:
                    logs.append(f"✗ Monotonicity: v_min={v_min_actual:.4f}, expected √(2g)={v_min_constant:.4f}")
                if not max_ok:
                    logs.append(f"✗ Monotonicity: v_max={v_max_actual:.4f}, expected 2√g={v_max_constant:.4f}")
                return False, logs
            
        except Exception as e:
            logs.append(f"✗ Monotonicity test error: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return results"""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Step 1 (Midpoint time)", self.test_step1),
            ("Step 2 (Point B)", self.test_step2),
            ("Step 3 (Point A)", self.test_step3),
            ("Step 4 (Velocity range)", self.test_step4),
            ("Step 5 (Energy & L)", self.test_step5_energy),
            ("Energy Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs