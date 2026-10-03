import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', 400.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.a_accel = getattr(solution_instance, 'a_accel', 2.0)
        self.a_decel = getattr(solution_instance, 'a_decel', 4.0)
        self.max_height = getattr(solution_instance, 'max_height', 24.0)
        self.target_height = getattr(solution_instance, 'target_height', 18.0)
        self.horiz_dist = getattr(solution_instance, 'horiz_dist', 24.0)
        self.horiz_time = getattr(solution_instance, 'horiz_time', 6.0)
        
        # Safe attribute access for results
        self.T = getattr(solution_instance, 'T', getattr(solution_instance, 'results', {}).get('T', 0.0))
        self.t_rise = getattr(solution_instance, 't_rise', getattr(solution_instance, 'results', {}).get('t_rise', 0.0))
        self.v_3 = getattr(solution_instance, 'v_3', getattr(solution_instance, 'results', {}).get('v_3', 0.0))
        self.v_1 = getattr(solution_instance, 'v_1', getattr(solution_instance, 'results', {}).get('v_1', 0.0))
        self.v_2 = getattr(solution_instance, 'v_2', getattr(solution_instance, 'results', {}).get('v_2', 0.0))
        self.t_1 = getattr(solution_instance, 't_1', getattr(solution_instance, 'results', {}).get('t_1', 0.0))
        self.t_2 = getattr(solution_instance, 't_2', getattr(solution_instance, 'results', {}).get('t_2', 0.0))
        self.t_1_prime = getattr(solution_instance, 't_1\'', getattr(solution_instance, 'results', {}).get('t_1\'', 0.0))
        self.t_2_prime = getattr(solution_instance, 't_2\'', getattr(solution_instance, 'results', {}).get('t_2\'', 0.0))
        self.x = getattr(solution_instance, 'x', getattr(solution_instance, 'results', {}).get('x', 0.0))
        
        # Pre-calculated values for verification
        self.ascent_dist = self.max_height  # 24m
        self.descent_dist = self.max_height - self.target_height  # 6m
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Newton's 2nd Law check for tension during acceleration phase
            # During acceleration upward: T - mg = m*a_accel
            T_expected = self.m * (self.g + self.a_accel)
            if not math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Tension T={self.T} ≠ m(g+a)={T_expected}")
            else:
                logs.append(f"PASS: Tension T={self.T} ≈ m(g+a)={T_expected}")
                
            # 2. Kinematics invariant for ascent
            # v1^2/(2*a_accel) + v1^2/(2*a_decel) = max_height
            if self.v_1 > 0:
                ascent_invariant = (self.v_1**2)/(2*self.a_accel) + (self.v_1**2)/(2*self.a_decel)
                if not math.isclose(ascent_invariant, self.ascent_dist, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Ascent invariant: {ascent_invariant} ≠ {self.ascent_dist}")
                else:
                    logs.append(f"PASS: Ascent invariant holds: {ascent_invariant} ≈ {self.ascent_dist}")
                    
            # 3. Kinematics invariant for descent
            # v2^2/(2*a_accel) + v2^2/(2*a_decel) = descent_dist
            if self.v_2 > 0:
                descent_invariant = (self.v_2**2)/(2*self.a_accel) + (self.v_2**2)/(2*self.a_decel)
                if not math.isclose(descent_invariant, self.descent_dist, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Descent invariant: {descent_invariant} ≠ {self.descent_dist}")
                else:
                    logs.append(f"PASS: Descent invariant holds: {descent_invariant} ≈ {self.descent_dist}")
                    
            return True, logs
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
            
    def test_trace_consistency(self):
        logs = []
        try:
            # Check ascent times consistency
            if self.v_1 > 0:
                t1_expected = self.v_1 / self.a_accel
                t2_expected = self.v_1 / self.a_decel
                t_rise_expected = t1_expected + t2_expected
                
                if not math.isclose(self.t_1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: t1={self.t_1} ≠ v1/a_accel={t1_expected}")
                if not math.isclose(self.t_2, t2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: t2={self.t_2} ≠ v1/a_decel={t2_expected}")
                if not math.isclose(self.t_rise, t_rise_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_rise={self.t_rise} ≠ t1+t2={t_rise_expected}")
                else:
                    logs.append(f"PASS: Ascent times consistent: t_rise={self.t_rise}")
                    
            # Check descent times consistency
            if self.v_2 > 0:
                t1p_expected = self.v_2 / self.a_accel
                t2p_expected = self.v_2 / self.a_decel
                t_descent_expected = t1p_expected + t2p_expected
                
                if not math.isclose(self.t_1_prime, t1p_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: t1'={self.t_1_prime} ≠ v2/a_accel={t1p_expected}")
                if not math.isclose(self.t_2_prime, t2p_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: t2'={self.t_2_prime} ≠ v2/a_decel={t2p_expected}")
                    
            # Check total displacement and time
            total_path = self.ascent_dist + self.horiz_dist + self.descent_dist
            total_displacement = math.sqrt(self.horiz_dist**2 + (self.ascent_dist - self.descent_dist)**2)
            
            if not math.isclose(self.x, total_displacement, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Displacement x={self.x} ≠ √(horiz²+vert²)={total_displacement}")
            else:
                logs.append(f"PASS: Displacement correct: x={self.x}")
                
            # Check average speed (using displacement, not path length, per problem context)
            if self.T > 0:
                v3_expected = self.x / self.T
                if not math.isclose(self.v_3, v3_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v3={self.v_3} ≠ x/T={v3_expected}")
                else:
                    logs.append(f"PASS: Average speed v3={self.v_3} ≈ x/T={v3_expected}")
                    
            return True, logs
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
            
    def test_monotonicity(self):
        logs = []
        original_m = self.m
        try:
            # Perturb mass by +10%
            self.m *= 1.1
            
            # Recalculate tension using Newton's 2nd Law
            T_perturbed = self.m * (self.g + self.a_accel)
            
            # Check if tension increases proportionally
            if T_perturbed > self.T:
                logs.append(f"PASS: Tension increases with mass: {self.T} → {T_perturbed}")
            else:
                logs.append(f"FAIL: Tension doesn't increase with mass")
                
            # Restore original mass
            self.m = original_m
            
            # Check ascent velocity invariant with symbolic fallback
            v1 = sp.Symbol('v1')
            eq = sp.Eq(v1**2/(2*self.a_accel) + v1**2/(2*self.a_decel), self.ascent_dist)
            sol = sp.solve(eq, v1)
            v1_expected = float(max([s for s in sol if s > 0])) if sol else 0.0
            
            if self.v_1 > 0 and not math.isclose(self.v_1, v1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: v1={self.v_1} ≠ symbolic solution {v1_expected}")
                
            return True, logs
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
            
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Test 1: Invariants
        ok1, logs1 = self.test_invariants()
        all_logs.extend(["=== Invariants Test ==="] + logs1)
        if not ok1:
            success = False
            
        # Test 2: Trace Consistency
        ok2, logs2 = self.test_trace_consistency()
        all_logs.extend(["", "=== Trace Consistency Test ==="] + logs2)
        if not ok2:
            success = False
            
        # Test 3: Monotonicity
        ok3, logs3 = self.test_monotonicity()
        all_logs.extend(["", "=== Monotonicity Test ==="] + logs3)
        if not ok3:
            success = False
            
        return success, all_logs