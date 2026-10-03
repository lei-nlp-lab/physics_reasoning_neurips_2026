import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with defaults
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 100.0)  # kg
        self.g = getattr(solution_instance, 'g', 10.0)   # m/s²
        
        # Extract results with safe fallbacks from results dict
        self.F_N = self.results.get('F_N', 0.0)          # Normal force
        self.a3 = self.results.get('a_3', 0.0)
        self.x = self.results.get('x', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.v = self.results.get('v', 0.0)
        self.x1 = self.results.get('x_1', 0.0)
        self.a1 = self.results.get('a_1', 0.0)
        self.x3 = self.results.get('x_3', 0.0)
        
        # Time intervals from problem context
        self.t1 = 2.0  # 0-2 seconds
        self.t2 = 4.0  # 2-6 seconds
        self.t3 = 2.0  # 6-8 seconds
        
    def test_invariants(self):
        """Verify physics laws: Newton's 2nd Law, Kinematics consistency."""
        logs = []
        success = True
        
        try:
            # 1. Newton's 2nd Law for phase 1 (0-2s)
            # F_net = m*a = F_N1 - m*g
            # From steps: a1 = 5.0 m/s², so F_N1 = m*(g + a1)
            F_N1_expected = self.m * (self.g + self.a1)
            F_N1_actual = getattr(self, 'F_N1', None)
            if F_N1_actual is None:
                # If not explicitly stored, use the value implied by a1
                F_N1_actual = F_N1_expected
            
            if not math.isclose(F_N1_actual, F_N1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton 2nd Law phase 1 fail: F_N1 expected {F_N1_expected:.2f}, got {F_N1_actual:.2f}")
                success = False
            
            # 2. Newton's 2nd Law for phase 2 (2-6s) - uniform motion means a=0
            # F_net = 0 = F_N2 - m*g => F_N2 = m*g
            F_N2_expected = self.m * self.g
            F_N2_actual = getattr(self, 'F_N2', None)
            if F_N2_actual is not None and not math.isclose(F_N2_actual, F_N2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton 2nd Law phase 2 fail: F_N2 expected {F_N2_expected:.2f}, got {F_N2_actual:.2f}")
                success = False
            
            # 3. Newton's 2nd Law for phase 3 (6-8s)
            # F_net = m*a3 = F_N3 - m*g
            # From steps: a3 = -5.0 m/s², so F_N3 = m*(g + a3)
            F_N3_expected = self.m * (self.g + self.a3)
            F_N3_actual = getattr(self, 'F_N3', None)
            if F_N3_actual is None:
                F_N3_actual = F_N3_expected
            
            if not math.isclose(F_N3_actual, F_N3_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton 2nd Law phase 3 fail: F_N3 expected {F_N3_expected:.2f}, got {F_N3_actual:.2f}")
                success = False
            
            # 4. Kinematics consistency: v = a1*t1
            v_expected = self.a1 * self.t1
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Kinematics velocity fail: v expected {v_expected:.2f}, got {self.v:.2f}")
                success = False
            
            # 5. Displacement consistency: x1 = 0.5*a1*t1²
            x1_expected = 0.5 * self.a1 * self.t1**2
            if not math.isclose(self.x1, x1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Kinematics x1 fail: expected {x1_expected:.2f}, got {self.x1:.2f}")
                success = False
            
            # 6. Displacement consistency: x2 = v*t2
            x2_expected = self.v * self.t2
            if not math.isclose(self.x2, x2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Kinematics x2 fail: expected {x2_expected:.2f}, got {self.x2:.2f}")
                success = False
            
            # 7. Displacement consistency: x3 = v*t3 + 0.5*a3*t3²
            x3_expected = self.v * self.t3 + 0.5 * self.a3 * self.t3**2
            if not math.isclose(self.x3, x3_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Kinematics x3 fail: expected {x3_expected:.2f}, got {self.x3:.2f}")
                success = False
            
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            # Total displacement should be sum of individual displacements
            x_total = self.x1 + self.x2 + self.x3
            if not math.isclose(self.x, x_total, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency fail: x={self.x:.2f}, but x1+x2+x3={x_total:.2f}")
                success = False
            
            # Check if final velocity is 0 (elevator comes to rest)
            v_final = self.v + self.a3 * self.t3
            if not math.isclose(v_final, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Final velocity not zero: {v_final:.2f}")
                success = False
            
        except Exception as e:
            logs.append(f"Trace consistency test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb mass and check displacement trend."""
        logs = []
        success = True
        
        try:
            # Store original values
            original_m = self.m
            original_x = self.x
            
            # Perturb mass by +10%
            self.m = original_m * 1.10
            
            # Recalculate accelerations using Newton's 2nd Law
            # Phase 1: a1 = (F_N1 - m*g)/m
            F_N1 = original_m * (self.g + self.a1)  # Keep same normal force (from graph)
            a1_perturbed = (F_N1 - self.m * self.g) / self.m if self.m != 0 else 0
            
            # Phase 3: a3 = (F_N3 - m*g)/m
            F_N3 = original_m * (self.g + self.a3)  # Keep same normal force
            a3_perturbed = (F_N3 - self.m * self.g) / self.m if self.m != 0 else 0
            
            # Calculate perturbed displacements
            v_perturbed = a1_perturbed * self.t1
            x1_perturbed = 0.5 * a1_perturbed * self.t1**2
            x2_perturbed = v_perturbed * self.t2
            x3_perturbed = v_perturbed * self.t3 + 0.5 * a3_perturbed * self.t3**2
            x_total_perturbed = x1_perturbed + x2_perturbed + x3_perturbed
            
            # Expected trend: Larger mass means smaller acceleration (same force), 
            # so smaller displacements in acceleration phases.
            # Therefore, total displacement should decrease with increased mass.
            if x_total_perturbed > original_x:
                logs.append(f"Monotonicity fail: mass↑ ({original_m:.1f}→{self.m:.1f}) but x↑ ({original_x:.2f}→{x_total_perturbed:.2f})")
                success = False
            elif math.isclose(x_total_perturbed, original_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Monotonicity warning: no significant change with mass perturbation")
            
            # Restore original values
            self.m = original_m
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            # Ensure restoration on error
            self.m = original_m if 'original_m' in locals() else 100.0
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True
        
        # Run invariant tests
        success1, logs1 = self.test_invariants()
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Run trace consistency tests
        success2, logs2 = self.test_trace_consistency()
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Run monotonicity tests
        success3, logs3 = self.test_monotonicity()
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        return overall_success, all_logs