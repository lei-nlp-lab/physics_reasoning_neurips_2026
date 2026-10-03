import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'results', {}).get('B', 0.0))
        self.t1 = getattr(solution_instance, 't_1', getattr(solution_instance, 'results', {}).get('t_1', 0.0))
        self.t2 = getattr(solution_instance, 't_2', getattr(solution_instance, 'results', {}).get('t_2', 0.0))
        self.t_total = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.v_land = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.Q = getattr(solution_instance, 'Q', getattr(solution_instance, 'results', {}).get('Q', 0.0))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.0))
        
        # Physical constants (problem given)
        self.h = 0.8
        self.l = 0.2
        self.m = 0.1
        self.R = 0.08
        self.v0 = 2.0
        self.F = 2.0
        self.g = 10.0
        
        # Derived quantities for checks
        self.v_boundary = None  # velocity when leaving field on descent
        self.I_entry = None     # induced current during entry
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Energy conservation: Q = 2mgl
            Q_expected = 2 * self.m * self.g * self.l
            if not math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Energy conservation failed: Q={self.Q:.3f}, expected {Q_expected:.3f}")
            else:
                logs.append(f"Energy conservation passed: Q ≈ 2mgl = {Q_expected:.3f} J")
                
            # 2. Newton's 1st law during entry: F = mg + BIl
            # First compute induced current during entry
            self.I_entry = self.B * self.l * self.v0 / self.R if self.B > 0 else 0.0
            force_balance_entry = self.F - (self.m * self.g + self.B * self.I_entry * self.l)
            if not math.isclose(force_balance_entry, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton's 1st law (entry) failed: F - (mg + BIl) = {force_balance_entry:.3f} N")
            else:
                logs.append(f"Newton's 1st law (entry) passed: F = mg + BIl")
                
            # 3. Newton's 1st law at boundary during descent: mg = BIl
            # Compute velocity at boundary when leaving field
            if self.B > 0:
                self.v_boundary = (self.m * self.g * self.R) / (self.B**2 * self.l**2)
                I_boundary = self.B * self.l * self.v_boundary / self.R
                force_balance_exit = self.m * self.g - self.B * I_boundary * self.l
                if not math.isclose(force_balance_exit, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Newton's 1st law (exit) failed: mg - BIl = {force_balance_exit:.3f} N")
                else:
                    logs.append(f"Newton's 1st law (exit) passed: mg = BIl at boundary")
            else:
                logs.append("Skipping exit force balance: B=0")
                
        except Exception as e:
            logs.append(f"Error in invariants: {e}")
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # 1. Time to fully enter field: t1 = l / v0
            t1_expected = self.l / self.v0
            if not math.isclose(self.t1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"t1 inconsistency: {self.t1:.3f} s, expected {t1_expected:.3f} s")
            else:
                logs.append(f"t1 consistent: l/v0 = {t1_expected:.3f} s")
                
            # 2. Time to highest point after entry: t2 = v0 / g
            t2_expected = self.v0 / self.g
            if not math.isclose(self.t2, t2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"t2 inconsistency: {self.t2:.3f} s, expected {t2_expected:.3f} s")
            else:
                logs.append(f"t2 consistent: v0/g = {t2_expected:.3f} s")
                
            # 3. Total time to highest point: t = t1 + t2
            t_expected = self.t1 + self.t2
            if not math.isclose(self.t_total, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Total time inconsistency: {self.t_total:.3f} s, expected {t_expected:.3f} s")
            else:
                logs.append(f"Total time consistent: t1 + t2 = {t_expected:.3f} s")
                
            # 4. Induced current consistency (if provided)
            if hasattr(self, 'I') and self.I > 0:
                I_expected = self.B * self.l * self.v0 / self.R
                if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Induced current inconsistency: I={self.I:.3f} A, expected {I_expected:.3f} A")
                else:
                    logs.append(f"Induced current consistent: Blv0/R = {I_expected:.3f} A")
                    
            # 5. Landing velocity from kinematic equation
            if self.v_boundary is None and self.B > 0:
                self.v_boundary = (self.m * self.g * self.R) / (self.B**2 * self.l**2)
            if self.v_boundary is not None:
                v_land_sq_expected = self.v_boundary**2 + 2 * self.g * (self.h - self.l)
                v_land_expected = math.sqrt(max(v_land_sq_expected, 0))
                if not math.isclose(self.v_land, v_land_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Landing velocity inconsistency: v={self.v_land:.3f} m/s, expected {v_land_expected:.3f} m/s")
                else:
                    logs.append(f"Landing velocity consistent: v² = v_boundary² + 2g(h-l)")
                    
        except Exception as e:
            logs.append(f"Error in trace consistency: {e}")
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_m = self.m
        try:
            # Perturb mass by +1%
            self.m = original_m * 1.01
            
            # Recalculate dependent quantities
            B_pert = math.sqrt(max((self.F - self.m * self.g) * self.R / (self.l**2 * self.v0), 0))
            v_boundary_pert = (self.m * self.g * self.R) / (B_pert**2 * self.l**2) if B_pert > 0 else 0
            v_land_pert = math.sqrt(v_boundary_pert**2 + 2 * self.g * (self.h - self.l))
            Q_pert = 2 * self.m * self.g * self.l
            
            # Restore original mass
            self.m = original_m
            
            # Check monotonic trends
            if B_pert < self.B:
                logs.append(f"Monotonicity: B decreases with increased mass (perturbed {B_pert:.3f} T vs original {self.B:.3f} T)")
            else:
                logs.append(f"Monotonicity warning: B did not decrease as expected")
                
            if v_boundary_pert > self.v_boundary:
                logs.append(f"Monotonicity: v_boundary increases with increased mass")
            else:
                logs.append(f"Monotonicity warning: v_boundary did not increase as expected")
                
            if v_land_pert > self.v_land:
                logs.append(f"Monotonicity: landing velocity increases with increased mass")
            else:
                logs.append(f"Monotonicity warning: landing velocity did not increase as expected")
                
            if Q_pert > self.Q:
                logs.append(f"Monotonicity: heat generated increases with increased mass")
            else:
                logs.append(f"Monotonicity warning: heat generated did not increase as expected")
                
        except Exception as e:
            logs.append(f"Error in monotonicity test: {e}")
            self.m = original_m  # Ensure restoration on error
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        try:
            # Run tests in sequence
            for test_name, test_method in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== {test_name} ===")
                logs = test_method()
                all_logs.extend(logs)
                
                # Check for critical failures in invariants and trace consistency
                if test_name in ["Invariants", "Trace Consistency"]:
                    error_keywords = ["failed", "inconsistency", "Error"]
                    if any(any(keyword in log.lower() for keyword in error_keywords) for log in logs):
                        success = False
                        
        except Exception as e:
            all_logs.append(f"Critical error during testing: {e}")
            all_logs.append(traceback.format_exc())
            success = False
            
        return success, all_logs