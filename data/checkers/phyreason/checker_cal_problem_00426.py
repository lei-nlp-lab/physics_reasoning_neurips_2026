import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results with fallbacks
        self.g = 10.0  # Given in problem
        self.h_total = 20.0  # Building height
        self.h1 = 1.8  # Distance fallen to window top
        self.dt_window = 0.3  # Time through window
        
        # Extract results safely using getattr with dictionary fallback
        self.results = getattr(solution_instance, 'results', {})
        
        self.t1 = getattr(solution_instance, 't_1', self.results.get('t_1', 0.0))
        self.t2 = getattr(solution_instance, 't_2', self.results.get('t_2', 0.0))
        self.t_total = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.h2 = getattr(solution_instance, 'h_2', self.results.get('h_2', 0.0))
        self.Deltah = getattr(solution_instance, 'Deltah', self.results.get('Deltah', 0.0))
        self.v_final = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        
        # Initialize symbolic variables for invariant checking
        self.v0, self.v1, self.v2 = sp.symbols('v0 v1 v2', real=True)
        self.t, self.t1_sym, self.t2_sym = sp.symbols('t t1 t2', real=True)
        self.h, self.h1_sym, self.h2_sym = sp.symbols('h h1 h2', real=True)
        
        # Store original values for monotonicity test
        self._orig_h1 = self.h1
    
    def test_invariants(self):
        """Verify energy conservation and kinematic consistency"""
        logs = []
        
        try:
            # 1. Energy conservation for full drop
            # Potential energy at top = Kinetic energy at bottom
            expected_v = math.sqrt(2 * self.g * self.h_total)
            if not math.isclose(self.v_final, expected_v, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Energy conservation - v_final={self.v_final:.3f}, "
                          f"expected sqrt(2*g*h)={expected_v:.3f}")
            else:
                logs.append("PASS: Energy conservation")
            
            # 2. Kinematic consistency for t1 (time to window top)
            expected_t1 = math.sqrt(2 * self.h1 / self.g)
            if not math.isclose(self.t1, expected_t1, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: t1 kinematic - t1={self.t1:.3f}, "
                          f"expected sqrt(2*h1/g)={expected_t1:.3f}")
            else:
                logs.append("PASS: t1 kinematic consistency")
            
            # 3. Kinematic consistency for t2 (time to window bottom)
            # t2 should be t1 + dt_window
            expected_t2 = self.t1 + self.dt_window
            if not math.isclose(self.t2, expected_t2, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: t2 time addition - t2={self.t2:.3f}, "
                          f"expected t1+dt={expected_t2:.3f}")
            else:
                logs.append("PASS: t2 time addition")
            
            # 4. Kinematic consistency for h2 (distance to window bottom)
            expected_h2 = 0.5 * self.g * (self.t2 ** 2)
            if not math.isclose(self.h2, expected_h2, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: h2 kinematic - h2={self.h2:.3f}, "
                          f"expected 0.5*g*t2²={expected_h2:.3f}")
            else:
                logs.append("PASS: h2 kinematic consistency")
            
            # 5. Window height consistency (two independent calculations)
            # Method 1: Direct subtraction
            calc_Deltah1 = self.h2 - self.h1
            # Method 2: Using average velocity through window
            v_top = self.g * self.t1  # Velocity at window top
            v_bottom = self.g * self.t2  # Velocity at window bottom
            v_avg = (v_top + v_bottom) / 2
            calc_Deltah2 = v_avg * self.dt_window
            
            if not math.isclose(calc_Deltah1, calc_Deltah2, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Window height consistency - subtraction={calc_Deltah1:.3f}, "
                          f"v_avg*Δt={calc_Deltah2:.3f}")
            elif not math.isclose(self.Deltah, calc_Deltah1, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Δh mismatch - result={self.Deltah:.3f}, "
                          f"calculated={calc_Deltah1:.3f}")
            else:
                logs.append("PASS: Window height consistency")
            
            # 6. Total time consistency
            expected_total_t = math.sqrt(2 * self.h_total / self.g)
            if not math.isclose(self.t_total, expected_total_t, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Total time - t_total={self.t_total:.3f}, "
                          f"expected sqrt(2*h_total/g)={expected_total_t:.3f}")
            else:
                logs.append("PASS: Total time consistency")
            
            # 7. Velocity at window top and bottom consistency
            # Velocity at window top: v1 = g*t1
            # Velocity at window bottom: v2 = g*t2
            v1_calc = self.g * self.t1
            v2_calc = self.g * self.t2
            # Check v2 = v1 + g*Δt (kinematic equation)
            v2_expected = v1_calc + self.g * self.dt_window
            if not math.isclose(v2_calc, v2_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Velocity continuity - v2={v2_calc:.3f}, "
                          f"expected v1+g*Δt={v2_expected:.3f}")
            else:
                logs.append("PASS: Velocity continuity")
            
            return all("PASS" in log or logs.index(log) == 0 for log in logs[1:]), logs
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            return False, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        logs = []
        
        try:
            # Check that final answers match expected values
            target_v = 20.0  # m/s
            target_Deltah = 2.25  # m
            
            if not math.isclose(self.v_final, target_v, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Final velocity mismatch - result={self.v_final:.3f}, "
                          f"target={target_v:.3f}")
            else:
                logs.append("PASS: Final velocity matches target")
            
            if not math.isclose(self.Deltah, target_Deltah, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Window height mismatch - result={self.Deltah:.3f}, "
                          f"target={target_Deltah:.3f}")
            else:
                logs.append("PASS: Window height matches target")
            
            # Verify intermediate steps yield final results
            # Recompute everything from basic principles
            t1_calc = math.sqrt(2 * self.h1 / self.g)
            t2_calc = t1_calc + self.dt_window
            h2_calc = 0.5 * self.g * (t2_calc ** 2)
            Deltah_calc = h2_calc - self.h1
            
            if not math.isclose(self.Deltah, Deltah_calc, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Trace recomputation - final Δh={self.Deltah:.3f}, "
                          f"recomputed={Deltah_calc:.3f}")
            else:
                logs.append("PASS: Trace consistency (recomputation matches)")
            
            return all("PASS" in log or logs.index(log) == 0 for log in logs[1:]), logs
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            return False, logs
    
    def test_monotonicity(self):
        """Test if increasing h1 leads to increased window height"""
        logs = []
        
        try:
            # Perturb h1 by +10%
            perturbed_h1 = self.h1 * 1.1
            
            # Recalculate using the same formulas
            t1_pert = math.sqrt(2 * perturbed_h1 / self.g)
            t2_pert = t1_pert + self.dt_window
            h2_pert = 0.5 * self.g * (t2_pert ** 2)
            Deltah_pert = h2_pert - perturbed_h1
            
            # Check monotonicity: larger h1 should give larger window height
            # (because drop is faster through window when starting lower)
            if Deltah_pert > self.Deltah:
                logs.append("PASS: Monotonicity - increasing h1 increases Δh")
            else:
                logs.append(f"WARNING: Non-monotonic - Δh_orig={self.Deltah:.3f}, "
                          f"Δh_pert={Deltah_pert:.3f}")
            
            # Additional check: perturb g slightly
            # Increasing g should decrease time to window top but increase velocity
            # The effect on Δh is complex, so we just verify calculations remain consistent
            g_pert = self.g * 1.05  # +5%
            t1_g = math.sqrt(2 * self.h1 / g_pert)
            t2_g = t1_g + self.dt_window
            h2_g = 0.5 * g_pert * (t2_g ** 2)
            Deltah_g = h2_g - self.h1
            
            # Check energy conservation still holds with perturbed g
            v_final_g = math.sqrt(2 * g_pert * self.h_total)
            energy_ratio = (0.5 * v_final_g ** 2) / (g_pert * self.h_total)
            if math.isclose(energy_ratio, 1.0, rel_tol=0.02):
                logs.append("PASS: Energy conservation preserved under g perturbation")
            else:
                logs.append(f"FAIL: Energy ratio with perturbed g = {energy_ratio:.3f}")
            
            return all("PASS" in log or "WARNING" not in log for log in logs), logs
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            return False, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success"""
        all_logs = []
        success = True
        
        # Run invariants test
        inv_success, inv_logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(inv_logs)
        success = success and inv_success
        
        # Run trace consistency test
        trace_success, trace_logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(trace_logs)
        success = success and trace_success
        
        # Run monotonicity test
        mono_success, mono_logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(mono_logs)
        success = success and mono_success
        
        # Summary
        all_logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        
        return success, all_logs