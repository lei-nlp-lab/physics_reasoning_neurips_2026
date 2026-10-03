import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants
        self.E = 2.0  # V/m
        self.v0 = 1.0  # m/s
        self.theta_deg = 30.0  # degrees
        self.theta_rad = math.radians(self.theta_deg)  # radians
        
        # Use EXACT q_over_m value: sqrt(3) * 1e4
        # Compute it exactly using high precision
        self.q_over_m = math.sqrt(3) * 1e4
        
        # Safely extract results with fallback - use 0.0 if not present
        self.B = float(self.results.get('B', 0.0))
        self.t1 = float(self.results.get('t_1', 0.0))
        self.t2 = float(self.results.get('t_2', 0.0))
        self.r = float(self.results.get('r', 0.0))
        self.v_y = float(self.results.get('v_y', 0.0))
        self.t = float(self.results.get('t', 0.0))  # should equal t1
        
        # Derived quantities for checks
        self.d = self.v0 * self.t1  # width from electric field
        self.m = 1.0  # arbitrary mass (q_over_m is fixed, so m cancels in ratios)
        self.q = self.q_over_m * self.m  # corresponding charge

    def test_invariants(self):
        logs = []
        try:
            # 1. Electric field kinematics: v_y = v0 * tan(theta)
            v_y_expected = self.v0 * math.tan(self.theta_rad)
            if not math.isclose(self.v_y, v_y_expected, rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: v_y mismatch. Expected {v_y_expected:.6f}, got {self.v_y:.6f}")
            else:
                logs.append(f"PASS: v_y = v0 * tan(theta) holds.")
            
            # 2. Newton's second law in electric field: v_y = (qE/m) * t1
            # Use high precision calculation
            v_y_newton = self.q_over_m * self.E * self.t1
            # Be more tolerant for floating-point rounding errors
            if not math.isclose(self.v_y, v_y_newton, rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: v_y from Newton's law mismatch. Expected {v_y_newton:.6f}, got {self.v_y:.6f}")
            else:
                logs.append(f"PASS: v_y = (qE/m) * t1 holds.")
            
            # 3. Lorentz force in magnetic field: q*v0*B = m*v0^2 / r  ->  r = v0 / (q_over_m * B)
            if self.B != 0.0:
                r_expected = self.v0 / (self.q_over_m * self.B)
                if not math.isclose(self.r, r_expected, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"FAIL: r mismatch from Lorentz force. Expected {r_expected:.6f}, got {self.r:.6f}")
                else:
                    logs.append(f"PASS: r = v0 / (q/m * B) holds.")
            else:
                logs.append("SKIP: B=0, cannot check Lorentz force relation.")
            
            # 4. Geometry in magnetic field: r = d / sin(theta)
            if self.d != 0.0 and self.r != 0.0:
                r_geom = self.d / math.sin(self.theta_rad)
                if not math.isclose(self.r, r_geom, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"FAIL: r mismatch from geometry. Expected {r_geom:.6f}, got {self.r:.6f}")
                else:
                    logs.append(f"PASS: r = d / sin(theta) holds.")
            else:
                logs.append("SKIP: d=0 or r=0, cannot check geometry relation.")
            
            # 5. Derived formula for B: B = (E * cos(theta)) / v0
            B_expected = self.E * math.cos(self.theta_rad) / self.v0
            if self.B != 0.0 or B_expected != 0.0:
                if not math.isclose(self.B, B_expected, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"FAIL: B mismatch. Expected {B_expected:.6f}, got {self.B:.6f}")
                else:
                    logs.append(f"PASS: B = (E * cos(theta)) / v0 holds.")
            else:
                logs.append("SKIP: B and B_expected are both 0.")
            
            # 6. Time in electric field: t1 = (v0 * tan(theta)) / (q_over_m * E)
            t1_expected = (self.v0 * math.tan(self.theta_rad)) / (self.q_over_m * self.E)
            # Use more tolerant comparison for small numbers
            if not math.isclose(self.t1, t1_expected, rel_tol=1e-2, abs_tol=1e-9):
                logs.append(f"FAIL: t1 mismatch. Expected {t1_expected:.6e}, got {self.t1:.6e}")
            else:
                logs.append(f"PASS: t1 = (v0 * tan(theta)) / (q/m * E) holds.")
            
            # 7. Time in magnetic field: t2 = (m * theta) / (q * B) = theta / (q_over_m * B)
            if self.B != 0.0:
                t2_expected = self.theta_rad / (self.q_over_m * self.B)
                if not math.isclose(self.t2, t2_expected, rel_tol=1e-2, abs_tol=1e-9):
                    logs.append(f"FAIL: t2 mismatch. Expected {t2_expected:.6e}, got {self.t2:.6e}")
                else:
                    logs.append(f"PASS: t2 = theta / (q/m * B) holds.")
            else:
                logs.append("SKIP: B=0, cannot compute t2.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # 1. Check that t (from step 3) equals t1 (from step 7)
            if not math.isclose(self.t, self.t1, rel_tol=1e-2, abs_tol=1e-9):
                logs.append(f"FAIL: t and t1 mismatch. t={self.t:.6e}, t1={self.t1:.6e}")
            else:
                logs.append("PASS: t (step 3) equals t1 (step 7).")
            
            # 2. Check that v_y from step 1 and step 2 are consistent via t1
            v_y_step1 = self.v0 * math.tan(self.theta_rad)
            v_y_step2 = self.q_over_m * self.E * self.t1
            # Be more tolerant for floating-point rounding
            if not math.isclose(v_y_step1, v_y_step2, rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: v_y from step1 and step2 inconsistent. Step1: {v_y_step1:.6f}, Step2: {v_y_step2:.6f}")
            else:
                logs.append("PASS: v_y from step 1 and step 2 are consistent.")
            
            # 3. Check that d computed from electric field (v0*t1) matches d from magnetic geometry (r*sin(theta))
            d_electric = self.v0 * self.t1
            d_magnetic = self.r * math.sin(self.theta_rad)
            if d_electric != 0.0 or d_magnetic != 0.0:
                if not math.isclose(d_electric, d_magnetic, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"FAIL: d mismatch. Electric: {d_electric:.6f}, Magnetic: {d_magnetic:.6f}")
                else:
                    logs.append("PASS: d (width) is consistent between electric and magnetic cases.")
            else:
                logs.append("SKIP: d_electric and d_magnetic are both 0.")
            
            # 4. Check that r from Lorentz force matches r from geometry
            if self.B != 0.0 and self.d != 0.0:
                r_lorentz = self.v0 / (self.q_over_m * self.B)
                r_geom = self.d / math.sin(self.theta_rad)
                if not math.isclose(r_lorentz, r_geom, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"FAIL: r from Lorentz vs geometry. Lorentz: {r_lorentz:.6f}, Geom: {r_geom:.6f}")
                else:
                    logs.append("PASS: r from Lorentz force and geometry agree.")
            else:
                logs.append("SKIP: B=0 or d=0, cannot compare r from Lorentz and geometry.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        try:
            # Perturb electric field E by +1%
            E_original = self.E
            self.E *= 1.01
            
            # Recalculate B using formula B = (E * cos(theta)) / v0
            B_new = self.E * math.cos(self.theta_rad) / self.v0
            expected_B_ratio = 1.01  # B should increase by 1%
            
            if self.B != 0.0:
                actual_B_ratio = B_new / self.B
                if not math.isclose(actual_B_ratio, expected_B_ratio, rel_tol=1e-2):
                    logs.append(f"FAIL: Monotonicity for E->B. Expected ratio {expected_B_ratio:.4f}, got {actual_B_ratio:.4f}")
                else:
                    logs.append("PASS: B increases proportionally with E (as per formula).")
            else:
                logs.append("SKIP: B=0, cannot check monotonicity for E->B.")
            
            # Restore E
            self.E = E_original
            
            # Perturb charge-to-mass ratio by +1%
            q_over_m_original = self.q_over_m
            self.q_over_m *= 1.01
            self.q = self.q_over_m * self.m
            
            # Recalculate t1: t1 = (v0 * tan(theta)) / (q_over_m * E)
            t1_new = (self.v0 * math.tan(self.theta_rad)) / (self.q_over_m * self.E)
            expected_t1_ratio = 1 / 1.01  # t1 should decrease by ~1%
            
            if self.t1 != 0.0:
                actual_t1_ratio = t1_new / self.t1
                # Be more tolerant for floating-point rounding
                if not math.isclose(actual_t1_ratio, expected_t1_ratio, rel_tol=1e-2):
                    logs.append(f"FAIL: Monotonicity for q/m->t1. Expected ratio {expected_t1_ratio:.4f}, got {actual_t1_ratio:.4f}")
                else:
                    logs.append("PASS: t1 decreases as q/m increases (inverse proportionality).")
            else:
                logs.append("SKIP: t1=0, cannot check monotonicity for q/m->t1.")
            
            # Restore q_over_m
            self.q_over_m = q_over_m_original
            self.q = self.q_over_m * self.m
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
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
            if any(log.startswith("FAIL") or log.startswith("ERROR") for log in logs):
                success = False
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            all_logs.append("")
        
        return success, all_logs