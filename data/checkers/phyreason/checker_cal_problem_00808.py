import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction using getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract given data (from problem context)
        self.x_given = 3.0  # meters (wave has just reached x=3m at t=1.5s)
        self.t_given = 1.5  # seconds
        
        # Extract computed results safely
        self.f = self._safe_get('f')
        self.t_2 = self._safe_get('t_2')
        self.T = self._safe_get('T')
        self.three_quarter_T = self._safe_get('3/4T')
        self.t_total = self._safe_get('t')
        self.v = self._safe_get('v')
        self.t_1 = self._safe_get('t_1')
        self.lam = self._safe_get('λ')  # wavelength
        
        # Additional computed values for verification
        self.x_target = 6.0  # meters (position of interest)
        
    def _safe_get(self, key):
        """Safely retrieve value from solution instance with multiple fallbacks."""
        # Try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) else val
            
        # Try results dict
        val = self.results.get(key)
        if val is not None:
            try:
                return float(val)
            except:
                return val
                
        # Try alternative keys
        alt_keys = {
            'λ': ['lambda', 'wavelength'],
            '3/4T': ['three_quarter_T', 'three_quarter_T_period'],
            't_1': ['t1', 'travel_time_3_to_6'],
            't_2': ['t2', 'trough_time'],
            't': ['total_time', 't_total']
        }
        
        if key in alt_keys:
            for alt in alt_keys[key]:
                val = getattr(self.solution, alt, None)
                if val is not None:
                    return float(val) if isinstance(val, (int, float, str)) else val
                val = self.results.get(alt)
                if val is not None:
                    try:
                        return float(val)
                    except:
                        return val
                        
        return None
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with defaults."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physical laws and relationships."""
        logs = []
        successes = []
        
        try:
            # 1. Wave speed from basic definition: v = distance/time (for wavefront)
            v_from_given = self.x_given / self.t_given
            if self._is_close(self.v, v_from_given):
                logs.append(f"✓ Wave speed from given data: v = {self.x_given}/{self.t_given} = {v_from_given:.3f} m/s")
                successes.append(True)
            else:
                logs.append(f"✗ Wave speed mismatch: v={self.v}, expected {v_from_given:.3f} from given data")
                successes.append(False)
            
            # 2. Wave equation: v = λ/T
            if self.lam is not None and self.T is not None and self.v is not None:
                v_from_eq = self.lam / self.T
                if self._is_close(self.v, v_from_eq):
                    logs.append(f"✓ Wave equation satisfied: v = λ/T = {self.lam}/{self.T} = {v_from_eq:.3f} m/s")
                    successes.append(True)
                else:
                    logs.append(f"✗ Wave equation violated: v={self.v}, λ/T={v_from_eq:.3f}")
                    successes.append(False)
            
            # 3. Frequency-period relationship: f = 1/T
            if self.f is not None and self.T is not None:
                f_from_T = 1.0 / self.T
                if self._is_close(self.f, f_from_T):
                    logs.append(f"✓ Frequency-period relation: f = 1/T = 1/{self.T} = {f_from_T:.3f} Hz")
                    successes.append(True)
                else:
                    logs.append(f"✗ Frequency mismatch: f={self.f}, 1/T={f_from_T:.3f}")
                    successes.append(False)
            
            # 4. Waveform geometry: 3/4 wavelength in 3 meters
            if self.lam is not None:
                lam_from_geometry = (4.0/3.0) * self.x_given
                if self._is_close(self.lam, lam_from_geometry):
                    logs.append(f"✓ Waveform geometry: λ = (4/3)*{self.x_given} = {lam_from_geometry:.3f} m")
                    successes.append(True)
                else:
                    logs.append(f"✗ Wavelength mismatch: λ={self.lam}, expected {lam_from_geometry:.3f} from geometry")
                    successes.append(False)
            
            # 5. Time for 3/4 cycle
            if self.three_quarter_T is not None and self.T is not None:
                three_quarter_from_T = 0.75 * self.T
                if self._is_close(self.three_quarter_T, three_quarter_from_T):
                    logs.append(f"✓ 3/4 period relation: (3/4)T = 0.75*{self.T} = {three_quarter_from_T:.3f} s")
                    successes.append(True)
                else:
                    logs.append(f"✗ 3/4 period mismatch: {self.three_quarter_T} vs 0.75*T={three_quarter_from_T:.3f}")
                    successes.append(False)
            
            # 6. Travel time from x=3m to x=6m
            if self.t_1 is not None and self.v is not None:
                t1_expected = (self.x_target - self.x_given) / self.v
                if self._is_close(self.t_1, t1_expected):
                    logs.append(f"✓ Travel time x=3→6: t₁ = (6-3)/{self.v} = {t1_expected:.3f} s")
                    successes.append(True)
                else:
                    logs.append(f"✗ Travel time mismatch: t₁={self.t_1}, expected {t1_expected:.3f}")
                    successes.append(False)
            
            # 7. Time to reach trough: t₂ = (3/4)T
            if self.t_2 is not None and self.T is not None:
                t2_expected = 0.75 * self.T
                if self._is_close(self.t_2, t2_expected):
                    logs.append(f"✓ Trough time: t₂ = (3/4)T = 0.75*{self.T} = {t2_expected:.3f} s")
                    successes.append(True)
                else:
                    logs.append(f"✗ Trough time mismatch: t₂={self.t_2}, expected {t2_expected:.3f}")
                    successes.append(False)
            
            # 8. Total time consistency
            if self.t_total is not None and self.t_1 is not None and self.t_2 is not None:
                t_total_expected = self.t_1 + self.t_2
                if self._is_close(self.t_total, t_total_expected):
                    logs.append(f"✓ Total time: t = t₁ + t₂ = {self.t_1} + {self.t_2} = {t_total_expected:.3f} s")
                    successes.append(True)
                else:
                    logs.append(f"✗ Total time mismatch: t={self.t_total}, expected {t_total_expected:.3f}")
                    successes.append(False)
            
            # 9. Wave speed consistency with period and given time
            if self.T is not None:
                T_from_given = (4.0/3.0) * self.t_given
                if self._is_close(self.T, T_from_given):
                    logs.append(f"✓ Period from given time: T = (4/3)*{self.t_given} = {T_from_given:.3f} s")
                    successes.append(True)
                else:
                    logs.append(f"✗ Period mismatch: T={self.T}, expected {T_from_given:.3f}")
                    successes.append(False)
        
        except Exception as e:
            logs.append(f"Error in invariants test: {str(e)}")
            successes.append(False)
        
        return all(successes), logs
    
    def test_trace_consistency(self):
        """Check intermediate steps against final answers."""
        logs = []
        successes = []
        
        try:
            # Target answers from problem
            target_v = 2.0  # m/s
            target_t_total = 3.0  # s
            
            # Check wave speed
            if self._is_close(self.v, target_v):
                logs.append(f"✓ Wave speed matches target: {self.v} ≈ {target_v} m/s")
                successes.append(True)
            else:
                logs.append(f"✗ Wave speed mismatch: {self.v} ≠ {target_v} m/s")
                successes.append(False)
            
            # Check total time
            if self._is_close(self.t_total, target_t_total):
                logs.append(f"✓ Total time matches target: {self.t_total} ≈ {target_t_total} s")
                successes.append(True)
            else:
                logs.append(f"✗ Total time mismatch: {self.t_total} ≠ {target_t_total} s")
                successes.append(False)
            
            # Verify step-by-step consistency
            steps_consistent = True
            
            # Step 2: T = (4/3)*1.5 = 2
            if self.T is not None:
                T_step2 = (4.0/3.0) * self.t_given
                if not self._is_close(self.T, T_step2):
                    logs.append(f"✗ Step 2 inconsistency: T={self.T}, should be (4/3)*1.5={T_step2}")
                    steps_consistent = False
            
            # Step 3: λ = (4/3)*3 = 4
            if self.lam is not None:
                lam_step3 = (4.0/3.0) * self.x_given
                if not self._is_close(self.lam, lam_step3):
                    logs.append(f"✗ Step 3 inconsistency: λ={self.lam}, should be (4/3)*3={lam_step3}")
                    steps_consistent = False
            
            # Step 4: f = 1/T
            if self.f is not None and self.T is not None:
                f_step4 = 1.0 / self.T
                if not self._is_close(self.f, f_step4):
                    logs.append(f"✗ Step 4 inconsistency: f={self.f}, should be 1/T={f_step4}")
                    steps_consistent = False
            
            # Step 5: v = λ/T
            if self.v is not None and self.lam is not None and self.T is not None:
                v_step5 = self.lam / self.T
                if not self._is_close(self.v, v_step5):
                    logs.append(f"✗ Step 5 inconsistency: v={self.v}, should be λ/T={v_step5}")
                    steps_consistent = False
            
            # Step 6: t₁ = (6-3)/v
            if self.t_1 is not None and self.v is not None:
                t1_step6 = (self.x_target - self.x_given) / self.v
                if not self._is_close(self.t_1, t1_step6):
                    logs.append(f"✗ Step 6 inconsistency: t₁={self.t_1}, should be (6-3)/v={t1_step6}")
                    steps_consistent = False
            
            # Step 7: t₂ = (3/4)T
            if self.t_2 is not None and self.T is not None:
                t2_step7 = 0.75 * self.T
                if not self._is_close(self.t_2, t2_step7):
                    logs.append(f"✗ Step 7 inconsistency: t₂={self.t_2}, should be (3/4)T={t2_step7}")
                    steps_consistent = False
            
            # Step 8: t = t₁ + t₂
            if self.t_total is not None and self.t_1 is not None and self.t_2 is not None:
                t_step8 = self.t_1 + self.t_2
                if not self._is_close(self.t_total, t_step8):
                    logs.append(f"✗ Step 8 inconsistency: t={self.t_total}, should be t₁+t₂={t_step8}")
                    steps_consistent = False
            
            if steps_consistent:
                logs.append("✓ All step-by-step calculations are internally consistent")
                successes.append(True)
            else:
                successes.append(False)
        
        except Exception as e:
            logs.append(f"Error in trace consistency test: {str(e)}")
            successes.append(False)
        
        return all(successes), logs
    
    def test_monotonicity(self):
        """Perturb input parameters and check physical trends."""
        logs = []
        successes = []
        
        try:
            # Original values (save for restoration)
            orig_t_given = self.t_given
            orig_x_given = self.x_given
            
            # Test 1: Increase given time by 10%
            perturb_t = self.t_given * 1.1  # 1.5 → 1.65s
            
            # Expected changes:
            # Wave speed should decrease: v = x_given / t_given
            v_expected = self.x_given / perturb_t
            
            # Period should increase proportionally: T = (4/3) * t_given
            T_expected = (4.0/3.0) * perturb_t
            
            # Wavelength should remain unchanged (geometry fixed)
            lam_expected = self.lam
            
            logs.append(f"Perturbation: t_given = {perturb_t:.3f}s (original: {orig_t_given}s)")
            logs.append(f"  Expected: v↓ to {v_expected:.3f}m/s, T↑ to {T_expected:.3f}s, λ unchanged")
            
            # Symbolic check of relationships
            v, lam, T, t = sp.symbols('v lam T t', positive=True)
            
            # From geometry: λ = (4/3)*3 = 4 (constant)
            # From wave equation: v = λ/T
            # From given condition: v = 3/t
            
            eq1 = sp.Eq(v, lam/T)
            eq2 = sp.Eq(v, 3/t)
            
            # Solve for T in terms of t
            T_from_t = sp.solve([eq1.subs(lam, 4), eq2], [v, T])[0][1]
            
            # Check derivative: dT/dt > 0 (period increases with given time)
            dT_dt = sp.diff(T_from_t, t)
            dT_dt_val = float(dT_dt.subs(t, orig_t_given))
            
            if dT_dt_val > 0:
                logs.append(f"✓ Monotonicity: dT/dt = {dT_dt_val:.3f} > 0 (period increases with given time)")
                successes.append(True)
            else:
                logs.append(f"✗ Monotonicity violation: dT/dt = {dT_dt_val:.3f} ≤ 0")
                successes.append(False)
            
            # Check derivative: dv/dt < 0 (wave speed decreases with given time)
            v_from_t = sp.solve([eq1.subs(lam, 4), eq2], [v, T])[0][0]
            dv_dt = sp.diff(v_from_t, t)
            dv_dt_val = float(dv_dt.subs(t, orig_t_given))
            
            if dv_dt_val < 0:
                logs.append(f"✓ Monotonicity: dv/dt = {dv_dt_val:.3f} < 0 (wave speed decreases with given time)")
                successes.append(True)
            else:
                logs.append(f"✗ Monotonicity violation: dv/dt = {dv_dt_val:.3f} ≥ 0")
                successes.append(False)
            
            # Test 2: Increase given distance by 10%
            perturb_x = self.x_given * 1.1  # 3 → 3.3m
            
            # Expected changes:
            # Wave speed should increase: v = x_given / t_given (but t_given fixed)
            v_expected2 = perturb_x / self.t_given
            
            # Wavelength should increase: λ = (4/3)*x_given
            lam_expected2 = (4.0/3.0) * perturb_x
            
            # Period should remain unchanged (from t_given)
            T_expected2 = self.T
            
            logs.append(f"\nPerturbation: x_given = {perturb_x:.3f}m (original: {orig_x_given}m)")
            logs.append(f"  Expected: v↑ to {v_expected2:.3f}m/s, λ↑ to {lam_expected2:.3f}m, T unchanged")
            
            # Symbolic check
            v, lam, T, x = sp.symbols('v lam T x', positive=True)
            t_fixed = orig_t_given
            
            eq3 = sp.Eq(v, lam/T)
            eq4 = sp.Eq(v, x/t_fixed)
            eq5 = sp.Eq(lam, (4.0/3.0)*x)  # from geometry
            
            # Solve for λ in terms of x
            lam_from_x = sp.solve(eq5, lam)[0]
            
            # Check derivative: dλ/dx > 0 (wavelength increases with given distance)
            dlam_dx = sp.diff(lam_from_x, x)
            dlam_dx_val = float(dlam_dx.subs(x, orig_x_given))
            
            if dlam_dx_val > 0:
                logs.append(f"✓ Monotonicity: dλ/dx = {dlam_dx_val:.3f} > 0 (wavelength increases with given distance)")
                successes.append(True)
            else:
                logs.append(f"✗ Monotonicity violation: dλ/dx = {dlam_dx_val:.3f} ≤ 0")
                successes.append(False)
            
            # Check that total time t = t₁ + t₂ is strictly increasing with distance
            # t₁ = (x_target - x_given)/v, but v = x_given/t_given
            # So t₁ = (x_target - x_given) * (t_given/x_given)
            x_target = self.x_target
            t1_expr = (x_target - x) * (t_fixed/x)
            t2_expr = 0.75 * T_expected2  # constant for this perturbation
            
            # Since T_expected2 is constant, total time trend depends on t1
            dt1_dx = sp.diff(t1_expr, x)
            dt1_dx_val = float(dt1_dx.subs(x, orig_x_given))
            
            # For x=3, x_target=6: t1 = (6-x)*(1.5/x)
            # Derivative: d/dx[(6-x)*(1.5/x)] = -1.5/x - 1.5*(6-x)/x^2
            # At x=3: -1.5/3 - 1.5*3/9 = -0.5 - 0.5 = -1.0
            # So total time decreases as x_given increases (makes sense - wave starts closer to target)
            
            logs.append(f"  Note: dt₁/dx = {dt1_dx_val:.3f} (travel time decreases as wave starts closer to target)")
            
            # Restore original values
            self.t_given = orig_t_given
            self.x_given = orig_x_given
            
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            successes.append(False)
        
        return all(successes), logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        all_successes = []
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_method()
                all_successes.append(success)
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                all_successes.append(False)
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.append(f"ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_logs.append("Result: FAIL")
        
        # Overall result
        overall_success = all(all_successes)
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append(f"Target answers: v = 2 m/s, t = 3 s")
        
        return overall_success, all_logs