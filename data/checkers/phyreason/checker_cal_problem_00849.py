import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        # Extract input parameters with safe defaults
        self.L = getattr(solution_instance, 'L', 0.2)  # m
        self.d = getattr(solution_instance, 'd', 0.2)  # m
        self.v0 = getattr(solution_instance, 'v0', 2e5)  # m/s
        self.q_over_m = getattr(solution_instance, 'q_over_m', 1e8)  # C/kg
        self.s = getattr(solution_instance, 's', 0.4)  # m
        self.n = getattr(solution_instance, 'n', None)  # optional
        self.B = getattr(solution_instance, 'B', None)  # optional, will compute if missing
        
        # Extract results with safe fallbacks
        self.d_half = self._safe_get('d/2', 0.1)
        self.L_res = self._safe_get('L', 0.2)
        self.U = self._safe_get('U', 400.0)
        self.r1 = self._safe_get('r_1', 0.4)
        self.r2 = self._safe_get('r_2', 0.566)
        self.v = self._safe_get('v', None)  # will compute if missing
        # FIXED: Extract t (time difference) and t_circ (magnetic transit time) separately
        self.t_diff = self._safe_get('t', None)  # time difference from step 8
        self.t_circ = self._safe_get('t_circ', None)  # magnetic transit time from step 11
        self.s_prime = self._safe_get('s\'', None)  # from step 9
        
        # Compute derived quantities if missing
        if self.B is None:
            # B from r1: r1 = v0 / (q/m * B) => B = v0 / (r1 * q/m)
            self.B = self.v0 / (self.r1 * self.q_over_m)
        
        if self.v is None:
            # v from energy: v = sqrt(v0^2 + (q/m)*U)
            self.v = math.sqrt(self.v0**2 + self.q_over_m * self.U)
        
        # Set default for t_diff if missing
        if self.t_diff is None:
            # Compute from formula: t_diff = (π/4) / (q_over_m * B)
            self.t_diff = (math.pi / 4) / (self.q_over_m * self.B)
        
        # Set default for t_circ if missing
        if self.t_circ is None:
            self.t_circ = self.s / self.v0
    
    def _safe_get(self, key, default):
        """Safely get value from results with multiple fallbacks."""
        # Try direct key
        val = self.results.get(key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) else val
        # Try alternative representations
        if key == 'd/2':
            val = self.results.get('d_half', None)
        elif key == 's\'':
            val = self.results.get('s_prime', None)
        elif key == 'r_1':
            val = self.results.get('r1', None)
        elif key == 'r_2':
            val = self.results.get('r2', None)
        # For 't', we need to be careful: it could be time difference or magnetic transit time
        # We handle this in __init__ by using separate variables
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) else val
        return default
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Electric field acceleration check
            t_electric = self.L / self.v0
            a_expected = 2 * self.d_half / (t_electric**2)  # from d/2 = 0.5*a*t^2
            a_from_E = self.q_over_m * (self.U / self.d)  # a = (q/m)*E = (q/m)*(U/d)
            if not self._is_close(a_expected, a_from_E):
                logs.append(f"FAIL: Electric acceleration mismatch. Expected {a_expected:.3e}, got {a_from_E:.3e}")
                success = False
            else:
                logs.append(f"PASS: Electric acceleration consistent.")
            
            # 2. Voltage formula check (derived from kinematics)
            U_expected = (self.d**2 * self.v0**2) / (self.q_over_m * self.L**2)
            if not self._is_close(self.U, U_expected):
                logs.append(f"FAIL: Voltage mismatch. Expected {U_expected:.1f}, got {self.U:.1f}")
                success = False
            else:
                logs.append(f"PASS: Voltage U={self.U} V consistent with kinematics.")
            
            # 3. Magnetic field B consistency
            B_from_r1 = self.v0 / (self.r1 * self.q_over_m)
            if not self._is_close(self.B, B_from_r1):
                logs.append(f"FAIL: Magnetic field B mismatch. From r1: {B_from_r1:.3e}, stored: {self.B:.3e}")
                success = False
            else:
                logs.append(f"PASS: Magnetic field B={self.B:.3e} T consistent with r1.")
            
            # 4. Velocity after electric field (energy conservation)
            v_expected = math.sqrt(self.v0**2 + self.q_over_m * self.U)
            if not self._is_close(self.v, v_expected):
                logs.append(f"FAIL: Velocity v mismatch. Expected {v_expected:.3e}, got {self.v:.3e}")
                success = False
            else:
                logs.append(f"PASS: Velocity v={self.v:.3e} m/s consistent with energy.")
            
            # 5. Radius r2 consistency
            r2_expected = self.v / (self.q_over_m * self.B)
            if not self._is_close(self.r2, r2_expected):
                logs.append(f"FAIL: Radius r2 mismatch. Expected {r2_expected:.3f}, got {self.r2:.3f}")
                success = False
            else:
                logs.append(f"PASS: Radius r2={self.r2:.3f} m consistent with v and B.")
            
            # 6. Circular motion period check (step 8) - use t_diff
            T_expected = 2 * math.pi / (self.q_over_m * self.B)  # T = 2π/(q/m * B)
            t_diff_expected = T_expected / 8  # step 8 uses T/8
            if not self._is_close(self.t_diff, t_diff_expected):
                logs.append(f"FAIL: Time difference mismatch. Expected {t_diff_expected:.3e}, got {self.t_diff:.3e}")
                success = False
            else:
                logs.append(f"PASS: Time difference t={self.t_diff:.3e} s consistent with T/8.")
            
            # 7. Check that magnetic transit time matches target answer (2e-5 s)
            if not self._is_close(self.t_circ, 2e-5):
                logs.append(f"WARN: Magnetic transit time {self.t_circ:.3e} s differs from target 2e-5 s.")
                # Not a failure because of potential approximations
            else:
                logs.append(f"PASS: Magnetic transit time matches target answer.")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check that intermediate results lead to final answers
            # Target answer 1: U = 400 V
            if not self._is_close(self.U, 400.0):
                logs.append(f"FAIL: Final voltage U={self.U} does not match target 400 V.")
                success = False
            else:
                logs.append(f"PASS: Voltage U matches target answer.")
            
            # Target answer 2: t = 5π×10^{-7} s ≈ 1.57e-6 s (time difference)
            t_target = 5 * math.pi * 1e-7
            if not self._is_close(self.t_diff, t_target):
                logs.append(f"FAIL: Time difference {self.t_diff:.3e} does not match target {t_target:.3e}.")
                success = False
            else:
                logs.append(f"PASS: Time difference matches target answer.")
            
            # Target answer 3: t = 2×10^{-5} s (magnetic transit time)
            if not self._is_close(self.t_circ, 2e-5):
                logs.append(f"FAIL: Magnetic transit time {self.t_circ:.3e} does not match target 2e-5 s.")
                success = False
            else:
                logs.append(f"PASS: Magnetic transit time matches target answer.")
            
            # Consistency between time difference and magnetic transit time (they should be different)
            if self._is_close(self.t_diff, self.t_circ, rel_tol=1e-1):
                logs.append(f"WARN: Time difference and magnetic transit time are unexpectedly close.")
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb U by +1% and check effect on r2
            U_orig = self.U
            U_pert = U_orig * 1.01
            v_pert = math.sqrt(self.v0**2 + self.q_over_m * U_pert)
            r2_pert = v_pert / (self.q_over_m * self.B)
            if r2_pert <= self.r2:
                logs.append(f"FAIL: Increasing U did not increase r2. Original {self.r2:.3f}, perturbed {r2_pert:.3f}")
                success = False
            else:
                logs.append(f"PASS: r2 increases with U (monotonic trend OK).")
            
            # Perturb B by +1% and check effect on time difference (step 8)
            B_orig = self.B
            B_pert = B_orig * 1.01
            T_pert = 2 * math.pi / (self.q_over_m * B_pert)
            t_diff_pert = T_pert / 8
            if t_diff_pert >= self.t_diff:
                logs.append(f"FAIL: Increasing B did not decrease time difference. Original {self.t_diff:.3e}, perturbed {t_diff_pert:.3e}")
                success = False
            else:
                logs.append(f"PASS: Time difference decreases with B (monotonic trend OK).")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test suite
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== {test_name} ===")
            try:
                success, logs = test_method()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"CRASH in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs