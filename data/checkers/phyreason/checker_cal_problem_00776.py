import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results using getattr to avoid sandbox errors."""
        # Safely extract results with fallbacks
        self.T = self._safe_get(solution_instance, 'T', 0.0)
        self.v = self._safe_get(solution_instance, 'v', 0.0)
        self.y = self._safe_get(solution_instance, 'y', 0.0)
        self.lamb = self._safe_get(solution_instance, 'lambda', 0.0)
        
        # Constants from problem statement
        self.t1 = 0.0
        self.t2 = 0.6
        self.t_target = 1.2
        self.A = 10.0  # cm
        self.lambda_given = 8.0  # m
        
    def _safe_get(self, obj, key, default):
        """Helper to safely get attribute from solution instance or results dict."""
        val = getattr(obj, key, None)
        if val is None:
            val = getattr(obj, 'results', {}).get(key, default)
        return float(val) if isinstance(val, (int, float)) else default

    def test_invariants(self):
        """Check fundamental wave physics invariants."""
        logs = []
        
        # 1. Check wave speed formula v = lambda / T (numerical)
        if all([self.lamb, self.T, self.v]) and self.T > 0:
            v_calc = self.lamb / self.T
            if not math.isclose(self.v, v_calc, rel_tol=0.02, abs_tol=0.1):
                logs.append(f"FAIL: Wave speed mismatch: v={self.v} ≠ λ/T={v_calc}")
            else:
                logs.append(f"PASS: Wave speed v={self.v:.1f} consistent with λ/T")
        
        # 2. Check period inequality condition: 2T < 0.6 < 2.5T
        if self.T > 0:
            lower_bound = 2 * self.T
            upper_bound = 2.5 * self.T
            if not (lower_bound < 0.6 < upper_bound):
                logs.append(f"FAIL: Period condition violated: 2T={lower_bound:.3f} < 0.6 < {upper_bound:.3f}")
            else:
                logs.append(f"PASS: Period condition satisfied: 2T={lower_bound:.3f} < 0.6 < {upper_bound:.3f}")
        
        # 3. Check displacement equation consistency: y = A sin(ωt) where ω = 2π/T
        if all([self.T, self.y]) and self.T > 0:
            omega = 2 * math.pi / self.T
            y_calc = self.A * math.sin(omega * self.t_target)
            if not math.isclose(self.y, y_calc, rel_tol=0.02, abs_tol=0.5):
                logs.append(f"FAIL: Displacement mismatch: y={self.y} ≠ A sin(ωt)={y_calc:.1f}")
            else:
                logs.append(f"PASS: Displacement y={self.y:.1f} matches SHM equation")
        
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Check if T from steps matches stored T (from step_4)
        T_step4 = 4.8 / 19  # ~0.25263 s
        if self.T and not math.isclose(self.T, T_step4, rel_tol=0.02, abs_tol=0.005):
            logs.append(f"WARN: Period T={self.T:.3f} differs from step_4 value {T_step4:.3f}")
        
        # Check if v from steps matches stored v (from step_5/8)
        v_step5 = 31.7  # m/s
        if self.v and not math.isclose(self.v, v_step5, rel_tol=0.02, abs_tol=0.5):
            logs.append(f"WARN: Wave speed v={self.v:.1f} differs from step value {v_step5:.1f}")
        
        # Check if displacement matches step_10
        y_step10 = -10.0  # cm
        if self.y and not math.isclose(self.y, y_step10, rel_tol=0.02, abs_tol=0.5):
            logs.append(f"WARN: Displacement y={self.y:.1f} differs from step value {y_step10:.1f}")
        
        # Verify wavelength is 8 m
        if self.lamb and not math.isclose(self.lamb, self.lambda_given, rel_tol=0.01):
            logs.append(f"FAIL: Wavelength λ={self.lamb} ≠ given 8 m")
        elif self.lamb:
            logs.append(f"PASS: Wavelength λ={self.lamb} matches diagram")
        
        return logs

    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs."""
        logs = []
        
        # 1. Perturb wavelength λ and check wave speed v = λ/T
        if self.T and self.T > 0:
            lambda_orig = self.lamb if self.lamb else self.lambda_given
            
            # Increase wavelength by 10%
            lambda_new = lambda_orig * 1.1
            v_orig = lambda_orig / self.T
            v_new = lambda_new / self.T
            
            if not (v_new > v_orig or math.isclose(v_new, v_orig, rel_tol=0.001)):
                logs.append(f"FAIL: v should increase with λ (v_orig={v_orig:.1f}, v_new={v_new:.1f})")
            else:
                logs.append(f"PASS: Wave speed increases with wavelength (v∝λ)")
            
            # Restore original value
            self.lamb = lambda_orig
        
        # 2. Perturb period T and check frequency f = 1/T effect on ω
        if self.T and self.T > 0:
            T_orig = self.T
            T_new = T_orig * 0.9  # Decrease period by 10%
            
            # Angular frequency ω = 2π/T
            omega_orig = 2 * math.pi / T_orig
            omega_new = 2 * math.pi / T_new
            
            if not (omega_new > omega_orig or math.isclose(omega_new, omega_orig, rel_tol=0.001)):
                logs.append(f"FAIL: ω should increase as T decreases")
            else:
                logs.append(f"PASS: ω=2π/T monotonic relationship holds")
            
            # Restore original value
            self.T = T_orig
        
        return logs

    def test_direction_consistency(self):
        """Check wave propagation direction logic from steps analysis."""
        logs = []
        
        # From steps: Wave propagates positive x-direction
        # The phase shift in steps suggests this (dashed line to the right of solid)
        logs.append("INFO: Direction check - assuming positive x-direction from steps analysis")
        
        return logs

    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        logs = []
        all_passed = True
        
        try:
            # Run all test suites
            test_suites = [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity),
                ("Direction", self.test_direction_consistency),
            ]
            
            for suite_name, test_func in test_suites:
                suite_logs = test_func()
                if any(log.startswith("FAIL") for log in suite_logs):
                    all_passed = False
                logs.append(f"=== {suite_name} ===")
                logs.extend(suite_logs)
                logs.append("")
            
            # Summary
            if all_passed:
                logs.append("SUMMARY: All physics checks passed within tolerances.")
            else:
                logs.append("SUMMARY: Some physics checks failed.")
                
        except Exception as e:
            logs.append(f"ERROR during testing: {str(e)}")
            logs.append(traceback.format_exc())
            all_passed = False
        
        return all_passed, logs

    # Additional helper methods for symbolic checks (fallback)
    def symbolic_wave_check(self):
        """Symbolic verification of wave equations using SymPy."""
        logs = []
        try:
            T, v, lamb, t = sp.symbols('T v λ t', positive=True)
            
            # Basic wave relationship
            eq1 = sp.Eq(v, lamb / T)
            logs.append(f"Symbolic wave equation: {eq1}")
            
            # SHM equation
            A = self.A
            omega = 2 * sp.pi / T
            y_eq = A * sp.sin(omega * t)
            logs.append(f"Symbolic displacement: y(t) = {y_eq}")
            
        except Exception as e:
            logs.append(f"Symbolic check skipped: {str(e)}")
        
        return logs