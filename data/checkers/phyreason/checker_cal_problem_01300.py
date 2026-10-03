import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants from problem
        self.g = 10.0  # m/s²
        self.L = 5.0   # rod length (m)
        self.h_cyl = 20.0  # cylinder top below suspension (m)
        self.h_cyl_bot = 25.0  # cylinder bottom below suspension (m)
        
        # Safe extraction of results with fallback
        self.t_A1 = self._safe_get('t_A1', getattr(solution_instance, 't_A1', None))
        self.t_A2 = self._safe_get('t_A2', getattr(solution_instance, 't_A2', None))
        self.t_1 = self._safe_get('t_1', getattr(solution_instance, 't_1', None))
        self.t_2 = self._safe_get('t_2', getattr(solution_instance, 't_2', None))
        self.t_B = self._safe_get('t_B', getattr(solution_instance, 't_B', None))
        
        # Expected analytical values
        self.t_A1_expected = math.sqrt(2 * (self.h_cyl - self.L) / self.g)  # lower end reaches A
        self.t_A2_expected = math.sqrt(2 * self.h_cyl / self.g)            # upper end reaches A
        self.t_B_expected = math.sqrt(2 * self.h_cyl_bot / self.g)         # upper end reaches B
        self.t_1_expected = self.t_A2_expected - self.t_A1_expected
        self.t_2_expected = self.t_B_expected - self.t_A1_expected
    
    def _safe_get(self, key, default=None):
        """Safely get value from results dictionary."""
        if key in self.results:
            val = self.results[key]
            if isinstance(val, str):
                try:
                    return float(val)
                except:
                    return default
            return float(val) if val is not None else default
        return default
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify kinematic consistency using free-fall equations."""
        logs = []
        success = True
        
        # Check free-fall equation: h = 1/2 g t^2
        if self.t_A1 is not None:
            h_A1_calc = 0.5 * self.g * self.t_A1**2
            h_A1_actual = self.h_cyl - self.L  # 15m
            if not self._is_close(h_A1_calc, h_A1_actual):
                logs.append(f"t_A1 violates free-fall: h_calc={h_A1_calc:.3f}, h_expected={h_A1_actual:.3f}")
                success = False
        
        if self.t_A2 is not None:
            h_A2_calc = 0.5 * self.g * self.t_A2**2
            h_A2_actual = self.h_cyl  # 20m
            if not self._is_close(h_A2_calc, h_A2_actual):
                logs.append(f"t_A2 violates free-fall: h_calc={h_A2_calc:.3f}, h_expected={h_A2_actual:.3f}")
                success = False
        
        if self.t_B is not None:
            h_B_calc = 0.5 * self.g * self.t_B**2
            h_B_actual = self.h_cyl_bot  # 25m
            if not self._is_close(h_B_calc, h_B_actual):
                logs.append(f"t_B violates free-fall: h_calc={h_B_calc:.3f}, h_expected={h_B_actual:.3f}")
                success = False
        
        # Check velocity-time relation: v = g*t
        if self.t_A1 is not None and self.t_A2 is not None:
            v_A1 = self.g * self.t_A1
            v_A2 = self.g * self.t_A2
            if v_A2 <= v_A1:  # Should be strictly greater since t_A2 > t_A1
                logs.append(f"Velocity monotonicity violated: v_A1={v_A1:.3f}, v_A2={v_A2:.3f}")
                success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Verify step-by-step consistency with final answers."""
        logs = []
        success = True
        
        # Check intermediate calculations match final answers
        if all(v is not None for v in [self.t_A1, self.t_A2, self.t_1]):
            t_1_calc = self.t_A2 - self.t_A1
            if not self._is_close(self.t_1, t_1_calc):
                logs.append(f"t_1 inconsistency: {self.t_1:.3f} vs calculated {t_1_calc:.3f}")
                success = False
        
        if all(v is not None for v in [self.t_A1, self.t_B, self.t_2]):
            t_2_calc = self.t_B - self.t_A1
            if not self._is_close(self.t_2, t_2_calc):
                logs.append(f"t_2 inconsistency: {self.t_2:.3f} vs calculated {t_2_calc:.3f}")
                success = False
        
        # Compare with expected analytical values
        if self.t_A1 is not None and not self._is_close(self.t_A1, self.t_A1_expected):
            logs.append(f"t_A1 mismatch: {self.t_A1:.3f} vs expected {self.t_A1_expected:.3f}")
            success = False
        
        if self.t_A2 is not None and not self._is_close(self.t_A2, self.t_A2_expected):
            logs.append(f"t_A2 mismatch: {self.t_A2:.3f} vs expected {self.t_A2_expected:.3f}")
            success = False
        
        if self.t_B is not None and not self._is_close(self.t_B, self.t_B_expected):
            logs.append(f"t_B mismatch: {self.t_B:.3f} vs expected {self.t_B_expected:.3f}")
            success = False
        
        if self.t_1 is not None and not self._is_close(self.t_1, self.t_1_expected):
            logs.append(f"t_1 mismatch: {self.t_1:.3f} vs expected {self.t_1_expected:.3f}")
            success = False
        
        if self.t_2 is not None and not self._is_close(self.t_2, self.t_2_expected):
            logs.append(f"t_2 mismatch: {self.t_2:.3f} vs expected {self.t_2_expected:.3f}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb rod length and check physical trend."""
        logs = []
        success = True
        
        # Save original
        original_L = self.L
        
        try:
            # Test 1: Increase rod length by 1m
            self.L = original_L + 1.0
            t_A1_inc = math.sqrt(2 * (self.h_cyl - self.L) / self.g)
            t_1_inc = math.sqrt(2 * self.h_cyl / self.g) - t_A1_inc
            
            # With longer rod, lower end starts closer to A → t_A1 decreases, t_1 increases
            if not (t_A1_inc < self.t_A1_expected):
                logs.append(f"Monotonicity failed: t_A1 should decrease with longer rod")
                success = False
            
            if not (t_1_inc > self.t_1_expected):
                logs.append(f"Monotonicity failed: t_1 should increase with longer rod")
                success = False
            
            # Test 2: Decrease rod length by 1m
            self.L = original_L - 1.0
            t_A1_dec = math.sqrt(2 * (self.h_cyl - self.L) / self.g)
            t_1_dec = math.sqrt(2 * self.h_cyl / self.g) - t_A1_dec
            
            # With shorter rod, lower end starts further from A → t_A1 increases, t_1 decreases
            if not (t_A1_dec > self.t_A1_expected):
                logs.append(f"Monotonicity failed: t_A1 should increase with shorter rod")
                success = False
            
            if not (t_1_dec < self.t_1_expected):
                logs.append(f"Monotonicity failed: t_1 should decrease with shorter rod")
                success = False
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
        finally:
            # Restore original
            self.L = original_L
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all checks and return summary."""
        logs = []
        all_success = True
        
        test_functions = [
            ("Kinematic Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in test_functions:
            try:
                success, test_logs = test_func()
                all_success = all_success and success
                logs.append(f"=== {name} ===")
                if test_logs:
                    logs.extend(test_logs)
                else:
                    logs.append("Passed")
                logs.append("")
            except Exception as e:
                logs.append(f"ERROR in {name}: {traceback.format_exc()}")
                all_success = False
        
        return all_success, logs