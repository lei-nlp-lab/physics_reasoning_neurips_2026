import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        """
        self.solution = solution_instance
        
        # Wave speed is given in problem statement (0.5 m/s)
        self.v = 0.5  # m/s
        
        # Safely extract results using getattr with fallback to results dict
        self.results = {}
        for key in ['λ', 'T', 'S', 'A']:
            # Try direct attribute, then results dict, then default
            value = getattr(solution_instance, key, None)
            if value is None:
                value = getattr(solution_instance, 'results', {}).get(key, 0.0)
            self.results[key] = float(value) if value is not None else 0.0
        
        # Extract specific values with units conversion
        self.A = self.results.get('A', 0.0)  # cm
        self.T = self.results.get('T', 0.0)  # s
        self.lambda_val = self.results.get('λ', 0.0)  # m
        self.S = self.results.get('S', 0.0)  # cm
        
        # Convert A and S to meters for consistency in checks
        self.A_m = self.A / 100.0  # m
        self.S_m = self.S / 100.0  # m
        
        self.logs = []

    def test_invariants(self):
        """
        Test fundamental physics invariants:
        1. Wavelength = Wave Speed × Period
        2. Distance in one period = 4 × Amplitude
        """
        try:
            # Test 1: λ = v * T
            expected_lambda = self.v * self.T
            if not math.isclose(self.lambda_val, expected_lambda, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: λ = v * T check: λ={self.lambda_val:.3f}, v*T={expected_lambda:.3f}")
                return False
            self.logs.append(f"PASS: λ = v * T check: {self.lambda_val:.3f} ≈ {expected_lambda:.3f}")
            
            # Test 2: S = 4A (in one period)
            expected_S_m = 4 * self.A_m
            if not math.isclose(self.S_m, expected_S_m, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: S = 4A check: S={self.S_m:.4f}m, 4A={expected_S_m:.4f}m")
                return False
            self.logs.append(f"PASS: S = 4A check: {self.S_m:.4f}m ≈ {expected_S_m:.4f}m")
            
            # Test 3: Symbolic verification (fallback)
            try:
                λ, v, T = sp.symbols('λ v T')
                eq1 = sp.Eq(λ, v * T)
                # Substitute numerical values
                check1 = eq1.subs({λ: self.lambda_val, v: self.v, T: self.T})
                if not check1:
                    self.logs.append("WARNING: Symbolic λ check failed, but numerical passed")
            except:
                self.logs.append("NOTE: Symbolic λ check skipped due to error")
            
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False

    def test_trace_consistency(self):
        """
        Check consistency between intermediate and final results.
        From steps analysis: A=4cm, T=4s should lead to λ=2m, S=16cm.
        """
        try:
            # Check amplitude from steps analysis (4 cm)
            if not math.isclose(self.A, 4.0, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Amplitude inconsistency: A={self.A}cm ≠ 4cm")
                return False
            self.logs.append(f"PASS: Amplitude A={self.A}cm matches expected 4cm")
            
            # Check period from steps analysis (4 s)
            if not math.isclose(self.T, 4.0, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Period inconsistency: T={self.T}s ≠ 4s")
                return False
            self.logs.append(f"PASS: Period T={self.T}s matches expected 4s")
            
            # Check wavelength from steps analysis (2 m)
            if not math.isclose(self.lambda_val, 2.0, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Wavelength inconsistency: λ={self.lambda_val}m ≠ 2m")
                return False
            self.logs.append(f"PASS: Wavelength λ={self.lambda_val}m matches expected 2m")
            
            # Check distance from steps analysis (16 cm)
            if not math.isclose(self.S, 16.0, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Distance inconsistency: S={self.S}cm ≠ 16cm")
                return False
            self.logs.append(f"PASS: Distance S={self.S}cm matches expected 16cm")
            
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False

    def test_monotonicity(self):
        """
        Perturb inputs and verify monotonic relationships:
        1. If wave speed increases (period constant), wavelength increases.
        2. If amplitude increases, distance in one period increases.
        """
        original_A = self.A
        original_v = self.v
        original_T = self.T
        
        try:
            # Test 1: Perturb wave speed (increase by 10%)
            perturbed_v = original_v * 1.1
            expected_lambda_inc = perturbed_v * original_T
            
            if not expected_lambda_inc > self.lambda_val:
                self.logs.append(f"FAIL: λ monotonicity w.r.t v: {expected_lambda_inc:.3f} ≤ {self.lambda_val:.3f}")
                return False
            self.logs.append(f"PASS: λ increases with v (perturbation test)")
            
            # Test 2: Perturb period (increase by 10%)
            perturbed_T = original_T * 1.1
            expected_lambda_inc2 = original_v * perturbed_T
            
            if not expected_lambda_inc2 > self.lambda_val:
                self.logs.append(f"FAIL: λ monotonicity w.r.t T: {expected_lambda_inc2:.3f} ≤ {self.lambda_val:.3f}")
                return False
            self.logs.append(f"PASS: λ increases with T (perturbation test)")
            
            # Test 3: Perturb amplitude (increase by 10%)
            perturbed_A = original_A * 1.1
            expected_S_inc = 4 * perturbed_A
            
            if not expected_S_inc > self.S:
                self.logs.append(f"FAIL: S monotonicity w.r.t A: {expected_S_inc:.3f} ≤ {self.S:.3f}")
                return False
            self.logs.append(f"PASS: S increases with A (perturbation test)")
            
            # Restore original values
            self.v = original_v
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            self.v = original_v  # Restore on error
            return False

    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} Test ---")
            try:
                if not test_func():
                    success = False
                    self.logs.append(f"{test_name} test failed.")
            except Exception as e:
                success = False
                self.logs.append(f"{test_name} test crashed: {str(e)}")
                self.logs.append(traceback.format_exc())
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution_instance)
# success, logs = checker.run_all_tests()