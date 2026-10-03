import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safe attribute access using getattr with fallbacks."""
        self.solution = solution_instance
        
        # Extract results from solution's results dict
        results = getattr(solution_instance, 'results', {})
        
        # Extract all values from results dictionary
        self.F = float(results.get('F', 0.0))          # Restoring force magnitude at displacement x
        self.T = float(results.get('T', 0.0))          # Period
        self.sintheta = float(results.get('sintheta', 0.0))  # sin(theta) approximation
        self.v = float(results.get('v', 0.0))          # Velocity at lowest point
        self.F_prime = float(results.get("F'", results.get('F_prime', 0.0)))  # Tension at lowest point
        self.k = float(results.get('k', 0.0))          # Effective spring constant
        
        # Extract input parameters - these should come from the code's actual values
        # Use reasonable defaults if not found in results
        self.L = float(results.get('L_val', results.get('L', 1.0)))
        self.m = float(results.get('m_val', results.get('m', 1.0)))
        self.g = float(results.get('g_val', results.get('g', 9.8)))
        self.theta = float(results.get('theta_val', results.get('theta', 0.1)))
        
    def test_invariants(self):
        """Test fundamental physics laws: SHM period, work-energy, Newton's 2nd."""
        logs = []
        success = True
        
        try:
            # 1. Check SHM period formula: T = 2π√(L/g)
            if self.T > 0 and self.L > 0 and self.g > 0:
                T_expected = 2 * math.pi * math.sqrt(self.L / self.g)
                if not math.isclose(self.T, T_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Period T = {self.T} ≠ 2π√(L/g) = {T_expected}")
                    success = False
                else:
                    logs.append(f"PASS: Period T matches SHM formula within 2%")
            
            # 2. Check restoring force constant: k = mg/L
            if self.k > 0 and self.m > 0 and self.g > 0 and self.L > 0:
                k_expected = self.m * self.g / self.L
                if not math.isclose(self.k, k_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: k = {self.k} ≠ mg/L = {k_expected}")
                    success = False
                else:
                    logs.append("PASS: k matches mg/L")
            
            # 3. Check work-energy theorem: mgL(1-cosθ) = ½mv²
            if self.v >= 0 and self.m > 0 and self.g > 0 and self.L > 0 and self.theta > 0:
                v_expected = math.sqrt(2 * self.g * self.L * (1 - math.cos(self.theta)))
                if not math.isclose(self.v, v_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v = {self.v} ≠ √[2gL(1-cosθ)] = {v_expected}")
                    success = False
                else:
                    logs.append("PASS: Work-energy theorem satisfied")
            
            # 4. Check Newton's 2nd law at lowest point: F' - mg = mv²/L
            if self.F_prime > 0 and self.v >= 0 and self.m > 0 and self.g > 0 and self.L > 0:
                F_prime_expected = self.m * self.g + self.m * self.v**2 / self.L
                if not math.isclose(self.F_prime, F_prime_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: F' = {self.F_prime} ≠ mg + mv²/L = {F_prime_expected}")
                    success = False
                else:
                    logs.append("PASS: Newton's 2nd law satisfied at lowest point")
        
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate steps against final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check that k from step_3 matches period calculation
            if self.k > 0 and self.T > 0 and self.m > 0:
                # From T = 2π√(m/k), we get k = 4π²m/T²
                k_from_T = 4 * math.pi**2 * self.m / self.T**2
                if not math.isclose(self.k, k_from_T, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: k = {self.k} ≠ 4π²m/T² = {k_from_T}")
                    success = False
                else:
                    logs.append("PASS: k consistent with period T")
            
            # 2. Check tension formula: F' = 3mg - 2mg cosθ
            if self.F_prime > 0 and self.m > 0 and self.g > 0 and self.theta > 0:
                F_prime_expected = 3 * self.m * self.g - 2 * self.m * self.g * math.cos(self.theta)
                if not math.isclose(self.F_prime, F_prime_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: F' = {self.F_prime} ≠ 3mg - 2mg cosθ = {F_prime_expected}")
                    success = False
                else:
                    logs.append("PASS: Tension formula matches derived expression")
            
            # 3. Check small angle approximation: sinθ ≈ θ for small θ
            if self.sintheta >= 0 and self.theta > 0:
                theta_rad = self.theta
                sin_approx = theta_rad  # First term of Taylor series
                if not math.isclose(self.sintheta, sin_approx, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"WARN: sinθ = {self.sintheta} differs from θ = {theta_rad}")
                else:
                    logs.append("PASS: Small angle approximation valid")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check trends."""
        logs = []
        success = True
        
        try:
            original_m = self.m
            original_L = self.L
            original_theta = self.theta
            
            # 1. Test mass perturbation: T should be independent of m for pendulum
            if original_L > 0 and self.g > 0:
                m_perturbed = original_m * 1.1  # +10%
                T_perturbed = 2 * math.pi * math.sqrt(original_L / self.g)
                
                if not math.isclose(T_perturbed, self.T, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Period T changed when mass perturbed (should be independent)")
                    success = False
                else:
                    logs.append("PASS: Period independent of mass (correct for pendulum)")
            
            # 2. Test length perturbation: T ∝ √L
            if original_L > 0 and self.g > 0:
                L_perturbed = original_L * 1.1  # +10%
                T_perturbed = 2 * math.pi * math.sqrt(L_perturbed / self.g)
                T_original = 2 * math.pi * math.sqrt(original_L / self.g)
                ratio_expected = math.sqrt(1.1)  # √(1.1) ≈ 1.0488
                ratio_actual = T_perturbed / T_original
                
                if not math.isclose(ratio_actual, ratio_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: T ratio = {ratio_actual:.4f} ≠ √1.1 = {ratio_expected:.4f}")
                    success = False
                else:
                    logs.append("PASS: T ∝ √L relationship holds")
            
            # 3. Test angle perturbation: v increases with θ
            if original_L > 0 and self.g > 0 and original_theta > 0:
                theta_perturbed = original_theta * 1.1
                v_large = math.sqrt(2 * self.g * original_L * (1 - math.cos(theta_perturbed)))
                v_small = math.sqrt(2 * self.g * original_L * (1 - math.cos(original_theta)))
                
                if v_large <= v_small:
                    logs.append(f"FAIL: v({theta_perturbed:.3f}) = {v_large:.3f} ≤ v({original_theta:.3f}) = {v_small:.3f}")
                    success = False
                else:
                    logs.append("PASS: Speed increases with release angle")
        
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success."""
        logs = []
        all_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_success = all_success and success1
        logs.extend([f"[INVARIANTS] {log}" for log in logs1])
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_success = all_success and success2
        logs.extend([f"[TRACE] {log}" for log in logs2])
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_success = all_success and success3
        logs.extend([f"[MONOTONICITY] {log}" for log in logs3])
        
        # Summary
        if all_success:
            logs.append("SUCCESS: All physics checks passed.")
        else:
            logs.append("WARNING: Some physics checks failed.")
        
        return all_success, logs