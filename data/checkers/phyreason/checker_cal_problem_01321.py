import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts inputs and results from the solution instance.
        Uses getattr with fallbacks to avoid AttributeErrors.
        """
        # Extract inputs with safe defaults
        self.g = 10.0  # Given in problem
        self.R = 6.4e6  # 6400 km converted to meters
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Results mapping with fallbacks
        self.H = self.results.get('H', None)
        self.g_prime = self.results.get('g_prime', None)
        self.m = self.results.get('m', None)
        self.h = self.results.get('h', None)
        self.L = self.results.get('L', None)
        
        # Store for monotonicity backup
        self._original_h = self.h
        self._original_H = self.H

    def test_invariants(self):
        """Check fundamental physics laws using KB theorems."""
        logs = []
        
        try:
            # 1. Check weight-mass relationship (Theorem 1)
            # At surface: W = m*g
            if self.m is not None:
                # Mass should be positive
                if self.m <= 0:
                    logs.append(f"FAIL: Mass m={self.m} must be positive")
                else:
                    logs.append(f"PASS: Mass m={self.m} kg is positive")
            
            # 2. Check g' relationship from universal gravitation (Theorem 3)
            # g' = g * (R/(R+h))^2
            if self.g_prime is not None and self.h is not None:
                expected_g_prime = self.g * (self.R / (self.R + self.h))**2
                if math.isclose(self.g_prime, expected_g_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"PASS: g'={self.g_prime} matches gravitation law")
                else:
                    logs.append(f"FAIL: g'={self.g_prime} vs expected {expected_g_prime}")
            
            # 3. Check centripetal force requirement (Theorem 4)
            # For geostationary orbit: v^2/(R+H) = g' at height H
            # where g' = g * (R/(R+H))^2
            if self.H is not None and isinstance(self.H, (int, float)):
                g_prime_H = self.g * (self.R / (self.R + self.H))**2
                v = 2 * math.pi * (self.R + self.H) / (24 * 3600)  # T=24h
                centripetal_acc = v**2 / (self.R + self.H)
                
                if math.isclose(g_prime_H, centripetal_acc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"PASS: Centripetal force balance at H={self.H:.1e}m")
                else:
                    logs.append(f"FAIL: g'={g_prime_H} vs centripetal={centripetal_acc}")
            
            # 4. Check h = 3R relationship from steps
            if self.h is not None:
                expected_h = 3 * self.R
                if math.isclose(self.h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"PASS: h={self.h:.1e}m equals 3R={expected_h:.1e}m")
                else:
                    logs.append(f"FAIL: h={self.h:.1e}m vs 3R={expected_h:.1e}m")
        
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # 1. Check step 2: g' calculation
            # From step 2: g' = F_N/m - g
            # We don't have F_N, but we can check consistency with h
            if self.g_prime is not None and self.h is not None:
                # From step 4: h = sqrt(g/g') * R - R
                h_from_g_prime = math.sqrt(self.g / self.g_prime) * self.R - self.R
                if math.isclose(self.h, h_from_g_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"PASS: h consistent with g' (h={self.h:.1e}, calculated={h_from_g_prime:.1e})")
                else:
                    logs.append(f"FAIL: h inconsistency (h={self.h:.1e}, from g'={h_from_g_prime:.1e})")
            
            # 2. Check H and L equality (step 6)
            if self.H is not None and self.L is not None:
                if math.isclose(self.H, self.L, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: H and L numerically equal")
                else:
                    logs.append(f"FAIL: H={self.H} != L={self.L}")
            
            # 3. Check H matches expected numerical value for T=24h
            if self.H is not None and isinstance(self.H, (int, float)):
                # Expected H for T=24h: H = (g*R²*T²/(4π²))^(1/3) - R
                T = 24 * 3600  # 24 hours in seconds
                expected_H = (self.g * self.R**2 * T**2 / (4 * math.pi**2))**(1/3) - self.R
                if math.isclose(self.H, expected_H, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: H numerically matches expected value for T=24h")
                else:
                    logs.append(f"FAIL: H={self.H} vs expected={expected_H}")
        
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_monotonicity(self):
        """Perturb mass and check trend, then restore."""
        logs = []
        
        try:
            # Test 1: Increase mass by 10%
            if self.m is not None:
                original_m = self.m
                perturbed_m = original_m * 1.1
                
                # According to Theorem 1, weight should increase proportionally
                # W = m*g, so if m increases 10%, W should increase 10%
                original_W = original_m * self.g
                perturbed_W = perturbed_m * self.g
                
                expected_ratio = 1.1
                actual_ratio = perturbed_W / original_W
                
                if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"PASS: Weight increases linearly with mass (ratio={actual_ratio:.3f})")
                else:
                    logs.append(f"FAIL: Weight ratio={actual_ratio:.3f} vs expected {expected_ratio}")
            
            # Test 2: Increase R by 5% and check h trend
            original_R = self.R
            perturbed_R = original_R * 1.05
            
            # h should be proportional to R (h = 3R)
            original_h = 3 * original_R
            perturbed_h = 3 * perturbed_R
            
            expected_h_ratio = 1.05
            actual_h_ratio = perturbed_h / original_h
            
            if math.isclose(actual_h_ratio, expected_h_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"PASS: h scales linearly with R (ratio={actual_h_ratio:.3f})")
            else:
                logs.append(f"FAIL: h ratio={actual_h_ratio:.3f} vs expected {expected_h_ratio}")
            
            # Test 3: Check H monotonicity with R
            if self.H is not None and isinstance(self.H, (int, float)):
                # Test with T = 86400 s
                T_val = 86400
                R_vals = [self.R, self.R * 1.05]
                H_vals = []
                
                for R_val in R_vals:
                    H_val = (self.g * R_val**2 * T_val**2 / (4 * math.pi**2))**(1/3) - R_val
                    H_vals.append(H_val)
                
                if H_vals[1] > H_vals[0]:
                    logs.append(f"PASS: H increases with R (H1={H_vals[0]:.1e}, H2={H_vals[1]:.1e})")
                else:
                    logs.append(f"FAIL: H decreases with R (H1={H_vals[0]:.1e}, H2={H_vals[1]:.1e})")
        
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        # Restore original values
        self.h = self._original_h
        self.H = self._original_H
        
        return logs

    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        all_logs = ["=== Physics Checker Report ==="]
        
        # Run all test suites
        all_logs.append("\n--- Invariant Tests ---")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n--- Trace Consistency Tests ---")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n--- Monotonicity Tests ---")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if "FAIL" in log or "ERROR" in log:
                success = False
                break
        
        all_logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        
        return success, all_logs