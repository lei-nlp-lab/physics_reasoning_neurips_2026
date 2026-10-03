import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract ONLY the results from the solution
        self.results = getattr(solution_instance, 'results', {})
        
        # Use standard Earth constants for verification (same as typical values used in solution)
        self.R = 6371000.0      # meters
        self.g = 9.8            # m/s²
        self.G = 6.67430e-11    # m³ kg⁻¹ s⁻²
        self.T = 86400.0        # seconds
        
        # Extract results with defaults
        self.M = self.results.get('M', 0.0)
        self.v_1 = self.results.get('v_1', 0.0)
        self.v = self.results.get('v', 0.0)
        self.h = self.results.get('h', 0.0)
        
        self.logs = []

    def test_invariants(self):
        """Verify fundamental physical laws."""
        try:
            # 1. Mass of Earth: M = g * R^2 / G
            expected_M = self.g * self.R**2 / self.G
            if not math.isclose(self.M, expected_M, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: M invariant. Expected {expected_M:.3e}, got {self.M:.3e}")
                return False
            self.logs.append(f"PASS: M = g*R²/G = {expected_M:.3e}")
            
            # 2. First cosmic velocity: v₁ = sqrt(g * R)
            expected_v1 = math.sqrt(self.g * self.R)
            if not math.isclose(self.v_1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v₁ invariant. Expected {expected_v1:.3f}, got {self.v_1:.3f}")
                return False
            self.logs.append(f"PASS: v₁ = √(gR) = {expected_v1:.3f} m/s")
            
            # 3. Orbital velocity: v = sqrt(G * M / (R + h))
            r_orbit = self.R + self.h
            expected_v = math.sqrt(self.G * self.M / r_orbit)
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v orbital invariant. Expected {expected_v:.3f}, got {self.v:.3f}")
                return False
            self.logs.append(f"PASS: v = √(GM/(R+h)) = {expected_v:.3f} m/s")
            
            # 4. Period-velocity relation: v = 2π(R+h)/T
            expected_v_from_T = 2 * math.pi * r_orbit / self.T
            if not math.isclose(self.v, expected_v_from_T, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v from T invariant. Expected {expected_v_from_T:.3f}, got {self.v:.3f}")
                return False
            self.logs.append(f"PASS: v = 2π(R+h)/T = {expected_v_from_T:.3f} m/s")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final altitude."""
        try:
            # From step 5: h = ∛(g R² T²/(4π²)) - R
            numerator = self.g * self.R**2 * self.T**2
            denominator = 4 * math.pi**2
            expected_h = (numerator / denominator) ** (1/3) - self.R
            
            if not math.isclose(self.h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: h trace consistency. Expected {expected_h:.3f}, got {self.h:.3f}")
                return False
            self.logs.append(f"PASS: h = ∛(gR²T²/(4π²)) - R = {expected_h:.3f} m")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb period T and verify altitude trend."""
        try:
            original_T = self.T
            # Increase T by 1%
            perturbed_T = original_T * 1.01
            # Recalculate h using the formula from step 5
            numerator = self.g * self.R**2 * perturbed_T**2
            denominator = 4 * math.pi**2
            perturbed_h = (numerator / denominator) ** (1/3) - self.R
            
            # For a satellite, if period increases, altitude should increase (T ∝ r^(3/2))
            if not perturbed_h > self.h:
                self.logs.append(f"FAIL: monotonicity. T increased by 1%, but h did not increase.")
                self.logs.append(f"  Original h: {self.h:.3f}, Perturbed h: {perturbed_h:.3f}")
                return False
            self.logs.append(f"PASS: h increases with T (as expected from orbital mechanics)")
            self.logs.append(f"  Original h: {self.h:.3f} m, Perturbed h: {perturbed_h:.3f} m")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all checks and return overall success."""
        self.logs.clear()
        success = True
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]
        
        for test in tests:
            if not test():
                success = False
                # Continue with other tests
                
        return success, self.logs