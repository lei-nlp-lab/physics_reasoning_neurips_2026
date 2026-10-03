import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (with safe extraction)
        self.P_ground_per_area = 1400.0  # W/m² (1.4 kW/m²)
        self.visible_fraction = 0.45
        
        # Constants - use more precise values to match typical calculations
        self.h = 6.62607015e-34       # Planck's constant (J·s) - CODATA 2018
        self.c = 299792458.0          # Speed of light (m/s)
        self.lambda_visible = 550e-9  # Typical visible wavelength (550 nm)
        self.R_sun_earth = 1.496e11   # Earth-Sun distance (m)
        self.r_earth = 6.371e6        # Earth radius (m)
        
        # Extract results with fallbacks
        self.n = self._safe_get('n', 0.0)           # photons/s/m²
        self.N = self._safe_get('N', 0.0)           # photons/s
        self.P_ground_total = self._safe_get('P_ground', 0.0)  # kW
        
        # For monotonicity backup
        self._original_n = self.n
        self._original_N = self.N
        self._original_P_ground_total = self.P_ground_total
        
    def _safe_get(self, key, default):
        """Safely get value from results with multiple fallbacks."""
        value = self.results.get(key, default)
        if value == default and hasattr(self, key):
            value = getattr(self, key, default)
        return float(value) if isinstance(value, (int, float)) else default
    
    def test_invariants(self):
        """Verify physical invariants using KB relationships."""
        logs = []
        success = True
        
        try:
            # Test 1: Photon energy relationship (numerical)
            # E_photon = h * c / λ
            E_num = self.h * self.c / self.lambda_visible
            n_calculated = (self.visible_fraction * self.P_ground_per_area) / E_num
            
            if not math.isclose(self.n, n_calculated, rel_tol=0.02, abs_tol=1e18):
                logs.append(f"FAIL: Photon flux n mismatch. Got {self.n:.2e}, expected {n_calculated:.2e}")
                success = False
            else:
                logs.append(f"PASS: Photon flux n consistent within 2%")
            
            # Test 2: Power conservation - simplified correct relationship
            # The fundamental relationship is: P_visible_at_earth = n * E_photon
            # where P_visible_at_earth = P_ground_per_area * visible_fraction
            P_visible_calculated = self.n * E_num
            P_visible_expected = self.P_ground_per_area * self.visible_fraction
            
            if not math.isclose(P_visible_calculated, P_visible_expected, rel_tol=0.02, abs_tol=1e-10):
                logs.append(f"FAIL: Power per m² mismatch. Got {P_visible_calculated:.2e} W/m², expected {P_visible_expected:.2e} W/m²")
                success = False
            else:
                logs.append(f"PASS: Power per square meter consistent")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        logs = []
        success = True
        
        try:
            # Step 1 to Step 2 consistency: N = n * 4πR²
            N_calculated = self.n * 4 * math.pi * self.R_sun_earth**2
            
            if not math.isclose(self.N, N_calculated, rel_tol=0.02, abs_tol=1e42):
                logs.append(f"FAIL: N inconsistent with n. Got {self.N:.2e}, expected {N_calculated:.2e}")
                success = False
            else:
                logs.append(f"PASS: N consistent with photon flux n")
            
            # Step 3: P_ground = P * πr² (converting units)
            # Note: P_ground_total is in kW, P_ground_per_area is in W/m²
            P_calculated_kW = (self.P_ground_per_area / 1000) * math.pi * self.r_earth**2
            
            if not math.isclose(self.P_ground_total, P_calculated_kW, rel_tol=0.02, abs_tol=1e12):
                logs.append(f"FAIL: P_ground mismatch. Got {self.P_ground_total:.2e}, expected {P_calculated_kW:.2e}")
                success = False
            else:
                logs.append(f"PASS: Total power P_ground consistent")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input and check output trends."""
        logs = []
        success = True
        
        try:
            # Backup original
            original_P = self.P_ground_per_area
            
            # Perturb: increase power density by 10%
            perturb_factor = 1.10
            self.P_ground_per_area *= perturb_factor
            
            # Recalculate using formulas
            E_photon = self.h * self.c / self.lambda_visible
            n_new = (self.visible_fraction * self.P_ground_per_area) / E_photon
            N_new = n_new * 4 * math.pi * self.R_sun_earth**2
            P_ground_new = (self.P_ground_per_area / 1000) * math.pi * self.r_earth**2
            
            # Check monotonic relationships from KB
            # 1. n should increase proportionally to P
            if not math.isclose(n_new / self._original_n, perturb_factor, rel_tol=0.01):
                logs.append(f"FAIL: Photon flux n not monotonic with P. Ratio: {n_new/self._original_n:.3f}")
                success = False
            
            # 2. N should increase proportionally to n (and thus P)
            if not math.isclose(N_new / self._original_N, perturb_factor, rel_tol=0.01):
                logs.append(f"FAIL: Total photons N not monotonic. Ratio: {N_new/self._original_N:.3f}")
                success = False
            
            # 3. P_ground should increase proportionally to P
            if not math.isclose(P_ground_new / self._original_P_ground_total, perturb_factor, rel_tol=0.01):
                logs.append(f"FAIL: Total power not monotonic. Ratio: {P_ground_new/self._original_P_ground_total:.3f}")
                success = False
            
            if success:
                logs.append(f"PASS: All quantities show correct monotonic behavior")
            
            # Restore original
            self.P_ground_per_area = original_P
            self.n = self._original_n
            self.N = self._original_N
            self.P_ground_total = self._original_P_ground_total
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
            # Attempt restoration on error
            self.P_ground_per_area = original_P if 'original_P' in locals() else 1400.0
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs

# Example usage guard
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'n': 1.75e21,
                'N': 4.9e44,
                'P_ground': 1.8e14
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    print(f"Overall success: {success}")
    for log in logs:
        print(log)