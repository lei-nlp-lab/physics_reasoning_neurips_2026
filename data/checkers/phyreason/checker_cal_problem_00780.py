import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Never access solution_instance.attr directly.
        """
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results with safe defaults
        self.R = self._safe_get('R', 1.0)  # Radius, default 1.0 for numeric tests
        self.c = self._safe_get('c', 3.0e8)  # Speed of light, default 3e8 for numeric tests
        
        # Geometry and angles from steps
        # IMPORTANT: The results may store angles in radians or degrees.
        # We'll retrieve them and convert to degrees for internal use.
        i_rad = self._safe_get('i', math.radians(60.0))
        r_rad = self._safe_get('r', math.radians(45.0))
        beta_rad = self._safe_get('beta', math.radians(60.0))
        if 'β' in self.results:
            beta_rad = self._safe_get('β', math.radians(60.0))
        EOC_rad = self._safe_get('∠EOC', math.radians(30.0))
        # Convert to degrees for checker logic
        self.i_deg = math.degrees(i_rad)
        self.r_deg = math.degrees(r_rad)
        self.beta_deg = math.degrees(beta_rad)
        self.EOC_deg = math.degrees(EOC_rad)
        
        # Calculated quantities
        self.n = self._safe_get('n', math.sqrt(6)/2)  # Refractive index
        self.v = self._safe_get('v', 0.0)  # Speed in medium
        self.t = self._safe_get('t', 0.0)  # Total time
        self.EF = self._safe_get('EF', self.R)  # Distance EF
        
        # Derived values for checks
        self.DE = math.sqrt(2) * self.R  # Given DE = √2 R
        self.sin_i = math.sin(math.radians(self.i_deg))
        self.sin_r = math.sin(math.radians(self.r_deg))
        self.sin_beta = math.sin(math.radians(self.beta_deg))
        
        # Expected values for comparison
        self.expected_n = math.sqrt(6) / 2
        self.expected_t = (math.sqrt(3) + 1) * self.R / self.c
    
    def _safe_get(self, key, default):
        """Safely get value from results with fallback."""
        value = self.results.get(key, default)
        # If value is string like "√6/2", try to evaluate
        if isinstance(value, str):
            try:
                # Replace common symbols
                cleaned = value.replace('√', 'sqrt(').replace('R', str(self.R))
                if 'sqrt(' in cleaned and not cleaned.endswith(')'):
                    cleaned += ')'
                return float(eval(cleaned))
            except:
                return default
        return float(value) if isinstance(value, (int, float)) else default
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Snell's Law at point D: n = sin(i)/sin(r)
            n_calculated = self.sin_i / self.sin_r
            if not math.isclose(self.n, n_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Snell's law at D: n = {self.n:.4f}, sin(i)/sin(r) = {n_calculated:.4f}")
                success = False
            else:
                logs.append(f"PASS: Snell's law at D holds: n = {self.n:.4f}")
            
            # 2. Speed relationship: v = c/n
            if self.n == 0:
                logs.append("ERROR: n is zero, cannot compute v = c/n")
                success = False
            else:
                v_expected = self.c / self.n
                if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Speed in medium: v = {self.v:.4e}, expected c/n = {v_expected:.4e}")
                    success = False
                else:
                    logs.append(f"PASS: Speed relationship v = c/n holds")
            
            # 3. Geometry check: ∠EOC should be 30°
            if not math.isclose(self.EOC_deg, 30.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ∠EOC = {self.EOC_deg:.2f}°, expected 30°")
                success = False
            else:
                logs.append(f"PASS: ∠EOC = {self.EOC_deg:.2f}° as expected")
            
            # 4. Snell's Law at point E: n * sin(β) = sin(γ) where γ is refraction angle in air
            # From geometry: ∠EFO = ∠EOC = 30° (triangle EFO is isosceles)
            # But ∠EFO is NOT the refraction angle γ. The refraction angle γ is the angle between
            # the refracted ray EF and the normal at point E.
            # Actually, we don't have γ in the results, so we skip this test.
            # Instead, we check that β = 60° as expected from geometry.
            if not math.isclose(self.beta_deg, 60.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: β = {self.beta_deg:.2f}°, expected 60°")
                success = False
            else:
                logs.append(f"PASS: β = {self.beta_deg:.2f}° as expected")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate steps lead to final answer."""
        logs = []
        success = True
        
        try:
            # 1. Check n matches expected √6/2
            if not math.isclose(self.n, self.expected_n, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: n = {self.n:.4f}, expected √6/2 ≈ {self.expected_n:.4f}")
                success = False
            else:
                logs.append(f"PASS: n = {self.n:.4f} matches √6/2")
            
            # 2. Check time formula: t = DE/v + EF/c
            # Calculate v from n
            if self.n == 0:
                logs.append("ERROR: n is zero, cannot compute v")
                success = False
            else:
                v_calc = self.c / self.n
                t_calc = self.DE / v_calc + self.EF / self.c
                
                if not math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t = {self.t:.4e}, DE/v + EF/c = {t_calc:.4e}")
                    success = False
                else:
                    logs.append(f"PASS: Time formula gives consistent result")
            
            # 3. Check t matches expected (√3+1)R/c
            if not math.isclose(self.t, self.expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t = {self.t:.4e}, expected (√3+1)R/c = {self.expected_t:.4e}")
                success = False
            else:
                logs.append(f"PASS: t = {self.t:.4e} matches (√3+1)R/c")
            
            # 4. Check EF = R from geometry
            if not math.isclose(self.EF, self.R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: EF = {self.EF:.4f}R, expected R")
                success = False
            else:
                logs.append(f"PASS: EF = R as expected")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb refractive index n, recalculate time, check physical trend.
        RESTORE original values after test.
        """
        logs = []
        success = True
        
        try:
            # Save original state
            original_n = self.n
            original_t = self.t
            
            # Guard against zero time
            if original_t == 0.0:
                # Recompute t using the formula to avoid division by zero later
                if original_n == 0:
                    logs.append("ERROR: n is zero, cannot compute t")
                    return False, logs
                original_t = self.DE / (self.c / original_n) + self.EF / self.c
                logs.append(f"NOTE: original_t was 0, recomputed as {original_t:.4e}")
            
            # Perturb n by +10% (makes medium optically denser)
            perturbed_n = original_n * 1.10
            
            # Recalculate speed in medium with perturbed n
            v_perturbed = self.c / perturbed_n
            
            # Recalculate time with perturbed speed
            # Note: Geometry (DE, EF) and angles remain unchanged
            t_perturbed = self.DE / v_perturbed + self.EF / self.c
            
            # Physical expectation: Higher n → slower v → longer travel time in medium
            # Since DE is traveled in medium, total time should increase
            # Use a more reasonable tolerance: just check if t_perturbed > original_t
            if not (t_perturbed > original_t):
                logs.append(f"FAIL: Monotonicity: n↑ from {original_n:.4f} to {perturbed_n:.4f}, t should ↑")
                logs.append(f"      Original t = {original_t:.4e}, Perturbed t = {t_perturbed:.4e}")
                success = False
            else:
                logs.append(f"PASS: Monotonicity: n↑ → t↑ as expected")
                logs.append(f"      Original t = {original_t:.4e}, Perturbed t = {t_perturbed:.4e} (+{100*(t_perturbed/original_t-1):.1f}%)")
            
            # RESTORE original values
            self.n = original_n
            self.t = original_t
            
            # Second test: Perturb R (size of hemisphere)
            original_R = self.R
            perturbed_R = original_R * 1.10
            
            # Recalculate with perturbed R
            # DE scales with R, EF scales with R
            DE_perturbed = math.sqrt(2) * perturbed_R
            EF_perturbed = perturbed_R
            t_R_perturbed = DE_perturbed / (self.c/self.n) + EF_perturbed / self.c
            
            # Physical expectation: Larger R → longer path → longer time
            # Time should scale linearly with R
            expected_ratio = perturbed_R / original_R
            actual_ratio = t_R_perturbed / original_t
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
                logs.append(f"FAIL: R scaling: R↑ from {original_R:.2f} to {perturbed_R:.2f}, t should scale linearly")
                logs.append(f"      Expected t ratio = {expected_ratio:.3f}, Actual = {actual_ratio:.3f}")
                success = False
            else:
                logs.append(f"PASS: R scaling: t ∝ R as expected (ratio = {actual_ratio:.3f})")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall result."""
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("\n1. PHYSICS INVARIANTS:")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n2. TRACE CONSISTENCY:")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n3. MONOTONICITY:")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs