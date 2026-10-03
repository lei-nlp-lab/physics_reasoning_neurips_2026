import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction from solution_instance
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.n_range = self.results.get('n_range', 0.0)
        self.n_min = self.results.get('n_min', 0.0)
        self.n_max = self.results.get('n_max', 0.0)
        self.theta = self.results.get('theta', 0.0)  # Deflection angle
        self.r_D = self.results.get('r_D', 0.0)  # Refraction angle at D
        self.r_E_incidence = self.results.get('r_E_incidence', 0.0)  # Incidence at E
        self.r_E_reflection = self.results.get('r_E_reflection', 0.0)  # Reflection at E
        self.alpha_F = self.results.get('alpha_F', 0.0)  # Incidence at F
        self.r_F = self.results.get('r_F', 0.0)  # Refraction at F
        
        self.logs = []
    
    def log(self, msg, status="INFO"):
        self.logs.append(f"[{status}] {msg}")
    
    def test_geometry_invariants(self):
        """Check geometric consistency from steps analysis"""
        try:
            # Step 1: r_D should be 30° (given geometry)
            if not math.isclose(self.r_D, 30.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"r_D = {self.r_D}°, expected 30°", "FAIL")
                return False
            
            # Step 2: r_E_incidence = r_D + 30° = 60°
            expected_r_E = self.r_D + 30.0
            if not math.isclose(self.r_E_incidence, expected_r_E, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"r_E_incidence = {self.r_E_incidence}°, expected {expected_r_E}°", "FAIL")
                return False
            
            # Law of reflection at E: incidence = reflection
            if not math.isclose(self.r_E_incidence, self.r_E_reflection, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Reflection at E violated: {self.r_E_incidence}° ≠ {self.r_E_reflection}°", "FAIL")
                return False
            
            # Step 3: α_F = 30°
            if not math.isclose(self.alpha_F, 30.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"α_F = {self.alpha_F}°, expected 30°", "FAIL")
                return False
            
            # Step 5: Deflection angle θ = 60°
            if not math.isclose(self.theta, 60.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"θ = {self.theta}°, expected 60°", "FAIL")
                return False
            
            self.log("Geometry invariants satisfied", "PASS")
            return True
        except Exception as e:
            self.log(f"Geometry test failed with error: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_snell_law_consistency(self):
        """Verify Snell's law at refraction points"""
        try:
            # Check that r_F is consistent with Snell's Law at F for SOME valid n
            # The solution may compute r_F using n_min, n_max, or any n in range
            # We'll verify that sin(r_F) = n * sin(alpha_F) for some n in [n_min, n_max)
            
            if self.n_min <= 0 or self.n_max <= 0:
                self.log("Skipping Snell's law consistency (missing n_min/n_max)", "WARN")
                return True
            
            # Compute what n would give the reported r_F
            # From Snell at F: n = sin(r_F) / sin(alpha_F)
            if abs(math.sin(math.radians(self.alpha_F))) < 1e-10:
                self.log("alpha_F is 0°, skipping Snell check", "WARN")
                return True
            
            n_from_r_F = math.sin(math.radians(self.r_F)) / math.sin(math.radians(self.alpha_F))
            
            # Check if this n is within the valid range [n_min, n_max)
            # Allow small tolerance for floating point
            if n_from_r_F >= self.n_min - 1e-2 and n_from_r_F <= self.n_max + 1e-2:
                self.log(f"Snell's law at F consistent: n={n_from_r_F:.4f} in range [{self.n_min:.4f}, {self.n_max:.4f})", "PASS")
                return True
            else:
                self.log(f"Snell's law at F violated: n={n_from_r_F:.4f} not in range [{self.n_min:.4f}, {self.n_max:.4f})", "FAIL")
                return False
                
        except Exception as e:
            self.log(f"Snell's law test failed: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_total_internal_reflection(self):
        """Verify TIR conditions at points E and F"""
        try:
            # At point E: TIR occurs, so incidence (60°) > critical angle θ_c
            # θ_c = arcsin(1/n) for prism→air interface
            # Condition: 60° > arcsin(1/n) => sin(60°) > 1/n => n > 1/sin(60°) = 2/√3 ≈ 1.1547
            expected_n_min = 2 / math.sqrt(3)  # ≈ 1.1547
            
            if not math.isclose(self.n_min, expected_n_min, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"n_min = {self.n_min}, expected {expected_n_min:.4f}", "FAIL")
                return False
            
            # At point F: No TIR (light exits), so incidence (30°) < critical angle
            # Condition: 30° < arcsin(1/n) => sin(30°) < 1/n => n < 1/sin(30°) = 2
            expected_n_max = 2.0
            
            if not math.isclose(self.n_max, expected_n_max, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"n_max = {self.n_max}, expected {expected_n_max}", "FAIL")
                return False
            
            # Range should be n_min ≤ n < n_max
            if not (self.n_min <= self.n_max - 1e-2):  # Allow small tolerance
                self.log(f"Invalid range: n_min={self.n_min}, n_max={self.n_max}", "FAIL")
                return False
            
            self.log("TIR conditions satisfied", "PASS")
            return True
        except Exception as e:
            self.log(f"TIR test failed: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check that the computed deflection angle matches expected 60°
            if not math.isclose(self.theta, 60.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Final θ={self.theta}° doesn't match target 60°", "FAIL")
                return False
            
            # Check that n_range is computed correctly as n_max - n_min
            expected_n_range = self.n_max - self.n_min
            if not math.isclose(self.n_range, expected_n_range, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"n_range = {self.n_range}, expected {expected_n_range}", "FAIL")
                return False
            
            # Check that n_min and n_max satisfy the inequality
            if not (self.n_min >= 2/math.sqrt(3) - 1e-2 and self.n_max <= 2.0 + 1e-2):
                self.log(f"Bounds out of range: n_min={self.n_min}, n_max={self.n_max}", "FAIL")
                return False
            
            self.log("Trace consistency verified", "PASS")
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """Perturb refractive index and check physical trends"""
        try:
            if self.n_min <= 0 or self.n_max <= 0:
                self.log("Skipping monotonicity test (invalid bounds)", "WARN")
                return True
            
            # Test 1: Increasing n makes TIR at E easier (critical angle decreases)
            # Critical angle θ_c = arcsin(1/n)
            n_test1 = self.n_min * 1.1  # Slightly above n_min
            n_test2 = self.n_min * 0.9  # Slightly below n_min
            
            # Avoid division by zero
            if abs(n_test1) < 1e-10 or abs(n_test2) < 1e-10:
                self.log("n values too small for monotonicity test", "WARN")
                return True
                
            θ_c1 = math.degrees(math.asin(1 / n_test1))
            θ_c2 = math.degrees(math.asin(1 / n_test2))
            
            # Higher n → smaller critical angle
            if not θ_c1 < θ_c2:
                self.log(f"Monotonicity violated: θ_c({n_test1})={θ_c1}°, θ_c({n_test2})={θ_c2}°", "FAIL")
                return False
            
            # Test 2: At point F, higher n makes TIR more likely (critical angle decreases)
            # But we require no TIR at F, so n must stay below 2
            # Check that just below n_max, incidence (30°) < θ_c, just above, incidence > θ_c
            n_below = self.n_max - 0.01
            n_above = self.n_max + 0.01
            
            if abs(n_below) < 1e-10 or abs(n_above) < 1e-10:
                self.log("n values near boundary too small", "WARN")
                return True
                
            θ_c_below = math.degrees(math.asin(1 / n_below))
            θ_c_above = math.degrees(math.asin(1 / n_above))
            
            # At n_below: 30° should be < θ_c
            # At n_above: 30° should be > θ_c (would cause TIR at F)
            if not (30.0 < θ_c_below and 30.0 > θ_c_above):
                self.log(f"Boundary condition at n_max inconsistent", "FAIL")
                return False
            
            self.log("Monotonicity trends verified", "PASS")
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed: {traceback.format_exc()}", "ERROR")
            return False
    
    def run_all_tests(self):
        """Execute all checks and return success status and logs"""
        self.logs = []
        
        tests = [
            ("Geometry Invariants", self.test_geometry_invariants),
            ("Snell's Law Consistency", self.test_snell_law_consistency),
            ("Total Internal Reflection", self.test_total_internal_reflection),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} failed", "FAIL")
            except Exception as e:
                success = False
                self.log(f"{test_name} crashed: {traceback.format_exc()}", "ERROR")
        
        return success, self.logs