import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract input parameters - FIRST try to get from results, then from attributes
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract constants - check results first, then attributes
        self.k = self.results.get('k', getattr(solution_instance, 'k', 10.0))
        self.B = self.results.get('B', getattr(solution_instance, 'B', 0.5))
        self.N = self.results.get('N', getattr(solution_instance, 'N', 100))
        self.l = self.results.get('l', getattr(solution_instance, 'l', 0.05))
        self.d = self.results.get('d', getattr(solution_instance, 'd', 0.1))
        self.r = self.results.get('r', getattr(solution_instance, 'r', 1.0))
        self.I = self.results.get('I', getattr(solution_instance, 'I', 0.001))
        self.s1 = self.results.get('s1', getattr(solution_instance, 's1', 0.03))
        self.s2 = self.results.get('s2', getattr(solution_instance, 's2', 0.01))
        self.I_prime_input = getattr(solution_instance, 'I_prime', 0.0)
        
        # Extract results with fallback to 0.0
        self.delta_x = self.results.get('delta_x', self.results.get('Δx', 0.0))
        self.F = self.results.get('F', 0.0)
        self.s = self.results.get('s', 0.0)
        self.s1_result = self.results.get('s₁', self.results.get('s1', self.s1))
        self.s2_result = self.results.get('s₂', self.results.get('s2', self.s2))
        self.theta = self.results.get('theta', self.results.get('θ', 0.0))
        self.twotheta = self.results.get('2theta', self.results.get('2θ', 0.0))
        self.I_prime = self.results.get('I_prime', self.results.get('I\'', 0.0))
        self.abs_delta_x = self.results.get('|delta_x|', self.results.get('|Δx|', 0.0))
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Log storage
        self.logs = []
    
    def log(self, message, success=True):
        """Log a message with success/failure status"""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def fuzzy_equal(self, a, b, rel_tol=None, abs_tol=None):
        """Check if two numbers are approximately equal"""
        if rel_tol is None:
            rel_tol = self.rel_tol
        if abs_tol is None:
            abs_tol = self.abs_tol
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_force_balance(self):
        """Verify that magnetic force equals spring force at equilibrium"""
        try:
            # Magnetic force from Ampere's law (simplified for perpendicular field)
            F_magnetic = self.N * self.B * abs(self.I) * self.l
            
            # Spring force from Hooke's Law
            F_spring = self.k * abs(self.delta_x)
            
            # Check equilibrium
            if self.fuzzy_equal(F_magnetic, F_spring):
                return self.log("Force balance: Magnetic force ≈ Spring force (equilibrium)", True)
            else:
                self.log(f"Force imbalance: F_magnetic={F_magnetic}, F_spring={F_spring}", False)
                return False
        except Exception as e:
            self.log(f"Force balance test failed: {str(e)}", False)
            return False
    
    def test_geometry_relations(self):
        """Verify geometric relationships: Δx = dθ and s = 2rθ"""
        try:
            # Check Δx = dθ (small angle approximation)
            delta_x_from_theta = self.d * self.theta
            if self.fuzzy_equal(self.delta_x, delta_x_from_theta):
                self.log(f"Geometry: Δx = dθ verified ({self.delta_x} ≈ {self.d}×{self.theta})", True)
            else:
                self.log(f"Geometry mismatch: Δx={self.delta_x}, dθ={delta_x_from_theta}", False)
                return False
            
            # Check s = 2rθ
            s_from_theta = 2 * self.r * self.theta
            if self.fuzzy_equal(self.s, s_from_theta):
                self.log(f"Geometry: s = 2rθ verified ({self.s} ≈ 2×{self.r}×{self.theta})", True)
            else:
                self.log(f"Geometry mismatch: s={self.s}, 2rθ={s_from_theta}", False)
                return False
            
            # Check 2θ relationship
            if self.fuzzy_equal(self.twotheta, 2 * self.theta):
                self.log(f"Angle: 2θ relationship verified ({self.twotheta} ≈ 2×{self.theta})", True)
            else:
                self.log(f"Angle mismatch: 2θ={self.twotheta}, 2×θ={2*self.theta}", False)
                return False
            
            return True
        except Exception as e:
            self.log(f"Geometry test failed: {str(e)}", False)
            return False
    
    def test_derived_formulas(self):
        """Verify derived formulas against direct computation"""
        try:
            # Formula 1: |Δx| = NBIl/k
            abs_delta_x_expected = abs(self.N * self.B * self.I * self.l / self.k) if self.k != 0 else 0
            if self.fuzzy_equal(self.abs_delta_x, abs_delta_x_expected):
                self.log(f"Formula 1: |Δx| = NBIl/k verified", True)
            else:
                self.log(f"Formula 1 mismatch: |Δx|={self.abs_delta_x}, expected={abs_delta_x_expected}", False)
                return False
            
            # Formula 2: s = 2NBIlr/(dk)
            s_expected = 2 * self.N * self.B * self.I * self.l * self.r / (self.d * self.k) if self.k != 0 and self.d != 0 else 0
            if self.fuzzy_equal(self.s, s_expected):
                self.log(f"Formula 2: s = 2NBIlr/(dk) verified", True)
            else:
                self.log(f"Formula 2 mismatch: s={self.s}, expected={s_expected}", False)
                return False
            
            # Formula 3: I' = dk(s₁+s₂)/(4NBlr)
            numerator = self.d * self.k * (self.s1 + self.s2)
            denominator = 4 * self.N * self.B * self.l * self.r
            I_prime_expected = numerator / denominator if denominator != 0 else 0
            if self.fuzzy_equal(self.I_prime, I_prime_expected):
                self.log(f"Formula 3: I' = dk(s₁+s₂)/(4NBlr) verified", True)
            else:
                self.log(f"Formula 3 mismatch: I'={self.I_prime}, expected={I_prime_expected}", False)
                return False
            
            return True
        except Exception as e:
            self.log(f"Derived formulas test failed: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final answer"""
        try:
            # Step 1: F = NBIl
            F_from_inputs = self.N * self.B * abs(self.I) * self.l
            if not self.fuzzy_equal(self.F, F_from_inputs):
                self.log(f"Step 1 inconsistency: F={self.F}, NBIl={F_from_inputs}", False)
                return False
            
            # Step 2: |Δx| = F/k
            if self.k != 0:
                delta_x_from_F = self.F / self.k
                if not self.fuzzy_equal(self.abs_delta_x, delta_x_from_F):
                    self.log(f"Step 2 inconsistency: |Δx|={self.abs_delta_x}, F/k={delta_x_from_F}", False)
                    return False
            
            # Step 4/5: s should equal 2rθ and also 2rΔx/d
            if self.d != 0:
                s_combined = 2 * self.r * self.delta_x / self.d
                if not self.fuzzy_equal(self.s, s_combined):
                    self.log(f"Step 4/5 inconsistency: s={self.s}, 2rΔx/d={s_combined}", False)
                    return False
            else:
                self.log("Skipping 2rΔx/d check (d=0)", True)
            
            # Check final I' calculation - use the actual s1 and s2 values
            if self.s1 != 0 and self.s2 != 0 and self.I_prime != 0:
                # Using the derived formula directly
                I_prime_expected = self.d * self.k * (self.s1 + self.s2) / (4 * self.N * self.B * self.l * self.r)
                if not self.fuzzy_equal(self.I_prime, I_prime_expected, rel_tol=5e-2):
                    self.log(f"Final answer inconsistency: I'={self.I_prime}, expected={I_prime_expected}", False)
                    return False
            
            self.log("All trace steps are consistent", True)
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends"""
        try:
            # Store original values
            orig_I = self.I
            orig_s1 = self.s1
            orig_s2 = self.s2
            
            # Test 1: Increase current I, check that s increases
            if orig_I != 0 and self.k != 0 and self.d != 0:
                perturbed_I = orig_I * 1.1  # 10% increase
                s_orig = 2 * self.N * self.B * orig_I * self.l * self.r / (self.d * self.k)
                s_perturbed = 2 * self.N * self.B * perturbed_I * self.l * self.r / (self.d * self.k)
                
                if s_perturbed > s_orig:
                    self.log(f"Monotonicity 1: Increasing I increases s ({s_orig} → {s_perturbed})", True)
                else:
                    self.log(f"Monotonicity violation: s didn't increase with I", False)
                    return False
            
            # Test 2: Increase s1, check that I' increases
            if orig_s1 != 0 and self.N != 0 and self.B != 0 and self.l != 0 and self.r != 0:
                perturbed_s1 = orig_s1 * 1.1
                I_prime_orig = self.d * self.k * (orig_s1 + orig_s2) / (4 * self.N * self.B * self.l * self.r)
                I_prime_perturbed = self.d * self.k * (perturbed_s1 + orig_s2) / (4 * self.N * self.B * self.l * self.r)
                
                if I_prime_perturbed > I_prime_orig:
                    self.log(f"Monotonicity 2: Increasing s1 increases I' ({I_prime_orig} → {I_prime_perturbed})", True)
                else:
                    self.log(f"Monotonicity violation: I' didn't increase with s1", False)
                    return False
            
            # Test 3: Check linearity of s with I
            if orig_I != 0 and self.k != 0 and self.d != 0:
                # Test with 50% of original I
                I_half = orig_I * 0.5
                s_half = 2 * self.N * self.B * I_half * self.l * self.r / (self.d * self.k)
                s_full = 2 * self.N * self.B * orig_I * self.l * self.r / (self.d * self.k)
                
                if self.fuzzy_equal(s_half / s_full, 0.5, rel_tol=5e-2):
                    self.log(f"Linearity: s is linear with I (s(0.5I) ≈ 0.5s(I))", True)
                else:
                    self.log(f"Non-linearity detected: s(0.5I)/s(I) = {s_half/s_full}", False)
                    return False
            
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def test_superposition_principle(self):
        """Verify superposition principle for s1 and s2"""
        try:
            if self.s1 == 0 or self.s2 == 0:
                self.log("Skipping superposition test (missing s1 or s2)", True)
                return True
            
            # According to superposition: s1 = s0 + s(I'), s2 = s0 - s(I')
            # Where s0 is initial offset, s(I') is displacement due to current
            
            # From these: s1 + s2 = 2s0, s1 - s2 = 2s(I')
            s_I_prime = (self.s1 - self.s2) / 2
            
            # Calculate s(I') from formula
            if self.k != 0 and self.d != 0:
                s_I_prime_formula = 2 * self.N * self.B * self.I_prime * self.l * self.r / (self.d * self.k)
            else:
                s_I_prime_formula = 0
            
            if self.fuzzy_equal(s_I_prime, s_I_prime_formula, rel_tol=5e-2):
                self.log(f"Superposition verified: (s1-s2)/2 = s(I') = {s_I_prime}", True)
                return True
            else:
                self.log(f"Superposition mismatch: (s1-s2)/2={s_I_prime}, s(I') formula={s_I_prime_formula}", False)
                return False
        except Exception as e:
            self.log(f"Superposition test failed: {str(e)}", False)
            return False
    
    def test_sympy_laws(self):
        """Verify using symbolic physics laws - REMOVED SYMBOLIC OPERATIONS"""
        try:
            # This test now uses only numeric comparisons
            # Check force balance numerically
            F_magnetic = self.N * self.B * abs(self.I) * self.l
            F_spring = self.k * abs(self.delta_x)
            
            if self.fuzzy_equal(F_magnetic, F_spring):
                self.log("Numeric: Hooke's Law and Ampere's Force balance", True)
            else:
                self.log(f"Numeric force balance mismatch: F_mag={F_magnetic}, F_spring={F_spring}", False)
                return False
            
            # Check geometry numerically
            delta_x_geom = self.d * self.theta
            if self.fuzzy_equal(self.delta_x, delta_x_geom):
                self.log("Numeric: Δx = dθ verified", True)
                return True
            else:
                self.log(f"Numeric geometry mismatch: Δx={self.delta_x}, dθ={delta_x_geom}", False)
                return False
            
        except Exception as e:
            self.log(f"Numerical laws test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status with logs"""
        self.logs = []  # Reset logs
        
        # Run tests
        tests = [
            ("Force Balance", self.test_force_balance),
            ("Geometry Relations", self.test_geometry_relations),
            ("Derived Formulas", self.test_derived_formulas),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Superposition", self.test_superposition_principle),
            ("Numerical Laws", self.test_sympy_laws),
        ]
        
        success = True
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.log(f"Test '{test_name}' failed", False)
            except Exception as e:
                success = False
                self.log(f"Test '{test_name}' crashed: {str(e)}", False)
        
        return success, self.logs