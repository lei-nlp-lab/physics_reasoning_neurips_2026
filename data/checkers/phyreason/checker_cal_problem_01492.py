import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract ALL values from results dictionary only
        results = getattr(solution_instance, 'results', {})
        
        # Input values
        self.h = results.get('h', None)
        self.v = results.get('v', None)
        self.G = results.get('G', None)
        self.R = results.get('R', None)
        
        # Result values
        self.g = results.get('g', None)
        self.M = results.get('M', None)
        self.V = results.get('V', None)
        self.rho = results.get('rho', results.get('\rho', None))
        
        self.solution = solution_instance
        self.logs = []
        
    def compare_values(self, val1, val2, name=""):
        """Robust numeric comparison"""
        try:
            # Convert to float if possible
            if hasattr(val1, 'evalf'):
                val1 = float(val1.evalf())
            if hasattr(val2, 'evalf'):
                val2 = float(val2.evalf())
            
            # Numeric comparison
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                return math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2)
            else:
                # Fallback to string comparison
                return str(val1) == str(val2)
        except Exception as e:
            self.logs.append(f"Warning: Could not compare {name}: {str(e)}")
            return str(val1) == str(val2)
    
    def test_invariants(self):
        """Verify fundamental physics laws"""
        logs = []
        success = True
        
        try:
            # 1. Check kinematic equation: v² = 2gh (from free fall)
            if self.g is not None and self.v is not None and self.h is not None:
                expected_v_squared = 2 * self.g * self.h
                v_squared = self.v**2
                if not self.compare_values(v_squared, expected_v_squared, "v² = 2gh"):
                    logs.append(f"FAIL: v² = {v_squared} ≠ 2gh = {expected_v_squared}")
                    success = False
                else:
                    logs.append("PASS: Kinematic equation v² = 2gh")
            
            # 2. Check Newton's Law: GM/R² = g
            if self.g is not None and self.M is not None and self.G is not None and self.R is not None:
                expected_g = self.G * self.M / (self.R**2)
                if not self.compare_values(self.g, expected_g, "g = GM/R²"):
                    logs.append(f"FAIL: g = {self.g} ≠ GM/R² = {expected_g}")
                    success = False
                else:
                    logs.append("PASS: Newton's Law GM/R² = g")
            
            # 3. Check density consistency: ρ = M/V
            if self.rho is not None and self.M is not None and self.V is not None:
                expected_rho = self.M / self.V
                if not self.compare_values(self.rho, expected_rho, "ρ = M/V"):
                    logs.append(f"FAIL: ρ = {self.rho} ≠ M/V = {expected_rho}")
                    success = False
                else:
                    logs.append("PASS: Density formula ρ = M/V")
                    
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
            
        self.logs.extend(logs)
        return success
    
    def test_trace_consistency(self):
        """Verify step-by-step consistency with STEPS ANALYSIS"""
        logs = []
        success = True
        
        try:
            # Check step 1: g = v²/(2h)
            if self.g is not None and self.v is not None and self.h is not None:
                expected_g_step1 = self.v**2 / (2 * self.h)
                if not self.compare_values(self.g, expected_g_step1, "g = v²/(2h)"):
                    logs.append(f"FAIL Step1: g = {self.g} ≠ v²/(2h) = {expected_g_step1}")
                    success = False
                else:
                    logs.append("PASS Step1: g = v²/(2h)")
            
            # Check step 2: GMm/R² = mg (should be identity if step1 correct)
            if self.g is not None and self.M is not None:
                logs.append("INFO Step2: Relationship GM/R² = g verified in invariants")
            
            # Check step 3: M = v²R²/(2Gh)
            if self.M is not None and self.v is not None and self.R is not None and self.G is not None and self.h is not None:
                expected_M_step3 = (self.v**2 * self.R**2) / (2 * self.G * self.h)
                if not self.compare_values(self.M, expected_M_step3, "M = v²R²/(2Gh)"):
                    logs.append(f"FAIL Step3: M = {self.M} ≠ v²R²/(2Gh) = {expected_M_step3}")
                    success = False
                else:
                    logs.append("PASS Step3: M = v²R²/(2Gh)")
            
            # Check step 4: V = (4/3)πR³
            if self.V is not None and self.R is not None:
                expected_V_step4 = (4/3) * math.pi * self.R**3
                if not self.compare_values(self.V, expected_V_step4, "V = (4/3)πR³"):
                    logs.append(f"FAIL Step4: V = {self.V} ≠ (4/3)πR³ = {expected_V_step4}")
                    success = False
                else:
                    logs.append("PASS Step4: V = (4/3)πR³")
            
            # Check step 5: ρ = 3v²/(8πG h R)
            if self.rho is not None and self.v is not None and self.G is not None and self.h is not None and self.R is not None:
                expected_rho_step5 = (3 * self.v**2) / (8 * math.pi * self.G * self.h * self.R)
                if not self.compare_values(self.rho, expected_rho_step5, "ρ = 3v²/(8πG h R)"):
                    logs.append(f"FAIL Step5: ρ = {self.rho} ≠ 3v²/(8πG h R) = {expected_rho_step5}")
                    success = False
                else:
                    logs.append("PASS Step5: ρ = 3v²/(8πG h R)")
                    
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
            
        self.logs.extend(logs)
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends"""
        logs = []
        success = True
        
        try:
            # Store original values
            original_h = self.h
            
            # Only test if we have numeric values
            if (isinstance(original_h, (int, float)) and 
                isinstance(self.v, (int, float)) and
                isinstance(self.R, (int, float)) and
                isinstance(self.G, (int, float))):
                
                # Increase h by 1%
                perturbed_h = original_h * 1.01
                
                # Recalculate expected results using formulas
                g_perturbed = self.v**2 / (2 * perturbed_h)
                M_perturbed = (self.v**2 * self.R**2) / (2 * self.G * perturbed_h)
                V_perturbed = (4/3) * math.pi * self.R**3  # Unchanged
                rho_perturbed = M_perturbed / V_perturbed
                
                # Check trends: if h increases, g, M, ρ should decrease
                if self.g is not None:
                    if g_perturbed < self.g:
                        logs.append("PASS Monotonicity: g decreases when h increases")
                    else:
                        logs.append(f"FAIL Monotonicity: g(perturbed)={g_perturbed} not < g(original)={self.g}")
                        success = False
                
                if self.M is not None:
                    if M_perturbed < self.M:
                        logs.append("PASS Monotonicity: M decreases when h increases")
                    else:
                        logs.append(f"FAIL Monotonicity: M(perturbed)={M_perturbed} not < M(original)={self.M}")
                        success = False
                
                if self.rho is not None:
                    if rho_perturbed < self.rho:
                        logs.append("PASS Monotonicity: ρ decreases when h increases")
                    else:
                        logs.append(f"FAIL Monotonicity: ρ(perturbed)={rho_perturbed} not < ρ(original)={self.rho}")
                        success = False
                        
                # Restore original value
                self.h = original_h
            else:
                logs.append("INFO Monotonicity: Skipped (non-numeric input)")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
            # Restore original value on error
            if 'original_h' in locals():
                self.h = original_h
        
        self.logs.extend(logs)
        return success
    
    def run_all_tests(self):
        """Run all validation tests"""
        self.logs = []
        overall_success = True
        
        # Run individual tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== Running {test_name} ===")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"CRITICAL ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Overall success: {overall_success}")
        
        return overall_success, self.logs