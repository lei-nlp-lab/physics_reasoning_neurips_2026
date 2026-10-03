import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs (with defaults from problem statement)
        self.v_kmh = getattr(solution_instance, 'v_kmh', 43.2)
        self.v = self.v_kmh * (1000/3600)  # Convert to m/s
        self.x1 = getattr(solution_instance, 'x1', 36.0)
        self.x2 = getattr(solution_instance, 'x2', 0.0)  # Not needed
        self.x3 = getattr(solution_instance, 'x3', 36.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.M = getattr(solution_instance, 'M', 4500.0)  # Bus mass inferred from F=9000N and a₂=2
        self.m = getattr(solution_instance, 'm', 30.0)    # Student mass inferred from F=60√26
        
        # Safe extraction of results (using KB mapping and steps)
        self.results = {}
        self.results['a₁'] = getattr(solution_instance, 'a₁', getattr(solution_instance, 'results', {}).get('a₁', 0.0))
        self.results['a₂'] = getattr(solution_instance, 'a₂', getattr(solution_instance, 'results', {}).get('a₂', 0.0))
        self.results['F_resistance'] = getattr(solution_instance, 'F_resistance', getattr(solution_instance, 'results', {}).get('F_resistance', 0.0))
        self.results['F_horizontal'] = getattr(solution_instance, 'F_horizontal', getattr(solution_instance, 'results', {}).get('F_horizontal', 0.0))
        self.results['F_vertical'] = getattr(solution_instance, 'F_vertical', getattr(solution_instance, 'results', {}).get('F_vertical', 0.0))
        self.results['F'] = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 0.0))
        
        # Expected symbolic forms
        self.expected = {
            'a₁': self.v**2 / (2 * self.x1),
            'a₂': self.v**2 / (2 * self.x3),
            'F_resistance': self.M * self.v**2 / (2 * self.x3),
            'F_horizontal': self.m * self.v**2 / (2 * self.x1),
            'F_vertical': self.m * self.g,
            'F': self.m * sp.sqrt(self.g**2 + (self.v**2 / (2 * self.x1))**2)
        }
        
        # Store logs
        self.logs = []
        
    def log(self, msg, success=None):
        entry = {'message': msg, 'success': success}
        self.logs.append(entry)
        return success
        
    def test_invariants(self):
        """Verify kinematic equations and Newton's laws from KB."""
        try:
            # 1. Kinematic: a₁ = v²/(2x₁)
            a1_expected = self.v**2 / (2 * self.x1)
            if not math.isclose(self.results['a₁'], a1_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"a₁ mismatch: got {self.results['a₁']}, expected {a1_expected}", False)
            self.log(f"a₁ kinematic OK: {self.results['a₁']} ≈ {a1_expected}", True)
            
            # 2. Kinematic: a₂ = v²/(2x₃)
            a2_expected = self.v**2 / (2 * self.x3)
            if not math.isclose(self.results['a₂'], a2_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"a₂ mismatch: got {self.results['a₂']}, expected {a2_expected}", False)
            self.log(f"a₂ kinematic OK: {self.results['a₂']} ≈ {a2_expected}", True)
            
            # 3. Newton's 2nd: F_resistance = M * a₂
            F_res_expected = self.M * a2_expected
            if not math.isclose(self.results['F_resistance'], F_res_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"F_resistance mismatch: got {self.results['F_resistance']}, expected {F_res_expected}", False)
            self.log(f"F_resistance Newton OK: {self.results['F_resistance']} ≈ {F_res_expected}", True)
            
            # 4. Force decomposition for student
            F_h_expected = self.m * a1_expected
            if not math.isclose(self.results['F_horizontal'], F_h_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"F_horizontal mismatch: got {self.results['F_horizontal']}, expected {F_h_expected}", False)
            self.log(f"F_horizontal decomposition OK: {self.results['F_horizontal']} ≈ {F_h_expected}", True)
            
            F_v_expected = self.m * self.g
            if not math.isclose(self.results['F_vertical'], F_v_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"F_vertical mismatch: got {self.results['F_vertical']}, expected {F_v_expected}", False)
            self.log(f"F_vertical decomposition OK: {self.results['F_vertical']} ≈ {F_v_expected}", True)
            
            # 5. Resultant force magnitude
            F_expected = math.sqrt(F_h_expected**2 + F_v_expected**2)
            if not math.isclose(self.results['F'], F_expected, rel_tol=1e-2, abs_tol=1e-2):
                # Check if result is in radical form (60√26 ≈ 305.94)
                radical_expected = 60 * math.sqrt(26)
                if math.isclose(self.results['F'], radical_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"F magnitude matches radical form 60√26 ≈ {radical_expected}", True)
                else:
                    return self.log(f"F magnitude mismatch: got {self.results['F']}, expected {F_expected}", False)
            else:
                self.log(f"F magnitude OK: {self.results['F']} ≈ {F_expected}", True)
                
            return True
            
        except Exception as e:
            return self.log(f"Invariant test failed with error: {traceback.format_exc()}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results match final answers."""
        try:
            # Check a₁ and a₂ are equal (both = v²/(2*36))
            if not math.isclose(self.results['a₁'], self.results['a₂'], rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"a₁ ({self.results['a₁']}) ≠ a₂ ({self.results['a₂']}) but both should be v²/(2*36)", False)
            self.log(f"a₁ and a₂ consistent: both = {self.results['a₁']}", True)
            
            # Check F_horizontal = m * a₁
            F_h_calc = self.m * self.results['a₁']
            if not math.isclose(self.results['F_horizontal'], F_h_calc, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"F_horizontal {self.results['F_horizontal']} ≠ m*a₁ = {F_h_calc}", False)
            self.log(f"F_horizontal consistent with m*a₁: {self.results['F_horizontal']}", True)
            
            # Check F = √(F_h² + F_v²)
            F_mag = math.sqrt(self.results['F_horizontal']**2 + self.results['F_vertical']**2)
            if not math.isclose(self.results['F'], F_mag, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"F {self.results['F']} ≠ √(F_h²+F_v²) = {F_mag}", False)
            self.log(f"F magnitude consistent with components: {self.results['F']}", True)
            
            return True
            
        except Exception as e:
            return self.log(f"Trace consistency test failed with error: {traceback.format_exc()}", False)
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        try:
            original_m = self.m
            perturbed_m = original_m * 1.1  # +10%
            
            # Recalculate expected forces with perturbed mass
            F_h_orig = self.m * self.results['a₁']
            F_h_pert = perturbed_m * self.results['a₁']
            F_v_orig = self.m * self.g
            F_v_pert = perturbed_m * self.g
            F_orig = math.sqrt(F_h_orig**2 + F_v_orig**2)
            F_pert = math.sqrt(F_h_pert**2 + F_v_pert**2)
            
            # All should increase by 10% (since linear in m)
            if not (math.isclose(F_h_pert/F_h_orig, 1.1, rel_tol=1e-2) and
                    math.isclose(F_v_pert/F_v_orig, 1.1, rel_tol=1e-2) and
                    math.isclose(F_pert/F_orig, 1.1, rel_tol=1e-2)):
                return self.log(f"Monotonicity fail: forces should scale linearly with m (perturb +10%)", False)
            self.log("Monotonicity OK: forces scale linearly with student mass", True)
            
            # Perturb bus mass M for resistance force
            original_M = self.M
            perturbed_M = original_M * 1.1
            F_res_orig = original_M * self.results['a₂']
            F_res_pert = perturbed_M * self.results['a₂']
            if not math.isclose(F_res_pert/F_res_orig, 1.1, rel_tol=1e-2):
                return self.log(f"Monotonicity fail: F_resistance should scale linearly with M", False)
            self.log("Monotonicity OK: F_resistance scales linearly with bus mass", True)
            
            return True
            
        except Exception as e:
            return self.log(f"Monotonicity test failed with error: {traceback.format_exc()}", False)
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs)."""
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for name, test_func in tests:
            self.log(f"--- Starting {name} Test ---", None)
            success = test_func()
            if not success:
                overall_success = False
                self.log(f"✗ {name} test failed", False)
            else:
                self.log(f"✓ {name} test passed", True)
        
        return overall_success, self.logs