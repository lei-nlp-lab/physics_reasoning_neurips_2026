import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract all values from results dictionary only
        results = getattr(solution_instance, 'results', {})
        
        # Input parameters - use safe get with default 0.0
        self.h = results.get('h', 0.0)
        self.v = results.get('v', 0.0)
        self.G = results.get('G', 0.0)
        self.R = results.get('R', 0.0)
        
        # Results from solution steps - handle both 'ρ' and 'rho' keys
        self.g = results.get('g', 0.0)
        self.M = results.get('M', 0.0)
        
        # Try Unicode rho first, then ASCII 'rho' as fallback
        self.rho = results.get('ρ', 0.0)
        if self.rho == 0.0:
            self.rho = results.get('rho', 0.0)
        
        # Store original values for monotonicity test
        self.original_h = self.h
        self.original_g = self.g
        self.original_M = self.M
        self.original_rho = self.rho
        
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def test_invariants(self):
        """Verify energy conservation and Newton's laws"""
        try:
            # Check for zero values that would cause division errors
            if abs(self.h) < 1e-15:
                self.log("ERROR: h is zero or too small, cannot compute expected_g")
                return False
            if abs(self.R) < 1e-15:
                self.log("ERROR: R is zero or too small, cannot compute expected_g_from_M or expected_rho")
                return False
            
            # 1. Check kinematic equation: v² = 2gh (from energy conservation)
            expected_g = self.v**2 / (2 * self.h)
            if not math.isclose(self.g, expected_g, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: g invariant violated. Expected {expected_g}, got {self.g}")
                return False
            self.log("PASS: Kinematic equation v² = 2gh holds")

            # 2. Check Newton's Law of Gravitation: g = GM/R²
            # Check if G is not zero to avoid division by zero
            if abs(self.G) < 1e-15:
                self.log("ERROR: G is zero or too small, cannot compute expected_g_from_M")
                return False
            expected_g_from_M = self.G * self.M / (self.R**2)
            if not math.isclose(self.g, expected_g_from_M, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Gravitational law violated. g should be GM/R² = {expected_g_from_M}, got {self.g}")
                return False
            self.log("PASS: Newton's law g = GM/R² holds")

            # 3. Check density definition: ρ = M / (4/3 π R³)
            expected_rho = self.M / (4/3 * math.pi * self.R**3)
            if not math.isclose(self.rho, expected_rho, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Density definition violated. Expected {expected_rho}, got {self.rho}")
                return False
            self.log("PASS: Density definition ρ = M/(4/3 π R³) holds")

            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check for zero values that would cause division errors
            if abs(self.R) < 1e-15:
                self.log("ERROR: R is zero or too small, cannot compute g_from_M or rho_from_M")
                return False
            if abs(self.h) < 1e-15 or abs(self.G) < 1e-15:
                self.log("ERROR: h or G is zero or too small, cannot compute formula values")
                return False
            
            # Check that g from step 1 matches g computed from M (step 2)
            g_from_M = self.G * self.M / (self.R**2)
            if not math.isclose(self.g, g_from_M, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: g from step1 ({self.g}) inconsistent with g from M ({g_from_M})")
                return False
            self.log("PASS: g consistent between step1 and step2")

            # Check that ρ from step 3 matches ρ computed from M
            rho_from_M = self.M / (4/3 * math.pi * self.R**3)
            if not math.isclose(self.rho, rho_from_M, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: ρ from step3 ({self.rho}) inconsistent with ρ from M ({rho_from_M})")
                return False
            self.log("PASS: ρ consistent with M")

            # Verify all formulas match the given steps
            expected_g_formula = self.v**2 / (2 * self.h)
            expected_M_formula = self.v**2 * self.R**2 / (2 * self.G * self.h)
            expected_rho_formula = 3 * self.v**2 / (8 * self.G * self.h * math.pi * self.R)

            if not (math.isclose(self.g, expected_g_formula, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.M, expected_M_formula, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.rho, expected_rho_formula, rel_tol=1e-2, abs_tol=1e-2)):
                self.log("FAIL: One or more formulas don't match step equations")
                return False

            self.log("PASS: All step formulas are consistent")
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            return False

    def test_monotonicity(self):
        """Perturb height and verify expected trends"""
        try:
            # Check for zero values
            if abs(self.h) < 1e-15:
                self.log("ERROR: h is zero or too small, cannot perturb")
                return False
            if abs(self.G) < 1e-15:
                self.log("ERROR: G is zero or too small, cannot compute perturbed values")
                return False
            if abs(self.R) < 1e-15:
                self.log("ERROR: R is zero or too small, cannot compute perturbed values")
                return False
            
            # Perturb h by +1%
            perturbed_h = self.h * 1.01
            
            # Recalculate using formulas from steps
            perturbed_g = self.v**2 / (2 * perturbed_h)
            perturbed_M = self.v**2 * self.R**2 / (2 * self.G * perturbed_h)
            perturbed_rho = 3 * self.v**2 / (8 * self.G * perturbed_h * math.pi * self.R)
            
            # Check monotonic trends
            # Since g = v²/(2h), increasing h should decrease g
            if not (perturbed_g < self.g):
                self.log(f"FAIL: g should decrease when h increases. Original: {self.g}, Perturbed: {perturbed_g}")
                return False
            self.log("PASS: g decreases with increased h (g ∝ 1/h)")
            
            # Since M = v²R²/(2Gh), increasing h should decrease M
            if not (perturbed_M < self.M):
                self.log(f"FAIL: M should decrease when h increases. Original: {self.M}, Perturbed: {perturbed_M}")
                return False
            self.log("PASS: M decreases with increased h (M ∝ 1/h)")
            
            # Since ρ = 3v²/(8GπRh), increasing h should decrease ρ
            if not (perturbed_rho < self.rho):
                self.log(f"FAIL: ρ should decrease when h increases. Original: {self.rho}, Perturbed: {perturbed_rho}")
                return False
            self.log("PASS: ρ decreases with increased h (ρ ∝ 1/h)")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success flag with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\nRunning {test_name} test...")
            if not test_func():
                success = False
                self.log(f"✗ {test_name} test failed")
            else:
                self.log(f"✓ {test_name} test passed")
        
        return success, self.logs