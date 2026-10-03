import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely initialize all parameters and results from the solution instance.
        Uses getattr and .get() with fallbacks to avoid AttributeError.
        """
        # Store the instance for potential method calls (but be cautious)
        self.solution = solution_instance
        
        # Extract input parameters (safe access with defaults)
        results = getattr(solution_instance, 'results', {})
        
        # Get all values from results dictionary with sensible defaults
        self.m = results.get('m', 0.08)
        self.R1 = results.get('R1', 1.0)
        self.l = results.get('l', 0.5)
        self.mu = results.get('mu', 0.5)
        self.E = results.get('E', 12.0)
        self.B = results.get('B', 0.6)
        self.g = results.get('g', 10.0)
        
        # Get trigonometric values directly from results or use defaults
        self.sin_alpha = results.get('sin37', 0.6)
        self.cos_alpha = results.get('cos37', 0.8)
        
        # Extract results (safe access with defaults)
        self.R2_min = results.get('R2_min', results.get('R_2_min', 3.5))
        self.R2_max = results.get('R2_max', results.get('R_2_max', 21.5))
        self.mg_parallel = results.get('mg_parallel', 0.48)
        self.F = results.get('F', 0.6)
        self.I_example = results.get('I', 2.0)  # Example current from steps
        self.F_prime = results.get("F'", results.get('F_prime', 0.48))
        
        # Derived quantities for verification
        self.mg = self.m * self.g
        self.mg_perp = self.mg * self.cos_alpha
        self.mg_parallel_calc = self.mg * self.sin_alpha
        
        # Example R2 for the step-by-step calculation (where I=2A)
        self.R2_example = self.E / self.I_example - self.R1
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []

    def log(self, message: str, success: Optional[bool] = None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def fuzzy_equal(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self) -> bool:
        """Test fundamental physical invariants."""
        success = True
        
        # 1. Check mg_parallel calculation
        if not self.fuzzy_equal(self.mg_parallel, self.mg_parallel_calc):
            self.log(f"mg_parallel mismatch: calculated {self.mg_parallel_calc}, expected {self.mg_parallel}", False)
            success = False
        else:
            self.log(f"mg_parallel correct: {self.mg_parallel} ≈ {self.mg_parallel_calc}", True)
        
        # 2. Check example Ampere force F = B * I * l
        F_calc = self.B * self.I_example * self.l
        if not self.fuzzy_equal(self.F, F_calc):
            self.log(f"Ampere force mismatch: calculated {F_calc}, expected {self.F}", False)
            success = False
        else:
            self.log(f"Ampere force correct: {self.F} ≈ {F_calc}", True)
        
        # 3. Check F' = F * cos(alpha)
        F_prime_calc = self.F * self.cos_alpha
        if not self.fuzzy_equal(self.F_prime, F_prime_calc):
            self.log(f"F' mismatch: calculated {F_prime_calc}, expected {self.F_prime}", False)
            success = False
        else:
            self.log(f"F' correct: {self.F_prime} ≈ {F_prime_calc}", True)
        
        # 4. Check that R2 range is physically plausible
        if self.R2_min < 0 or self.R2_max > 30:  # Max R2 given as 30Ω
            self.log(f"R2 range outside plausible bounds: [{self.R2_min}, {self.R2_max}]", False)
            success = False
        else:
            self.log(f"R2 range within plausible bounds: [{self.R2_min}, {self.R2_max}]", True)
        
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer."""
        success = True
        
        # 1. Verify that at R2_min, the equilibrium condition holds (bar about to slide UP)
        # For R2_min (minimum resistance, maximum current), rod about to slide UP
        # Condition: F*cosα = mg*sinα + μ*(m*g*cosα - F*sinα)
        I_min = self.E / (self.R1 + self.R2_min)
        F_min = self.B * I_min * self.l
        F_prime_min = F_min * self.cos_alpha
        # CORRECTED: Normal force N = m*g*cosα - F*sinα
        N_min = self.mg_perp - F_min * self.sin_alpha
        
        lhs_up = F_prime_min
        rhs_up = self.mg_parallel + self.mu * N_min
        
        if not self.fuzzy_equal(lhs_up, rhs_up):
            self.log(f"R2_min (slide UP) condition mismatch: F'={lhs_up}, mg_parallel+μN={rhs_up}", False)
            success = False
        else:
            self.log(f"R2_min (slide UP) condition satisfied: {lhs_up} ≈ {rhs_up}", True)
        
        # 2. Verify that at R2_max, the equilibrium condition holds (bar about to slide DOWN)
        # For R2_max (maximum resistance, minimum current), rod about to slide DOWN
        # Condition: mg*sinα = F*cosα + μ*(m*g*cosα - F*sinα)
        I_max = self.E / (self.R1 + self.R2_max)
        F_max = self.B * I_max * self.l
        F_prime_max = F_max * self.cos_alpha
        # CORRECTED: Normal force N = m*g*cosα - F*sinα
        N_max = self.mg_perp - F_max * self.sin_alpha
        
        lhs_down = self.mg_parallel
        rhs_down = F_prime_max + self.mu * N_max
        
        if not self.fuzzy_equal(lhs_down, rhs_down):
            self.log(f"R2_max (slide DOWN) condition mismatch: mg_parallel={lhs_down}, F'+μN={rhs_down}", False)
            success = False
        else:
            self.log(f"R2_max (slide DOWN) condition satisfied: {lhs_down} ≈ {rhs_down}", True)
        
        # 3. Check that R2_example (5Ω) lies within the range
        if not (self.R2_min <= self.R2_example <= self.R2_max):
            self.log(f"Example R2 ({self.R2_example}) not in range [{self.R2_min}, {self.R2_max}]", False)
            success = False
        else:
            self.log(f"Example R2 ({self.R2_example}) within range [{self.R2_min}, {self.R2_max}]", True)
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb mass and verify expected trends.
        Restores original mass after test.
        """
        original_m = self.m
        success = True
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            perturbed_mg_parallel = perturbed_m * self.g * self.sin_alpha
            
            # Compute new R2_min and R2_max using the CORRECT equilibrium equations
            # For impending slide UP (R2_min): F*cosα = mg*sinα + μ*(m*g*cosα - F*sinα)
            # Solve for I: I = [mg*(sinα + μ*cosα)] / [B*l*(cosα + μ*sinα)]
            numerator_up = perturbed_m * self.g * (self.sin_alpha + self.mu * self.cos_alpha)
            denominator_up = self.B * self.l * (self.cos_alpha + self.mu * self.sin_alpha)
            I_up_new = numerator_up / denominator_up
            R2_min_new = self.E / I_up_new - self.R1
            
            # For impending slide DOWN (R2_max): mg*sinα = F*cosα + μ*(m*g*cosα - F*sinα)
            # Solve for I: I = [mg*(sinα - μ*cosα)] / [B*l*(cosα - μ*sinα)]
            numerator_down = perturbed_m * self.g * (self.sin_alpha - self.mu * self.cos_alpha)
            denominator_down = self.B * self.l * (self.cos_alpha - self.mu * self.sin_alpha)
            I_down_new = numerator_down / denominator_down
            R2_max_new = self.E / I_down_new - self.R1
            
            # Check monotonicity: increasing mass should decrease both R2_min and R2_max
            # (larger mass requires larger current to hold, so smaller R2)
            if R2_min_new > self.R2_min:
                self.log(f"Monotonicity violated for R2_min: increased mass but R2_min increased ({R2_min_new} > {self.R2_min})", False)
                success = False
            else:
                self.log(f"R2_min monotonicity: {R2_min_new} ≤ {self.R2_min} (mass increased)", True)
            
            if R2_max_new > self.R2_max:
                self.log(f"Monotonicity violated for R2_max: increased mass but R2_max increased ({R2_max_new} > {self.R2_max})", False)
                success = False
            else:
                self.log(f"R2_max monotonicity: {R2_max_new} ≤ {self.R2_max} (mass increased)", True)
            
            # Additional check: perturbed mg_parallel should be larger
            if perturbed_mg_parallel <= self.mg_parallel:
                self.log(f"mg_parallel monotonicity violated: {perturbed_mg_parallel} ≤ {self.mg_parallel}", False)
                success = False
            else:
                self.log(f"mg_parallel monotonicity: {perturbed_mg_parallel} > {self.mg_parallel}", True)
                
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {str(e)}", False)
            success = False
        finally:
            # Restore original mass
            self.m = original_m
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            try:
                self.log(f"Running {name} test...", None)
                success = test_func()
                overall_success = overall_success and success
            except Exception as e:
                self.log(f"{name} test crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        self.log(f"All tests completed. Overall success: {overall_success}", None)
        return overall_success, self.logs