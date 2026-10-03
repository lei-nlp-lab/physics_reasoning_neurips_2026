import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Follows robustness rules: never access attributes directly without fallbacks.
        """
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.h = getattr(solution_instance, 'h', 1.0)
        self.R = getattr(solution_instance, 'R', 0.5)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        
        # Convert theta to radians for symbolic use
        self.theta_rad = math.radians(self.theta_deg)
        
        # Extract results from dictionary with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.K_B = self.results.get('1/2*m*v_B2', 0.0)  # Kinetic energy at B (first pass)
        self.F_N = self.results.get('F_N', 0.0)         # Normal force at B
        self.F = self.results.get('F', 0.0)            # Force on track at B
        self.L = self.results.get('L', 0.0)            # Max distance up incline
        self.K_B_return = self.results.get("1/2*m*v_B'2", 0.0)  # Kinetic energy at B (return)
        
        # Derived quantities (computed for checks)
        self.v_B = math.sqrt(2 * self.K_B / self.m) if self.K_B > 0 else 0.0
        self.v_B_return = math.sqrt(2 * self.K_B_return / self.m) if self.K_B_return > 0 else 0.0
        
        # Log storage
        self.logs = []
        
    def log(self, msg: str):
        """Helper to append logs."""
        self.logs.append(msg)
    
    def fuzzy_equal(self, a: float, b: float, context: str = "") -> bool:
        """Robust fuzzy comparison with physics tolerances."""
        is_close = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        if not is_close:
            self.log(f"Mismatch in {context}: {a} vs {b}")
        return is_close
    
    def test_energy_conservation_step1(self) -> bool:
        """Check energy conservation from P to B (first descent)."""
        try:
            # Theoretical: m*g*(h + R) = 0.5*m*v_B^2
            potential_loss = self.m * self.g * (self.h + self.R)
            kinetic_at_B = self.K_B
            
            if not self.fuzzy_equal(potential_loss, kinetic_at_B, "Step1 energy conservation"):
                self.log(f"Energy conservation violation: PE loss = {potential_loss}, KE at B = {kinetic_at_B}")
                return False
            
            # Also check velocity consistency
            v_B_theory = math.sqrt(2 * self.g * (self.h + self.R))
            if not self.fuzzy_equal(self.v_B, v_B_theory, "Velocity at B"):
                return False
            
            self.log("✓ Step 1 energy conservation passed")
            return True
        except Exception as e:
            self.log(f"Error in energy conservation test: {e}")
            return False
    
    def test_newton_second_law_step2(self) -> bool:
        """Check normal force at B using Newton's second law."""
        try:
            # At bottom: F_N = m*g + m*v_B^2/R
            centripetal_term = self.m * self.v_B**2 / self.R
            F_N_theory = self.m * self.g + centripetal_term
            
            if not self.fuzzy_equal(self.F_N, F_N_theory, "Normal force at B"):
                self.log(f"Newton's 2nd law mismatch: F_N = {self.F_N}, theory = {F_N_theory}")
                return False
            
            self.log("✓ Step 2 Newton's second law passed")
            return True
        except Exception as e:
            self.log(f"Error in Newton's second law test: {e}")
            return False
    
    def test_newton_third_law_step3(self) -> bool:
        """Check Newton's third law: force on track equals normal force."""
        try:
            if not self.fuzzy_equal(self.F, self.F_N, "Newton's third law (action-reaction)"):
                self.log(f"Newton's 3rd law mismatch: F = {self.F}, F_N = {self.F_N}")
                return False
            
            self.log("✓ Step 3 Newton's third law passed")
            return True
        except Exception as e:
            self.log(f"Error in Newton's third law test: {e}")
            return False
    
    def test_work_energy_incline(self) -> bool:
        """Check work-energy theorem for incline motion (steps 4-5)."""
        try:
            # Using the equation from step 4:
            # m*g*(h + R - R*cosθ - L*sinθ) - μ*m*g*L*cosθ = 0
            # Rearranged to solve for L:
            # L = [m*g*(h + R - R*cosθ)] / [m*g*(sinθ + μ*cosθ)]
            # Cancel m*g
            
            numerator = self.h + self.R - self.R * self.cos37
            denominator = self.sin37 + self.mu * self.cos37
            L_theory = numerator / denominator if denominator != 0 else 0.0
            
            if not self.fuzzy_equal(self.L, L_theory, "Max distance L"):
                self.log(f"Work-energy mismatch: L = {self.L}, theory = {L_theory}")
                return False
            
            # Check the work-energy equation directly
            work_gravity = self.m * self.g * (self.h + self.R - self.R * self.cos37 - self.L * self.sin37)
            work_friction = self.mu * self.m * self.g * self.L * self.cos37
            net_work = work_gravity - work_friction
            
            if not self.fuzzy_equal(net_work, 0.0, "Net work on incline"):
                self.log(f"Work-energy equation not zero: {net_work}")
                return False
            
            self.log("✓ Steps 4-5 work-energy theorem passed")
            return True
        except Exception as e:
            self.log(f"Error in work-energy test: {e}")
            return False
    
    def test_return_energy_step6(self) -> bool:
        """Check kinetic energy at B after return trip."""
        try:
            # Energy loss = 2 * μ * m * g * L * cosθ (up and down)
            energy_loss = 2 * self.mu * self.m * self.g * self.L * self.cos37
            K_B_return_theory = self.K_B - energy_loss
            
            if not self.fuzzy_equal(self.K_B_return, K_B_return_theory, "Return kinetic energy"):
                self.log(f"Return energy mismatch: {self.K_B_return} vs {K_B_return_theory}")
                return False
            
            self.log("✓ Step 6 return energy calculation passed")
            return True
        except Exception as e:
            self.log(f"Error in return energy test: {e}")
            return False
    
    def test_slide_out_condition(self) -> bool:
        """Check if slider can slide out from point A."""
        try:
            # To slide out from A, must have enough energy to reach A from B on return
            # Energy at B (return): K_B_return
            # Energy needed to reach A: m*g*R (height difference from B to A)
            energy_needed = self.m * self.g * self.R
            
            can_slide_out = self.K_B_return > energy_needed
            
            # Check consistency with target answer
            target_answer = "You can slide out from point A."
            expected_outcome = can_slide_out or math.isclose(self.K_B_return, energy_needed, rel_tol=1e-2)
            
            # The target answer states it CAN slide out, so we expect True
            if not expected_outcome:
                self.log(f"Slide out condition failed: K_B_return={self.K_B_return}, needed={energy_needed}")
                return False
            
            self.log("✓ Slide out condition check passed")
            return True
        except Exception as e:
            self.log(f"Error in slide out test: {e}")
            return False
    
    def test_invariants(self) -> bool:
        """Run all invariant tests based on physics laws."""
        tests = [
            self.test_energy_conservation_step1,
            self.test_newton_second_law_step2,
            self.test_newton_third_law_step3,
            self.test_work_energy_incline,
            self.test_return_energy_step6,
            self.test_slide_out_condition
        ]
        
        success = True
        for test in tests:
            if not test():
                success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        try:
            # Check that F_N and F match the target answer (70 N)
            target_F = 70.0
            if not (self.fuzzy_equal(self.F_N, target_F, "F_N vs target") and 
                    self.fuzzy_equal(self.F, target_F, "F vs target")):
                self.log("Normal force doesn't match target 70N")
                return False
            
            # Check that L matches the correct physics value (1.1 m, not 1.2 m)
            # The correct value from the work-energy equation is:
            # L = (h + R - R*cos37) / (sin37 + μ*cos37)
            correct_L = (self.h + self.R - self.R * self.cos37) / (self.sin37 + self.mu * self.cos37)
            if not self.fuzzy_equal(self.L, correct_L, "L vs correct physics value"):
                self.log(f"L doesn't match correct physics value {correct_L:.1f} m: {self.L}")
                return False
            
            # Check that kinetic energies are positive and reasonable
            if self.K_B <= 0 or self.K_B_return <= 0:
                self.log("Kinetic energy should be positive")
                return False
            
            self.log("✓ Trace consistency check passed")
            return True
        except Exception as e:
            self.log(f"Error in trace consistency: {e}")
            return False
    
    def test_monotonicity(self) -> bool:
        """
        Perturb mass and check if results change in expected direction.
        Uses formulas, not solution methods.
        """
        original_m = self.m
        perturb_factor = 1.1  # 10% increase
        
        try:
            # Perturb mass
            self.m *= perturb_factor
            
            # Recalculate expected values using physical formulas
            # 1. Kinetic energy at B should scale linearly with m
            K_B_new = self.m * self.g * (self.h + self.R)
            expected_ratio_K = perturb_factor
            actual_ratio_K = K_B_new / self.K_B if self.K_B != 0 else 0
            
            if not self.fuzzy_equal(actual_ratio_K, expected_ratio_K, "K_B monotonicity"):
                self.log(f"K_B ratio mismatch: {actual_ratio_K} vs {expected_ratio_K}")
                self.m = original_m
                return False
            
            # 2. Normal force at B should scale linearly with m
            v_B_new = math.sqrt(2 * self.g * (self.h + self.R))  # Independent of m
            F_N_new = self.m * self.g + self.m * v_B_new**2 / self.R
            expected_ratio_F = perturb_factor
            actual_ratio_F = F_N_new / self.F_N if self.F_N != 0 else 0
            
            if not self.fuzzy_equal(actual_ratio_F, expected_ratio_F, "F_N monotonicity"):
                self.log(f"F_N ratio mismatch: {actual_ratio_F} vs {expected_ratio_F}")
                self.m = original_m
                return False
            
            # 3. L should be independent of m (cancels out)
            L_new = (self.h + self.R - self.R * self.cos37) / (self.sin37 + self.mu * self.cos37)
            if not self.fuzzy_equal(L_new, self.L, "L independence from m"):
                self.log(f"L changed with m: {L_new} vs {self.L}")
                self.m = original_m
                return False
            
            # 4. Return kinetic energy should scale linearly with m
            K_B_return_new = K_B_new - 2 * self.mu * self.m * self.g * self.L * self.cos37
            expected_ratio_return = perturb_factor
            actual_ratio_return = K_B_return_new / self.K_B_return if self.K_B_return != 0 else 0
            
            if not self.fuzzy_equal(actual_ratio_return, expected_ratio_return, "K_B_return monotonicity"):
                self.log(f"K_B_return ratio mismatch: {actual_ratio_return} vs {expected_ratio_return}")
                self.m = original_m
                return False
            
            # Restore original mass
            self.m = original_m
            
            self.log("✓ Monotonicity tests passed")
            return True
            
        except Exception as e:
            self.log(f"Error in monotonicity test: {e}")
            self.m = original_m  # Ensure mass is restored
            return False
    
    def run_all_tests(self) -> tuple[bool, list[str]]:
        """Execute all tests and return success status with logs."""
        self.logs.clear()
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"\n=== Running {test_name} ===")
            if not test_func():
                success = False
                self.log(f"✗ {test_name} failed")
            else:
                self.log(f"✓ {test_name} passed")
        
        return success, self.logs