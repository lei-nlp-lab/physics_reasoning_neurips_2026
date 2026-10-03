import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from results dictionary."""
        try:
            # Get results dictionary (default to empty dict)
            results = getattr(solution_instance, 'results', {})
            
            # Inputs (from problem statement) - get from results
            self.h = results.get('h', 0.0)
            self.t = results.get('t', 0.0)
            self.R = results.get('R', 0.0)
            self.G = results.get('G', 0.0)
            
            # Results (from solution steps)
            self.M = results.get('M', 0.0)
            self.g = results.get('g', 0.0)
            self.F_g = results.get('F_g', 0.0)
            self.F_c = results.get('F_c', 0.0)
            self.v_1 = results.get('v_1', 0.0)
            
            self.logs = []
            self.success = True
        except Exception as e:
            self.logs.append(f"Initialization error: {traceback.format_exc()}")
            self.success = False

    def test_invariants(self):
        """Check fundamental physics laws from KB."""
        try:
            # Check for zero values that would cause division errors
            if self.t == 0.0:
                self.logs.append("Warning: t = 0.0, skipping free-fall invariant test")
                self.success = False
                return
            if self.R == 0.0:
                self.logs.append("Warning: R = 0.0, skipping gravitational invariant test")
                self.success = False
                return
            
            # 1. Free-fall displacement: h = (1/2) * g * t^2
            expected_h = 0.5 * self.g * self.t**2
            if not math.isclose(self.h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"Free-fall invariant violated: h={self.h}, (1/2)gt²={expected_h}")
                self.success = False
            else:
                self.logs.append("✓ Free-fall displacement consistent")
            
            # 2. Weight formula: g = G * M / R^2 (from step_3 and Newton's law)
            expected_g = self.G * self.M / self.R**2
            if not math.isclose(self.g, expected_g, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"Gravitational invariant violated: g={self.g}, G*M/R²={expected_g}")
                self.success = False
            else:
                self.logs.append("✓ Gravitational acceleration consistent")
            
            # 3. Centripetal force: F_c = m*v_1^2/R, but since F_c = m*g from step_5, we have v_1^2 = g*R
            expected_v_squared = self.g * self.R
            actual_v_squared = self.v_1**2
            if not math.isclose(actual_v_squared, expected_v_squared, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"Centripetal invariant violated: v_1²={actual_v_squared}, g*R={expected_v_squared}")
                self.success = False
            else:
                self.logs.append("✓ Centripetal velocity consistent")
                
        except Exception as e:
            self.logs.append(f"Invariant test error: {traceback.format_exc()}")
            self.success = False

    def test_trace_consistency(self):
        """Cross-check intermediate steps with final results."""
        try:
            # Check for zero values that would cause division errors
            if self.t == 0.0:
                self.logs.append("Warning: t = 0.0, skipping trace consistency tests")
                self.success = False
                return
            if self.R == 0.0:
                self.logs.append("Warning: R = 0.0, skipping trace consistency tests")
                self.success = False
                return
            if self.G == 0.0:
                self.logs.append("Warning: G = 0.0, skipping trace consistency tests")
                self.success = False
                return
            
            # Step_1 to Step_2: g = 2h/t^2
            expected_g_step2 = 2 * self.h / self.t**2
            if not math.isclose(self.g, expected_g_step2, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"Step 2 inconsistency: g={self.g}, 2h/t²={expected_g_step2}")
                self.success = False
            else:
                self.logs.append("✓ Step 2 (g=2h/t²) consistent")
            
            # Step_3 to Step_4: M = g*R^2/G = 2h*R^2/(G*t^2)
            expected_M_step4 = self.g * self.R**2 / self.G
            if not math.isclose(self.M, expected_M_step4, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"Step 4 inconsistency: M={self.M}, gR²/G={expected_M_step4}")
                self.success = False
            else:
                self.logs.append("✓ Step 4 (M=gR²/G) consistent")
            
            # Step_5 to Step_6: v_1 = sqrt(g*R) = sqrt(2h*R)/t
            expected_v_step6_1 = math.sqrt(self.g * self.R)
            expected_v_step6_2 = math.sqrt(2 * self.h * self.R) / self.t
            if not (math.isclose(self.v_1, expected_v_step6_1, rel_tol=1e-2, abs_tol=1e-2) and 
                    math.isclose(self.v_1, expected_v_step6_2, rel_tol=1e-2, abs_tol=1e-2)):
                self.logs.append(f"Step 6 inconsistency: v_1={self.v_1}, sqrt(gR)={expected_v_step6_1}, sqrt(2hR)/t={expected_v_step6_2}")
                self.success = False
            else:
                self.logs.append("✓ Step 6 (v_1=√(gR)=√(2hR)/t) consistent")
                
        except Exception as e:
            self.logs.append(f"Trace consistency error: {traceback.format_exc()}")
            self.success = False

    def test_monotonicity(self):
        """Perturb input h and check trends (must restore original)."""
        try:
            # Check for zero values that would cause division errors
            if self.t == 0.0:
                self.logs.append("Warning: t = 0.0, skipping monotonicity test")
                self.success = False
                return
            if self.R == 0.0:
                self.logs.append("Warning: R = 0.0, skipping monotonicity test")
                self.success = False
                return
            if self.G == 0.0:
                self.logs.append("Warning: G = 0.0, skipping monotonicity test")
                self.success = False
                return
            
            original_h = self.h
            perturb_factors = [0.9, 1.1]  # Decrease and increase by 10%
            
            for factor in perturb_factors:
                # Create perturbed copy
                h_perturbed = original_h * factor
                
                # Recalculate using formulas from steps (analytically)
                g_perturbed = 2 * h_perturbed / self.t**2
                M_perturbed = g_perturbed * self.R**2 / self.G
                v_1_perturbed = math.sqrt(2 * h_perturbed * self.R) / self.t
                
                # Check monotonic trends
                if factor > 1:  # h increased
                    if not (g_perturbed > self.g and M_perturbed > self.M and v_1_perturbed > self.v_1):
                        self.logs.append(f"Monotonicity violated when h increased: g {g_perturbed} > {self.g}? M {M_perturbed} > {self.M}? v_1 {v_1_perturbed} > {self.v_1}?")
                        self.success = False
                else:  # h decreased
                    if not (g_perturbed < self.g and M_perturbed < self.M and v_1_perturbed < self.v_1):
                        self.logs.append(f"Monotonicity violated when h decreased: g {g_perturbed} < {self.g}? M {M_perturbed} < {self.M}? v_1 {v_1_perturbed} < {self.v_1}?")
                        self.success = False
            
            # Restore original h
            self.h = original_h
            if self.success:
                self.logs.append("✓ Monotonicity test passed")
                
        except Exception as e:
            self.logs.append(f"Monotonicity test error: {traceback.format_exc()}")
            self.success = False

    def run_all_tests(self):
        """Execute all checks and return final status."""
        if not hasattr(self, 'success'):
            self.success = True
            self.logs = []
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        return self.success, self.logs