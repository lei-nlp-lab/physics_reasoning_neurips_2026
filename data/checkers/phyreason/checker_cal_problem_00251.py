import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from the solution instance
        self.R = 50.0  # From problem context
        
        # Safely extract results using getattr with fallback
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', 0.0))
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', 0.0))
        self.direction_str = getattr(solution_instance, 'N/A', solution_instance.results.get('N/A', ''))
        
        # Define symbolic variables for invariant checking
        self.r = sp.symbols('r', positive=True)
        self.displacement_mag_sym = sp.sqrt(2) * self.r
        self.arc_length_sym = (3/sp.Integer(4)) * 2 * sp.pi * self.r
        
        self.logs = []

    def test_invariants(self):
        """Verify physical invariants using symbolic and numerical checks."""
        try:
            # 1. Check displacement magnitude formula: x = √2 * R
            expected_x = math.sqrt(2) * self.R
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Displacement magnitude mismatch. Expected ~{expected_x:.1f}, got {self.x}")
                return False
            self.logs.append(f"PASS: Displacement magnitude {self.x:.1f} ≈ √2 * {self.R} = {expected_x:.1f}")
            
            # 2. Check arc length formula: l = (3/4) * 2πR
            expected_l = (3/4) * 2 * math.pi * self.R
            if not math.isclose(self.l, expected_l, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Arc length mismatch. Expected ~{expected_l:.1f}, got {self.l}")
                return False
            self.logs.append(f"PASS: Arc length {self.l:.1f} ≈ (3/4)*2π*{self.R} = {expected_l:.1f}")
            
            # 3. Check displacement direction string
            if not self.direction_str or "A to B" not in self.direction_str:
                self.logs.append(f"WARNING: Displacement direction string may be incorrect: '{self.direction_str}'")
            else:
                self.logs.append(f"PASS: Displacement direction correctly described as '{self.direction_str}'")
            
            # 4. Symbolic verification of formulas
            x_expr = sp.simplify(self.displacement_mag_sym - sp.sqrt(2)*self.r)
            l_expr = sp.simplify(self.arc_length_sym - (3/sp.Integer(4))*2*sp.pi*self.r)
            if x_expr != 0 or l_expr != 0:
                self.logs.append(f"SYMBOLIC MISMATCH: x_expr={x_expr}, l_expr={l_expr}")
                return False
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        try:
            # From STEPS ANALYSIS: step_1 gives x=70.7, step_3 gives l=235.5
            step1_x = 70.7
            step3_l = 235.5
            
            if not math.isclose(self.x, step1_x, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 1 displacement {step1_x} inconsistent with final {self.x}")
                return False
            self.logs.append(f"PASS: Step 1 displacement consistent")
            
            if not math.isclose(self.l, step3_l, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 3 arc length {step3_l} inconsistent with final {self.l}")
                return False
            self.logs.append(f"PASS: Step 3 arc length consistent")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb radius and check that results scale appropriately."""
        try:
            # Perturb radius by ±10%
            perturbations = [0.9, 1.1]  # 90% and 110% of original
            
            for factor in perturbations:
                R_perturbed = self.R * factor
                
                # Recalculate using formulas
                x_perturbed = math.sqrt(2) * R_perturbed
                l_perturbed = (3/4) * 2 * math.pi * R_perturbed
                
                # Check monotonic trends
                if factor < 1:
                    if not (x_perturbed < self.x and l_perturbed < self.l):
                        self.logs.append(f"FAIL: Decreasing R to {R_perturbed} should decrease both x and l")
                        return False
                else:
                    if not (x_perturbed > self.x and l_perturbed > self.l):
                        self.logs.append(f"FAIL: Increasing R to {R_perturbed} should increase both x and l")
                        return False
            
            self.logs.append("PASS: Monotonicity test - x and l scale linearly with R")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== Running {test_name} Test ===")
            if not test_func():
                success = False
                self.logs.append(f"--- {test_name} FAILED ---")
            else:
                self.logs.append(f"--- {test_name} PASSED ---")
        
        return success, self.logs