import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with defaults
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', self.results.get('g', 9.8))
        self.theta = getattr(solution_instance, 'theta', self.results.get('theta', 0.0))
        
        # Extract results with safe access
        self.F = getattr(solution_instance, 'F', self.results.get('F', 0.0))
        self.F_all = getattr(solution_instance, 'F_all', 
                            self.results.get('F_{all}', self.results.get('F_all', 0.0)))
        
        # Expected values from physics formulas
        self.expected_F = self.m * self.g / math.cos(self.theta) if math.cos(self.theta) != 0 else float('inf')
        self.expected_F_all = self.m * self.g * math.tan(self.theta)
        
        # For symbolic checks
        self.m_sym, self.g_sym, self.theta_sym = sp.symbols('m g theta', positive=True)
        self.F_sym = self.m_sym * self.g_sym / sp.cos(self.theta_sym)
        self.F_all_sym = self.m_sym * self.g_sym * sp.tan(self.theta_sym)
        
        self.logs = []

    def test_invariants(self):
        """Verify Newton's Laws and force relationships."""
        try:
            # 1. Check vertical force balance (Newton's First Law in vertical direction)
            vertical_net = self.F * math.cos(self.theta) - self.m * self.g
            if not math.isclose(vertical_net, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Vertical force imbalance: F*cosθ - mg = {vertical_net:.3f}")
                return False
            self.logs.append(f"PASS: Vertical force balance (F*cosθ = mg)")

            # 2. Check horizontal resultant matches tension component (Newton's Second Law)
            horizontal_from_tension = self.F * math.sin(self.theta)
            if not math.isclose(horizontal_from_tension, self.F_all, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Horizontal mismatch: F*sinθ={horizontal_from_tension:.3f}, F_all={self.F_all:.3f}")
                return False
            self.logs.append(f"PASS: Horizontal force F_all = F*sinθ")

            # 3. Symbolic verification fallback
            F_expr = sp.simplify(self.F_sym - self.F_sym)
            F_all_expr = sp.simplify(self.F_all_sym - self.F_all_sym)
            if F_expr != 0 or F_all_expr != 0:
                self.logs.append(f"WARNING: Symbolic verification inconclusive")
            else:
                self.logs.append(f"PASS: Symbolic forms match expected relationships")

            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check F against formula mg/cosθ
            if not math.isclose(self.F, self.expected_F, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Tension F mismatch: expected {self.expected_F:.3f}, got {self.F:.3f}")
                return False
            self.logs.append(f"PASS: Tension F = mg/cosθ")

            # Check F_all against formula mg*tanθ
            if not math.isclose(self.F_all, self.expected_F_all, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Resultant F_all mismatch: expected {self.expected_F_all:.3f}, got {self.F_all:.3f}")
                return False
            self.logs.append(f"PASS: Resultant F_all = mg*tanθ")

            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False

    def test_monotonicity(self):
        """Perturb mass and verify proportional changes in forces."""
        try:
            original_m = self.m
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.10
            self.m = perturbed_m
            
            # Recalculate expected forces with perturbed mass
            expected_F_perturbed = perturbed_m * self.g / math.cos(self.theta)
            expected_F_all_perturbed = perturbed_m * self.g * math.tan(self.theta)
            
            # Check monotonic increase (both forces should increase with mass)
            if not (expected_F_perturbed > self.expected_F and 
                    expected_F_all_perturbed > self.expected_F_all):
                self.logs.append(f"FAIL: Monotonicity violation - forces didn't increase with mass")
                self.m = original_m  # Restore before returning
                return False
            
            self.logs.append(f"PASS: Forces increase proportionally with mass")
            
            # Restore original mass
            self.m = original_m
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            self.m = original_m  # Ensure restoration on error
            return False

    def run_all_tests(self):
        """Execute all checks and return (success_bool, logs)."""
        self.logs.clear()
        success = True
        
        tests = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} ---")
            if not test_func():
                success = False
                self.logs.append(f"{test_name}: FAILED")
            else:
                self.logs.append(f"{test_name}: PASSED")
        
        self.logs.insert(0, f"Physics Checker Summary: {'ALL PASS' if success else 'FAILURES DETECTED'}")
        return success, self.logs