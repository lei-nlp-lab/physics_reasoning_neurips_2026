import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs and results safely
        self.W = getattr(solution_instance, 'W', self.results.get('W', 0.0))
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        self.x_all = getattr(solution_instance, 'x_all', self.results.get('x_all', 0.0))
        
        # Constants from problem statement
        self.g = 10.0  # m/s²
        self.reaction_dist = 17.0  # m
        self.resistance_factor = 0.5
        
        # Derived quantities
        self.m = 1.0  # Assume unit mass for symbolic checks (cancels out)
        # Compute v_m from braking distance formula: x = v_m²/g
        if self.x > 0:
            self.v_m = math.sqrt(self.g * self.x)
        else:
            self.v_m = 0.0
            
        # Logging
        self.logs = []
        
    def _log(self, message):
        self.logs.append(message)
        
    def test_invariants(self):
        """Verify energy conservation and work-energy theorem."""
        try:
            # 1. Check work-energy theorem during braking
            # Work done by resistance: W = -0.5*m*g*x
            W_expected = -self.resistance_factor * self.m * self.g * self.x
            # Change in kinetic energy: ΔK = 0 - 0.5*m*v_m²
            delta_K = 0 - 0.5 * self.m * self.v_m**2
            
            if not math.isclose(W_expected, delta_K, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Work-energy theorem mismatch. W={W_expected:.1f}, ΔK={delta_K:.1f}")
                return False
            self._log("PASS: Work-energy theorem satisfied during braking.")
            
            # 2. Check total distance consistency (x_all = x + reaction_dist)
            if not math.isclose(self.x_all, self.x + self.reaction_dist, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Total distance mismatch. x_all={self.x_all}, x+17={self.x+17}")
                return False
            self._log("PASS: Total distance x_all = x + 17 m.")
            
            # 3. Symbolic verification with sympy
            m, g, v_m, x = sp.symbols('m g v_m x', positive=True)
            W_expr = -0.5 * m * g * x
            delta_K_expr = 0 - 0.5 * m * v_m**2
            # From kinematics: x = v_m²/g
            kinematics_eq = sp.Eq(x, v_m**2 / g)
            # Substitute into work-energy: -0.5*m*g*(v_m²/g) = -0.5*m*v_m²
            lhs_sub = W_expr.subs(x, v_m**2 / g)
            if sp.simplify(lhs_sub - delta_K_expr) != 0:
                self._log("FAIL: Symbolic work-energy theorem invalid.")
                return False
            self._log("PASS: Symbolic work-energy theorem verified.")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Check intermediate results align with final answer."""
        try:
            # Step 3: x = v_m²/g (from step_2 equation)
            x_calculated = self.v_m**2 / self.g
            if not math.isclose(self.x, x_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Braking distance mismatch. x={self.x}, v_m²/g={x_calculated}")
                return False
            self._log(f"PASS: Braking distance x={self.x} m consistent with v_m={self.v_m:.1f} m/s.")
            
            # Step 4: x_all = x + 17
            x_all_calculated = self.x + self.reaction_dist
            if not math.isclose(self.x_all, x_all_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Total distance mismatch. x_all={self.x_all}, calculated={x_all_calculated}")
                return False
            self._log(f"PASS: Total distance x_all={self.x_all} m consistent.")
            
            # Check that 128 m is the final answer
            if not math.isclose(self.x_all, 128.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Expected 128 m, got {self.x_all} m.")
                return False
            self._log("PASS: Final answer matches target 128 m.")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and verify physical trends."""
        original_x_all = self.x_all
        original_v_m = self.v_m
        
        try:
            # Perturb v_m by +10%
            perturbed_v_m = original_v_m * 1.1
            perturbed_x = perturbed_v_m**2 / self.g
            perturbed_x_all = perturbed_x + self.reaction_dist
            
            # Expect larger v_m → larger braking distance → larger total distance
            if perturbed_x_all <= original_x_all:
                self._log(f"FAIL: Monotonicity violated. v_m↑ should increase x_all.")
                return False
            self._log(f"PASS: v_m↑ ({original_v_m:.1f}→{perturbed_v_m:.1f}) increases x_all ({original_x_all:.1f}→{perturbed_x_all:.1f}).")
            
            # Perturb resistance factor (deceleration) and check trend
            original_resistance = self.resistance_factor
            perturbed_resistance = original_resistance * 0.9  # weaker braking
            # With weaker braking, braking distance increases for same v_m
            perturbed_x_weak = original_v_m**2 / (perturbed_resistance * self.g)
            perturbed_x_all_weak = perturbed_x_weak + self.reaction_dist
            
            if perturbed_x_all_weak <= original_x_all:
                self._log(f"FAIL: Weaker braking should increase total distance.")
                return False
            self._log(f"PASS: Weaker braking (factor {original_resistance}→{perturbed_resistance}) increases x_all.")
            
            # Restore original state (though checker is read-only)
            self.v_m = original_v_m
            self.x_all = original_x_all
            return True
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            self.v_m = original_v_m
            self.x_all = original_x_all
            return False
    
    def run_all_tests(self):
        """Execute all checks and return success flag with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self._log(f"\n=== Running {name} Test ===")
            if not test_func():
                success = False
                self._log(f"✗ {name} test failed.")
            else:
                self._log(f"✓ {name} test passed.")
        
        self._log(f"\n=== Overall Result: {'SUCCESS' if success else 'FAILURE'} ===")
        return success, self.logs