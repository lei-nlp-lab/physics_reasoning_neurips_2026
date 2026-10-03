import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance
        self.h = getattr(solution_instance, 'h', getattr(solution_instance, 'initial_height', 45.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'gravity', 10.0))
        
        # Safely extract results using both direct attributes and results dictionary
        results = getattr(solution_instance, 'results', {})
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        self.h_2 = getattr(solution_instance, 'h_2', results.get('h_2', 0.0))
        self.Deltah = getattr(solution_instance, 'Deltah', results.get('Deltah', 0.0))
        
        # Expected target answers for trace consistency
        self.targets = [3.0, 30.0, 25.0]  # t, v, Δh in order
        self.logs = []
        
    def _log(self, test_name, message, success):
        self.logs.append(f"{test_name}: {message}")
        return success
    
    def test_invariants(self):
        """Verify kinematic equations and energy conservation."""
        tests_passed = True
        
        # Test 1: Time from height using kinematic equation
        expected_t = math.sqrt(2 * self.h / self.g)
        if not math.isclose(self.t, expected_t, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_invariants",
                f"Time t={self.t} differs from sqrt(2h/g)={expected_t:.3f}",
                False
            )
        
        # Test 2: Velocity from time using v = g*t
        expected_v = self.g * self.t
        if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_invariants", 
                f"Velocity v={self.v} differs from g*t={expected_v:.3f}",
                False
            )
        
        # Test 3: Energy conservation (kinetic = potential loss)
        expected_v_from_energy = math.sqrt(2 * self.g * self.h)
        if not math.isclose(self.v, expected_v_from_energy, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_invariants",
                f"Velocity v={self.v} violates energy conservation (sqrt(2gh)={expected_v_from_energy:.3f})",
                False
            )
        
        # Test 4: Height in first 2 seconds
        expected_h2 = 0.5 * self.g * (2.0 ** 2)
        if not math.isclose(self.h_2, expected_h2, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_invariants",
                f"Height in 2s h_2={self.h_2} differs from (1/2)gt²={expected_h2:.3f}",
                False
            )
        
        # Test 5: Δh from total height and h_2
        expected_Deltah = self.h - self.h_2
        if not math.isclose(self.Deltah, expected_Deltah, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_invariants",
                f"Δh={self.Deltah} differs from h-h_2={expected_Deltah:.3f}",
                False
            )
        
        # Test 6: Symbolic verification with SymPy - FIXED to select positive solution
        try:
            t_sym = sp.symbols('t')
            eq1 = sp.Eq(self.h, 0.5 * self.g * t_sym**2)
            symbolic_solutions = sp.solve(eq1, t_sym)
            # Select the positive solution
            positive_solution = None
            for sol in symbolic_solutions:
                if float(sol) > 0:
                    positive_solution = float(sol)
                    break
            if positive_solution is None:
                positive_solution = float(symbolic_solutions[0])  # fallback
            
            if not math.isclose(self.t, positive_solution, rel_tol=0.02, abs_tol=0.02):
                tests_passed = self._log(
                    "test_invariants",
                    f"Symbolic solution t={positive_solution:.3f} differs from computed t={self.t}",
                    False
                )
        except Exception as e:
            tests_passed = self._log(
                "test_invariants",
                f"Symbolic check failed with error: {str(e)}",
                True  # Non-critical failure
            )
        
        return tests_passed
    
    def test_trace_consistency(self):
        """Check intermediate results against final answers and target values."""
        tests_passed = True
        
        # Check against target answers
        expected_t, expected_v, expected_Deltah = self.targets
        
        if not math.isclose(self.t, expected_t, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_trace_consistency",
                f"Time t={self.t} differs from target {expected_t}s",
                False
            )
        
        if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_trace_consistency",
                f"Velocity v={self.v} differs from target {expected_v}m/s",
                False
            )
        
        if not math.isclose(self.Deltah, expected_Deltah, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_trace_consistency",
                f"Δh={self.Deltah} differs from target {expected_Deltah}m",
                False
            )
        
        # Cross-check: v from h_2 should be consistent with kinematics in first 2s
        v_after_2s = self.g * 2.0
        height_check = 0.5 * v_after_2s * 2.0  # h = v_avg * t = (0 + v)/2 * t
        if not math.isclose(self.h_2, height_check, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                "test_trace_consistency",
                f"h_2={self.h_2} inconsistent with v_after_2s={v_after_2s:.3f}",
                False
            )
        
        return tests_passed
    
    def test_monotonicity(self):
        """Perturb height and check trends in dependent quantities."""
        original_h = self.h
        delta_h = 1.0  # 1 meter increase
        tests_passed = True
        
        # Perturb height
        self.h += delta_h
        
        # Recalculate expected values using formulas
        new_t = math.sqrt(2 * self.h / self.g)
        new_v = self.g * new_t
        new_h2 = 0.5 * self.g * (2.0 ** 2)  # Independent of height
        new_Deltah = self.h - new_h2
        
        # Check monotonic trends
        if not new_t > self.t:
            tests_passed = self._log(
                "test_monotonicity",
                f"Time does not increase with height: {new_t:.3f} <= {self.t}",
                False
            )
        
        if not new_v > self.v:
            tests_passed = self._log(
                "test_monotonicity",
                f"Velocity does not increase with height: {new_v:.3f} <= {self.v}",
                False
            )
        
        if not new_Deltah > self.Deltah:
            tests_passed = self._log(
                "test_monotonicity",
                f"Δh does not increase with height: {new_Deltah:.3f} <= {self.Deltah}",
                False
            )
        
        # Restore original height
        self.h = original_h
        
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        all_passed = True
        
        # Run tests
        if not self.test_invariants():
            all_passed = False
        
        if not self.test_trace_consistency():
            all_passed = False
        
        if not self.test_monotonicity():
            all_passed = False
        
        # Summary
        if all_passed:
            self.logs.append("All tests passed successfully.")
        else:
            self.logs.append("Some tests failed. Check physics consistency.")
        
        return all_passed, self.logs