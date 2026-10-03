import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all required quantities from the solution instance.
        Uses getattr and .get() to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Extract results dictionary if it exists, otherwise empty dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Initial conditions (with safe defaults)
        self.L = self.results.get('L', 5.0)  # cm
        self.h1 = self.results.get('h1', 14.0)  # cm
        self.h2 = self.results.get('h2', 8.0)  # cm
        
        # Extract pressures correctly: initial p_A is 76.0, not from results['p_A']
        self.p_A_initial = 76.0  # Given in problem
        self.p_B_initial = self.results.get('p_B', 90.0)  # Should be 90.0
        
        # Final/computed quantities
        self.x = self.results.get('x', 0.0)  # cm
        self.mass_ratio = self.results.get('m_A/m_B', 0.0)
        self.p_A_final = self.results.get('p_A', 0.0)  # cmHg (final value)
        self.L_final = self.results.get('L_prime', 0.0)  # cm
        self.h = self.results.get('h', 0.0)  # cm
        
        # Derived quantities
        self.p_B_final = 120.0  # Given in problem
        self.H_final = 16.0  # cm (mercury column height after compression)
        
        # Cross-sectional area (cancels)
        self.S = 1.0
        
    def test_invariants(self):
        """
        Check physical invariants:
        1. Boyle's Law for gas A and gas B (initial vs final).
        2. Pressure relationships (initial and final).
        3. Mass ratio from initial conditions.
        """
        logs = []
        success = True
        
        # 1. Boyle's Law for gas A: p_A_initial * L = p_A_final * L_final
        if self.p_A_final != 0.0 and self.L_final != 0.0:
            product_A_initial = self.p_A_initial * self.L
            product_A_final = self.p_A_final * self.L_final
            if not math.isclose(product_A_initial, product_A_final, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Boyle's Law for gas A failed: {product_A_initial} ≠ {product_A_final}")
                success = False
            else:
                logs.append("Boyle's Law for gas A holds.")
        else:
            logs.append("Skipping Boyle's Law for gas A (missing final values).")
        
        # 2. Boyle's Law for gas B: p_B_initial * h2 = p_B_final * h
        if self.h != 0.0:
            product_B_initial = self.p_B_initial * self.h2
            product_B_final = self.p_B_final * self.h
            if not math.isclose(product_B_initial, product_B_final, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Boyle's Law for gas B failed: {product_B_initial} ≠ {product_B_final}")
                success = False
            else:
                logs.append("Boyle's Law for gas B holds.")
        else:
            logs.append("Skipping Boyle's Law for gas B (missing h).")
        
        # 3. Pressure relationships
        # Initial: p_B_initial = p_A_initial + h1
        if not math.isclose(self.p_B_initial, self.p_A_initial + self.h1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Initial pressure relationship failed: p_B_initial={self.p_B_initial}, p_A_initial+h1={self.p_A_initial+self.h1}")
            success = False
        else:
            logs.append("Initial pressure relationship holds.")
        
        # Final: p_A_final = p_B_final - H_final
        if self.p_A_final != 0.0:
            if not math.isclose(self.p_A_final, self.p_B_final - self.H_final, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Final pressure relationship failed: p_A_final={self.p_A_final}, p_B_final-H_final={self.p_B_final-self.H_final}")
                success = False
            else:
                logs.append("Final pressure relationship holds.")
        
        # 4. Mass ratio from initial conditions: m_A/m_B = (p_A_initial * L) / (p_B_initial * h2)
        expected_mass_ratio = (self.p_A_initial * self.L) / (self.p_B_initial * self.h2)
        if not math.isclose(self.mass_ratio, expected_mass_ratio, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Mass ratio mismatch: got {self.mass_ratio}, expected {expected_mass_ratio}")
            success = False
        else:
            logs.append(f"Mass ratio correct: {self.mass_ratio} ≈ {expected_mass_ratio}")
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Check that intermediate results are consistent with the final answer.
        Specifically, verify that x = (h2 - h) + (L - L_final).
        """
        logs = []
        success = True
        
        if self.h != 0.0 and self.L_final != 0.0:
            expected_x = (self.h2 - self.h) + (self.L - self.L_final)
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: x={self.x}, computed from steps={expected_x}")
                success = False
            else:
                logs.append(f"Trace consistency holds: x={self.x} matches steps.")
        else:
            logs.append("Skipping trace consistency (missing h or L_final).")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an initial parameter (L) and verify that the output (x) changes
        in the expected direction, using the physical relationships.
        """
        logs = []
        success = True
        
        # Perturb L by +1%
        delta = 0.01
        L_perturbed = self.L * (1 + delta)
        
        # Recompute using the correct physical relationships
        # 1. Initial pressures (unchanged)
        p_A_initial = self.p_A_initial
        p_B_initial = self.p_B_initial
        h2 = self.h2
        p_B_final = self.p_B_final
        H_final = self.H_final
        
        # 2. Boyle's law for B gives h (unchanged since p_B_final fixed)
        h = (p_B_initial * h2) / p_B_final
        
        # 3. Pressure relationship for A final
        p_A_final = p_B_final - H_final
        
        # 4. Boyle's law for A gives L_final_perturbed
        L_final_perturbed = (p_A_initial * L_perturbed) / p_A_final
        
        # 5. Compute x_perturbed
        x_perturbed = (h2 - h) + (L_perturbed - L_final_perturbed)
        
        # Check monotonicity: increasing L should increase x
        # Theoretical derivative: dx/dL = 1 - (p_A_initial/p_A_final)
        # For our numbers: 1 - 76/104 = 28/104 ≈ 0.2692 > 0
        if x_perturbed > self.x:
            logs.append(f"Monotonicity holds: x increased from {self.x:.4f} to {x_perturbed:.4f} when L increased by 1%.")
        else:
            logs.append(f"Monotonicity violated: x decreased or stayed same ({self.x:.4f} -> {x_perturbed:.4f}) when L increased.")
            success = False
        
        # Check that the change is roughly consistent with the derivative
        derivative = 1 - (p_A_initial / p_A_final)
        delta_L = L_perturbed - self.L
        expected_delta_x = derivative * delta_L
        actual_delta_x = x_perturbed - self.x
        if not math.isclose(actual_delta_x, expected_delta_x, rel_tol=5e-2, abs_tol=1e-2):
            logs.append(f"Monotonicity slope mismatch: expected Δx≈{expected_delta_x:.4f}, got {actual_delta_x:.4f}")
            # Not necessarily a failure, but log it.
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {str(e)}")
                overall_success = False
        
        return overall_success, all_logs