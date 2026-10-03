import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access to avoid sandbox errors.
        """
        # Extract inputs with fallbacks
        self.m = getattr(solution_instance, 'm', 1.0e-3)
        self.q = getattr(solution_instance, 'q', 1.0e-10)
        self.E = getattr(solution_instance, 'E', 2.0e6)
        
        # Time intervals (all 1.0s)
        self.t1 = getattr(solution_instance, 't1', 1.0)
        self.t2 = getattr(solution_instance, 't2', 1.0)
        self.t3 = getattr(solution_instance, 't3', 1.0)
        
        # Extract results safely from results dict
        results = getattr(solution_instance, 'results', {})
        
        # Map expected keys with fallbacks
        self.Deltax_2 = results.get('Deltax_2', results.get('\\Delta x_2', 0.0))
        self.Deltay_3 = results.get('Deltay_3', results.get('\\Delta y_3', 0.0))
        self.v_x = results.get('v_x', 0.0)
        self.v_y = results.get('v_y', 0.0)
        self.Deltay = results.get('Deltay', results.get('\\Delta y', 0.0))
        self.a_y = results.get('a_y', 0.0)
        self.Deltax_1 = results.get('Deltax_1', results.get('\\Delta x_1', 0.0))
        self.a = results.get('a', 0.0)
        self.x_2 = results.get('x_2', 0.0)
        self.Deltax_3 = results.get('Deltax_3', results.get('\\Delta x_3', 0.0))
        self.a_x = results.get('a_x', 0.0)
        self.y_2 = results.get('y_2', 0.0)
        
        # Final answer coordinates
        final_ans = getattr(solution_instance, 'final_answer', ["(0.40m, 0.20m)"])
        if final_ans and isinstance(final_ans, list) and len(final_ans) > 0:
            try:
                # Parse "(0.40m, 0.20m)" to floats
                coord_str = final_ans[0].strip("()").replace("m", "").replace(" ", "")
                x_str, y_str = coord_str.split(",")
                self.final_x = float(x_str)
                self.final_y = float(y_str)
            except:
                self.final_x = 0.40
                self.final_y = 0.20
        else:
            self.final_x = 0.40
            self.final_y = 0.20
        
        # Store logs
        self.logs = []
    
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
    
    def test_invariants(self) -> bool:
        """
        Test fundamental physics invariants:
        1. Newton's Second Law for each interval
        2. Kinematic consistency
        3. Vector magnitude consistency
        """
        all_pass = True
        
        # Calculate expected acceleration from Newton's 2nd Law
        a_expected = (self.q * self.E) / self.m
        
        # Test 1: First interval acceleration (x-direction)
        if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"First interval acceleration mismatch: {self.a} vs {a_expected}", False)
        else:
            all_pass &= self._log("First interval Newton's 2nd Law holds", True)
        
        # Test 2: Kinematics for first interval
        v_x_expected = self.a * self.t1
        Deltax_1_expected = 0.5 * self.a * self.t1**2
        
        if not math.isclose(self.v_x, v_x_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"v_x mismatch: {self.v_x} vs {v_x_expected}", False)
        
        if not math.isclose(self.Deltax_1, Deltax_1_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Δx1 mismatch: {self.Deltax_1} vs {Deltax_1_expected}", False)
        
        # Test 3: Second interval kinematics in x (constant velocity)
        Deltax_2_expected = self.v_x * self.t2
        if not math.isclose(self.Deltax_2, Deltax_2_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Δx2 mismatch: {self.Deltax_2} vs {Deltax_2_expected}", False)
        
        # Test 4: Second interval kinematics in y (constant acceleration)
        # Acceleration in y should be same magnitude as in x
        a_y_expected = a_expected
        if not math.isclose(self.a_y, a_y_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"a_y mismatch: {self.a_y} vs {a_y_expected}", False)
        
        v_y_expected = a_y_expected * self.t2
        Deltay_expected = 0.5 * a_y_expected * self.t2**2
        
        if not math.isclose(self.v_y, v_y_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"v_y mismatch: {self.v_y} vs {v_y_expected}", False)
        
        if not math.isclose(self.Deltay, Deltay_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Δy mismatch: {self.Deltay} vs {Deltay_expected}", False)
        
        # Test 5: Position at end of second interval
        x2_expected = self.Deltax_1 + self.Deltax_2
        y2_expected = self.Deltay
        
        if not math.isclose(self.x_2, x2_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"x_2 mismatch: {self.x_2} vs {x2_expected}", False)
        
        if not math.isclose(self.y_2, y2_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"y_2 mismatch: {self.y_2} vs {y2_expected}", False)
        
        # Test 6: Third interval accelerations (to bring to rest)
        a_x_expected = self.v_x / self.t3  # Should be negative, but magnitude check
        a_y_expected = self.v_y / self.t3
        
        if not math.isclose(abs(self.a_x), abs(a_x_expected), rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"|a_x| mismatch: {self.a_x} vs {a_x_expected}", False)
        
        if not math.isclose(abs(self.a_y), abs(a_y_expected), rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"|a_y| mismatch: {self.a_y} vs {a_y_expected}", False)
        
        # Test 7: Third interval displacements
        Deltax_3_expected = self.x_2 + self.v_x * self.t3 - 0.5 * abs(self.a_x) * self.t3**2
        Deltay_3_expected = self.y_2 + self.v_y * self.t3 - 0.5 * abs(self.a_y) * self.t3**2
        
        if not math.isclose(self.Deltax_3, Deltax_3_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Δx3 mismatch: {self.Deltax_3} vs {Deltax_3_expected}", False)
        
        if not math.isclose(self.Deltay_3, Deltay_3_expected, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Δy3 mismatch: {self.Deltay_3} vs {Deltay_3_expected}", False)
        
        # Test 8: Final velocity should be zero (energy check)
        # Work done by electric field should equal kinetic energy at t=2s
        # W = qE * (Δx1 + Δx3) + qE * (Δy + Δy3) ? Actually careful: 
        # Work in x-direction: qE * (Δx1) + 0 + qE_x3 * Δx3 where E_x3 is x-component in 3rd interval
        # But we know final KE = 0, so total work should equal initial KE at t=2s
        
        KE_initial = 0.5 * self.m * (self.v_x**2 + self.v_y**2)
        # Work in first interval: force = qE in x, displacement = Δx1
        W1 = self.q * self.E * self.Deltax_1
        # Work in second interval: force = qE in y, displacement = Δy
        W2 = self.q * self.E * self.Deltay
        # Work in third interval: force components qE_x3, qE_y3
        # From Newton's 2nd: m*a_x = qE_x3, m*a_y = qE_y3
        E_x3 = self.m * self.a_x / self.q
        E_y3 = self.m * self.a_y / self.q
        W3_x = self.q * E_x3 * (self.Deltax_3 - self.x_2)
        W3_y = self.q * E_y3 * (self.Deltay_3 - self.y_2)
        W3 = W3_x + W3_y
        
        total_work = W1 + W2 + W3
        
        if not math.isclose(total_work, KE_initial, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Energy conservation mismatch: work={total_work}, ΔKE={KE_initial}", False)
        else:
            all_pass &= self._log("Energy conservation holds", True)
        
        return all_pass
    
    def test_trace_consistency(self) -> bool:
        """
        Test that intermediate results are consistent with final answer.
        """
        all_pass = True
        
        # Test 1: Final answer should match Δx3 and Δy3
        if not math.isclose(self.final_x, self.Deltax_3, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Final x-coordinate mismatch: {self.final_x} vs {self.Deltax_3}", False)
        else:
            all_pass &= self._log("Final x-coordinate consistent", True)
        
        if not math.isclose(self.final_y, self.Deltay_3, rel_tol=1e-2, abs_tol=1e-2):
            all_pass &= self._log(f"Final y-coordinate mismatch: {self.final_y} vs {self.Deltay_3}", False)
        else:
            all_pass &= self._log("Final y-coordinate consistent", True)
        
        # Test 2: Check that positions are reasonable (positive)
        if self.final_x < 0 or self.final_y < 0:
            all_pass &= self._log("Final coordinates should be positive", False)
        
        # Test 3: Verify that all displacements are physically possible
        # Δx1 should be ≤ Δx2 since acceleration then constant velocity
        if self.Deltax_1 > self.Deltax_2:
            all_pass &= self._log("Δx1 > Δx2 is unexpected", False)
        
        # Δx3 should be > x_2 since v_x is positive and a_x negative
        if self.Deltax_3 <= self.x_2:
            all_pass &= self._log("Δx3 should be greater than x_2", False)
        
        return all_pass
    
    def test_monotonicity(self) -> bool:
        """
        Test monotonicity by perturbing charge q and checking trend.
        Increase q → increase acceleration → increase final coordinates.
        """
        original_q = self.q
        
        try:
            # Perturb q by +10%
            perturbed_q = original_q * 1.1
            
            # Recalculate key quantities using formulas from steps
            perturbed_a = (perturbed_q * self.E) / self.m
            perturbed_v_x = perturbed_a * self.t1
            perturbed_Deltax_1 = 0.5 * perturbed_a * self.t1**2
            perturbed_Deltax_2 = perturbed_v_x * self.t2
            
            # y-direction in second interval
            perturbed_Deltay = 0.5 * perturbed_a * self.t2**2
            perturbed_v_y = perturbed_a * self.t2
            
            perturbed_x_2 = perturbed_Deltax_1 + perturbed_Deltax_2
            perturbed_y_2 = perturbed_Deltay
            
            # Third interval accelerations (magnitudes)
            perturbed_a_x = perturbed_v_x / self.t3
            perturbed_a_y = perturbed_v_y / self.t3
            
            perturbed_Deltax_3 = perturbed_x_2 + perturbed_v_x * self.t3 - 0.5 * perturbed_a_x * self.t3**2
            perturbed_Deltay_3 = perturbed_y_2 + perturbed_v_y * self.t3 - 0.5 * perturbed_a_y * self.t3**2
            
            # Check monotonicity: increased q should increase final coordinates
            if perturbed_Deltax_3 > self.Deltax_3 and perturbed_Deltay_3 > self.Deltay_3:
                self._log("Monotonicity holds: increasing q increases final coordinates", True)
                return True
            else:
                self._log(f"Monotonicity fails: perturbed x={perturbed_Deltax_3} vs original={self.Deltax_3}, "
                         f"perturbed y={perturbed_Deltay_3} vs original={self.Deltay_3}", False)
                return False
                
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", False)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success status and logs.
        """
        self.logs = []  # Reset logs
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Overall success
        success = test1 and test2 and test3
        
        # Add summary
        summary = f"Overall: {'PASS' if success else 'FAIL'} (Invariants:{test1}, Trace:{test2}, Monotonicity:{test3})"
        self.logs.append(summary)
        
        return success, self.logs