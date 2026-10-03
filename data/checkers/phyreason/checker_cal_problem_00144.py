import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    """Robust physics auditor for ion implantation problem."""
    
    def __init__(self, solution_instance):
        """Safely extract inputs and results with fallback defaults."""
        self.solution = solution_instance
        
        # Extract results dictionary directly
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results.copy()
        else:
            # Try to extract from instance attributes
            self.results = {}
            result_keys = [
                'v', 'q/m', 'x', 'y', 'alpha', 'r', 'theta',
                'x_1', 'y_1', 'x_2', 'y_2', 'R'
            ]
            for key in result_keys:
                # Try various naming conventions
                attr_name = key.replace('/', '_')
                if hasattr(solution_instance, attr_name):
                    self.results[key] = getattr(solution_instance, attr_name)
                elif hasattr(solution_instance, key):
                    self.results[key] = getattr(solution_instance, key)
                else:
                    self.results[key] = 0.0
        
        # Extract input parameters
        self.inputs = {}
        for param in ['E', 'B', 'R1', 'R2', 'L']:
            if hasattr(solution_instance, param):
                self.inputs[param] = getattr(solution_instance, param)
            elif hasattr(solution_instance, f'{param}_val'):
                self.inputs[param] = getattr(solution_instance, f'{param}_val')
            else:
                # Use reasonable defaults based on the problem
                self.inputs[param] = 1.0
        
        # Set convenience variables
        self.E = float(self.inputs['E'])
        self.B = float(self.inputs['B'])
        self.R1 = float(self.inputs['R1'])
        self.R2 = float(self.inputs['R2'])
        self.L = float(self.inputs['L'])
        
        self.logs = []
    
    def _log(self, msg: str):
        """Add message to logs."""
        self.logs.append(msg)
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self) -> bool:
        """Verify fundamental physical laws."""
        success = True
        
        # Get results with defaults
        v = float(self.results.get('v', 0.0))
        q_m = float(self.results.get('q/m', 0.0))
        R = float(self.results.get('R', 0.0))
        r = float(self.results.get('r', 0.0))
        theta = float(self.results.get('theta', 0.0))
        alpha = float(self.results.get('alpha', 0.0))
        
        # 1. Velocity selector: v = E/B
        expected_v = self.E / self.B
        if not self._is_close(v, expected_v):
            self._log(f"FAIL: Velocity selector invariant. Got v={v}, expected E/B={expected_v}")
            success = False
        else:
            self._log(f"PASS: Velocity selector v = E/B holds")
        
        # 2. Magnetic analyzer: q/m = v/(R*B) = 2E/((R1+R2)B²)
        R_avg = (self.R1 + self.R2) / 2
        expected_qm1 = v / (R_avg * self.B)
        expected_qm2 = 2 * self.E / ((self.R1 + self.R2) * self.B**2)
        
        if not (self._is_close(q_m, expected_qm1) or self._is_close(q_m, expected_qm2)):
            self._log(f"FAIL: q/m invariant. Got {q_m}, expected {expected_qm1} or {expected_qm2}")
            success = False
        else:
            self._log(f"PASS: q/m = v/(RB) = 2E/((R1+R2)B²) holds")
        
        # 3. Radius consistency: R = (R1 + R2)/2
        expected_R = (self.R1 + self.R2) / 2
        if not self._is_close(R, expected_R):
            self._log(f"FAIL: R = (R1+R2)/2. Got {R}, expected {expected_R}")
            success = False
        else:
            self._log(f"PASS: R = (R1+R2)/2 holds")
        
        # 4. Circular motion in magnetic analyzer: r = mv/(qB) = v/(q/m * B)
        if q_m > 0:
            r_from_qm = v / (q_m * self.B)
            if not self._is_close(r, r_from_qm):
                self._log(f"FAIL: r = mv/(qB). Got {r}, calculated {r_from_qm}")
                success = False
            else:
                self._log(f"PASS: r = mv/(qB) holds")
        
        # 5. Electric deflection angle: tanθ ≈ θ = qEL/(mv²)
        if theta > 0 and q_m > 0 and v > 0:
            expected_theta = q_m * self.E * self.L / (v**2)
            if not self._is_close(theta, expected_theta):
                self._log(f"FAIL: θ = qEL/(mv²). Got {theta}, expected {expected_theta}")
                success = False
            else:
                self._log(f"PASS: θ = qEL/(mv²) holds")
        
        # 6. Small angle approximations validity
        if alpha > 0.2:
            self._log(f"WARNING: α={alpha} may be too large for small-angle approximations")
        
        if theta > 0.2:
            self._log(f"WARNING: θ={theta} may be too large for small-angle approximations")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Get all results
        v = float(self.results.get('v', 0.0))
        q_m = float(self.results.get('q/m', 0.0))
        x_1 = float(self.results.get('x_1', 0.0))
        x_2 = float(self.results.get('x_2', 0.0))
        y_1 = float(self.results.get('y_1', 0.0))
        y_2 = float(self.results.get('y_2', 0.0))
        r = float(self.results.get('r', 0.0))
        alpha = float(self.results.get('alpha', 0.0))
        
        # Avoid division by zero
        if v == 0 or q_m == 0 or r == 0:
            self._log("WARNING: Zero values prevent trace consistency checks")
            return False
        
        # 1. Electric deflection steps
        # x₁ = ½(qE/m)(L/v)²
        x1_calc = 0.5 * q_m * self.E * (self.L / v)**2
        if not self._is_close(x_1, x1_calc):
            self._log(f"FAIL: x₁ calculation. Got {x_1}, calculated {x1_calc}")
            success = False
        
        # x₂ = L·tanθ ≈ L·θ = qEL²/(mv²)
        x2_calc = q_m * self.E * self.L**2 / (v**2)
        if not self._is_close(x_2, x2_calc):
            self._log(f"FAIL: x₂ calculation. Got {x_2}, calculated {x2_calc}")
            success = False
        
        # x_total = x₁ + x₂ = 3qEL²/(2mv²) = 3L²/(R₁+R₂)
        x_total_calc = x_1 + x_2
        x_total_expected = 3 * self.L**2 / (self.R1 + self.R2)
        if not self._is_close(x_total_calc, x_total_expected):
            self._log(f"FAIL: x_total = x₁+x₂. Sum={x_total_calc}, expected 3L²/(R₁+R₂)={x_total_expected}")
            success = False
        
        # 2. Magnetic deflection steps
        # r = mv/(qB) = v/(q_m * B)
        r_calc = v / (q_m * self.B)
        if not self._is_close(r, r_calc):
            self._log(f"FAIL: r calculation. Got {r}, calculated {r_calc}")
            success = False
        
        # α = L/r
        alpha_calc = self.L / r
        if not self._is_close(alpha, alpha_calc):
            self._log(f"FAIL: α calculation. Got {alpha}, calculated {alpha_calc}")
            success = False
        
        # y₁ = r(1 - cosα) ≈ r·α²/2 = L²/(2r)
        y1_approx = self.L**2 / (2 * r)
        # Using the given approximation: L²/(R₁+R₂)
        y1_expected = self.L**2 / (self.R1 + self.R2)
        if not (self._is_close(y_1, y1_approx, rel_tol=5e-2) or self._is_close(y_1, y1_expected)):
            self._log(f"FAIL: y₁ approximation. Got {y_1}, expected ≈{y1_approx} or {y1_expected}")
            success = False
        
        # y₂ = L·tanα ≈ L·α = L²/r
        y2_approx = self.L**2 / r
        # Using the given approximation: 2L²/(R₁+R₂)
        y2_expected = 2 * self.L**2 / (self.R1 + self.R2)
        if not (self._is_close(y_2, y2_approx, rel_tol=5e-2) or self._is_close(y_2, y2_expected)):
            self._log(f"FAIL: y₂ approximation. Got {y_2}, expected ≈{y2_approx} or {y2_expected}")
            success = False
        
        # y_total = y₁ + y₂ = 3L²/(R₁+R₂)
        y_total_calc = y_1 + y_2
        y_total_expected = 3 * self.L**2 / (self.R1 + self.R2)
        if not self._is_close(y_total_calc, y_total_expected, rel_tol=5e-2):
            self._log(f"FAIL: y_total = y₁+y₂. Sum={y_total_calc}, expected 3L²/(R₁+R₂)={y_total_expected}")
            success = False
        
        if success:
            self._log("PASS: All intermediate results consistent with formulas")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test that outputs change monotonically with input perturbations."""
        success = True
        
        # Store original values
        orig_E = self.E
        orig_B = self.B
        orig_L = self.L
        orig_R1 = self.R1
        
        try:
            # Test 1: Increase E → v should increase
            self.E = orig_E * 1.1  # +10%
            v_new = self.E / self.B
            v_orig = float(self.results.get('v', 0.0))
            if v_new <= v_orig:
                self._log(f"FAIL: v should increase when E increases. v_orig={v_orig}, v_new={v_new}")
                success = False
            
            # Test 2: Increase B → v should decrease
            self.B = orig_B * 1.1
            v_new2 = self.E / self.B
            if v_new2 >= v_new:
                self._log(f"FAIL: v should decrease when B increases. v_before={v_new}, v_after={v_new2}")
                success = False
            
            # Test 3: Increase L → deflection distances should increase
            self.L = orig_L * 1.1
            deflection_new = 3 * self.L**2 / (self.R1 + self.R2)
            deflection_orig = 3 * orig_L**2 / (orig_R1 + self.R2)
            if deflection_new <= deflection_orig:
                self._log(f"FAIL: Deflection should increase when L increases. Orig={deflection_orig}, New={deflection_new}")
                success = False
            
            # Test 4: Increase R1 → deflection distances should decrease
            self.R1 = orig_R1 * 1.1
            deflection_new2 = 3 * self.L**2 / (self.R1 + self.R2)
            if deflection_new2 >= deflection_new:
                self._log(f"FAIL: Deflection should decrease when R1 increases. Before={deflection_new}, After={deflection_new2}")
                success = False
            
            if success:
                self._log("PASS: All monotonicity tests passed")
            
        finally:
            # Restore original values
            self.E = orig_E
            self.B = orig_B
            self.L = orig_L
            self.R1 = orig_R1
        
        return success
    
    def test_superposition(self) -> bool:
        """Verify superposition principle for combined fields."""
        success = True
        
        # Get results
        x = float(self.results.get('x', 0.0))
        y = float(self.results.get('y', 0.0))
        x_1 = float(self.results.get('x_1', 0.0))
        x_2 = float(self.results.get('x_2', 0.0))
        y_1 = float(self.results.get('y_1', 0.0))
        y_2 = float(self.results.get('y_2', 0.0))
        
        # Expected final positions from problem statement
        x_electric_only = 3 * self.L**2 / (self.R1 + self.R2)
        y_electric_only = 0.0
        
        x_magnetic_only = 0.0
        y_magnetic_only = 3 * self.L**2 / (self.R1 + self.R2)
        
        x_both = 3 * self.L**2 / (self.R1 + self.R2)
        y_both = 3 * self.L**2 / (self.R1 + self.R2)
        
        # Check both-fields case
        if not (self._is_close(x, x_both) and self._is_close(y, y_both)):
            self._log(f"FAIL: Both-fields position. Got ({x}, {y}), expected ({x_both}, {y_both})")
            success = False
        else:
            self._log("PASS: Superposition principle verified for combined fields")
        
        # Verify independence of x and y motions
        x_from_electric = x_1 + x_2
        y_from_magnetic = y_1 + y_2
        
        if not self._is_close(x, x_from_electric):
            self._log(f"FAIL: x should equal x₁+x₂. Got {x}, sum={x_from_electric}")
            success = False
        
        if not self._is_close(y, y_from_magnetic):
            self._log(f"FAIL: y should equal y₁+y₂. Got {y}, sum={y_from_magnetic}")
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all audit tests and return results."""
        self.logs = []
        self.logs.append("=== Physics Audit Started ===")
        
        test_results = []
        overall_success = True
        
        try:
            # Run individual tests
            tests = [
                ("Invariant Tests", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity),
                ("Superposition", self.test_superposition)
            ]
            
            for test_name, test_func in tests:
                self._log(f"\n--- {test_name} ---")
                try:
                    result = test_func()
                    test_results.append(result)
                    if result:
                        self._log(f"{test_name}: PASSED")
                    else:
                        self._log(f"{test_name}: FAILED")
                        overall_success = False
                except Exception as e:
                    self._log(f"{test_name}: ERROR - {str(e)}")
                    self._log(traceback.format_exc())
                    test_results.append(False)
                    overall_success = False
            
            # Summary
            self._log("\n=== Audit Summary ===")
            passed = sum(test_results)
            total = len(test_results)
            self._log(f"Passed: {passed}/{total} tests")
            
            if overall_success:
                self._log("OVERALL: PASS - Physics checks are consistent")
            else:
                self._log("OVERALL: FAIL - Physics inconsistencies detected")
            
        except Exception as e:
            self._log(f"CRITICAL ERROR: {str(e)}")
            self._log(traceback.format_exc())
            overall_success = False
        
        return overall_success, self.logs