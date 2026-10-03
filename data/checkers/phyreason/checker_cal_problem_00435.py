import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract inputs and results safely using getattr/dict.get."""
        self.solution = solution_instance
        
        # Extract given inputs (known forces and angle)
        self.F1 = 20.0  # Given
        self.F2 = 6.0   # Given
        self.F3 = 10.0  # Given
        self.theta_deg = 37.0  # Given
        self.theta_rad = math.radians(self.theta_deg)
        self.cos_theta = 0.8  # Given cos37°
        self.sin_theta = 0.6  # Given sin37°
        
        # Extract results from solution with safe fallbacks
        self.F_x = self._safe_get('F_x', 0.0)
        self.F_y = self._safe_get('F_y', 0.0)
        self.F_mag = self._safe_get('F', 0.0)
        self.phi_deg = self._safe_get('phi', 0.0)
        self.phi_rad = math.radians(self.phi_deg) if self.phi_deg else 0.0
        
    def _safe_get(self, key, default):
        """Safely retrieve attribute from solution instance or results dict."""
        # First try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) else float(val)
        # Then try results dict
        if hasattr(self.solution, 'results'):
            val = self.solution.results.get(key, default)
            return float(val) if isinstance(val, (int, float, str)) else float(val)
        return float(default)
    
    def test_invariants(self):
        """Test fundamental physics laws: vector addition and trigonometry."""
        logs = []
        success = True
        
        try:
            # 1. Verify x-component decomposition
            expected_F_x = self.F1 * self.cos_theta - self.F3  # F1 right, F3 left
            if not math.isclose(self.F_x, expected_F_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_x = {self.F_x}, expected {expected_F_x}")
                success = False
            else:
                logs.append(f"PASS: F_x decomposition correct")
            
            # 2. Verify y-component decomposition
            expected_F_y = self.F1 * self.sin_theta - self.F2  # F1 up, F2 down
            if not math.isclose(self.F_y, expected_F_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_y = {self.F_y}, expected {expected_F_y}")
                success = False
            else:
                logs.append(f"PASS: F_y decomposition correct")
            
            # 3. Verify Pythagorean magnitude
            expected_F_mag = math.sqrt(self.F_x**2 + self.F_y**2)
            if not math.isclose(self.F_mag, expected_F_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_magnitude = {self.F_mag}, expected {expected_F_mag}")
                success = False
            else:
                logs.append(f"PASS: F magnitude via Pythagorean theorem correct")
            
            # 4. Verify angle calculation (handle division by zero)
            if abs(self.F_x) > 1e-6:
                expected_phi_rad = math.atan2(self.F_y, self.F_x)
                expected_phi_deg = math.degrees(expected_phi_rad)
                if not math.isclose(self.phi_deg, expected_phi_deg, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: φ = {self.phi_deg}°, expected {expected_phi_deg}°")
                    success = False
                else:
                    logs.append(f"PASS: Angle φ = arctan(F_y/F_x) correct")
            else:
                logs.append("SKIP: F_x = 0, cannot compute angle")
                
            # 5. Symbolic check (as additional verification)
            F1_x_sym = self.F1 * sp.cos(self.theta_rad)
            F1_y_sym = self.F1 * sp.sin(self.theta_rad)
            F_net_x_sym = F1_x_sym - self.F3  # F3 left
            F_net_y_sym = F1_y_sym - self.F2  # F2 down
            F_net_mag_sym = sp.sqrt(F_net_x_sym**2 + F_net_y_sym**2)
            
            # Evaluate numerically
            F_net_x_val = float(F_net_x_sym.evalf())
            F_net_y_val = float(F_net_y_sym.evalf())
            F_net_mag_val = float(F_net_mag_sym.evalf())
            
            if (math.isclose(self.F_x, F_net_x_val, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.F_y, F_net_y_val, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.F_mag, F_net_mag_val, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append("PASS: Symbolic vector addition consistent")
            else:
                logs.append(f"WARN: Symbolic check shows small discrepancies")
                # Don't fail on symbolic check alone due to rounding
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps from STEPS ANALYSIS."""
        logs = []
        success = True
        
        try:
            # Step 1: F_x = F1*cos37 - F3 = 20*0.8 - 10 = 16 - 10 = 6
            expected_F_x_step = 6.0
            if not math.isclose(self.F_x, expected_F_x_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step1 F_x = {self.F_x}, should be {expected_F_x_step}")
                success = False
            else:
                logs.append(f"PASS: Step1 F_x = 6N correct")
            
            # Step 2: F_y = F1*sin37 - F2 = 20*0.6 - 6 = 12 - 6 = 6
            expected_F_y_step = 6.0
            if not math.isclose(self.F_y, expected_F_y_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step2 F_y = {self.F_y}, should be {expected_F_y_step}")
                success = False
            else:
                logs.append(f"PASS: Step2 F_y = 6N correct")
            
            # Step 3: F = sqrt(6² + 6²) = 6√2 ≈ 8.485
            expected_F_step = 6 * math.sqrt(2)
            if not math.isclose(self.F_mag, expected_F_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step3 F = {self.F_mag}, should be {expected_F_step:.3f}")
                success = False
            else:
                logs.append(f"PASS: Step3 F = 6√2 ≈ 8.485N correct")
            
            # Step 4: φ = arctan(6/6) = 45°
            expected_phi_step = 45.0
            if not math.isclose(self.phi_deg, expected_phi_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step4 φ = {self.phi_deg}°, should be {expected_phi_step}°")
                success = False
            else:
                logs.append(f"PASS: Step4 φ = 45° correct")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb an input and verify output trends match physical intuition."""
        logs = []
        success = True
        
        try:
            # Store original values
            original_F1 = self.F1
            
            # Test 1: Increase F1 by 5% -> net force magnitude should increase
            # (F1 contributes positively to both components)
            perturbed_F1 = self.F1 * 1.05
            perturbed_F_x = perturbed_F1 * self.cos_theta - self.F3
            perturbed_F_y = perturbed_F1 * self.sin_theta - self.F2
            perturbed_F = math.sqrt(perturbed_F_x**2 + perturbed_F_y**2)
            
            if not perturbed_F > self.F_mag:
                logs.append(f"FAIL: Increasing F1 from {self.F1} to {perturbed_F1} "
                           f"did not increase net force magnitude")
                success = False
            else:
                logs.append(f"PASS: Increasing F1 increases net force (monotonic)")
            
            # Test 2: Decrease F2 by 5% -> net force magnitude should increase
            # (F2 opposes F1's y-component, so reducing opposition increases net)
            original_F2 = self.F2
            perturbed_F2 = self.F2 * 0.95
            perturbed_F_y2 = self.F1 * self.sin_theta - perturbed_F2
            perturbed_F2_mag = math.sqrt(self.F_x**2 + perturbed_F_y2**2)
            
            if not perturbed_F2_mag > self.F_mag:
                logs.append(f"FAIL: Decreasing F2 from {self.F2} to {perturbed_F2} "
                           f"did not increase net force magnitude")
                success = False
            else:
                logs.append(f"PASS: Decreasing F2 increases net force (monotonic)")
            
            # Test 3: Increase F3 by 5% -> net force magnitude should decrease
            # (F3 opposes F1's x-component, so increasing opposition decreases net)
            original_F3 = self.F3
            perturbed_F3 = self.F3 * 1.05
            perturbed_F_x3 = self.F1 * self.cos_theta - perturbed_F3
            perturbed_F3_mag = math.sqrt(perturbed_F_x3**2 + self.F_y**2)
            
            if not perturbed_F3_mag < self.F_mag:
                logs.append(f"FAIL: Increasing F3 from {self.F3} to {perturbed_F3} "
                           f"did not decrease net force magnitude")
                success = False
            else:
                logs.append(f"PASS: Increasing F3 decreases net force (monotonic)")
            
            # Restore original values (conceptually - we don't actually modify solution)
            logs.append("Monotonicity tests completed; original parameters unchanged")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                success, logs = test_method()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: {test_name} FAILED")
                else:
                    all_logs.append(f"Result: {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs