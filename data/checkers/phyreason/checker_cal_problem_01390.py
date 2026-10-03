import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.m = getattr(solution_instance, 'm', 1.0)  # Given: 1 kg
        self.v0 = getattr(solution_instance, 'v0', 3.0)  # Given: 3 m/s horizontal
        self.x = getattr(solution_instance, 'x', 1.2)  # Given: 1.2 m horizontal distance
        
        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.v_y = getattr(solution_instance, 'v_y', results.get('v_y', 0.0))
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        self.theta = getattr(solution_instance, 'theta', results.get('theta', 0.0))
        self.P_G = getattr(solution_instance, 'P_G', results.get('P_G', 0.0))
        
        # Extract g from solution instance or use default
        self.g = results.get('g', getattr(solution_instance, 'g', 10.0))
        
        # For monotonicity test
        self._original_v0 = self.v0
        
    def test_invariants(self):
        logs = []
        
        # 1. Horizontal motion check: x = v0 * t
        try:
            expected_x = self.v0 * self.t
            if not math.isclose(self.x, expected_x, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Horizontal motion invariant violated: x={self.x}, v0*t={expected_x}")
            else:
                logs.append("✓ Horizontal motion: x = v0*t holds")
        except Exception as e:
            logs.append(f"Horizontal motion check failed: {e}")
        
        # 2. Free-fall vertical motion: v_y = g * t
        try:
            expected_v_y = self.g * self.t
            if not math.isclose(self.v_y, expected_v_y, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Vertical motion invariant violated: v_y={self.v_y}, g*t={expected_v_y}")
            else:
                logs.append("✓ Vertical motion: v_y = g*t holds")
        except Exception as e:
            logs.append(f"Vertical motion check failed: {e}")
        
        # 3. Energy conservation: 1/2*m*v^2 = 1/2*m*v0^2 + m*g*h
        try:
            # Calculate height from free-fall: h = 1/2*g*t^2
            h = 0.5 * self.g * self.t**2
            initial_energy = 0.5 * self.m * self.v0**2 + self.m * self.g * h
            final_energy = 0.5 * self.m * self.v**2
            if not math.isclose(initial_energy, final_energy, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Energy conservation violated: initial={initial_energy}, final={final_energy}")
            else:
                logs.append("✓ Energy conservation holds")
        except Exception as e:
            logs.append(f"Energy conservation check failed: {e}")
        
        # 4. Pythagorean check: v^2 = v0^2 + v_y^2
        try:
            expected_v_sq = self.v0**2 + self.v_y**2
            if not math.isclose(self.v**2, expected_v_sq, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Pythagorean violation: v²={self.v**2}, v0²+v_y²={expected_v_sq}")
            else:
                logs.append("✓ Pythagorean relation holds")
        except Exception as e:
            logs.append(f"Pythagorean check failed: {e}")
        
        # 5. Angle check: tan(θ) = v_y / v0
        try:
            if abs(self.v0) < 1e-10:  # Avoid division by zero
                logs.append("Angle check skipped: v0 is zero")
            else:
                expected_theta = math.degrees(math.atan(self.v_y / self.v0))
                if not math.isclose(self.theta, expected_theta, rel_tol=0.02, abs_tol=1.0):
                    logs.append(f"Angle mismatch: θ={self.theta}°, expected={expected_theta}°")
                else:
                    logs.append("✓ Angle relation tan(θ)=v_y/v0 holds")
        except Exception as e:
            logs.append(f"Angle check failed: {e}")
        
        # 6. Power check: P_G = m*g*v_y
        try:
            expected_P_G = self.m * self.g * self.v_y
            if not math.isclose(self.P_G, expected_P_G, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Power mismatch: P_G={self.P_G}, m*g*v_y={expected_P_G}")
            else:
                logs.append("✓ Instantaneous power formula holds")
        except Exception as e:
            logs.append(f"Power check failed: {e}")
        
        return len(logs) == 0 or all('✓' in log for log in logs), logs
    
    def test_trace_consistency(self):
        logs = []
        
        # Check step-by-step consistency from STEPS ANALYSIS
        # Step 1: t = x / v0
        try:
            if abs(self.v0) < 1e-10:  # Avoid division by zero
                logs.append("Step 1 check skipped: v0 is zero")
            else:
                expected_t = self.x / self.v0
                if not math.isclose(self.t, expected_t, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Step 1 inconsistency: t={self.t}, x/v0={expected_t}")
                else:
                    logs.append("✓ Step 1: t = x/v0 consistent")
        except Exception as e:
            logs.append(f"Step 1 check failed: {e}")
        
        # Step 2: v_y = g*t
        try:
            expected_v_y = self.g * self.t
            if not math.isclose(self.v_y, expected_v_y, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Step 2 inconsistency: v_y={self.v_y}, g*t={expected_v_y}")
            else:
                logs.append("✓ Step 2: v_y = g*t consistent")
        except Exception as e:
            logs.append(f"Step 2 check failed: {e}")
        
        # Step 3: v = sqrt(v0^2 + v_y^2)
        try:
            expected_v = math.sqrt(self.v0**2 + self.v_y**2)
            if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Step 3 inconsistency: v={self.v}, sqrt(v0²+v_y²)={expected_v}")
            else:
                logs.append("✓ Step 3: v = sqrt(v0²+v_y²) consistent")
        except Exception as e:
            logs.append(f"Step 3 check failed: {e}")
        
        # Step 4: θ = arctan(v_y/v0)
        try:
            if abs(self.v0) < 1e-10:  # Avoid division by zero
                logs.append("Step 4 check skipped: v0 is zero")
            else:
                expected_theta = math.degrees(math.atan(self.v_y / self.v0))
                if not math.isclose(self.theta, expected_theta, rel_tol=0.02, abs_tol=1.0):
                    logs.append(f"Step 4 inconsistency: θ={self.theta}°, arctan(v_y/v0)={expected_theta}°")
                else:
                    logs.append("✓ Step 4: θ = arctan(v_y/v0) consistent")
        except Exception as e:
            logs.append(f"Step 4 check failed: {e}")
        
        # Step 5: P_G = m*g*v_y
        try:
            expected_P_G = self.m * self.g * self.v_y
            if not math.isclose(self.P_G, expected_P_G, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Step 5 inconsistency: P_G={self.P_G}, m*g*v_y={expected_P_G}")
            else:
                logs.append("✓ Step 5: P_G = m*g*v_y consistent")
        except Exception as e:
            logs.append(f"Step 5 check failed: {e}")
        
        return len(logs) == 0 or all('✓' in log for log in logs), logs
    
    def test_monotonicity(self):
        logs = []
        
        try:
            # Perturb initial velocity by ±10%
            perturbations = [0.9, 1.1]
            
            for factor in perturbations:
                perturbed_v0 = self.v0 * factor
                
                # Recalculate using physics formulas with the SAME g as used in original calculation
                t_perturbed = self.x / perturbed_v0
                v_y_perturbed = self.g * t_perturbed
                v_perturbed = math.sqrt(perturbed_v0**2 + v_y_perturbed**2)
                
                # Check monotonic trends - CORRECTED: Only check t and v_y which have clear monotonic relationships
                # t is inversely proportional to v0
                if factor > 1:  # Increased v0
                    if not (t_perturbed < self.t or math.isclose(t_perturbed, self.t, rel_tol=0.01)):
                        logs.append(f"Monotonicity violation: t should decrease when v0 increases")
                    if not (v_y_perturbed < self.v_y or math.isclose(v_y_perturbed, self.v_y, rel_tol=0.01)):
                        logs.append(f"Monotonicity violation: v_y should decrease when v0 increases")
                    
                else:  # Decreased v0
                    if not (t_perturbed > self.t or math.isclose(t_perturbed, self.t, rel_tol=0.01)):
                        logs.append(f"Monotonicity violation: t should increase when v0 decreases")
                    if not (v_y_perturbed > self.v_y or math.isclose(v_y_perturbed, self.v_y, rel_tol=0.01)):
                        logs.append(f"Monotonicity violation: v_y should increase when v0 decreases")
            
            if not logs:
                logs.append("✓ Monotonicity test passed: t and v_y behave consistently with perturbations")
            else:
                # Add note about v not having guaranteed monotonicity
                logs.append("Note: Resultant speed v does not have guaranteed monotonicity with v0 in projectile motion")
            
        except Exception as e:
            logs.append(f"Monotonicity test failed: {e}")
        
        # For this specific problem, we accept the test if only v monotonicity fails
        # (since that's expected physics behavior)
        filtered_logs = [log for log in logs if "v should" not in log]
        success = len(filtered_logs) == 0 or all('✓' in log for log in filtered_logs)
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not test_success:
                    success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.append(f"Test crashed: {e}")
                all_logs.append(traceback.format_exc())
                success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        if success:
            all_logs.append("✓ All physics checks passed within tolerance")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return success, all_logs