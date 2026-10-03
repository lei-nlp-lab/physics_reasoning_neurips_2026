import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        Uses getattr with fallbacks to avoid AttributeError in sandbox.
        """
        # Constants from problem context
        self.m = 1500.0  # kg
        self.g = 10.0    # m/s²
        self.mu = math.sqrt(3) / 3  # friction coefficient
        self.a_truck = 8.0  # m/s² (truck braking acceleration)
        
        # Extract results safely - use getattr with nested dict fallback
        self.results = {}
        
        # Map expected keys
        keys_needed = ['F_N', 'a_1', 'Deltax', 'a_2', 'x_2', 'x_1']
        for key in keys_needed:
            # Try direct attribute
            val = getattr(solution_instance, key, None)
            if val is None and hasattr(solution_instance, 'results'):
                val = solution_instance.results.get(key, None)
            if val is not None:
                if isinstance(val, (int, float)):
                    self.results[key] = float(val)
                elif isinstance(val, str):
                    try:
                        if 'sqrt' in val:
                            val = val.replace('sqrt', 'math.sqrt')
                            self.results[key] = eval(val)
                        else:
                            self.results[key] = float(val)
                    except:
                        self.results[key] = 0.0
                else:
                    # Try to convert SymPy numbers
                    try:
                        self.results[key] = float(val)
                    except:
                        self.results[key] = 0.0
            else:
                self.results[key] = 0.0
        
        # Store for convenience
        self.F_N = self.results.get('F_N', 0.0)
        self.a_1 = self.results.get('a_1', 0.0)
        self.a_2 = self.results.get('a_2', self.a_truck)
        self.x_1 = self.results.get('x_1', 0.0)
        self.x_2 = self.results.get('x_2', 0.0)
        self.Deltax = self.results.get('Deltax', 0.0)

    def test_invariants(self):
        """Verify fundamental physics invariants."""
        logs = []
        
        # 1. Check equilibrium condition for F_N (Step 1)
        # F_N * cos30° = 0.5 * m * g
        cos30 = math.cos(math.radians(30))  # √3/2 ≈ 0.8660254
        expected_F_N = (0.5 * self.m * self.g) / cos30
        
        if self.F_N:
            if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_N equilibrium check. Expected ~{expected_F_N:.2f}, got {self.F_N:.2f}")
            else:
                logs.append(f"PASS: F_N equilibrium check ({self.F_N:.2f} ≈ {expected_F_N:.2f})")
        
        # 2. Check friction force relation for a_1 (Step 4)
        # Correct formula: a_1 = 2 * μ * F_N / m (magnitude) because two contact points
        if self.a_1:
            expected_a1 = 2 * self.mu * self.F_N / self.m
            # Compare absolute values because a_1 might be negative
            if not math.isclose(abs(self.a_1), expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: a_1 friction check. Got |{self.a_1:.2f}|, expected 2*μ*F_N/m={expected_a1:.2f}")
            else:
                logs.append(f"PASS: a_1 friction check (|{self.a_1:.2f}| ≈ {expected_a1:.2f})")
        
        # 3. Check truck acceleration matches given value
        if self.a_2 and not math.isclose(self.a_2, self.a_truck, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Truck acceleration mismatch. Expected {self.a_truck}, got {self.a_2}")
        elif self.a_2:
            logs.append(f"PASS: Truck acceleration matches ({self.a_2})")
        
        # 4. Check kinematics consistency between x_1, x_2, a_1, a_2 (Steps 6-8)
        # x_1 = v0²/(2|a_1|), x_2 = v0²/(2a_2) => x_1/x_2 = a_2/|a_1|
        if all(v is not None for v in [self.x_1, self.x_2, self.a_1, self.a_2]):
            ratio_left = self.x_1 / self.x_2 if self.x_2 != 0 else float('inf')
            ratio_right = self.a_2 / abs(self.a_1) if self.a_1 != 0 else float('inf')
            
            if not math.isclose(ratio_left, ratio_right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinematics ratio mismatch. x1/x2={ratio_left:.2f}, a2/|a1|={ratio_right:.2f}")
            else:
                logs.append(f"PASS: Kinematics ratio consistent (x1/x2 = a2/|a1| = {ratio_left:.2f})")
        
        # 5. Check Δx = x_1 - x_2 (Step 8)
        if all(v is not None for v in [self.Deltax, self.x_1, self.x_2]):
            if not math.isclose(self.Deltax, self.x_1 - self.x_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δx = x1 - x2 violated. {self.Deltax} ≠ {self.x_1 - self.x_2}")
            else:
                logs.append(f"PASS: Δx = x1 - x2 satisfied ({self.Deltax} = {self.x_1} - {self.x_2})")
        
        # 6. Energy/work theorem: Friction work = kinetic energy loss
        # Correct: Work done by friction on upper pipe = - (2*μ*F_N) * x_1 (negative because force opposes displacement)
        # Change in kinetic energy = 0 - 0.5*m*v0² = -0.5*m*v0²
        # So |work| = 0.5*m*v0²
        # Also v0² = 2*|a_1|*x_1
        if all(v is not None for v in [self.a_1, self.x_1, self.F_N]):
            v0_sq_from_kinematics = 2 * abs(self.a_1) * self.x_1
            work_friction_magnitude = 2 * self.mu * self.F_N * self.x_1
            ke_loss = 0.5 * self.m * v0_sq_from_kinematics
            
            if not math.isclose(work_friction_magnitude, ke_loss, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy balance. Friction work magnitude={work_friction_magnitude:.2f}, ΔKE magnitude={ke_loss:.2f}")
            else:
                logs.append(f"PASS: Energy conservation check")
        
        return logs

    def test_trace_consistency(self):
        """Check intermediate results are consistent with final answer."""
        logs = []
        
        # Step 1: F_N from equilibrium
        if self.F_N:
            cos30 = math.cos(math.radians(30))
            expected_F_N = (0.5 * self.m * self.g) / cos30
            if math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("TRACE PASS: Step 1 (F_N equilibrium) consistent")
            else:
                logs.append(f"TRACE WARN: Step 1 F_N deviates: {self.F_N:.2f} vs {expected_F_N:.2f}")
        
        # Step 4: a_1 from friction
        if self.a_1 and self.F_N:
            expected_a1 = 2 * self.mu * self.F_N / self.m
            if math.isclose(abs(self.a_1), expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("TRACE PASS: Step 4 (a_1 from friction) consistent")
            else:
                logs.append(f"TRACE WARN: Step 4 a_1 deviates: |{self.a_1:.2f}| vs {expected_a1:.2f}")
        
        # Steps 6-8: Kinematics chain
        if all(v is not None for v in [self.a_1, self.a_2, self.x_1, self.x_2, self.Deltax]):
            # Derived from same v0
            v0_sq_1 = 2 * abs(self.a_1) * self.x_1
            v0_sq_2 = 2 * self.a_2 * self.x_2
            
            if math.isclose(v0_sq_1, v0_sq_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("TRACE PASS: Consistent v0 from both motions")
            else:
                logs.append(f"TRACE FAIL: Inconsistent v0: {v0_sq_1:.2f} vs {v0_sq_2:.2f}")
            
            # Δx relationship
            if math.isclose(self.Deltax, self.x_1 - self.x_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("TRACE PASS: Δx correctly computed from x1 - x2")
            else:
                logs.append("TRACE FAIL: Δx doesn't match x1 - x2")
        
        # Final answer check (target values)
        expected_F_N_val = 5000 * math.sqrt(3)  # ≈ 8660.254
        expected_Deltax = 1.8
        
        if self.F_N and math.isclose(self.F_N, expected_F_N_val, rel_tol=1e-2, abs_tol=1e-2):
            logs.append("TARGET PASS: F_N = 5000√3 N verified")
        elif self.F_N:
            logs.append(f"TARGET FAIL: F_N {self.F_N:.2f} ≠ 5000√3 ≈ {expected_F_N_val:.2f}")
        
        if self.Deltax and math.isclose(self.Deltax, expected_Deltax, rel_tol=1e-2, abs_tol=1e-2):
            logs.append("TARGET PASS: Δx = 1.8 m verified")
        elif self.Deltax:
            logs.append(f"TARGET FAIL: Δx {self.Deltax:.2f} ≠ 1.8")
        
        return logs

    def test_monotonicity(self):
        """
        Perturb mass m and check expected trend.
        Important: Restore original m after test.
        """
        logs = []
        
        if not self.F_N:
            logs.append("MONOTONICITY SKIP: F_N not available")
            return logs
        
        original_m = self.m
        perturbation = 0.01  # 1% increase
        
        try:
            # Increase mass
            self.m = original_m * (1 + perturbation)
            cos30 = math.cos(math.radians(30))
            new_F_N = (0.5 * self.m * self.g) / cos30
            
            # Check proportional increase
            expected_ratio = (1 + perturbation)
            actual_ratio = new_F_N / self.F_N if self.F_N != 0 else 0
            
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"MONOTONICITY PASS: F_N scales linearly with m (ratio: {actual_ratio:.4f})")
            else:
                logs.append(f"MONOTONICITY FAIL: F_N scaling violated. Expected ratio {expected_ratio:.4f}, got {actual_ratio:.4f}")
            
        except Exception as e:
            logs.append(f"MONOTONICITY ERROR: {str(e)}")
        finally:
            # Restore original mass
            self.m = original_m
        
        return logs

    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        Success is defined as no critical failures (FAIL or TARGET FAIL).
        """
        all_logs = []
        
        try:
            all_logs.append("=== PhysicsChecker Report ===")
            
            # Run tests
            inv_logs = self.test_invariants()
            trace_logs = self.test_trace_consistency()
            mono_logs = self.test_monotonicity()
            
            all_logs.extend(inv_logs)
            all_logs.extend(trace_logs)
            all_logs.extend(mono_logs)
            
            # Determine success
            fail_keywords = ['FAIL', 'ERROR']
            success = True
            for log in all_logs:
                if any(keyword in log for keyword in fail_keywords):
                    success = False
                    break
            
            return success, all_logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            return False, [error_msg]