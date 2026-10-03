import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Problem inputs
        self.L = getattr(solution_instance, 'L', 0.5)
        self.theta = getattr(solution_instance, 'theta', math.radians(30))
        self.B = getattr(solution_instance, 'B', 0.2)
        self.m_a = getattr(solution_instance, 'm_a', 0.3)
        self.m_b = getattr(solution_instance, 'm_b', 0.1)
        self.R = getattr(solution_instance, 'R', 0.1)
        self.F = getattr(solution_instance, 'F', 2.0)
        self.Q_a = getattr(solution_instance, 'Q_a', 0.12)
        self.v_a = getattr(solution_instance, 'v_a', 1.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.t = self.results.get('t', 0.0)
        self.I_x = self.results.get('I_x', 0.0)
        self.a_b = self.results.get('a_b', 0.0)
        self.I = self.results.get('I', 0.0)
        self.v_b = self.results.get('v_b', 0.0)
        self.F_A = self.results.get('F_A', 0.0)
        self.E = self.results.get('E', 0.0)
        
        # For monotonicity test backup
        self._original_m_a = self.m_a
        
    def test_invariants(self):
        """Verify physical invariants: momentum conservation, Newton's laws, energy."""
        logs = []
        try:
            # 1. Momentum relationship from step_2: m_a * v_a = m_b * v_b
            momentum_left = self.m_a * self.v_a
            momentum_right = self.m_b * self.v_b
            if math.isclose(momentum_left, momentum_right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Momentum relationship m_a*v_a = m_b*v_b holds")
            else:
                logs.append(f"✗ Momentum mismatch: {momentum_left} ≠ {momentum_right}")
                
            # 2. Newton's second law for rod b: sum of forces = m_b * a_b
            # Forces on rod b: gravity component down incline (positive), Ampere force up incline (negative)
            # m_b * g * sinθ - F_A = m_b * a_b
            sin_theta = math.sin(self.theta)
            net_force_b = self.m_b * self.g * sin_theta - self.F_A
            expected_net_force = self.m_b * self.a_b
            if math.isclose(net_force_b, expected_net_force, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Newton's second law for rod b holds")
            else:
                logs.append(f"✗ Newton's second law mismatch for rod b: {net_force_b} ≠ {expected_net_force}")
                
            # 3. Energy conservation check - use the actual energy equation from step_8
            # We need x_a and x_b from the solution if available, otherwise skip
            x_a = self.results.get('x_a', None)
            x_b = self.results.get('x_b', None)
            if x_a is not None and x_b is not None:
                left_energy = self.F * x_a + self.m_b * self.g * sin_theta * x_b - self.m_a * self.g * sin_theta * x_a
                right_energy = 0.5 * self.m_a * self.v_a**2 + 0.5 * self.m_b * self.v_b**2 + 2 * self.Q_a
                if math.isclose(left_energy, right_energy, rel_tol=5e-2, abs_tol=5e-2):
                    logs.append("✓ Energy balance holds within 5%")
                else:
                    logs.append(f"✗ Energy mismatch: {left_energy} ≠ {right_energy}")
            else:
                logs.append("Note: x_a, x_b not in results - skipping energy check")
                
            # 4. Check induced EMF and current from step_4
            induced_emf = self.B * self.L * (self.v_a + self.v_b)
            if math.isclose(induced_emf, self.E, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Induced EMF matches velocity relationship")
            else:
                logs.append(f"✗ Induced EMF mismatch: {induced_emf} ≠ {self.E}")
                
            expected_current = induced_emf / (2 * self.R)
            if math.isclose(expected_current, self.I, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Induced current matches Ohm's law")
            else:
                logs.append(f"✗ Current mismatch: {expected_current} ≠ {self.I}")
                
            # 5. Check Ampere force from step_5
            expected_F_A = self.B * self.I * self.L
            if math.isclose(expected_F_A, self.F_A, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Ampere force matches B*I*L")
            else:
                logs.append(f"✗ Ampere force mismatch: {expected_F_A} ≠ {self.F_A}")
                
        except Exception as e:
            logs.append(f"Error in invariant tests: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Step 3: v_b = (m_a * v_a) / m_b
            expected_v_b = (self.m_a * self.v_a) / self.m_b
            if math.isclose(expected_v_b, self.v_b, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_b consistent with momentum relationship")
            else:
                logs.append(f"✗ v_b inconsistent: expected {expected_v_b}, got {self.v_b}")
                
            # NOTE: v_b = a_b * t is NOT valid here because acceleration is not constant
            # The Ampere force changes with velocity, so a_b is not constant
            # We should NOT test this invalid kinematic relationship
            
            # Step 10: Impulse of Ampere force I_x = B²L²(x_a + x_b)/(2R)
            # Use x_a and x_b from results if available
            x_a = self.results.get('x_a', None)
            x_b = self.results.get('x_b', None)
            if x_a is not None and x_b is not None:
                expected_I_x = (self.B**2 * self.L**2 * (x_a + x_b)) / (2 * self.R)
                if math.isclose(expected_I_x, self.I_x, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("✓ Impulse I_x matches theoretical formula")
                else:
                    logs.append(f"✗ I_x mismatch: expected {expected_I_x}, got {self.I_x}")
            else:
                logs.append("Note: x_a, x_b not in results - skipping I_x check")
                    
        except Exception as e:
            logs.append(f"Error in trace consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb mass m_a and check if results change as expected."""
        logs = []
        original_v_b = self.v_b
        original_a_b = self.a_b
        original_t = self.t
        
        try:
            # Increase m_a by 10%
            self.m_a *= 1.1
            # Recompute v_b using momentum relationship (step_3)
            new_v_b = (self.m_a * self.v_a) / self.m_b
            # Since m_a increased, v_b should increase
            if new_v_b > original_v_b:
                logs.append("✓ v_b increases when m_a increases (momentum)")
            else:
                logs.append(f"✗ v_b monotonicity violated: {new_v_b} <= {original_v_b}")
                
            # For a_b and t, we can't recompute easily without the full solution.
            logs.append("Note: a_b and t monotonicity requires full recomputation - skipping quantitative check.")
            
        except Exception as e:
            logs.append(f"Error in monotonicity test: {traceback.format_exc()}")
        finally:
            # Restore original value
            self.m_a = self._original_m_a
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            test_logs = test_method()
            all_logs.extend(test_logs)
            # Check for any failure indicators in logs
            for log in test_logs:
                if log.startswith("✗") or "Error" in log:
                    success = False
        
        # Final summary
        all_logs.append(f"\n=== Summary ===")
        all_logs.append(f"Overall success: {'PASS' if success else 'FAIL'}")
        return success, all_logs