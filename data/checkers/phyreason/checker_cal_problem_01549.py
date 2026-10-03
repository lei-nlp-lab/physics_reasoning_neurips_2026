import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        try:
            self.solution = solution_instance
            # Extract results - these should always be present
            self.x = solution_instance.results.get('x', 0.0)
            self.E_k = solution_instance.results.get('E_k', 0.0)
            self.a = solution_instance.results.get('a', 0.0)
            
            # Extract constants with reasonable defaults if not found
            # Use the same values as the solution code for consistency
            self.m = solution_instance.results.get('m', 9.11e-31)
            self.e = solution_instance.results.get('e', 1.602e-19)
            self.v0 = solution_instance.results.get('v0', 1.0e6)
            self.E = solution_instance.results.get('E', 1000.0)
            
            # Store original for monotonicity tests
            self.original_inputs = {'m': self.m, 'e': self.e, 'E': self.E, 'v0': self.v0}
            
        except Exception as e:
            self.error_msg = f"Initialization failed: {str(e)}"
            raise

    def test_invariants(self):
        """Check fundamental physics laws from KB."""
        logs = []
        try:
            # Newton's second law: F = m*a, and F = e*E (magnitude, ignoring sign)
            # Note: The force magnitude is e*E, but acceleration a is already magnitude
            force_from_field = self.e * self.E
            force_from_accel = self.m * self.a
            
            # Use relative tolerance for comparison since values can be very large/small
            # Handle division by zero in relative error calculation
            if math.isclose(force_from_field, force_from_accel, rel_tol=1e-2):
                logs.append("✓ Newton's second law: F = eE = m*a holds")
            else:
                # Calculate relative error for debugging, avoiding division by zero
                denom = max(abs(force_from_field), abs(force_from_accel))
                if denom > 0:
                    rel_error = abs(force_from_field - force_from_accel) / denom
                    logs.append(f"✗ Newton's second law mismatch: eE={force_from_field:.3e}, m*a={force_from_accel:.3e}, rel_error={rel_error:.3e}")
                else:
                    logs.append(f"✗ Newton's second law mismatch: eE={force_from_field:.3e}, m*a={force_from_accel:.3e} (denominator zero)")
            
            # Work-energy theorem: net work = change in kinetic energy
            # For maximum distance x, final velocity is 0
            K_initial = 0.5 * self.m * self.v0**2
            K_final = 0.0  # at maximum distance
            W_net = -force_from_field * self.x  # negative work (force opposes motion)
            delta_K = K_final - K_initial
            
            if math.isclose(W_net, delta_K, rel_tol=1e-2):
                logs.append("✓ Work-energy theorem holds for maximum distance")
            else:
                denom = max(abs(W_net), abs(delta_K))
                if denom > 0:
                    rel_error = abs(W_net - delta_K) / denom
                    logs.append(f"✗ Work-energy theorem mismatch: W_net={W_net:.3e}, ΔK={delta_K:.3e}, rel_error={rel_error:.3e}")
                else:
                    logs.append(f"✗ Work-energy theorem mismatch: W_net={W_net:.3e}, ΔK={delta_K:.3e} (denominator zero)")
            
            # Check kinetic energy at half distance using work-energy
            x_half = self.x / 2.0
            W_half = -force_from_field * x_half
            K_half_expected = K_initial + W_half
            if math.isclose(K_half_expected, self.E_k, rel_tol=1e-2):
                logs.append("✓ Work-energy theorem holds for half distance")
            else:
                denom = max(abs(K_half_expected), abs(self.E_k))
                if denom > 0:
                    rel_error = abs(K_half_expected - self.E_k) / denom
                    logs.append(f"✗ Half-distance energy mismatch: expected={K_half_expected:.3e}, got={self.E_k:.3e}, rel_error={rel_error:.3e}")
                else:
                    logs.append(f"✗ Half-distance energy mismatch: expected={K_half_expected:.3e}, got={self.E_k:.3e} (denominator zero)")
                
        except Exception as e:
            logs.append(f"✗ Invariant test failed: {str(e)}")
            logs.append(traceback.format_exc())
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check acceleration formula: a = eE/m
            # Handle division by zero
            if self.m != 0:
                a_expected = self.e * self.E / self.m
                if math.isclose(self.a, a_expected, rel_tol=1e-2):
                    logs.append("✓ Acceleration matches formula a = eE/m")
                else:
                    denom = max(abs(self.a), abs(a_expected))
                    if denom > 0:
                        rel_error = abs(self.a - a_expected) / denom
                        logs.append(f"✗ Acceleration mismatch: expected={a_expected:.3e}, got={self.a:.3e}, rel_error={rel_error:.3e}")
                    else:
                        logs.append(f"✗ Acceleration mismatch: expected={a_expected:.3e}, got={self.a:.3e} (denominator zero)")
            else:
                logs.append("✗ Cannot check acceleration: mass is zero")
            
            # Check maximum distance formula: x = mv₀²/(2eE)
            if self.e != 0 and self.E != 0:
                x_expected = self.m * self.v0**2 / (2 * self.e * self.E)
                if math.isclose(self.x, x_expected, rel_tol=1e-2):
                    logs.append("✓ Maximum distance matches formula x = mv₀²/(2eE)")
                else:
                    denom = max(abs(self.x), abs(x_expected))
                    if denom > 0:
                        rel_error = abs(self.x - x_expected) / denom
                        logs.append(f"✗ Distance mismatch: expected={x_expected:.3e}, got={self.x:.3e}, rel_error={rel_error:.3e}")
                    else:
                        logs.append(f"✗ Distance mismatch: expected={x_expected:.3e}, got={self.x:.3e} (denominator zero)")
            else:
                logs.append("✗ Cannot check distance: e or E is zero")
            
            # Check kinetic energy at half distance: E_k = (1/4)mv₀²
            E_k_expected = 0.25 * self.m * self.v0**2
            if math.isclose(self.E_k, E_k_expected, rel_tol=1e-2):
                logs.append("✓ Kinetic energy at half distance matches formula")
            else:
                denom = max(abs(self.E_k), abs(E_k_expected))
                if denom > 0:
                    rel_error = abs(self.E_k - E_k_expected) / denom
                    logs.append(f"✗ Energy mismatch: expected={E_k_expected:.3e}, got={self.E_k:.3e}, rel_error={rel_error:.3e}")
                else:
                    logs.append(f"✗ Energy mismatch: expected={E_k_expected:.3e}, got={self.E_k:.3e} (denominator zero)")
                
        except Exception as e:
            logs.append(f"✗ Trace consistency test failed: {str(e)}")
            logs.append(traceback.format_exc())
        return logs

    def test_monotonicity(self):
        """Perturb inputs and check trend consistency, then restore."""
        logs = []
        try:
            # Perturb mass by +10%
            m_perturbed = self.m * 1.1
            
            # Recalculate expected results using formulas
            # Handle division by zero
            if m_perturbed != 0:
                a_new = self.e * self.E / m_perturbed
            else:
                a_new = float('inf')
                
            if self.e != 0 and self.E != 0:
                x_new = m_perturbed * self.v0**2 / (2 * self.e * self.E)
            else:
                x_new = float('inf')
                
            E_k_new = 0.25 * m_perturbed * self.v0**2
            
            # Check trends
            # For acceleration: should decrease when mass increases
            if a_new < self.a or math.isclose(a_new, self.a, rel_tol=1e-2):
                logs.append("✓ Acceleration decreases or stays same with increased mass")
            else:
                logs.append(f"✗ Acceleration monotonicity failed: old={self.a:.3e}, new={a_new:.3e}")
            
            # For distance: should increase when mass increases
            if x_new > self.x or math.isclose(x_new, self.x, rel_tol=1e-2):
                logs.append("✓ Maximum distance increases or stays same with increased mass")
            else:
                logs.append(f"✗ Distance monotonicity failed: old={self.x:.3e}, new={x_new:.3e}")
            
            # For energy: should increase when mass increases
            if E_k_new > self.E_k or math.isclose(E_k_new, self.E_k, rel_tol=1e-2):
                logs.append("✓ Kinetic energy at half distance increases or stays same with increased mass")
            else:
                logs.append(f"✗ Energy monotonicity failed: old={self.E_k:.3e}, new={E_k_new:.3e}")
            
            # Restore original values
            self.m = self.original_inputs['m']
            
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed: {str(e)}")
            logs.append(traceback.format_exc())
            # Attempt to restore even if test fails
            self.m = self.original_inputs.get('m', self.m)
        return logs

    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        logs = []
        success = True
        
        # Run tests in order
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                test_logs = test_func()
                if any(log.startswith('✗') for log in test_logs):
                    success = False
                logs.append(f"\n=== {test_name} ===")
                logs.extend(test_logs)
            except Exception as e:
                success = False
                logs.append(f"\n=== {test_name} ===")
                logs.append(f"✗ Test crashed: {str(e)}")
                logs.append(traceback.format_exc())
        
        return success, logs