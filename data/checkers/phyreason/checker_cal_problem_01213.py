import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        # Extract inputs from problem statement (with safe defaults)
        self.v0 = getattr(solution_instance, 'v0', 0.0)          # initial velocity before kick
        self.v1 = getattr(solution_instance, 'v1', 30.0)         # velocity after kick
        self.contact_foot = getattr(solution_instance, 'contact_foot', 0.15)  # contact time with foot
        self.distance = getattr(solution_instance, 'distance', 11.0)  # distance traveled in air
        self.contact_goalkeeper = getattr(solution_instance, 'contact_goalkeeper', 0.2)  # contact time with goalkeeper
        self.v2 = getattr(solution_instance, 'v2', -5.0)         # velocity after block (negative for opposite direction)
        
        # Extract results from solution_instance.results (with safe defaults)
        results = getattr(solution_instance, 'results', {})
        self.a = results.get('a', 0.0)                           # acceleration during foot contact
        self.t2 = results.get('t_2', 0.0)                        # time in air
        self.a2 = results.get('a_2', 0.0)                        # acceleration during goalkeeper block
        
        # Store original values for restoration after perturbation tests
        self._original_inputs = {
            'v1': self.v1,
            'distance': self.distance,
            'contact_foot': self.contact_foot
        }

    def test_invariants(self):
        """Verify fundamental physics laws using fuzzy matching."""
        logs = []
        try:
            # Step 1: Check acceleration definition during foot contact
            a_calc = (self.v1 - self.v0) / self.contact_foot
            if math.isclose(self.a, a_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Step 1: a = (v1 - v0)/t1 matches definition of average acceleration")
            else:
                logs.append(f"✗ Step 1: Expected a ≈ {a_calc:.1f}, got {self.a:.1f}")
            
            # Step 2: Check uniform motion equation for time in air
            t2_calc = self.distance / self.v1
            if math.isclose(self.t2, t2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Step 2: t2 = distance/v1 matches uniform motion equation")
            else:
                logs.append(f"✗ Step 2: Expected t2 ≈ {t2_calc:.3f}, got {self.t2:.3f}")
            
            # Step 3: Check acceleration definition during goalkeeper block
            a2_calc = (self.v2 - self.v1) / self.contact_goalkeeper
            if math.isclose(self.a2, a2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Step 3: a2 = (v2 - v1)/t3 matches definition of average acceleration")
            else:
                logs.append(f"✗ Step 3: Expected a2 ≈ {a2_calc:.1f}, got {self.a2:.1f}")
            
            # Check direction - based on sign of a2
            if self.a2 < 0:
                logs.append("✓ Negative a2 indicates direction opposite to initial velocity")
            else:
                logs.append("⚠ a2 is not negative, may not indicate opposite direction")
                
        except ZeroDivisionError:
            logs.append("✗ Division by zero encountered in physics calculations")
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
        
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that all calculated values are within expected ranges
            if math.isclose(self.a, 200.0, rel_tol=1e-2, abs_tol=2.0):
                logs.append("✓ Step 1 acceleration (200 m/s²) consistent with target")
            else:
                logs.append(f"✗ Step 1 acceleration {self.a:.1f} m/s² differs from 200 m/s²")
            
            if math.isclose(self.t2, 0.37, rel_tol=1e-2, abs_tol=0.01):
                logs.append("✓ Step 2 time (0.37 s) consistent with target")
            else:
                logs.append(f"✗ Step 2 time {self.t2:.3f} s differs from 0.37 s")
            
            if math.isclose(self.a2, -175.0, rel_tol=1e-2, abs_tol=2.0):
                logs.append("✓ Step 3 acceleration (-175 m/s²) consistent with target")
            elif math.isclose(abs(self.a2), 175.0, rel_tol=1e-2, abs_tol=2.0):
                logs.append("✓ Step 3 acceleration magnitude (175 m/s²) consistent with target")
            else:
                logs.append(f"✗ Step 3 acceleration {self.a2:.1f} m/s² differs from -175 m/s²")
            
            # Verify the sign of a2 matches physical expectation
            if self.a2 < 0 and self.v2 < 0:
                logs.append("✓ Negative a2 sign consistent with rebound direction")
            elif self.a2 > 0 and self.v2 > 0:
                logs.append("⚠ Positive a2 sign may indicate incorrect direction assumption")
            else:
                logs.append("✗ a2 sign inconsistent with velocity change direction")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
        
        return logs

    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        logs = []
        original_v1 = self.v1
        original_distance = self.distance
        original_contact_foot = self.contact_foot
        
        try:
            # Test 1: Increase initial velocity by 10%
            self.v1 *= 1.1
            a_new = (self.v1 - self.v0) / original_contact_foot
            t2_new = original_distance / self.v1
            a2_new = (self.v2 - self.v1) / self.contact_goalkeeper
            
            # Restore original
            self.v1 = original_v1
            
            # Check trends
            if a_new > self.a:
                logs.append("✓ Increasing v1 increases foot acceleration (monotonic trend)")
            else:
                logs.append("✗ Increasing v1 should increase foot acceleration")
            
            if t2_new < self.t2:
                logs.append("✓ Increasing v1 decreases travel time (inverse relationship)")
            else:
                logs.append("✗ Increasing v1 should decrease travel time")
            
            if abs(a2_new) > abs(self.a2):
                logs.append("✓ Increasing v1 increases goalkeeper acceleration magnitude")
            else:
                logs.append("✗ Increasing v1 should increase goalkeeper acceleration magnitude")
            
            # Test 2: Increase contact time with foot by 10%
            self.contact_foot *= 1.1
            a_new2 = (self.v1 - self.v0) / self.contact_foot
            
            # Restore
            self.contact_foot = original_contact_foot
            
            if a_new2 < self.a:
                logs.append("✓ Increasing contact time decreases acceleration (inverse relationship)")
            else:
                logs.append("✗ Increasing contact time should decrease acceleration")
                
        except ZeroDivisionError:
            logs.append("✗ Division by zero in monotonicity test")
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            # Ensure restoration even on error
            self.v1 = original_v1
            self.contact_foot = original_contact_foot
        
        return logs

    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        logs = []
        all_passed = True
        
        # Run each test
        for test_name, test_func in [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Trends", self.test_monotonicity)
        ]:
            logs.append(f"\n=== Running {test_name} ===")
            test_logs = test_func()
            
            # Check for errors or failures
            has_errors = any(log.startswith('✗') or 'Error' in log for log in test_logs)
            if has_errors:
                all_passed = False
            
            logs.extend(test_logs)
        
        # Summary
        logs.append(f"\n=== Summary ===")
        if all_passed:
            logs.append("✓ All physics checks passed within tolerance")
        else:
            logs.append("✗ Some physics checks failed")
        
        return all_passed, logs