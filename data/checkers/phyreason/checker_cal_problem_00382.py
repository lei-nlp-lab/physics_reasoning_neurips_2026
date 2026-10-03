import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        self.inputs = {}
        self.results = {}
        self.logs = []

        # Input parameters from problem context
        self.inputs['v_p'] = 9.1  # km/s
        self.inputs['v_s'] = 3.7  # km/s
        self.inputs['delta_t'] = 5.4  # s

        # Safely extract results from solution_instance
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract distance 's' - ensure it's a float
        s_val = results_dict.get('s')
        if s_val is None:
            self.results['s'] = 0.0
        elif isinstance(s_val, (int, float)):
            self.results['s'] = float(s_val)
        elif hasattr(s_val, 'evalf'):
            # Handle SymPy objects
            try:
                self.results['s'] = float(s_val.evalf())
            except:
                self.results['s'] = 0.0
        else:
            # Try conversion, but if it fails, use 0.0
            try:
                self.results['s'] = float(s_val)
            except (TypeError, ValueError):
                self.results['s'] = 0.0
        
        # Target answers for verification
        self.target_distance = 34.0  # km
        self.target_observation = "Vibration up and down"

    def log(self, message, level="INFO"):
        """Helper method to log messages."""
        self.logs.append(f"{level}: {message}")

    def test_invariants(self):
        """Verify physical invariants using numeric checks only."""
        try:
            # Numerical check with extracted values
            lhs = self.results['s']
            # Protect against division by zero
            denominator = (1/self.inputs['v_s'] - 1/self.inputs['v_p'])
            if abs(denominator) < 1e-12:
                self.log("Denominator too small in invariant test", "ERROR")
                return False
                
            rhs = self.inputs['delta_t'] / denominator
            
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Distance invariant holds: s={lhs:.2f} km, formula gives {rhs:.2f} km")
                return True
            else:
                self.log(f"Distance invariant violated: s={lhs:.2f} km, formula gives {rhs:.2f} km", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"Invariant test failed: {str(e)}", "ERROR")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Check distance calculation
            denominator = (1/self.inputs['v_s'] - 1/self.inputs['v_p'])
            if abs(denominator) < 1e-12:
                self.log("Denominator too small in trace consistency", "ERROR")
                return False
                
            s_calc = self.inputs['delta_t'] / denominator
            
            if math.isclose(self.results['s'], s_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace consistency: Calculated {s_calc:.1f} km matches result {self.results['s']:.1f} km")
                return True
            else:
                self.log(f"Trace inconsistency: Calculated {s_calc:.1f} km, but got {self.results['s']:.1f} km", "WARNING")
                return False
            
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}", "ERROR")
            return False

    def test_monotonicity(self):
        """Test that perturbing input parameters produces expected monotonic trends."""
        try:
            # Test 1: Increase time difference should increase distance
            denominator = (1/self.inputs['v_s'] - 1/self.inputs['v_p'])
            if abs(denominator) < 1e-12:
                self.log("Denominator too small in monotonicity test", "ERROR")
                return False
                
            original_dt = self.inputs['delta_t']
            s_original = original_dt / denominator
            
            perturbed_dt = original_dt * 1.1  # +10%
            s_perturbed = perturbed_dt / denominator
            
            if s_perturbed > s_original:
                self.log(f"Monotonicity check 1: Increasing Δt from {original_dt} to {perturbed_dt} s "
                        f"increases distance from {s_original:.2f} to {s_perturbed:.2f} km (PASS)")
                monotonicity_1 = True
            else:
                self.log(f"Monotonicity check 1: Distance didn't increase with Δt increase", "WARNING")
                monotonicity_1 = False

            # Test 2: Increase P-wave speed should INCREASE time difference for fixed distance
            # CORRECTED: As v_p increases, 1/v_p decreases, so (1/v_s - 1/v_p) increases
            # Therefore Δt = s * (1/v_s - 1/v_p) should increase for fixed s
            original_vp = self.inputs['v_p']
            perturbed_vp = original_vp * 1.1  # +10%
            
            # For same distance s, compute new time difference
            s_fixed = s_original
            new_denominator = (1/self.inputs['v_s'] - 1/perturbed_vp)
            if abs(new_denominator) < 1e-12:
                self.log("New denominator too small in monotonicity test 2", "WARNING")
                monotonicity_2 = False
            else:
                new_dt = s_fixed * new_denominator
                
                # CORRECTED: new_dt should be GREATER than original_dt when v_p increases
                if new_dt > original_dt:
                    self.log(f"Monotonicity check 2: Increasing v_p increases Δt from {original_dt:.2f} to {new_dt:.2f} s for fixed distance (PASS)")
                    monotonicity_2 = True
                else:
                    self.log(f"Monotonicity check 2: Δt didn't increase with v_p increase", "WARNING")
                    monotonicity_2 = False

            return monotonicity_1 and monotonicity_2
            
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}", "ERROR")
            return False

    def run_all_tests(self):
        """Execute all tests and return success boolean with logs."""
        self.logs = []  # Reset logs
        
        try:
            test_results = []
            
            # Test 1: Physical invariants
            self.log("=" * 50)
            self.log("Test 1: Physical Invariants")
            test1 = self.test_invariants()
            test_results.append(test1)
            
            # Test 2: Trace consistency
            self.log("=" * 50)
            self.log("Test 2: Trace Consistency")
            test2 = self.test_trace_consistency()
            test_results.append(test2)
            
            # Test 3: Monotonicity
            self.log("=" * 50)
            self.log("Test 3: Monotonicity")
            test3 = self.test_monotonicity()
            test_results.append(test3)
            
            # Test 4: Target answer matching (fuzzy numerical match for distance)
            self.log("=" * 50)
            self.log("Test 4: Target Answer Verification")
            distance_match = math.isclose(self.results['s'], self.target_distance, 
                                         rel_tol=1e-2, abs_tol=1e-2)
            if distance_match:
                self.log(f"Distance matches target: {self.results['s']:.1f} km ≈ {self.target_distance} km")
            else:
                self.log(f"Distance mismatch: {self.results['s']:.1f} km ≠ {self.target_distance} km", "WARNING")
            test_results.append(distance_match)
            
            # Note: We don't test observation string since it's not in results
            # The observation will be verified in the [subans] section
            
            # Final assessment - only check the tests that actually run
            success = all(test_results)
            self.log("=" * 50)
            self.log(f"OVERALL RESULT: {'PASS' if success else 'FAIL'}")
            self.log(f"Individual tests: {test_results}")
            
            return success, self.logs
            
        except Exception as e:
            self.log(f"Test suite crashed: {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            return False, self.logs