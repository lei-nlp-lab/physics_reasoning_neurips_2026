import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr with fallback to results dictionary.
        """
        self.sol = solution_instance
        
        # Extract results safely
        self.results = solution_instance.results if hasattr(solution_instance, 'results') else {}
        
        # Extract fundamental inputs as NUMERIC values
        # First try to get from results dict, then from instance attributes
        # Use default numeric values if not found
        self.m = float(self.results.get('m', getattr(solution_instance, 'm', 1.0)))
        self.q = float(self.results.get('q', getattr(solution_instance, 'q', 1.0)))
        self.B = float(self.results.get('B', getattr(solution_instance, 'B', 1.0)))
        self.v0 = float(self.results.get('v0', getattr(solution_instance, 'v0', 1.0)))
        self.d = float(self.results.get('d', getattr(solution_instance, 'd', 1.0)))
        
        # Map expected keys with fallbacks - ensure all are float
        self.cosθ = float(self.results.get('costheta', 0.0))
        self.x = float(self.results.get('x', 0.0))
        self.v_y = float(self.results.get('v_y', 0.0))
        self.r = float(self.results.get('r', 0.0))
        self.R_prime = float(self.results.get("R''", self.results.get('R_prime', 0.0)))
        self.v_y2 = float(self.results.get('v_y2', 0.0))
        self.a = float(self.results.get('a', 0.0))
        self.R = float(self.results.get('R', 0.0))
        self.U_0 = float(self.results.get('U_0', 0.0))
        self.U_x = float(self.results.get('U_x', 0.0))
        self.x_0 = float(self.results.get('x_0', 0.0))
        self.L = float(self.results.get('L', 0.0))
        
        # For monotonicity testing
        self._original_m = float(self.m)
        
        # Logging
        self.logs = []
        
    def log(self, msg, success=None):
        """Helper to log messages."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        return success
    
    def fuzzy_eq(self, a, b, msg, rel_tol=1e-2, abs_tol=1e-2):
        """Check if two numbers are approximately equal."""
        try:
            a_float = float(a)
            b_float = float(b)
            if math.isclose(a_float, b_float, rel_tol=rel_tol, abs_tol=abs_tol):
                return self.log(f"{msg}: {a_float} ≈ {b_float}", True)
            else:
                return self.log(f"{msg}: {a_float} ≠ {b_float} (outside tolerance)", False)
        except Exception as e:
            return self.log(f"{msg}: Error in comparison - {str(e)}", False)
    
    def test_invariants(self):
        """Verify physical invariants based on KB theorems."""
        try:
            # 1. Lorentz force as centripetal force for reference ion
            R_expected = self.m * self.v0 / (self.q * self.B)
            success1 = self.fuzzy_eq(self.R, R_expected, 
                                    f"Lorentz centripetal: R = mv0/(qB)")
            
            # 2. Newton's second law for electric field acceleration
            a_expected = self.q * self.U_0 / (self.m * self.d)
            success2 = self.fuzzy_eq(self.a, a_expected,
                                    f"Newton's 2nd: a = qU0/(md)")
            
            # 3. Kinematic equation for vertical motion
            v_y2_expected = 2 * self.a * self.d
            success3 = self.fuzzy_eq(self.v_y2, v_y2_expected,
                                    f"Kinematic: v_y² = 2ad")
            
            # 4. Velocity decomposition consistency
            # For reference ion: v_y should equal v0
            success4 = self.fuzzy_eq(self.v_y, self.v0,
                                    f"Velocity decomposition: v_y = v0 for reference ion")
            
            # 5. Geometric relationship: x_0 = 2R
            x0_expected = 2 * self.R
            success5 = self.fuzzy_eq(self.x_0, x0_expected,
                                    f"Geometry: x0 = 2R")
            
            # 6. Length L = 2d (from step_5)
            L_expected = 2 * self.d
            success6 = self.fuzzy_eq(self.L, L_expected,
                                    f"Analyzer length: L = 2d")
            
            # 7. U_0 formula derivation
            U0_expected = self.m * self.v0**2 / (2 * self.q)
            success7 = self.fuzzy_eq(self.U_0, U0_expected,
                                    f"U0 derivation: U0 = mv0²/(2q)")
            
            return all([success1, success2, success3, success4, success5, success6, success7])
            
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Final target answers from problem statement
            target1 = 2 * self.m * self.v0 / (self.q * self.B)  # x0
            target2 = 2 * self.d  # L
            target3 = self.m * self.v0**2 / (2 * self.q)  # U0
            target4_coeff = self.q * self.B**2 / (8 * self.m)  # Ux coefficient
            
            # Check x0 matches target1
            success1 = self.fuzzy_eq(self.x_0, target1, "x0 matches target answer")
            
            # Check L matches target2
            success2 = self.fuzzy_eq(self.L, target2, "L matches target answer")
            
            # Check U0 matches target3
            success3 = self.fuzzy_eq(self.U_0, target3, "U0 matches target answer")
            
            # Check Ux formula for a sample x value
            # Use the midpoint of the valid x range
            x_min = self.m * self.v0 / (2 * self.q * self.B)
            x_max = 2 * math.sqrt(2) * self.m * self.v0 / (self.q * self.B)
            x_sample = (x_min + x_max) / 2
            
            # Calculate expected Ux from derived formula
            Ux_expected = target4_coeff * x_sample**2
            
            # Recalculate using the step-by-step derivation
            # From step_14: v_y = Bqx/(2m)
            v_y_calc = self.B * self.q * x_sample / (2 * self.m)
            # From step_15-16: v_y² = 2ad = 2*(qUx/(md))*d
            Ux_calc = v_y_calc**2 * self.m / (2 * self.q)
            
            success4 = self.fuzzy_eq(Ux_calc, Ux_expected, 
                                    f"Ux formula consistency for x={x_sample:.3f}")
            
            # Also check that for the reference ion (x = x_0), Ux should equal U_0
            Ux_at_x0 = target4_coeff * self.x_0**2
            success5 = self.fuzzy_eq(Ux_at_x0, self.U_0,
                                    f"Ux(x0) = U0 consistency check")
            
            return all([success1, success2, success3, success4, success5])
            
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and check trends, then restore original value."""
        original_success = True
        try:
            # Store original results that depend on m
            original_R = float(self.R)
            original_x0 = float(self.x_0)
            original_U0 = float(self.U_0)
            
            # Perturb mass by +10%
            perturbed_m = self._original_m * 1.1
            
            # Recalculate expected values with perturbed mass
            R_perturbed = perturbed_m * self.v0 / (self.q * self.B)
            x0_perturbed = 2 * R_perturbed
            U0_perturbed = perturbed_m * self.v0**2 / (2 * self.q)
            
            # Check monotonic trends
            # R and x0 should increase with m (direct proportionality)
            if R_perturbed > original_R:
                self.log(f"R increases with m: {original_R:.3f} → {R_perturbed:.3f}", True)
            else:
                self.log(f"R monotonicity failed: {original_R:.3f} → {R_perturbed:.3f}", False)
                original_success = False
            
            if x0_perturbed > original_x0:
                self.log(f"x0 increases with m: {original_x0:.3f} → {x0_perturbed:.3f}", True)
            else:
                self.log(f"x0 monotonicity failed: {original_x0:.3f} → {x0_perturbed:.3f}", False)
                original_success = False
            
            # U0 should also increase with m
            if U0_perturbed > original_U0:
                self.log(f"U0 increases with m: {original_U0:.3f} → {U0_perturbed:.3f}", True)
            else:
                self.log(f"U0 monotonicity failed: {original_U0:.3f} → {U0_perturbed:.3f}", False)
                original_success = False
            
            # Restore original mass
            self.m = self._original_m
            
            # Also test Ux coefficient dependence on m
            coeff_original = self.q * self.B**2 / (8 * self._original_m)
            coeff_perturbed = self.q * self.B**2 / (8 * perturbed_m)
            
            # Ux coefficient should decrease with increasing m (inverse proportionality)
            if coeff_perturbed < coeff_original:
                self.log(f"Ux coefficient decreases with m: {coeff_original:.3e} → {coeff_perturbed:.3e}", True)
            else:
                self.log(f"Ux coefficient monotonicity failed: {coeff_original:.3e} → {coeff_perturbed:.3e}", False)
                original_success = False
            
            return original_success
            
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}", False)
            # Try to restore original mass even if error occurs
            self.m = self._original_m
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        try:
            self.log("=== PhysicsChecker Start ===")
            
            # Run tests
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            # Summary
            all_success = inv_success and trace_success and mono_success
            self.log(f"\n=== Summary ===")
            self.log(f"Invariants: {'PASS' if inv_success else 'FAIL'}")
            self.log(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
            self.log(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}")
            self.log(f"Overall: {'PASS' if all_success else 'FAIL'}")
            
            return all_success, self.logs
            
        except Exception as e:
            error_msg = f"Critical error in run_all_tests: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs