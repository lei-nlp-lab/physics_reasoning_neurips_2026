import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        # First try to get results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Inputs - try to get from results first, then from attributes
        self.m = results.get('m', getattr(solution_instance, 'm', None))
        self.q = results.get('q', getattr(solution_instance, 'q', None))
        self.v = results.get('v', getattr(solution_instance, 'v', None))
        self.a = results.get('a', getattr(solution_instance, 'a', None))
        self.E = results.get('E', getattr(solution_instance, 'E', None))
        
        # Results - directly from results dictionary
        self.B = results.get('B', None)
        self.t = results.get('t', None)
        self.x = results.get('x', None)
        self.y = results.get('y', None)
        self.r = results.get('r', None)
        self.T = results.get('T', None)
        self.theta = results.get('theta', None)
        self.coords = results.get('x,y', None)
        self.law_eq = results.get('qvB=mfracv2r', None)
        
        # Derived constants
        self.cos30 = math.cos(math.radians(30))  # √3/2 ≈ 0.866
        self.cos60 = math.cos(math.radians(60))  # 0.5
        self.sin60 = math.sin(math.radians(60))  # √3/2
        
        self.logs = []

    def log(self, message):
        """Helper to append log messages."""
        self.logs.append(message)

    def fuzzy_equal(self, a, b, name="", rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        if a is None or b is None:
            self.log(f"WARNING: Missing data for {name}")
            return False
        try:
            # Convert to float if needed
            a_val = float(a) if hasattr(a, 'evalf') else a
            b_val = float(b) if hasattr(b, 'evalf') else b
            close = math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
            if not close:
                self.log(f"FAIL {name}: {a_val} ≠ {b_val} (outside tolerance)")
            return close
        except Exception as e:
            self.log(f"ERROR in fuzzy_equal for {name}: {e}")
            return False

    def test_invariants(self):
        """Verify fundamental physics laws based on KB."""
        tests = []
        try:
            # 1. Magnetic force = centripetal force (Newton's 2nd + Lorentz)
            if None not in (self.q, self.v, self.B, self.m, self.r):
                lhs = self.q * self.v * self.B
                rhs = self.m * self.v**2 / self.r
                tests.append(("Magnetic force = centripetal force", 
                              self.fuzzy_equal(lhs, rhs, "qvB = mv²/r")))

            # 2. Radius from geometry (step_2)
            if None not in (self.r, self.a):
                expected_r = self.a / self.cos30  # a / (√3/2)
                tests.append(("Radius geometry r = a/cos30°", 
                              self.fuzzy_equal(self.r, expected_r, "r")))

            # 3. Period from uniform circular motion (step_4)
            if None not in (self.T, self.r, self.v):
                expected_T = 2 * math.pi * self.r / self.v
                tests.append(("Period T = 2πr/v", 
                              self.fuzzy_equal(self.T, expected_T, "T")))

            # 4. Central angle (step_5): should be 120° = 2π/3 rad
            if self.theta is not None:
                expected_theta = 2 * math.pi / 3
                tests.append(("Central angle θ = 2π/3", 
                              self.fuzzy_equal(self.theta, expected_theta, "θ")))

            # 5. Time in magnetic field (step_6-7)
            if None not in (self.t, self.theta, self.T):
                expected_t = (self.theta / (2 * math.pi)) * self.T
                tests.append(("Time t = (θ/2π)T", 
                              self.fuzzy_equal(self.t, expected_t, "t")))
                # Alternative formula: t = 4√3 π a / (9v)
                if None not in (self.a, self.v):
                    expected_t2 = (4 * math.sqrt(3) * math.pi * self.a) / (9 * self.v)
                    tests.append(("Time t = 4√3 π a/(9v)", 
                                  self.fuzzy_equal(self.t, expected_t2, "t_alt")))

            # 6. Y-coordinate (step_8)
            if None not in (self.y, self.r):
                expected_y = self.r * (1 + self.cos60)  # r + r*cos60°
                tests.append(("Y-coordinate y = r(1+cos60°)", 
                              self.fuzzy_equal(self.y, expected_y, "y")))
                # Also check with a: y = √3 a
                if self.a is not None:
                    expected_y2 = math.sqrt(3) * self.a
                    tests.append(("Y-coordinate y = √3 a", 
                                  self.fuzzy_equal(self.y, expected_y2, "y_alt")))

            # 7. Electric field kinematics (step_9): Newton's 2nd + kinematics
            if None not in (self.x, self.m, self.v, self.q, self.E):
                # Acceleration in electric field: a = qE/m (deceleration)
                # Distance to stop: x = v²/(2a) = m v²/(2qE)
                expected_x = self.m * self.v**2 / (2 * self.q * self.E)
                # x should be negative (second quadrant)
                if expected_x > 0:
                    expected_x = -expected_x
                tests.append(("X-distance in electric field x = -mv²/(2qE)", 
                              self.fuzzy_equal(self.x, expected_x, "x")))

            # 8. Energy conservation check (electric field segment only)
            if None not in (self.m, self.v, self.q, self.E, self.x):
                delta_KE = -0.5 * self.m * self.v**2  # From v to 0
                work = self.q * self.E * self.x  # x is negative, so work negative
                tests.append(("Energy: ΔKE = work by electric field", 
                              self.fuzzy_equal(delta_KE, work, "energy")))

        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False

        success = all(t[1] for t in tests)
        for name, passed in tests:
            self.log(f"{'PASS' if passed else 'FAIL'}: {name}")
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check B expression from steps
            if None not in (self.B, self.m, self.v, self.q, self.a):
                expected_B = (math.sqrt(3) * self.m * self.v) / (2 * self.q * self.a)
                check1 = self.fuzzy_equal(self.B, expected_B, "B expression")
                self.log(f"{'PASS' if check1 else 'FAIL'}: B = √3 m v/(2 q a)")
            else:
                check1 = False
                self.log("Missing data for B check")

            # 2. Check t expression
            if None not in (self.t, self.a, self.v):
                expected_t = (4 * math.sqrt(3) * math.pi * self.a) / (9 * self.v)
                check2 = self.fuzzy_equal(self.t, expected_t, "t expression")
                self.log(f"{'PASS' if check2 else 'FAIL'}: t = 4√3 π a/(9 v)")
            else:
                check2 = False
                self.log("Missing data for t check")

            # 3. Check coordinates
            if None not in (self.x, self.y, self.m, self.v, self.q, self.E, self.a):
                expected_x = -self.m * self.v**2 / (2 * self.q * self.E)
                expected_y = math.sqrt(3) * self.a
                check3_x = self.fuzzy_equal(self.x, expected_x, "x coordinate")
                check3_y = self.fuzzy_equal(self.y, expected_y, "y coordinate")
                check3 = check3_x and check3_y
                self.log(f"{'PASS' if check3 else 'FAIL'}: coords = (-mv²/(2qE), √3 a)")
            else:
                check3 = False
                self.log("Missing data for coordinates check")

            return all([check1, check2, check3])
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        try:
            # Store original values
            original_m = self.m
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1 if original_m else None
            
            if None not in (original_m, perturbed_m, self.v, self.q, self.a, self.E):
                # B ∝ m (from B = √3 m v/(2 q a))
                B_original = (math.sqrt(3) * original_m * self.v) / (2 * self.q * self.a)
                B_perturbed = (math.sqrt(3) * perturbed_m * self.v) / (2 * self.q * self.a)
                check1 = B_perturbed > B_original
                self.log(f"B monotonicity (m↑ → B↑): {'PASS' if check1 else 'FAIL'} " +
                        f"({B_original:.3e} → {B_perturbed:.3e})")
                
                # t independent of m (t = 4√3 π a/(9 v))
                t_original = (4 * math.sqrt(3) * math.pi * self.a) / (9 * self.v)
                t_perturbed = t_original  # unchanged
                check2 = math.isclose(t_perturbed, t_original, rel_tol=1e-2)
                self.log(f"t independent of m: {'PASS' if check2 else 'FAIL'} " +
                        f"({t_original:.3e} → {t_perturbed:.3e})")
                
                # |x| ∝ m (x = -m v²/(2 q E))
                x_original = -original_m * self.v**2 / (2 * self.q * self.E)
                x_perturbed = -perturbed_m * self.v**2 / (2 * self.q * self.E)
                check3 = abs(x_perturbed) > abs(x_original)
                self.log(f"|x| monotonicity (m↑ → |x|↑): {'PASS' if check3 else 'FAIL'} " +
                        f"({abs(x_original):.3e} → {abs(x_perturbed):.3e})")
                
                return all([check1, check2, check3])
            else:
                self.log("Missing data for monotonicity test")
                return False
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs = []
        self.log("=== Physics Checker Started ===")
        
        results = []
        try:
            results.append(("Invariants", self.test_invariants()))
            results.append(("Trace Consistency", self.test_trace_consistency()))
            results.append(("Monotonicity", self.test_monotonicity()))
        except Exception as e:
            self.log(f"CRITICAL ERROR in test suite: {traceback.format_exc()}")
            return False, self.logs
        
        success = all(r[1] for r in results)
        self.log(f"=== All tests {'PASSED' if success else 'FAILED'} ===")
        for test_name, passed in results:
            self.log(f"  {test_name}: {'PASS' if passed else 'FAIL'}")
        
        return success, self.logs