import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.logs = []
        
        try:
            # Safely extract input parameters from the solution instance
            if hasattr(solution_instance, 'results'):
                results_dict = solution_instance.results
            else:
                # If solution_instance itself is a dictionary
                results_dict = solution_instance if isinstance(solution_instance, dict) else {}
                self.logs.append("Warning: solution_instance doesn't have 'results' attribute, treating as dict")
            
            # Extract with safe defaults - handle missing keys gracefully
            self.U = float(results_dict.get('U', 100.0))
            self.m = float(results_dict.get('m', 9.11e-31))
            self.q = float(results_dict.get('q', 1.6e-19))
            self.d = float(results_dict.get('d', 0.01))
            
            # Ensure parameters are reasonable
            if abs(self.d) < 1e-30:
                self.d = 0.01
                self.logs.append("Warning: d was too small, set to 0.01")
            if abs(self.q) < 1e-30:
                self.q = 1.6e-19
                self.logs.append("Warning: q was too small, set to 1.6e-19")
            if abs(self.m) < 1e-30:
                self.m = 9.11e-31
                self.logs.append("Warning: m was too small, set to 9.11e-31")
            
            # Extract results with safe defaults
            self.v0 = float(results_dict.get('v_0', 0.0))
            self.B0 = float(results_dict.get('B_0', 0.0))
            self.vy = float(results_dict.get('v_y', 0.0))
            self.theta = float(results_dict.get('theta', 0.0))  # in radians
            self.tantheta = float(results_dict.get('tantheta', 0.0))
            self.v = float(results_dict.get('v', 0.0))
            self.a = float(results_dict.get('a', 0.0))
            self.r = float(results_dict.get('r', 0.0))
            self.L = float(results_dict.get('L', 0.0))
            
            # Constants computed from the actual parameters used in the solution
            self.L_expected = 2 * math.sqrt(3) * self.d
            self.r_expected = math.sqrt(3) * self.d
            self.tantheta_expected = 1 / math.sqrt(3)
            self.theta_expected_rad = math.atan(self.tantheta_expected)  # in radians
            
            # Expected formulas from physics - handle division by d and q safely
            # FIXED: Remove the problematic condition and compute directly
            try:
                # Compute v0_expected
                self.v0_expected = math.sqrt(3 * self.q * self.U / self.m)
                # Compute B0_expected
                self.B0_expected = (2 / self.d) * math.sqrt(self.U * self.m / (3 * self.q))
            except ZeroDivisionError:
                self.v0_expected = 0.0
                self.B0_expected = 0.0
                self.logs.append("Warning: Division by zero in expected value calculation")
            except Exception as e:
                self.v0_expected = 0.0
                self.B0_expected = 0.0
                self.logs.append(f"Warning: Error computing expected values: {str(e)}")
                
        except Exception as e:
            # If anything goes wrong during initialization, use safe defaults
            self.logs.append(f"Error during initialization: {str(e)}")
            self.U = 100.0
            self.m = 9.11e-31
            self.q = 1.6e-19
            self.d = 0.01
            self.v0 = 0.0
            self.B0 = 0.0
            self.vy = 0.0
            self.theta = 0.0
            self.tantheta = 0.0
            self.v = 0.0
            self.a = 0.0
            self.r = 0.0
            self.L = 0.0
            self.L_expected = 2 * math.sqrt(3) * self.d
            self.r_expected = math.sqrt(3) * self.d
            self.tantheta_expected = 1 / math.sqrt(3)
            self.theta_expected_rad = math.atan(self.tantheta_expected)
            # Compute expected values with safe defaults
            self.v0_expected = math.sqrt(3 * self.q * self.U / self.m)
            self.B0_expected = (2 / self.d) * math.sqrt(self.U * self.m / (3 * self.q))

    def log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success

    def test_invariants(self):
        """Verify conservation of energy and Newton's laws"""
        successes = []
        
        try:
            # 1. Energy conservation: KE_final - KE_initial = work done by electric field
            # Work done = q * (potential difference) = q * (U/2) since particle moves from midpoint to plate
            ke_initial = 0.5 * self.m * self.v0**2
            ke_final = 0.5 * self.m * self.v**2
            work = self.q * (self.U / 2)
            energy_balance = (ke_final - ke_initial) - work
            
            # Use relative tolerance for energy balance
            successes.append(self.log(
                f"Energy conservation: ΔKE = {ke_final - ke_initial:.3e}, work = {work:.3e}, diff = {energy_balance:.3e}",
                math.isclose(energy_balance, 0, rel_tol=1e-2, abs_tol=abs(work)*1e-2)
            ))
            
            # 2. Newton's second law in vertical direction: F = m*a = q*E
            if abs(self.d) > 1e-30:
                E = self.U / (2 * self.d)
                force_electric = self.q * E
                force_from_a = self.m * self.a
                successes.append(self.log(
                    f"Newton's 2nd law: qE = {force_electric:.3e}, m*a = {force_from_a:.3e}",
                    math.isclose(force_electric, force_from_a, rel_tol=1e-2, abs_tol=abs(force_electric)*1e-2)
                ))
            else:
                successes.append(self.log("Newton's 2nd law: d too small, skipping", True))
            
            # 3. Lorentz force as centripetal force (at point K)
            if abs(self.B0) > 1e-30 and abs(self.v) > 1e-30 and abs(self.r) > 1e-30:
                magnetic_force = self.q * self.B0 * self.v
                centripetal_force = self.m * self.v**2 / self.r
                successes.append(self.log(
                    f"Lorentz force: qvB = {magnetic_force:.3e}, mv²/r = {centripetal_force:.3e}",
                    math.isclose(magnetic_force, centripetal_force, rel_tol=1e-2, abs_tol=abs(magnetic_force)*1e-2)
                ))
            else:
                successes.append(self.log("Lorentz force: B0, v, or r too small, skipping", True))
                
        except Exception as e:
            self.logs.append(f"✗ Invariants test failed with error: {str(e)}")
            successes.append(False)
        
        return all(successes)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        successes = []
        
        try:
            # Check geometric relationships
            successes.append(self.log(
                f"Plate length L: expected {self.L_expected:.3e}, got {self.L:.3e}",
                math.isclose(self.L, self.L_expected, rel_tol=1e-2, abs_tol=abs(self.L_expected)*1e-2)
            ))
            
            # Check v0 from kinematics
            if abs(self.v0) > 1e-30 and abs(self.L) > 1e-30:
                t = self.L / self.v0  # time to cross plates
                if abs(t) > 1e-30:
                    a_expected = 2 * self.d / t**2
                    successes.append(self.log(
                        f"Acceleration from kinematics: a = {self.a:.3e}, expected {a_expected:.3e}",
                        math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=abs(a_expected)*1e-2)
                    ))
                else:
                    successes.append(self.log("Time t too small, skipping acceleration check", True))
            else:
                successes.append(self.log("v0 or L too small, skipping kinematics check", True))
            
            # Check vertical velocity component
            if abs(self.v0) > 1e-30 and abs(self.L) > 1e-30:
                t = self.L / self.v0
                if abs(t) > 1e-30:
                    vy_from_t = 2 * self.d / t
                    successes.append(self.log(
                        f"Vertical velocity: v_y = {self.vy:.3e}, expected {vy_from_t:.3e}",
                        math.isclose(self.vy, vy_from_t, rel_tol=1e-2, abs_tol=abs(vy_from_t)*1e-2)
                    ))
                else:
                    successes.append(self.log("Time t too small, skipping v_y check", True))
            else:
                successes.append(self.log("v0 or L too small, skipping v_y check", True))
            
            # Check velocity magnitude at K
            v_mag = math.sqrt(self.v0**2 + self.vy**2)
            successes.append(self.log(
                f"Velocity magnitude: v = {self.v:.3e}, expected {v_mag:.3e}",
                math.isclose(self.v, v_mag, rel_tol=1e-2, abs_tol=abs(v_mag)*1e-2)
            ))
            
            # Check tangent of angle
            if abs(self.v0) > 1e-30:
                tantheta_calc = self.vy / self.v0
                successes.append(self.log(
                    f"tanθ: {self.tantheta:.3e}, expected {tantheta_calc:.3e}",
                    math.isclose(self.tantheta, tantheta_calc, rel_tol=1e-2, abs_tol=abs(tantheta_calc)*1e-2)
                ))
            else:
                successes.append(self.log("v0 too small, skipping tanθ check", True))
            
            # Check angle - theta is stored in radians, so compare in radians
            if abs(self.v0) > 1e-30:
                theta_calc_rad = math.atan2(self.vy, self.v0)
                successes.append(self.log(
                    f"Angle θ: {self.theta:.3e} rad, expected {theta_calc_rad:.3e} rad",
                    math.isclose(self.theta, theta_calc_rad, rel_tol=1e-2, abs_tol=abs(theta_calc_rad)*1e-2)
                ))
            else:
                successes.append(self.log("v0 too small, skipping angle check", True))
            
            # Check radius of circular motion
            successes.append(self.log(
                f"Radius r: {self.r:.3e}, expected {self.r_expected:.3e}",
                math.isclose(self.r, self.r_expected, rel_tol=1e-2, abs_tol=abs(self.r_expected)*1e-2)
            ))
            
            # Verify target answers using the correct expected formulas
            # FIXED: Use the properly computed expected values
            successes.append(self.log(
                f"v0: {self.v0:.3e}, expected {self.v0_expected:.3e}",
                math.isclose(self.v0, self.v0_expected, rel_tol=1e-2, abs_tol=abs(self.v0_expected)*1e-2)
            ))
            successes.append(self.log(
                f"B0: {self.B0:.3e}, expected {self.B0_expected:.3e}",
                math.isclose(self.B0, self.B0_expected, rel_tol=1e-2, abs_tol=abs(self.B0_expected)*1e-2)
            ))
            
        except Exception as e:
            self.logs.append(f"✗ Trace consistency test failed with error: {str(e)}")
            successes.append(False)
        
        return all(successes)

    def test_monotonicity(self):
        """Perturb input U and check trends"""
        successes = []
        
        try:
            # Save original values
            U_orig = self.U
            v0_orig = self.v0
            B0_orig = self.B0
            
            # Perturb U by +10%
            self.U *= 1.1
            
            # Recalculate expected values using formulas
            # FIXED: Use the same calculation logic as in __init__
            try:
                v0_new = math.sqrt(3 * self.q * self.U / self.m)
                B0_new = (2 / self.d) * math.sqrt(self.U * self.m / (3 * self.q))
            except ZeroDivisionError:
                v0_new = 0
                B0_new = 0
            except Exception:
                v0_new = 0
                B0_new = 0
            
            # Check monotonic trends
            successes.append(self.log(
                f"U increased → v0 increased: {v0_new:.3e} > {v0_orig:.3e}",
                v0_new > v0_orig or math.isclose(v0_new, v0_orig, rel_tol=1e-10)
            ))
            successes.append(self.log(
                f"U increased → B0 increased: {B0_new:.3e} > {B0_orig:.3e}",
                B0_new > B0_orig or math.isclose(B0_new, B0_orig, rel_tol=1e-10)
            ))
            
            # Restore original values
            self.U = U_orig
            
        except Exception as e:
            self.logs.append(f"✗ Monotonicity test failed with error: {str(e)}")
            successes.append(False)
        
        return all(successes)

    def run_all_tests(self):
        """Run all checks and return success flag with logs"""
        self.logs = []
        overall_success = True
        
        try:
            self.logs.append("=== Physics Checker Started ===")
            self.logs.append(f"Parameters: U={self.U}, q={self.q}, m={self.m}, d={self.d}")
            self.logs.append(f"Expected v0: {self.v0_expected:.3e}, Expected B0: {self.B0_expected:.3e}")
            
            # Run tests
            for test_name, test_func in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                self.logs.append(f"\n--- Testing {test_name} ---")
                try:
                    if not test_func():
                        overall_success = False
                except Exception as e:
                    self.logs.append(f"✗ {test_name} failed with error: {str(e)}")
                    self.logs.append(traceback.format_exc())
                    overall_success = False
            
            self.logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
            return overall_success, self.logs
            
        except Exception as e:
            self.logs.append(f"✗ Checker crashed: {str(e)}")
            return False, self.logs