import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.v1 = getattr(solution_instance, 'v1', 0.0)
        
        # Extract results with safe defaults
        results = getattr(solution_instance, 'results', {})
        self.E_p = results.get('E_p', 0.0)        # Elastic potential at A
        self.W = results.get('W', 0.0)           # Friction work from B to C
        self.x = results.get('x', 0.0)           # Horizontal displacement from C
        self.v2 = results.get('v_2', 0.0)        # Velocity at C
        self.t = results.get('t', 0.0)           # Time of flight from C
        
        # Initialize logs
        self.logs = []
        
    def log(self, msg, success=None):
        """Helper to log messages and track success status"""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        
    def fuzzy_equal(self, a, b, msg="", rel_tol=1e-2, abs_tol=1e-2):
        """Check if two numbers are close within tolerance"""
        result = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        self.log(f"{msg}: {a:.4f} ≈ {b:.4f}", success=result)
        return result
        
    def test_step1_elastic_potential(self):
        """Check E_p = 0.5*m*v1^2 (spring energy converts to kinetic at B)"""
        expected = 0.5 * self.m * self.v1**2
        return self.fuzzy_equal(self.E_p, expected, 
                              f"Step 1: Elastic potential E_p = 0.5*m*v1²")
        
    def test_step2_work_energy_theorem(self):
        """Check W = -2mgR + 0.5*m*v2² - 0.5*m*v1² (net work B to C)"""
        # Note: W here is friction work, but this formula is for net work
        # Actually net work = work_by_gravity + work_by_friction
        # work_by_gravity = -mg(2R)
        # So: -2mgR + W_friction = 0.5*m*v2² - 0.5*m*v1²
        # Therefore: W_friction = 0.5*m*v2² - 0.5*m*v1² + 2mgR
        kinetic_change = 0.5 * self.m * (self.v2**2 - self.v1**2)
        gravity_work = -self.m * self.g * 2 * self.R
        expected_net_work = gravity_work + self.W  # Should equal kinetic_change
        return self.fuzzy_equal(kinetic_change, expected_net_work,
                              f"Step 2: Work-energy theorem (B to C)")
        
    def test_step3_centripetal_at_C(self):
        """Check mg = m*v2²/R (critical condition at top of track)"""
        centripetal = self.m * self.v2**2 / self.R
        return self.fuzzy_equal(self.m * self.g, centripetal,
                              f"Step 3: Centripetal force at C")
        
    def test_step4_friction_work(self):
        """Check W = 2.5*mgR - 0.5*m*v1² (after substituting v2² = gR)"""
        expected = 2.5 * self.m * self.g * self.R - 0.5 * self.m * self.v1**2
        return self.fuzzy_equal(self.W, expected,
                              f"Step 4: Friction work W")
        
    def test_step5_v2_value(self):
        """Check v2 = √(gR) from centripetal condition"""
        expected = math.sqrt(self.g * self.R)
        return self.fuzzy_equal(self.v2, expected,
                              f"Step 5: Velocity at C v₂ = √(gR)")
        
    def test_step6_free_fall_time(self):
        """Check 2R = 0.5*g*t² (vertical fall from height 2R)"""
        fall_distance = 0.5 * self.g * self.t**2
        return self.fuzzy_equal(2 * self.R, fall_distance,
                              f"Step 6: Free fall time from C")
        
    def test_step7_horizontal_motion(self):
        """Check x = v2*t (uniform horizontal motion)"""
        expected = self.v2 * self.t
        return self.fuzzy_equal(self.x, expected,
                              f"Step 7: Horizontal displacement x = v₂t")
        
    def test_step8_final_distance(self):
        """Check x = 2R (landing point distance from B)"""
        return self.fuzzy_equal(self.x, 2 * self.R,
                              f"Step 8: Distance from landing to B")
        
    def test_energy_conservation_A_to_B(self):
        """Check spring energy fully converts to kinetic at B (smooth)"""
        # A: spring potential, B: kinetic
        energy_A = self.E_p
        energy_B = 0.5 * self.m * self.v1**2
        return self.fuzzy_equal(energy_A, energy_B,
                              f"Energy conservation A→B (smooth)")
        
    def test_energy_balance_B_to_C(self):
        """Check work-energy balance with friction"""
        # Initial energy at B
        E_B = 0.5 * self.m * self.v1**2
        
        # Energy at C (height 2R)
        E_C = self.m * self.g * 2 * self.R + 0.5 * self.m * self.v2**2
        
        # Friction work should account for difference
        energy_diff = E_C - E_B
        return self.fuzzy_equal(self.W, energy_diff,
                              f"Energy balance B→C with friction")
        
    def test_invariants(self):
        """Main physics invariants"""
        self.log("\n=== Testing Physics Invariants ===")
        
        tests = [
            self.test_step1_elastic_potential,
            self.test_step3_centripetal_at_C,
            self.test_step5_v2_value,
            self.test_step6_free_fall_time,
            self.test_step8_final_distance,
            self.test_energy_conservation_A_to_B,
        ]
        
        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                self.log(f"Test failed with error: {str(e)}", success=False)
                results.append(False)
                
        return all(results)
        
    def test_trace_consistency(self):
        """Check intermediate results are consistent with final answers"""
        self.log("\n=== Testing Trace Consistency ===")
        
        # Check the derived equations chain
        try:
            # From centripetal: v2² = gR
            # Friction work: W = 0.5*m*v2² - 0.5*m*v1² + 2mgR
            expected_from_derivation = 0.5 * self.m * self.g * self.R - 0.5 * self.m * self.v1**2 + 2 * self.m * self.g * self.R
            expected_from_derivation = 2.5 * self.m * self.g * self.R - 0.5 * self.m * self.v1**2
            derivation_ok = self.fuzzy_equal(self.W, expected_from_derivation,
                                           "Derivation: W_friction from v2²=gR")
            
        except Exception as e:
            self.log(f"Trace consistency error: {str(e)}", success=False)
            derivation_ok = False
            
        # Check x = v2*t and t from free fall are consistent
        try:
            t_from_fall = math.sqrt(4 * self.R / self.g)
            x_from_motion = self.v2 * t_from_fall
            motion_ok = self.fuzzy_equal(self.x, x_from_motion,
                                       "Consistency: x from v₂ and free fall time")
        except Exception as e:
            self.log(f"Motion consistency error: {str(e)}", success=False)
            motion_ok = False
            
        return derivation_ok and motion_ok
        
    def test_monotonicity(self):
        """Perturb parameters and check trends"""
        self.log("\n=== Testing Monotonicity ===")
        
        # Store original values
        original_m = self.m
        original_R = self.R
        original_v1 = self.v1
        original_E_p = self.E_p
        original_W = self.W
        original_x = self.x
        
        try:
            # Test 1: Increase mass by 10%
            self.m = original_m * 1.1
            new_E_p = 0.5 * self.m * original_v1**2
            new_W = 2.5 * self.m * self.g * self.R - 0.5 * self.m * original_v1**2
            new_x = 2 * self.R
            
            # E_p should increase with m (if v1 fixed)
            E_p_trend = math.isclose(new_E_p, original_E_p * 1.1, rel_tol=1e-2)
            # W should increase with m (if other parameters fixed)
            W_trend = math.isclose(new_W, original_W * 1.1, rel_tol=1e-2)
            # x should not change with m
            x_trend = math.isclose(new_x, original_x, rel_tol=1e-2)
            
            self.log(f"Mass increase: E_p trend correct ({original_E_p:.2f} → {new_E_p:.2f})", 
                   success=E_p_trend)
            self.log(f"Mass increase: W trend correct ({original_W:.2f} → {new_W:.2f})",
                   success=W_trend)
            self.log(f"Mass increase: x invariant ({original_x:.2f} → {new_x:.2f})",
                   success=x_trend)
            
            # Restore mass, increase radius
            self.m = original_m
            self.R = original_R * 1.1
            
            new_E_p = 0.5 * self.m * original_v1**2
            new_W = 2.5 * self.m * self.g * self.R - 0.5 * self.m * original_v1**2
            new_x = 2 * self.R
            
            # E_p should not change with R
            E_p_trend2 = math.isclose(new_E_p, original_E_p, rel_tol=1e-2)
            # W should increase with R (positive coefficient)
            W_trend2 = new_W > original_W
            # x should increase with R
            x_trend2 = new_x > original_x
            
            self.log(f"Radius increase: E_p invariant ({original_E_p:.2f} → {new_E_p:.2f})",
                   success=E_p_trend2)
            self.log(f"Radius increase: W trend correct ({original_W:.2f} → {new_W:.2f})",
                   success=W_trend2)
            self.log(f"Radius increase: x trend correct ({original_x:.2f} → {new_x:.2f})",
                   success=x_trend2)
            
            monotonicity_ok = (E_p_trend and W_trend and x_trend and 
                             E_p_trend2 and W_trend2 and x_trend2)
            
        except Exception as e:
            self.log(f"Monotonicity test error: {str(e)}", success=False)
            monotonicity_ok = False
            
        finally:
            # Restore original values
            self.m = original_m
            self.R = original_R
            self.v1 = original_v1
            self.E_p = original_E_p
            self.W = original_W
            self.x = original_x
            
        return monotonicity_ok
        
    def run_all_tests(self):
        """Execute all tests and return overall success"""
        self.logs = []  # Clear previous logs
        
        try:
            self.log("=" * 60)
            self.log("PhysicsChecker - Semicircular Track Problem")
            self.log("=" * 60)
            
            # Run test suites
            invariants_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            monotonicity_ok = self.test_monotonicity()
            
            # Additional step tests
            self.log("\n=== Additional Step Tests ===")
            step_tests = [
                self.test_step2_work_energy_theorem,
                self.test_step4_friction_work,
                self.test_step7_horizontal_motion,
                self.test_energy_balance_B_to_C,
            ]
            
            step_results = []
            for test in step_tests:
                try:
                    step_results.append(test())
                except Exception as e:
                    self.log(f"Step test error: {str(e)}", success=False)
                    step_results.append(False)
            
            steps_ok = all(step_results)
            
            # Overall success
            overall_success = (invariants_ok and trace_ok and 
                             monotonicity_ok and steps_ok)
            
            self.log("\n" + "=" * 60)
            self.log(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
            self.log("=" * 60)
            
            return overall_success, self.logs
            
        except Exception as e:
            self.log(f"Checker crashed: {traceback.format_exc()}", success=False)
            return False, self.logs