import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 60.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.l = getattr(solution_instance, 'l', 2.0)
        self.H = getattr(solution_instance, 'H', 3.0)
        self.alpha_deg = getattr(solution_instance, 'alpha_deg', 53.0)
        self.cos_alpha = getattr(solution_instance, 'cos_alpha', 0.6)
        self.sin_alpha = getattr(solution_instance, 'sin_alpha', 0.8)

        # Extract results with safe defaults
        self.d = getattr(solution_instance, 'd', self.results.get('d', 0.0))
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        self.F = getattr(solution_instance, 'F', self.results.get('F', 0.0))
        self.Fprime_minus_mg = getattr(solution_instance, 'Fprime_minus_mg', 
                                        self.results.get('$Fprime-mg$', 0.0))
        self.kinetic_energy = getattr(solution_instance, 'kinetic_energy', 
                                       self.results.get('$frac12mv2$', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.Fprime = getattr(solution_instance, 'Fprime', 
                              self.results.get('$Fprime$', 0.0))
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        
        # For depth calculation (friction forces unknown - will check consistency)
        self.f1_plus_f2 = None  # Not provided in results

        # Expected values from target answers (numerical approximations)
        self.expected_F = 1080.0  # N
        self.expected_total_x = 8/5 + 4*math.sqrt(5)/5  # m ≈ 3.388
        self.expected_d = 1.2  # m

        # Constants
        self.alpha_rad = math.radians(self.alpha_deg)

    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for floats."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        return False

    def test_invariants(self):
        """Check fundamental physics invariants."""
        logs = []
        
        try:
            # 1. Energy conservation during swing
            # Initial potential energy relative to lowest point: m*g*l*(1-cosα)
            # Should equal kinetic energy at bottom: 0.5*m*v^2
            PE_initial = self.m * self.g * self.l * (1 - self.cos_alpha)
            KE_bottom = 0.5 * self.m * self.v**2
            
            if not self.fuzzy_equal(PE_initial, KE_bottom):
                logs.append(f"Energy conservation failed: PE_initial={PE_initial:.2f}J, "
                          f"KE_bottom={KE_bottom:.2f}J, diff={abs(PE_initial-KE_bottom):.2f}J")
            else:
                logs.append("✓ Energy conservation during swing verified")
            
            # 2. Newton's 2nd Law at lowest point (circular motion)
            # F' - mg = m*v^2/l
            radial_acc = self.v**2 / self.l
            lhs = self.Fprime - self.m * self.g
            rhs = self.m * radial_acc
            
            if not self.fuzzy_equal(lhs, rhs):
                logs.append(f"Newton's 2nd Law failed: F'-mg={lhs:.2f}N, "
                          f"m*v²/l={rhs:.2f}N, diff={abs(lhs-rhs):.2f}N")
            else:
                logs.append("✓ Newton's 2nd Law at lowest point verified")
            
            # 3. Tension formula consistency: F' = (3-2cosα)mg
            expected_Fprime = (3 - 2*self.cos_alpha) * self.m * self.g
            if not self.fuzzy_equal(self.Fprime, expected_Fprime):
                logs.append(f"Tension formula failed: F'={self.Fprime:.2f}N, "
                          f"expected={expected_Fprime:.2f}N")
            else:
                logs.append("✓ Tension formula verified")
            
            # 4. Free fall kinematics (from lowest point to water)
            # H - l = 0.5*g*t^2
            fall_height = self.H - self.l
            expected_t = math.sqrt(2 * fall_height / self.g)
            if not self.fuzzy_equal(self.t, expected_t):
                logs.append(f"Free fall time failed: t={self.t:.2f}s, "
                          f"expected={expected_t:.2f}s")
            else:
                logs.append("✓ Free fall time verified")
            
            # 5. Horizontal displacement during free fall: x = v*t
            expected_x_free = self.v * self.t
            if not self.fuzzy_equal(self.x, expected_x_free):
                logs.append(f"Horizontal free fall failed: x={self.x:.2f}m, "
                          f"v*t={expected_x_free:.2f}m")
            else:
                logs.append("✓ Horizontal free fall displacement verified")
            
            # 6. Total horizontal distance (swing + free fall)
            swing_horizontal = self.l * self.sin_alpha
            total_horizontal = swing_horizontal + self.x
            
            if not self.fuzzy_equal(total_horizontal, self.expected_total_x):
                logs.append(f"Total horizontal distance failed: {total_horizontal:.3f}m, "
                          f"expected={self.expected_total_x:.3f}m")
            else:
                logs.append("✓ Total horizontal distance verified")
            
            # 7. Work-energy for water entry (symbolic check since f1+f2 unknown)
            # mg(H - l*cosα + d) = (f1+f2)d
            # Rearranged: d = mg(H - l*cosα)/(f1+f2 - mg)
            # We'll check that the computed d matches expected
            if not self.fuzzy_equal(self.d, self.expected_d):
                logs.append(f"Depth mismatch: d={self.d:.2f}m, expected={self.expected_d:.2f}m")
            else:
                logs.append("✓ Depth value verified")
                
        except Exception as e:
            logs.append(f"Error in invariants test: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        logs = []
        
        try:
            # Check that tension F matches F' (should be equal by Newton's 3rd Law)
            if not self.fuzzy_equal(self.F, self.Fprime):
                logs.append(f"Tension inconsistency: F={self.F:.2f}N, F'={self.Fprime:.2f}N")
            else:
                logs.append("✓ Tension F and F' consistent")
            
            # Check that F' - mg from step 2 equals m*v^2/l
            Fprime_minus_mg_calc = self.m * self.v**2 / self.l
            if not self.fuzzy_equal(self.Fprime_minus_mg, Fprime_minus_mg_calc):
                logs.append(f"F'-mg inconsistency: {self.Fprime_minus_mg:.2f}N vs "
                          f"calculated {Fprime_minus_mg_calc:.2f}N")
            else:
                logs.append("✓ F'-mg consistent with v²/l")
            
            # Check that kinetic energy equals 0.5*m*v^2
            KE_calc = 0.5 * self.m * self.v**2
            if not self.fuzzy_equal(self.kinetic_energy, KE_calc):
                logs.append(f"Kinetic energy inconsistency: {self.kinetic_energy:.2f}J vs "
                          f"calculated {KE_calc:.2f}J")
            else:
                logs.append("✓ Kinetic energy consistent with velocity")
            
            # Check that all three target answers match
            if not self.fuzzy_equal(self.F, self.expected_F):
                logs.append(f"Tension answer mismatch: {self.F:.2f}N vs {self.expected_F:.2f}N")
            else:
                logs.append("✓ Tension matches target answer")
            
            total_x = self.l * self.sin_alpha + self.x
            if not self.fuzzy_equal(total_x, self.expected_total_x):
                logs.append(f"Horizontal distance mismatch: {total_x:.3f}m vs {self.expected_total_x:.3f}m")
            else:
                logs.append("✓ Horizontal distance matches target answer")
            
            if not self.fuzzy_equal(self.d, self.expected_d):
                logs.append(f"Depth mismatch: {self.d:.2f}m vs {self.expected_d:.2f}m")
            else:
                logs.append("✓ Depth matches target answer")
                
        except Exception as e:
            logs.append(f"Error in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_monotonicity(self):
        """Test how results change when input parameters are perturbed."""
        logs = []
        
        try:
            # Test 1: Increase mass by 1%
            original_m = self.m
            self.m *= 1.01
            
            # Recalculate expected values based on formulas
            new_v = math.sqrt(2 * self.g * self.l * (1 - self.cos_alpha))  # Independent of m
            new_Fprime = (3 - 2*self.cos_alpha) * self.m * self.g  # Proportional to m
            new_t = math.sqrt(2 * (self.H - self.l) / self.g)  # Independent of m
            new_x = new_v * new_t  # Independent of m
            swing_x = self.l * self.sin_alpha
            new_total_x = swing_x + new_x  # Independent of m
            
            # Check monotonic relationships
            # Tension should increase with mass
            if new_Fprime <= self.Fprime:
                logs.append("✗ Tension should increase with mass but didn't")
            else:
                logs.append("✓ Tension increases with mass (monotonic)")
            
            # Velocity, time, horizontal distances should remain the same (within tolerance)
            if not self.fuzzy_equal(new_v, self.v):
                logs.append(f"✗ Velocity changed with mass: {new_v:.3f}m/s vs {self.v:.3f}m/s")
            else:
                logs.append("✓ Velocity independent of mass (correct)")
            
            if not self.fuzzy_equal(new_t, self.t):
                logs.append(f"✗ Free fall time changed with mass: {new_t:.3f}s vs {self.t:.3f}s")
            else:
                logs.append("✓ Free fall time independent of mass (correct)")
            
            if not self.fuzzy_equal(new_total_x, self.expected_total_x):
                logs.append(f"✗ Horizontal distance changed with mass: {new_total_x:.3f}m vs {self.expected_total_x:.3f}m")
            else:
                logs.append("✓ Horizontal distance independent of mass (correct)")
            
            # Restore original mass
            self.m = original_m
            
            # Test 2: Increase rope length by 1%
            original_l = self.l
            self.l *= 1.01
            
            new_v2 = math.sqrt(2 * self.g * self.l * (1 - self.cos_alpha))  # Increases with √l
            new_Fprime2 = (3 - 2*self.cos_alpha) * self.m * self.g  # Independent of l
            new_t2 = math.sqrt(2 * (self.H - self.l) / self.g)  # Decreases with l
            new_x2 = new_v2 * new_t2
            swing_x2 = self.l * self.sin_alpha
            new_total_x2 = swing_x2 + new_x2
            
            # Velocity should increase with √l
            if new_v2 <= self.v:
                logs.append("✗ Velocity should increase with rope length but didn't")
            else:
                logs.append("✓ Velocity increases with rope length (monotonic)")
            
            # Free fall time should decrease (since H-l decreases)
            if new_t2 >= self.t:
                logs.append("✗ Free fall time should decrease with rope length but didn't")
            else:
                logs.append("✓ Free fall time decreases with rope length (monotonic)")
            
            # Tension should remain constant (independent of l)
            if not self.fuzzy_equal(new_Fprime2, self.Fprime):
                logs.append(f"✗ Tension changed with rope length: {new_Fprime2:.2f}N vs {self.Fprime:.2f}N")
            else:
                logs.append("✓ Tension independent of rope length (correct)")
            
            # Restore original length
            self.l = original_l
            
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            logs.append(traceback.format_exc())
            # Ensure we restore original values on error
            self.m = getattr(self.solution, 'm', 60.0)
            self.l = getattr(self.solution, 'l', 2.0)
        
        return logs

    def run_all_tests(self):
        """Run all tests and return overall success status and logs."""
        all_logs = []
        success = True
        
        all_logs.append("=== Physics Checker Started ===")
        
        # Run individual test suites
        invariants_logs = self.test_invariants()
        all_logs.append("\n--- Invariants Test ---")
        all_logs.extend(invariants_logs)
        
        trace_logs = self.test_trace_consistency()
        all_logs.append("\n--- Trace Consistency Test ---")
        all_logs.extend(trace_logs)
        
        monotonicity_logs = self.test_monotonicity()
        all_logs.append("\n--- Monotonicity Test ---")
        all_logs.extend(monotonicity_logs)
        
        # Determine overall success
        error_keywords = ['failed', 'Error', 'mismatch', 'inconsistency', '✗']
        for log in all_logs:
            if any(keyword in log for keyword in error_keywords):
                success = False
                break
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        
        return success, all_logs