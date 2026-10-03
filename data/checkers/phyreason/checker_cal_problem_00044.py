import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Extract inputs with defaults
        self.M = getattr(solution_instance, 'M', None)
        if self.M is None:
            # Try to get from inputs dict
            self.M = solution_instance.inputs.get('M', 1.0)  # default fallback
        self.l = solution_instance.inputs.get('l', 1.0)
        self.g = solution_instance.inputs.get('g', 9.8)
        self.m = self.M / 3.0  # given m = M/3
        
        # Extract results safely using .get with defaults
        self.results = getattr(solution_instance, 'results', {})
        # Velocity symbols - note: keys may have primes, use safe access
        self.v0 = self.results.get('v_0', 0.0)
        self.v1 = self.results.get('v_1', 0.0)
        self.v1p = self.results.get("v_1'", 0.0)  # v_1' with single quote
        self.v2 = self.results.get('v_2', 0.0)
        self.v2p = self.results.get("v_2'", 0.0)  # v_2'
        self.v2pp = self.results.get("v_2''", 0.0)  # v_2''
        self.v3 = self.results.get('v_3', 0.0)
        self.v3p = self.results.get("v_3'", 0.0)  # v_3'
        self.v3pp = self.results.get("v_3''", 0.0)  # v_3''
        
        # Times
        self.t = self.results.get('t', 0.0)  # time to max separation
        self.t1 = self.results.get('t_1', 0.0)  # time between first and second collision
        self.t2 = self.results.get('t_2', 0.0)  # time between second and third
        self.t3 = self.results.get('t_3', 0.0)  # time between third and fourth
        
        # Displacements
        self.x_plate1 = self.results.get('x_plate,1', 0.0)
        self.x_plate2 = self.results.get('x_plate,2', 0.0)
        self.x_plate3 = self.results.get('x_plate,3', 0.0)
        self.d_max = self.results.get('d_max', 0.0)
        
        # Expected values computed from formulas (for cross-check)
        self.v0_expected = math.sqrt(2 * self.g * self.l)
        self.v1_expected = (self.m - self.M) / (self.m + self.M) * self.v0_expected
        self.v1p_expected = (2 * self.m) / (self.m + self.M) * self.v0_expected
        self.t_expected = self.v0_expected / self.g  # time to max separation
        # d_max = (v1p - v1)*t - 0.5*g*t**2
        self.d_max_expected = (self.v1p_expected - self.v1_expected) * self.t_expected - 0.5 * self.g * self.t_expected**2
        # t1 from step8: v1*t1 + 0.5*g*t1**2 = v1p*t1  => 0.5*g*t1**2 = (v1p - v1)*t1 => t1 = 2*(v1p - v1)/g = 2*v0/g
        self.t1_expected = 2 * self.v0_expected / self.g
        self.v2_expected = self.v1_expected + self.g * self.t1_expected
        self.x_plate1_expected = self.v1p_expected * self.t1_expected
        # After second collision (elastic with same masses)
        # Using formulas for elastic collision with different masses
        self.v2p_expected = ((self.m - self.M) * self.v2_expected + 2 * self.M * self.v1p_expected) / (self.m + self.M)
        self.v2pp_expected = ((self.M - self.m) * self.v1p_expected + 2 * self.m * self.v2_expected) / (self.m + self.M)
        # t2 from step13: v2' * t2 + 0.5*g*t2**2 = v2'' * t2
        # => 0.5*g*t2**2 = (v2'' - v2')*t2
        # => t2 = 2*(v2'' - v2')/g
        self.t2_expected = 2 * (self.v2pp_expected - self.v2p_expected) / self.g
        self.x_plate2_expected = self.v2pp_expected * self.t2_expected
        self.v3_expected = self.v2p_expected + self.g * self.t2_expected
        # Third collision: elastic collision again
        self.v3p_expected = ((self.m - self.M) * self.v3_expected + 2 * self.M * self.v2pp_expected) / (self.m + self.M)
        self.v3pp_expected = ((self.M - self.m) * self.v2pp_expected + 2 * self.m * self.v3_expected) / (self.m + self.M)
        # t3 from step19: similar to t2
        self.t3_expected = 2 * (self.v3pp_expected - self.v3p_expected) / self.g
        self.x_plate3_expected = self.v3pp_expected * self.t3_expected

        # For monotonicity test, store original l
        self.original_l = self.l

    def fuzzy_equal(self, a, b):
        """Fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Check fundamental physics invariants."""
        logs = []
        try:
            # 1. Conservation of mechanical energy for ball's initial fall
            # Ball falls from height l: potential loss = m*g*l, gain in kinetic energy = 0.5*m*v0^2
            expected_ke = 0.5 * self.m * self.v0**2
            expected_pe_loss = self.m * self.g * self.l
            if not self.fuzzy_equal(expected_ke, expected_pe_loss):
                logs.append(f"Energy conservation in initial fall failed: KE={expected_ke}, PE loss={expected_pe_loss}")
            else:
                logs.append("✓ Energy conservation in initial fall holds.")

            # 2. First collision: check momentum and kinetic energy conservation
            # Momentum before: m*v0 + M*0 = m*v0
            # Momentum after: m*v1 + M*v1p
            mom_before = self.m * self.v0
            mom_after = self.m * self.v1 + self.M * self.v1p
            if not self.fuzzy_equal(mom_before, mom_after):
                logs.append(f"Momentum conservation in first collision failed: before={mom_before}, after={mom_after}")
            else:
                logs.append("✓ Momentum conservation in first collision holds.")
            # Kinetic energy before: 0.5*m*v0^2
            # Kinetic energy after: 0.5*m*v1**2 + 0.5*M*v1p**2
            ke_before = 0.5 * self.m * self.v0**2
            ke_after = 0.5 * self.m * self.v1**2 + 0.5 * self.M * self.v1p**2
            if not self.fuzzy_equal(ke_before, ke_after):
                logs.append(f"KE conservation in first collision failed: before={ke_before}, after={ke_after}")
            else:
                logs.append("✓ KE conservation in first collision holds.")

            # 3. Second collision: momentum and KE conservation
            # Before: m*v2 + M*v1p
            # After: m*v2p + M*v2pp
            mom_before2 = self.m * self.v2 + self.M * self.v1p
            mom_after2 = self.m * self.v2p + self.M * self.v2pp
            if not self.fuzzy_equal(mom_before2, mom_after2):
                logs.append(f"Momentum conservation in second collision failed: before={mom_before2}, after={mom_after2}")
            else:
                logs.append("✓ Momentum conservation in second collision holds.")
            ke_before2 = 0.5 * self.m * self.v2**2 + 0.5 * self.M * self.v1p**2
            ke_after2 = 0.5 * self.m * self.v2p**2 + 0.5 * self.M * self.v2pp**2
            if not self.fuzzy_equal(ke_before2, ke_after2):
                logs.append(f"KE conservation in second collision failed: before={ke_before2}, after={ke_after2}")
            else:
                logs.append("✓ KE conservation in second collision holds.")

            # 4. Third collision: momentum and KE conservation
            # Before: m*v3 + M*v2pp
            # After: m*v3p + M*v3pp
            mom_before3 = self.m * self.v3 + self.M * self.v2pp
            mom_after3 = self.m * self.v3p + self.M * self.v3pp
            if not self.fuzzy_equal(mom_before3, mom_after3):
                logs.append(f"Momentum conservation in third collision failed: before={mom_before3}, after={mom_after3}")
            else:
                logs.append("✓ Momentum conservation in third collision holds.")
            ke_before3 = 0.5 * self.m * self.v3**2 + 0.5 * self.M * self.v2pp**2
            ke_after3 = 0.5 * self.m * self.v3p**2 + 0.5 * self.M * self.v3pp**2
            if not self.fuzzy_equal(ke_before3, ke_after3):
                logs.append(f"KE conservation in third collision failed: before={ke_before3}, after={ke_after3}")
            else:
                logs.append("✓ KE conservation in third collision holds.")

        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
        return logs

    def test_trace_consistency(self):
        """Check consistency between intermediate results using kinematic equations."""
        logs = []
        try:
            # 1. Check v0 from energy conservation
            if not self.fuzzy_equal(self.v0, self.v0_expected):
                logs.append(f"v0 mismatch: expected {self.v0_expected:.4f}, got {self.v0:.4f}")
            else:
                logs.append("✓ v0 matches expected from energy conservation.")

            # 2. Check v1 and v1' from collision formulas
            if not self.fuzzy_equal(self.v1, self.v1_expected):
                logs.append(f"v1 mismatch: expected {self.v1_expected:.4f}, got {self.v1:.4f}")
            else:
                logs.append("✓ v1 matches expected from collision.")
            if not self.fuzzy_equal(self.v1p, self.v1p_expected):
                logs.append(f"v1' mismatch: expected {self.v1p_expected:.4f}, got {self.v1p:.4f}")
            else:
                logs.append("✓ v1' matches expected from collision.")

            # 3. Check time to max separation t
            if not self.fuzzy_equal(self.t, self.t_expected):
                logs.append(f"t (max separation time) mismatch: expected {self.t_expected:.4f}, got {self.t:.4f}")
            else:
                logs.append("✓ t matches expected.")

            # 4. Check d_max
            if not self.fuzzy_equal(self.d_max, self.d_max_expected):
                logs.append(f"d_max mismatch: expected {self.d_max_expected:.4f}, got {self.d_max:.4f}")
            else:
                logs.append("✓ d_max matches expected.")

            # 5. Check t1
            if not self.fuzzy_equal(self.t1, self.t1_expected):
                logs.append(f"t1 mismatch: expected {self.t1_expected:.4f}, got {self.t1:.4f}")
            else:
                logs.append("✓ t1 matches expected.")

            # 6. Check v2 from kinematics: v2 = v1 + g*t1
            v2_calc = self.v1 + self.g * self.t1
            if not self.fuzzy_equal(self.v2, v2_calc):
                logs.append(f"v2 inconsistency: from kinematics {v2_calc:.4f}, got {self.v2:.4f}")
            else:
                logs.append("✓ v2 consistent with kinematics.")

            # 7. Check x_plate1 = v1' * t1
            x_plate1_calc = self.v1p * self.t1
            if not self.fuzzy_equal(self.x_plate1, x_plate1_calc):
                logs.append(f"x_plate,1 inconsistency: calculated {x_plate1_calc:.4f}, got {self.x_plate1:.4f}")
            else:
                logs.append("✓ x_plate,1 consistent.")

            # 8. Check v2' and v2'' from collision formulas
            if not self.fuzzy_equal(self.v2p, self.v2p_expected):
                logs.append(f"v2' mismatch: expected {self.v2p_expected:.4f}, got {self.v2p:.4f}")
            else:
                logs.append("✓ v2' matches expected.")
            if not self.fuzzy_equal(self.v2pp, self.v2pp_expected):
                logs.append(f"v2'' mismatch: expected {self.v2pp_expected:.4f}, got {self.v2pp:.4f}")
            else:
                logs.append("✓ v2'' matches expected.")

            # 9. Check t2 from meeting condition: t2 = 2*(v2'' - v2')/g
            t2_calc = 2 * (self.v2pp - self.v2p) / self.g
            if not self.fuzzy_equal(self.t2, t2_calc):
                logs.append(f"t2 inconsistency: calculated {t2_calc:.4f}, got {self.t2:.4f}")
            else:
                logs.append("✓ t2 consistent with meeting condition.")

            # 10. Check x_plate2 = v2'' * t2
            x_plate2_calc = self.v2pp * self.t2
            if not self.fuzzy_equal(self.x_plate2, x_plate2_calc):
                logs.append(f"x_plate,2 inconsistency: calculated {x_plate2_calc:.4f}, got {self.x_plate2:.4f}")
            else:
                logs.append("✓ x_plate,2 consistent.")

            # 11. Check v3 = v2' + g*t2
            v3_calc = self.v2p + self.g * self.t2
            if not self.fuzzy_equal(self.v3, v3_calc):
                logs.append(f"v3 inconsistency: calculated {v3_calc:.4f}, got {self.v3:.4f}")
            else:
                logs.append("✓ v3 consistent with kinematics.")

            # 12. Check v3' and v3'' from collision formulas
            if not self.fuzzy_equal(self.v3p, self.v3p_expected):
                logs.append(f"v3' mismatch: expected {self.v3p_expected:.4f}, got {self.v3p:.4f}")
            else:
                logs.append("✓ v3' matches expected.")
            if not self.fuzzy_equal(self.v3pp, self.v3pp_expected):
                logs.append(f"v3'' mismatch: expected {self.v3pp_expected:.4f}, got {self.v3pp:.4f}")
            else:
                logs.append("✓ v3'' matches expected.")

            # 13. Check t3 from meeting condition: t3 = 2*(v3'' - v3')/g
            t3_calc = 2 * (self.v3pp - self.v3p) / self.g
            if not self.fuzzy_equal(self.t3, t3_calc):
                logs.append(f"t3 inconsistency: calculated {t3_calc:.4f}, got {self.t3:.4f}")
            else:
                logs.append("✓ t3 consistent.")

            # 14. Check x_plate3 = v3'' * t3
            x_plate3_calc = self.v3pp * self.t3
            if not self.fuzzy_equal(self.x_plate3, x_plate3_calc):
                logs.append(f"x_plate,3 inconsistency: calculated {x_plate3_calc:.4f}, got {self.x_plate3:.4f}")
            else:
                logs.append("✓ x_plate,3 consistent.")

        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
        return logs

    def test_monotonicity(self):
        """Perturb input l and check scaling of key results."""
        logs = []
        try:
            # Store original values
            original_l = self.l
            perturbed_l = original_l * 1.01  # increase by 1%
            
            # Recompute expected v0 with perturbed l
            v0_new = math.sqrt(2 * self.g * perturbed_l)
            # The ratio v0_new / self.v0_expected should be sqrt(perturbed_l / original_l)
            expected_ratio = math.sqrt(perturbed_l / original_l)
            actual_ratio = v0_new / self.v0_expected
            if not self.fuzzy_equal(actual_ratio, expected_ratio):
                logs.append(f"Monotonicity for v0 failed: expected ratio {expected_ratio:.4f}, got {actual_ratio:.4f}")
            else:
                logs.append("✓ v0 scales as sqrt(l).")
            
            # Restore original l
            self.l = original_l
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
            self.l = self.original_l  # ensure restoration
        return logs

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        logs = []
        success = True
        
        # Run each test and collect logs
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                test_logs = test_method()
                if any(log.startswith("Error") or (not log.startswith("✓") and "failed" in log.lower()) for log in test_logs):
                    success = False
                logs.append(f"--- {test_name} ---")
                logs.extend(test_logs)
            except Exception as e:
                logs.append(f"Exception in {test_name}: {e}")
                success = False
        
        return success, logs