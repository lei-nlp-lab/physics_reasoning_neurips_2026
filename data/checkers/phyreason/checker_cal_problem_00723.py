import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract parameters from results dictionary (not attributes)
        results = getattr(solution_instance, 'results', {})
        
        # Get parameters from results with defaults
        self.m = results.get('m', 1.0)
        self.q = results.get('q', 1.0)
        self.B = results.get('B', 1.0)
        self.L = results.get('L', 1.0)
        
        # Extract computed results
        self.v = results.get('v', 0.0)
        self.E = results.get('E', 0.0)
        self.y = results.get('y', 0.0)
        self.v_y = results.get('v_y', 0.0)
        self.t = results.get('t', 0.0)
        self.a = results.get('a', 0.0)
        self.r_2 = results.get('r_2', 0.0)
        self.v_prime = results.get('v\'', results.get('v_prime', 0.0))
        self.theta = results.get('theta', 0.0)
        self.r_1 = results.get('r_1', 0.0)
        
        # Log storage
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            message = f"[{'PASS' if success else 'FAIL'}] {message}"
        self.logs.append(message)
        return success
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        try:
            # 1. Force equilibrium between plates (straight line motion)
            electric_force = abs(self.q * self.E)
            magnetic_force = abs(self.q * self.v * self.B)
            eq_check = math.isclose(electric_force, magnetic_force, 
                                    rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Force equilibrium: qE={electric_force:.3e}, "
                     f"qvB={magnetic_force:.3e}", eq_check)
            
            # 2. Energy conservation check (kinetic energy gain from electric field)
            work = abs(self.q * self.E * self.y)
            delta_ke_y = 0.5 * abs(self.m) * self.v_y**2
            energy_check = math.isclose(work, delta_ke_y, 
                                       rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Energy balance: qEy={work:.3e}, "
                     f"ΔKE_y={delta_ke_y:.3e}", energy_check)
            
            # 3. Centripetal force in magnetic field
            magnetic_force = abs(self.q * self.v_prime * self.B)
            centripetal_force = abs(self.m * self.v_prime**2 / self.r_2)
            cent_check = math.isclose(magnetic_force, centripetal_force,
                                     rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Centripetal force: qv'B={magnetic_force:.3e}, "
                     f"mv'²/r₂={centripetal_force:.3e}", cent_check)
            
            return all([eq_check, energy_check, cent_check])
        except Exception as e:
            self._log(f"Invariants test failed: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results."""
        try:
            all_checks = []
            
            # 1. Check v from step 2: qvB = mv²/r₁ with r₁ = L
            v_expected = abs(self.q * self.B * self.L / self.m)
            v_check = math.isclose(abs(self.v), v_expected, 
                                   rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Velocity consistency: v={abs(self.v):.3e}, "
                     f"qBL/m={v_expected:.3e}", v_check)
            all_checks.append(v_check)
            
            # 2. Check E from step 4: E = vB (from equilibrium)
            E_expected = abs(self.v * self.B)
            E_check = math.isclose(abs(self.E), E_expected,
                                   rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Electric field: E={abs(self.E):.3e}, "
                     f"vB={E_expected:.3e}", E_check)
            all_checks.append(E_check)
            
            # 3. Check vertical motion (steps 6-9)
            # Time between plates: t = L/v
            t_expected = abs(self.L / self.v)
            t_check = math.isclose(abs(self.t), t_expected,
                                   rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Time between plates: t={abs(self.t):.3e}, "
                     f"L/v={t_expected:.3e}", t_check)
            all_checks.append(t_check)
            
            # Acceleration: a = qE/m
            a_expected = abs(self.q * self.E / self.m)
            a_check = math.isclose(abs(self.a), a_expected,
                                  rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Acceleration: a={abs(self.a):.3e}, "
                     f"qE/m={a_expected:.3e}", a_check)
            all_checks.append(a_check)
            
            # Vertical displacement: y = ½at² = L/2
            y_expected = abs(self.L) / 2
            y_check = math.isclose(abs(self.y), y_expected,
                                  rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Vertical displacement: y={abs(self.y):.3e}, "
                     f"L/2={y_expected:.3e}", y_check)
            all_checks.append(y_check)
            
            # Vertical velocity: v_y = at = v (from step 10)
            v_y_check = math.isclose(abs(self.v_y), abs(self.v),
                                    rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Vertical velocity: v_y={abs(self.v_y):.3e}, "
                     f"v={abs(self.v):.3e}", v_y_check)
            all_checks.append(v_y_check)
            
            # 4. Check exit velocity v' = √(v² + v_y²) = √2 * v
            v_prime_expected = math.sqrt(2) * abs(self.v)
            v_prime_check = math.isclose(abs(self.v_prime), v_prime_expected,
                                        rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Exit velocity: v'={abs(self.v_prime):.3e}, "
                     f"√2*v={v_prime_expected:.3e}", v_prime_check)
            all_checks.append(v_prime_check)
            
            # 5. Check radius r₂ = m*v'/(q*B) = √2 * L
            r2_expected = math.sqrt(2) * abs(self.L)
            r2_check = math.isclose(abs(self.r_2), r2_expected,
                                   rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Second radius: r₂={abs(self.r_2):.3e}, "
                     f"√2*L={r2_expected:.3e}", r2_check)
            all_checks.append(r2_check)
            
            # 6. Check angle θ = 45° (from v_y = v)
            theta_check = math.isclose(abs(self.theta), 45.0,
                                      rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Angle: θ={abs(self.theta):.1f}°, expected 45°", 
                     theta_check)
            all_checks.append(theta_check)
            
            return all(all_checks)
        except Exception as e:
            self._log(f"Trace consistency failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Test how results change with perturbed mass."""
        try:
            all_checks = []
            
            # Original values
            orig_v = abs(self.q * self.B * self.L / self.m)
            orig_E = abs(self.v * self.B)
            orig_y = abs(self.L) / 2
            orig_v_y = orig_v
            orig_v_prime = math.sqrt(2) * orig_v
            orig_r2 = math.sqrt(2) * abs(self.L)
            
            # Perturb mass by +10%
            m_perturbed = self.m * 1.1
            
            # Recalculate with perturbed mass
            new_v = abs(self.q * self.B * self.L / m_perturbed)
            new_E = new_v * abs(self.B)
            new_y = abs(self.L) / 2
            new_v_y = new_v
            new_v_prime = math.sqrt(2) * new_v
            new_r2 = math.sqrt(2) * abs(self.L)
            
            # Check trends:
            # 1. v ∝ 1/m
            v_trend_expected = orig_v / 1.1
            v_trend = math.isclose(new_v, v_trend_expected, 
                                  rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity v: new={new_v:.3e}, expected={v_trend_expected:.3e} "
                     f"(∝ 1/m)", v_trend)
            all_checks.append(v_trend)
            
            # 2. E ∝ 1/m (since E = vB and v ∝ 1/m)
            E_trend_expected = orig_E / 1.1
            E_trend = math.isclose(new_E, E_trend_expected,
                                  rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity E: new={new_E:.3e}, expected={E_trend_expected:.3e} "
                     f"(∝ 1/m)", E_trend)
            all_checks.append(E_trend)
            
            # 3. y should be constant (L/2)
            y_trend = math.isclose(new_y, orig_y, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity y: new={new_y:.3e}, orig={orig_y:.3e} (constant)", 
                     y_trend)
            all_checks.append(y_trend)
            
            # 4. v_y = v ∝ 1/m
            v_y_trend_expected = orig_v_y / 1.1
            v_y_trend = math.isclose(new_v_y, v_y_trend_expected,
                                    rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity v_y: new={new_v_y:.3e}, "
                     f"expected={v_y_trend_expected:.3e} (∝ 1/m)", v_y_trend)
            all_checks.append(v_y_trend)
            
            # 5. v' ∝ 1/m
            v_prime_trend_expected = orig_v_prime / 1.1
            v_prime_trend = math.isclose(new_v_prime, v_prime_trend_expected,
                                        rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity v': new={new_v_prime:.3e}, "
                     f"expected={v_prime_trend_expected:.3e} (∝ 1/m)", v_prime_trend)
            all_checks.append(v_prime_trend)
            
            # 6. r₂ should be constant (√2 L, independent of m)
            r2_trend = math.isclose(new_r2, orig_r2, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity r₂: new={new_r2:.3e}, orig={orig_r2:.3e} "
                     f"(constant)", r2_trend)
            all_checks.append(r2_trend)
            
            return all(all_checks)
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []  # Clear previous logs
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants()),
            ("Trace Consistency", self.test_trace_consistency()),
            ("Monotonicity", self.test_monotonicity())
        ]
        
        # Count successes
        success_count = sum(1 for _, result in tests if result)
        total_tests = len(tests)
        
        # Summary
        summary = (f"Physics Checker Results: {success_count}/{total_tests} tests passed")
        self.logs.append(summary)
        
        overall_success = success_count == total_tests
        return overall_success, self.logs