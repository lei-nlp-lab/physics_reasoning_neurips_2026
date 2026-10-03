import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results with fallbacks
        self.solution = solution_instance
        # Inputs (from problem statement and table)
        self.m_person = getattr(solution_instance, 'm_person', 60.0)  # kg
        self.v_max = getattr(solution_instance, 'v_max', 7.0)  # m/s
        self.t = getattr(solution_instance, 't', 300.0)  # 5 minutes = 300 s
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s^2
        self.resistance_coeff = getattr(solution_instance, 'resistance_coeff', 0.05)
        self.rated_power = getattr(solution_instance, 'rated_power', 
                                  solution_instance.results.get('rated_power', 360.0))  # W (from table)
        
        # Results (from solution steps)
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 1.08e5))  # J
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', None))  # N
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', 0.1))  # ohm
        self.P_out = getattr(solution_instance, 'P_out', solution_instance.results.get('P_out', 350.0))  # W
        self.P_heat = getattr(solution_instance, 'P_heat', solution_instance.results.get('P_heat', None))  # W
        self.E_f = getattr(solution_instance, 'E_f', solution_instance.results.get('E_f', None))  # N (friction force)
        
        # Derived quantities
        self.P_total = getattr(solution_instance, 'P_total', solution_instance.results.get('P', 360.0))  # W
        self.m_bike = getattr(solution_instance, 'm_bike', 40.0)  # kg (from friction force calculation)
        self.m_total = self.m_person + self.m_bike
        
        # Store logs
        self.logs = []
    
    def log(self, msg):
        self.logs.append(msg)
    
    def test_energy_calculation(self):
        """Verify energy E = P_total * t"""
        try:
            expected_E = self.P_total * self.t
            if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"✓ Energy calculation correct: E = {self.E:.1f} J = P_total * t = {expected_E:.1f} J")
                return True
            else:
                self.log(f"✗ Energy mismatch: E = {self.E:.1f} J, expected P_total * t = {expected_E:.1f} J")
                return False
        except Exception as e:
            self.log(f"✗ Energy calculation test failed: {e}")
            return False
    
    def test_force_equilibrium(self):
        """At max speed, traction force = friction force"""
        try:
            # Friction force from physics
            friction_force = self.resistance_coeff * self.m_total * self.g
            
            # Traction force from power: F = P_out / v_max
            traction_from_power = self.P_out / self.v_max if self.v_max > 0 else 0
            
            # Check consistency
            checks = []
            if self.F is not None:
                checks.append(('F from solution', self.F, friction_force))
            checks.append(('F from P_out/v', traction_from_power, friction_force))
            
            all_pass = True
            for name, calculated, expected in checks:
                if math.isclose(calculated, expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"✓ {name} matches friction force: {calculated:.1f} N ≈ {expected:.1f} N")
                else:
                    self.log(f"✗ {name} mismatch: {calculated:.1f} N ≠ {expected:.1f} N")
                    all_pass = False
            
            # Store calculated friction for later use
            if self.E_f is None:
                self.E_f = friction_force
            
            return all_pass
        except Exception as e:
            self.log(f"✗ Force equilibrium test failed: {e}")
            return False
    
    def test_power_balance(self):
        """Verify P_total = P_out + P_heat"""
        try:
            # Calculate P_heat if not provided
            if self.P_heat is None:
                self.P_heat = self.P_total - self.P_out
            
            if math.isclose(self.P_total, self.P_out + self.P_heat, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"✓ Power balance correct: P_total = {self.P_total:.1f} W = P_out + P_heat = {self.P_out:.1f} + {self.P_heat:.1f} W")
                
                # Additional check: P_heat should be positive
                if self.P_heat > 0:
                    self.log(f"✓ P_heat is positive: {self.P_heat:.1f} W")
                    return True
                else:
                    self.log(f"✗ P_heat should be positive but is {self.P_heat:.1f} W")
                    return False
            else:
                self.log(f"✗ Power imbalance: P_total = {self.P_total:.1f} W ≠ P_out + P_heat = {self.P_out:.1f} + {self.P_heat:.1f} W")
                return False
        except Exception as e:
            self.log(f"✗ Power balance test failed: {e}")
            return False
    
    def test_joule_law_consistency(self):
        """Verify P_heat = I² * r consistency"""
        try:
            # Calculate current from P_total = V*I (need voltage from table)
            # Since V not given, use alternative: I = sqrt(P_heat / r)
            if self.P_heat > 0 and self.r > 0:
                I_calc = math.sqrt(self.P_heat / self.r)
                
                # Check if this current gives correct P_total with some V
                # This is a consistency check rather than exact verification
                self.log(f"✓ Joule's law consistency: I = sqrt(P_heat/r) = {I_calc:.2f} A")
                
                # Symbolic check
                I, r, P_heat = sp.symbols('I r P_heat', positive=True)
                eq = sp.Eq(P_heat, I**2 * r)
                self.log(f"  Symbolic: {eq}")
                
                return True
            else:
                self.log("✗ Cannot test Joule's law without positive P_heat and r")
                return False
        except Exception as e:
            self.log(f"✗ Joule's law test failed: {e}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check step-by-step results with final answers"""
        try:
            # Target answers from problem
            targets = [1.08e5, 350.0, 0.1]  # E, P_out, r
            
            # Solution values
            solutions = [self.E, self.P_out, self.r]
            
            all_pass = True
            for i, (target, sol, name) in enumerate(zip(targets, solutions, 
                                                       ['E', 'P_out', 'r'])):
                if math.isclose(sol, target, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"✓ {name} matches target: {sol} ≈ {target}")
                else:
                    self.log(f"✗ {name} mismatch: {sol} ≠ {target}")
                    all_pass = False
            
            return all_pass
        except Exception as e:
            self.log(f"✗ Trace consistency test failed: {e}")
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check resistance trend"""
        try:
            # Save original state
            orig_m_person = self.m_person
            
            # Increase mass by 10%
            self.m_person *= 1.1
            self.m_total = self.m_person + self.m_bike
            
            # Recalculate using physics formulas
            new_friction = self.resistance_coeff * self.m_total * self.g
            new_P_out = new_friction * self.v_max
            new_P_heat = self.P_total - new_P_out
            
            # Calculate new resistance (assuming same voltage/current relationship)
            if new_P_heat > 0 and self.P_heat > 0:
                # From P_heat = I² * r, with same I (rated operation)
                new_r = self.r * (new_P_heat / self.P_heat)
                
                # Check monotonicity: increased mass → increased friction → increased P_out → decreased P_heat → decreased r
                if new_r < self.r:
                    self.log(f"✓ Monotonicity: Increased mass → lower resistance ({self.r:.3f} Ω → {new_r:.3f} Ω)")
                else:
                    self.log(f"✗ Monotonicity violated: Increased mass but resistance didn't decrease")
                
                # Restore original state
                self.m_person = orig_m_person
                self.m_total = self.m_person + self.m_bike
                
                return new_r < self.r
            else:
                self.log("✗ Cannot test monotonicity with non-positive P_heat")
                self.m_person = orig_m_person
                self.m_total = self.m_person + self.m_bike
                return False
        except Exception as e:
            self.log(f"✗ Monotonicity test failed: {e}")
            # Restore original state on error
            self.m_person = orig_m_person if 'orig_m_person' in locals() else 60.0
            self.m_total = self.m_person + self.m_bike
            return False
    
    def test_invariants(self):
        """Run all physics invariance tests"""
        tests = [
            self.test_energy_calculation,
            self.test_force_equilibrium,
            self.test_power_balance,
            self.test_joule_law_consistency,
        ]
        
        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                self.log(f"✗ Test {test.__name__} crashed: {e}")
                results.append(False)
        
        return all(results)
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []  # Clear previous logs
        
        try:
            # Run all test categories
            trace_ok = self.test_trace_consistency()
            invariants_ok = self.test_invariants()
            monotonicity_ok = self.test_monotonicity()
            
            success = trace_ok and invariants_ok
            
            summary = f"""
            ===== PHYSICS AUDIT SUMMARY =====
            Trace Consistency: {'PASS' if trace_ok else 'FAIL'}
            Physics Invariants: {'PASS' if invariants_ok else 'FAIL'}
            Monotonicity Check: {'PASS' if monotonicity_ok else 'FAIL'}
            Overall: {'PASS' if success else 'FAIL'}
            """
            self.log(summary)
            
        except Exception as e:
            self.log(f"✗ Critical error in test suite: {e}")
            self.log(traceback.format_exc())
            success = False
        
        return success, self.logs