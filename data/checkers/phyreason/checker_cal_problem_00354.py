import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 6.0))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', 2.0))
        self.R1 = getattr(solution_instance, 'R1', solution_instance.results.get('R1', 2.0))
        self.R2 = getattr(solution_instance, 'R2', solution_instance.results.get('R2', 3.0))
        self.R3 = getattr(solution_instance, 'R3', solution_instance.results.get('R3', 6.0))
        self.C = getattr(solution_instance, 'C', solution_instance.results.get('C', 10e-6))  # 10 μF
        
        # Safe extraction of results using key mapping
        self.R_total = getattr(solution_instance, 'R_total', 
                               solution_instance.results.get('R_texttotal', 
                               solution_instance.results.get('R_{\\text{total}}', 0.0)))
        self.I = getattr(solution_instance, 'I', solution_instance.results.get('I', 0.0))
        self.U = getattr(solution_instance, 'U', solution_instance.results.get('U', 0.0))
        self.U1 = getattr(solution_instance, 'U1', solution_instance.results.get('U_1', 0.0))
        self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', 0.0))
        
        self.logs = []
        self.success = True
        
    def test_invariants(self):
        """Verify Ohm's Law and circuit conservation principles"""
        try:
            # 1. Check equivalent resistance calculation
            R_parallel = (self.R2 * self.R3) / (self.R2 + self.R3)
            expected_R_total = self.R1 + R_parallel
            if not math.isclose(self.R_total, expected_R_total, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"Resistance invariant FAIL: R_total={self.R_total:.3f}, expected={expected_R_total:.3f}")
                self.success = False
            
            # 2. Check Ohm's Law for closed circuit
            expected_I = self.E / (self.R_total + self.r)
            if not math.isclose(self.I, expected_I, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"Current invariant FAIL: I={self.I:.3f}, expected={expected_I:.3f}")
                self.success = False
            
            # 3. Check terminal voltage from two perspectives
            # From EMF and internal resistance
            expected_U1 = self.E - self.I * self.r
            # From external circuit
            expected_U2 = self.I * self.R_total
            if not (math.isclose(self.U, expected_U1, rel_tol=0.02, abs_tol=0.02) and 
                    math.isclose(self.U, expected_U2, rel_tol=0.02, abs_tol=0.02)):
                self.logs.append(f"Voltage invariant FAIL: U={self.U:.3f}, expected from E-r={expected_U1:.3f}, from I*R={expected_U2:.3f}")
                self.success = False
            
            # 4. Check voltage across R1
            expected_U1_voltage = self.I * self.R1
            if not math.isclose(self.U1, expected_U1_voltage, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"U1 invariant FAIL: U1={self.U1:.3f}, expected={expected_U1_voltage:.3f}")
                self.success = False
            
            # 5. Check capacitor charge
            expected_Q = self.C * self.U1
            if not math.isclose(self.Q, expected_Q, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"Charge invariant FAIL: Q={self.Q:.3e}, expected={expected_Q:.3e}")
                self.success = False
                
            # 6. Symbolic check as fallback
            try:
                I_sym, R_sym, r_sym, E_sym, U_sym = sp.symbols('I R r E U')
                eq1 = sp.Eq(U_sym, E_sym - I_sym * r_sym)
                eq2 = sp.Eq(U_sym, I_sym * R_sym)
                # Substitute values
                subs_dict = {E_sym: self.E, r_sym: self.r, I_sym: self.I, R_sym: self.R_total, U_sym: self.U}
                if not (eq1.subs(subs_dict).simplify() == True and eq2.subs(subs_dict).simplify() == True):
                    self.logs.append("Symbolic check raised inconsistency")
                    self.success = False
            except Exception as e:
                self.logs.append(f"Symbolic check error (non-critical): {str(e)}")
            
            if self.success:
                self.logs.append("All invariants PASS")
                
        except Exception as e:
            self.logs.append(f"Invariant test error: {str(e)}")
            self.success = False
    
    def test_trace_consistency(self):
        """Check intermediate results align with final answer"""
        try:
            # Recompute steps and compare
            # Step 1: R_parallel
            R_parallel = (self.R2 * self.R3) / (self.R2 + self.R3)
            expected_R_total_step = self.R1 + R_parallel
            
            # Step 2: Current
            expected_I_step = self.E / (expected_R_total_step + self.r)
            
            # Step 3: Terminal voltage
            expected_U_step = self.E - expected_I_step * self.r
            
            # Step 4: Voltage across R1
            expected_U1_step = expected_I_step * self.R1
            
            # Step 5: Charge
            expected_Q_step = self.C * expected_U1_step
            
            # Compare with results (allow 2% tolerance)
            checks = [
                ("R_total", self.R_total, expected_R_total_step),
                ("I", self.I, expected_I_step),
                ("U", self.U, expected_U_step),
                ("U1", self.U1, expected_U1_step),
                ("Q", self.Q, expected_Q_step)
            ]
            
            all_pass = True
            for name, actual, expected in checks:
                if not math.isclose(actual, expected, rel_tol=0.02, abs_tol=0.02):
                    self.logs.append(f"Trace {name}: {actual:.3f} vs expected {expected:.3f}")
                    all_pass = False
            
            if all_pass:
                self.logs.append("Trace consistency PASS")
            else:
                self.success = False
                
        except Exception as e:
            self.logs.append(f"Trace consistency error: {str(e)}")
            self.success = False
    
    def test_monotonicity(self):
        """Perturb R1 and verify expected trends"""
        original_R1 = self.R1
        try:
            # Store original results
            original_results = {
                'I': self.I,
                'U': self.U,
                'U1': self.U1,
                'Q': self.Q
            }
            
            # Perturb R1 by +10%
            perturbed_R1 = original_R1 * 1.1
            
            # Recompute everything with perturbed R1
            R_parallel = (self.R2 * self.R3) / (self.R2 + self.R3)
            perturbed_R_total = perturbed_R1 + R_parallel
            perturbed_I = self.E / (perturbed_R_total + self.r)
            perturbed_U = self.E - perturbed_I * self.r
            perturbed_U1 = perturbed_I * perturbed_R1
            perturbed_Q = self.C * perturbed_U1
            
            # Check trends (qualitative)
            # When R1 increases:
            # 1. Total resistance increases → current should decrease
            # 2. Terminal voltage should increase (U = E - I*r, I decreases)
            # 3. U1 = I*R1: I decreases, R1 increases → effect ambiguous, needs calculation
            # 4. Q = C*U1 → follows U1 trend
            
            trend_checks = []
            
            # Current should decrease
            if perturbed_I < original_results['I']:
                trend_checks.append(("I decreases", True))
            else:
                trend_checks.append(("I should decrease", False))
            
            # Terminal voltage should increase
            if perturbed_U > original_results['U']:
                trend_checks.append(("U increases", True))
            else:
                trend_checks.append(("U should increase", False))
            
            # U1 trend: compute derivative sign
            # dU1/dR1 = d(I*R1)/dR1 = R1*dI/dR1 + I
            # From circuit: I = E/(R1 + R_parallel + r)
            # dI/dR1 = -E/(R1 + R_parallel + r)^2
            # So dU1/dR1 = -E*R1/(R1 + R_parallel + r)^2 + E/(R1 + R_parallel + r)
            #            = E*(R1 + R_parallel + r - R1)/(R1 + R_parallel + r)^2
            #            = E*(R_parallel + r)/(R1 + R_parallel + r)^2 > 0
            # Therefore U1 should increase with R1
            if perturbed_U1 > original_results['U1']:
                trend_checks.append(("U1 increases", True))
            else:
                trend_checks.append(("U1 should increase", False))
            
            # Q follows U1
            if perturbed_Q > original_results['Q']:
                trend_checks.append(("Q increases", True))
            else:
                trend_checks.append(("Q should increase", False))
            
            # Evaluate all trends
            all_trends_pass = all(check[1] for check in trend_checks)
            
            if all_trends_pass:
                self.logs.append("Monotonicity test PASS")
            else:
                failed = [name for name, passed in trend_checks if not passed]
                self.logs.append(f"Monotonicity FAIL: {failed}")
                self.success = False
                
            # Restore original R1
            self.R1 = original_R1
            
        except Exception as e:
            self.logs.append(f"Monotonicity test error: {str(e)}")
            # Restore on error
            self.R1 = original_R1
            self.success = False
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)"""
        self.logs = []
        self.success = True
        
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
        except Exception as e:
            self.logs.append(f"Critical test error: {str(e)}\n{traceback.format_exc()}")
            self.success = False
        
        return self.success, self.logs