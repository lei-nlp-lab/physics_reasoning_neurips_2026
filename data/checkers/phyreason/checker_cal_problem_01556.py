import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # The solution_instance is the module where the code ran
        self.solution = solution_instance
        
        # Extract input parameters from the num_vals dictionary used in the code
        # Default to the same values used in the solving script
        self.U0 = getattr(solution_instance, 'U0', 1000.0)
        self.U = getattr(solution_instance, 'U', 200.0)
        self.d = getattr(solution_instance, 'd', 0.01)
        self.e = getattr(solution_instance, 'e', 1.602e-19)
        self.m = getattr(solution_instance, 'm', 9.109e-31)
        
        # Extract results from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Ensure all required keys exist with defaults
        required_keys = ['y', 'E_k', 't', 'a', 'l', 'v_0']
        for key in required_keys:
            if key not in self.results:
                self.results[key] = 0.0
        
        self.logs = []
    
    def test_invariants(self):
        """Verify Energy Conservation, Newton's 2nd Law, and Kinematics"""
        logs = []
        try:
            # 1. Work-Energy Theorem for initial acceleration (Step 1)
            if self.m != 0:
                v0_from_energy = math.sqrt(2 * self.e * self.U0 / self.m)
            else:
                v0_from_energy = 0.0
                
            if not math.isclose(self.results['v_0'], v0_from_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_0 inconsistent with energy conservation. "
                           f"Got {self.results['v_0']:.3e}, expected {v0_from_energy:.3e}")
            else:
                logs.append("PASS: Work-Energy Theorem holds for initial acceleration")
            
            # 2. Newton's 2nd Law in plates (Step 3)
            if self.m != 0 and self.d != 0:
                a_from_force = self.e * self.U / (self.m * self.d)
            else:
                a_from_force = 0.0
                
            if not math.isclose(self.results['a'], a_from_force, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration inconsistent with Newton's 2nd Law. "
                           f"Got {self.results['a']:.3e}, expected {a_from_force:.3e}")
            else:
                logs.append("PASS: Newton's 2nd Law holds in deflection region")
            
            # 3. Kinematics displacement (Step 4)
            y_from_kinematics = 0.5 * self.results['a'] * self.results['t']**2
            if not math.isclose(self.results['y'], y_from_kinematics, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Displacement y inconsistent with kinematics. "
                           f"Got {self.results['y']:.3e}, expected {y_from_kinematics:.3e}")
            else:
                logs.append("PASS: Kinematics equation holds for deflection")
            
            # 4. Deflection condition y = d/2
            if not math.isclose(self.results['y'], self.d/2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Deflection y should be d/2. "
                           f"Got {self.results['y']:.3e}, expected {self.d/2:.3e}")
            else:
                logs.append("PASS: Electron grazes plate (y = d/2)")
            
            # 5. Time definition (Step 2)
            if self.results['v_0'] != 0:
                t_from_velocity = self.results['l'] / self.results['v_0']
            else:
                t_from_velocity = 0.0
                
            if not math.isclose(self.results['t'], t_from_velocity, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Time t inconsistent with l/v0. "
                           f"Got {self.results['t']:.3e}, expected {t_from_velocity:.3e}")
            else:
                logs.append("PASS: Time definition t = l/v0 holds")
            
            # 6. Total energy conservation (Step 6)
            E_k_expected = self.e * (self.U0 + self.U/2)
            if not math.isclose(self.results['E_k'], E_k_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final kinetic energy inconsistent with energy conservation. "
                           f"Got {self.results['E_k']:.3e}, expected {E_k_expected:.3e}")
            else:
                logs.append("PASS: Total energy conservation holds")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        logs = []
        try:
            # 1. Verify l from kinematic chain
            if self.U != 0:
                l_calculated = self.d * math.sqrt(2 * self.U0 / self.U)
            else:
                l_calculated = 0.0
            
            if not math.isclose(self.results['l'], l_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Plate length l inconsistent with derived formula. "
                           f"Got {self.results['l']:.3e}, expected {l_calculated:.3e}")
            else:
                logs.append("PASS: Plate length l consistent with kinematic chain")
            
            # 2. Verify all steps lead to same final E_k
            E_k_alt = self.e * self.U0 + 0.5 * self.e * self.U
            
            if not math.isclose(self.results['E_k'], E_k_alt, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_k inconsistent via alternative calculation. "
                           f"Got {self.results['E_k']:.3e}, expected {E_k_alt:.3e}")
            else:
                logs.append("PASS: Final E_k consistent via multiple paths")
            
            # 3. Check that t, a, v0 satisfy all relationships simultaneously
            if self.U != 0:
                l_from_chain = self.d * self.results['v_0'] * math.sqrt(self.m/(self.e * self.U))
            else:
                l_from_chain = 0.0
            
            if not math.isclose(self.results['l'], l_from_chain, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Chain t→a→y→l inconsistent. "
                           f"Got l={self.results['l']:.3e}, expected {l_from_chain:.3e}")
            else:
                logs.append("PASS: All intermediate results consistent with final l")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends"""
        logs = []
        original_U0 = self.U0
        original_U = self.U
        
        try:
            # Test 1: Increase U0, l should increase, E_k should increase
            perturb_factor = 1.1
            self.U0 *= perturb_factor
            
            # Recalculate using formulas
            if original_U != 0:
                l_new = self.d * math.sqrt(2 * self.U0 / original_U)
            else:
                l_new = 0.0
                
            E_k_new = self.e * (self.U0 + original_U/2)
            
            if l_new > self.results['l'] and E_k_new > self.results['E_k']:
                logs.append("PASS: Increasing U0 increases both l and E_k (monotonic)")
            else:
                logs.append(f"FAIL: Expected l and E_k to increase with U0. "
                           f"l: {self.results['l']:.3e}→{l_new:.3e}, "
                           f"E_k: {self.results['E_k']:.3e}→{E_k_new:.3e}")
            
            # Restore U0 and perturb U
            self.U0 = original_U0
            self.U *= perturb_factor
            
            # Recalculate
            if self.U != 0:
                l_new = self.d * math.sqrt(2 * self.U0 / self.U)
            else:
                l_new = 0.0
                
            E_k_new = self.e * (self.U0 + self.U/2)
            
            # Increasing U should decrease l but increase E_k
            if l_new < self.results['l'] and E_k_new > self.results['E_k']:
                logs.append("PASS: Increasing U decreases l but increases E_k (monotonic)")
            else:
                logs.append(f"FAIL: Expected l to decrease and E_k to increase with U. "
                           f"l: {self.results['l']:.3e}→{l_new:.3e}, "
                           f"E_k: {self.results['E_k']:.3e}→{E_k_new:.3e}")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        finally:
            # Restore original values
            self.U0 = original_U0
            self.U = original_U
        
        self.logs.extend(logs)
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def run_all_tests(self):
        """Run all tests and return summary"""
        all_success = True
        all_logs = []
        
        # Run invariants test
        success, logs = self.test_invariants()
        all_success = all_success and success
        all_logs.extend(["=== Invariants Test ==="] + logs)
        
        # Run trace consistency test
        success, logs = self.test_trace_consistency()
        all_success = all_success and success
        all_logs.extend(["\n=== Trace Consistency Test ==="] + logs)
        
        # Run monotonicity test
        success, logs = self.test_monotonicity()
        all_success = all_success and success
        all_logs.extend(["\n=== Monotonicity Test ==="] + logs)
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall result: {'PASS' if all_success else 'FAIL'}")
        
        return all_success, all_logs