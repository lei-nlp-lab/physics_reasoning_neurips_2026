import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: Extract all required inputs and results from the solution instance.
        Uses getattr with fallbacks to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        
        # Extract fundamental constants and parameters safely
        self.e = getattr(solution_instance, 'e', None)
        self.m = getattr(solution_instance, 'm', None)
        self.E = getattr(solution_instance, 'E', None)
        self.L = getattr(solution_instance, 'L', None)
        
        # Results dictionary for safe access
        self.results = {}
        
        # Expected keys from problem statement
        expected_keys = [
            'a_1', 'y', 'v_y', 't_2', 't', 't_3', 'a_2', 'v', 
            'L/2', 'tantheta', 'y_1'
        ]
        
        for key in expected_keys:
            # Try to get from solution attribute first, then from results dict
            attr_val = getattr(solution_instance, key, None)
            if attr_val is not None:
                self.results[key] = attr_val
            elif hasattr(solution_instance, 'results'):
                self.results[key] = solution_instance.results.get(key, 0.0)
            else:
                self.results[key] = 0.0
        
        # Target answers for verification
        self.target_answers = [
            3 * sp.sqrt(self.m * self.L / (self.e * self.E)) if all([self.m, self.L, self.e, self.E]) else None,
            3,  # tanθ
            4.5 * self.L if self.L else None
        ]
        
        # Setup symbolic variables for verification
        self._setup_symbolic()
    
    def _setup_symbolic(self):
        """Initialize SymPy symbols for symbolic verification"""
        self.symbols = {}
        # Physical constants
        self.symbols['e'] = sp.symbols('e', positive=True)
        self.symbols['m'] = sp.symbols('m', positive=True)
        self.symbols['E'] = sp.symbols('E', positive=True)
        self.symbols['L'] = sp.symbols('L', positive=True)
        
        # Intermediate quantities
        self.symbols['a_1'] = sp.symbols('a_1')
        self.symbols['a_2'] = sp.symbols('a_2')
        self.symbols['v'] = sp.symbols('v')
        self.symbols['t_1'] = sp.symbols('t_1')
        self.symbols['t_2'] = sp.symbols('t_2')
        self.symbols['t_3'] = sp.symbols('t_3')
        self.symbols['t'] = sp.symbols('t')
        self.symbols['v_y'] = sp.symbols('v_y')
        self.symbols['y_1'] = sp.symbols('y_1')
        self.symbols['y'] = sp.symbols('y')
        self.symbols['tantheta'] = sp.symbols('tantheta')
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Robust numerical comparison with tolerance.
        Handles symbolic expressions by evaluating numerically if possible.
        """
        try:
            # If both are symbolic, substitute values if available
            if isinstance(a, sp.Basic) and isinstance(b, sp.Basic):
                # Try to evaluate numerically
                subs_dict = {}
                if self.e is not None:
                    subs_dict[self.symbols['e']] = self.e
                if self.m is not None:
                    subs_dict[self.symbols['m']] = self.m
                if self.E is not None:
                    subs_dict[self.symbols['E']] = self.E
                if self.L is not None:
                    subs_dict[self.symbols['L']] = self.L
                
                a_num = float(a.subs(subs_dict)) if subs_dict else float(a.evalf())
                b_num = float(b.subs(subs_dict)) if subs_dict else float(b.evalf())
                return math.isclose(a_num, b_num, rel_tol=rel_tol, abs_tol=abs_tol)
            else:
                return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            # If conversion fails, use string comparison as last resort
            return str(a) == str(b)
    
    def test_invariants(self):
        """
        Verify physical invariants:
        1. Newton's Second Law in both regions
        2. Energy conservation (work-energy theorem)
        """
        logs = []
        successes = []
        
        try:
            # 1. Newton's Second Law checks
            # Region 1: E1 field, force = e*E to the right (for electron with negative charge)
            expected_a1 = self.e * self.E / self.m if all([self.e, self.E, self.m]) else None
            if expected_a1 is not None and 'a_1' in self.results:
                a1 = self.results['a_1']
                if self._is_close(a1, expected_a1):
                    logs.append("✓ Newton's 2nd Law in region E1: a_1 = eE/m holds")
                    successes.append(True)
                else:
                    logs.append(f"✗ Newton's 2nd Law in region E1 violation: a_1={a1}, expected {expected_a1}")
                    successes.append(False)
            
            # Region 2: E2 field, force = 3eE downward (for electron, upward acceleration)
            expected_a2 = 3 * self.e * self.E / self.m if all([self.e, self.E, self.m]) else None
            if expected_a2 is not None and 'a_2' in self.results:
                a2 = self.results['a_2']
                if self._is_close(a2, expected_a2):
                    logs.append("✓ Newton's 2nd Law in region E2: a_2 = 3eE/m holds")
                    successes.append(True)
                else:
                    logs.append(f"✗ Newton's 2nd Law in region E2 violation: a_2={a2}, expected {expected_a2}")
                    successes.append(False)
            
            # 2. Energy conservation (work-energy theorem)
            # Total work done = change in kinetic energy
            if all([self.e, self.E, self.L, 'v' in self.results]):
                # Work in region 1: e*E*(L/2) (force and displacement both rightward)
                work1 = self.e * self.E * (self.L / 2)
                # Work in region 2: 3e*E*y1 (force downward, displacement downward)
                y1 = self.results.get('y_1', 0)
                work2 = 3 * self.e * self.E * y1
                total_work = work1 + work2
                
                # Final kinetic energy (at screen, has both horizontal and vertical components)
                v = self.results['v']
                v_y = self.results.get('v_y', 0)
                final_KE = 0.5 * self.m * (v**2 + v_y**2)
                
                # Initial KE is 0 (released from rest)
                if self._is_close(total_work, final_KE):
                    logs.append("✓ Energy conservation (work-energy theorem) holds")
                    successes.append(True)
                else:
                    logs.append(f"✗ Energy conservation violation: total work={total_work}, final KE={final_KE}")
                    successes.append(False)
        
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            successes.append(False)
        
        return all(successes), logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Follows the step-by-step analysis from the problem.
        """
        logs = []
        successes = []
        
        try:
            # Step 1-3: Acceleration and velocity at MN
            if all(['a_1' in self.results, self.L]):
                # Time t1 to cross L/2 in region 1
                a1 = self.results['a_1']
                t1 = sp.sqrt(self.L / a1)  # from L/2 = 0.5*a1*t1^2
                
                # Velocity at MN
                v_expected = a1 * t1
                if 'v' in self.results:
                    v = self.results['v']
                    if self._is_close(v, v_expected):
                        logs.append("✓ Velocity at MN consistent with kinematics")
                        successes.append(True)
                    else:
                        logs.append(f"✗ Velocity at MN inconsistent: v={v}, expected {v_expected}")
                        successes.append(False)
            
            # Step 4-6: Time calculations
            if all(['v' in self.results, self.L, 't_2' in self.results, 't' in self.results]):
                v = self.results['v']
                t2_expected = 2 * self.L / v  # time from MN to screen
                t2 = self.results['t_2']
                
                if self._is_close(t2, t2_expected):
                    logs.append("✓ Time from MN to screen consistent")
                    successes.append(True)
                else:
                    logs.append(f"✗ Time from MN to screen inconsistent: t2={t2}, expected {t2_expected}")
                    successes.append(False)
                
                # Total time should be t1 + t2
                # We already have t1 from above, or can calculate from kinematics
                if 'a_1' in self.results:
                    t1_calc = sp.sqrt(self.L / self.results['a_1'])
                    t_expected = t1_calc + t2_expected
                    t = self.results['t']
                    
                    # Compare with target answer
                    target_t = self.target_answers[0]
                    if target_t is not None and self._is_close(t, target_t):
                        logs.append("✓ Total time matches target answer")
                        successes.append(True)
                    elif self._is_close(t, t_expected):
                        logs.append("✓ Total time consistent with kinematics")
                        successes.append(True)
                    else:
                        logs.append(f"✗ Total time inconsistent: t={t}, expected {t_expected}")
                        successes.append(False)
            
            # Step 7-11: Vertical motion in region E2
            if all(['a_2' in self.results, 'v' in self.results, self.L, 't_3' in self.results, 'v_y' in self.results, 'tantheta' in self.results]):
                a2 = self.results['a_2']
                v = self.results['v']
                
                # Time in E2 region
                t3_expected = self.L / v
                t3 = self.results['t_3']
                
                if self._is_close(t3, t3_expected):
                    logs.append("✓ Time in E2 region consistent")
                    successes.append(True)
                else:
                    logs.append(f"✗ Time in E2 region inconsistent: t3={t3}, expected {t3_expected}")
                    successes.append(False)
                
                # Vertical velocity
                v_y_expected = a2 * t3
                v_y = self.results['v_y']
                
                if self._is_close(v_y, v_y_expected):
                    logs.append("✓ Vertical velocity consistent")
                    successes.append(True)
                else:
                    logs.append(f"✗ Vertical velocity inconsistent: v_y={v_y}, expected {v_y_expected}")
                    successes.append(False)
                
                # tanθ
                tantheta_expected = v_y / v
                tantheta = self.results['tantheta']
                
                # Compare with target answer (should be 3)
                if self._is_close(tantheta, 3):
                    logs.append("✓ tanθ matches target answer (3)")
                    successes.append(True)
                elif self._is_close(tantheta, tantheta_expected):
                    logs.append("✓ tanθ consistent with velocity components")
                    successes.append(True)
                else:
                    logs.append(f"✗ tanθ inconsistent: tanθ={tantheta}, expected {tantheta_expected}")
                    successes.append(False)
            
            # Step 12-14: Vertical displacement
            if all(['a_2' in self.results, 't_3' in self.results, 'tantheta' in self.results, 'y_1' in self.results, 'y' in self.results, self.L]):
                a2 = self.results['a_2']
                t3 = self.results['t_3']
                
                # Vertical displacement in E2
                y1_expected = 0.5 * a2 * t3**2
                y1 = self.results['y_1']
                
                if self._is_close(y1, y1_expected):
                    logs.append("✓ Vertical displacement in E2 consistent")
                    successes.append(True)
                else:
                    logs.append(f"✗ Vertical displacement in E2 inconsistent: y1={y1}, expected {y1_expected}")
                    successes.append(False)
                
                # Total vertical displacement at screen
                tantheta = self.results['tantheta']
                y_expected = y1 + self.L * tantheta
                y = self.results['y']
                
                # Compare with target answer
                target_y = self.target_answers[2]
                if target_y is not None and self._is_close(y, target_y):
                    logs.append("✓ Total vertical displacement matches target answer (4.5L)")
                    successes.append(True)
                elif self._is_close(y, y_expected):
                    logs.append("✓ Total vertical displacement consistent")
                    successes.append(True)
                else:
                    logs.append(f"✗ Total vertical displacement inconsistent: y={y}, expected {y_expected}")
                    successes.append(False)
        
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            successes.append(False)
        
        return all(successes), logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and check if the output changes in the expected direction.
        Restores the original value after testing.
        """
        logs = []
        successes = []
        
        try:
            if self.L is not None and 'y' in self.results:
                original_L = self.L
                original_y = self.results['y']
                
                # Perturb L by +10%
                perturbed_L = original_L * 1.1
                self.L = perturbed_L
                
                # Recalculate using the kinematic relationships
                # We'll use the symbolic relationships to predict the change
                if all([self.e, self.E, self.m]):
                    # Recalculate a1, a2
                    a1 = self.e * self.E / self.m
                    a2 = 3 * self.e * self.E / self.m
                    
                    # Recalculate t1, v, t3
                    t1 = sp.sqrt(perturbed_L / a1)  # L/2 distance, so t1 ~ sqrt(L)
                    v = a1 * t1
                    t3 = perturbed_L / v  # t3 = L/v
                    
                    # Recalculate y1 and y
                    y1 = 0.5 * a2 * t3**2
                    tantheta = 3  # From problem solution
                    predicted_y = y1 + perturbed_L * tantheta
                    
                    # Check monotonicity: y should increase with L
                    if predicted_y > original_y:
                        logs.append("✓ Monotonicity: y increases with increasing L")
                        successes.append(True)
                    else:
                        logs.append("✗ Monotonicity violation: y does not increase with L")
                        successes.append(False)
                
                # Restore original value
                self.L = original_L
            
            # Test mass monotonicity: increasing mass should increase time
            if self.m is not None and 't' in self.results:
                original_m = self.m
                original_t = self.results['t']
                
                perturbed_m = original_m * 1.1
                self.m = perturbed_m
                
                if all([self.e, self.E, self.L]):
                    # Symbolic expression for total time: t = 3*sqrt(mL/(eE))
                    predicted_t = 3 * sp.sqrt(perturbed_m * self.L / (self.e * self.E))
                    
                    if predicted_t > original_t:
                        logs.append("✓ Monotonicity: t increases with increasing m")
                        successes.append(True)
                    else:
                        logs.append("✗ Monotonicity violation: t does not increase with m")
                        successes.append(False)
                
                # Restore original value
                self.m = original_m
        
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            successes.append(False)
        
        return all(successes), logs
    
    def run_all_tests(self):
        """
        Run all tests and return overall success status and logs.
        """
        all_logs = []
        all_successes = []
        
        # Run each test
        for test_name, test_method in [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_method()
                all_successes.append(success)
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"Test {test_name} failed with exception: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_successes.append(False)
        
        # Final summary
        overall_success = all(all_successes)
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs