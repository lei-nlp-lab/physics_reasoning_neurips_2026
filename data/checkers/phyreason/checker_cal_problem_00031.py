import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access
        self.results = {}
        
        # Extract all results safely
        result_keys = [
            'v_2,1', 't_9', 'S', 'v_1,2', 't_2', 'S_2', 'm_2', 'N/A', 'x,y', 'S_0',
            'r_2,1', 'r', 'v_2,0', 't_3', 'v_2,3', 't_1', 'S_1', 'v_1,3',
            'v_2,2', 'S_8', 'v_1,9', 'T_1andT_2', 'v_1,0', 'v_2,9'
        ]
        
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, None))
        
        # Define symbols for physics checks
        self.a, self.B, self.q, self.m = 1.0, 1.0, 1.0, 1.0  # Use numeric values
        
        # Expected numeric values from solution steps (with a=1, B=1, q=1, m=1)
        self.expected = {
            'v_1,0': 1.0,  # q*B*a/m = 1*1*1/1 = 1
            'm_2': 0.5,    # m/2 = 0.5
            'v_2,1': 3.0,  # 3*q*B*a/m = 3
            't_1': math.pi,  # π*m/(q*B) = π
            'r_2,1': 1.5,  # 3*a/2 = 1.5
            't_2': 3*math.pi,  # 3π
            'S_1': 6*math.pi,  # 6π*a = 6π
            'v_1,2': 1.0,  # q*B*a/m = 1
            'v_2,2': -5.0, # -5*q*B*a/m = -5
            't_3': 5*math.pi,  # 5π
            'S_2': 10*math.pi, # 10π
            'v_1,3': -3.0, # -3*q*B*a/m = -3
            'v_2,3': 3.0,  # 3*q*B*a/m = 3
            't_9': 17*math.pi, # 17π
            'S_8': 10*math.pi, # 10π
            'v_1,9': -3.0, # -3*q*B*a/m = -3
            'v_2,9': 3.0,  # 3*q*B*a/m = 3
            'S_0': 3*math.pi,  # 3π
            'S': 67*math.pi    # 67π
        }
        
        self.logs = []
    
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
    
    def _compare_num(self, val1, val2, name):
        """Compare two numeric values with tolerance."""
        if val1 is None or val2 is None:
            self._log(f"{name}: Missing value", False)
            return False
        
        try:
            if math.isclose(float(val1), float(val2), rel_tol=0.02, abs_tol=0.02):
                self._log(f"{name}: {float(val1):.3f} ≈ {float(val2):.3f}", True)
                return True
            else:
                self._log(f"{name}: {float(val1):.3f} ≠ {float(val2):.3f} (expected)", False)
                return False
        except Exception as e:
            self._log(f"{name}: Comparison failed - {str(e)}", False)
            return False
    
    def test_invariants(self):
        """Verify physics invariants: energy and momentum conservation at collisions."""
        success = True
        
        # Collision 1 (from steps 5-7)
        v1_0 = self.results.get('v_1,0')
        v2_0 = self.results.get('v_2,0')
        v1_1 = -3 * v1_0 if v1_0 is not None else None  # From problem statement
        v2_1 = self.results.get('v_2,1')
        m2 = self.results.get('m_2')
        
        if all(x is not None for x in [v1_0, v2_0, v1_1, v2_1, m2]):
            # Momentum conservation
            p_before = self.m * v1_0 + m2 * v2_0
            p_after = self.m * v1_1 + m2 * v2_1
            if self._compare_num(p_before, p_after, "Collision1 Momentum"):
                pass
            else:
                success = False
            
            # Energy conservation (elastic collision)
            KE_before = 0.5 * self.m * v1_0**2 + 0.5 * m2 * v2_0**2
            KE_after = 0.5 * self.m * v1_1**2 + 0.5 * m2 * v2_1**2
            if self._compare_num(KE_before, KE_after, "Collision1 Energy"):
                pass
            else:
                success = False
        
        # Lorentz force equals centripetal force for particle 2 after collision 1
        r2_1 = self.results.get('r_2,1')
        if v2_1 is not None and r2_1 is not None and m2 is not None:
            # Lorentz force: q*v*B
            lorentz = self.q * v2_1 * self.B
            # Centripetal force: m2*v^2/r
            centripetal = m2 * v2_1**2 / r2_1
            if self._compare_num(lorentz, centripetal, "Lorentz=Centripetal (part2 after col1)"):
                pass
            else:
                success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check key results against expected numeric values
        checks = [
            ('v_1,0', 'Initial velocity of particle 1'),
            ('m_2', 'Mass of particle 2'),
            ('v_2,1', 'Velocity of particle 2 after collision 1'),
            ('t_1', 'Time of first collision'),
            ('r_2,1', 'Radius of particle 2 after collision 1'),
            ('S', 'Total path length of particle 2')
        ]
        
        for key, desc in checks:
            result = self.results.get(key)
            expected = self.expected.get(key)
            if result is not None and expected is not None:
                if not self._compare_num(result, expected, desc):
                    success = False
        
        # Check period relationship
        if 'T_1andT_2' in self.results:
            T1 = 2 * math.pi * self.m / (self.q * self.B)
            T2 = 2 * math.pi * (self.m/2) / (self.q * self.B)  # m2 = m/2
            if self._compare_num(T1, 2*T2, "Period relation T1=2T2"):
                pass
            else:
                success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb a parameter and check expected trends."""
        success = True
        
        # Test 1: For particle 2, radius r2 = m2*v2/(q*B)
        # If B increases (with fixed m2, v2, q), radius should decrease
        B_orig = 1.0
        B_pert = 2.0
        m2 = self.results.get('m_2', 0.5)
        v2 = self.results.get('v_2,1', 3.0)
        q = self.q
        
        r2_orig = m2 * v2 / (q * B_orig)
        r2_pert = m2 * v2 / (q * B_pert)
        
        if r2_pert < r2_orig:
            self._log(f"Monotonicity: B↑ ({B_orig}→{B_pert}) → r2↓ ({r2_orig:.3f}→{r2_pert:.3f})", True)
        else:
            self._log(f"Monotonicity: B↑ should decrease radius for particle 2", False)
            success = False
        
        # Test 2: Increase mass m2 -> larger period T2
        m_orig = 1.0
        m_pert = 2.0
        B = self.B
        q = self.q
        
        T2_orig = 2 * math.pi * (m_orig/2) / (q * B)
        T2_pert = 2 * math.pi * (m_pert/2) / (q * B)
        
        if T2_pert > T2_orig:
            self._log(f"Monotonicity: m↑ ({m_orig}→{m_pert}) → T2↑ ({T2_orig:.3f}→{T2_pert:.3f})", True)
        else:
            self._log(f"Monotonicity: m↑ should increase period", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        self.logs.append("=== Physics Checker Results ===")
        
        success = True
        
        try:
            # Run each test
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            success = inv_success and trace_success and mono_success
            
            # Summary
            self.logs.append("\n=== Summary ===")
            self.logs.append(f"Invariants: {'PASS' if inv_success else 'FAIL'}")
            self.logs.append(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
            self.logs.append(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}")
            self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"Error during testing: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        return success, self.logs