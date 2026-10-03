import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts inputs and results from the solution instance.
        Uses getattr with fallback to results dictionary to prevent AttributeError.
        """
        # Physical constants
        self.M = getattr(solution_instance, 'M', 0.5)
        self.q = getattr(solution_instance, 'q', 3.0)
        self.m = getattr(solution_instance, 'm', 0.5)
        self.mu1 = getattr(solution_instance, 'mu1', 0.2)
        self.mu2 = getattr(solution_instance, 'mu2', 0.4)
        self.E = getattr(solution_instance, 'E', 3.0)
        self.t = getattr(solution_instance, 't', 2.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Results dictionary (fallback if attributes not directly accessible)
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallbacks
        self.a_1 = getattr(solution_instance, 'a_1', self.results.get('a_1', None))
        self.a_2 = getattr(solution_instance, 'a_2', self.results.get('a_2', None))
        self.v_1 = getattr(solution_instance, 'v_1', self.results.get('v_1', None))
        self.v_2 = getattr(solution_instance, 'v_2', self.results.get('v_2', None))
        self.v = getattr(solution_instance, 'v', self.results.get('v', None))
        self.t_prime = getattr(solution_instance, 't_prime', self.results.get('t\'', self.results.get('t_prime', None)))
        self.x_1 = getattr(solution_instance, 'x_1', self.results.get('x_1', None))
        self.x_2 = getattr(solution_instance, 'x_2', self.results.get('x_2', None))
        self.x_1_prime = getattr(solution_instance, 'x_1_prime', self.results.get('x_1\'', self.results.get('x_1_prime', None)))
        self.x_2_prime = getattr(solution_instance, 'x_2_prime', self.results.get('x_2\'', self.results.get('x_2_prime', None)))
        self.a_prime = getattr(solution_instance, 'a_prime', self.results.get('a\'', self.results.get('a_prime', None)))
        self.a_dprime = getattr(solution_instance, 'a_dprime', self.results.get('a\'\'', self.results.get('a_dprime', None)))
        self.delta_x = getattr(solution_instance, 'delta_x', self.results.get('delta_x', self.results.get('Δx', None)))
        
        # For symbolic checks
        self.symbols = {
            'M': sp.symbols('M'),
            'm': sp.symbols('m'),
            'q': sp.symbols('q'),
            'mu1': sp.symbols('mu1'),
            'mu2': sp.symbols('mu2'),
            'E': sp.symbols('E'),
            'g': sp.symbols('g'),
            't': sp.symbols('t'),
            't_prime': sp.symbols('t_prime'),
            'a_1': sp.symbols('a_1'),
            'a_2': sp.symbols('a_2'),
            'v_1': sp.symbols('v_1'),
            'v_2': sp.symbols('v_2'),
            'v': sp.symbols('v'),
            'a_prime': sp.symbols('a_prime'),
            'a_dprime': sp.symbols('a_dprime'),
            'x_1': sp.symbols('x_1'),
            'x_2': sp.symbols('x_2'),
            'x_1_prime': sp.symbols('x_1_prime'),
            'x_2_prime': sp.symbols('x_2_prime'),
            'delta_x': sp.symbols('delta_x')
        }
        
        self.numeric_values = {
            'M': self.M, 'm': self.m, 'q': self.q, 'mu1': self.mu1, 'mu2': self.mu2,
            'E': self.E, 'g': self.g, 't': self.t
        }

    def is_close(self, a, b):
        """Fuzzy numerical matching with 1-2% tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        
        # 1. Newton's 2nd Law for small dye block (step 1 & 3)
        if self.a_1 is not None:
            expected_a1 = self.mu1 * self.g
            if not self.is_close(self.a_1, expected_a1):
                logs.append(f"FAIL: a_1 = {self.a_1}, expected μ1*g = {expected_a1}")
            else:
                logs.append(f"PASS: a_1 = μ1*g = {expected_a1}")
        else:
            logs.append("SKIP: a_1 not available")
        
        # 2. Newton's 2nd Law for insulating plate (step 2 & 4)
        if self.a_2 is not None:
            # Force balance: qE - μ1*m*g - μ2*(m+M)*g = M*a_2
            net_force = self.q * self.E - self.mu1 * self.m * self.g - self.mu2 * (self.m + self.M) * self.g
            expected_a2 = net_force / self.M
            if not self.is_close(self.a_2, expected_a2):
                logs.append(f"FAIL: a_2 = {self.a_2}, expected (qE-μ1mg-μ2(m+M)g)/M = {expected_a2}")
            else:
                logs.append(f"PASS: a_2 satisfies Newton's 2nd Law = {expected_a2}")
        else:
            logs.append("SKIP: a_2 not available")
        
        # 3. Kinematics consistency for velocities after 2s (step 5-7)
        if all(v is not None for v in [self.v_1, self.a_1, self.v_2, self.a_2]):
            expected_v1 = self.a_1 * self.t
            expected_v2 = self.a_2 * self.t
            if not self.is_close(self.v_1, expected_v1):
                logs.append(f"FAIL: v_1 = {self.v_1}, expected a_1*t = {expected_v1}")
            else:
                logs.append(f"PASS: v_1 = a_1*t = {expected_v1}")
            if not self.is_close(self.v_2, expected_v2):
                logs.append(f"FAIL: v_2 = {self.v_2}, expected a_2*t = {expected_v2}")
            else:
                logs.append(f"PASS: v_2 = a_2*t = {expected_v2}")
        else:
            logs.append("SKIP: v_1 or v_2 not available")
        
        # 4. Deceleration after electric field removed (step 8)
        if self.a_prime is not None:
            expected_aprime = (self.mu1 * self.m * self.g + self.mu2 * (self.m + self.M) * self.g) / self.M
            if not self.is_close(self.a_prime, expected_aprime):
                logs.append(f"FAIL: a' = {self.a_prime}, expected (μ1mg+μ2(m+M)g)/M = {expected_aprime}")
            else:
                logs.append(f"PASS: a' satisfies Newton's 2nd Law = {expected_aprime}")
        else:
            logs.append("SKIP: a' not available")
        
        # 5. Velocity equality condition (step 9-11)
        if all(v is not None for v in [self.t_prime, self.v, self.v_1, self.a_1, self.v_2, self.a_prime]):
            # Two equations should give same v and t'
            v_from_block = self.v_1 + self.a_1 * self.t_prime
            v_from_plate = self.v_2 - self.a_prime * self.t_prime
            if not (self.is_close(v_from_block, v_from_plate) and self.is_close(self.v, v_from_block)):
                logs.append(f"FAIL: Velocity equality inconsistent. v={self.v}, v1+a1*t'={v_from_block}, v2-a'*t'={v_from_plate}")
            else:
                logs.append(f"PASS: Velocity equality consistent: v = {self.v}")
            
            # Check t' formula
            expected_tprime = (self.v_2 - self.v_1) / (self.a_1 + self.a_prime) if (self.a_1 + self.a_prime) != 0 else None
            if expected_tprime and not self.is_close(self.t_prime, expected_tprime):
                logs.append(f"FAIL: t' = {self.t_prime}, expected (v2-v1)/(a1+a') = {expected_tprime}")
            elif expected_tprime:
                logs.append(f"PASS: t' = (v2-v1)/(a1+a') = {expected_tprime}")
        else:
            logs.append("SKIP: Missing variables for velocity equality check")
        
        # 6. Acceleration after velocities equal (step 16)
        if self.a_dprime is not None:
            expected_adprime = (self.mu2 * (self.m + self.M) * self.g - self.mu1 * self.m * self.g) / self.M
            if not self.is_close(self.a_dprime, expected_adprime):
                logs.append(f"FAIL: a'' = {self.a_dprime}, expected (μ2(m+M)g-μ1mg)/M = {expected_adprime}")
            else:
                logs.append(f"PASS: a'' satisfies Newton's 2nd Law = {expected_adprime}")
        else:
            logs.append("SKIP: a'' not available")
        
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final trace length."""
        logs = []
        
        # Check displacements before E removed (steps 12-13)
        if all(x is not None for x in [self.x_1, self.a_1, self.x_2, self.a_2]):
            expected_x1 = 0.5 * self.a_1 * self.t**2
            expected_x2 = 0.5 * self.a_2 * self.t**2
            if not self.is_close(self.x_1, expected_x1):
                logs.append(f"FAIL: x_1 = {self.x_1}, expected 0.5*a_1*t^2 = {expected_x1}")
            else:
                logs.append(f"PASS: x_1 = 0.5*a_1*t^2 = {expected_x1}")
            if not self.is_close(self.x_2, expected_x2):
                logs.append(f"FAIL: x_2 = {self.x_2}, expected 0.5*a_2*t^2 = {expected_x2}")
            else:
                logs.append(f"PASS: x_2 = 0.5*a_2*t^2 = {expected_x2}")
        else:
            logs.append("SKIP: x_1 or x_2 not available")
        
        # Check displacements after E removed until equal velocity (steps 14-15)
        if all(x is not None for x in [self.x_1_prime, self.x_2_prime, self.v_1, self.v_2, self.v, self.t_prime]):
            expected_x1_prime = 0.5 * (self.v_1 + self.v) * self.t_prime
            expected_x2_prime = 0.5 * (self.v_2 + self.v) * self.t_prime
            if not self.is_close(self.x_1_prime, expected_x1_prime):
                logs.append(f"FAIL: x_1' = {self.x_1_prime}, expected 0.5*(v1+v)*t' = {expected_x1_prime}")
            else:
                logs.append(f"PASS: x_1' = 0.5*(v1+v)*t' = {expected_x1_prime}")
            if not self.is_close(self.x_2_prime, expected_x2_prime):
                logs.append(f"FAIL: x_2' = {self.x_2_prime}, expected 0.5*(v2+v)*t' = {expected_x2_prime}")
            else:
                logs.append(f"PASS: x_2' = 0.5*(v2+v)*t' = {expected_x2_prime}")
        else:
            logs.append("SKIP: x_1' or x_2' not available")
        
        # Check final trace length (step 17)
        if self.delta_x is not None:
            if all(x is not None for x in [self.x_1, self.x_2, self.x_1_prime, self.x_2_prime]):
                expected_delta_x = (self.x_2 + self.x_2_prime) - (self.x_1 + self.x_1_prime)
                if not self.is_close(self.delta_x, expected_delta_x):
                    logs.append(f"FAIL: Δx = {self.delta_x}, expected (x2+x2')-(x1+x1') = {expected_delta_x}")
                else:
                    logs.append(f"PASS: Δx = (x2+x2')-(x1+x1') = {expected_delta_x}")
            else:
                logs.append("SKIP: Cannot compute Δx from intermediate displacements")
        
        # Compare with target answers
        target_a1, target_a2 = 2.0, 8.0
        target_v = 6.0
        target_delta_x = 18.0
        
        if self.a_1 is not None and not self.is_close(self.a_1, target_a1):
            logs.append(f"FAIL: a_1 = {self.a_1}, target answer = {target_a1}")
        if self.a_2 is not None and not self.is_close(self.a_2, target_a2):
            logs.append(f"FAIL: a_2 = {self.a_2}, target answer = {target_a2}")
        if self.v is not None and not self.is_close(self.v, target_v):
            logs.append(f"FAIL: v = {self.v}, target answer = {target_v}")
        if self.delta_x is not None and not self.is_close(self.delta_x, target_delta_x):
            logs.append(f"FAIL: Δx = {self.delta_x}, target answer = {target_delta_x}")
        
        return logs

    def test_monotonicity(self):
        """Perturb mass m and check if trace length changes predictably."""
        logs = []
        
        if self.delta_x is None:
            logs.append("SKIP: Δx not available for monotonicity test")
            return logs
        
        # Perturbation factor
        perturb_factor = 1.01  # 1% increase
        m_original = self.m
        self.m *= perturb_factor
        
        try:
            # Recalculate using same formulas as in steps
            a1_new = self.mu1 * self.g
            net_force = self.q * self.E - self.mu1 * self.m * self.g - self.mu2 * (self.m + self.M) * self.g
            a2_new = net_force / self.M
            
            v1_new = a1_new * self.t
            v2_new = a2_new * self.t
            
            a_prime_new = (self.mu1 * self.m * self.g + self.mu2 * (self.m + self.M) * self.g) / self.M
            
            if (a1_new + a_prime_new) != 0:
                t_prime_new = (v2_new - v1_new) / (a1_new + a_prime_new)
                v_new = v1_new + a1_new * t_prime_new
                
                x1_new = 0.5 * a1_new * self.t**2
                x2_new = 0.5 * a2_new * self.t**2
                x1_prime_new = 0.5 * (v1_new + v_new) * t_prime_new
                x2_prime_new = 0.5 * (v2_new + v_new) * t_prime_new
                
                delta_x_new = (x2_new + x2_prime_new) - (x1_new + x1_prime_new)
                
                # Check if trace length decreased (increased m -> more friction -> shorter trace)
                if delta_x_new < self.delta_x:
                    logs.append(f"PASS: Δx decreases when m increases (Δx_original={self.delta_x:.3f}, Δx_new={delta_x_new:.3f})")
                else:
                    logs.append(f"WARN: Δx did not decrease as expected when m increased")
            else:
                logs.append("SKIP: a1+a' = 0, cannot compute t'")
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
        finally:
            # Restore original value
            self.m = m_original
        
        return logs

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run each test
            for test_name, test_func in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== {test_name} Test ===")
                test_logs = test_func()
                all_logs.extend(test_logs)
                
                # Check for any FAIL messages
                if any("FAIL:" in log for log in test_logs):
                    success = False
            
            # Summary
            all_logs.append(f"\n=== SUMMARY ===")
            if success:
                all_logs.append("ALL TESTS PASSED")
            else:
                all_logs.append("SOME TESTS FAILED")
                
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs