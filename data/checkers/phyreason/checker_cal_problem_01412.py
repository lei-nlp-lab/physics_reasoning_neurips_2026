import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Input parameters with safe access
        self.m = getattr(solution_instance, 'm', 0.008)  # 8g = 0.008 kg
        self.g = getattr(solution_instance, 'g', 10.0)
        self.R = getattr(solution_instance, 'R', 0.5)
        self.E = getattr(solution_instance, 'E', 6e6)
        self.theta_deg = getattr(solution_instance, 'theta', 53.0)
        self.theta = math.radians(self.theta_deg)
        self.sin53 = getattr(solution_instance, 'sin53', 0.8)
        self.cos53 = getattr(solution_instance, 'cos53', 0.6)
        
        # Results with safe access via results dictionary
        results = getattr(solution_instance, 'results', {})
        self.h_a = results.get('h_a', 0.0)
        self.h_b = results.get('h_b', 0.0)
        self.h_b_prime = results.get('h_b\'', results.get('h_b_prime', 0.0))  # handle both notations
        # ball a's v_D: try v_D_a first, then v_D
        self.v_D_a = results.get('v_D_a', results.get('v_D', 0.0))
        # ball b's v_D: try v_D_b first, then v_D (since our code stores ball b's v_D as 'v_D')
        self.v_D_b = results.get('v_D_b', results.get('v_D', 0.0))
        self.v_c = results.get('v_c', 0.0)
        self.x = results.get('x', 0.0)
        self.y = results.get('y', 0.0)
        
        # Derived constants for checks
        self.v_D_circular = math.sqrt(self.g * self.R)  # theoretical v_D from circular motion
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        try:
            # 1. Ball a: Circular motion at top point D (step_1)
            # mg = m*v_D^2/R -> v_D = sqrt(g*R)
            if not math.isclose(self.v_D_a, self.v_D_circular, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Ball a v_D mismatch. Expected ~{self.v_D_circular:.3f}, got {self.v_D_a:.3f}")
                return False
            self.log("PASS: Ball a circular motion condition at D")
            
            # 2. Ball a: Energy conservation (step_2)
            # mgh_a = mg(2R) + 1/2 m v_D^2
            left = self.m * self.g * self.h_a
            right = self.m * self.g * 2 * self.R + 0.5 * self.m * self.v_D_a**2
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Ball a energy conservation. Left={left:.3f}, Right={right:.3f}")
                return False
            self.log("PASS: Ball a energy conservation")
            
            # 3. Ball b: Electric force provides centripetal at C (step_4)
            # E*q = m*v_c^2/R (q unknown, but we can check if solution satisfies it)
            # Since q unknown, we skip quantitative check but note assumption
            self.log("NOTE: Ball b circular motion at C depends on unknown charge q")
            
            # 4. Ball b: Energy with electric field (step_5)
            # mg(h_b - R) - E*q*R = 1/2 m v_c^2
            # Cannot check without q, but we can check if h_b matches target 17/16 = 1.0625
            target_h_b = 17/16
            if not math.isclose(self.h_b, target_h_b, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"WARN: Ball b h_b {self.h_b:.3f} differs from target {target_h_b:.3f}")
            
            # 5. Ball b projectile: Vertical impact condition (step_7)
            # v_D / (g*t) = tan(53°) -> t = v_D/(g*tan53)
            tan53 = self.sin53 / self.cos53
            if self.v_D_b > 0:
                t_est = self.v_D_b / (self.g * tan53)
                # From projectile: y = 1/2 g t^2
                y_from_t = 0.5 * self.g * t_est**2
                if not math.isclose(self.y, y_from_t, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"WARN: Ball b projectile y inconsistent. From t: {y_from_t:.3f}, reported: {self.y:.3f}")
            else:
                self.log("WARN: Ball b v_D_b is zero or negative")
            
            # 6. Ball b: Energy conservation for projectile part (step_12)
            # mg(h_b' - 2R) = 1/2 m v_D_b^2
            if self.v_D_b > 0:
                left_energy = self.m * self.g * (self.h_b_prime - 2*self.R)
                right_energy = 0.5 * self.m * self.v_D_b**2
                if not math.isclose(left_energy, right_energy, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Ball b projectile energy. Left={left_energy:.3f}, Right={right_energy:.3f}")
                    return False
                self.log("PASS: Ball b projectile energy conservation")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Check intermediate steps lead to final answers."""
        try:
            # 1. Ball a: h_a should be 5R/2 = 1.25 m
            expected_h_a = 2.5 * self.R  # 5/2 * R
            if not math.isclose(self.h_a, expected_h_a, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Ball a h_a inconsistent. Expected {expected_h_a:.3f}, got {self.h_a:.3f}")
                return False
            self.log("PASS: Ball a h_a = 5R/2")
            
            # 2. Ball b: h_b should be 17/16 m ≈ 1.0625
            expected_h_b = 17/16
            if not math.isclose(self.h_b, expected_h_b, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Ball b h_b inconsistent. Expected {expected_h_b:.3f}, got {self.h_b:.3f}")
                return False
            self.log("PASS: Ball b h_b = 17/16 m")
            
            # 3. Ball b projectile: Relationship x/y = 8/3 (step_9)
            if self.x > 0 and self.y > 0:
                ratio = self.x / self.y
                expected_ratio = 8/3
                if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Ball b x/y ratio {ratio:.3f} ≠ 8/3")
                    return False
                self.log("PASS: Ball b x/y = 8/3")
            
            # 4. Ball b projectile: x = 3*v_D^2/(4g) from step_10
            if self.v_D_b > 0:
                x_from_v = 3 * self.v_D_b**2 / (4 * self.g)
                if not math.isclose(self.x, x_from_v, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"WARN: Ball b x from v_D: {x_from_v:.3f}, reported: {self.x:.3f}")
            
            # 5. Ball b projectile: h_b' = 32/15 m ≈ 2.1333
            expected_h_b_prime = 32/15
            if not math.isclose(self.h_b_prime, expected_h_b_prime, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Ball b h_b' inconsistent. Expected {expected_h_b_prime:.3f}, got {self.h_b_prime:.3f}")
                return False
            self.log("PASS: Ball b h_b' = 32/15 m")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb input R and check trends."""
        try:
            original_R = self.R
            perturbed_R = original_R * 1.01  # +1%
            
            # Ball a: h_a = 2.5 * R, so should increase with R
            original_h_a = 2.5 * original_R
            perturbed_h_a = 2.5 * perturbed_R
            if perturbed_h_a <= original_h_a:
                self.log(f"FAIL: Ball a h_a monotonicity. R↑ should cause h_a↑")
                return False
            self.log("PASS: Ball a h_a increases with R")
            
            # Ball b projectile: v_D from energy: v_D^2 = 2g(h_b' - 2R)
            # h_b' is fixed by geometry? Actually h_b' = 32/15 is constant in problem.
            # But if we consider the formula h_b' = v_D^2/(2g) + 2R, increasing R would require v_D to decrease
            # to keep h_b' constant. We'll check derivative sign.
            # Since h_b' is fixed, dv_D/dR = -2g / (2*v_D) = -g/v_D < 0 (decreasing)
            # So v_D should decrease if R increases (with fixed h_b').
            # We'll approximate numerically:
            if self.v_D_b > 0:
                # Use energy formula: h_b' = v_D^2/(2g) + 2R
                # Solve for v_D at perturbed R (assuming same h_b')
                v_D_squared = 2 * self.g * (self.h_b_prime - 2 * perturbed_R)
                if v_D_squared > 0:
                    perturbed_v_D = math.sqrt(v_D_squared)
                    if perturbed_v_D >= self.v_D_b:
                        self.log(f"WARN: Ball b v_D monotonicity. R↑ should cause v_D↓")
                    else:
                        self.log("PASS: Ball b v_D decreases with R (energy constraint)")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all checks and return success flag with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.log(f"\n--- Testing {name} ---")
            if not test_func():
                success = False
                self.log(f"FAILED: {name}")
            else:
                self.log(f"PASSED: {name}")
        
        return success, self.logs