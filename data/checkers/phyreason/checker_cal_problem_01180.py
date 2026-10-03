import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely (hardcoded from problem)
        self.v_A = 10.0
        self.v_B = 30.0
        self.d0 = 750.0  # initial separation (A ahead of B)
        self.stop_dist = 1800.0  # distance for B to stop
        
        # Extract results safely with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.a = self.results.get('a', 0.0)  # deceleration magnitude (positive)
        self.t0 = self.results.get('t_0', 0.0)  # collision time in first scenario
        
        # For part 2 collision: x_A should be displacement of A at collision
        # For part 3: we need t_prime, x_B (new situation), Deltax
        # The results dict should contain all required variables
        self.t_prime = self.results.get('t_prime', 0.0)
        self.Deltax = self.results.get('Deltax', 0.0)
        
        # x_A and x_B could be from either part. We'll test both scenarios separately.
        # We'll compute expected values based on physics rather than assuming which is stored.
        
        # Derived quantities
        self.a_B = -self.a  # acceleration of B (negative)
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Check deceleration from stopping distance
            expected_a = self.v_B**2 / (2 * self.stop_dist)
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Deceleration a={self.a} does not match expected {expected_a} from v_B^2/(2*stop_dist)")
                success = False
            else:
                logs.append(f"PASS: Deceleration a={self.a} matches kinematic stopping condition")
            
            # 2. Check collision condition in first scenario (A at constant speed)
            # Compute expected collision time from physics
            # Equation: d0 + v_A*t0 = v_B*t0 + 0.5*a_B*t0^2
            # 0.5*a_B*t0^2 + (v_B - v_A)*t0 - d0 = 0
            # a_B = -a (negative)
            a_coeff = 0.5 * self.a_B  # = -0.5*a
            b_coeff = self.v_B - self.v_A  # = 20
            c_coeff = -self.d0  # = -750
            
            disc = b_coeff**2 - 4*a_coeff*c_coeff
            if disc >= 0:
                t1 = (-b_coeff - math.sqrt(disc)) / (2*a_coeff)
                t2 = (-b_coeff + math.sqrt(disc)) / (2*a_coeff)
                # Take the smaller positive root
                t_collision_expected = min(t1, t2) if t1>0 and t2>0 else max(t1, t2)
                
                if not math.isclose(self.t0, t_collision_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Collision time t0={self.t0}s does not match expected {t_collision_expected}s")
                    success = False
                else:
                    logs.append(f"PASS: Collision time t0={self.t0}s matches physics")
            else:
                logs.append("WARN: No real collision time from physics equation")
            
            # 3. Check second scenario (A accelerates after delay Δt)
            # Parameters from problem: Δt = 2.0 s, a1 = 0.2 m/s² (from ground truth step 12)
            Delta_t = 2.0
            a1 = 0.2  # CORRECTED from 0.5 to 0.2
            
            # Check if t_prime makes speeds equal
            v_B_at_tprime = self.v_B + self.a_B * self.t_prime
            v_A_at_tprime = self.v_A + a1 * (self.t_prime - Delta_t)
            if not math.isclose(v_B_at_tprime, v_A_at_tprime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Speeds at t' not equal: v_B={v_B_at_tprime}, v_A={v_A_at_tprime}")
                success = False
            else:
                logs.append(f"PASS: Speeds equal at t'={self.t_prime}s")
            
            # Compute expected x_B and x_A from t_prime
            x_B_expected = self.v_B * self.t_prime + 0.5 * self.a_B * self.t_prime**2
            x_A_expected = self.v_A * Delta_t + self.v_A * (self.t_prime - Delta_t) + 0.5 * a1 * (self.t_prime - Delta_t)**2
            Deltax_expected = self.d0 + x_A_expected - x_B_expected
            
            # Get x_B and x_A from results (they should be from part 3)
            x_B_result = self.results.get('x_B', 0.0)
            x_A_result = self.results.get('x_A', 0.0)
            
            # Check x_B consistency
            if not math.isclose(x_B_result, x_B_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: x_B={x_B_result} differs from expected {x_B_expected}m (part 3)")
            
            # Check x_A consistency  
            if not math.isclose(x_A_result, x_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: x_A={x_A_result} differs from expected {x_A_expected}m (part 3)")
            
            # Check Deltax
            if not math.isclose(self.Deltax, Deltax_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Deltax={self.Deltax} does not match expected {Deltax_expected}m")
                success = False
            else:
                logs.append(f"PASS: Deltax={self.Deltax}m consistent with physics")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check that the time when speeds are equal in first scenario (t=80s) is consistent
            t_eq = (self.v_B - self.v_A) / self.a if self.a != 0 else 0
            logs.append(f"Computed t_eq (when speeds equal in part 2) = {t_eq}s")
            
            # Check positions at t_eq from steps: x_B=1600m, x_A=800m (displacements from t=0)
            x_B_eq = self.v_B * t_eq + 0.5 * self.a_B * t_eq**2
            x_A_eq = self.v_A * t_eq  # displacement of A from t=0
            
            if not math.isclose(x_B_eq, 1600.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: x_B at t_eq={x_B_eq} differs from step value 1600m")
            else:
                logs.append(f"PASS: x_B at t_eq={x_B_eq}m matches 1600m")
                
            if not math.isclose(x_A_eq, 800.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: x_A displacement at t_eq={x_A_eq} differs from step value 800m")
            else:
                logs.append(f"PASS: x_A at t_eq={x_A_eq}m matches 800m")
            
            # Relative distance at t_eq: initial separation + (x_A_eq - x_B_eq)
            rel_dist = self.d0 + (x_A_eq - x_B_eq)
            logs.append(f"Relative distance at t_eq={rel_dist}m")
            
            # Verify that at t0 (collision time), speeds are not equal (since t0=60 < t_eq=80)
            if self.t0 < t_eq:
                logs.append("PASS: Collision occurs before speeds become equal (t0 < t_eq)")
            else:
                logs.append("WARN: Collision time t0 >= t_eq, may indicate inconsistency")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb initial separation d0 and check collision time trend
            original_d0 = self.d0
            perturbed_d0 = original_d0 + 10.0  # increase by 10m
            
            # Solve collision time for perturbed case
            # Equation: d0 + v_A*t = v_B*t + 0.5*a_B*t^2
            a_coeff = 0.5 * self.a_B  # = -0.125
            b_coeff = self.v_B - self.v_A  # = 20
            c_coeff = -perturbed_d0  # = -760
            
            disc = b_coeff**2 - 4*a_coeff*c_coeff
            if disc >= 0:
                t1 = (-b_coeff - math.sqrt(disc)) / (2*a_coeff)
                t2 = (-b_coeff + math.sqrt(disc)) / (2*a_coeff)
                # Take the smaller positive root (first collision)
                t_collision_perturbed = min(t1, t2) if t1>0 and t2>0 else max(t1, t2)
                logs.append(f"Perturbed d0={perturbed_d0}m gives collision time t={t_collision_perturbed}s")
                if t_collision_perturbed > self.t0:
                    logs.append("PASS: Increasing initial separation delays collision (monotonic)")
                else:
                    logs.append("FAIL: Collision time did not increase with increased separation")
                    success = False
            else:
                logs.append("WARN: No real collision time for perturbed d0 (no collision)")
            
            # Restore d0
            self.d0 = original_d0
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            test_success, test_logs = test_func()
            all_logs.append(f"=== {name} ===")
            all_logs.extend(test_logs)
            if not test_success:
                overall_success = False
        
        return overall_success, all_logs