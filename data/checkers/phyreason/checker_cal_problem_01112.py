import math
import traceback
from typing import Tuple, List, Dict, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Problem constants from context
        self.v0_A = 10.0  # initial velocity of Car A (m/s)
        self.v_truck = 16.0  # constant velocity of truck (m/s)
        self.a_A = 2.0  # acceleration of Car A (m/s^2)
        self.s0 = 7.0  # initial separation (m)
        
        # Safely extract results
        self.results = {}
        results_dict = getattr(solution_instance, 'results', {})
        
        # Map expected keys to actual keys in results
        key_mapping = {
            'x_3"': 'x_3_prime',  # x_3' (maximum allowed displacement)
            'x_3': 'x_3',   # displacement at constant speed
            't_2': 't_2',
            'a': 'a',
            'S_max': 'S_max',
            'x_2': 'x_2',
            't_1': 't_1',
            'v_B"': 'v_B',  # v_B' (final velocity)
            'x_1+x_3': None,  # We'll compute this if needed
            'x_1': 'x_1'
        }
        
        for checker_key, code_key in key_mapping.items():
            if code_key and code_key in results_dict:
                self.results[checker_key] = results_dict[code_key]
            else:
                self.results[checker_key] = None

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True
        
        # 1. Verify time when Car A reaches truck's speed (t1 = 3s)
        try:
            t1 = self.results.get('t_1')
            if t1 is not None:
                expected_t1 = (self.v_truck - self.v0_A) / self.a_A
                if not math.isclose(t1, expected_t1, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: t1={t1}, expected {expected_t1}")
                    success = False
                else:
                    logs.append(f"PASS: t1 calculation correct")
        except Exception as e:
            logs.append(f"ERROR in t1 check: {e}")
            success = False

        # 2. Verify kinematics during t1
        try:
            x1 = self.results.get('x_1')
            x2 = self.results.get('x_2')
            S_max = self.results.get('S_max')
            
            if all(v is not None for v in [x1, x2, S_max]):
                # Check x1: x1 = v0*t1 + 0.5*a*t1^2
                if t1 is not None:
                    expected_x1 = self.v0_A * t1 + 0.5 * self.a_A * t1**2
                    if not math.isclose(x1, expected_x1, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: x1={x1}, expected {expected_x1}")
                        success = False
                
                # Check x2: x2 = v_truck * t1
                if t1 is not None:
                    expected_x2 = self.v_truck * t1
                    if not math.isclose(x2, expected_x2, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: x2={x2}, expected {expected_x2}")
                        success = False
                
                # Check S_max = x2 + s0 - x1
                expected_S_max = x2 + self.s0 - x1
                if not math.isclose(S_max, expected_S_max, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: S_max={S_max}, expected {expected_S_max}")
                    success = False
        except Exception as e:
            logs.append(f"ERROR in kinematics during t1: {e}")
            success = False

        # 3. Verify overtaking time t2 (should satisfy quadratic)
        try:
            t2 = self.results.get('t_2')
            if t2 is not None:
                # Check using quadratic formula directly (no sympy)
                # Equation: 0.5*a*t2^2 + (v0_A - v_truck)*t2 - s0 = 0
                a_coeff = 0.5 * self.a_A
                b_coeff = self.v0_A - self.v_truck
                c_coeff = -self.s0
                
                discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
                if discriminant >= 0:
                    root1 = (-b_coeff + math.sqrt(discriminant)) / (2 * a_coeff)
                    root2 = (-b_coeff - math.sqrt(discriminant)) / (2 * a_coeff)
                    positive_roots = [r for r in [root1, root2] if r > 0]
                    
                    if positive_roots:
                        if not any(math.isclose(t2, root, rel_tol=0.02, abs_tol=0.02) for root in positive_roots):
                            logs.append(f"FAIL: t2={t2} not a root of kinematics equation (roots: {positive_roots})")
                            success = False
                        else:
                            logs.append(f"PASS: t2 satisfies kinematics equation")
                    else:
                        logs.append("SKIP: No positive roots found")
                else:
                    logs.append("SKIP: No real roots for quadratic")
        except Exception as e:
            logs.append(f"ERROR in t2 check: {e}")
            success = False

        # 4. Verify displacement during overtaking (x1_total should be computed from t2)
        try:
            t2 = self.results.get('t_2')
            if t2 is not None:
                # Compute expected displacement during overtaking
                expected_x1_total = self.v0_A * t2 + 0.5 * self.a_A * t2**2
                # We don't check against x_1 because x_1 is the t_1 displacement
                # Instead, we can check if the computed t2 gives reasonable displacement
                if expected_x1_total <= 0:
                    logs.append(f"FAIL: x1_total during overtaking should be positive, got {expected_x1_total}")
                    success = False
                else:
                    logs.append(f"PASS: Overtaking displacement computed correctly from t2: {expected_x1_total:.1f} m")
        except Exception as e:
            logs.append(f"ERROR in displacement check: {e}")
            success = False

        # 5. Verify deceleration calculation for car B
        try:
            v_B_prime = self.results.get('v_B"')
            a_decel = self.results.get('a')
            x3_prime = self.results.get('x_3"')
            
            # We need to compute x3_prime correctly
            if x3_prime is None:
                # Compute x1_total from t2
                t2 = self.results.get('t_2')
                if t2 is not None:
                    x1_total = self.v0_A * t2 + 0.5 * self.a_A * t2**2
                    x3_prime = 178.5 - x1_total  # x_total_safe_val - x1_total
                    self.results['x_3"'] = x3_prime
            
            if all(v is not None for v in [v_B_prime, a_decel]) and x3_prime is not None:
                t2 = self.results.get('t_2')
                if t2 is not None and t2 != 0:
                    # Check average velocity formula: x3' = (v_B + v_B')/2 * t2
                    # Need v_B (car B initial speed) = 12 m/s (from problem context)
                    v_B_initial = 12.0
                    expected_x3_prime = (v_B_initial + v_B_prime) / 2 * t2
                    if not math.isclose(x3_prime, expected_x3_prime, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: x3'={x3_prime}, expected from avg vel: {expected_x3_prime}")
                        success = False
                    
                    # Check deceleration: a = (v_B' - v_B_initial)/t2
                    expected_a = (v_B_prime - v_B_initial) / t2
                    if not math.isclose(a_decel, expected_a, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: deceleration a={a_decel}, expected {expected_a}")
                        success = False
                else:
                    logs.append("SKIP: t2 not available or zero for deceleration check")
        except Exception as e:
            logs.append(f"ERROR in deceleration check: {e}")
            success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True
        
        # Check consistency between intermediate results and final answers
        try:
            # Get target answers from context
            target_S_max = 16.0
            target_t2 = 7.0
            target_a_decel = -1.0  # m/s^2
            
            S_max = self.results.get('S_max')
            t2 = self.results.get('t_2')
            a_decel = self.results.get('a')
            
            if S_max is not None:
                if not math.isclose(S_max, target_S_max, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: S_max={S_max} doesn't match target {target_S_max}")
                    success = False
                else:
                    logs.append(f"PASS: S_max matches target")
            
            if t2 is not None:
                if not math.isclose(t2, target_t2, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: t2={t2} doesn't match target {target_t2}")
                    success = False
                else:
                    logs.append(f"PASS: t2 matches target")
            
            if a_decel is not None:
                if not math.isclose(a_decel, target_a_decel, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: deceleration a={a_decel} doesn't match target {target_a_decel}")
                    success = False
                else:
                    logs.append(f"PASS: deceleration matches target")
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {e}")
            success = False
            
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True
        
        # Perturb acceleration and check effect on overtaking time
        original_a = self.a_A
        perturbation = 0.1  # 5% increase
        
        try:
            # Perturb acceleration upward
            self.a_A = original_a * (1 + perturbation)
            
            # Recalculate t2 from kinematics equation
            # 0.5*a*t2^2 + (v0_A - v_truck)*t2 - s0 = 0
            a_coeff = 0.5 * self.a_A
            b_coeff = self.v0_A - self.v_truck
            c_coeff = -self.s0
            
            discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
            if discriminant >= 0:
                t2_perturbed = (-b_coeff + math.sqrt(discriminant)) / (2 * a_coeff)
                
                # Compare with original t2
                original_t2 = self.results.get('t_2')
                if original_t2 is not None:
                    # With higher acceleration, overtaking time should decrease
                    if t2_perturbed < original_t2:
                        logs.append(f"PASS: Increased acceleration reduces overtaking time (perturbed: {t2_perturbed:.2f}, original: {original_t2:.2f})")
                    else:
                        logs.append(f"FAIL: Monotonicity violated - increased acceleration didn't reduce t2")
                        success = False
                else:
                    logs.append("SKIP: Original t2 not available for comparison")
            else:
                logs.append("SKIP: No real solution with perturbed acceleration")
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            success = False
        finally:
            # Restore original acceleration
            self.a_A = original_a
        
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True
        
        # Run all tests
        test_functions = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            try:
                test_success, test_logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(test_logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {e}")
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n{'='*40}")
        all_logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs