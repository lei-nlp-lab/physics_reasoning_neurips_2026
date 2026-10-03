import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        # Extract inputs with safe defaults (from problem context and steps)
        self.a1 = getattr(solution_instance, 'a1', getattr(solution_instance, 'a_1', 2.0))  # acceleration m/s^2
        self.vB = getattr(solution_instance, 'vB', getattr(solution_instance, 'v_B', 10.0))  # constant speed m/s
        self.v = getattr(solution_instance, 'v', 24.0)  # max speed for car A
        self.a2 = getattr(solution_instance, 'a2', getattr(solution_instance, 'a_2', -6.0))  # deceleration m/s^2
        
        # Extract results from solution_instance.results (safe access)
        results = getattr(solution_instance, 'results', {})
        self.t1 = results.get('t_1', 0.0)
        self.t2 = results.get('t_2', 0.0)
        self.x1 = results.get('x_1', 0.0)
        self.x2 = results.get('x_2', 0.0)
        
        # Fix: Use proper string escaping for keys with apostrophes
        # Try multiple possible key names
        self.x1_prime = results.get("x_1'", results.get('x_1_prime', 0.0))
        self.x2_prime = results.get("x_2'", results.get('x_2_prime', 0.0))
        
        self.delta_s = results.get('delta_s', results.get('Δs', 0.0))
        self.t = results.get('t', 0.0)
        self.t_all = results.get('t_all', 0.0)
        
        # Constants for numerical checks
        self.eps_rel = 1e-2
        self.eps_abs = 1e-2
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Check: At t1, speeds should be equal (v_A = a1*t1 = vB)
            v_A_at_t1 = self.a1 * self.t1
            if not math.isclose(v_A_at_t1, self.vB, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: At t1={self.t1}, v_A={v_A_at_t1} ≠ vB={self.vB}")
                success = False
            else:
                logs.append(f"PASS: Speeds equal at t1 (v_A={v_A_at_t1} = vB={self.vB})")
                
            # 2. Check: At t2, displacements should be equal (0.5*a1*t2^2 = vB*t2)
            x_A_at_t2 = 0.5 * self.a1 * self.t2**2
            x_B_at_t2 = self.vB * self.t2
            if not math.isclose(x_A_at_t2, x_B_at_t2, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: At t2={self.t2}, x_A={x_A_at_t2} ≠ x_B={x_B_at_t2}")
                success = False
            else:
                logs.append(f"PASS: Displacements equal at t2 (x_A={x_A_at_t2} = x_B={x_B_at_t2})")
                
            # 3. Check: x1 formula (x1 = v^2/(2*a1))
            x1_calc = self.v**2 / (2 * self.a1)
            if not math.isclose(self.x1, x1_calc, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: x1={self.x1} ≠ v^2/(2*a1)={x1_calc}")
                success = False
            else:
                logs.append(f"PASS: x1 formula correct (x1={self.x1} = {x1_calc})")
                
            # 4. Check: x2 formula (x2 = vB * (v/a1))
            t_acc = self.v / self.a1  # time to reach speed v
            x2_calc = self.vB * t_acc
            if not math.isclose(self.x2, x2_calc, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: x2={self.x2} ≠ vB*(v/a1)={x2_calc}")
                success = False
            else:
                logs.append(f"PASS: x2 formula correct (x2={self.x2} = {x2_calc})")
                
            # 5. Check: x1' formula (x1' = v^2/(2*|a2|))
            x1_prime_calc = self.v**2 / (2 * abs(self.a2))
            if not math.isclose(self.x1_prime, x1_prime_calc, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: x1'={self.x1_prime} ≠ v^2/(2*|a2|)={x1_prime_calc}")
                success = False
            else:
                logs.append(f"PASS: x1' formula correct (x1'={self.x1_prime} = {x1_prime_calc})")
                
            # 6. Check: x2' formula (x2' = vB * (v/|a2|))
            t_dec = self.v / abs(self.a2)  # time to decelerate from v to 0
            x2_prime_calc = self.vB * t_dec
            if not math.isclose(self.x2_prime, x2_prime_calc, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: x2'={self.x2_prime} ≠ vB*(v/|a2|)={x2_prime_calc}")
                success = False
            else:
                logs.append(f"PASS: x2' formula correct (x2'={self.x2_prime} = {x2_prime_calc})")
                
            # 7. Check: Δs = (x1 + x1') - (x2 + x2')
            delta_s_calc = (self.x1 + self.x1_prime) - (self.x2 + self.x2_prime)
            if not math.isclose(self.delta_s, delta_s_calc, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: Δs={self.delta_s} ≠ (x1+x1')-(x2+x2')={delta_s_calc}")
                success = False
            else:
                logs.append(f"PASS: Δs formula correct (Δs={self.delta_s} = {delta_s_calc})")
                
            # 8. Check: t = Δs / vB
            t_calc = self.delta_s / self.vB
            if not math.isclose(self.t, t_calc, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: t={self.t} ≠ Δs/vB={t_calc}")
                success = False
            else:
                logs.append(f"PASS: t formula correct (t={self.t} = {t_calc})")
                
            # 9. Check: t_all = v/a1 + v/|a2| + t
            t_all_calc = (self.v / self.a1) + (self.v / abs(self.a2)) + self.t
            if not math.isclose(self.t_all, t_all_calc, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: t_all={self.t_all} ≠ v/a1+v/|a2|+t={t_all_calc}")
                success = False
            else:
                logs.append(f"PASS: t_all formula correct (t_all={self.t_all} = {t_all_calc})")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Cross-check that intermediate results align with final answers
            # Target answers: ["$5s$, $10s$", "$19.2s$"]
            
            # Check t1 and t2 match expected values from steps
            if not math.isclose(self.t1, 5.0, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: t1={self.t1} ≠ expected 5.0")
                success = False
            else:
                logs.append(f"PASS: t1 matches expected value (5.0)")
                
            if not math.isclose(self.t2, 10.0, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: t2={self.t2} ≠ expected 10.0")
                success = False
            else:
                logs.append(f"PASS: t2 matches expected value (10.0)")
                
            # Check t_all matches expected value
            if not math.isclose(self.t_all, 19.2, rel_tol=self.eps_rel, abs_tol=self.eps_abs):
                logs.append(f"FAIL: t_all={self.t_all} ≠ expected 19.2")
                success = False
            else:
                logs.append(f"PASS: t_all matches expected value (19.2)")
                
            # Verify that all calculated times are positive
            times = [self.t1, self.t2, self.t, self.t_all]
            time_names = ['t1', 't2', 't', 't_all']
            for t_val, name in zip(times, time_names):
                if t_val <= 0:
                    logs.append(f"FAIL: {name}={t_val} is not positive")
                    success = False
                    
            # Verify distances are non-negative
            distances = [self.x1, self.x2, self.x1_prime, self.x2_prime, self.delta_s]
            dist_names = ['x1', 'x2', 'x1\'', 'x2\'', 'Δs']
            for d_val, name in zip(distances, dist_names):
                if d_val < 0:
                    logs.append(f"WARNING: {name}={d_val} is negative (check sign conventions)")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb an input parameter and check trends match physical expectations
            # Choose a1 (acceleration) for perturbation
            original_a1 = self.a1
            perturbed_a1 = original_a1 * 1.1  # Increase by 10%
            
            # Recalculate t1 and t2 using formulas
            # t1 = vB / a1
            t1_original = self.vB / original_a1
            t1_perturbed = self.vB / perturbed_a1
            
            # t2 = 2 * vB / a1
            t2_original = 2 * self.vB / original_a1
            t2_perturbed = 2 * self.vB / perturbed_a1
            
            # According to theorem: if acceleration increases, catch-up time should decrease
            if not (t1_perturbed < t1_original and t2_perturbed < t2_original):
                logs.append(f"FAIL: Monotonicity violated for a1 perturbation")
                logs.append(f"  Original: t1={t1_original}, t2={t2_original}")
                logs.append(f"  Perturbed (+10% a1): t1={t1_perturbed}, t2={t2_perturbed}")
                success = False
            else:
                logs.append(f"PASS: Monotonicity holds for a1 perturbation")
                logs.append(f"  As a1 increases, t1 and t2 decrease (as expected)")
                
            # Restore original value
            self.a1 = original_a1
            
            # Test another monotonic relationship: if vB increases, Δs should increase
            # Δs = (x1 + x1') - (x2 + x2') = [v^2/(2a1) + v^2/(2|a2|)] - [vB*(v/a1) + vB*(v/|a2|)]
            # = v^2*(1/(2a1) + 1/(2|a2|)) - vB*v*(1/a1 + 1/|a2|)
            # Derivative with respect to vB is negative: d(Δs)/dvB = -v*(1/a1 + 1/|a2|)
            # So if vB increases, Δs decreases
            original_vB = self.vB
            perturbed_vB = original_vB * 1.1
            
            # Calculate Δs for both cases
            term = self.v * (1/self.a1 + 1/abs(self.a2))
            delta_s_original = (self.v**2/(2*self.a1) + self.v**2/(2*abs(self.a2))) - original_vB * term
            delta_s_perturbed = (self.v**2/(2*self.a1) + self.v**2/(2*abs(self.a2))) - perturbed_vB * term
            
            if not (delta_s_perturbed < delta_s_original):
                logs.append(f"FAIL: Monotonicity violated for vB perturbation")
                logs.append(f"  Original: Δs={delta_s_original}")
                logs.append(f"  Perturbed (+10% vB): Δs={delta_s_perturbed}")
                success = False
            else:
                logs.append(f"PASS: Monotonicity holds for vB perturbation")
                logs.append(f"  As vB increases, Δs decreases (as expected)")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*60}")
            all_logs.append(f"Running {test_name}")
            all_logs.append(f"{'='*60}")
            
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRITICAL ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n{'='*60}")
        all_logs.append(f"SUMMARY: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        all_logs.append(f"{'='*60}")
        
        return overall_success, all_logs

# Example usage (for testing the checker itself):
if __name__ == "__main__":
    # Create a mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.a1 = 2.0
            self.vB = 10.0
            self.v = 24.0
            self.a2 = -6.0
            self.results = {
                't_1': 5.0,
                't_2': 10.0,
                'x_1': 144.0,
                'x_2': 120.0,
                "x_1'": 48.0,
                "x_2'": 40.0,
                'delta_s': 32.0,
                't': 3.2,
                't_all': 19.2
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nFinal success status: {success}")