import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.inputs = {}
        self.results = {}
        
        # Known constants from problem statement
        self.inputs['v1'] = 16.0
        self.inputs['a1'] = 2.0
        self.inputs['v2'] = 4.0
        self.inputs['a2'] = 6.0
        self.inputs['t_3s'] = 3.0
        self.inputs['initial_sep'] = 18.0
        
        # Extract results from solution_instance
        result_keys = [
            'x_1', 't_1', 'v_a', 'x_A', 'x_B', 'Deltax', 'Deltax_m',
            'v_A1', 'v_B1', 't_A', 's_0', 'Deltax_0', 't_0', 'v_all',
            'x_A-x_B', 'a_0'
        ]
        
        for key in result_keys:
            # Try to get from results dict
            if hasattr(solution_instance, 'results'):
                if isinstance(solution_instance.results, dict):
                    self.results[key] = solution_instance.results.get(key, None)
                else:
                    self.results[key] = getattr(solution_instance.results, key, None)
            else:
                self.results[key] = getattr(solution_instance, key, None)
        
        # Store expected final answers
        self.expected_answers = [
            "39 m",
            "27 m",
            "a0 >= 6 m/s^2"
        ]
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            v1, a1, v2, a2, t3 = self.inputs['v1'], self.inputs['a1'], self.inputs['v2'], self.inputs['a2'], self.inputs['t_3s']
            
            # 1. Check velocities at t=3s
            v_A3_expected = v1 - a1 * t3
            if self.results['v_A1'] is not None:
                if not math.isclose(v_A3_expected, float(self.results['v_A1']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_A at t=3s expected {v_A3_expected}, got {self.results['v_A1']}")
                    success = False
            
            v_B3_expected = v2 + a2 * t3
            if self.results['v_B1'] is not None:
                if not math.isclose(v_B3_expected, float(self.results['v_B1']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_B at t=3s expected {v_B3_expected}, got {self.results['v_B1']}")
                    success = False
            
            # 2. Check displacement of A in first 3s
            x_A3_form1 = v1 * t3 - 0.5 * a1 * t3**2
            x_A3_form2 = ((v1 + v_A3_expected) / 2) * t3
            if not math.isclose(x_A3_form1, x_A3_form2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinematic inconsistency for A displacement: {x_A3_form1} vs {x_A3_form2}")
                success = False
            if self.results['x_1'] is not None:
                if not math.isclose(x_A3_form1, float(self.results['x_1']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x_A(3) expected {x_A3_form1}, got {self.results['x_1']}")
                    success = False
            
            # 3. Check velocity equality time t1
            t1_expected = (v1 - v2) / (a1 + a2)
            if self.results['t_1'] is not None:
                if not math.isclose(t1_expected, float(self.results['t_1']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t1 expected {t1_expected}, got {self.results['t_1']}")
                    success = False
            
            # 4. Check velocities at t1
            v_at_t1 = v1 - a1 * t1_expected
            if self.results['v_a'] is not None:
                if not math.isclose(v_at_t1, float(self.results['v_a']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v at t1 expected {v_at_t1}, got {self.results['v_a']}")
                    success = False
            
            # 5. Check maximum distance in first 3s
            delta_x_at_t1 = (v1 - v2) / 2 * t1_expected
            max_dist_expected = self.inputs['initial_sep'] + delta_x_at_t1
            if self.results['Deltax_m'] is not None:
                if not math.isclose(max_dist_expected, float(self.results['Deltax_m']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: max distance expected {max_dist_expected}, got {self.results['Deltax_m']}")
                    success = False
            
            # 6. Check collision avoidance condition
            a0_val = self.results.get('a_0')
            if a0_val is not None:
                a0_val = float(a0_val)
                s0 = float(self.results.get('s_0', 18.0))
                threshold = 72.0 / s0 + 2.0
                if a0_val < threshold - 1e-2:
                    logs.append(f"FAIL: collision condition violated: a0={a0_val} < {threshold}")
                    success = False
                else:
                    logs.append(f"PASS: a0={a0_val} meets threshold {threshold}")
            
            if success:
                logs.append("PASS: All kinematic invariants satisfied")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # 1. Check step_1 result (39 m)
            if self.results.get('x_1') is not None:
                if not math.isclose(float(self.results['x_1']), 39.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x_1={self.results['x_1']} ≠ 39 m")
                    success = False
            
            # 2. Check step_6 result (27 m)
            if self.results.get('Deltax_m') is not None:
                if not math.isclose(float(self.results['Deltax_m']), 27.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Deltax_m={self.results['Deltax_m']} ≠ 27 m")
                    success = False
            
            # 3. Check step_15 result (a0 >= 6)
            a0_val = self.results.get('a_0')
            if a0_val is not None:
                a0_val = float(a0_val)
                if a0_val < 6.0 - 1e-2:
                    logs.append(f"FAIL: a0={a0_val} < 6 m/s²")
                    success = False
            
            # 4. Consistency of t_A with v_A1 and a1
            if all(k in self.results for k in ['t_A', 'v_A1']):
                t_A_expected = float(self.results['v_A1']) / self.inputs['a1']
                if not math.isclose(t_A_expected, float(self.results['t_A']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_A expected {t_A_expected}, got {self.results['t_A']}")
                    success = False
            
            # 5. Check that Deltax equals (v1 - v2)/2 * t_1
            if all(k in self.results for k in ['Deltax', 't_1']):
                v1, v2 = self.inputs['v1'], self.inputs['v2']
                t1 = float(self.results['t_1'])
                Deltax_expected = (v1 - v2) / 2 * t1
                if not math.isclose(Deltax_expected, float(self.results['Deltax']), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Deltax expected {Deltax_expected}, got {self.results['Deltax']}")
                    success = False
            
            if success:
                logs.append("PASS: All trace consistency checks passed")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Extract baseline
            s0 = float(self.results.get('s_0', 18.0))
            a0_min_baseline = 72.0 / s0 + 2.0
            
            # Perturb s0 upward by 10%
            s0_perturbed = s0 * 1.1
            a0_min_perturbed = 72.0 / s0_perturbed + 2.0
            
            # Check monotonicity: larger s0 should require smaller a0_min
            if a0_min_perturbed >= a0_min_baseline + 1e-2:
                logs.append(f"FAIL: Monotonicity violated: s0↑ ({s0}→{s0_perturbed}) but a0_min ({a0_min_baseline}→{a0_min_perturbed}) not ↓")
                success = False
            else:
                logs.append(f"PASS: Monotonicity holds: s0↑ → a0_min↓ ({a0_min_baseline}→{a0_min_perturbed})")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Kinematic Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                for log in logs:
                    all_logs.append(log)
            except Exception as e:
                all_logs.append(f"CRASH: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs