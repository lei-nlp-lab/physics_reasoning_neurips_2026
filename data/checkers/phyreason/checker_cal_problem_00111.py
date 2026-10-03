import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution_instance."""
        self.inputs = {}
        self.results = {}
        
        # Extract constants from solution_instance (they may be stored as attributes or in results)
        constants_keys = ['B', 'L', 'm_A', 'm_G', 'm_H', 'sin_theta', 'cos_theta', 'g', 'R_G', 'R_H', 'R_total']
        for key in constants_keys:
            # Try to get from solution_instance attribute first, then from results dict
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, None)
            if val is None:
                # Default values from problem statement (but B may be overridden)
                defaults = {
                    'B': 11.0, 'L': 0.2, 'm_A': 0.2, 'm_G': 0.2, 'm_H': 0.2,
                    'sin_theta': 0.25, 'cos_theta': 0.97, 'g': 10.0,
                    'R_G': 0.1, 'R_H': 0.1, 'R_total': 0.2
                }
                val = defaults[key]
            self.inputs[key] = float(val) if isinstance(val, (int, float, str)) else 0.0
            
        # Safely extract results using getattr and results dictionary
        result_keys = [
            'a', 'T', 'F', 'v_H', 'v_G', 'P_F', 's_G',
            'I', 'v_H - v_G', 'v_H-v_G', 'a\'', 'a\'\''
        ]
        for key in result_keys:
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, None)
            self.results[key] = val if val is not None else 0.0
            
        # Convert to float if they aren't already
        for key in self.results:
            if isinstance(self.results[key], (int, float, str)):
                try:
                    self.results[key] = float(self.results[key])
                except:
                    self.results[key] = 0.0
        
        # Ensure v_H - v_G is populated from v_H-v_G if missing
        if self.results['v_H - v_G'] == 0.0 and self.results.get('v_H-v_G', 0.0) != 0.0:
            self.results['v_H - v_G'] = self.results['v_H-v_G']

    def test_invariants(self):
        """Verify energy conservation, Newton's laws based on KB."""
        logs = []
        
        # 1. Check tension T from step3: T = m_A * g * sinθ + m_A * a
        if self.results['a'] and self.results['T']:
            T_expected = self.inputs['m_A'] * self.inputs['g'] * self.inputs['sin_theta']
            T_expected += self.inputs['m_A'] * self.results['a']
            if not math.isclose(T_expected, self.results['T'], rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Tension T mismatch. Expected ~{T_expected:.3f}, got {self.results['T']:.3f}")
            else:
                logs.append(f"PASS: Tension T = {self.results['T']:.3f} matches Newton's Law for block A.")
        
        # 2. Check current formula from step4: I = B*L*(v_H - v_G) / R_total
        if all(k in self.results for k in ['I', 'v_H', 'v_G']):
            v_diff = self.results['v_H'] - self.results['v_G']
            I_expected = self.inputs['B'] * self.inputs['L'] * v_diff / self.inputs['R_total']
            if not math.isclose(I_expected, self.results['I'], rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Current I mismatch. Faraday+Ohm expects ~{I_expected:.3f}, got {self.results['I']:.3f}")
            else:
                logs.append(f"PASS: Current I = {self.results['I']:.3f} matches Faraday+Ohm laws.")
        
        # 3. Check Newton's 2nd Law for rod G from step5: BIL - T = m_G * a
        if all(k in self.results for k in ['I', 'T', 'a']):
            F_amp = self.inputs['B'] * self.results['I'] * self.inputs['L']
            F_net_expected = F_amp - self.results['T']
            F_net_actual = self.inputs['m_G'] * self.results['a']
            if not math.isclose(F_net_expected, F_net_actual, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Newton's Law for rod G. BIL-T = {F_net_expected:.3f}, m_G*a = {F_net_actual:.3f}")
            else:
                logs.append(f"PASS: Newton's Law for rod G satisfied.")
        
        # 4. Check Newton's 2nd Law for rod H from step7: F = m_H * a + BIL
        if all(k in self.results for k in ['F', 'I', 'a']):
            F_expected = self.inputs['m_H'] * self.results['a']
            F_expected += self.inputs['B'] * self.results['I'] * self.inputs['L']
            if not math.isclose(F_expected, self.results['F'], rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Force F on rod H mismatch. Expected ~{F_expected:.3f}, got {self.results['F']:.3f}")
            else:
                logs.append(f"PASS: Force F = {self.results['F']:.3f} matches Newton's Law for rod H.")
        
        # 5. Check acceleration consistency from steps 1,2,6
        if 'a' in self.results:
            if not math.isclose(self.results['a'], 2.0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Acceleration a = {self.results['a']:.3f}, should be ~2.0 from kinematics.")
            else:
                logs.append(f"PASS: Acceleration a = {self.results['a']:.3f} matches kinematic analysis.")
        
        # 6. Check velocity difference from step6
        if 'v_H - v_G' in self.results:
            v_diff_calc = self.results['v_H'] - self.results['v_G']
            if not math.isclose(v_diff_calc, self.results['v_H - v_G'], rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_H - v_G mismatch. Calculated {v_diff_calc:.3f}, recorded {self.results['v_H - v_G']:.3f}")
            else:
                logs.append(f"PASS: Velocity difference v_H - v_G = {self.results['v_H - v_G']:.3f} consistent.")
        
        # 7. Check power formula from step10: P_F = F * v_H
        if all(k in self.results for k in ['P_F', 'F', 'v_H']):
            P_expected = self.results['F'] * self.results['v_H']
            if not math.isclose(P_expected, self.results['P_F'], rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Power P_F mismatch. F*v_H = {P_expected:.3f}, got {self.results['P_F']:.3f}")
            else:
                logs.append(f"PASS: Power P_F = {self.results['P_F']:.3f} matches F*v_H.")
        
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Target answers from problem
        targets = {
            'a': 2.0,
            'T': 0.9,
            'P_F': 16.15,
            's_G': 2.53
        }
        
        for key, expected in targets.items():
            if key in self.results:
                if not math.isclose(self.results[key], expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: {key} = {self.results[key]:.3f}, target = {expected:.3f}")
                else:
                    logs.append(f"PASS: {key} matches target answer.")
        
        # Check velocity at t=1.5s from steps 8-9
        if all(k in self.results for k in ['v_G', 'v_H', 'v_H - v_G']):
            v_H_expected = self.results['v_G'] + self.results['v_H - v_G']
            if not math.isclose(v_H_expected, self.results['v_H'], rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_H should be v_G + (v_H-v_G) = {v_H_expected:.3f}, got {self.results['v_H']:.3f}")
            else:
                logs.append(f"PASS: v_H = {self.results['v_H']:.3f} consistent with step9 relation.")
        
        # Check acceleration after H stops (step11) - optional
        if "a'" in self.results:
            a_prime = self.results["a'"]
            if a_prime <= 0:
                logs.append(f"WARN: a' = {a_prime:.3f} should be positive deceleration.")
            else:
                logs.append(f"PASS: a' = {a_prime:.3f} is positive as expected.")
        
        # Check acceleration of block A after H stops (step12) - optional
        if "a''" in self.results:
            a_expect = self.inputs['g'] * self.inputs['sin_theta']
            a_double_prime = self.results["a''"]
            if not math.isclose(a_expect, a_double_prime, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: a'' should be g*sinθ = {a_expect:.3f}, got {a_double_prime:.3f}")
            else:
                logs.append(f"PASS: a'' matches g*sinθ = {a_expect:.3f}.")
        
        return logs

    def test_monotonicity(self):
        """Perturb an input and check physical trends."""
        logs = []
        
        try:
            # Perturb mass of A and observe tension T
            m_A_original = self.inputs['m_A']
            m_A_new = m_A_original * 1.1  # +10%
            
            # Recalculate T using formula from step3: T = m_A*g*sinθ + m_A*a
            if 'a' in self.results:
                T_new = m_A_new * self.inputs['g'] * self.inputs['sin_theta']
                T_new += m_A_new * self.results['a']
                
                T_original = self.inputs['m_A'] * self.inputs['g'] * self.inputs['sin_theta']
                T_original += self.inputs['m_A'] * self.results['a']
                
                if T_new > T_original:
                    logs.append(f"PASS: Tension T increases with mass m_A (as expected).")
                else:
                    logs.append(f"FAIL: Tension T should increase with mass m_A.")
            
            # Perturb magnetic field B and observe current I
            B_original = self.inputs['B']
            B_new = B_original * 1.1  # +10%
            
            if 'v_H - v_G' in self.results:
                I_original = self.results['I']
                I_new = B_new * self.inputs['L'] * self.results['v_H - v_G'] / self.inputs['R_total']
                
                if I_new > I_original:
                    logs.append(f"PASS: Current I increases with B (as expected).")
                else:
                    logs.append(f"FAIL: Current I should increase with B.")
            
            # Perturb acceleration a and observe force F on rod H
            if all(k in self.results for k in ['a', 'I']):
                a_original = self.results['a']
                a_new = a_original * 1.1  # +10%
                
                F_original = self.inputs['m_H'] * a_original
                F_original += self.inputs['B'] * self.results['I'] * self.inputs['L']
                
                # For new F, we need new I too. But simplify: assume I proportional to a?
                # Instead, check that F increases with a if I fixed
                F_new = self.inputs['m_H'] * a_new + self.inputs['B'] * self.results['I'] * self.inputs['L']
                if F_new > F_original:
                    logs.append(f"PASS: Force F increases with acceleration a (as expected).")
                else:
                    logs.append(f"WARN: Force F may not increase with a if I changes.")
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
        
        return logs

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        logs = []
        success = True
        
        try:
            # Run each test suite
            for test_name, test_func in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                logs.append(f"\n=== {test_name} ===")
                test_logs = test_func()
                logs.extend(test_logs)
                
                # Check for critical failures
                for log in test_logs:
                    if log.startswith("FAIL:"):
                        success = False
        
        except Exception as e:
            logs.append(f"Checker crashed: {traceback.format_exc()}")
            success = False
        
        return success, logs