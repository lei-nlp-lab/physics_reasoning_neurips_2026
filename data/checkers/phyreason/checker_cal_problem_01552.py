import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with safe defaults
        self.inputs['m'] = getattr(solution_instance, 'm', 1.0)
        self.inputs['g'] = getattr(solution_instance, 'g', 9.8)
        self.inputs['R'] = getattr(solution_instance, 'R', 1.0)
        self.inputs['theta'] = getattr(solution_instance, 'theta', 53 * math.pi/180)
        self.inputs['k_electric'] = 3/4
        self.inputs['h_A'] = 12 * getattr(solution_instance, 'R', 1.0)
        
        # Extract results with safe fallbacks
        results_dict = getattr(solution_instance, 'results', {})
        
        # Map expected keys to actual keys in results_dict
        self.results['beta'] = results_dict.get('beta', 0.0)
        self.results['centripetal_D'] = results_dict.get('centripetal_D', results_dict.get('m*v_D^2/R', results_dict.get('m*v_D2/R', 0.0)))
        self.results['F_resultant'] = results_dict.get('F_resultant', results_dict.get('F', 0.0))
        self.results['h_min'] = results_dict.get('h_min', 0.0)
        self.results['N_prime'] = results_dict.get('N_prime', results_dict.get('N\'', 0.0))
        self.results['KE_D'] = results_dict.get('KE_D', results_dict.get('1/2mv_D^2', results_dict.get('1/2mv_D2', 0.0)))
        self.results['h_max'] = results_dict.get('h_max', 0.0)
        self.results['h_range'] = results_dict.get('h_range', results_dict.get('h', 0.0))
        self.results['N'] = results_dict.get('N', 0.0)
        self.results['delta_E'] = results_dict.get('delta_E', results_dict.get('ΔE', 0.0))
        
        # Use exact trigonometric values for consistency with problem statement
        # tan53° = 4/3, sin53° = 4/5, cos53° = 3/5
        self.tan_theta_exact = 4/3
        self.sin_theta_exact = 4/5
        self.cos_theta_exact = 3/5
        
    def test_invariants(self):
        logs = []
        try:
            m, g, R = self.inputs['m'], self.inputs['g'], self.inputs['R']
            k = self.inputs['k_electric']
            
            # 1. Energy loss ΔE - use exact tanθ = 4/3
            F_electric = k * m * g
            horiz_dist_AB = 12 * R / self.tan_theta_exact  # 12R / (4/3) = 9R
            delta_E_expected = F_electric * horiz_dist_AB  # (3/4)mg * 9R = (27/4)mgR
            
            if math.isclose(self.results['delta_E'], delta_E_expected, 
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Energy loss ΔE matches electric work calculation")
            else:
                logs.append(f"✗ Energy loss mismatch: got {self.results['delta_E']}, expected {delta_E_expected}")
            
            # 2. Newton's 2nd Law at point D
            if math.isclose(self.results['centripetal_D'], 
                           F_electric + self.results['N_prime'],
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Newton's 2nd Law satisfied at point D")
            else:
                logs.append(f"✗ Centripetal force mismatch at D: {self.results['centripetal_D']} ≠ {F_electric} + {self.results['N_prime']}")
            
            # 3. Newton's 3rd Law: N = N'
            if math.isclose(self.results['N'], self.results['N_prime'],
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Newton's 3rd Law satisfied (N = N')")
            else:
                logs.append(f"✗ Normal force mismatch: N={self.results['N']}, N'={self.results['N_prime']}")
            
            # 4. Resultant force magnitude check
            F_expected = m * g * math.sqrt(1 + k**2)
            if math.isclose(self.results['F_resultant'], F_expected,
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Resultant force magnitude correct")
            else:
                logs.append(f"✗ Resultant force mismatch: got {self.results['F_resultant']}, expected {F_expected}")
            
            # 5. Angle β check: tanβ = k = 3/4
            beta_expected = math.atan(k)
            if math.isclose(self.results['beta'], beta_expected,
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Angle β correct")
            else:
                logs.append(f"✗ Angle β mismatch: got {self.results['beta']}, expected {beta_expected}")
                
        except Exception as e:
            logs.append(f"✗ Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            m, g, R = self.inputs['m'], self.inputs['g'], self.inputs['R']
            k = self.inputs['k_electric']
            h_A = 12 * R
            
            # Use exact trigonometric values
            F_electric = k * m * g
            
            # 1. Check KE at D consistency with exact trig
            # KE_D = mg(12R - R) - F_electric * (12R/(4/3) + 2R + R)
            # = 11mgR - (3/4)mg * (9R + 3R) = 11mgR - (3/4)mg * 12R = 11mgR - 9mgR = 2mgR
            work_electric = F_electric * (12*R/self.tan_theta_exact + 2*R + R)
            KE_D_expected = m * g * (h_A - R) - work_electric
            
            if math.isclose(self.results['KE_D'], KE_D_expected,
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ KE at D consistent with work-energy theorem")
            else:
                logs.append(f"✗ KE_D mismatch: got {self.results['KE_D']}, expected {KE_D_expected}")
            
            # 2. Check centripetal force at D from KE_D
            v_D_sq = 2 * self.results['KE_D'] / m
            centripetal_expected = m * v_D_sq / R
            
            if math.isclose(self.results['centripetal_D'], centripetal_expected,
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Centripetal force at D consistent with KE_D")
            else:
                logs.append(f"✗ Centripetal force from KE_D mismatch: got {self.results['centripetal_D']}, expected {centripetal_expected}")
            
            # 3. Check N' from centripetal force
            N_prime_expected = self.results['centripetal_D'] - F_electric
            if math.isclose(self.results['N_prime'], N_prime_expected,
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ N' consistent with centripetal force")
            else:
                logs.append(f"✗ N' mismatch: got {self.results['N_prime']}, expected {N_prime_expected}")
            
            # 4. Check h_min calculation consistency
            # Use exact values: β = arctan(3/4), cosβ = 4/5, sinβ = 3/5
            beta = math.atan(k)
            cos_beta_exact = 4/5
            sin_beta_exact = 3/5
            
            # Required centripetal force: F_resultant = mg√(1+k²) = (5/4)mg
            v1_sq_required = self.results['F_resultant'] * R / m
            KE_M_required = 0.5 * m * v1_sq_required
            
            # Energy from h_min to point M with exact trig:
            height_diff = self.results['h_min'] - R - R * cos_beta_exact
            horiz_dist = self.results['h_min']/self.tan_theta_exact + 2*R + R*sin_beta_exact
            energy_balance = m*g*height_diff - F_electric*horiz_dist
            
            if math.isclose(energy_balance, KE_M_required,
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ h_min satisfies critical energy condition")
            else:
                logs.append(f"✗ h_min energy balance mismatch: {energy_balance} vs {KE_M_required}")
                
        except Exception as e:
            logs.append(f"✗ Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_k = self.inputs['k_electric']
        
        try:
            m, g, R = self.inputs['m'], self.inputs['g'], self.inputs['R']
            
            # Perturb electric force ratio
            perturbed_k = original_k * 1.1
            
            # Save original N'
            original_N_prime = self.results['N_prime']
            
            # Recalculate N' for perturbed k
            # From exact derivation: N' = mg * [22 - 25k] where 25 comes from:
            # T = 12/(4/3) + 3 = 9 + 3 = 12, so 2T + 1 = 25
            perturbed_N_prime = m*g * (22 - perturbed_k * 25)
            
            # Check trend
            if original_N_prime > perturbed_N_prime:
                logs.append("✓ Monotonicity: N' decreases as electric force increases")
            else:
                logs.append(f"✗ Monotonicity violation: N' should decrease but got {original_N_prime} -> {perturbed_N_prime}")
            
            # Check h_min trend
            original_h_min = self.results['h_min']
            
            # For perturbed k, estimate h_min using exact formulas
            beta_perturbed = math.atan(perturbed_k)
            cos_beta = 4/5  # Using exact value for consistency
            sin_beta = 3/5
            F_coeff_perturbed = math.sqrt(1 + perturbed_k**2)
            KE_M_required = 0.5 * m * g * R * F_coeff_perturbed
            
            # Solve for h_min: (h_min - R - R*cosβ) - k*(h_min/(4/3) + 2R + R*sinβ) = 0.5*F_coeff
            numerator = KE_M_required + m*g*R*(1 + cos_beta) + perturbed_k*m*g*(2*R + R*sin_beta)
            denominator = m*g - perturbed_k*m*g/self.tan_theta_exact
            if abs(denominator) > 1e-10:
                perturbed_h_min = numerator / denominator
            else:
                perturbed_h_min = float('inf')
            
            if perturbed_h_min > original_h_min:
                logs.append("✓ Monotonicity: h_min increases with electric force")
            else:
                logs.append(f"✗ Monotonicity: h_min trend unexpected: {original_h_min} -> {perturbed_h_min}")
                
        except Exception as e:
            logs.append(f"✗ Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run individual tests
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_method()
            if any(log.startswith("✗") for log in logs):
                success = False
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
        
        # Final target answer checks
        all_logs.append("=== Target Answers ===")
        try:
            m, g, R = self.inputs['m'], self.inputs['g'], self.inputs['R']
            
            # Check ΔE = (27/4) mgR
            target_delta_E = (27/4) * m * g * R
            if math.isclose(self.results['delta_E'], target_delta_E,
                           rel_tol=1e-2, abs_tol=1e-2):
                all_logs.append("✓ ΔE = (27/4) mgR matches target")
            else:
                all_logs.append(f"✗ ΔE mismatch: got {self.results['delta_E']}, expected {target_delta_E}")
                success = False
            
            # Check N = (13/4) mg
            target_N = (13/4) * m * g
            if math.isclose(self.results['N'], target_N,
                           rel_tol=1e-2, abs_tol=1e-2):
                all_logs.append("✓ N = (13/4) mg matches target")
            else:
                all_logs.append(f"✗ N mismatch: got {self.results['N']}, expected {target_N}")
                success = False
            
            # Check h condition with tolerance
            # Target: h ≥ 10R OR 24/7 R ≤ h ≤ 40/7 R
            h_value = self.results['h_range']
            
            # Use math.isclose for bounds comparison
            condition1 = h_value >= 10*R or math.isclose(h_value, 10*R, rel_tol=1e-2)
            
            lower_bound = (24/7) * R
            upper_bound = (40/7) * R
            condition2 = (h_value >= lower_bound or math.isclose(h_value, lower_bound, rel_tol=1e-2)) and \
                        (h_value <= upper_bound or math.isclose(h_value, upper_bound, rel_tol=1e-2))
            
            if condition1 or condition2:
                all_logs.append("✓ h condition satisfied")
            else:
                all_logs.append(f"✗ h condition failed: h={h_value}, expected h≥{10*R} or {lower_bound}≤h≤{upper_bound}")
                success = False
                
        except Exception as e:
            all_logs.append(f"✗ Error in target checks: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs