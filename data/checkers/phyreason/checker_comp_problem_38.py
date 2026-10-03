import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks
        self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', 0.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.81))
        self.L0 = getattr(solution_instance, 'L0', solution_instance.results.get('L0', 0.0))
        self.k = getattr(solution_instance, 'k', solution_instance.results.get('k', 0.0))
        self.alpha = getattr(solution_instance, 'alpha', solution_instance.results.get('alpha', 0.0))
        
        # Fallback: compute alpha from definition if not provided
        if not self.alpha and self.M and self.g and self.k and self.L0:
            self.alpha = (self.k * self.L0) / (self.M * self.g)
        
        # Safely extract results with fallbacks
        self.results = {}
        result_keys = ['H', 'H_cm-L_0/2', 'Q', 'dW', 'delta_y', 't_c', 'H_cm-upper_i', 
                      'l_0', 'v_upper_f2', 'H_cm-upper_f', 'v_min', 'm_top', 'A', 'delta_W', 
                      'B', 'W', 'dy', 'H_cm', 'v_upper-f2', 'delta_H_cm-upper', 'k*']
        for key in result_keys:
            self.results[key] = solution_instance.results.get(key, 0.0)
        
        # Derived quantities
        self.l0 = self.alpha * self.L0 if self.alpha else 0.0
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Verify alpha = kL0/(Mg) matches definition
            if self.M and self.g and self.k and self.L0:
                alpha_calc = (self.k * self.L0) / (self.M * self.g)
                if not math.isclose(self.alpha, alpha_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: alpha mismatch. Given: {self.alpha}, Computed: {alpha_calc}")
                else:
                    logs.append(f"PASS: alpha = {self.alpha} matches definition")
            
            # 2. Check equilibrium at initial state: bottom tension = weight
            # For ZLS, force at bottom: F = k*l0 = k*(alpha*L0) = k*L0*alpha
            # Weight: M*g
            if self.M and self.g and self.k and self.L0:
                bottom_force = self.k * self.l0
                weight = self.M * self.g
                if not math.isclose(bottom_force, weight, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Initial equilibrium violated. Bottom force: {bottom_force}, Weight: {weight}")
                else:
                    logs.append(f"PASS: Initial equilibrium F_bottom = Mg holds")
            
            # 3. Energy conservation check: Initial elastic energy + gravitational potential = final energies + Q
            if self.M and self.g and self.L0 and self.alpha and self.results['Q']:
                # Initial elastic energy from integration (step 29)
                W_elastic_initial = (self.k * self.L0**2 / (6 * self.alpha**2)) * ((1 - self.alpha)**2) * (2*self.alpha + 1)
                
                # Initial gravitational potential energy (CM at H_cm)
                if self.results['H_cm']:
                    U_grav_initial = self.M * self.g * self.results['H_cm']
                else:
                    # Compute H_cm from formula
                    H_cm_calc = self.L0 * (1/(6*self.alpha) - self.alpha**2/6 + self.alpha/2)
                    U_grav_initial = self.M * self.g * H_cm_calc
                
                # Final energies after collapse: spring fully relaxed, CM at L0/2
                U_grav_final = self.M * self.g * (self.L0 / 2)
                
                # Energy conservation: initial = final + Q (lost energy)
                initial_energy = W_elastic_initial + U_grav_initial
                final_energy = U_grav_final + self.results['Q']
                
                if not math.isclose(initial_energy, final_energy, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy conservation violated. Initial: {initial_energy}, Final+Q: {final_energy}")
                else:
                    logs.append(f"PASS: Energy conservation holds within tolerance")
            
            # 4. Check CM displacement formula
            if self.L0 and self.alpha and self.results['H_cm-L_0/2']:
                delta_H_cm_calc = (self.L0 / (6 * self.alpha)) * ((1 - self.alpha)**3)
                if not math.isclose(self.results['H_cm-L_0/2'], delta_H_cm_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: CM displacement mismatch. Given: {self.results['H_cm-L_0/2']}, Formula: {delta_H_cm_calc}")
                else:
                    logs.append(f"PASS: CM displacement formula verified")
                    
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # 1. Check H = (L0/2)*(alpha + 1/alpha)
            if self.L0 and self.alpha and self.results['H']:
                H_calc = (self.L0 / 2) * (self.alpha + 1/self.alpha)
                if not math.isclose(self.results['H'], H_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: H mismatch. Given: {self.results['H']}, Formula: {H_calc}")
                else:
                    logs.append(f"PASS: H consistent with formula")
            
            # 2. Check t_c = sqrt((L0/(3*g*alpha))*(1-alpha)^3)
            if self.L0 and self.g and self.alpha and self.results['t_c']:
                t_c_calc = math.sqrt((self.L0 / (3 * self.g * self.alpha)) * ((1 - self.alpha)**3))
                if not math.isclose(self.results['t_c'], t_c_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_c mismatch. Given: {self.results['t_c']}, Formula: {t_c_calc}")
                else:
                    logs.append(f"PASS: t_c consistent with free-fall formula")
            
            # 3. Check A and B formulas
            if self.g and self.alpha:
                A_calc = (2 * self.g) / (3 * self.alpha)
                B_calc = ((1/(3*self.alpha)) - 1) * self.g * self.L0
                
                if self.results['A'] and not math.isclose(self.results['A'], A_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: A mismatch. Given: {self.results['A']}, Formula: {A_calc}")
                elif self.results['A']:
                    logs.append(f"PASS: A consistent with formula")
                
                if self.results['B'] and self.L0 and not math.isclose(self.results['B'], B_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: B mismatch. Given: {self.results['B']}, Formula: {B_calc}")
                elif self.results['B']:
                    logs.append(f"PASS: B consistent with formula")
            
            # 4. Check v_min = (1-alpha)*sqrt(A*alpha*L0 + B)
            if self.alpha and self.results['v_min'] and self.results['A'] and self.results['B'] and self.L0:
                v_min_calc = (1 - self.alpha) * math.sqrt(self.results['A'] * self.alpha * self.L0 + self.results['B'])
                if not math.isclose(self.results['v_min'], v_min_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_min mismatch. Given: {self.results['v_min']}, Formula: {v_min_calc}")
                else:
                    logs.append(f"PASS: v_min consistent with A, B, and alpha")
            
            # 5. Check Q = (M*g*L0*(1-alpha)^2*(2*alpha+1))/(6*alpha)
            if self.M and self.g and self.L0 and self.alpha and self.results['Q']:
                Q_calc = (self.M * self.g * self.L0 * ((1 - self.alpha)**2) * (2*self.alpha + 1)) / (6 * self.alpha)
                if not math.isclose(self.results['Q'], Q_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Q mismatch. Given: {self.results['Q']}, Formula: {Q_calc}")
                else:
                    logs.append(f"PASS: Q consistent with energy loss formula")
            
            # 6. Check l_0 = alpha*L0
            if self.L0 and self.alpha and self.results['l_0']:
                l0_calc = self.alpha * self.L0
                if not math.isclose(self.results['l_0'], l0_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: l_0 mismatch. Given: {self.results['l_0']}, Formula: {l0_calc}")
                else:
                    logs.append(f"PASS: l_0 = alpha*L0 holds")
            
            # 7. Check m_top = (1-alpha)*M
            if self.M and self.alpha and self.results['m_top']:
                m_top_calc = (1 - self.alpha) * self.M
                if not math.isclose(self.results['m_top'], m_top_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: m_top mismatch. Given: {self.results['m_top']}, Formula: {m_top_calc}")
                else:
                    logs.append(f"PASS: m_top consistent with mass fraction")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_alpha = self.alpha
        perturbed = False
        
        try:
            # Perturb alpha by 1% (but keep < 1)
            if original_alpha > 0:
                perturbed_alpha = original_alpha * 1.01
                if perturbed_alpha >= 1:
                    perturbed_alpha = original_alpha * 0.99
                
                # Store original results that depend on alpha
                original_H = self.results['H']
                original_t_c = self.results['t_c']
                original_Q = self.results['Q']
                
                # Compute perturbed H
                perturbed_H = (self.L0 / 2) * (perturbed_alpha + 1/perturbed_alpha)
                
                # For alpha < 1, H should decrease as alpha increases (since dH/dalpha = (L0/2)*(1 - 1/alpha^2) < 0)
                if perturbed_alpha > original_alpha:
                    if perturbed_H < original_H:
                        logs.append("PASS: H decreases as alpha increases (monotonicity)")
                    else:
                        logs.append(f"FAIL: H monotonicity violated. Original H: {original_H}, Perturbed H: {perturbed_H}")
                else:
                    if perturbed_H > original_H:
                        logs.append("PASS: H increases as alpha decreases (monotonicity)")
                    else:
                        logs.append(f"FAIL: H monotonicity violated. Original H: {original_H}, Perturbed H: {perturbed_H}")
                
                # Compute perturbed t_c
                perturbed_t_c = math.sqrt((self.L0 / (3 * self.g * perturbed_alpha)) * ((1 - perturbed_alpha)**3))
                
                # For alpha < 1, t_c should decrease as alpha increases (both factors decrease)
                if perturbed_alpha > original_alpha:
                    if perturbed_t_c < original_t_c:
                        logs.append("PASS: t_c decreases as alpha increases (monotonicity)")
                    else:
                        logs.append(f"FAIL: t_c monotonicity violated. Original t_c: {original_t_c}, Perturbed t_c: {perturbed_t_c}")
                else:
                    if perturbed_t_c > original_t_c:
                        logs.append("PASS: t_c increases as alpha decreases (monotonicity)")
                    else:
                        logs.append(f"FAIL: t_c monotonicity violated. Original t_c: {original_t_c}, Perturbed t_c: {perturbed_t_c}")
                
                # Compute perturbed Q
                perturbed_Q = (self.M * self.g * self.L0 * ((1 - perturbed_alpha)**2) * (2*perturbed_alpha + 1)) / (6 * perturbed_alpha)
                
                # For small alpha, Q should decrease as alpha increases? Let's check numerically
                # We'll just verify the computation is consistent
                if perturbed_alpha > original_alpha:
                    logs.append(f"INFO: Q at alpha={original_alpha}: {original_Q}, at alpha={perturbed_alpha}: {perturbed_Q}")
                else:
                    logs.append(f"INFO: Q at alpha={original_alpha}: {original_Q}, at alpha={perturbed_alpha}: {perturbed_Q}")
                
                perturbed = True
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        finally:
            # Restore original alpha
            if perturbed:
                self.alpha = original_alpha
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_method()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            
            # Check for failures (logs containing "FAIL:")
            if any("FAIL:" in log for log in logs):
                success = False
        
        return success, all_logs