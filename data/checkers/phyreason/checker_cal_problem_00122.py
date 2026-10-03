import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Extract inputs with defaults (some might be in results)
        self.m0 = getattr(solution_instance, 'm0', self.inputs.get('m0', self.results.get('m0', 1.0)))
        self.d = getattr(solution_instance, 'd', self.inputs.get('d', self.results.get('d', 0.01)))
        self.h1 = getattr(solution_instance, 'h1', self.inputs.get('h1', self.results.get('h1', 0.001)))
        self.h2 = getattr(solution_instance, 'h2', self.inputs.get('h2', self.results.get('h2', 0.0015)))
        self.t = getattr(solution_instance, 't', self.inputs.get('t', self.results.get('t', 1.0)))
        self.U = getattr(solution_instance, 'U', self.inputs.get('U', self.results.get('U', 100.0)))
        self.g = getattr(solution_instance, 'g', self.inputs.get('g', self.results.get('g', 9.8)))
        
        # Extract results with safe access using the key mapping
        self.f = self.results.get('f', 0.0)
        self.q_A = self.results.get('q_A', 0.0)
        self.v_2 = self.results.get('v_2', 0.0)  # velocity of B
        self.q = self.results.get('q', 0.0)      # charge of B
        self.delta_E_p = self.results.get('delta_E_p', 0.0)
        self.W_e = self.results.get('W_e', 0.0)
        self.F_E = self.results.get('F_E', 0.0)
        self.k = self.results.get('k', 0.0)
        self.v_1 = self.results.get('v_1', 0.0)  # velocity of A and B initially
        self.v_all = self.results.get('v_all', 0.0)   # combined droplet velocity in case 1 (from momentum)
        self.v_2_prime = self.results.get('v_2\'', self.results.get('v_2\'\'', 0.0))  # final uniform velocity magnitude
        self.F_prime = self.results.get('F\'', 0.0)   # electric force on combined droplet
        
        # For convenience, compute derived quantities
        self.v1_computed = self.h1 / self.t if self.t != 0 else 0.0
        self.v2_computed = self.h2 / self.t if self.t != 0 else 0.0
        self.k_computed = (self.m0**(2/3) * self.g * self.t) / self.h1 if self.h1 != 0 else 0.0
        self.q_computed = (self.m0 * self.g * self.d * (self.h1 + self.h2)) / (self.h1 * self.U) if self.h1 != 0 and self.U != 0 else 0.0
        self.delta_E_p_computed = - (self.m0 * self.g * self.h2 * (self.h1 + self.h2)) / self.h1 if self.h1 != 0 else 0.0
        self.F_prime_computed = (self.m0 * self.g * (self.h1 + self.h2)) / self.h1 if self.h1 != 0 else 0.0
        
        # Determine case for combined droplet
        if self.F_prime_computed > 2 * self.m0 * self.g:
            self.v_combined_computed = (self.h2 - self.h1) / (2**(1/3) * self.t) if self.t != 0 else 0.0
            self.direction_combined = "upwards"
        else:
            self.v_combined_computed = (self.h1 - self.h2) / (2**(1/3) * self.t) if self.t != 0 else 0.0
            self.direction_combined = "downwards"
    
    def fuzzy_equal(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        logs = []
        try:
            # 1. Force equilibrium for initial falling (step_2 and step_3)
            if not self.fuzzy_equal(self.f, self.m0 * self.g):
                logs.append(f"FAIL: Air resistance f = {self.f} does not equal m0*g = {self.m0 * self.g}")
            else:
                logs.append("PASS: Initial force equilibrium f = m0*g")
            
            f_from_formula = self.k * self.m0**(1/3) * self.v_1
            if not self.fuzzy_equal(self.f, f_from_formula):
                logs.append(f"FAIL: f from drag formula {f_from_formula} does not match stored f {self.f}")
            else:
                logs.append("PASS: Drag formula f = k * m0^(1/3) * v_1 holds")
            
            # 2. Electric force equilibrium for B (step_6)
            left = self.U * self.q / self.d if self.d != 0 else 0.0
            right = self.m0 * self.g + self.k * self.m0**(1/3) * self.v_2
            if not self.fuzzy_equal(left, right):
                logs.append(f"FAIL: Electric force equilibrium fails: left={left}, right={right}")
            else:
                logs.append("PASS: Electric force equilibrium for B holds")
            
            # 3. Work-energy relation (step_8 and step_9): ΔE_p = -W_e
            if not self.fuzzy_equal(self.delta_E_p, -self.W_e):
                logs.append(f"FAIL: ΔE_p = {self.delta_E_p} != -W_e = {-self.W_e}")
            else:
                logs.append("PASS: ΔE_p = -W_e holds")
            
            W_e_computed = (self.U / self.d) * self.q * self.h2 if self.d != 0 else 0.0
            if not self.fuzzy_equal(self.W_e, W_e_computed):
                logs.append(f"FAIL: W_e computed {W_e_computed} != stored {self.W_e}")
            else:
                logs.append("PASS: W_e = (U/d)*q*h2 holds")
            
            # 4. Conservation of momentum for collision (step_13 and step_16)
            if self.h2 > self.h1:
                p_initial = self.m0 * self.v_2 - self.m0 * self.v_1
                p_final = 2 * self.m0 * self.v_all
                if not self.fuzzy_equal(p_initial, p_final):
                    logs.append(f"FAIL: Momentum conservation case 1: p_initial={p_initial}, p_final={p_final}")
                else:
                    logs.append("PASS: Momentum conservation for case 1 (h2>h1)")
            else:
                p_initial = self.m0 * (self.v_2 - self.v_1)
                p_final = -2 * self.m0 * self.v_2_prime
                if not self.fuzzy_equal(p_initial, p_final):
                    logs.append(f"FAIL: Momentum conservation case 2: p_initial={p_initial}, p_final={p_final}")
                else:
                    logs.append("PASS: Momentum conservation for case 2 (h1>h2)")
            
            # 5. Force equilibrium for combined droplet (step_14 and step_17)
            # Corrected: For upward case, equilibrium uses v_2_prime (final uniform velocity), not v_all.
            # For downward case, equilibrium uses v_2_prime as well.
            if self.h2 > self.h1:
                left = self.F_prime
                right = 2 * self.m0 * self.g + self.k * (2 * self.m0)**(1/3) * self.v_2_prime
                if not self.fuzzy_equal(left, right):
                    logs.append(f"FAIL: Force equilibrium for combined droplet (upward): left={left}, right={right}")
                else:
                    logs.append("PASS: Force equilibrium for combined droplet (upward)")
            else:
                left = 2 * self.m0 * self.g
                right = self.F_prime + self.k * (2 * self.m0)**(1/3) * self.v_2_prime
                if not self.fuzzy_equal(left, right):
                    logs.append(f"FAIL: Force equilibrium for combined droplet (downward): left={left}, right={right}")
                else:
                    logs.append("PASS: Force equilibrium for combined droplet (downward)")
                    
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            if not self.fuzzy_equal(self.v_1, self.v1_computed):
                logs.append(f"FAIL: v_1 = {self.v_1} != h1/t = {self.v1_computed}")
            else:
                logs.append("PASS: v_1 = h1/t")
            
            if not self.fuzzy_equal(self.k, self.k_computed):
                logs.append(f"FAIL: k = {self.k} != computed k = {self.k_computed}")
            else:
                logs.append("PASS: k formula holds")
            
            if not self.fuzzy_equal(self.v_2, self.v2_computed):
                logs.append(f"FAIL: v_2 = {self.v_2} != h2/t = {self.v2_computed}")
            else:
                logs.append("PASS: v_2 = h2/t")
            
            if not self.fuzzy_equal(self.q, self.q_computed):
                logs.append(f"FAIL: q = {self.q} != computed q = {self.q_computed}")
            else:
                logs.append("PASS: q formula holds")
            
            delta_E_p_from_formula = - (self.U / self.d) * self.q * self.h2 if self.d != 0 else 0.0
            if not self.fuzzy_equal(self.delta_E_p, delta_E_p_from_formula):
                logs.append(f"FAIL: ΔE_p = {self.delta_E_p} != computed ΔE_p = {delta_E_p_from_formula}")
            else:
                logs.append("PASS: ΔE_p formula via work holds")
            
            if not self.fuzzy_equal(self.delta_E_p, self.delta_E_p_computed):
                logs.append(f"FAIL: ΔE_p = {self.delta_E_p} != derived expression {self.delta_E_p_computed}")
            else:
                logs.append("PASS: ΔE_p matches derived expression")
            
            if not self.fuzzy_equal(self.F_prime, self.F_prime_computed):
                logs.append(f"FAIL: F' = {self.F_prime} != computed F' = {self.F_prime_computed}")
            else:
                logs.append("PASS: F' formula holds")
            
            if self.h2 > self.h1:
                if not self.fuzzy_equal(self.v_2_prime, self.v_combined_computed):
                    logs.append(f"FAIL: v_2' = {self.v_2_prime} != computed {self.v_combined_computed} for h2>h1")
                else:
                    logs.append("PASS: v_2' matches expression for h2>h1")
            else:
                if not self.fuzzy_equal(self.v_2_prime, self.v_combined_computed):
                    logs.append(f"FAIL: v_2' = {self.v_2_prime} != computed {self.v_combined_computed} for h1>h2")
                else:
                    logs.append("PASS: v_2' matches expression for h1>h2")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_h1 = self.h1
        try:
            perturbed_h1 = original_h1 * 1.1
            k_pert = (self.m0**(2/3) * self.g * self.t) / perturbed_h1
            q_pert = (self.m0 * self.g * self.d * (perturbed_h1 + self.h2)) / (perturbed_h1 * self.U)
            delta_E_p_pert = - (self.m0 * self.g * self.h2 * (perturbed_h1 + self.h2)) / perturbed_h1
            F_prime_pert = (self.m0 * self.g * (perturbed_h1 + self.h2)) / perturbed_h1
            
            if k_pert < self.k_computed:
                logs.append("PASS: k decreases when h1 increases (as expected)")
            else:
                logs.append(f"WARNING: k monotonicity not as expected: k_pert={k_pert}, original={self.k_computed}")
            
            if q_pert < self.q_computed:
                logs.append("PASS: q decreases when h1 increases (as expected)")
            else:
                logs.append(f"WARNING: q monotonicity not as expected: q_pert={q_pert}, original={self.q_computed}")
            
            if abs(delta_E_p_pert) < abs(self.delta_E_p_computed):
                logs.append("PASS: |ΔE_p| decreases when h1 increases (as expected)")
            else:
                logs.append(f"WARNING: |ΔE_p| monotonicity not as expected: |ΔE_p_pert|={abs(delta_E_p_pert)}, original={abs(self.delta_E_p_computed)}")
            
            if F_prime_pert < self.F_prime_computed:
                logs.append("PASS: F' decreases when h1 increases (as expected)")
            else:
                logs.append(f"WARNING: F' monotonicity not as expected: F'_pert={F_prime_pert}, original={self.F_prime_computed}")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        finally:
            self.h1 = original_h1
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        invariants_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        monotonicity_logs = self.test_monotonicity()
        
        all_logs.append("=== PhysicsChecker Report ===")
        all_logs.append("--- test_invariants ---")
        all_logs.extend(invariants_logs)
        all_logs.append("--- test_trace_consistency ---")
        all_logs.extend(trace_logs)
        all_logs.append("--- test_monotonicity ---")
        all_logs.extend(monotonicity_logs)
        
        # Check for any FAIL or ERROR messages
        for log in invariants_logs + trace_logs + monotonicity_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        return success, all_logs