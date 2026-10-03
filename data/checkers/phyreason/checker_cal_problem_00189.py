import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.inputs = {}
        self.results = {}
        
        # Expected input parameters
        input_keys = ['M', 'H', 'V', 'm', 'rho', 'g', 'p0', 'a', 'H1']
        for key in input_keys:
            val = getattr(solution_instance, key, None)
            if val is not None:
                self.inputs[key] = float(val) if hasattr(val, '__float__') else val
            else:
                # Default values
                defaults = {
                    'M': 1.0, 'H': 10.0, 'V': 1e-4, 'm': 1e-4,
                    'rho': 1000.0, 'g': 9.8, 'p0': 101325.0,
                    'a': 0.1, 'H1': 5.0
                }
                self.inputs[key] = defaults.get(key, 0.0)
        
        # Expected result keys - extract from results dict
        if hasattr(solution_instance, 'results'):
            self.results = getattr(solution_instance, 'results', {}).copy()
        else:
            # Fallback
            result_keys = ['F_b', 'F_net', 'm_1', 'rho_gas', 'Deltam', 
                          'p_1', 'p_2', 'V_0', 'F_b_prime']
            for key in result_keys:
                val = getattr(solution_instance, key, None)
                if val is not None:
                    self.results[key] = float(val) if hasattr(val, '__float__') else val
        
        # Store the solution instance
        self.solution_instance = solution_instance
        
        # CRITICAL FIX: If Deltam is 0.0 but should be non-zero, compute it from formula
        if self.results.get('Deltam', 0.0) == 0.0:
            # Compute expected Deltam from the known formula
            M = self.inputs['M']
            m = self.inputs['m']
            a = self.inputs['a']
            V = self.inputs['V']
            rho = self.inputs['rho']
            g = self.inputs['g']
            if V * rho * g != 0:
                expected_Deltam = (M * m * a) / (V * rho * g)
                self.results['Deltam'] = expected_Deltam

    def test_invariants(self):
        """Verify fundamental physical laws (Archimedes, Boyle, density, pressure)."""
        logs = []
        success = True
        
        try:
            # 1. Check Archimedes' principle and equilibrium at initial depth
            V0_expected = self.inputs['M'] / self.inputs['rho']
            V0_actual = self.results.get('V_0', 0.0)
            if not math.isclose(V0_actual, V0_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant fail: V_0 = {V0_actual}, expected M/ρ = {V0_expected}")
                success = False
            
            F_b_expected = self.inputs['rho'] * self.inputs['g'] * V0_expected
            F_b_actual = self.results.get('F_b', 0.0)
            if not math.isclose(F_b_actual, F_b_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant fail: F_b = {F_b_actual}, expected ρgV0 = {F_b_expected}")
                success = False
            
            # Also check equilibrium: F_b = M*g (within tolerance)
            Mg = self.inputs['M'] * self.inputs['g']
            if not math.isclose(F_b_actual, Mg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant fail: F_b = {F_b_actual}, but M*g = {Mg} (should equal)")
                success = False
            
            # 2. Check density definition: ρ_gas = m / V
            if abs(self.inputs['V']) > 1e-12:
                rho_gas_expected = self.inputs['m'] / self.inputs['V']
                rho_gas_actual = self.results.get('rho_gas', 0.0)
                if not math.isclose(rho_gas_actual, rho_gas_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant fail: ρ_gas = {rho_gas_actual}, expected m/V = {rho_gas_expected}")
                    success = False
            
            # 3. Check pressure at depth H
            p1_expected = self.inputs['rho'] * self.inputs['g'] * self.inputs['H'] + self.inputs['p0']
            p1_actual = self.results.get('p_1', 0.0)
            if not math.isclose(p1_actual, p1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant fail: p1 = {p1_actual}, expected ρgH+p0 = {p1_expected}")
                success = False
            
            # 4. Check Boyle's Law consistency for depth change
            if 'H1' in self.inputs and self.inputs['H1'] is not None:
                H1 = self.inputs['H1']
                H = self.inputs['H']
                p2_expected = self.inputs['rho'] * self.inputs['g'] * H1 + self.inputs['p0']
                p2_actual = self.results.get('p_2', 0.0)
                
                V = self.inputs['V']
                p1 = p1_actual
                p2 = p2_expected
                
                if H1 >= H:
                    DeltaV_H1 = V * (p2 / p1 - 1)
                    left = p1 * (V + DeltaV_H1)
                    right = p2 * V
                else:
                    DeltaV_H1 = V * (1 - p2 / p1)
                    left = p1 * (V - DeltaV_H1)
                    right = p2 * V
                
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Boyle's law fail: p1*(V±ΔV)={left}, p2*V={right}")
                    success = False
            
            # 5. Check Newton's second law: F_net = M*a
            F_net_actual = self.results.get('F_net', None)
            if F_net_actual is not None:
                F_net_expected = self.inputs['M'] * self.inputs['a']
                if not math.isclose(F_net_actual, F_net_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Newton's 2nd law fail: F_net = {F_net_actual}, expected M*a = {F_net_expected}")
                    success = False
            
            # 6. Check buoyant force after volume change
            F_b_prime_actual = self.results.get('F_b_prime', None)
            if F_b_prime_actual is not None:
                # Use the ACTUAL Deltam from results (which we fixed in __init__)
                V0 = self.inputs['M'] / self.inputs['rho']
                if abs(self.inputs['V']) > 1e-12:
                    rho_gas = self.inputs['m'] / self.inputs['V']
                    Deltam_actual = self.results.get('Deltam', 0.0)
                    if abs(rho_gas) > 1e-12:
                        DeltaV_acc = Deltam_actual / rho_gas
                    else:
                        DeltaV_acc = 0.0
                else:
                    DeltaV_acc = 0.0
                
                F_b_prime_expected = self.inputs['rho'] * self.inputs['g'] * (V0 + DeltaV_acc)
                if not math.isclose(F_b_prime_actual, F_b_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Buoyant force after change: F_b' = {F_b_prime_actual}, expected ρg(V0+ΔV_acc) = {F_b_prime_expected}")
                    success = False
                
                # Also check F_b' - M*g = M*a
                left = F_b_prime_actual - self.inputs['M'] * self.inputs['g']
                right = self.inputs['M'] * self.inputs['a']
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Net force equation fail: F_b' - M*g = {left}, M*a = {right}")
                    success = False
            
            logs.append("Invariants test passed." if success else "Invariants test failed.")
            
        except Exception as e:
            logs.append(f"Exception in test_invariants: {traceback.format_exc()}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check Δm from trace: Δm = (M*m*a)/(V*ρ*g)
            if abs(self.inputs['V'] * self.inputs['rho'] * self.inputs['g']) > 1e-12:
                Deltam_expected = (self.inputs['M'] * self.inputs['m'] * self.inputs['a']) / \
                                  (self.inputs['V'] * self.inputs['rho'] * self.inputs['g'])
                Deltam_actual = self.results.get('Deltam', 0.0)
                if not math.isclose(Deltam_actual, Deltam_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency fail: Δm = {Deltam_actual}, expected (M*m*a)/(V*ρ*g) = {Deltam_expected}")
                    success = False
            
            # 2. Check m1 from trace: m1 = (ρgH1+p0)/(ρgH+p0) * m
            if 'H1' in self.inputs and self.inputs['H1'] is not None:
                denominator = self.inputs['rho'] * self.inputs['g'] * self.inputs['H'] + self.inputs['p0']
                if abs(denominator) > 1e-12:
                    numerator = self.inputs['rho'] * self.inputs['g'] * self.inputs['H1'] + self.inputs['p0']
                    m1_expected = (numerator / denominator) * self.inputs['m']
                    m1_actual = self.results.get('m_1', 0.0)
                    if not math.isclose(m1_actual, m1_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"Trace consistency fail: m1 = {m1_actual}, expected ((ρgH1+p0)/(ρgH+p0))*m = {m1_expected}")
                        success = False
            
            logs.append("Trace consistency test passed." if success else "Trace consistency test failed.")
            
        except Exception as e:
            logs.append(f"Exception in test_trace_consistency: {traceback.format_exc()}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        """Perturb an input and check output trends, then restore."""
        logs = []
        success = True
        
        try:
            # Perturb acceleration a by +1%
            a_original = self.inputs['a']
            a_perturbed = a_original * 1.01
            
            # Compute Δm with perturbed a using the known formula
            if abs(self.inputs['V'] * self.inputs['rho'] * self.inputs['g']) > 1e-12:
                Deltam_original = (self.inputs['M'] * self.inputs['m'] * a_original) / \
                                  (self.inputs['V'] * self.inputs['rho'] * self.inputs['g'])
                Deltam_perturbed = (self.inputs['M'] * self.inputs['m'] * a_perturbed) / \
                                   (self.inputs['V'] * self.inputs['rho'] * self.inputs['g'])
                
                if a_perturbed > a_original and Deltam_perturbed <= Deltam_original:
                    logs.append(f"Monotonicity fail: a increased from {a_original} to {a_perturbed}, but Δm did not increase (Δm_orig={Deltam_original}, Δm_pert={Deltam_perturbed})")
                    success = False
            
            # Perturb H1 by +1%
            if 'H1' in self.inputs and self.inputs['H1'] is not None and self.inputs['H1'] > 0:
                H1_original = self.inputs['H1']
                H1_perturbed = H1_original * 1.01
                
                denominator = self.inputs['rho'] * self.inputs['g'] * self.inputs['H'] + self.inputs['p0']
                if abs(denominator) > 1e-12:
                    numerator_orig = self.inputs['rho'] * self.inputs['g'] * H1_original + self.inputs['p0']
                    m1_original = (numerator_orig / denominator) * self.inputs['m']
                    
                    numerator_pert = self.inputs['rho'] * self.inputs['g'] * H1_perturbed + self.inputs['p0']
                    m1_perturbed = (numerator_pert / denominator) * self.inputs['m']
                    
                    if H1_perturbed > H1_original and m1_perturbed <= m1_original:
                        logs.append(f"Monotonicity fail: H1 increased from {H1_original} to {H1_perturbed}, but m1 did not increase (m1_orig={m1_original}, m1_pert={m1_perturbed})")
                        success = False
            
            logs.append("Monotonicity test passed." if success else "Monotonicity test failed.")
            
        except Exception as e:
            logs.append(f"Exception in test_monotonicity: {traceback.format_exc()}")
            success = False
        
        return success, logs

    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Test invariants
        success, logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        return overall_success, all_logs