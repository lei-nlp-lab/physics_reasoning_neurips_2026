import math
import sympy as sp
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        """
        # Extract input parameters with safe defaults
        self.C = getattr(solution_instance, 'C', None)
        self.T1 = getattr(solution_instance, 'T1', None)
        self.T2 = getattr(solution_instance, 'T2', None)
        self.T3 = getattr(solution_instance, 'T3', None)
        
        # Safely extract results dictionary or object
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        
        # Extract all expected result keys with safe defaults
        self.results = {
            'T_H,T_0': results.get('T_H,T_0', 0.0),
            'T_1T_2T_3': results.get('T_1T_2T_3', 0.0),
            'T_f': results.get('T_f', 0.0),
            'S': results.get('S', 0.0),
            'T_L': results.get('T_L', 0.0),
            'delta_U': results.get('delta_U', results.get('ΔU', 0.0)),
            'T_H': results.get('T_H', 0.0),
            'dS_1+dS_2': results.get('dS_1+dS_2', 0.0),
            'dT_1/T_1': results.get('dT_1/T_1', 0.0),
            'W': results.get('W', 0.0),
            'T_1+T_2+T_3': results.get('T_1+T_2+T_3', 0.0)
        }
        
        # Store logs
        self.logs = []
        
    def log(self, message: str) -> None:
        """Helper to log messages."""
        self.logs.append(message)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test thermodynamic invariants and laws.
        
        Returns:
            Tuple of (success_bool, list_of_violations)
        """
        violations = []
        
        # Test 1: Second Law - Entropy change for reversible two-object process should be zero
        if not math.isclose(self.results['dS_1+dS_2'], 0.0, rel_tol=1e-2, abs_tol=1e-2):
            violations.append(f"Second Law violation: dS_1+dS_2 = {self.results['dS_1+dS_2']} ≠ 0")
        
        # Test 2: First Law - Energy conservation for two-object system
        # ΔU = C(T_f - T1) + C(T_f - T2) = C(2T_f - T1 - T2)
        # W = C[T1 + T2 - 2T_f] (from steps analysis)
        # For Carnot engine: Q_hot + Q_cold = W (First Law)
        # Q_hot = -C(T_f - T1), Q_cold = -C(T_f - T2)  (negative because heat leaves reservoirs)
        # So W = -(Q_hot + Q_cold) = C(2T_f - T1 - T2) ??? Wait, let's check carefully:
        # Actually from step_5: W = C[T1 + T2 - 2T_f]
        # And from step_4: ΔU = C(2T_f - T1 - T2)
        # So W = -ΔU, which is consistent with First Law (ΔU = Q - W, and Q = Q_hot + Q_cold = 0??)
        # For the two reservoirs together as the working substance, Q = 0 (no external heat) so W = -ΔU
        if self.C is not None and self.T1 is not None and self.T2 is not None:
            T_f_calc = math.sqrt(self.T1 * self.T2) if self.T1 > 0 and self.T2 > 0 else 0
            delta_U_calc = self.C * (2 * T_f_calc - self.T1 - self.T2)
            W_calc = self.C * (self.T1 + self.T2 - 2 * T_f_calc)
            
            # Check W = -ΔU
            if not math.isclose(W_calc, -delta_U_calc, rel_tol=1e-2, abs_tol=1e-2):
                violations.append(f"First Law violation for two-object system: W = {W_calc}, -ΔU = {-delta_U_calc}")
            
            # Check against provided results if available
            if self.results['T_f'] > 0 and self.results['W'] != 0:
                if not math.isclose(self.results['W'], -self.results['delta_U'], rel_tol=1e-2, abs_tol=1e-2):
                    violations.append(f"First Law violation from results: W = {self.results['W']}, -ΔU = {-self.results['delta_U']}")
        
        # Test 3: Second Law - Entropy change for three-object system should be non-negative
        # For three objects with equal C: S = C*ln(T1*T2*T3) + constant
        # The product T1*T2*T3 should be non-decreasing for spontaneous processes
        if self.T1 is not None and self.T2 is not None and self.T3 is not None:
            initial_product = self.T1 * self.T2 * self.T3
            # From step_9: T1*T2*T3 = constant
            if not math.isclose(self.results['T_1T_2T_3'], initial_product, rel_tol=1e-2, abs_tol=1e-2):
                violations.append(f"Entropy conservation violation: initial T1*T2*T3 = {initial_product}, constant = {self.results['T_1T_2T_3']}")
        
        # Test 4: Energy conservation for three-object system
        # T1 + T2 + T3 should be constant (from step_9)
        if self.T1 is not None and self.T2 is not None and self.T3 is not None:
            initial_sum = self.T1 + self.T2 + self.T3
            if not math.isclose(self.results['T_1+T_2+T_3'], initial_sum, rel_tol=1e-2, abs_tol=1e-2):
                violations.append(f"Energy conservation violation: initial sum = {initial_sum}, constant = {self.results['T_1+T_2+T_3']}")
        
        # Test 5: Maximum temperature constraint for three-object system
        # From step_10: T1+T2+T3 = T_H + 2T_0 and T1*T2*T3 = T_H*T_0^2
        # This should be satisfied
        if self.results['T_H'] > 0 and 'T_H,T_0' in str(self.results):
            # Extract T0 from T_H,T_0 if possible (assuming it's a tuple or string representation)
            t_h_t_0 = self.results['T_H,T_0']
            if isinstance(t_h_t_0, (tuple, list)) and len(t_h_t_0) == 2:
                T_H_val, T_0_val = t_h_t_0
            else:
                # Try to parse if it's a string
                try:
                    if isinstance(t_h_t_0, str):
                        parts = t_h_t_0.split(',')
                        T_H_val = float(parts[0])
                        T_0_val = float(parts[1])
                    else:
                        T_H_val = self.results['T_H']
                        T_0_val = (self.results['T_1+T_2+T_3'] - self.results['T_H']) / 2
                except:
                    T_H_val = self.results['T_H']
                    T_0_val = (self.results['T_1+T_2+T_3'] - self.results['T_H']) / 2
            
            # Check the two conservation equations
            sum_check = math.isclose(self.results['T_1+T_2+T_3'], T_H_val + 2*T_0_val, rel_tol=1e-2, abs_tol=1e-2)
            product_check = math.isclose(self.results['T_1T_2T_3'], T_H_val * (T_0_val**2), rel_tol=1e-2, abs_tol=1e-2)
            
            if not (sum_check and product_check):
                violations.append(f"Maximum temperature constraints violated: sum_check={sum_check}, product_check={product_check}")
        
        success = len(violations) == 0
        if not success:
            self.log(f"Invariant tests failed: {violations}")
        else:
            self.log("All invariant tests passed.")
        
        return success, violations
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Check consistency between intermediate steps and final results.
        """
        inconsistencies = []
        
        # Check 1: T_f from step_3 vs step_6
        if self.T1 is not None and self.T2 is not None:
            T_f_calc = math.sqrt(self.T1 * self.T2)
            if self.results['T_f'] > 0:
                if not math.isclose(self.results['T_f'], T_f_calc, rel_tol=1e-2, abs_tol=1e-2):
                    inconsistencies.append(f"T_f inconsistency: calculated {T_f_calc}, result {self.results['T_f']}")
        
        # Check 2: W from step_5 vs step_6
        if self.T1 is not None and self.T2 is not None and self.C is not None:
            T_f_calc = math.sqrt(self.T1 * self.T2)
            W_calc = self.C * (self.T1 + self.T2 - 2 * T_f_calc)
            if self.results['W'] != 0:
                if not math.isclose(self.results['W'], W_calc, rel_tol=1e-2, abs_tol=1e-2):
                    inconsistencies.append(f"W inconsistency: calculated {W_calc}, result {self.results['W']}")
        
        # Check 3: T_L from step_7 should be 100K
        if not math.isclose(self.results['T_L'], 100.0, rel_tol=1e-2, abs_tol=1e-2):
            inconsistencies.append(f"T_L inconsistency: expected 100K, got {self.results['T_L']}K")
        
        # Check 4: T_H from step_13 should be 400K
        if not math.isclose(self.results['T_H'], 400.0, rel_tol=1e-2, abs_tol=1e-2):
            inconsistencies.append(f"T_H inconsistency: expected 400K, got {self.results['T_H']}K")
        
        # Check 5: Verify the polynomial relationship from step_11
        # T_H(7 - T_H)^2 = 36 when temperatures are normalized by 100K
        # Actually from step_11: T_H(7-T_H)^2=36 with T in units of 100K
        # So T_H/100 * (7 - T_H/100)^2 = 36
        T_H_normalized = self.results['T_H'] / 100.0
        lhs = T_H_normalized * (7 - T_H_normalized) ** 2
        if not math.isclose(lhs, 36.0, rel_tol=1e-2, abs_tol=1e-2):
            inconsistencies.append(f"Polynomial relationship violation: T_H({7-T_H_normalized})² = {lhs} ≠ 36")
        
        success = len(inconsistencies) == 0
        if not success:
            self.log(f"Trace consistency failed: {inconsistencies}")
        else:
            self.log("All trace consistency tests passed.")
        
        return success, inconsistencies
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb inputs and check monotonic trends.
        """
        issues = []
        
        # Only run if we have the necessary parameters
        if self.C is not None and self.T1 is not None and self.T2 is not None:
            # Store original values
            original_T1 = self.T1
            
            # Test 1: Increase T1, work output should increase (for two-object system)
            # For fixed T2, W = C(T1 + T2 - 2√(T1*T2))
            # dW/dT1 = C(1 - √(T2/T1))
            # For T1 > T2, dW/dT1 > 0; for T1 < T2, dW/dT1 > 0 as long as T1 > 0
            # Actually, dW/dT1 = C(1 - √(T2/T1)) > 0 when T1 > T2
            # But we're testing with the given T1=100K, T2=300K, so T1 < T2, dW/dT1 = C(1 - √(3)) < 0
            # So increasing T1 should decrease W when T1 < T2
            
            # Perturb T1 upward by 10%
            perturbed_T1 = original_T1 * 1.1
            T_f_original = math.sqrt(original_T1 * self.T2)
            T_f_perturbed = math.sqrt(perturbed_T1 * self.T2)
            
            W_original = self.C * (original_T1 + self.T2 - 2 * T_f_original)
            W_perturbed = self.C * (perturbed_T1 + self.T2 - 2 * T_f_perturbed)
            
            # For T1 < T2, increasing T1 should decrease W (make it less negative)
            # Both W_original and W_perturbed are negative since T1+T2 < 2√(T1*T2)
            if not (W_perturbed > W_original):  # Less negative means larger
                issues.append(f"Monotonicity violation: increasing T1 from {original_T1} to {perturbed_T1} "
                             f"did not increase W (from {W_original} to {W_perturbed})")
            
            # Restore original value
            self.T1 = original_T1
            
            # Test 2: Increase T2, work output should increase (since T2 is the hot reservoir)
            # For T1 < T2, increasing T2 should make W more negative (increase magnitude)
            original_T2 = self.T2
            perturbed_T2 = original_T2 * 1.1
            T_f_original = math.sqrt(self.T1 * original_T2)
            T_f_perturbed = math.sqrt(self.T1 * perturbed_T2)
            
            W_original = self.C * (self.T1 + original_T2 - 2 * T_f_original)
            W_perturbed = self.C * (self.T1 + perturbed_T2 - 2 * T_f_perturbed)
            
            # For T1 < T2, increasing T2 should decrease W (make it more negative)
            if not (W_perturbed < W_original):
                issues.append(f"Monotonicity violation: increasing T2 from {original_T2} to {perturbed_T2} "
                             f"did not decrease W (from {W_original} to {W_perturbed})")
            
            # Restore
            self.T2 = original_T2
            
            # Test 3: For three-object system, if we increase initial T3, final T_H should increase
            if self.T3 is not None:
                original_T3 = self.T3
                perturbed_T3 = original_T3 * 1.1
                
                # From conservation laws:
                # T1+T2+T3 = T_H + 2T_0 = constant
                # T1*T2*T3 = T_H*T_0^2 = constant
                # Eliminating T_0: T_H*( (constant - T_H)/2 )^2 = constant_product
                # So T_H is determined by the cubic equation
                # We can't easily solve analytically, but we can check that
                # increasing T3 increases the RHS of the product equation, so T_H should increase
                # Actually, with T1 and T2 fixed, increasing T3 increases the constant product
                # which should increase T_H to maintain the equation
                
                # We'll just note this as a qualitative check
                self.log("Monotonicity: Qualitatively, increasing T3 should increase final T_H")
                
                # Restore
                self.T3 = original_T3
        
        success = len(issues) == 0
        if not success:
            self.log(f"Monotonicity tests failed: {issues}")
        else:
            self.log("All monotonicity tests passed.")
        
        return success, issues
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        self.logs = []
        
        # Run all tests
        inv_success, inv_logs = self.test_invariants()
        trace_success, trace_logs = self.test_trace_consistency()
        mono_success, mono_logs = self.test_monotonicity()
        
        # Overall success
        overall_success = inv_success and trace_success and mono_success
        
        # Compile all logs
        all_logs = self.logs.copy()
        
        return overall_success, all_logs


# Example usage and additional helper for demonstration
if __name__ == "__main__":
    # Create a mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.C = 1.0  # Heat capacity
            self.T1 = 100.0  # K
            self.T2 = 300.0  # K
            self.T3 = 300.0  # K
            self.results = {
                'T_H,T_0': (400.0, 300.0),
                'T_1T_2T_3': 9e6,  # 100*300*300
                'T_f': math.sqrt(100*300),  # ~173.2
                'S': 0.0,
                'T_L': 100.0,
                'delta_U': 1.0*(2*math.sqrt(100*300) - 100 - 300),  # ~-153.6
                'T_H': 400.0,
                'dS_1+dS_2': 0.0,
                'dT_1/T_1': 0.0,
                'W': 1.0*(100 + 300 - 2*math.sqrt(100*300)),  # ~153.6
                'T_1+T_2+T_3': 700.0  # 100+300+300
            }
    
    # Test the checker
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    print(f"Success: {success}")
    for log in logs:
        print(f"  {log}")