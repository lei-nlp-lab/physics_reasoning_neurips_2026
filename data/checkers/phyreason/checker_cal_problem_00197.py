import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize all inputs and results using safe attribute access.
        """
        try:
            # Extract inputs safely (from instance or .inputs dict)
            self.V1 = getattr(solution_instance, 'V1', 
                             getattr(solution_instance, 'inputs', {}).get('V1', 1.0))
            self.p1 = getattr(solution_instance, 'p1', 
                             getattr(solution_instance, 'inputs', {}).get('p1', 1.0))
            self.T1 = getattr(solution_instance, 'T1', 
                             getattr(solution_instance, 'inputs', {}).get('T1', 300.0))
            self.k = getattr(solution_instance, 'k', 
                            getattr(solution_instance, 'inputs', {}).get('k', 0.5))
            self.T2 = getattr(solution_instance, 'T2', 
                             getattr(solution_instance, 'inputs', {}).get('T2', 250.0))
            
            # Extract results safely
            results = getattr(solution_instance, 'results', {})
            self.V = results.get('V', 0.0)
            self.p = results.get('p', 0.0)
            
            # Store original for monotonicity
            self._original_k = self.k
            self._original_V1 = self.V1
        except Exception as e:
            print(f"Warning: Initialization error: {e}")
            # Default fallbacks
            self.V1 = self.p1 = self.T1 = self.k = self.T2 = 1.0
            self.V = self.p = 0.0
            self._original_k = self._original_V1 = 1.0
    
    def test_invariants(self):
        """
        Verify fundamental gas laws:
        1. Boyle's Law (step 1): p1*V1 = k*p1*(V1 + V)
        2. Charles's Law (step 2): (k*p1)/T1 = p/T2
        """
        logs = []
        success = True
        
        # Test 1: Boyle's Law (step 1)
        try:
            left = self.p1 * self.V1
            right = self.k * self.p1 * (self.V1 + self.V)
            if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Boyle's Law satisfied: p1*V1 ≈ k*p1*(V1+V)")
            else:
                logs.append(f"✗ Boyle's Law violation: {left} ≠ {right}")
                success = False
        except Exception as e:
            logs.append(f"✗ Boyle's Law test failed: {e}")
            success = False
        
        # Test 2: Charles's Law (step 2)
        try:
            left = (self.k * self.p1) / self.T1
            right = self.p / self.T2
            if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Charles's Law satisfied: (k*p1)/T1 ≈ p/T2")
            else:
                logs.append(f"✗ Charles's Law violation: {left} ≠ {right}")
                success = False
        except Exception as e:
            logs.append(f"✗ Charles's Law test failed: {e}")
            success = False
        
        # Test 3: Ideal gas law consistency (numerical fallback)
        try:
            # Compute implied moles from initial tank
            R = 8.314  # J/(mol·K)
            n_initial = (self.p1 * self.V1) / (R * self.T1)
            
            # Moles after inflation: tank + balloon at pressure k*p1
            n_total = (self.k * self.p1 * (self.V1 + self.V)) / (R * self.T1)
            
            if math.isclose(n_initial, n_total, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Mole conservation satisfied")
            else:
                logs.append(f"✗ Mole conservation violation: {n_initial} ≠ {n_total}")
                success = False
        except Exception as e:
            logs.append(f"✗ Mole conservation test failed: {e}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify V matches formula V = (1-k)/k * V1
        Verify p matches formula p = (k*p1*T2)/T1
        """
        logs = []
        success = True
        
        # Check V formula
        try:
            V_expected = (1 - self.k) / self.k * self.V1
            if math.isclose(self.V, V_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ V matches formula: V = (1-k)/k * V1")
            else:
                logs.append(f"✗ V mismatch: {self.V} ≠ {V_expected}")
                success = False
        except Exception as e:
            logs.append(f"✗ V formula test failed: {e}")
            success = False
        
        # Check p formula
        try:
            p_expected = (self.k * self.p1 * self.T2) / self.T1
            if math.isclose(self.p, p_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ p matches formula: p = (k*p1*T2)/T1")
            else:
                logs.append(f"✗ p mismatch: {self.p} ≠ {p_expected}")
                success = False
        except Exception as e:
            logs.append(f"✗ p formula test failed: {e}")
            success = False
        
        # Cross-check using ideal gas law with constant moles in balloon
        try:
            # Moles in balloon after inflation
            R = 8.314
            n_balloon = (self.k * self.p1 * self.V) / (R * self.T1)
            # Moles in balloon at altitude (should be same, volume constant)
            n_altitude = (self.p * self.V) / (R * self.T2)
            
            if math.isclose(n_balloon, n_altitude, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Balloon moles consistent between steps")
            else:
                logs.append(f"✗ Balloon moles inconsistent: {n_balloon} ≠ {n_altitude}")
                success = False
        except Exception as e:
            logs.append(f"✗ Balloon moles test failed: {e}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb k and V1 to check expected trends:
        1. Increase k → V decreases, p increases
        2. Increase V1 → V increases, p unchanged
        """
        logs = []
        success = True
        
        try:
            # Test 1: Perturb k (+1%)
            perturbed_k = self.k * 1.01
            V_perturbed = (1 - perturbed_k) / perturbed_k * self.V1
            p_perturbed = (perturbed_k * self.p1 * self.T2) / self.T1
            
            # Check trends
            V_trend_ok = (self.k < perturbed_k) and (self.V > V_perturbed)
            p_trend_ok = (self.k < perturbed_k) and (self.p < p_perturbed)
            
            if V_trend_ok:
                logs.append("✓ V decreases when k increases")
            else:
                logs.append(f"✗ V trend violation: k↑, V should ↓ (V={self.V}, V_pert={V_perturbed})")
                success = False
            
            if p_trend_ok:
                logs.append("✓ p increases when k increases")
            else:
                logs.append(f"✗ p trend violation: k↑, p should ↑ (p={self.p}, p_pert={p_perturbed})")
                success = False
        except Exception as e:
            logs.append(f"✗ k perturbation test failed: {e}")
            success = False
        
        try:
            # Test 2: Perturb V1 (+1%)
            perturbed_V1 = self.V1 * 1.01
            V_perturbed = (1 - self.k) / self.k * perturbed_V1
            p_perturbed = (self.k * self.p1 * self.T2) / self.T1  # Should not change
            
            # Check trends
            V_trend_ok = (self.V1 < perturbed_V1) and (self.V < V_perturbed)
            p_trend_ok = math.isclose(p_perturbed, self.p, rel_tol=1e-2, abs_tol=1e-2)
            
            if V_trend_ok:
                logs.append("✓ V increases when V1 increases")
            else:
                logs.append(f"✗ V trend violation: V1↑, V should ↑ (V={self.V}, V_pert={V_perturbed})")
                success = False
            
            if p_trend_ok:
                logs.append("✓ p unchanged when V1 increases")
            else:
                logs.append(f"✗ p should be independent of V1 (p={self.p}, p_pert={p_perturbed})")
                success = False
        except Exception as e:
            logs.append(f"✗ V1 perturbation test failed: {e}")
            success = False
        
        # Restore original values
        self.k = self._original_k
        self.V1 = self._original_V1
        
        return success, logs
    
    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Test crashed: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs

# Example usage (commented out for production):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)