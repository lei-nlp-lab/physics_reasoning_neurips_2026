import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs
        self.m1 = getattr(solution_instance, 'm1', 2.0)
        self.m2 = getattr(solution_instance, 'm2', 1.0)
        self.alpha = getattr(solution_instance, 'alpha', 37.0)  # in degrees
        self.mu1 = getattr(solution_instance, 'mu1', 0.5)
        self.mu2 = getattr(solution_instance, 'mu2', 0.4)
        self.mu3 = getattr(solution_instance, 'mu3', 0.3)
        self.mu4 = getattr(solution_instance, 'mu4', 0.2)
        self.mu5 = getattr(solution_instance, 'mu5', 0.1)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin_alpha = 0.6  # given sin37° = 0.6
        self.cos_alpha = 0.8  # given cos37° = 0.8
        
        # Extract results safely using the provided key mapping
        self.results = {}
        key_mapping = {
            'F_{f1}': 'F_f1',
            'F': 'F',
            'F_{f3mx}': 'F_f3mx',
            'F_{N1}': 'F_N1',
            'F_{N2}': 'F_N2',
            'F_{N3}': 'F_N3',
            'F_{fl}': 'F_fl',
            'F_{f4}': 'F_f4',
            'N/A': 'N_A',  # placeholder
            'F_{T1}': 'F_T1',
            'F_{f2mx}': 'F_f2mx',
            'F_{f4mx}': 'F_f4mx',
            'F_{N4}': 'F_N4',
            'F_{N5}': 'F_N5'
        }
        for latex_key, py_key in key_mapping.items():
            self.results[py_key] = getattr(solution_instance, py_key, None)
            if self.results[py_key] is None:
                # Fallback to results dictionary if attribute not present
                self.results[py_key] = getattr(solution_instance, 'results', {}).get(py_key, 0.0)
        
        # Pre-compute expected values for cross-checking
        self.expected = self._compute_expected()
        
    def _compute_expected(self):
        """Compute expected values based on inputs and physics laws."""
        # Normal forces (vertical equilibrium)
        F_T1 = (self.mu1 * self.m1 * self.g) / (self.mu1 * self.cos_alpha + self.sin_alpha)
        F_N1 = self.m1 * self.g - F_T1 * self.cos_alpha
        F_N2 = F_N1 + self.m2 * self.g
        F_N3 = F_N2 + self.m2 * self.g
        F_N4 = F_N3 + self.m2 * self.g
        F_N5 = F_N4 + self.m2 * self.g
        
        # Friction forces (kinetic/max static, since slow motion)
        F_f1 = self.mu1 * F_N1
        F_f2 = self.mu2 * F_N2
        F_f3 = self.mu3 * F_N3
        F_f4 = self.mu4 * F_N4
        F_f5 = self.mu5 * F_N5
        
        # Applied force F (from horizontal equilibrium of whole system or step_24)
        # According to step_24: F = F_f4 + F_f1 + F_f2 + F_f5
        F = F_f4 + F_f1 + F_f2 + F_f5
        
        return {
            'F_T1': F_T1,
            'F_N1': F_N1,
            'F_N2': F_N2,
            'F_N3': F_N3,
            'F_N4': F_N4,
            'F_N5': F_N5,
            'F_f1': F_f1,
            'F_f2': F_f2,
            'F_f3': F_f3,
            'F_f4': F_f4,
            'F_f5': F_f5,
            'F': F
        }
    
    def test_invariants(self):
        """Check fundamental physics invariants."""
        logs = []
        
        # 1. Vertical force equilibrium for each block
        # Block A: F_T1*cos_alpha + F_N1 = m1*g
        val = self.results.get('F_T1', 0) * self.cos_alpha + self.results.get('F_N1', 0)
        target = self.m1 * self.g
        if not math.isclose(val, target, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Vertical equilibrium on A: {val} vs {target}")
        
        # Block B: F_N1 + m2*g = F_N2
        val = self.results.get('F_N1', 0) + self.m2 * self.g
        target = self.results.get('F_N2', 0)
        if not math.isclose(val, target, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Vertical equilibrium on B: {val} vs {target}")
        
        # Block C: F_N2 + m2*g = F_N3
        val = self.results.get('F_N2', 0) + self.m2 * self.g
        target = self.results.get('F_N3', 0)
        if not math.isclose(val, target, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Vertical equilibrium on C: {val} vs {target}")
        
        # Block D: F_N3 + m2*g = F_N4
        val = self.results.get('F_N3', 0) + self.m2 * self.g
        target = self.results.get('F_N4', 0)
        if not math.isclose(val, target, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Vertical equilibrium on D: {val} vs {target}")
        
        # Block E: F_N4 + m2*g = F_N5
        val = self.results.get('F_N4', 0) + self.m2 * self.g
        target = self.results.get('F_N5', 0)
        if not math.isclose(val, target, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Vertical equilibrium on E: {val} vs {target}")
        
        # 2. Friction forces equal mu * normal force
        pairs = [
            ('F_f1', 'F_N1', self.mu1),
            ('F_f2mx', 'F_N2', self.mu2),
            ('F_f3mx', 'F_N3', self.mu3),
            ('F_f4mx', 'F_N4', self.mu4)
        ]
        for f_key, n_key, mu in pairs:
            f_val = self.results.get(f_key, 0)
            n_val = self.results.get(n_key, 0)
            expected = mu * n_val
            if not math.isclose(f_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Friction law for {f_key}: {f_val} vs {expected}")
        
        # 3. Horizontal equilibrium on A: F_f1 = F_T1 * sin_alpha
        left = self.results.get('F_f1', 0)
        right = self.results.get('F_T1', 0) * self.sin_alpha
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Horizontal equilibrium on A: {left} vs {right}")
        
        # 4. Check that F_f4 equals F_f4mx (since slow motion, kinetic = max static)
        if not math.isclose(self.results.get('F_f4', 0), self.results.get('F_f4mx', 0), 
                           rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F_f4 should equal F_f4mx for slow motion")
        
        return len(logs) == 0, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        # Step 24: F = F_f4 + F_f1 + F_f2 + F_f5
        # Note: F_f2 is stored as F_f2mx in results, and F_f5 is not stored directly.
        # Compute F_f5 from mu5 and F_N5
        F_f5 = self.mu5 * self.results.get('F_N5', 0)
        computed_F = (self.results.get('F_f4', 0) + 
                     self.results.get('F_f1', 0) + 
                     self.results.get('F_f2mx', 0) + 
                     F_f5)
        target_F = self.results.get('F', 0)
        if not math.isclose(computed_F, target_F, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Trace consistency F: {computed_F} vs {target_F}")
        
        # Step 5: F_T1 formula
        computed_T1 = (self.mu1 * self.m1 * self.g) / (self.mu1 * self.cos_alpha + self.sin_alpha)
        target_T1 = self.results.get('F_T1', 0)
        if not math.isclose(computed_T1, target_T1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F_T1 formula: {computed_T1} vs {target_T1}")
        
        # Step 6: F_N1 from vertical equilibrium
        computed_N1 = self.m1 * self.g - target_T1 * self.cos_alpha
        target_N1 = self.results.get('F_N1', 0)
        if not math.isclose(computed_N1, target_N1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F_N1 from equilibrium: {computed_N1} vs {target_N1}")
        
        return len(logs) == 0, logs
    
    def test_monotonicity(self):
        """Perturb an input and check the trend."""
        logs = []
        
        # Perturb m1 by +10%
        original_m1 = self.m1
        self.m1 *= 1.1
        
        # Recompute expected values with perturbed m1
        perturbed = self._compute_expected()
        
        # Check that F increases when m1 increases (since more weight -> more normal forces -> more friction)
        original_F = self.expected['F']
        perturbed_F = perturbed['F']
        if perturbed_F <= original_F:
            logs.append(f"FAIL: Monotonicity w.r.t m1: F did not increase when m1 increased")
        
        # Restore original value
        self.m1 = original_m1
        
        return len(logs) == 0, logs
    
    def run_all_tests(self):
        """Run all tests and return success and logs."""
        all_logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                if not test_success:
                    success = False
                    all_logs.append(f"{test_name} failed:")
                    all_logs.extend(logs)
                else:
                    all_logs.append(f"{test_name} passed.")
            except Exception as e:
                success = False
                all_logs.append(f"{test_name} raised exception: {e}")
                all_logs.append(traceback.format_exc())
        
        return success, all_logs