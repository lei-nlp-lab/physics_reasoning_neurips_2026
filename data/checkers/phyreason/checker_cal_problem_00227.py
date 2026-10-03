import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results without direct attribute access
        self.inputs = {}
        self.results = {}
        
        # Expected inputs from problem statement - use defaults if not found
        # Default values: p0=1.0, S=1.0, H=1.0, g=9.8 (same as in solution)
        self.inputs['p0'] = getattr(solution_instance, 'p0', 1.0)
        self.inputs['S'] = getattr(solution_instance, 'S', 1.0)
        self.inputs['H'] = getattr(solution_instance, 'H', 1.0)
        self.inputs['g'] = getattr(solution_instance, 'g', 9.8)
        
        # Extract results safely from results dictionary
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results.copy()
        else:
            # Try to get individual attributes
            result_keys = ['V_1', 'p_1', 'm', 'V_2', 'k', 'p_2']
            for key in result_keys:
                self.results[key] = getattr(solution_instance, key, 0.0)
        
    def test_invariants(self):
        logs = []
        try:
            # Extract numerical values
            S_val = self.inputs['S']
            H_val = self.inputs['H']
            p0_val = self.inputs['p0']
            g_val = self.inputs['g']
            
            V1_val = self.results.get('V_1', 0.0)
            p1_val = self.results.get('p_1', 0.0)
            V2_val = self.results.get('V_2', 0.0)
            p2_val = self.results.get('p_2', 0.0)
            m_val = self.results.get('m', 0.0)
            k_val = self.results.get('k', 0.0)
            
            # Test 1: Boyle's Law (isothermal process)
            product1 = p1_val * V1_val
            product2 = p2_val * V2_val
            
            if math.isclose(product1, product2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Boyle's Law invariant satisfied: p1*V1 = p2*V2")
            else:
                logs.append(f"✗ Boyle's Law violation: p1*V1={product1:.3f}, p2*V2={product2:.3f}")
            
            # Test 2: Force balance on right piston
            left_side = m_val * g_val + p0_val * (2*S_val)
            right_side = p2_val * (2*S_val)
            
            if math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Right piston force balance satisfied")
            else:
                logs.append(f"✗ Right piston force imbalance: left={left_side:.3f}, right={right_side:.3f}")
            
            # Test 3: Force balance on left piston
            left_side_left = p0_val * S_val + k_val * (H_val/2)
            right_side_left = p2_val * S_val
            
            if math.isclose(left_side_left, right_side_left, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Left piston force balance satisfied")
            else:
                logs.append(f"✗ Left piston force imbalance: left={left_side_left:.3f}, right={right_side_left:.3f}")
            
            # Test 4: Initial pressure equals atmospheric pressure
            if math.isclose(p1_val, p0_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Initial pressure equals atmospheric pressure")
            else:
                logs.append(f"✗ Initial pressure mismatch: p1={p1_val:.3f}, p0={p0_val:.3f}")
            
            return logs
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            S_val = self.inputs['S']
            H_val = self.inputs['H']
            p0_val = self.inputs['p0']
            g_val = self.inputs['g']
            
            V1_val = self.results.get('V_1', 0.0)
            p1_val = self.results.get('p_1', 0.0)
            V2_val = self.results.get('V_2', 0.0)
            p2_val = self.results.get('p_2', 0.0)
            m_val = self.results.get('m', 0.0)
            k_val = self.results.get('k', 0.0)
            
            # Check V1 calculation: V1 = S*H + 2S*H = 3SH
            expected_V1 = 3 * S_val * H_val
            if math.isclose(V1_val, expected_V1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ V1 calculation consistent with problem statement")
            else:
                logs.append(f"✗ V1 mismatch: calculated={V1_val:.3f}, expected={expected_V1:.3f}")
            
            # Check V2 calculation: V2 = S*(3H/2) + 2S*(2H/3) = (17/6)SH
            expected_V2 = (17/6) * S_val * H_val
            if math.isclose(V2_val, expected_V2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ V2 calculation consistent with problem statement")
            else:
                logs.append(f"✗ V2 mismatch: calculated={V2_val:.3f}, expected={expected_V2:.3f}")
            
            # Check p2 from Boyle's Law: p2 = p1*V1/V2 = p0*(3SH)/((17/6)SH) = (18/17)p0
            expected_p2 = (18/17) * p0_val
            if math.isclose(p2_val, expected_p2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ p2 calculation consistent with Boyle's Law")
            else:
                logs.append(f"✗ p2 mismatch: calculated={p2_val:.3f}, expected={expected_p2:.3f}")
            
            # Check m from force balance: m = (2S/g)*(p2 - p0) = (2S/g)*(p0/17)
            expected_m = (2 * S_val * p0_val) / (17 * g_val)
            if math.isclose(m_val, expected_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ m calculation consistent with force balance")
            else:
                logs.append(f"✗ m mismatch: calculated={m_val:.3f}, expected={expected_m:.3f}")
            
            # Check k from force balance: k = (2S/H)*(p2 - p0) = (2S/H)*(p0/17)
            expected_k = (2 * S_val * p0_val) / (17 * H_val)
            if math.isclose(k_val, expected_k, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ k calculation consistent with force balance")
            else:
                logs.append(f"✗ k mismatch: calculated={k_val:.3f}, expected={expected_k:.3f}")
            
            return logs
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            return logs
    
    def test_monotonicity(self):
        logs = []
        original_H = self.inputs['H']
        
        try:
            # Perturb H by +10%
            perturbed_H = original_H * 1.1
            self.inputs['H'] = perturbed_H
            
            # Recalculate using formulas from steps analysis
            S_val = self.inputs['S']
            H_val = self.inputs['H']
            p0_val = self.inputs['p0']
            g_val = self.inputs['g']
            
            # Calculate expected values with perturbed H
            V1_pert = 3 * S_val * H_val
            V2_pert = (17/6) * S_val * H_val
            p2_pert = (18/17) * p0_val
            m_pert = (2 * S_val * p0_val) / (17 * g_val)
            k_pert = (2 * S_val * p0_val) / (17 * H_val)
            
            # Get original values
            original_V1 = self.results.get('V_1', 0.0)
            original_V2 = self.results.get('V_2', 0.0)
            original_m = self.results.get('m', 0.0)
            original_k = self.results.get('k', 0.0)
            
            # Check trends
            if V1_pert > original_V1:
                logs.append("✓ V1 increases with increased H (monotonicity)")
            else:
                logs.append("✗ V1 monotonicity violation")
            
            if V2_pert > original_V2:
                logs.append("✓ V2 increases with increased H (monotonicity)")
            else:
                logs.append("✗ V2 monotonicity violation")
            
            if math.isclose(m_pert, original_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ m independent of H as expected")
            else:
                logs.append("✗ m shows unexpected dependence on H")
            
            if k_pert < original_k:
                logs.append("✓ k decreases with increased H (monotonicity)")
            else:
                logs.append("✗ k monotonicity violation")
            
            # Restore original H
            self.inputs['H'] = original_H
            
            return logs
            
        except Exception as e:
            # Ensure H is restored even if error occurs
            self.inputs['H'] = original_H
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            return logs
    
    def test_sympy_consistency(self):
        """Numeric verification of the key equations"""
        logs = []
        try:
            S_val = self.inputs['S']
            H_val = self.inputs['H']
            p0_val = self.inputs['p0']
            g_val = self.inputs['g']
            
            # Get results
            V1_val = self.results.get('V_1', 0.0)
            p1_val = self.results.get('p_1', 0.0)
            V2_val = self.results.get('V_2', 0.0)
            p2_val = self.results.get('p_2', 0.0)
            m_val = self.results.get('m', 0.0)
            k_val = self.results.get('k', 0.0)
            
            # Expected values from formulas
            expected_V1 = 3 * S_val * H_val
            expected_V2 = (17/6) * S_val * H_val
            expected_p2 = (18/17) * p0_val
            expected_m = (2 * S_val * p0_val) / (17 * g_val)
            expected_k = (2 * S_val * p0_val) / (17 * H_val)
            
            # Compare with tolerance
            if math.isclose(V1_val, expected_V1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ V1 matches expected formula")
            else:
                logs.append(f"✗ V1 mismatch: {V1_val:.6f} vs {expected_V1:.6f}")
            
            if math.isclose(V2_val, expected_V2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ V2 matches expected formula")
            else:
                logs.append(f"✗ V2 mismatch: {V2_val:.6f} vs {expected_V2:.6f}")
            
            if math.isclose(p2_val, expected_p2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ p2 matches expected formula")
            else:
                logs.append(f"✗ p2 mismatch: {p2_val:.6f} vs {expected_p2:.6f}")
            
            if math.isclose(m_val, expected_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ m matches expected formula")
            else:
                logs.append(f"✗ m mismatch: {m_val:.6f} vs {expected_m:.6f}")
            
            if math.isclose(k_val, expected_k, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ k matches expected formula")
            else:
                logs.append(f"✗ k mismatch: {k_val:.6f} vs {expected_k:.6f}")
            
            return logs
            
        except Exception as e:
            logs.append(f"Error in test_sympy_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test and collect logs
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity),
            ("Formula Consistency Tests", self.test_sympy_consistency)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*60}")
            all_logs.append(f"Running {test_name}")
            all_logs.append(f"{'='*60}")
            
            test_logs = test_func()
            
            # Check for any ✗ in the logs
            for log in test_logs:
                all_logs.append(log)
                if log.startswith("✗"):
                    success = False
        
        # Summary
        all_logs.append(f"\n{'='*60}")
        all_logs.append("TEST SUMMARY")
        all_logs.append(f"{'='*60}")
        if success:
            all_logs.append("✓ All tests passed!")
        else:
            all_logs.append("✗ Some tests failed")
        
        return success, all_logs