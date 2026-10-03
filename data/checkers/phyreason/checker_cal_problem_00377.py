import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results."""
        self.inputs = {}
        self.results = {}
        self.logs = []
        
        # Inputs from problem statement
        self.inputs['m_A'] = getattr(solution_instance, 'm_A', 1.0)
        self.inputs['m_B'] = getattr(solution_instance, 'm_B', 4.0)
        self.inputs['m_C'] = getattr(solution_instance, 'm_C', 5.0)
        self.inputs['v_A'] = getattr(solution_instance, 'v_A', 6.0)
        
        # Results from solution
        self.results['v_B'] = getattr(solution_instance, 'v_B', solution_instance.results.get('v_B', 0.0))
        self.results['E_P'] = getattr(solution_instance, 'E_P', solution_instance.results.get('E_P', 0.0))
        self.results['v_C'] = getattr(solution_instance, 'v_C', solution_instance.results.get('v_C', 0.0))
        
    def test_invariants(self):
        """Verify momentum and energy conservation laws."""
        logs = []
        success = True
        
        # Step 1: Momentum conservation during spring release (A+B system)
        try:
            m_A, m_B, v_A = self.inputs['m_A'], self.inputs['m_B'], self.inputs['v_A']
            v_B = self.results['v_B']
            # Initial momentum = 0, final = m_A*v_A + m_B*v_B
            p_final = m_A * v_A + m_B * v_B
            if not math.isclose(p_final, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Momentum conservation violated during spring release: p_final = {p_final:.3f} ≠ 0")
                success = False
            else:
                logs.append(f"PASS: Momentum conservation during spring release: p_final ≈ 0")
        except Exception as e:
            logs.append(f"ERROR in momentum check: {traceback.format_exc()}")
            success = False
        
        # Step 2: Energy conservation during spring release
        try:
            m_A, m_B, v_A = self.inputs['m_A'], self.inputs['m_B'], self.inputs['v_A']
            v_B, E_P = self.results['v_B'], self.results['E_P']
            KE_A = 0.5 * m_A * v_A**2
            KE_B = 0.5 * m_B * v_B**2
            total_KE = KE_A + KE_B
            if not math.isclose(total_KE, E_P, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation violated: KE_total = {total_KE:.3f} ≠ E_P = {E_P:.3f}")
                success = False
            else:
                logs.append(f"PASS: Energy conservation: KE_total ≈ E_P ≈ {total_KE:.3f} J")
        except Exception as e:
            logs.append(f"ERROR in energy check: {traceback.format_exc()}")
            success = False
        
        # Step 3: Momentum conservation during collision (A+B+C system)
        try:
            m_A, m_B, m_C = self.inputs['m_A'], self.inputs['m_B'], self.inputs['m_C']
            v_C = self.results['v_C']
            # Before collision: p_A + p_B + p_C = m_A*v_A + m_B*v_B + m_C*0 = 0 (from step1)
            # After collision: (m_A + m_B + m_C) * v_C
            p_after = (m_A + m_B + m_C) * v_C
            if not math.isclose(p_after, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Momentum conservation violated during collision: p_after = {p_after:.3f} ≠ 0")
                success = False
            else:
                logs.append(f"PASS: Momentum conservation during collision: p_after ≈ 0")
        except Exception as e:
            logs.append(f"ERROR in collision momentum check: {traceback.format_exc()}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Verify v_B from momentum equation
        try:
            m_A, m_B, v_A = self.inputs['m_A'], self.inputs['m_B'], self.inputs['v_A']
            v_B_expected = - (m_A * v_A) / m_B
            v_B_actual = self.results['v_B']
            if not math.isclose(v_B_actual, v_B_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B mismatch: expected {v_B_expected:.3f}, got {v_B_actual:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_B consistent with momentum equation: {v_B_actual:.3f} m/s")
        except Exception as e:
            logs.append(f"ERROR in v_B consistency: {traceback.format_exc()}")
            success = False
        
        # Verify E_P from energy equation
        try:
            m_A, m_B, v_A = self.inputs['m_A'], self.inputs['m_B'], self.inputs['v_A']
            v_B = self.results['v_B']
            E_P_expected = 0.5 * m_A * v_A**2 + 0.5 * m_B * v_B**2
            E_P_actual = self.results['E_P']
            if not math.isclose(E_P_actual, E_P_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_P mismatch: expected {E_P_expected:.3f}, got {E_P_actual:.3f}")
                success = False
            else:
                logs.append(f"PASS: E_P consistent with energy equation: {E_P_actual:.3f} J")
        except Exception as e:
            logs.append(f"ERROR in E_P consistency: {traceback.format_exc()}")
            success = False
        
        # Verify v_C from momentum equation
        try:
            m_A, m_B, m_C = self.inputs['m_A'], self.inputs['m_B'], self.inputs['m_C']
            v_C_expected = 0.0  # From conservation: 0 = (m_A + m_B + m_C) * v_C
            v_C_actual = self.results['v_C']
            if not math.isclose(v_C_actual, v_C_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_C mismatch: expected {v_C_expected:.3f}, got {v_C_actual:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_C consistent with momentum equation: {v_C_actual:.3f} m/s")
        except Exception as e:
            logs.append(f"ERROR in v_C consistency: {traceback.format_exc()}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def test_monotonicity(self):
        """Perturb mass m_A and check expected trends."""
        logs = []
        success = True
        
        original_m_A = self.inputs['m_A']
        perturbed_m_A = original_m_A * 1.1  # Increase by 10%
        
        try:
            # Recalculate v_B for perturbed mass (keeping v_A fixed at 6 m/s)
            v_B_perturbed = - (perturbed_m_A * self.inputs['v_A']) / self.inputs['m_B']
            v_B_original = self.results['v_B']
            
            # Trend: If m_A increases, |v_B| should increase (since v_B = - (m_A*v_A)/m_B)
            if abs(v_B_perturbed) <= abs(v_B_original):
                logs.append(f"FAIL: Monotonicity violated: |v_B| should increase when m_A increases")
                success = False
            else:
                logs.append(f"PASS: |v_B| increases with m_A (as expected)")
            
            # Recalculate E_P for perturbed mass
            E_P_perturbed = 0.5 * perturbed_m_A * self.inputs['v_A']**2 + 0.5 * self.inputs['m_B'] * v_B_perturbed**2
            E_P_original = self.results['E_P']
            
            # Trend: If m_A increases, E_P should increase (more mass → more stored energy)
            if E_P_perturbed <= E_P_original:
                logs.append(f"FAIL: Monotonicity violated: E_P should increase when m_A increases")
                success = False
            else:
                logs.append(f"PASS: E_P increases with m_A (as expected)")
            
            # v_C remains 0 regardless (from momentum conservation)
            logs.append("INFO: v_C remains 0 for any m_A (momentum conservation)")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success."""
        self.logs = ["PhysicsChecker Report:", "=" * 40]
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        overall = inv_ok and trace_ok and mono_ok
        
        # Summary
        self.logs.append("=" * 40)
        self.logs.append(f"Overall: {'PASS' if overall else 'FAIL'}")
        self.logs.append(f"  Invariants: {'PASS' if inv_ok else 'FAIL'}")
        self.logs.append(f"  Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.logs.append(f"  Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
        
        return overall, self.logs