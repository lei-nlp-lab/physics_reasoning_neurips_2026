import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', 2.0))
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v0', 1.0))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.81))
        
        # Safe attribute access for results with fallback
        self.results = {}
        self.results['v'] = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        self.results['t'] = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        self.results['x'] = getattr(solution_instance, 'x', solution_instance.results.get('x', 0.0))
        self.results['delta_E'] = getattr(solution_instance, 'delta_E', solution_instance.results.get('delta_E', 0.0))
        
        # Set up symbolic variables for invariant testing
        self.m_sym, self.v0_sym, self.g_sym, self.h_sym = sp.symbols('m v0 g h', positive=True)
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Conservation of momentum during collision (horizontal direction)
            # Initial momentum: m*v0 + 0 (block stationary)
            # Final momentum: m*(v0/2) + M*v
            # According to step_1: v = v0/4 with M=2m
            
            # Symbolic check
            v_expected = self.v0_sym / 4
            momentum_initial = self.m_sym * self.v0_sym
            momentum_final = self.m_sym * (self.v0_sym / 2) + (2*self.m_sym) * v_expected
            momentum_eq = sp.Eq(momentum_initial, momentum_final)
            
            if momentum_eq.simplify():
                logs.append("✓ Momentum conservation satisfied symbolically")
            else:
                logs.append(f"✗ Momentum conservation failed symbolically: {momentum_eq}")
            
            # Numerical check with tolerance
            v_expected_num = self.v0 / 4
            if math.isclose(self.results['v'], v_expected_num, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Numerical v = {self.results['v']:.4f} matches expected {v_expected_num:.4f}")
            else:
                logs.append(f"✗ Numerical v mismatch: got {self.results['v']:.4f}, expected {v_expected_num:.4f}")
            
            # 2. Energy loss calculation
            # ΔE = initial KE - final KE (bullet + block)
            # Initial KE: 0.5*m*v0^2
            # Final KE: 0.5*m*(v0/2)^2 + 0.5*M*v^2
            # With M=2m and v=v0/4: ΔE = (5/16)*m*v0^2
            
            # Symbolic check
            delta_E_expected = (5*self.m_sym*self.v0_sym**2) / 16
            KE_initial = 0.5 * self.m_sym * self.v0_sym**2
            KE_final = 0.5 * self.m_sym * (self.v0_sym/2)**2 + 0.5 * (2*self.m_sym) * (self.v0_sym/4)**2
            energy_eq = sp.Eq(delta_E_expected, KE_initial - KE_final)
            
            if energy_eq.simplify():
                logs.append("✓ Energy loss formula satisfied symbolically")
            else:
                logs.append(f"✗ Energy loss formula failed symbolically: {energy_eq}")
            
            # Numerical check
            delta_E_expected_num = (5 * self.m * self.v0**2) / 16
            if math.isclose(self.results['delta_E'], delta_E_expected_num, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Numerical ΔE = {self.results['delta_E']:.4f} matches expected {delta_E_expected_num:.4f}")
            else:
                logs.append(f"✗ Numerical ΔE mismatch: got {self.results['delta_E']:.4f}, expected {delta_E_expected_num:.4f}")
            
            # 3. Check that energy loss is positive
            if self.results['delta_E'] > 0:
                logs.append("✓ Energy loss is positive (as expected for inelastic interaction)")
            else:
                logs.append(f"✗ Energy loss should be positive, got {self.results['delta_E']:.4f}")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Check that v, t, and x are internally consistent:
            # 1. t should satisfy h = 0.5*g*t^2
            t_expected = math.sqrt(2 * self.h / self.g)
            if math.isclose(self.results['t'], t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Fall time t = {self.results['t']:.4f} matches kinematic equation")
            else:
                logs.append(f"✗ Fall time mismatch: got {self.results['t']:.4f}, expected sqrt(2h/g) = {t_expected:.4f}")
            
            # 2. x should satisfy x = v*t
            x_expected = self.results['v'] * self.results['t']
            if math.isclose(self.results['x'], x_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Horizontal distance x = {self.results['x']:.4f} matches v*t = {x_expected:.4f}")
            else:
                logs.append(f"✗ Horizontal distance mismatch: got {self.results['x']:.4f}, expected v*t = {x_expected:.4f}")
            
            # 3. Overall expression: x should equal (v0/4)*sqrt(2h/g)
            x_full_expected = (self.v0 / 4) * math.sqrt(2 * self.h / self.g)
            if math.isclose(self.results['x'], x_full_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ x matches full expression (v0/4)*sqrt(2h/g) = {x_full_expected:.4f}")
            else:
                logs.append(f"✗ x doesn't match full expression: got {self.results['x']:.4f}, expected {x_full_expected:.4f}")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_v0 = self.v0
        try:
            # Perturb v0 upward by 10%
            perturbed_v0 = original_v0 * 1.1
            
            # Recalculate expected values using formulas
            v_perturbed_expected = perturbed_v0 / 4
            t_expected = math.sqrt(2 * self.h / self.g)  # unchanged
            x_perturbed_expected = v_perturbed_expected * t_expected
            delta_E_perturbed_expected = (5 * self.m * perturbed_v0**2) / 16
            
            # Check monotonicity:
            # 1. v should increase with v0
            if v_perturbed_expected > self.results['v']:
                logs.append("✓ v increases when v0 increases (linear relationship v = v0/4)")
            else:
                logs.append("✗ v should increase with v0")
            
            # 2. x should increase with v0
            if x_perturbed_expected > self.results['x']:
                logs.append("✓ x increases when v0 increases (linear relationship x ∝ v0)")
            else:
                logs.append("✗ x should increase with v0")
            
            # 3. ΔE should increase with v0 (quadratically)
            if delta_E_perturbed_expected > self.results['delta_E']:
                logs.append("✓ ΔE increases when v0 increases (quadratic relationship ΔE ∝ v0²)")
            else:
                logs.append("✗ ΔE should increase with v0")
                
            # Restore original v0
            self.v0 = original_v0
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            # Ensure v0 is restored even if error occurs
            self.v0 = original_v0
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        all_logs.append("=== PhysicsChecker Report ===")
        all_logs.append(f"Inputs: m={self.m}, M={self.M}, v0={self.v0}, h={self.h}, g={self.g}")
        all_logs.append(f"Results: v={self.results['v']:.4f}, t={self.results['t']:.4f}, x={self.results['x']:.4f}, ΔE={self.results['delta_E']:.4f}")
        all_logs.append("")
        
        # Run tests
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"--- {test_name} ---")
            test_logs = test_func()
            all_logs.extend(test_logs)
            
            # Check for any errors or failures in logs
            for log in test_logs:
                if log.startswith("✗") or "Error" in log:
                    success = False
            
            all_logs.append("")
        
        # Overall verdict
        if success:
            all_logs.append("=== All checks passed ===")
        else:
            all_logs.append("=== Some checks failed ===")
        
        return success, all_logs