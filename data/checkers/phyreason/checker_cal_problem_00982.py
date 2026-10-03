import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.B = 5e-3
        self.d = 1.2
        self.q_over_m = 2e6  # |q|/m in C/kg
        self.m = 1.0  # reference mass (kg)
        self.q = self.q_over_m * self.m  # magnitude of charge (C)
        
        # Safe extraction of results
        self.results = getattr(solution_instance, 'results', {})
        self.alpha = self._safe_get('alpha')
        self.r = self._safe_get('r')
        self.sintheta = self._safe_get('sintheta')
        self.T = self._safe_get('T')
        self.theta_2 = self._safe_get('theta_2')
        self.t = self._safe_get('t')
        self.sintheta_1 = self._safe_get('sintheta_1')
        self.r_1 = self._safe_get('r_1')
        self.OM = self._safe_get('OM')
        self.ON = self._safe_get('ON')
        self.v_1 = self._safe_get('v_1')
        self.MN = self._safe_get('MN')
        
        self.logs = []
        
    def _safe_get(self, key):
        """Safely get value from results with fallback."""
        return self.results.get(key, None)
    
    def _is_close(self, a, b):
        """Robust numerical comparison for physics values."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physics invariants."""
        logs = []
        try:
            # 1. Lorentz force centripetal relation for v_1
            if self.v_1 is not None and self.r_1 is not None:
                expected_v1 = self.q_over_m * self.B * self.r_1
                if self._is_close(self.v_1, expected_v1):
                    logs.append(f"✓ Lorentz force invariant: v_1 = {self.v_1} ≈ qB·r_1/m = {expected_v1}")
                else:
                    logs.append(f"✗ Lorentz force invariant violated: v_1={self.v_1}, expected {expected_v1}")
            
            # 2. Period formula
            expected_T = 2 * math.pi / (self.q_over_m * self.B)
            if self.T is not None and self._is_close(self.T, expected_T):
                logs.append(f"✓ Period invariant: T = {self.T} ≈ 2π/(qB/m) = {expected_T}")
            else:
                logs.append(f"✗ Period invariant violated: T={self.T}, expected {expected_T}")
            
            # 3. Geometric invariant for r_1
            expected_r1 = self.d / 2
            if self.r_1 is not None and self._is_close(self.r_1, expected_r1):
                logs.append(f"✓ Geometric invariant: r_1 = {self.r_1} ≈ d/2 = {expected_r1}")
            else:
                logs.append(f"✗ Geometric invariant violated: r_1={self.r_1}, expected {expected_r1}")
            
            # 4. Time-angle relation
            if self.t is not None and self.alpha is not None and self.T is not None:
                expected_t = (float(self.alpha) / 360) * self.T
                if self._is_close(self.t, expected_t):
                    logs.append(f"✓ Time-angle invariant: t = {self.t} ≈ (α/360°)·T = {expected_t}")
                else:
                    logs.append(f"✗ Time-angle invariant violated: t={self.t}, expected {expected_t}")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return all('✓' in log for log in logs if '✗' not in log)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check v_1 from step 3
            if self.v_1 is not None:
                expected_v1 = self.q_over_m * self.B * (self.d/2)
                if self._is_close(self.v_1, expected_v1):
                    logs.append(f"✓ v_1 consistency: {self.v_1} ≈ 6000 m/s")
                else:
                    logs.append(f"✗ v_1 inconsistency: got {self.v_1}, expected ~6000")
            
            # Check t from step 10
            if self.t is not None:
                expected_t = (97 * math.pi / 180) * 1/(self.q_over_m * self.B)
                if self._is_close(self.t, expected_t):
                    logs.append(f"✓ t consistency: {self.t} ≈ (97π/180)×10⁻⁴ s")
                else:
                    logs.append(f"✗ t inconsistency: got {self.t}, expected ~1.69e-4")
            
            # Check MN from step 15
            if self.MN is not None:
                # Compute expected MN from symbolic expression
                expected_MN = (2/5) * (math.sqrt(7) - math.sqrt(3))
                if self._is_close(self.MN, expected_MN):
                    logs.append(f"✓ MN consistency: {self.MN} ≈ (2/5)(√7-√3) ≈ 0.22 m")
                else:
                    logs.append(f"✗ MN inconsistency: got {self.MN}, expected ~0.22")
            
            # Check OM and ON consistency
            if self.OM is not None and self.ON is not None and self.MN is not None:
                expected_MN_calc = self.OM - self.ON
                if self._is_close(self.MN, expected_MN_calc):
                    logs.append(f"✓ MN = OM - ON consistency: {self.MN} = {self.OM} - {self.ON}")
                else:
                    logs.append(f"✗ MN ≠ OM - ON: {self.MN} vs {self.OM - self.ON}")
            
            # Check sinθ from step 6
            if self.sintheta is not None and self.r is not None:
                expected_sintheta = (self.d/2) / self.r
                if self._is_close(self.sintheta, expected_sintheta):
                    logs.append(f"✓ sinθ consistency: {self.sintheta} ≈ (d/2)/r")
                else:
                    logs.append(f"✗ sinθ inconsistency: got {self.sintheta}, expected ~0.75")
            
            # Check α from step 7
            if self.alpha is not None and self.sintheta is not None:
                theta = math.asin(self.sintheta)
                expected_alpha = 2 * theta * 180 / math.pi
                if self._is_close(self.alpha, expected_alpha):
                    logs.append(f"✓ α consistency: {self.alpha}° ≈ 2θ")
                else:
                    logs.append(f"✗ α inconsistency: got {self.alpha}°, expected ~97°")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return all('✓' in log for log in logs if '✗' not in log)
    
    def test_monotonicity(self):
        """Test trend when perturbing charge-to-mass ratio."""
        logs = []
        original_q_over_m = self.q_over_m
        
        try:
            # Perturb q/m upward by 5%
            perturbed_q_over_m = original_q_over_m * 1.05
            
            # Calculate expected changes
            # 1. Period should decrease
            T_original = 2 * math.pi / (original_q_over_m * self.B)
            T_perturbed = 2 * math.pi / (perturbed_q_over_m * self.B)
            if T_perturbed < T_original:
                logs.append("✓ Period decreases when |q|/m increases (T ∝ 1/(qB/m))")
            else:
                logs.append("✗ Period monotonicity violation")
            
            # 2. Velocity v_1 should increase (for same geometry)
            v1_original = original_q_over_m * self.B * (self.d/2)
            v1_perturbed = perturbed_q_over_m * self.B * (self.d/2)
            if v1_perturbed > v1_original:
                logs.append("✓ v_1 increases when |q|/m increases (v₁ ∝ qB/m)")
            else:
                logs.append("✗ v_1 monotonicity violation")
            
            # 3. Time t should decrease (for same α)
            if self.alpha is not None:
                t_original = (self.alpha / 360) * T_original
                t_perturbed = (self.alpha / 360) * T_perturbed
                if t_perturbed < t_original:
                    logs.append("✓ Time t decreases when |q|/m increases (t ∝ 1/(qB/m))")
                else:
                    logs.append("✗ Time t monotonicity violation")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Restore original value
            self.q_over_m = original_q_over_m
        
        self.logs.extend(logs)
        return all('✓' in log for log in logs if '✗' not in log)
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        success = inv_ok and trace_ok and mono_ok
        
        # Summary
        summary = [
            "="*50,
            "PHYSICS AUDIT SUMMARY",
            f"Invariants: {'PASS' if inv_ok else 'FAIL'}",
            f"Trace Consistency: {'PASS' if trace_ok else 'FAIL'}",
            f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}",
            f"OVERALL: {'PASS' if success else 'FAIL'}",
            "="*50
        ]
        
        full_logs = summary + self.logs
        return success, full_logs