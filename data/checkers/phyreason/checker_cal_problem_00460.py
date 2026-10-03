import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 2.0))
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', 10.0))
        self.theta_deg = 37.0  # Fixed from problem
        self.theta_rad = math.radians(self.theta_deg)
        self.s = getattr(solution_instance, 's', solution_instance.results.get('s', 5.0))
        self.mu = getattr(solution_instance, 'mu', solution_instance.results.get('mu', 0.25))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 10.0))
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        # Safe attribute access for results
        self.W = getattr(solution_instance, 'W', solution_instance.results.get('W', 0.0))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        self.W_f = getattr(solution_instance, 'W_f', solution_instance.results.get('W_f', 0.0))
        
        # Store original values for perturbation tests
        self.original_m = float(self.m)
        self.original_F = float(self.F)
        self.logs = []
        
    def log(self, message, success=True):
        """Helper to log test results"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
    
    def test_invariants(self):
        """Verify fundamental physics laws are satisfied"""
        results = []
        
        # 1. Work done by thrust (step 1)
        W_expected = self.F * self.s * self.cos37
        if math.isclose(float(self.W), float(W_expected), rel_tol=1e-2, abs_tol=1e-2):
            results.append(self.log(f"Work by thrust: {self.W:.2f} ≈ F*s*cosθ = {W_expected:.2f}"))
        else:
            results.append(self.log(f"Work by thrust mismatch: {self.W:.2f} ≠ {W_expected:.2f}", False))
        
        # 2. Friction force during thrust (step 2)
        N = self.m * self.g + self.F * self.sin37
        f_expected = self.mu * N
        if math.isclose(float(self.f), float(f_expected), rel_tol=1e-2, abs_tol=1e-2):
            results.append(self.log(f"Friction force: {self.f:.2f} ≈ μ*(mg+Fsinθ) = {f_expected:.2f}"))
        else:
            results.append(self.log(f"Friction force mismatch: {self.f:.2f} ≠ {f_expected:.2f}", False))
        
        # 3. Velocity via work-energy for thrust phase (step 3)
        # Net work during thrust = W_thrust - W_friction_thrust
        W_net = self.W - (self.f * self.s)
        v_expected = math.sqrt(2 * W_net / self.m) if W_net > 0 else 0.0
        if math.isclose(float(self.v), float(v_expected), rel_tol=1e-2, abs_tol=1e-2):
            results.append(self.log(f"Velocity: {self.v:.3f} ≈ sqrt(2*(W-f*s)/m) = {v_expected:.3f}"))
        else:
            results.append(self.log(f"Velocity mismatch: {self.v:.3f} ≠ {v_expected:.3f}", False))
        
        # 4. Total work by friction (step 4) - from energy conservation
        # Initial KE = 0, final KE = 0, so total work by non-conservative forces = 0
        # W_thrust + W_friction_total = 0 → W_friction_total = -W_thrust
        if math.isclose(float(self.W_f), -float(self.W), rel_tol=1e-2, abs_tol=1e-2):
            results.append(self.log(f"Total friction work: {self.W_f:.2f} ≈ -W = {-self.W:.2f}"))
        else:
            results.append(self.log(f"Total friction work mismatch: {self.W_f:.2f} ≠ {-self.W:.2f}", False))
        
        # 5. Friction force after thrust (consistency check)
        # During thrust: f_thrust = μ*(mg + Fsinθ) = 6.5 N
        # After thrust: f_slide = μ*mg = 0.25*2*10 = 5 N
        # The object slides distance d = v²/(2*μ*g) with f_slide
        # Work by friction after thrust = -f_slide * d = -5 * d
        f_slide = self.mu * self.m * self.g
        if self.v > 0:
            d_slide = self.v**2 / (2 * self.mu * self.g)
            W_f_slide_expected = -f_slide * d_slide
            W_f_thrust_actual = -self.f * self.s
            W_f_total_actual = self.W_f
            # Check: W_f_total ≈ W_f_thrust + W_f_slide
            W_f_total_expected = W_f_thrust_actual + W_f_slide_expected
            if math.isclose(float(W_f_total_actual), float(W_f_total_expected), rel_tol=1e-2, abs_tol=1e-2):
                results.append(self.log(f"Friction work decomposition consistent: total = {W_f_total_actual:.2f}, thrust phase = {W_f_thrust_actual:.2f}, sliding phase = {W_f_slide_expected:.2f}"))
            else:
                results.append(self.log(f"Friction work decomposition inconsistent: total {W_f_total_actual:.2f} ≠ sum {W_f_total_expected:.2f}", False))
        
        return all(results)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        results = []
        
        # 1. Check work-energy theorem for thrust phase
        # W_net = ΔK = ½mv² - 0
        delta_K = 0.5 * self.m * self.v**2
        W_net_calc = self.W - self.f * self.s
        if math.isclose(float(delta_K), float(W_net_calc), rel_tol=1e-2, abs_tol=1e-2):
            results.append(self.log(f"Work-energy for thrust phase: ΔK = {delta_K:.2f} ≈ W_net = {W_net_calc:.2f}"))
        else:
            results.append(self.log(f"Work-energy violation: ΔK = {delta_K:.2f}, W_net = {W_net_calc:.2f}", False))
        
        # 2. Check energy conservation for entire process
        # Initial KE = 0, final KE = 0
        # W_thrust + W_friction_total = 0
        energy_balance = self.W + self.W_f
        if math.isclose(float(energy_balance), 0.0, rel_tol=1e-2, abs_tol=1e-2):
            results.append(self.log(f"Energy conservation: W + W_f = {energy_balance:.2f} ≈ 0"))
        else:
            results.append(self.log(f"Energy violation: W + W_f = {energy_balance:.2f} ≠ 0", False))
        
        # 3. Verify friction forces are consistent with normal forces
        N_during = self.m * self.g + self.F * self.sin37
        f_during_expected = self.mu * N_during
        if math.isclose(float(self.f), float(f_during_expected), rel_tol=1e-2, abs_tol=1e-2):
            results.append(self.log(f"Friction during thrust consistent: f = {self.f:.2f} ≈ μN = {f_during_expected:.2f}"))
        else:
            results.append(self.log(f"Friction inconsistency: f = {self.f:.2f}, μN = {f_during_expected:.2f}", False))
        
        return all(results)
    
    def test_monotonicity(self):
        """Perturb inputs and check trends"""
        results = []
        
        # Perturb mass by +10%
        try:
            original_m = float(self.m)
            perturbed_m = original_m * 1.1
            
            # Recalculate expected results using formulas
            # Work by thrust remains same (independent of m)
            W_original = self.W
            
            # Friction force during thrust: f = μ*(mg + Fsinθ)
            f_original = self.f
            f_perturbed = self.mu * (perturbed_m * self.g + self.F * self.sin37)
            
            # Velocity: v = sqrt(2*(W - f*s)/m)
            v_original = self.v
            W_net_perturbed = self.W - (f_perturbed * self.s)
            if W_net_perturbed > 0:
                v_perturbed = math.sqrt(2 * W_net_perturbed / perturbed_m)
                
                # Check monotonicity trends
                # f should increase with m (since normal force increases)
                if f_perturbed > f_original:
                    results.append(self.log(f"Friction force increases with mass: f({original_m:.1f}) = {f_original:.2f} < f({perturbed_m:.1f}) = {f_perturbed:.2f}"))
                else:
                    results.append(self.log(f"Friction force trend wrong: should increase with mass", False))
                
                # v should decrease with m (larger mass means less acceleration)
                if v_perturbed < v_original:
                    results.append(self.log(f"Velocity decreases with mass: v({original_m:.1f}) = {v_original:.3f} > v({perturbed_m:.1f}) = {v_perturbed:.3f}"))
                else:
                    results.append(self.log(f"Velocity trend wrong: should decrease with mass", False))
            else:
                results.append(self.log(f"Cannot compute v_perturbed: W_net = {W_net_perturbed:.2f} ≤ 0"))
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            results.append(self.log(f"Monotonicity test failed: {str(e)}", False))
            traceback.print_exc()
        
        # Perturb force magnitude by +10%
        try:
            original_F = float(self.F)
            perturbed_F = original_F * 1.1
            
            # Recalculate expected results
            # Work by thrust: W = F*s*cosθ
            W_original = self.W
            W_perturbed = perturbed_F * self.s * self.cos37
            
            # Friction force: f = μ*(mg + Fsinθ)
            f_original = self.f
            f_perturbed = self.mu * (self.m * self.g + perturbed_F * self.sin37)
            
            # Check trends
            # Both W and f should increase with F
            if W_perturbed > W_original:
                results.append(self.log(f"Work increases with force: W({original_F:.1f}) = {W_original:.2f} < W({perturbed_F:.1f}) = {W_perturbed:.2f}"))
            else:
                results.append(self.log(f"Work trend wrong: should increase with force", False))
            
            if f_perturbed > f_original:
                results.append(self.log(f"Friction increases with force: f({original_F:.1f}) = {f_original:.2f} < f({perturbed_F:.1f}) = {f_perturbed:.2f}"))
            else:
                results.append(self.log(f"Friction trend wrong: should increase with force", False))
            
            # Restore original force
            self.F = original_F
            
        except Exception as e:
            results.append(self.log(f"Force perturbation test failed: {str(e)}", False))
            traceback.print_exc()
        
        return all(results)
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []
        success = True
        
        try:
            if not self.test_invariants():
                success = False
            
            if not self.test_trace_consistency():
                success = False
            
            if not self.test_monotonicity():
                success = False
                
        except Exception as e:
            self.logs.append(f"ERROR: Test execution failed: {str(e)}")
            success = False
        
        # Final summary
        if success:
            self.logs.append("SUMMARY: All physics checks passed")
        else:
            self.logs.append("SUMMARY: Some physics checks failed")
        
        return success, self.logs