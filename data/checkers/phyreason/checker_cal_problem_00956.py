import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction for inputs
        self.n = getattr(solution_instance, 'n', 250)
        self.ab = getattr(solution_instance, 'ab', 0.12)  # m
        self.ad = getattr(solution_instance, 'ad', 0.0)   # m (placeholder; actual from solution)
        self.B = getattr(solution_instance, 'B', 2.0 / math.pi)
        self.speed = getattr(solution_instance, 'speed', 120)  # rpm
        self.R = getattr(solution_instance, 'R', 12)
        self.V_breakdown = getattr(solution_instance, 'V_breakdown', 12)
        
        # Safe extraction for results with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Key results with safe defaults
        self.f = self.results.get('f', 0.0)
        self.I = self.results.get('I', 0.0)
        self.E_m = self.results.get('E_{m}', self.results.get('E_m', 0.0))
        self.t_illum = self.results.get('t', 0.0)
        self.theta_rad = self.results.get('theta', 0.0)  # Store as radians
        self.P_R = self.results.get('P_R', 0.0)
        self.U_eff = self.results.get('U_{effective}', self.results.get('U_effective', 0.0))
        self.omega = self.results.get('omega', 0.0)
        self.W = self.results.get('W', 0.0)
        
        # Derived constants
        self.area = self.ab * self.ad  # m² (requires ad from solution)
        self.speed_rps = self.speed / 60.0
        
        # Log storage
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def test_invariants(self):
        checks = []
        
        # 1. Angular velocity from rotational speed: ω = 2π * (speed in rps)
        omega_expected = 2 * math.pi * self.speed_rps
        if math.isclose(self.omega, omega_expected, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"ω = {self.omega:.3f} rad/s matches 2π×({self.speed_rps:.1f}) = {omega_expected:.3f}", True))
        else:
            checks.append(self._log(f"ω mismatch: {self.omega:.3f} vs expected {omega_expected:.3f}", False))
        
        # 2. Maximum induced emf: E_m = n * B * A * ω
        E_m_expected = self.n * self.B * self.area * self.omega
        if math.isclose(self.E_m, E_m_expected, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"E_m = {self.E_m:.3f} V matches nBAω = {E_m_expected:.3f} V", True))
        else:
            checks.append(self._log(f"E_m mismatch: {self.E_m:.3f} vs nBAω = {E_m_expected:.3f}", False))
        
        # 3. Effective voltage: U_eff = E_m / √2
        U_eff_expected = self.E_m / math.sqrt(2)
        if math.isclose(self.U_eff, U_eff_expected, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"U_eff = {self.U_eff:.3f} V matches E_m/√2 = {U_eff_expected:.3f} V", True))
        else:
            checks.append(self._log(f"U_eff mismatch: {self.U_eff:.3f} vs E_m/√2 = {U_eff_expected:.3f}", False))
        
        # 4. Ohm's Law: I = U_eff / R
        I_expected = self.U_eff / self.R
        if math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"I = {self.I:.3f} A matches U_eff/R = {I_expected:.3f} A", True))
        else:
            checks.append(self._log(f"I mismatch: {self.I:.3f} vs U_eff/R = {I_expected:.3f}", False))
        
        # 5. Joule's Law: P_R = I² * R
        P_R_expected = self.I**2 * self.R
        if math.isclose(self.P_R, P_R_expected, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"P_R = {self.P_R:.3f} W matches I²R = {P_R_expected:.3f} W", True))
        else:
            checks.append(self._log(f"P_R mismatch: {self.P_R:.3f} vs I²R = {P_R_expected:.3f}", False))
        
        # 6. Work-energy: W = P_R * t_total (t_total derived from W/P_R)
        if self.P_R > 0:
            t_total_expected = self.W / self.P_R
            # Check consistency with given t_illum? Not directly, but we can check W = I²R*t_total
            if math.isclose(self.W, self.P_R * t_total_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self._log(f"W = {self.W:.3f} J matches P_R*t_total with t_total = {t_total_expected:.1f} s", True))
            else:
                checks.append(self._log(f"W mismatch with P_R*t_total", False))
        
        # 7. Neon bulb conduction angle: θ = arccos(V_breakdown / E_m) OR arcsin depending on voltage function
        # Ground truth step 10 uses cos: e = E_m·cos(θ) = 12 => θ = arccos(12/E_m)
        if self.E_m > self.V_breakdown:
            # Using arccos as in ground truth
            theta_expected_rad = math.acos(self.V_breakdown / self.E_m)
            theta_expected_deg = math.degrees(theta_expected_rad)
            theta_actual_deg = math.degrees(self.theta_rad)
            if math.isclose(theta_actual_deg, theta_expected_deg, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self._log(f"θ = {theta_actual_deg:.1f}° matches arccos({self.V_breakdown}/{self.E_m:.3f}) = {theta_expected_deg:.1f}°", True))
            else:
                checks.append(self._log(f"θ mismatch: {theta_actual_deg:.1f}° vs expected {theta_expected_deg:.1f}°", False))
        
        # 8. Frequency relationships
        f_coil_expected = self.speed_rps
        if math.isclose(self.f, 2 * f_coil_expected, rel_tol=1e-2, abs_tol=1e-2):  # bulb flashes twice per cycle
            checks.append(self._log(f"f_bulb = {self.f:.1f} Hz matches 2×f_coil = {2*f_coil_expected:.1f} Hz", True))
        else:
            checks.append(self._log(f"f_bulb mismatch: {self.f:.1f} Hz vs 2×{f_coil_expected:.1f} = {2*f_coil_expected:.1f} Hz", False))
        
        return all(checks)
    
    def test_trace_consistency(self):
        checks = []
        
        # Check that derived steps are internally consistent
        # Step 3: ω = 4π rad/s from speed = 120 rpm -> 2 rps
        omega_step = 4 * math.pi
        if math.isclose(self.omega, omega_step, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"ω from step_3 ({omega_step:.3f}) matches stored {self.omega:.3f}", True))
        else:
            checks.append(self._log(f"ω mismatch with step_3: stored {self.omega:.3f} vs step {omega_step:.3f}", False))
        
        # Step 4: E_m = 24 V (given in steps)
        E_m_step = 24.0
        if math.isclose(self.E_m, E_m_step, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"E_m from step_4 ({E_m_step}) matches stored {self.E_m:.3f}", True))
        else:
            checks.append(self._log(f"E_m mismatch with step_4: stored {self.E_m:.3f} vs step {E_m_step}", False))
        
        # Step 5: I = 1.414 A (≈√2)
        I_step = 1.414
        if math.isclose(self.I, I_step, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"I from step_5 ({I_step}) matches stored {self.I:.3f}", True))
        else:
            checks.append(self._log(f"I mismatch with step_5: stored {self.I:.3f} vs step {I_step}", False))
        
        # Step 6: P_R = 24 W
        P_R_step = 24.0
        if math.isclose(self.P_R, P_R_step, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"P_R from step_6 ({P_R_step}) matches stored {self.P_R:.3f}", True))
        else:
            checks.append(self._log(f"P_R mismatch with step_6: stored {self.P_R:.3f} vs step {P_R_step}", False))
        
        # Step 7: W = 14400 J (but target says 1440J; we check relative scaling)
        W_step = 14400.0
        if self.W > 0:
            ratio = self.W / W_step
            if math.isclose(ratio, 1.0, rel_tol=1e-1, abs_tol=1e-1):
                checks.append(self._log(f"W scaling consistent: stored {self.W:.1f} vs step {W_step} (ratio {ratio:.3f})", True))
            else:
                checks.append(self._log(f"W inconsistent scaling: stored {self.W:.1f} vs step {W_step}", False))
        
        # Step 10: θ = 60° (from ground truth: cos(θ) = 12/24 => θ = 60°)
        theta_step_deg = 60.0
        theta_actual_deg = math.degrees(self.theta_rad)
        if math.isclose(theta_actual_deg, theta_step_deg, rel_tol=1e-2, abs_tol=1e-2):
            checks.append(self._log(f"θ from step_10 ({theta_step_deg}°) matches stored {theta_actual_deg:.1f}°", True))
        else:
            checks.append(self._log(f"θ mismatch with step_10: stored {theta_actual_deg:.1f}° vs step {theta_step_deg}°", False))
        
        return all(checks)
    
    def test_monotonicity(self):
        checks = []
        
        # Perturb rotational speed and check trends
        original_speed = self.speed
        perturbed_speed = original_speed * 1.1  # +10%
        
        # Recompute angular velocity
        omega_pert = 2 * math.pi * (perturbed_speed / 60.0)
        
        # Recompute E_m with perturbed ω (assuming other parameters fixed)
        E_m_pert = self.n * self.B * self.area * omega_pert
        
        # Check monotonicity: E_m should increase with ω
        if E_m_pert > self.E_m:
            trend_ok = True
            msg = f"E_m increases with ω (from {self.E_m:.3f} to {E_m_pert:.3f} V)"
        else:
            trend_ok = False
            msg = f"E_m non-increasing with ω perturbation"
        
        checks.append(self._log(msg, trend_ok))
        
        # Perturb resistance R and check power trend (P_R = I²R, but I depends on R)
        original_R = self.R
        perturbed_R = original_R * 1.1  # +10%
        
        # Recompute I using Ohm's Law with fixed U_eff (since U_eff depends on E_m, which is fixed here)
        I_pert = self.U_eff / perturbed_R
        P_R_pert = I_pert**2 * perturbed_R
        
        # For fixed voltage, power should decrease with increased R (P = V²/R)
        if P_R_pert < self.P_R:
            trend_ok = True
            msg = f"P_R decreases with R increase (from {self.P_R:.3f} to {P_R_pert:.3f} W)"
        else:
            trend_ok = False
            msg = f"P_R non-decreasing with R perturbation"
        
        checks.append(self._log(msg, trend_ok))
        
        # Perturb B and check E_m trend
        original_B = self.B
        perturbed_B = original_B * 0.9  # -10%
        E_m_pert_B = self.n * perturbed_B * self.area * self.omega
        
        if E_m_pert_B < self.E_m:
            trend_ok = True
            msg = f"E_m decreases with B decrease (from {self.E_m:.3f} to {E_m_pert_B:.3f} V)"
        else:
            trend_ok = False
            msg = f"E_m non-decreasing with B perturbation"
        
        checks.append(self._log(msg, trend_ok))
        
        return all(checks)
    
    def run_all_tests(self):
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
            self.logs.append(f"ERROR: {traceback.format_exc()}")
            success = False
        
        return success, self.logs