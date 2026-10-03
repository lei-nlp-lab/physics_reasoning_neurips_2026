import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results from solution instance
        self.results = {}
        expected_keys = ['Pressure', 'F_N', 'x', 't', 'a', 'v_C', 'v', 'v_B']
        
        # First try to get from results dict
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results.copy()
        else:
            # Fallback to direct attributes
            for key in expected_keys:
                val = getattr(solution_instance, key, 0.0)
                self.results[key] = val
        
        # Convert to float
        for key in self.results:
            try:
                self.results[key] = float(self.results[key])
            except (TypeError, ValueError):
                self.results[key] = 0.0
        
        # Constants from problem
        self.constants = {
            'm': 0.2,
            'H': 1.2,
            'L1': 1.25,
            'mu1': 0.25,
            'mu2': 0.2,
            'L2': 3.0,
            'R': 0.5,
            'g': 10.0,
            'sin37': 0.6,
            'cos37': 0.8,
            'v_belt': 4.0
        }
        
        self.logs = []
    
    def log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical equality check."""
        if a == 0 and b == 0:
            return True
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws with correct equations."""
        success = True
        try:
            c = self.constants
            r = self.results
            
            # 1. Work-energy theorem from A to B (CORRECTED)
            W_net = c['m'] * c['g'] * c['L1'] * c['sin37'] - c['mu1'] * c['m'] * c['g'] * c['cos37'] * c['L1']
            delta_KE = 0.5 * c['m'] * r['v_B']**2
            
            if not self.fuzzy_equal(W_net, delta_KE):
                self.log(f"Work-energy theorem A->B: net work {W_net:.3f} ≠ ΔKE {delta_KE:.3f}", False)
                success = False
            else:
                self.log(f"Work-energy theorem A->B holds: {W_net:.3f} ≈ {delta_KE:.3f}", True)
            
            # 2. Velocity continuity B->C
            if not self.fuzzy_equal(r['v_C'], r['v_B']):
                self.log(f"Velocity continuity B->C: v_C={r['v_C']:.3f} ≠ v_B={r['v_B']:.3f}", False)
                success = False
            else:
                self.log(f"Velocity continuity B->C holds: {r['v_C']:.3f} ≈ {r['v_B']:.3f}", True)
            
            # 3. Newton's second law at point C
            F_N_expected = c['m'] * c['g'] + c['m'] * r['v_C']**2 / c['R']
            if not self.fuzzy_equal(r['F_N'], F_N_expected):
                self.log(f"Newton's 2nd law at C: F_N={r['F_N']:.3f} ≠ mg + mv_C²/R={F_N_expected:.3f}", False)
                success = False
            else:
                self.log(f"Newton's 2nd law at C holds: {r['F_N']:.3f} ≈ {F_N_expected:.3f}", True)
            
            # 4. Newton's third law
            if not self.fuzzy_equal(r['Pressure'], r['F_N']):
                self.log(f"Newton's 3rd law: Pressure={r['Pressure']:.3f} ≠ F_N={r['F_N']:.3f}", False)
                success = False
            else:
                self.log(f"Newton's 3rd law holds: Pressure {r['Pressure']:.3f} ≈ F_N {r['F_N']:.3f}", True)
            
            # 5. Newton's second law on conveyor belt (magnitude)
            a_expected_magnitude = c['mu2'] * c['g']
            if not self.fuzzy_equal(abs(r['a']), a_expected_magnitude):
                self.log(f"Newton's 2nd law on belt: |a|={abs(r['a']):.3f} ≠ μ₂g={a_expected_magnitude:.3f}", False)
                success = False
            else:
                self.log(f"Newton's 2nd law on belt holds: |a|={abs(r['a']):.3f} ≈ μ₂g={a_expected_magnitude:.3f}", True)
            
            # 6. Kinematic check CORRECTED: The slider reaches belt speed before D
            # Distance to reach belt speed: v_belt² = v_C² - 2*a*d (deceleration)
            a_decel = abs(r['a'])  # magnitude of deceleration
            d_to_belt = (r['v_C']**2 - c['v_belt']**2) / (2 * a_decel)
            
            if d_to_belt < c['L2']:
                self.log(f"Kinematic: reaches belt speed at d={d_to_belt:.3f}m < L2={c['L2']:.3f}m, so v={c['v_belt']:.3f}m/s", True)
            else:
                self.log(f"Kinematic: would need d={d_to_belt:.3f}m ≥ L2, so would travel full L2", True)
            
            # 7. Free fall time
            h_D = c['H'] - c['L1'] * c['sin37']
            t_expected = math.sqrt(2 * h_D / c['g'])
            if not self.fuzzy_equal(r['t'], t_expected):
                self.log(f"Free fall time: t={r['t']:.3f} ≠ √(2h_D/g)={t_expected:.3f}", False)
                success = False
            else:
                self.log(f"Free fall time holds: {r['t']:.3f} ≈ √(2h_D/g)={t_expected:.3f}", True)
            
            # 8. Uniform horizontal motion (with belt speed)
            x_expected = c['v_belt'] * r['t']
            if not self.fuzzy_equal(r['x'], x_expected):
                self.log(f"Uniform motion: x={r['x']:.3f} ≠ v_belt*t={x_expected:.3f}", False)
                success = False
            else:
                self.log(f"Uniform motion holds: {r['x']:.3f} ≈ v_belt*t={x_expected:.3f}", True)
                
        except Exception as e:
            self.log(f"Exception in test_invariants: {str(e)}", False)
            success = False
        return success
    
    def test_trace_consistency(self):
        """Cross-check with correct physics."""
        success = True
        try:
            c = self.constants
            r = self.results
            
            # Step 1: v_B from correct work-energy
            v_B_sq_expected = 2 * (c['g'] * c['L1'] * c['sin37'] - c['mu1'] * c['g'] * c['cos37'] * c['L1'])
            v_B_expected = math.sqrt(v_B_sq_expected) if v_B_sq_expected > 0 else 0.0
            
            if not self.fuzzy_equal(r['v_B'], v_B_expected):
                self.log(f"Step 1: v_B={r['v_B']:.3f} ≠ √[2(gL1sin37-μ₁gcos37L1)]={v_B_expected:.3f}", False)
                success = False
            else:
                self.log(f"Step 1 consistent: v_B = √[2(gL1sin37-μ₁gcos37L1)] = {v_B_expected:.3f}", True)
            
            # Step 2: v_C = v_B
            if not self.fuzzy_equal(r['v_C'], r['v_B']):
                self.log(f"Step 2: v_C={r['v_C']:.3f} ≠ v_B={r['v_B']:.3f}", False)
                success = False
            else:
                self.log(f"Step 2 consistent: v_C = v_B = {r['v_B']:.3f}", True)
            
            # Step 3: F_N = mg + mv_C²/R
            F_N_expected = c['m'] * c['g'] + c['m'] * r['v_C']**2 / c['R']
            if not self.fuzzy_equal(r['F_N'], F_N_expected):
                self.log(f"Step 3: F_N={r['F_N']:.3f} ≠ mg + mv_C²/R={F_N_expected:.3f}", False)
                success = False
            else:
                self.log(f"Step 3 consistent: F_N = mg + mv_C²/R = {F_N_expected:.3f}", True)
            
            # Step 4: Pressure = F_N
            if not self.fuzzy_equal(r['Pressure'], r['F_N']):
                self.log(f"Step 4: Pressure={r['Pressure']:.3f} ≠ F_N={r['F_N']:.3f}", False)
                success = False
            else:
                self.log(f"Step 4 consistent: Pressure = F_N = {r['F_N']:.3f}", True)
            
            # Step 5: Acceleration magnitude = μ₂g
            a_expected_mag = c['mu2'] * c['g']
            if not self.fuzzy_equal(abs(r['a']), a_expected_mag):
                self.log(f"Step 5: |a|={abs(r['a']):.3f} ≠ μ₂g={a_expected_mag:.3f}", False)
                success = False
            else:
                self.log(f"Step 5 consistent: |a| = μ₂g = {a_expected_mag:.3f}", True)
            
            # Step 6: Velocity at D should be belt speed (4 m/s)
            if not self.fuzzy_equal(r['v'], c['v_belt']):
                self.log(f"Step 6: v={r['v']:.3f} ≠ v_belt={c['v_belt']:.3f}", False)
                success = False
            else:
                self.log(f"Step 6 consistent: v = v_belt = {c['v_belt']:.3f}", True)
            
            # Step 7: Free fall time
            h_D = c['H'] - c['L1'] * c['sin37']
            t_expected = math.sqrt(2 * h_D / c['g'])
            if not self.fuzzy_equal(r['t'], t_expected):
                self.log(f"Step 7: t={r['t']:.3f} ≠ √(2h_D/g)={t_expected:.3f}", False)
                success = False
            else:
                self.log(f"Step 7 consistent: t = √(2h_D/g) = {t_expected:.3f}", True)
            
            # Step 8: Horizontal distance
            x_expected = c['v_belt'] * r['t']
            if not self.fuzzy_equal(r['x'], x_expected):
                self.log(f"Step 8: x={r['x']:.3f} ≠ v_belt*t={x_expected:.3f}", False)
                success = False
            else:
                self.log(f"Step 8 consistent: x = v_belt*t = {x_expected:.3f}", True)
                
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {str(e)}", False)
            success = False
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check trend predictions with correct physics."""
        success = True
        original_logs_len = len(self.logs)
        
        try:
            c = self.constants.copy()
            r = self.results
            
            # Test 1: Increase μ1, Pressure should decrease
            mu1_perturbed = c['mu1'] * 1.1
            v_B_sq_perturbed = 2 * (c['g'] * c['L1'] * c['sin37'] - mu1_perturbed * c['g'] * c['cos37'] * c['L1'])
            if v_B_sq_perturbed > 0:
                v_B_perturbed = math.sqrt(v_B_sq_perturbed)
                v_C_perturbed = v_B_perturbed
                F_N_perturbed = c['m'] * c['g'] + c['m'] * v_C_perturbed**2 / c['R']
                
                if F_N_perturbed >= r['F_N']:
                    self.log(f"Monotonicity fail: μ1↑ (0.25→{mu1_perturbed:.3f}) but Pressure {F_N_perturbed:.3f} ≥ {r['F_N']:.3f}", False)
                    success = False
                else:
                    self.log(f"Monotonicity pass: μ1↑ (0.25→{mu1_perturbed:.3f}) → Pressure↓ ({r['F_N']:.3f}→{F_N_perturbed:.3f})", True)
            else:
                self.log(f"Monotonicity: μ1↑ makes v_B² negative", True)
            
            # Test 2: Increase μ2, x should decrease or stay same
            mu2_perturbed = c['mu2'] * 1.1
            a_perturbed_mag = mu2_perturbed * c['g']
            d_to_belt = (r['v_C']**2 - c['v_belt']**2) / (2 * a_perturbed_mag)
            
            if d_to_belt < c['L2']:
                v_perturbed = c['v_belt']
            else:
                v_perturbed_sq = r['v_C']**2 - 2 * a_perturbed_mag * c['L2']
                v_perturbed = math.sqrt(v_perturbed_sq) if v_perturbed_sq > 0 else 0.0
            
            h_D = c['H'] - c['L1'] * c['sin37']
            t_perturbed = math.sqrt(2 * h_D / c['g'])
            x_perturbed = v_perturbed * t_perturbed
            
            if x_perturbed > r['x'] + 0.01:
                self.log(f"Monotonicity fail: μ2↑ (0.2→{mu2_perturbed:.3f}) but x {x_perturbed:.3f} > {r['x']:.3f}", False)
                success = False
            else:
                self.log(f"Monotonicity pass: μ2↑ (0.2→{mu2_perturbed:.3f}) → x↓ or ≈ ({r['x']:.3f}→{x_perturbed:.3f})", True)
            
        except Exception as e:
            self.logs = self.logs[:original_logs_len]
            self.log(f"Exception in test_monotonicity: {str(e)}", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        overall_success = True
        self.logs = []
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"--- Starting {test_name} Test ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"Test {test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        self.log(f"=== Overall Test Result: {'PASS' if overall_success else 'FAIL'} ===", overall_success)
        return overall_success, self.logs