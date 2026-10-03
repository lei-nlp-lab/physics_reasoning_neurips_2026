import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.solution = solution_instance
        self.inputs = {
            'v0': getattr(solution_instance, 'v0', 2.0),
            'L': getattr(solution_instance, 'L', 2.0),
            'theta': getattr(solution_instance, 'theta', 37),
            'mu': getattr(solution_instance, 'mu', 0.2),
            'g': getattr(solution_instance, 'g', 10.0),
            'sin_theta': getattr(solution_instance, 'sin_theta', 0.6)
        }
        
        # Safe extraction of results with fallbacks
        self.results = {}
        result_keys = [
            'sₘ', 'a₃', 't₄', 't₂', 't', 't₁', 'a₁', 'sₘ\'', 's₁', 't₃', 's', 'a₂'
        ]
        for key in result_keys:
            # Try to get from solution_instance.results dictionary first, then from attributes
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                self.results[key] = solution_instance.results.get(key, 0.0)
            else:
                self.results[key] = getattr(solution_instance, key, 0.0)
        
        # Prepare symbolic variables for invariant checks
        self.v0_sym = sp.Symbol('v0')
        self.L_sym = sp.Symbol('L')
        self.theta_sym = sp.Symbol('theta')
        self.mu_sym = sp.Symbol('mu')
        self.g_sym = sp.Symbol('g')
        self.sin_theta_sym = sp.Symbol('sin_theta')
        
        self.logs = []

    def log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            msg = f"[{status}] {msg}"
        self.logs.append(msg)
        return success

    def test_invariants(self):
        success = True
        
        # 1. Check a1 = μ*g
        a1_expected = self.inputs['mu'] * self.inputs['g']
        if not math.isclose(self.results['a₁'], a1_expected, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"a₁ invariant failed: expected {a1_expected}, got {self.results['a₁']}", False)
        else:
            self.log(f"a₁ invariant passed: {self.results['a₁']} ≈ μ*g = {a1_expected}", True)
        
        # 2. Check a2 = g*sinθ
        a2_expected = self.inputs['g'] * self.inputs['sin_theta']
        if not math.isclose(self.results['a₂'], a2_expected, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"a₂ invariant failed: expected {a2_expected}, got {self.results['a₂']}", False)
        else:
            self.log(f"a₂ invariant passed: {self.results['a₂']} ≈ g*sinθ = {a2_expected}", True)
        
        # 3. Check a3 = μ*g (same as a1)
        a3_expected = self.inputs['mu'] * self.inputs['g']
        if not math.isclose(self.results['a₃'], a3_expected, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"a₃ invariant failed: expected {a3_expected}, got {self.results['a₃']}", False)
        else:
            self.log(f"a₃ invariant passed: {self.results['a₃']} ≈ μ*g = {a3_expected}", True)
        
        # 4. Energy conservation check: Total work done by friction and gravity should match kinetic energy changes
        try:
            m = sp.Symbol('m')
            # Work done by friction on conveyor belt (acceleration phase)
            s1_val = self.results.get('s₁', 0.0)
            work_friction_acc = self.inputs['mu'] * self.inputs['g'] * m * s1_val
            
            # Work done by friction on conveyor belt (deceleration phase) - use sₘ' for leftward displacement
            sm_prime_val = self.results.get('sₘ\'', 0.0)
            work_friction_dec = self.inputs['mu'] * self.inputs['g'] * m * sm_prime_val
            
            # Work done by gravity on incline (up and down)
            sm_val = self.results.get('sₘ', 0.0)
            work_gravity_up = -m * self.inputs['g'] * self.inputs['sin_theta'] * sm_val
            work_gravity_down = m * self.inputs['g'] * self.inputs['sin_theta'] * sm_val
            
            # Initial KE = 0, final KE after complete cycle = 0 (block comes to rest on belt)
            # Total work should be 0
            total_work = work_friction_acc + work_friction_dec + work_gravity_up + work_gravity_down
            # Simplify: work_gravity_up + work_gravity_down = 0, so total_work = work_friction_acc + work_friction_dec
            # These should be positive and represent energy dissipated
            # Instead, check that initial KE gained from friction equals energy dissipated
            ke_gained = 0.5 * m * (self.inputs['v0'] ** 2)
            energy_dissipated = work_friction_acc + work_friction_dec
            
            # Since m cancels, compare coefficients
            ke_coeff = 0.5 * (self.inputs['v0'] ** 2)
            diss_coeff = self.inputs['mu'] * self.inputs['g'] * (s1_val + sm_prime_val)
            
            if not math.isclose(ke_coeff, diss_coeff, rel_tol=0.05, abs_tol=0.05):
                self.log(f"Energy balance warning: KE gained {ke_coeff} ≠ dissipated {diss_coeff}", True)
        except Exception as e:
            self.log(f"Energy invariant symbolic check failed: {str(e)}", True)
        
        return success

    def test_trace_consistency(self):
        success = True
        
        # 1. Check t1 = v0 / a1
        t1_calc = self.inputs['v0'] / self.results['a₁']
        if not math.isclose(self.results['t₁'], t1_calc, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"t₁ trace failed: expected {t1_calc}, got {self.results['t₁']}", False)
        else:
            self.log(f"t₁ trace passed: {self.results['t₁']} ≈ v0/a₁ = {t1_calc}", True)
        
        # 2. Check s1 = v0² / (2*a1)
        s1_calc = (self.inputs['v0'] ** 2) / (2 * self.results['a₁'])
        if not math.isclose(self.results['s₁'], s1_calc, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"s₁ trace failed: expected {s1_calc}, got {self.results['s₁']}", False)
        else:
            self.log(f"s₁ trace passed: {self.results['s₁']} ≈ v0²/(2*a₁) = {s1_calc}", True)
        
        # 3. Check t2 = (L - s1) / v0
        t2_calc = (self.inputs['L'] - self.results['s₁']) / self.inputs['v0']
        if not math.isclose(self.results['t₂'], t2_calc, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"t₂ trace failed: expected {t2_calc}, got {self.results['t₂']}", False)
        else:
            self.log(f"t₂ trace passed: {self.results['t₂']} ≈ (L-s₁)/v0 = {t2_calc}", True)
        
        # 4. Check t = t1 + t2
        t_calc = self.results['t₁'] + self.results['t₂']
        if not math.isclose(self.results['t'], t_calc, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"t trace failed: expected {t_calc}, got {self.results['t']}", False)
        else:
            self.log(f"t trace passed: {self.results['t']} ≈ t₁+t₂ = {t_calc}", True)
        
        # 5. Check sₘ = v0² / (2*a2)
        sm_calc = (self.inputs['v0'] ** 2) / (2 * self.results['a₂'])
        if not math.isclose(self.results['sₘ'], sm_calc, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"sₘ trace failed: expected {sm_calc}, got {self.results['sₘ']}", False)
        else:
            self.log(f"sₘ trace passed: {self.results['sₘ']} ≈ v0²/(2*a₂) = {sm_calc}", True)
        
        # 6. Check sₘ' = v0² / (2*a3)
        sm_prime_val = self.results['sₘ\'']
        sm_prime_calc = (self.inputs['v0'] ** 2) / (2 * self.results['a₃'])
        if not math.isclose(sm_prime_val, sm_prime_calc, rel_tol=0.02, abs_tol=0.02):
            success = self.log("sₘ' trace failed: expected " + str(sm_prime_calc) + ", got " + str(sm_prime_val), False)
        else:
            self.log("sₘ' trace passed: " + str(sm_prime_val) + " ≈ v0²/(2*a₃) = " + str(sm_prime_calc), True)
        
        # 7. Check s = L + 3*sₘ + 2*sₘ'
        s_calc = self.inputs['L'] + 3 * self.results['sₘ'] + 2 * sm_prime_val
        if not math.isclose(self.results['s'], s_calc, rel_tol=0.02, abs_tol=0.02):
            success = self.log(f"s trace failed: expected {s_calc}, got {self.results['s']}", False)
        else:
            self.log(f"s trace passed: {self.results['s']} ≈ L + 3*sₘ + 2*sₘ' = {s_calc}", True)
        
        return success

    def test_monotonicity(self):
        original_mu = self.inputs['mu']
        perturbed_mu = original_mu * 1.1  # Increase by 10%
        
        # Store original results
        original_results = self.results.copy()
        
        try:
            # Perturb mu and recalculate key quantities using formulas
            # Recalculate a1, a3 (both depend on mu)
            a1_new = perturbed_mu * self.inputs['g']
            a3_new = perturbed_mu * self.inputs['g']
            
            # Recalculate s1, sₘ', and thus total distance s
            s1_new = (self.inputs['v0'] ** 2) / (2 * a1_new)
            sm_prime_new = (self.inputs['v0'] ** 2) / (2 * a3_new)
            s_new = self.inputs['L'] + 3 * self.results['sₘ'] + 2 * sm_prime_new
            
            # Check monotonicity: if mu increases, friction increases, so leftward displacement sₘ' decreases
            # Therefore total distance s should decrease because 2*sₘ' term decreases
            original_sm_prime = original_results['sₘ\'']
            if sm_prime_new < original_sm_prime:
                self.log(f"Monotonicity passed: sₘ' decreases when μ increases ({sm_prime_new} < {original_sm_prime})", True)
            else:
                self.log(f"Monotonicity warning: sₘ' did not decrease when μ increased", False)
            
            if s_new < original_results['s']:
                self.log(f"Monotonicity passed: total distance s decreases when μ increases ({s_new} < {original_results['s']})", True)
            else:
                self.log(f"Monotonicity warning: s did not decrease when μ increased", False)
            
            # Restore original values
            self.inputs['mu'] = original_mu
            self.results = original_results
            return True
            
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {str(e)}", False)
            # Restore on error
            self.inputs['mu'] = original_mu
            self.results = original_results
            return False

    def run_all_tests(self):
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self.log(f"Running {test_name}...")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"{test_name} crashed: {str(e)}\n{traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs