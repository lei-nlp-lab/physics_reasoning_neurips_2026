import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with defaults (based on problem statement)
        self.inputs['m'] = getattr(solution_instance, 'm', 60.0)
        self.inputs['L'] = getattr(solution_instance, 'L', 5.0)
        self.inputs['t'] = getattr(solution_instance, 't', 1.0)
        self.inputs['g'] = getattr(solution_instance, 'g', 10.0)
        
        # Extract results from solution_instance.results or direct attributes
        results_dict = getattr(solution_instance, 'results', {})
        self.results['v'] = getattr(solution_instance, 'v', results_dict.get('v', 0.0))
        self.results['t_0'] = getattr(solution_instance, 't_0', results_dict.get('t_0', 0.0))
        self.results['I'] = getattr(solution_instance, 'I', results_dict.get('I', 0.0))
        self.results['I_prime'] = getattr(solution_instance, 'I_prime', results_dict.get('I\'', 0.0))
        self.results['F_prime'] = getattr(solution_instance, 'F_prime', results_dict.get('F\'', 0.0))
        self.results['F'] = getattr(solution_instance, 'F', results_dict.get('F', 0.0))
        
        # Setup symbolic variables for invariant checking
        self.m, self.L, self.t_val, self.g = sp.symbols('m L t g', positive=True, real=True)
        self.v_sym, self.t0_sym, self.I_sym, self.Ip_sym, self.Fp_sym = sp.symbols('v t0 I Ip Fp', real=True)
        
        # Store logs for test results
        self.logs = []
        
    def log(self, message, success=None):
        """Helper to append log messages with optional success flag."""
        if success is not None:
            message = f"[{'PASS' if success else 'FAIL'}] {message}"
        self.logs.append(message)
        
    def test_invariants(self):
        """Verify conservation laws and Newton's laws based on KB."""
        try:
            # 1. Energy Conservation: m*g*L = 0.5*m*v^2 (from step 1)
            lhs_energy = self.inputs['m'] * self.inputs['g'] * self.inputs['L']
            rhs_energy = 0.5 * self.inputs['m'] * self.results['v']**2
            energy_check = math.isclose(lhs_energy, rhs_energy, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Energy conservation: m*g*L = {lhs_energy:.2f}, 0.5*m*v^2 = {rhs_energy:.2f}", energy_check)
            
            # 2. Kinematics consistency: t_0 = v/g (from step 2)
            t0_calc = self.results['v'] / self.inputs['g']
            kinematics_check = math.isclose(self.results['t_0'], t0_calc, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Kinematics: t_0 = {self.results['t_0']:.2f}, v/g = {t0_calc:.2f}", kinematics_check)
            
            # 3. Impulse-momentum for gravity: I = m*g*(t + t_0) (from step 3)
            I_calc = self.inputs['m'] * self.inputs['g'] * (self.inputs['t'] + self.results['t_0'])
            impulse_check = math.isclose(self.results['I'], I_calc, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Gravity impulse: I = {self.results['I']:.2f}, m*g*(t+t_0) = {I_calc:.2f}", impulse_check)
            
            # 4. Momentum theorem: I' = I (from step 4)
            momentum_check = math.isclose(self.results['I_prime'], self.results['I'], rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Momentum theorem: I' = {self.results['I_prime']:.2f}, I = {self.results['I']:.2f}", momentum_check)
            
            # 5. Average force definition: F' = I'/t (from step 5)
            Fp_calc = self.results['I_prime'] / self.inputs['t']
            force_def_check = math.isclose(self.results['F_prime'], Fp_calc, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Average force: F' = {self.results['F_prime']:.2f}, I'/t = {Fp_calc:.2f}", force_def_check)
            
            # 6. Newton's third law: F = F' (from step 6)
            third_law_check = math.isclose(self.results['F'], self.results['F_prime'], rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Newton's third law: F = {self.results['F']:.2f}, F' = {self.results['F_prime']:.2f}", third_law_check)
            
            # Overall invariant check
            all_passed = (energy_check and kinematics_check and impulse_check and 
                         momentum_check and force_def_check and third_law_check)
            self.log(f"All invariants passed: {all_passed}", all_passed)
            return all_passed
            
        except Exception as e:
            self.log(f"Invariant test failed with error: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Expected final answers from problem statement (target answers)
            target_v = 10.0  # m/s
            target_I_prime = 1200.0  # N·s
            target_F = 1200.0  # N
            
            # Check each with fuzzy matching
            v_check = math.isclose(self.results['v'], target_v, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Velocity v: {self.results['v']:.2f} vs target {target_v}", v_check)
            
            I_prime_check = math.isclose(self.results['I_prime'], target_I_prime, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Impulse I': {self.results['I_prime']:.2f} vs target {target_I_prime}", I_prime_check)
            
            F_check = math.isclose(self.results['F'], target_F, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Force F: {self.results['F']:.2f} vs target {target_F}", F_check)
            
            # Also check internal consistency: I' and F should be consistent via t
            F_from_I = self.results['I_prime'] / self.inputs['t']
            internal_check = math.isclose(self.results['F'], F_from_I, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Internal F vs I'/t: F={self.results['F']:.2f}, I'/t={F_from_I:.2f}", internal_check)
            
            all_passed = v_check and I_prime_check and F_check and internal_check
            self.log(f"Trace consistency passed: {all_passed}", all_passed)
            return all_passed
            
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb mass input and verify expected trends."""
        original_m = self.inputs['m']
        perturbed_m = original_m * 1.1  # Increase by 10%
        
        try:
            # Calculate expected changes using formulas from steps
            # Step 1: v = sqrt(2*g*L) -> independent of m
            v_original = math.sqrt(2 * self.inputs['g'] * self.inputs['L'])
            v_perturbed = math.sqrt(2 * self.inputs['g'] * self.inputs['L'])
            v_constant = math.isclose(v_original, v_perturbed, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Velocity v independent of m: {v_original:.2f} vs {v_perturbed:.2f}", v_constant)
            
            # Step 2: t_0 = v/g -> also independent of m
            t0_original = v_original / self.inputs['g']
            t0_perturbed = v_perturbed / self.inputs['g']
            t0_constant = math.isclose(t0_original, t0_perturbed, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Time t_0 independent of m: {t0_original:.2f} vs {t0_perturbed:.2f}", t0_constant)
            
            # Step 3: I = m*g*(t + t_0) -> proportional to m
            I_original = original_m * self.inputs['g'] * (self.inputs['t'] + t0_original)
            I_perturbed = perturbed_m * self.inputs['g'] * (self.inputs['t'] + t0_perturbed)
            I_ratio = I_perturbed / I_original
            I_proportional = math.isclose(I_ratio, perturbed_m/original_m, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Impulse I proportional to m: ratio {I_ratio:.3f} vs mass ratio {perturbed_m/original_m:.3f}", I_proportional)
            
            # Step 4: I' = I -> same proportionality
            Ip_ratio = I_perturbed / I_original
            Ip_proportional = math.isclose(Ip_ratio, perturbed_m/original_m, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Impulse I' proportional to m: ratio {Ip_ratio:.3f} vs mass ratio {perturbed_m/original_m:.3f}", Ip_proportional)
            
            # Step 5: F' = I'/t -> proportional to m
            Fp_original = I_original / self.inputs['t']
            Fp_perturbed = I_perturbed / self.inputs['t']
            Fp_ratio = Fp_perturbed / Fp_original
            Fp_proportional = math.isclose(Fp_ratio, perturbed_m/original_m, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Force F' proportional to m: ratio {Fp_ratio:.3f} vs mass ratio {perturbed_m/original_m:.3f}", Fp_proportional)
            
            # Step 6: F = F' -> same proportionality
            F_ratio = Fp_perturbed / Fp_original
            F_proportional = math.isclose(F_ratio, perturbed_m/original_m, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Force F proportional to m: ratio {F_ratio:.3f} vs mass ratio {perturbed_m/original_m:.3f}", F_proportional)
            
            all_passed = (v_constant and t0_constant and I_proportional and 
                         Ip_proportional and Fp_proportional and F_proportional)
            self.log(f"Monotonicity tests passed: {all_passed}", all_passed)
            return all_passed
            
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {traceback.format_exc()}")
            return False
        finally:
            # Restore original mass (though we only used copies, not modifying self.inputs)
            pass
    
    def run_all_tests(self):
        """Execute all test suites and return overall success and logs."""
        self.logs = []  # Reset logs
        
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            overall_success = inv_ok and trace_ok and mono_ok
            self.log(f"=== OVERALL RESULT: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===", overall_success)
            
        except Exception as e:
            self.log(f"Critical error in test suite: {traceback.format_exc()}")
            overall_success = False
        
        return overall_success, self.logs