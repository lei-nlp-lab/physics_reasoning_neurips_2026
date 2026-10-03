import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.inputs = {}
        self.results = {}
        
        # Input parameters (with fallbacks)
        self.inputs['m'] = getattr(solution_instance, 'm', 
                                 getattr(solution_instance, 'results', {}).get('m', None))
        self.inputs['g'] = getattr(solution_instance, 'g', 
                                 getattr(solution_instance, 'results', {}).get('g', None))
        self.inputs['h'] = getattr(solution_instance, 'h', 
                                 getattr(solution_instance, 'results', {}).get('h', None))
        self.inputs['theta'] = getattr(solution_instance, 'theta', 
                                     getattr(solution_instance, 'results', {}).get('theta', None))
        self.inputs['t0'] = getattr(solution_instance, 't0', 
                                  getattr(solution_instance, 'results', {}).get('t0', None))
        
        # Result quantities
        self.results['v'] = getattr(solution_instance, 'v', 
                                  getattr(solution_instance, 'results', {}).get('v', None))
        self.results['P_G'] = getattr(solution_instance, 'P_G', 
                                    getattr(solution_instance, 'results', {}).get('P_G', None))
        
        # Symbolic variables for verification
        self.m_sym, self.g_sym, self.h_sym, self.t0_sym = sp.symbols('m g h t0', positive=True)
        self.v_sym = sp.symbols('v', positive=True)
        self.P_sym = sp.symbols('P', positive=True)
        
        self.logs = []
        
    def log(self, message, is_error=False):
        """Helper to log messages."""
        entry = f"[{'ERROR' if is_error else 'INFO'}] {message}"
        self.logs.append(entry)
        return is_error
    
    def test_invariants(self):
        """Check fundamental physical laws."""
        errors = 0
        
        # 1. Energy conservation (work-energy theorem)
        try:
            if None in (self.inputs['m'], self.inputs['g'], self.inputs['h'], self.results['v']):
                self.log("Missing data for energy conservation check", is_error=True)
                errors += 1
            else:
                m_val = float(self.inputs['m'])
                g_val = float(self.inputs['g'])
                h_val = float(self.inputs['h'])
                v_val = float(self.results['v'])
                
                # Work done by gravity
                W_gravity = m_val * g_val * h_val
                # Change in kinetic energy
                delta_K = 0.5 * m_val * v_val**2
                
                if not math.isclose(W_gravity, delta_K, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Energy conservation violated: W_gravity={W_gravity:.3f}, "
                           f"ΔK={delta_K:.3f}, ratio={W_gravity/delta_K:.3f}", is_error=True)
                    errors += 1
                else:
                    self.log(f"Energy conservation satisfied: W_gravity={W_gravity:.3f} ≈ ΔK={delta_K:.3f}")
        except Exception as e:
            self.log(f"Energy conservation check failed: {e}", is_error=True)
            errors += 1
        
        # 2. Symbolic verification of work-energy theorem
        try:
            # Symbolic equation: m*g*h = 1/2*m*v^2 -> v = sqrt(2*g*h)
            eq_energy = sp.Eq(self.m_sym * self.g_sym * self.h_sym, 
                              0.5 * self.m_sym * self.v_sym**2)
            v_expected = sp.sqrt(2 * self.g_sym * self.h_sym)
            
            # Substitute numerical values if available
            if None not in (self.inputs['g'], self.inputs['h']):
                g_val = float(self.inputs['g'])
                h_val = float(self.inputs['h'])
                v_expected_num = math.sqrt(2 * g_val * h_val)
                if self.results['v'] is not None:
                    v_val = float(self.results['v'])
                    if not math.isclose(v_val, v_expected_num, rel_tol=1e-2, abs_tol=1e-2):
                        self.log(f"Symbolic work-energy mismatch: v={v_val:.3f}, "
                               f"expected sqrt(2gh)={v_expected_num:.3f}", is_error=True)
                        errors += 1
                    else:
                        self.log(f"Symbolic work-energy verified: v ≈ sqrt(2gh)")
        except Exception as e:
            self.log(f"Symbolic energy check failed: {e}", is_error=True)
            errors += 1
        
        return errors == 0
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        errors = 0
        
        # Check velocity formula
        try:
            if None not in (self.inputs['g'], self.inputs['h'], self.results['v']):
                g_val = float(self.inputs['g'])
                h_val = float(self.inputs['h'])
                v_val = float(self.results['v'])
                v_expected = math.sqrt(2 * g_val * h_val)
                
                if not math.isclose(v_val, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Velocity inconsistency: v={v_val:.3f}, "
                           f"sqrt(2gh)={v_expected:.3f}", is_error=True)
                    errors += 1
                else:
                    self.log(f"Velocity consistent with work-energy theorem")
        except Exception as e:
            self.log(f"Velocity consistency check failed: {e}", is_error=True)
            errors += 1
        
        # Check average power formula
        try:
            if None not in (self.inputs['m'], self.inputs['g'], self.inputs['h'], 
                          self.inputs['t0'], self.results['P_G']):
                m_val = float(self.inputs['m'])
                g_val = float(self.inputs['g'])
                h_val = float(self.inputs['h'])
                t0_val = float(self.inputs['t0'])
                P_val = float(self.results['P_G'])
                P_expected = m_val * g_val * h_val / t0_val
                
                if not math.isclose(P_val, P_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Power inconsistency: P_G={P_val:.3f}, "
                           f"mgh/t0={P_expected:.3f}", is_error=True)
                    errors += 1
                else:
                    self.log(f"Average power consistent with definition P = W/t")
        except Exception as e:
            self.log(f"Power consistency check failed: {e}", is_error=True)
            errors += 1
        
        return errors == 0
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        errors = 0
        
        # Store original values
        orig_h = self.inputs['h']
        orig_t0 = self.inputs['t0']
        
        # 1. Perturb h: velocity should increase with h (v ∝ √h)
        try:
            if None not in (self.inputs['g'], orig_h):
                g_val = float(self.inputs['g'])
                h_val = float(orig_h)
                v_orig = math.sqrt(2 * g_val * h_val)
                
                # Increase h by 10%
                h_perturbed = h_val * 1.1
                v_perturbed = math.sqrt(2 * g_val * h_perturbed)
                
                if not v_perturbed > v_orig:
                    self.log(f"Monotonicity violation: v(√h) should increase with h, "
                           f"but v({h_val:.3f})={v_orig:.3f}, v({h_perturbed:.3f})={v_perturbed:.3f}", 
                           is_error=True)
                    errors += 1
                else:
                    self.log(f"Monotonicity: v increases with h (√h dependence)")
        except Exception as e:
            self.log(f"Monotonicity check for h failed: {e}", is_error=True)
            errors += 1
        
        # 2. Perturb t0: average power should decrease with t0 (P ∝ 1/t0)
        try:
            if None not in (self.inputs['m'], self.inputs['g'], self.inputs['h'], orig_t0):
                m_val = float(self.inputs['m'])
                g_val = float(self.inputs['g'])
                h_val = float(self.inputs['h'])
                t0_val = float(orig_t0)
                P_orig = m_val * g_val * h_val / t0_val
                
                # Increase t0 by 10%
                t0_perturbed = t0_val * 1.1
                P_perturbed = m_val * g_val * h_val / t0_perturbed
                
                if not P_perturbed < P_orig:
                    self.log(f"Monotonicity violation: P ∝ 1/t0 should decrease with t0, "
                           f"but P({t0_val:.3f})={P_orig:.3f}, P({t0_perturbed:.3f})={P_perturbed:.3f}", 
                           is_error=True)
                    errors += 1
                else:
                    self.log(f"Monotonicity: P decreases with t0 (inverse dependence)")
        except Exception as e:
            self.log(f"Monotonicity check for t0 failed: {e}", is_error=True)
            errors += 1
        
        # Restore original values (not strictly needed but good practice)
        self.inputs['h'] = orig_h
        self.inputs['t0'] = orig_t0
        
        return errors == 0
    
    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} failed", is_error=True)
            except Exception as e:
                success = False
                self.log(f"{test_name} crashed: {e}\n{traceback.format_exc()}", is_error=True)
        
        self.log(f"All tests completed. Overall: {'PASS' if success else 'FAIL'}")
        return success, self.logs