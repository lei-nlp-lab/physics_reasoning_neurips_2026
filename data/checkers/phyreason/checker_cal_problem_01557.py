import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: never directly access solution_instance.attr
        self.inputs = {}
        self.results = {}
        
        # Extract inputs from solution_instance (some might be in solution_instance.inputs or directly as attributes)
        # Using getattr with default, and if not found, try .get() if it's a dict-like object
        self.inputs['q'] = getattr(solution_instance, 'q', getattr(solution_instance, 'inputs', {}).get('q', 3e-4))
        self.inputs['d'] = getattr(solution_instance, 'd', getattr(solution_instance, 'inputs', {}).get('d', 0.1))
        self.inputs['E'] = getattr(solution_instance, 'E', getattr(solution_instance, 'inputs', {}).get('E', 12.0))
        self.inputs['r'] = getattr(solution_instance, 'r', getattr(solution_instance, 'inputs', {}).get('r', 2.0))
        self.inputs['R1'] = getattr(solution_instance, 'R1', getattr(solution_instance, 'inputs', {}).get('R1', 4.0))
        self.inputs['R2'] = getattr(solution_instance, 'R2', getattr(solution_instance, 'inputs', {}).get('R2', 12.0))
        self.inputs['R3'] = getattr(solution_instance, 'R3', getattr(solution_instance, 'inputs', {}).get('R3', 12.0))
        self.inputs['R4'] = getattr(solution_instance, 'R4', getattr(solution_instance, 'inputs', {}).get('R4', 12.0))
        self.inputs['g'] = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        self.inputs['theta'] = getattr(solution_instance, 'theta', getattr(solution_instance, 'inputs', {}).get('theta', math.radians(37)))
        # Precompute sin and cos for convenience
        self.inputs['sin_th'] = math.sin(self.inputs['theta'])
        self.inputs['cos_th'] = math.cos(self.inputs['theta'])
        self.inputs['tan_th'] = math.tan(self.inputs['theta'])
        
        # Extract results safely
        results_dict = getattr(solution_instance, 'results', {})
        self.results['m'] = getattr(solution_instance, 'm', results_dict.get('m', 0.0))
        self.results['R'] = getattr(solution_instance, 'R', results_dict.get('R', 0.0))
        self.results['I'] = getattr(solution_instance, 'I', results_dict.get('I', 0.0))
        self.results['E0'] = getattr(solution_instance, 'E_0', results_dict.get('E_0', results_dict.get('E0', 0.0)))
        self.results['U'] = getattr(solution_instance, 'U', results_dict.get('U', 0.0))
        
        # Initialize logs
        self.logs = []
        self.success = True

    def log(self, message, is_error=False):
        """Helper to log messages and update success flag."""
        self.logs.append(message)
        if is_error:
            self.success = False
        return self.success

    def test_invariants(self):
        """Verify fundamental physical laws (Ohm's Law, force equilibrium, etc.)."""
        try:
            # 1. Ohm's Law for closed circuit: I = E / (R_total)
            R_parallel = (self.inputs['R2'] * self.inputs['R3']) / (self.inputs['R2'] + self.inputs['R3'])
            R_external = R_parallel + self.inputs['R1']
            R_total = R_external + self.inputs['r']
            I_expected = self.inputs['E'] / R_total
            if not math.isclose(self.results['I'], I_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Ohm's Law mismatch. Expected I={I_expected:.3f}, got {self.results['I']:.3f}", True)
            else:
                self.log(f"PASS: Ohm's Law check: I = E/(R+r) = {self.results['I']:.3f} A")
            
            # 2. Terminal voltage: U = E - I*r
            U_expected = self.inputs['E'] - self.results['I'] * self.inputs['r']
            if not math.isclose(self.results['U'], U_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Terminal voltage mismatch. Expected U={U_expected:.3f}, got {self.results['U']:.3f}", True)
            else:
                self.log(f"PASS: Terminal voltage check: U = E - I*r = {self.results['U']:.3f} V")
            
            # 3. Uniform electric field: E0 = U/d
            E0_expected = self.results['U'] / self.inputs['d']
            if not math.isclose(self.results['E0'], E0_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Electric field mismatch. Expected E0={E0_expected:.3f}, got {self.results['E0']:.3f}", True)
            else:
                self.log(f"PASS: Electric field check: E0 = U/d = {self.results['E0']:.3f} V/m")
            
            # 4. Force equilibrium: m = (E0 * q) / (g * tanθ)
            m_expected = (self.results['E0'] * self.inputs['q']) / (self.inputs['g'] * self.inputs['tan_th'])
            if not math.isclose(self.results['m'], m_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Mass equilibrium mismatch. Expected m={m_expected:.6f}, got {self.results['m']:.6f}", True)
            else:
                self.log(f"PASS: Force equilibrium check: m = (E0*q)/(g*tanθ) = {self.results['m']:.6f} kg")
            
            # 5. Check that R_external matches reported R (from step 1)
            if not math.isclose(self.results['R'], R_external, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"WARN: External resistance mismatch. Computed R={R_external:.3f}, got {self.results['R']:.3f}")
            else:
                self.log(f"PASS: External resistance R = {self.results['R']:.3f} Ω")
                
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}", True)
        return self.success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer (step-by-step)."""
        try:
            # Step 1: R = (R2*R3)/(R2+R3) + R1
            R_calc = (self.inputs['R2'] * self.inputs['R3']) / (self.inputs['R2'] + self.inputs['R3']) + self.inputs['R1']
            if not math.isclose(self.results['R'], R_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step 1 (R) inconsistency. Computed {R_calc:.3f}, got {self.results['R']:.3f}", True)
            else:
                self.log(f"PASS: Step 1 consistency: R = {self.results['R']:.3f} Ω")
            
            # Step 2: I = E / (R + r)
            I_calc = self.inputs['E'] / (self.results['R'] + self.inputs['r'])
            if not math.isclose(self.results['I'], I_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step 2 (I) inconsistency. Computed {I_calc:.3f}, got {self.results['I']:.3f}", True)
            else:
                self.log(f"PASS: Step 2 consistency: I = {self.results['I']:.3f} A")
            
            # Step 3: U = E - I*r
            U_calc = self.inputs['E'] - self.results['I'] * self.inputs['r']
            if not math.isclose(self.results['U'], U_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step 3 (U) inconsistency. Computed {U_calc:.3f}, got {self.results['U']:.3f}", True)
            else:
                self.log(f"PASS: Step 3 consistency: U = {self.results['U']:.3f} V")
            
            # Step 4: E0 = U/d
            E0_calc = self.results['U'] / self.inputs['d']
            if not math.isclose(self.results['E0'], E0_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step 4 (E0) inconsistency. Computed {E0_calc:.3f}, got {self.results['E0']:.3f}", True)
            else:
                self.log(f"PASS: Step 4 consistency: E0 = {self.results['E0']:.3f} V/m")
            
            # Step 5: m = (E0 * q) / (g * tanθ)
            m_calc = (self.results['E0'] * self.inputs['q']) / (self.inputs['g'] * self.inputs['tan_th'])
            if not math.isclose(self.results['m'], m_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step 5 (m) inconsistency. Computed {m_calc:.6f}, got {self.results['m']:.6f}", True)
            else:
                self.log(f"PASS: Step 5 consistency: m = {self.results['m']:.6f} kg")
                
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}", True)
        return self.success

    def test_monotonicity(self):
        """Perturb an input and verify the expected trend, then restore."""
        original_E = self.inputs['E']
        perturbed_E = original_E * 1.05  # Increase by 5%
        
        try:
            # Compute expected change: I should increase with increased E
            R_total = self.results['R'] + self.inputs['r']
            I_original = self.results['I']
            I_perturbed = perturbed_E / R_total
            
            if not I_perturbed > I_original:
                self.log(f"FAIL: Monotonicity violation. Increasing E should increase I, but got I_original={I_original:.3f}, I_perturbed={I_perturbed:.3f}", True)
            else:
                self.log(f"PASS: Monotonicity check: I increases with E (I goes {I_original:.3f} -> {I_perturbed:.3f} when E increases by 5%)")
            
            # Perturb angle and check m trend
            original_theta = self.inputs['theta']
            perturbed_theta = original_theta * 1.05  # Increase by 5% (in radians)
            perturbed_tan = math.tan(perturbed_theta)
            m_original = self.results['m']
            # Using the same E0 and q, g to compute perturbed m
            m_perturbed = (self.results['E0'] * self.inputs['q']) / (self.inputs['g'] * perturbed_tan)
            
            # Since tan increases with angle (for acute angles), m should decrease
            if not m_perturbed < m_original:
                self.log(f"FAIL: Monotonicity violation. Increasing θ should decrease m, but got m_original={m_original:.6f}, m_perturbed={m_perturbed:.6f}", True)
            else:
                self.log(f"PASS: Monotonicity check: m decreases with θ (m goes {m_original:.6f} -> {m_perturbed:.6f} when θ increases by 5%)")
            
            # Restore is implicit since we only used local variables
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}", True)
        return self.success

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.success = True
        self.logs = []
        
        self.log("=== Starting Physics Audit ===")
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        self.log("=== Audit Complete ===")
        
        return self.success, self.logs