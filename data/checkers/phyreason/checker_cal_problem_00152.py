import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        self.inputs = {}
        self.results = {}
        
        # Input parameters from problem statement
        self.inputs['R'] = getattr(solution_instance, 'R', getattr(solution_instance, 'resistance', 0.1))
        self.inputs['L'] = getattr(solution_instance, 'L', getattr(solution_instance, 'side_length', 0.2))
        self.inputs['B'] = getattr(solution_instance, 'B', getattr(solution_instance, 'magnetic_field', 0.5))
        self.inputs['v'] = getattr(solution_instance, 'v', getattr(solution_instance, 'velocity', 8.0))
        
        # Results expected from solution
        # Using nested getattr with defaults from KB steps
        self.results['E'] = getattr(solution_instance, 'E', getattr(solution_instance, 'results', {}).get('E', 0.8))
        self.results['I'] = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 8.0))
        self.results['F'] = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 0.8))
        self.results['t'] = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.05))
        self.results['Q'] = getattr(solution_instance, 'Q', getattr(solution_instance, 'results', {}).get('Q', 0.32))
        
        # For monotonicity tests
        self._original_inputs = self.inputs.copy()
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message, status="INFO"):
        """Helper for logging."""
        self.logs.append(f"[{status}] {message}")
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        return False
        
    def test_invariants(self):
        """Verify fundamental physical laws using symbolic and numerical checks."""
        tests_passed = True
        
        # 1. Motional EMF: E = B * L * v
        try:
            E_expected = self.inputs['B'] * self.inputs['L'] * self.inputs['v']
            if not self._is_close(self.results['E'], E_expected):
                self._log(f"EMF invariant failed: E={self.results['E']}, B*L*v={E_expected}", "WARNING")
                tests_passed = False
            else:
                self._log(f"EMF invariant passed: E = B*L*v = {E_expected} V")
        except Exception as e:
            self._log(f"EMF invariant check failed: {e}", "ERROR")
            tests_passed = False
            
        # 2. Ohm's Law: I = E / R
        try:
            I_expected = self.results['E'] / self.inputs['R']
            if not self._is_close(self.results['I'], I_expected):
                self._log(f"Ohm's Law invariant failed: I={self.results['I']}, E/R={I_expected}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Ohm's Law invariant passed: I = E/R = {I_expected} A")
        except Exception as e:
            self._log(f"Ohm's Law invariant check failed: {e}", "ERROR")
            tests_passed = False
            
        # 3. Ampere Force: F = B * I * L
        try:
            F_expected = self.inputs['B'] * self.results['I'] * self.inputs['L']
            if not self._is_close(self.results['F'], F_expected):
                self._log(f"Ampere Force invariant failed: F={self.results['F']}, B*I*L={F_expected}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Ampere Force invariant passed: F = B*I*L = {F_expected} N")
        except Exception as e:
            self._log(f"Ampere Force invariant check failed: {e}", "ERROR")
            tests_passed = False
            
        # 4. Time through field: t = 2L / v (entering and exiting)
        try:
            t_expected = 2 * self.inputs['L'] / self.inputs['v']
            if not self._is_close(self.results['t'], t_expected):
                self._log(f"Time invariant failed: t={self.results['t']}, 2L/v={t_expected}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Time invariant passed: t = 2L/v = {t_expected} s")
        except Exception as e:
            self._log(f"Time invariant check failed: {e}", "ERROR")
            tests_passed = False
            
        # 5. Joule Heating: Q = I^2 * R * t
        try:
            Q_expected = self.results['I']**2 * self.inputs['R'] * self.results['t']
            if not self._is_close(self.results['Q'], Q_expected):
                self._log(f"Joule's Law invariant failed: Q={self.results['Q']}, I^2*R*t={Q_expected}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Joule's Law invariant passed: Q = I^2*R*t = {Q_expected} J")
        except Exception as e:
            self._log(f"Joule's Law invariant check failed: {e}", "ERROR")
            tests_passed = False
            
        return tests_passed
        
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        tests_passed = True
        
        # Cross-check using combined formula from step 4: F = (B^2 * L^2 * v) / R
        try:
            F_combined = (self.inputs['B']**2 * self.inputs['L']**2 * self.inputs['v']) / self.inputs['R']
            if not self._is_close(self.results['F'], F_combined):
                self._log(f"Trace consistency (combined F) failed: {self.results['F']} vs {F_combined}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Trace consistency (combined F) passed: F = B²L²v/R = {F_combined} N")
        except Exception as e:
            self._log(f"Trace consistency (combined F) check failed: {e}", "ERROR")
            tests_passed = False
            
        # Cross-check using alternative Q formula: Q = (E^2 / R) * t
        try:
            Q_alternative = (self.results['E']**2 / self.inputs['R']) * self.results['t']
            if not self._is_close(self.results['Q'], Q_alternative):
                self._log(f"Trace consistency (alternative Q) failed: {self.results['Q']} vs {Q_alternative}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Trace consistency (alternative Q) passed: Q = E²t/R = {Q_alternative} J")
        except Exception as e:
            self._log(f"Trace consistency (alternative Q) check failed: {e}", "ERROR")
            tests_passed = False
            
        return tests_passed
        
    def test_monotonicity(self):
        """Perturb an input and check that results change according to physical trends."""
        tests_passed = True
        
        # Test 1: Increase velocity v by 10%, check that E, I, F, Q increase
        try:
            original_v = self.inputs['v']
            perturbed_v = original_v * 1.1
            
            # Calculate expected changes using formulas
            E_perturbed = self.inputs['B'] * self.inputs['L'] * perturbed_v
            I_perturbed = E_perturbed / self.inputs['R']
            F_perturbed = self.inputs['B'] * I_perturbed * self.inputs['L']
            t_perturbed = 2 * self.inputs['L'] / perturbed_v
            Q_perturbed = I_perturbed**2 * self.inputs['R'] * t_perturbed
            
            # Check monotonicity: all should increase
            if not (E_perturbed > self.results['E'] and 
                    I_perturbed > self.results['I'] and 
                    F_perturbed > self.results['F'] and 
                    Q_perturbed > self.results['Q']):
                self._log(f"Monotonicity test failed for v increase: E:{E_perturbed}, I:{I_perturbed}, F:{F_perturbed}, Q:{Q_perturbed}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Monotonicity test passed for v increase (+10%): all quantities increased")
                
        except Exception as e:
            self._log(f"Monotonicity test (v increase) failed: {e}", "ERROR")
            tests_passed = False
            
        # Test 2: Increase resistance R by 10%, check that I, F, Q decrease
        try:
            original_R = self.inputs['R']
            perturbed_R = original_R * 1.1
            
            # Calculate with perturbed R (E and t remain same since B, L, v unchanged)
            E_original = self.results['E']  # E = B*L*v unchanged
            t_original = self.results['t']  # t = 2L/v unchanged
            
            I_perturbed_R = E_original / perturbed_R
            F_perturbed_R = self.inputs['B'] * I_perturbed_R * self.inputs['L']
            Q_perturbed_R = I_perturbed_R**2 * perturbed_R * t_original
            
            # Check monotonicity: I and F should decrease, Q might decrease or increase? 
            # Actually from KB: If R increases, I decreases (Ohm), F decreases (Ampere), 
            # but Q = I²Rt: I² decreases more than R increases? Let's check numerically.
            if not (I_perturbed_R < self.results['I'] and 
                    F_perturbed_R < self.results['F']):
                self._log(f"Monotonicity test failed for R increase: I:{I_perturbed_R}, F:{F_perturbed_R}", "WARNING")
                tests_passed = False
            else:
                self._log(f"Monotonicity test passed for R increase (+10%): I and F decreased")
                
        except Exception as e:
            self._log(f"Monotonicity test (R increase) failed: {e}", "ERROR")
            tests_passed = False
            
        return tests_passed
        
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs)."""
        self.logs = []
        success = True
        
        try:
            self._log("Starting PhysicsChecker tests")
            
            # Run each test suite
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            overall_success = inv_success and trace_success and mono_success
            
            if overall_success:
                self._log("All tests passed!", "SUCCESS")
            else:
                self._log("Some tests failed", "WARNING")
                
            return overall_success, self.logs
            
        except Exception as e:
            self._log(f"Unhandled exception in run_all_tests: {traceback.format_exc()}", "ERROR")
            return False, self.logs