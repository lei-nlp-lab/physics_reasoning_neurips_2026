import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely (problem constants)
        self.E = getattr(solution_instance, 'E', 32.0)      # Electromotive force (V)
        self.r = getattr(solution_instance, 'r', 1.0)       # Internal resistance (Ω)
        self.R = getattr(solution_instance, 'R', 4.0)       # Fixed resistor (Ω)
        self.R_prime = getattr(solution_instance, 'R_prime', 3.0)  # Motor coil resistance (Ω)
        self.U = getattr(solution_instance, 'U', 22.0)      # Voltmeter reading (V)
        
        # Extract results safely with defaults
        self.I = self.results.get('I', 0.0)                 # Current in normal operation (A)
        self.P_motor = self.results.get('P_Motor', 0.0)     # Motor output power (W)
        self.I_prime = self.results.get('I_prime', 0.0)     # Current when stuck (A)
        self.P_thermal = self.results.get('P_textthermal', 0.0)  # Thermal power when stuck (W)
        
        # Target answers for validation
        self.target_p_motor = 32.0  # W
        self.target_p_thermal = 48.0  # W
        
        # Setup symbolic variables for law verification
        self._setup_symbolic()
    
    def _setup_symbolic(self):
        """Initialize SymPy symbols for physical laws."""
        I, V, R, t, P, Q, E, r, R_prime, U = sp.symbols('I V R t P Q E r R_prime U', positive=True)
        self.syms = {
            'I': I, 'V': V, 'R': R, 't': t, 'P': P, 'Q': Q,
            'E': E, 'r': r, 'R_prime': R_prime, 'U': U
        }
        
        # Joule's Law expressions
        self.joule_P = I**2 * R
        self.joule_Q = I**2 * R * t
        
        # Ohm's Law expressions
        self.ohm_V = I * R
        self.ohm_P_IV = I * V
        self.ohm_P_I2R = I**2 * R
        self.ohm_P_V2R = V**2 / R
    
    def _is_close(self, a, b):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify physical laws from KB."""
        logs = []
        success = True
        
        try:
            # 1. Check Ohm's Law for normal operation circuit
            # From step_1: I = (E - U) / (R + r)
            expected_I = (self.E - self.U) / (self.R + self.r)
            if not self._is_close(self.I, expected_I):
                logs.append(f"FAIL: Ohm's Law (normal). Expected I={expected_I:.2f}A, got {self.I:.2f}A")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law (normal). I = {self.I:.2f}A matches (E-U)/(R+r)")
            
            # 2. Check motor power formula from step_2: P_motor = U*I - I²*R'
            expected_P_motor = self.U * self.I - self.I**2 * self.R_prime
            if not self._is_close(self.P_motor, expected_P_motor):
                logs.append(f"FAIL: Motor power formula. Expected {expected_P_motor:.2f}W, got {self.P_motor:.2f}W")
                success = False
            else:
                logs.append(f"PASS: Motor power formula. P_motor = U*I - I²R' holds")
            
            # 3. Check Ohm's Law for stuck motor: I' = E / (R + R' + r)
            expected_I_prime = self.E / (self.R + self.R_prime + self.r)
            if not self._is_close(self.I_prime, expected_I_prime):
                logs.append(f"FAIL: Ohm's Law (stuck). Expected I'={expected_I_prime:.2f}A, got {self.I_prime:.2f}A")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law (stuck). I' = E/(R+R'+r) holds")
            
            # 4. Check Joule's Law for thermal power: P_thermal = I'² * R'
            expected_P_thermal = self.I_prime**2 * self.R_prime
            if not self._is_close(self.P_thermal, expected_P_thermal):
                logs.append(f"FAIL: Joule's Law. Expected P_thermal={expected_P_thermal:.2f}W, got {self.P_thermal:.2f}W")
                success = False
            else:
                logs.append(f"PASS: Joule's Law. P_thermal = I'²R' holds")
            
            # 5. Verify energy conservation (power balance) for normal operation
            # Total power from source: P_source = E * I
            # Power dissipated in resistors: P_R = I²*R, P_r = I²*r, P_R' = I²*R'
            # Motor output power should satisfy: P_source = P_R + P_r + P_R' + P_motor
            P_source = self.E * self.I
            P_dissipated = (self.I**2 * self.R) + (self.I**2 * self.r) + (self.I**2 * self.R_prime) + self.P_motor
            if not self._is_close(P_source, P_dissipated):
                logs.append(f"FAIL: Power balance. Source={P_source:.2f}W ≠ Dissipated={P_dissipated:.2f}W")
                success = False
            else:
                logs.append(f"PASS: Power conservation holds (normal operation)")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check target answers match
            if not self._is_close(self.P_motor, self.target_p_motor):
                logs.append(f"FAIL: P_motor {self.P_motor:.2f}W ≠ target {self.target_p_motor:.2f}W")
                success = False
            else:
                logs.append(f"PASS: P_motor matches target 32W")
            
            if not self._is_close(self.P_thermal, self.target_p_thermal):
                logs.append(f"FAIL: P_thermal {self.P_thermal:.2f}W ≠ target {self.target_p_thermal:.2f}W")
                success = False
            else:
                logs.append(f"PASS: P_thermal matches target 48W")
            
            # Verify current values are consistent with steps
            if not self._is_close(self.I, 2.0):
                logs.append(f"WARN: I={self.I:.2f}A differs from step_1 (2A)")
            
            if not self._is_close(self.I_prime, 4.0):
                logs.append(f"WARN: I'={self.I_prime:.2f}A differs from step_3 (4A)")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input parameters and verify expected trends."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase E by 10% → both currents should increase
            E_perturbed = self.E * 1.1
            
            # Recalculate using formulas (not calling solution methods)
            I_new = (E_perturbed - self.U) / (self.R + self.r)
            I_prime_new = E_perturbed / (self.R + self.R_prime + self.r)
            
            if I_new <= self.I:
                logs.append(f"FAIL: Increasing E should increase I, but {I_new:.2f}A ≤ {self.I:.2f}A")
                success = False
            else:
                logs.append(f"PASS: I increases with E (from {self.I:.2f}A to {I_new:.2f}A)")
            
            if I_prime_new <= self.I_prime:
                logs.append(f"FAIL: Increasing E should increase I', but {I_prime_new:.2f}A ≤ {self.I_prime:.2f}A")
                success = False
            else:
                logs.append(f"PASS: I' increases with E (from {self.I_prime:.2f}A to {I_prime_new:.2f}A)")
            
            # Test 2: Increase R by 10% → both currents should decrease
            R_perturbed = self.R * 1.1
            I_new_R = (self.E - self.U) / (R_perturbed + self.r)
            I_prime_new_R = self.E / (R_perturbed + self.R_prime + self.r)
            
            if I_new_R >= self.I:
                logs.append(f"FAIL: Increasing R should decrease I, but {I_new_R:.2f}A ≥ {self.I:.2f}A")
                success = False
            else:
                logs.append(f"PASS: I decreases with R (from {self.I:.2f}A to {I_new_R:.2f}A)")
            
            if I_prime_new_R >= self.I_prime:
                logs.append(f"FAIL: Increasing R should decrease I', but {I_prime_new_R:.2f}A ≥ {self.I_prime:.2f}A")
                success = False
            else:
                logs.append(f"PASS: I' decreases with R (from {self.I_prime:.2f}A to {I_prime_new_R:.2f}A)")
            
            # Test 3: Verify Joule's Law trend: If I' increases, P_thermal increases quadratically
            P_thermal_new = I_prime_new**2 * self.R_prime
            if P_thermal_new <= self.P_thermal:
                logs.append(f"FAIL: P_thermal should increase with I', but {P_thermal_new:.2f}W ≤ {self.P_thermal:.2f}W")
                success = False
            else:
                logs.append(f"PASS: P_thermal increases quadratically with I' (Joule's Law)")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all verification tests."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRASH in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.insert(0, f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs