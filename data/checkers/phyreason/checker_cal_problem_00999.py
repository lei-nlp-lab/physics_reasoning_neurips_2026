import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using getattr/results fallback."""
        self.solution = solution_instance
        
        # Input parameters (with safe defaults)
        self.V_container = 3.0  # L
        self.V_liquid_initial = 2.0  # L
        self.n_pumps = 15
        self.V_pump = 0.1  # L
        self.P_atm = 1e5  # Pa
        self.P_pump = 1e5  # Pa
        
        # Results extraction with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract using safe pattern: solution_instance.attr -> results.get -> default
        self.V = getattr(solution_instance, 'V', self.results.get('V', None))
        self.V_1 = getattr(solution_instance, 'V_1', self.results.get('V_1', None))
        self.p_1 = getattr(solution_instance, 'p_1', self.results.get('p_1', None))
        self.p_3 = getattr(solution_instance, 'p_3', self.results.get('p_3', None))
        self.V_2 = getattr(solution_instance, 'V_2', self.results.get('V_2', None))
        self.V_3 = getattr(solution_instance, 'V_3', self.results.get('V_3', None))
        self.p_2 = getattr(solution_instance, 'p_2', self.results.get('p_2', None))
        
        # Derived quantities
        self.V_gas_initial = self.V_container - self.V_liquid_initial  # Should be 1.0 L
        
        self.logs = []
        
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def test_invariants(self):
        """Check Boyle's Law and isothermal invariants."""
        try:
            # Check Boyle's Law for compression: p1*V1 = p2*V2
            if None not in (self.p_1, self.V_1, self.p_2, self.V_2):
                left = self.p_1 * self.V_1
                right = self.p_2 * self.V_2
                if math.isclose(left, right, rel_tol=0.02, abs_tol=100.0):
                    self._log(f"Boyle's Law (compression): p1*V1 = {left:.2f} ≈ p2*V2 = {right:.2f}", True)
                else:
                    self._log(f"Boyle's Law (compression): p1*V1 = {left:.2f} ≠ p2*V2 = {right:.2f}", False)
                    return False
            
            # Check Boyle's Law for expansion: p2*V2 = p3*V3
            if None not in (self.p_2, self.V_2, self.p_3, self.V_3):
                left = self.p_2 * self.V_2
                right = self.p_3 * self.V_3
                if math.isclose(left, right, rel_tol=0.02, abs_tol=100.0):
                    self._log(f"Boyle's Law (expansion): p2*V2 = {left:.2f} ≈ p3*V3 = {right:.2f}", True)
                else:
                    self._log(f"Boyle's Law (expansion): p2*V2 = {left:.2f} ≠ p3*V3 = {right:.2f}", False)
                    return False
            
            # Check initial conditions
            if self.p_1 is not None:
                if math.isclose(self.p_1, self.P_atm, rel_tol=0.02, abs_tol=100.0):
                    self._log(f"Initial pressure p1 = {self.p_1:.0f} Pa ≈ atmospheric {self.P_atm:.0f} Pa", True)
                else:
                    self._log(f"Initial pressure mismatch: p1 = {self.p_1:.0f} ≠ atmospheric {self.P_atm:.0f}", False)
                    return False
            
            if self.p_3 is not None:
                if math.isclose(self.p_3, self.P_atm, rel_tol=0.02, abs_tol=100.0):
                    self._log(f"Final pressure p3 = {self.p_3:.0f} Pa ≈ atmospheric {self.P_atm:.0f} Pa", True)
                else:
                    self._log(f"Final pressure mismatch: p3 = {self.p_3:.0f} ≠ atmospheric {self.P_atm:.0f}", False)
                    return False
            
            # Check that temperature is constant (implied by Boyle's Law checks)
            if None not in (self.p_1, self.V_1, self.p_2, self.V_2, self.p_3, self.V_3):
                # pV should be constant throughout
                pV1 = self.p_1 * self.V_1
                pV2 = self.p_2 * self.V_2
                pV3 = self.p_3 * self.V_3
                if (math.isclose(pV1, pV2, rel_tol=0.02, abs_tol=100.0) and 
                    math.isclose(pV2, pV3, rel_tol=0.02, abs_tol=100.0)):
                    self._log(f"Isothermal condition: pV ≈ constant = {pV1:.2f} Pa·L", True)
                else:
                    self._log(f"pV not constant: {pV1:.2f}, {pV2:.2f}, {pV3:.2f}", False)
                    return False
            
            return True
        except Exception as e:
            self._log(f"test_invariants failed: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Check intermediate calculations match final answers."""
        try:
            # Check V_1 calculation
            expected_V1 = self.V_gas_initial + self.n_pumps * self.V_pump  # 1.0 + 15*0.1 = 2.5 L
            if self.V_1 is not None:
                if math.isclose(self.V_1, expected_V1, rel_tol=0.02, abs_tol=0.05):
                    self._log(f"V1 = {self.V_1:.2f} L ≈ expected {expected_V1:.2f} L (1.0 + 15*0.1)", True)
                else:
                    self._log(f"V1 mismatch: {self.V_1:.2f} ≠ {expected_V1:.2f}", False)
                    return False
            
            # Check V_2 is the gas volume after pumping
            if self.V_2 is not None:
                if math.isclose(self.V_2, self.V_gas_initial, rel_tol=0.02, abs_tol=0.05):
                    self._log(f"V2 = {self.V_2:.2f} L ≈ initial gas volume {self.V_gas_initial:.2f} L", True)
                else:
                    self._log(f"V2 mismatch: {self.V_2:.2f} ≠ {self.V_gas_initial:.2f}", False)
                    return False
            
            # Check p_2 calculation from Boyle's Law
            if None not in (self.p_1, self.V_1, self.V_2, self.p_2):
                expected_p2 = self.p_1 * self.V_1 / self.V_2
                if math.isclose(self.p_2, expected_p2, rel_tol=0.02, abs_tol=1000.0):
                    self._log(f"p2 = {self.p_2:.0f} Pa ≈ p1*V1/V2 = {expected_p2:.0f} Pa", True)
                else:
                    self._log(f"p2 mismatch: {self.p_2:.0f} ≠ {expected_p2:.0f}", False)
                    return False
            
            # Check V_3 calculation
            if None not in (self.p_2, self.V_2, self.p_3, self.V_3):
                expected_V3 = self.p_2 * self.V_2 / self.p_3
                if math.isclose(self.V_3, expected_V3, rel_tol=0.02, abs_tol=0.05):
                    self._log(f"V3 = {self.V_3:.2f} L ≈ p2*V2/p3 = {expected_V3:.2f} L", True)
                else:
                    self._log(f"V3 mismatch: {self.V_3:.2f} ≠ {expected_V3:.2f}", False)
                    return False
            
            # Check final answer V (remaining disinfectant)
            if self.V is not None and self.V_3 is not None:
                expected_V = self.V_container - self.V_3  # 3.0 - V3
                if math.isclose(self.V, expected_V, rel_tol=0.02, abs_tol=0.05):
                    self._log(f"V = {self.V:.2f} L ≈ V_container - V3 = {expected_V:.2f} L", True)
                else:
                    self._log(f"V mismatch: {self.V:.2f} ≠ {expected_V:.2f}", False)
                    return False
            
            # Check against target answers
            if self.p_2 is not None:
                target_p2 = 2.5e5  # Pa
                if math.isclose(self.p_2, target_p2, rel_tol=0.02, abs_tol=1000.0):
                    self._log(f"p2 = {self.p_2:.1e} Pa ≈ target 2.5e5 Pa", True)
                else:
                    self._log(f"p2 = {self.p_2:.1e} Pa ≠ target 2.5e5 Pa", False)
                    return False
            
            if self.V is not None:
                target_V = 0.5  # L
                if math.isclose(self.V, target_V, rel_tol=0.02, abs_tol=0.05):
                    self._log(f"V = {self.V:.2f} L ≈ target 0.5 L", True)
                else:
                    self._log(f"V = {self.V:.2f} L ≠ target 0.5 L", False)
                    return False
            
            return True
        except Exception as e:
            self._log(f"test_trace_consistency failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb input (n_pumps) and verify expected trend."""
        try:
            # Store original values
            original_n_pumps = self.n_pumps
            
            # Perturb by +1 pump
            perturbed_n_pumps = original_n_pumps + 1
            
            # Recalculate using formulas (simulating the effect)
            V1_perturbed = self.V_gas_initial + perturbed_n_pumps * self.V_pump
            p2_perturbed = self.P_atm * V1_perturbed / self.V_gas_initial
            V3_perturbed = p2_perturbed * self.V_gas_initial / self.P_atm
            V_perturbed = self.V_container - V3_perturbed
            
            # Check monotonic trends
            if None not in (self.p_2, self.V):
                # More pumps → higher pressure
                if p2_perturbed > self.p_2:
                    self._log(f"Monotonicity (pressure): +1 pump → p2 increases {self.p_2:.0f} → {p2_perturbed:.0f} Pa", True)
                else:
                    self._log(f"Monotonicity failed: p2 should increase with more pumps", False)
                    return False
                
                # More pumps → less remaining disinfectant
                if V_perturbed < self.V:
                    self._log(f"Monotonicity (volume): +1 pump → V decreases {self.V:.2f} → {V_perturbed:.2f} L", True)
                else:
                    self._log(f"Monotonicity failed: V should decrease with more pumps", False)
                    return False
            else:
                self._log("Skipping monotonicity test (missing data)", None)
                return True  # Not a failure, just missing data
            
            return True
        except Exception as e:
            self._log(f"test_monotonicity failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        success = True
        
        tests = [
            ("Isothermal Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                success = False
        
        return success, self.logs