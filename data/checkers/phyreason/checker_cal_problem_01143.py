import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results from solution instance or its results dictionary
        results = getattr(solution_instance, 'results', {})
        self.U = getattr(solution_instance, 'U', results.get('U', 0.0))
        self.I1 = getattr(solution_instance, 'I_1', results.get('I_1', 0.0))
        self.U2 = getattr(solution_instance, 'U_2', results.get('U_2', 0.0))
        self.ratio_U1_U2 = getattr(solution_instance, 'U_1/U_2', results.get('U_1/U_2', 0.0))
        self.ratio_I1_I2 = getattr(solution_instance, 'I_1/I_2', results.get('I_1/I_2', 0.0))
        
        # Extract input parameters with defaults from problem statement
        self.R1 = getattr(solution_instance, 'R1', 100.0)
        self.R2 = getattr(solution_instance, 'R2', 7.5)
        self.n = 4.0  # turns ratio n1/n2 = 4:1
        
        # Known peak voltage from problem
        self.V_peak = 220.0 * math.sqrt(2)
        
        # Initialize logs
        self.logs = []
    
    def test_invariants(self):
        """Check fundamental physics laws: transformer ratios, KVL, Ohm's law, power conservation."""
        # 1. Check effective voltage from peak
        U_calc = self.V_peak / math.sqrt(2)
        if not math.isclose(self.U, U_calc, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Effective voltage U mismatch. Expected {U_calc:.2f}, got {self.U:.2f}")
        
        # 2. Compute derived quantities
        U1 = self.U - self.I1 * self.R1
        I2 = self.U2 / self.R2
        
        # 3. Transformer voltage ratio
        if not math.isclose(U1 / self.U2, self.n, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Transformer voltage ratio U1/U2 = {U1/self.U2:.3f}, expected {self.n:.1f}")
        
        # 4. Transformer current ratio
        if not math.isclose(self.I1 / I2, 1/self.n, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Transformer current ratio I1/I2 = {self.I1/I2:.3f}, expected {1/self.n:.3f}")
        
        # 5. Ohm's law for secondary
        if not math.isclose(self.U2, I2 * self.R2, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Ohm's law for secondary: U2 = {self.U2:.2f}, I2*R2 = {I2*self.R2:.2f}")
        
        # 6. Kirchhoff's voltage law for primary loop
        if not math.isclose(self.U, self.I1 * self.R1 + U1, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: KVL for primary loop: U = {self.U:.2f}, I1*R1+U1 = {self.I1*self.R1+U1:.2f}")
        
        # 7. Power conservation (ideal transformer, resistive loads)
        P_source = self.U * self.I1
        P_diss = self.I1**2 * self.R1 + I2**2 * self.R2
        if not math.isclose(P_source, P_diss, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Power conservation: P_source = {P_source:.2f} W, P_diss = {P_diss:.2f} W")
        
        if not any(log.startswith("FAIL") for log in self.logs[-7:]):
            self.logs.append("All invariant tests passed.")
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        # Expected values from steps analysis
        if not math.isclose(self.U, 220.0, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: U should be 220 V, got {self.U:.2f} V")
        if not math.isclose(self.I1, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: I1 should be 1 A, got {self.I1:.2f} A")
        if not math.isclose(self.U2, 30.0, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: U2 should be 30 V, got {self.U2:.2f} V")
        if not math.isclose(self.ratio_U1_U2, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Ratio U1/U2 should be 4, got {self.ratio_U1_U2:.3f}")
        if not math.isclose(self.ratio_I1_I2, 0.25, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: Ratio I1/I2 should be 0.25, got {self.ratio_I1_I2:.3f}")
        
        if not any(log.startswith("FAIL") for log in self.logs[-5:]):
            self.logs.append("All trace consistency tests passed.")
    
    def test_monotonicity(self):
        """Perturb R2 and verify expected trends in I1 and U2."""
        R2_original = self.R2
        R2_perturbed = R2_original * 1.01  # 1% increase
        
        # Formulas derived from transformer and circuit laws
        I1_original_calc = self.U / (self.R1 + self.n**2 * R2_original)
        U2_original_calc = self.n * R2_original * I1_original_calc
        
        I1_perturbed = self.U / (self.R1 + self.n**2 * R2_perturbed)
        U2_perturbed = self.n * R2_perturbed * I1_perturbed
        
        # Verify calculated values match reported results (within tolerance)
        if not math.isclose(I1_original_calc, self.I1, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: I1 from formula ({I1_original_calc:.3f} A) does not match result ({self.I1:.3f} A)")
        if not math.isclose(U2_original_calc, self.U2, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"FAIL: U2 from formula ({U2_original_calc:.3f} V) does not match result ({self.U2:.3f} V)")
        
        # Check monotonic trends: I1 decreases, U2 increases when R2 increases
        if I1_perturbed >= I1_original_calc:
            self.logs.append(f"FAIL: I1 did not decrease when R2 increased (from {I1_original_calc:.3f} to {I1_perturbed:.3f} A)")
        if U2_perturbed <= U2_original_calc:
            self.logs.append(f"FAIL: U2 did not increase when R2 increased (from {U2_original_calc:.3f} to {U2_perturbed:.3f} V)")
        
        if not any(log.startswith("FAIL") for log in self.logs[-4:]):
            self.logs.append("All monotonicity tests passed.")
    
    def run_all_tests(self):
        """Execute all checks and return success status and log messages."""
        self.logs = []
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success: no failures in logs
        success = not any(log.startswith("FAIL") for log in self.logs)
        return success, self.logs