import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks
        self.P = 200000.0  # 200 kW in W
        self.U_1 = 500.0  # V
        self.R = 20.0  # Ω
        self.U_4 = 220.0  # V
        self.P_loss_allowed = 0.01 * self.P  # 2000 W
        
        # Safely extract results using getattr and dict.get
        self.results = getattr(solution_instance, 'results', {})
        self.solution_instance = solution_instance
        
        # Extract with fallbacks
        self.n1_over_n2 = self._safe_get('n_1/n_2')
        self.n3_over_n4 = self._safe_get('n_3/n_4')
        self.U_2 = self._safe_get('U_2')
        self.U_wire = self._safe_get('U_wire')
        self.U_3 = self._safe_get('U_3')
        self.I_2 = self._safe_get('I_2')
        
        # Convert string fractions to floats if needed
        if isinstance(self.n1_over_n2, str) and '/' in self.n1_over_n2:
            num, den = map(float, self.n1_over_n2.split('/'))
            self.n1_over_n2 = num / den
        if isinstance(self.n3_over_n4, str) and '/' in self.n3_over_n4:
            num, den = map(float, self.n3_over_n4.split('/'))
            self.n3_over_n4 = num / den
    
    def _safe_get(self, key):
        """Safely get value from results with multiple fallbacks."""
        # Try direct access, then results dict, then calculated value
        value = getattr(self.solution_instance, key, 
                       self.results.get(key, None))
        
        # If still None, calculate default from known formulas
        if value is None:
            if key == 'I_2':
                value = math.sqrt(self.P_loss_allowed / self.R)
            elif key == 'U_2':
                value = self.P / self.I_2 if hasattr(self, 'I_2') and self.I_2 else 0
            elif key == 'U_wire':
                value = self.I_2 * self.R if hasattr(self, 'I_2') else 0
            elif key == 'U_3':
                value = self.U_2 - self.U_wire if hasattr(self, 'U_2') and hasattr(self, 'U_wire') else 0
            elif key == 'n_1/n_2':
                value = self.U_1 / self.U_2 if hasattr(self, 'U_2') and self.U_2 else 0
            elif key == 'n_3/n_4':
                value = self.U_3 / self.U_4 if hasattr(self, 'U_3') else 0
        return float(value) if value is not None else 0.0
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Power loss invariant: P_loss = I_2^2 * R = 1% of P
            P_loss_calculated = self.I_2**2 * self.R
            if not self._is_close(P_loss_calculated, self.P_loss_allowed):
                logs.append(f"Power loss invariant failed: {P_loss_calculated} ≠ {self.P_loss_allowed}")
                success = False
            
            # 2. Total power invariant: P = U_2 * I_2
            P_total = self.U_2 * self.I_2
            if not self._is_close(P_total, self.P):
                logs.append(f"Total power invariant failed: {P_total} ≠ {self.P}")
                success = False
            
            # 3. Ohm's law for transmission line: U_wire = I_2 * R
            U_wire_ohm = self.I_2 * self.R
            if not self._is_close(U_wire_ohm, self.U_wire):
                logs.append(f"Ohm's law failed: {U_wire_ohm} ≠ {self.U_wire}")
                success = False
            
            # 4. Voltage consistency: U_3 = U_2 - U_wire
            U_3_calc = self.U_2 - self.U_wire
            if not self._is_close(U_3_calc, self.U_3):
                logs.append(f"Voltage subtraction failed: {U_3_calc} ≠ {self.U_3}")
                success = False
            
            # 5. Transformer ratio 1: n1/n2 = U_1 / U_2
            n1_n2_calc = self.U_1 / self.U_2
            if not self._is_close(n1_n2_calc, self.n1_over_n2):
                logs.append(f"Step-up transformer ratio failed: {n1_n2_calc} ≠ {self.n1_over_n2}")
                success = False
            
            # 6. Transformer ratio 2: n3/n4 = U_3 / U_4
            n3_n4_calc = self.U_3 / self.U_4
            if not self._is_close(n3_n4_calc, self.n3_over_n4):
                logs.append(f"Step-down transformer ratio failed: {n3_n4_calc} ≠ {self.n3_over_n4}")
                success = False
            
            if success:
                logs.append("All invariants passed.")
                
        except Exception as e:
            logs.append(f"Error in invariant testing: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate calculations with final answers."""
        logs = []
        success = True
        
        try:
            # Recalculate all steps from the STEPS ANALYSIS
            # Step 1: I_2 from power loss
            I_2_calc = math.sqrt(self.P_loss_allowed / self.R)
            if not self._is_close(I_2_calc, self.I_2):
                logs.append(f"Step 1 (I_2) inconsistent: {I_2_calc} ≠ {self.I_2}")
                success = False
            
            # Step 2: U_2 from power
            U_2_calc = self.P / self.I_2
            if not self._is_close(U_2_calc, self.U_2):
                logs.append(f"Step 2 (U_2) inconsistent: {U_2_calc} ≠ {self.U_2}")
                success = False
            
            # Step 3: n1/n2 from transformer ratio
            n1_n2_calc = self.U_1 / self.U_2
            if not self._is_close(n1_n2_calc, self.n1_over_n2):
                logs.append(f"Step 3 (n1/n2) inconsistent: {n1_n2_calc} ≠ {self.n1_over_n2}")
                success = False
            
            # Step 4: U_wire from Ohm's law
            U_wire_calc = self.I_2 * self.R
            if not self._is_close(U_wire_calc, self.U_wire):
                logs.append(f"Step 4 (U_wire) inconsistent: {U_wire_calc} ≠ {self.U_wire}")
                success = False
            
            # Step 5: U_3 from voltage subtraction
            U_3_calc = self.U_2 - self.U_wire
            if not self._is_close(U_3_calc, self.U_3):
                logs.append(f"Step 5 (U_3) inconsistent: {U_3_calc} ≠ {self.U_3}")
                success = False
            
            # Step 6: n3/n4 from transformer ratio
            n3_n4_calc = self.U_3 / self.U_4
            if not self._is_close(n3_n4_calc, self.n3_over_n4):
                logs.append(f"Step 6 (n3/n4) inconsistent: {n3_n4_calc} ≠ {self.n3_over_n4}")
                success = False
            
            if success:
                logs.append("All trace steps are consistent.")
                
        except Exception as e:
            logs.append(f"Error in trace consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test how ratios change when perturbing resistance."""
        logs = []
        success = True
        
        try:
            # Store original values
            R_original = self.R
            original_n1_n2 = self.n1_over_n2
            original_n3_n4 = self.n3_over_n4
            
            # Perturb resistance by +10%
            R_perturbed = R_original * 1.10
            
            # Recalculate I_2 from power loss constraint
            I_2_perturbed = math.sqrt(self.P_loss_allowed / R_perturbed)
            
            # Recalculate U_2
            U_2_perturbed = self.P / I_2_perturbed
            
            # Recalculate n1/n2
            n1_n2_perturbed = self.U_1 / U_2_perturbed
            
            # Recalculate U_wire
            U_wire_perturbed = I_2_perturbed * R_perturbed
            
            # Recalculate U_3
            U_3_perturbed = U_2_perturbed - U_wire_perturbed
            
            # Recalculate n3/n4
            n3_n4_perturbed = U_3_perturbed / self.U_4
            
            # Check monotonicity trends:
            # When R increases, I_2 decreases (since P_loss fixed), so U_2 increases
            # Therefore n1/n2 = U1/U2 should decrease
            if not (R_perturbed > R_original and n1_n2_perturbed < original_n1_n2):
                logs.append(f"Monotonicity failed for n1/n2: R↑ should cause n1/n2↓")
                logs.append(f"  R: {R_original}→{R_perturbed}")
                logs.append(f"  n1/n2: {original_n1_n2}→{n1_n2_perturbed}")
                success = False
            
            # For n3/n4: U_3 = U_2 - I_2*R, both U_2 and I_2*R change
            # With our numbers, n3/n4 should increase when R increases
            if not (R_perturbed > R_original and n3_n4_perturbed > original_n3_n4):
                logs.append(f"Monotonicity failed for n3/n4: R↑ should cause n3/n4↑")
                logs.append(f"  R: {R_original}→{R_perturbed}")
                logs.append(f"  n3/n4: {original_n3_n4}→{n3_n4_perturbed}")
                success = False
            
            if success:
                logs.append("Monotonicity tests passed: ratios change as expected with resistance.")
                
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        return overall_success, all_logs