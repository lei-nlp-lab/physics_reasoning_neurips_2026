import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults from problem statement
        self.mA = getattr(solution_instance, 'mA', 2.4)
        self.mB = getattr(solution_instance, 'mB', 3.2)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        self.tan37 = self.sin37 / self.cos37
        
        # Safe extraction of results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.F_OD = results.get('F_OD', 0.0)
        self.F_OE = results.get('F_OE', 0.0)
        self.F_f = results.get('F_f', 0.0)
        self.F_fC = results.get('F_fC', 0.0)
        self.m_Agsin37circ = results.get('m_Agsin37circ', 0.0)
        
        # CORRECTED expected values from physics:
        # F_OE = (m_B * g)/cos37 = (3.2*10)/0.8 = 40.0
        # F_OD = F_OE * sin37 = 40 * 0.6 = 24.0
        # m_A*g*sin37 = 2.4*10*0.6 = 14.4
        # F_f = F_OD - m_A*g*sin37 = 24 - 14.4 = 9.6
        # F_fC = F_OD = 24.0 (from horizontal equilibrium of system)
        self.expected_F_OE = 40.0
        self.expected_F_OD = 24.0
        self.expected_F_f = 9.6
        self.expected_F_fC = 24.0  # CORRECTED: Should equal F_OD, not F_OE
        self.expected_m_Agsin37circ = 14.4
        
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append((success, msg))
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physical laws and step consistency."""
        success = True
        
        # Step 1: Equilibrium at point O
        # Vertical equilibrium: F_OE * cos37 = m_B * g  => F_OE = m_B * g / cos37
        # Horizontal equilibrium: F_OD = F_OE * sin37
        calc_F_OE = self.mB * self.g / self.cos37
        calc_F_OD = calc_F_OE * self.sin37
        
        if not self._isclose(self.F_OE, calc_F_OE):
            self._log(f"Step 1: F_OE mismatch. Expected {calc_F_OE}, got {self.F_OE}.", False)
            success = False
        else:
            self._log(f"Step 1: F_OE correct at {self.F_OE}.", True)
            
        if not self._isclose(self.F_OD, calc_F_OD):
            self._log(f"Step 1: F_OD mismatch. Expected {calc_F_OD}, got {self.F_OD}.", False)
            success = False
        else:
            self._log(f"Step 1: F_OD correct at {self.F_OD}.", True)
        
        # Step 2: Gravity component on A along incline
        calc_m_Agsin37circ = self.mA * self.g * self.sin37
        if not self._isclose(self.m_Agsin37circ, calc_m_Agsin37circ):
            self._log(f"Step 2: m_A*g*sin37 mismatch. Expected {calc_m_Agsin37circ}, got {self.m_Agsin37circ}.", False)
            success = False
        else:
            self._log(f"Step 2: m_A*g*sin37 correct at {self.m_Agsin37circ}.", True)
        
        # Step 3: Equilibrium on block A along incline
        # F_f = F_OD - m_A*g*sin37 (friction opposes motion down incline)
        calc_F_f = self.F_OD - self.m_Agsin37circ
        if not self._isclose(self.F_f, calc_F_f):
            # Allow for sign/direction misinterpretation, check magnitude
            if not self._isclose(abs(self.F_f), abs(calc_F_f)):
                self._log(f"Step 3: F_f mismatch. Expected {calc_F_f} (F_OD - m_A*g*sin37), got {self.F_f}.", False)
                success = False
            else:
                self._log(f"Step 3: F_f magnitude matches {abs(calc_F_f)}.", True)
        else:
            self._log(f"Step 3: F_f correct at {self.F_f}.", True)
        
        # Step 5: Horizontal equilibrium of system
        # CORRECTED: F_fC should equal F_OD, not F_OE
        if not self._isclose(self.F_fC, self.F_OD):
            self._log(f"Step 5: F_fC mismatch with F_OD. F_fC={self.F_fC}, F_OD={self.F_OD}.", False)
            success = False
        else:
            self._log(f"Step 5: F_fC = F_OD = {self.F_fC}.", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check if results match expected values from steps analysis
        expected_pairs = [
            (self.F_OE, self.expected_F_OE, "F_OE"),
            (self.F_OD, self.expected_F_OD, "F_OD"),
            (self.F_f, self.expected_F_f, "F_f"),
            (self.F_fC, self.expected_F_fC, "F_fC"),  # Now expects 24.0
            (self.m_Agsin37circ, self.expected_m_Agsin37circ, "m_A*g*sin37")
        ]
        
        for val, exp, name in expected_pairs:
            if not self._isclose(val, exp):
                self._log(f"Trace: {name} mismatch. Expected {exp}, got {val}.", False)
                success = False
        
        if success:
            self._log("All results match expected values from steps.", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb an input (mB) and verify expected trends."""
        success = True
        
        original_mB = self.mB
        perturbation = 0.01  # 1% increase
        
        # Increase mB by 1%
        self.mB = original_mB * (1 + perturbation)
        
        # Recalculate expected tensions
        new_F_OE = self.mB * self.g / self.cos37
        new_F_OD = new_F_OE * self.sin37
        
        # Check monotonicity: both should increase
        if new_F_OE <= self.F_OE or new_F_OD <= self.F_OD:
            self._log(f"Monotonicity: Increasing mB should increase tensions. Old F_OE={self.F_OE}, new={new_F_OE}; Old F_OD={self.F_OD}, new={new_F_OD}.", False)
            success = False
        else:
            self._log(f"Monotonicity: Tensions increase correctly with mB.", True)
        
        # Restore original mB
        self.mB = original_mB
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} failed with exception: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs