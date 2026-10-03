import sympy as sp
import math
import traceback
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (with safe defaults)
        self.m = getattr(solution_instance, 'm', 2.0e-3)
        self.d = getattr(solution_instance, 'd', 0.2)
        self.E_emf = getattr(solution_instance, 'E_emf', 15.0)  # Battery EMF
        self.r = getattr(solution_instance, 'r', 2.0)
        self.R1 = getattr(solution_instance, 'R1', 8.0)
        self.R2 = getattr(solution_instance, 'R2', 20.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        self.tan37 = self.sin37 / self.cos37  # Compute from given
        
        # Results with safe access
        self.U = self.results.get('U', 0.0)
        self.q = self.results.get('q', 0.0)
        self.a = self.results.get('a', 0.0)
        self.I = self.results.get('I', 0.0)
        self.E0 = self.results.get('E0', 0.0)  # Electric field
        
        # For monotonicity test backup
        self._original_m = self.m
        
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison for physics values"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        
        # 1. Ohm's Law for total circuit
        try:
            I_expected = self.E_emf / (self.r + self.R1 + self.R2)
            if not self.fuzzy_equal(self.I, I_expected):
                logs.append(f"Ohm's Law violation: I={self.I}, expected {I_expected}")
                all_pass = False
            else:
                logs.append("✓ Ohm's Law for total circuit satisfied")
        except Exception as e:
            logs.append(f"Error in Ohm's Law check: {str(e)}")
            all_pass = False
        
        # 2. Voltage across R2
        try:
            U_expected = self.I * self.R2
            if not self.fuzzy_equal(self.U, U_expected):
                logs.append(f"R2 voltage mismatch: U={self.U}, expected {U_expected}")
                all_pass = False
            else:
                logs.append("✓ Voltage across R2 consistent with Ohm's Law")
        except Exception as e:
            logs.append(f"Error in R2 voltage check: {str(e)}")
            all_pass = False
        
        # 3. Electric field from capacitor
        try:
            E0_expected = self.U / self.d
            if not self.fuzzy_equal(self.E0, E0_expected):
                logs.append(f"Electric field mismatch: E0={self.E0}, expected {E0_expected}")
                all_pass = False
            else:
                logs.append("✓ Electric field strength consistent with U/d")
        except Exception as e:
            logs.append(f"Error in electric field check: {str(e)}")
            all_pass = False
        
        # 4. Force equilibrium at angle (tangential components)
        try:
            # Electric force = q*E0, weight component = m*g*tan(37°)
            electric_force = self.q * self.E0
            weight_component = self.m * self.g * self.tan37
            
            if not self.fuzzy_equal(electric_force, weight_component):
                logs.append(f"Force equilibrium violation: qE0={electric_force}, mg*tan37={weight_component}")
                all_pass = False
            else:
                logs.append("✓ Force equilibrium satisfied (tangential components)")
        except Exception as e:
            logs.append(f"Error in force equilibrium check: {str(e)}")
            all_pass = False
        
        # 5. Net force after thread cut (Newton's 2nd Law)
        try:
            # From geometry: Tension = m*g/cos37°, net force = Tension when thread cut
            net_force_expected = self.m * self.g / self.cos37
            net_force_actual = self.m * self.a
            
            if not self.fuzzy_equal(net_force_actual, net_force_expected):
                logs.append(f"Newton's 2nd Law violation: ma={net_force_actual}, mg/cos37={net_force_expected}")
                all_pass = False
            else:
                logs.append("✓ Newton's 2nd Law consistent with force analysis")
        except Exception as e:
            logs.append(f"Error in Newton's 2nd Law check: {str(e)}")
            all_pass = False
        
        return all_pass, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        
        # Check step-by-step consistency from the steps analysis
        try:
            # Step 1: Current calculation
            I_calc = self.E_emf / (self.r + self.R1 + self.R2)
            if not self.fuzzy_equal(self.I, I_calc):
                logs.append(f"Step 1 inconsistency: I={self.I}, calculated {I_calc}")
                all_pass = False
            
            # Step 2-3: Voltage across R2
            U_calc = self.I * self.R2
            if not self.fuzzy_equal(self.U, U_calc):
                logs.append(f"Step 2-3 inconsistency: U={self.U}, calculated {U_calc}")
                all_pass = False
            
            # Step 4: Electric field
            E0_calc = self.U / self.d
            if not self.fuzzy_equal(self.E0, E0_calc):
                logs.append(f"Step 4 inconsistency: E0={self.E0}, calculated {E0_calc}")
                all_pass = False
            
            # Step 5-6: Charge calculation
            q_calc = (self.m * self.g * self.tan37) / self.E0
            if not self.fuzzy_equal(self.q, q_calc, abs_tol=1e-6):
                logs.append(f"Step 5-6 inconsistency: q={self.q}, calculated {q_calc}")
                all_pass = False
            
            # Step 7-9: Acceleration calculation
            a_calc = self.g / self.cos37
            if not self.fuzzy_equal(self.a, a_calc):
                logs.append(f"Step 7-9 inconsistency: a={self.a}, calculated {a_calc}")
                all_pass = False
            
            if all_pass:
                logs.append("✓ All trace steps are consistent")
                
        except Exception as e:
            logs.append(f"Error in trace consistency check: {str(e)}")
            all_pass = False
            
        return all_pass, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        
        try:
            # Perturb mass upward and check acceleration trend
            perturbation = 1.1  # +10%
            self.m = self._original_m * perturbation
            
            # Recalculate using formulas from trace consistency
            I_perturbed = self.E_emf / (self.r + self.R1 + self.R2)
            U_perturbed = I_perturbed * self.R2
            E0_perturbed = U_perturbed / self.d
            q_perturbed = (self.m * self.g * self.tan37) / E0_perturbed
            a_perturbed = self.g / self.cos37  # Note: a is independent of mass!
            
            # Check: Acceleration should NOT change with mass (from a = g/cos37)
            if not self.fuzzy_equal(a_perturbed, self.a):
                logs.append(f"Monotonicity violation: a changed from {self.a} to {a_perturbed} with mass perturbation")
                all_pass = False
            else:
                logs.append("✓ Acceleration independent of mass (as expected from a = g/cos37)")
            
            # Restore original mass
            self.m = self._original_m
            
            # Check charge monotonicity with mass
            q_original = (self.m * self.g * self.tan37) / (self.U / self.d)
            if q_perturbed > q_original:
                logs.append("✓ Charge increases with mass (positive correlation)")
            else:
                logs.append("✗ Charge does not increase with mass")
                all_pass = False
                
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            all_pass = False
            # Ensure restoration even on error
            self.m = self._original_m
            
        return all_pass, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
                status = "PASS" if success else "FAIL"
                all_logs.append(f"Result: {status}")
            except Exception as e:
                all_logs.append(f"Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs