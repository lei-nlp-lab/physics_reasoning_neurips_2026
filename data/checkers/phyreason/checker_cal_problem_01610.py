import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with defaults
        self.inputs['L'] = getattr(solution_instance, 'L', 0.1)
        self.inputs['R'] = getattr(solution_instance, 'R', 0.01)
        self.inputs['m'] = getattr(solution_instance, 'm', 0.1)
        self.inputs['F'] = getattr(solution_instance, 'F', 0.1)
        self.inputs['t'] = getattr(solution_instance, 't', 2.0)
        self.inputs['B'] = getattr(solution_instance, 'B', 0.2)
        
        # Extract results with safe access
        results_dict = getattr(solution_instance, 'results', {})
        self.results['a_1'] = results_dict.get('a_1', 0.0)
        self.results['v'] = results_dict.get('v', 0.0)
        self.results['a'] = results_dict.get('a', 0.0)
        self.results['v_m'] = results_dict.get('v_m', 0.0)
        self.results['I'] = results_dict.get('I', 0.0)
        self.results['I_1'] = results_dict.get('I_1', 0.0)
        self.results['E'] = results_dict.get('E', 0.0)
        self.results['E_1'] = results_dict.get('E_1', 0.0)
        
        # Expected answers from problem statement
        self.expected_v = 2.0  # m/s
        self.expected_a = 0.2  # m/s²
        self.expected_v_m = 2.5  # m/s
        
        self.logs = []
        
    def _log(self, msg, success=None):
        """Helper to log messages and track success."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            msg = f"[{status}] {msg}"
        self.logs.append(msg)
        return success if success is not None else True
    
    def _fuzzy_eq(self, a, b, msg, rel_tol=1e-2, abs_tol=1e-2):
        """Check if two values are close with physics tolerance."""
        is_close = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        return self._log(f"{msg}: {a} ≈ {b} (tolerance {rel_tol})", is_close)
    
    def test_invariants(self):
        """Test fundamental physics laws."""
        success = True
        
        try:
            # 1. Newton's Second Law before magnetic field: F = m * a_1
            F_calc = self.inputs['m'] * self.results['a_1']
            success &= self._fuzzy_eq(
                self.inputs['F'], F_calc,
                f"Newton's 2nd Law (pre-field): F={self.inputs['F']} = m*a_1={F_calc}"
            )
            
            # 2. Kinematics before magnetic field: v = a_1 * t
            v_calc = self.results['a_1'] * self.inputs['t']
            success &= self._fuzzy_eq(
                self.results['v'], v_calc,
                f"Kinematics (pre-field): v={self.results['v']} = a_1*t={v_calc}"
            )
            
            # 3. Newton's 2nd Law at entry: F - B*I*L = m*a
            if self.results['I'] != 0.0:  # Only if I was calculated
                F_net = self.inputs['F'] - self.inputs['B'] * self.results['I'] * self.inputs['L']
                m_times_a = self.inputs['m'] * self.results['a']
                success &= self._fuzzy_eq(
                    F_net, m_times_a,
                    f"Newton's 2nd Law (at entry): F-BIL={F_net} = m*a={m_times_a}"
                )
            
            # 4. Faraday's Law at entry: E = B*L*v
            E_calc = self.inputs['B'] * self.inputs['L'] * self.results['v']
            success &= self._fuzzy_eq(
                self.results['E'], E_calc,
                f"Faraday's Law (at entry): E={self.results['E']} = B*L*v={E_calc}"
            )
            
            # 5. Ohm's Law at entry: I = E/R
            if self.results['E'] != 0.0:
                I_calc = self.results['E'] / self.inputs['R']
                success &= self._fuzzy_eq(
                    self.results['I'], I_calc,
                    f"Ohm's Law (at entry): I={self.results['I']} = E/R={I_calc}"
                )
            
            # 6. Force equilibrium at terminal velocity: F = B*I_1*L
            if self.results['I_1'] != 0.0:
                F_ampere = self.inputs['B'] * self.results['I_1'] * self.inputs['L']
                success &= self._fuzzy_eq(
                    self.inputs['F'], F_ampere,
                    f"Force equilibrium (terminal): F={self.inputs['F']} = B*I_1*L={F_ampere}"
                )
            
            # 7. Faraday's Law at terminal: E_1 = B*L*v_m
            E1_calc = self.inputs['B'] * self.inputs['L'] * self.results['v_m']
            success &= self._fuzzy_eq(
                self.results['E_1'], E1_calc,
                f"Faraday's Law (terminal): E_1={self.results['E_1']} = B*L*v_m={E1_calc}"
            )
            
            # 8. Ohm's Law at terminal: I_1 = E_1/R
            if self.results['E_1'] != 0.0:
                I1_calc = self.results['E_1'] / self.inputs['R']
                success &= self._fuzzy_eq(
                    self.results['I_1'], I1_calc,
                    f"Ohm's Law (terminal): I_1={self.results['I_1']} = E_1/R={I1_calc}"
                )
            
        except Exception as e:
            self._log(f"Exception in test_invariants: {traceback.format_exc()}", False)
            success = False
            
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        try:
            # Check that pre-entry velocity matches expected 2 m/s
            success &= self._fuzzy_eq(
                self.results['v'], self.expected_v,
                f"Entry velocity consistency: v={self.results['v']}, expected={self.expected_v}"
            )
            
            # Check that acceleration at entry matches expected 0.2 m/s²
            success &= self._fuzzy_eq(
                self.results['a'], self.expected_a,
                f"Entry acceleration consistency: a={self.results['a']}, expected={self.expected_a}"
            )
            
            # Check that terminal velocity matches expected 2.5 m/s
            success &= self._fuzzy_eq(
                self.results['v_m'], self.expected_v_m,
                f"Terminal velocity consistency: v_m={self.results['v_m']}, expected={self.expected_v_m}"
            )
            
            # Verify terminal velocity formula: v_m = F*R/(B²*L²)
            v_m_formula = (self.inputs['F'] * self.inputs['R'] / 
                          (self.inputs['B']**2 * self.inputs['L']**2))
            success &= self._fuzzy_eq(
                self.results['v_m'], v_m_formula,
                f"Terminal velocity formula: v_m={self.results['v_m']} = F*R/(B²L²)={v_m_formula}"
            )
            
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {traceback.format_exc()}", False)
            success = False
            
        return success
    
    def test_monotonicity(self):
        """Perturb an input and check if outputs change as expected."""
        success = True
        
        try:
            # Save original values
            original_F = self.inputs['F']
            
            # Perturb F by +10%
            perturbed_F = original_F * 1.1
            self.inputs['F'] = perturbed_F
            
            # Recalculate using physics formulas
            # 1. New pre-field acceleration
            a1_new = perturbed_F / self.inputs['m']
            
            # 2. New entry velocity
            v_new = a1_new * self.inputs['t']
            
            # 3. New terminal velocity
            v_m_new = (perturbed_F * self.inputs['R'] / 
                      (self.inputs['B']**2 * self.inputs['L']**2))
            
            # Check monotonicity: if F increases, all these should increase
            success &= self._log(
                f"Monotonicity check: F↑ ({original_F}→{perturbed_F}) → "
                f"a1↑ ({self.results['a_1']}→{a1_new})",
                a1_new > self.results['a_1']
            )
            
            success &= self._log(
                f"Monotonicity check: F↑ ({original_F}→{perturbed_F}) → "
                f"v_m↑ ({self.results['v_m']}→{v_m_new})",
                v_m_new > self.results['v_m']
            )
            
            # Restore original F
            self.inputs['F'] = original_F
            
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {traceback.format_exc()}", False)
            success = False
            # Ensure restoration even on error
            if 'original_F' in locals():
                self.inputs['F'] = original_F
                
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs.clear()
        self.logs.append("=== Physics Checker Started ===")
        
        overall_success = True
        
        # Run tests
        self.logs.append("\n--- Testing Physics Invariants ---")
        overall_success &= self.test_invariants()
        
        self.logs.append("\n--- Testing Trace Consistency ---")
        overall_success &= self.test_trace_consistency()
        
        self.logs.append("\n--- Testing Monotonicity ---")
        overall_success &= self.test_monotonicity()
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs