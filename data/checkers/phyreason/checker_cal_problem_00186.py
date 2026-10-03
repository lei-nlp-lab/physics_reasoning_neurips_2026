import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        """
        self.solution = solution_instance
        
        # Input constants (with units converted to SI)
        self.V_total = 380e-6          # m³ (total bottle volume)
        self.V_water_surface = 80e-6   # m³ (water at surface)
        self.V_water_final = 230e-6    # m³ (water after stabilization)
        self.p0 = 1.0e5                # Pa (atmospheric pressure)
        self.rho = 1.0e3               # kg/m³ (water density)
        self.g = 10.0                  # m/s² (gravity)
        
        # Gas volumes
        self.V0 = self.V_total - self.V_water_surface  # Initial gas volume: 300 mL = 300e-6 m³
        self.V = self.V_total - self.V_water_final     # Final gas volume: 150 mL = 150e-6 m³
        
        # Extract results from solution (with safe fallbacks)
        self.p = getattr(solution_instance, 'p', 
                        getattr(solution_instance, 'results', {}).get('p', 0.0))
        self.h = getattr(solution_instance, 'h', 
                        getattr(solution_instance, 'results', {}).get('h', 0.0))
        
        # Expected values from KB analysis
        self.p_expected = 2.0e5  # Pa
        self.h_expected = 10.0   # m
        
        self.logs = []
    
    def test_invariants(self):
        """Verify Boyle's Law and pressure-depth relation."""
        try:
            # 1. Boyle's Law: p0*V0 = p*V
            left = self.p0 * self.V0
            right = self.p * self.V
            
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Boyle's Law violated: p0*V0={left:.2f}, p*V={right:.2f}")
                return False
            self.logs.append("PASS: Boyle's Law satisfied (p0*V0 = p*V)")
            
            # 2. Pressure-depth relation: p = p0 + ρ*g*h
            p_calc = self.p0 + self.rho * self.g * self.h
            
            if not math.isclose(self.p, p_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Pressure-depth relation violated: p={self.p:.2f}, p0+ρgh={p_calc:.2f}")
                return False
            self.logs.append("PASS: Pressure-depth relation satisfied (p = p0 + ρ*g*h)")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Recalculate p from Boyle's Law
            p_calc = self.p0 * (self.V0 / self.V)
            if not math.isclose(self.p, p_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Pressure inconsistency: given p={self.p:.2e}, calculated p={p_calc:.2e}")
                return False
            self.logs.append(f"PASS: Pressure consistency check (p = p0*V0/V = {p_calc:.2e} Pa)")
            
            # Recalculate h from pressure-depth relation
            h_calc = (self.p - self.p0) / (self.rho * self.g)
            if not math.isclose(self.h, h_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Depth inconsistency: given h={self.h:.2f}, calculated h={h_calc:.2f}")
                return False
            self.logs.append(f"PASS: Depth consistency check (h = (p-p0)/(ρg) = {h_calc:.2f} m)")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb input and verify physical trends."""
        try:
            # Perturb initial gas volume V0 by +1%
            V0_perturbed = self.V0 * 1.01
            p_perturbed = self.p0 * (V0_perturbed / self.V)
            h_perturbed = (p_perturbed - self.p0) / (self.rho * self.g)
            
            # Expected trend: Increasing V0 should increase p (Boyle's Law)
            if not (p_perturbed > self.p):
                self.logs.append(f"FAIL: Monotonicity violated for V0->p: {p_perturbed:.2e} <= {self.p:.2e}")
                return False
            
            # Expected trend: Increased p should increase h
            if not (h_perturbed > self.h):
                self.logs.append(f"FAIL: Monotonicity violated for p->h: {h_perturbed:.2f} <= {self.h:.2f}")
                return False
            
            self.logs.append("PASS: Monotonicity checks (increasing V0 increases p and h)")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def test_symbolic_laws(self):
        """Verify using symbolic physics laws."""
        try:
            # Symbolic Boyle's Law
            p0, V0, p, V = sp.symbols('p0 V0 p V')
            eq1 = sp.Eq(p0*V0, p*V)
            # Substitute numerical values
            subs_dict = {p0: self.p0, V0: self.V0, p: self.p, V: self.V}
            eq1_check = eq1.subs(subs_dict)
            if not eq1_check:
                self.logs.append("FAIL: Symbolic Boyle's Law check failed")
                return False
            self.logs.append("PASS: Symbolic Boyle's Law satisfied")
            
            # Symbolic pressure-depth relation
            rho, g, h = sp.symbols('rho g h')
            eq2 = sp.Eq(p, p0 + rho*g*h)
            subs_dict.update({rho: self.rho, g: self.g, h: self.h})
            eq2_check = eq2.subs(subs_dict)
            if not eq2_check:
                self.logs.append("FAIL: Symbolic pressure-depth relation check failed")
                return False
            self.logs.append("PASS: Symbolic pressure-depth relation satisfied")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_symbolic_laws: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Symbolic Laws", self.test_symbolic_laws)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== {test_name} Test ===")
            if not test_func():
                success = False
        
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs