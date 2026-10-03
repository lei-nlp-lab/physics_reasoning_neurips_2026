import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Safely initialize from solution instance with fallbacks."""
        # Extract inputs from problem context
        self.n = 2000
        self.S = 40e-4  # cm^2 to m^2
        self.r = 1.0
        self.R1 = 2.0
        self.R2 = 5.0
        self.d = 0.1  # cm to m
        self.g = 10.0
        self.dB_dt = 2.0  # T/s from step_2
        
        # Safely extract results with multiple fallbacks
        self.results = {}
        expected_keys = ['P', 'fracqm', 'fracDeltaBDeltat', 'I', 'U', 'E']
        for key in expected_keys:
            # Try direct attribute, then results dict, then set to None
            value = getattr(solution_instance, key, 
                          getattr(solution_instance, 'results', {}).get(key, None))
            self.results[key] = value
        
        # Target answers for verification
        self.target_P = 8.0  # W
        self.target_qm = 0.1  # C/kg
        
        # For symbolic computations
        self.symbols = self._init_symbols()
        
    def _init_symbols(self):
        """Initialize SymPy symbols for physical laws."""
        n, S, r, R1, R2, d, g, dB_dt = sp.symbols('n S r R1 R2 d g dB_dt')
        P, qm, I, U, E = sp.symbols('P qm I U E')
        return {
            'n': n, 'S': S, 'r': r, 'R1': R1, 'R2': R2,
            'd': d, 'g': g, 'dB_dt': dB_dt,
            'P': P, 'qm': qm, 'I': I, 'U': U, 'E': E
        }
    
    def _fuzzy_eq(self, a: float, b: float) -> bool:
        """Fuzzy numerical equality check with 1% tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def _eval_formula(self, expr_str: str, subs_dict: dict) -> float:
        """Safely evaluate a symbolic expression with numerical fallback."""
        try:
            expr = sp.sympify(expr_str)
            return float(expr.subs(subs_dict))
        except:
            # Numerical fallback using Python arithmetic
            return eval(expr_str, {"__builtins__": {}}, subs_dict)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physical laws (Faraday's Law, Ohm's Law, Equilibrium)."""
        logs = []
        success = True
        
        # Build substitution dictionary for symbolic checks
        subs_dict = {
            self.symbols['n']: self.n,
            self.symbols['S']: self.S,
            self.symbols['r']: self.r,
            self.symbols['R1']: self.R1,
            self.symbols['R2']: self.R2,
            self.symbols['d']: self.d,
            self.symbols['g']: self.g,
            self.symbols['dB_dt']: self.dB_dt,
            self.symbols['P']: self.results.get('P', 0.0),
            self.symbols['qm']: self.results.get('fracqm', 0.0),
            self.symbols['I']: self.results.get('I', 0.0),
            self.symbols['U']: self.results.get('U', 0.0),
            self.symbols['E']: self.results.get('E', 0.0)
        }
        
        # 1. Faraday's Law: E = n * (dB/dt) * S
        faraday_emf = self.n * self.dB_dt * self.S
        if not self._fuzzy_eq(self.results.get('E', 0.0), faraday_emf):
            logs.append(f"Faraday's Law violated: E={self.results.get('E'):.3f}, "
                       f"expected {faraday_emf:.3f}")
            success = False
        else:
            logs.append("✓ Faraday's Law satisfied")
        
        # 2. Ohm's Law for circuit: E = I * (R1 + R2 + r)
        circuit_resistance = self.R1 + self.R2 + self.r
        ohm_emf = self.results.get('I', 0.0) * circuit_resistance
        if not self._fuzzy_eq(self.results.get('E', 0.0), ohm_emf):
            logs.append(f"Ohm's Law (circuit) violated: E={self.results.get('E'):.3f}, "
                       f"expected {ohm_emf:.3f}")
            success = False
        else:
            logs.append("✓ Ohm's Law (circuit) satisfied")
        
        # 3. Power formula: P = I² * R1
        power_calc = (self.results.get('I', 0.0) ** 2) * self.R1
        if not self._fuzzy_eq(self.results.get('P', 0.0), power_calc):
            logs.append(f"Power formula violated: P={self.results.get('P'):.3f}, "
                       f"expected {power_calc:.3f}")
            success = False
        else:
            logs.append("✓ Power formula satisfied")
        
        # 4. Capacitor voltage: U = I * R2
        voltage_calc = self.results.get('I', 0.0) * self.R2
        if not self._fuzzy_eq(self.results.get('U', 0.0), voltage_calc):
            logs.append(f"Capacitor voltage violated: U={self.results.get('U'):.3f}, "
                       f"expected {voltage_calc:.3f}")
            success = False
        else:
            logs.append("✓ Capacitor voltage formula satisfied")
        
        # 5. Equilibrium condition: mg = q(U/d) => q/m = g*d/U
        if self.results.get('U', 0.0) != 0:
            qm_calc = (self.g * self.d) / self.results.get('U', 1.0)
            if not self._fuzzy_eq(self.results.get('fracqm', 0.0), qm_calc):
                logs.append(f"Equilibrium condition violated: q/m={self.results.get('fracqm'):.4f}, "
                           f"expected {qm_calc:.4f}")
                success = False
            else:
                logs.append("✓ Equilibrium condition satisfied")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check that intermediate results lead to final answers."""
        logs = []
        success = True
        
        # Get values with fallback defaults
        I = self.results.get('I', 0.0)
        U = self.results.get('U', 0.0)
        P = self.results.get('P', 0.0)
        qm = self.results.get('fracqm', 0.0)
        E = self.results.get('E', 0.0)
        
        # 1. Check power consistency: P should equal I²*R1
        power_from_I = I**2 * self.R1
        if not self._fuzzy_eq(P, power_from_I):
            logs.append(f"Power inconsistency: P={P:.3f}, but I²*R1={power_from_I:.3f}")
            success = False
        else:
            logs.append(f"✓ Power consistent: P={P:.3f} = I²*R1={power_from_I:.3f}")
        
        # 2. Check specific charge consistency: q/m should equal (g*d)/U
        if U != 0:
            qm_from_U = (self.g * self.d) / U
            if not self._fuzzy_eq(qm, qm_from_U):
                logs.append(f"Specific charge inconsistency: q/m={qm:.4f}, "
                           f"but (g*d)/U={qm_from_U:.4f}")
                success = False
            else:
                logs.append(f"✓ Specific charge consistent: q/m={qm:.4f} = (g*d)/U={qm_from_U:.4f}")
        
        # 3. Check EMF consistency through both paths
        faraday_emf = self.n * self.dB_dt * self.S
        ohm_emf = I * (self.R1 + self.R2 + self.r)
        
        if not (self._fuzzy_eq(E, faraday_emf) and self._fuzzy_eq(E, ohm_emf)):
            logs.append(f"EMF inconsistency: E={E:.3f}, Faraday={faraday_emf:.3f}, "
                       f"Ohm={ohm_emf:.3f}")
            success = False
        else:
            logs.append(f"✓ EMF consistent: E={E:.3f} = Faraday={faraday_emf:.3f} = Ohm={ohm_emf:.3f}")
        
        # 4. Check against target answers
        if not self._fuzzy_eq(P, self.target_P):
            logs.append(f"Power mismatch: P={P:.3f} vs target {self.target_P:.3f}")
            success = False
        else:
            logs.append(f"✓ Power matches target: {P:.3f} W")
            
        if not self._fuzzy_eq(qm, self.target_qm):
            logs.append(f"Specific charge mismatch: q/m={qm:.4f} vs target {self.target_qm:.4f}")
            success = False
        else:
            logs.append(f"✓ Specific charge matches target: {qm:.4f} C/kg")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb dB/dt and verify expected trends, then restore."""
        logs = []
        success = True
        
        # Store original values
        original_dB_dt = self.dB_dt
        original_I = self.results.get('I', 0.0)
        original_P = self.results.get('P', 0.0)
        original_U = self.results.get('U', 0.0)
        original_qm = self.results.get('fracqm', 0.0)
        
        try:
            # Perturb dB/dt by +10%
            perturbed_dB_dt = original_dB_dt * 1.1
            
            # Calculate chain of effects using formulas
            perturbed_E = self.n * perturbed_dB_dt * self.S
            perturbed_I = perturbed_E / (self.R1 + self.R2 + self.r)
            perturbed_P = perturbed_I**2 * self.R1
            perturbed_U = perturbed_I * self.R2
            perturbed_qm = (self.g * self.d) / perturbed_U if perturbed_U != 0 else 0
            
            # Check expected trends
            # When dB/dt increases:
            # - E increases (Faraday's law: E ∝ dB/dt)
            # - I increases (Ohm's law: I = E/R_total)
            # - P increases (P ∝ I²)
            # - U increases (U = I*R₂)
            # - q/m decreases (q/m = g*d/U ∝ 1/U)
            
            trend_checks = []
            
            # E should increase
            if perturbed_E > original_dB_dt * self.n * self.S * 0.99:  # Using original formula
                trend_checks.append(("E", "increases", True))
            else:
                trend_checks.append(("E", "increases", False))
            
            # I should increase
            if perturbed_I > original_I:
                trend_checks.append(("I", "increases", True))
            else:
                trend_checks.append(("I", "increases", False))
            
            # P should increase
            if perturbed_P > original_P:
                trend_checks.append(("P", "increases", True))
            else:
                trend_checks.append(("P", "increases", False))
            
            # U should increase
            if perturbed_U > original_U:
                trend_checks.append(("U", "increases", True))
            else:
                trend_checks.append(("U", "increases", False))
            
            # q/m should decrease (since U increased)
            if perturbed_qm < original_qm:
                trend_checks.append(("q/m", "decreases", True))
            else:
                trend_checks.append(("q/m", "decreases", False))
            
            # Log results
            for var, expected_trend, check_passed in trend_checks:
                if check_passed:
                    logs.append(f"✓ {var} {expected_trend} as expected when dB/dt increases")
                else:
                    logs.append(f"✗ {var} did not {expected_trend} when dB/dt increased")
                    success = False
            
            # Summary
            if success:
                logs.append("All monotonicity trends verified correctly.")
            else:
                logs.append("Some monotonicity trends violated.")
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
        finally:
            # Restore original value
            self.dB_dt = original_dB_dt
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n1. Physical Invariants Test:")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n2. Trace Consistency Test:")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n3. Monotonicity Test:")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        # Summary
        all_logs.append("\n=== Summary ===")
        if overall_success:
            all_logs.append("✅ All tests passed!")
        else:
            all_logs.append("❌ Some tests failed.")
        
        return overall_success, all_logs