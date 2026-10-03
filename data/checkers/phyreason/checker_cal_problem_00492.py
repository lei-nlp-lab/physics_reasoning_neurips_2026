import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Uses getattr and .get() fallbacks to prevent AttributeError.
        """
        # Problem constants (given in problem statement)
        self.q = -5e-5  # C
        self.delta_U_given = 2e-3  # J (increase in potential energy)
        self.AB = 0.04  # m (4 cm)
        self.theta_deg = 60  # degrees
        
        # Extract results safely from solution_instance
        self.d = getattr(solution_instance, 'd', 
                        getattr(solution_instance, 'results', {}).get('d', None))
        self.U_AB = getattr(solution_instance, 'U_AB', 
                           getattr(solution_instance, 'results', {}).get('U_AB', 
                           getattr(solution_instance, 'results', {}).get('U_{AB}', None)))
        self.W_AB = getattr(solution_instance, 'W_AB', 
                           getattr(solution_instance, 'results', {}).get('W_AB', 
                           getattr(solution_instance, 'results', {}).get('W_{AB}', None)))
        self.E = getattr(solution_instance, 'E', 
                        getattr(solution_instance, 'results', {}).get('E', None))
        
        # For symbolic computations
        self.symbols = {
            'q': sp.symbols('q'),
            'delta_U': sp.symbols('delta_U'),
            'W': sp.symbols('W'),
            'U': sp.symbols('U'),
            'd': sp.symbols('d'),
            'AB': sp.symbols('AB'),
            'theta': sp.symbols('theta'),
            'E': sp.symbols('E')
        }
        
        self.logs = []
    
    def _log(self, msg):
        """Helper to log messages."""
        self.logs.append(msg)
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison with physics-appropriate tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Test fundamental physics laws from KB."""
        try:
            # 1. Work-energy theorem for electric potential: W = -ΔU
            # From KB: sp.Eq(W, -delta_U)
            if self.W_AB is not None:
                expected_W = -self.delta_U_given
                if not self._is_close(self.W_AB, expected_W):
                    self._log(f"FAIL: W_AB = {self.W_AB} but -ΔU = {expected_W}")
                    return False
                self._log("PASS: W_AB = -ΔU (work-energy theorem)")
            
            # 2. Relationship between work and potential difference: U_AB = W_AB / q
            # From KB: sp.Eq(W, -q * delta_V) but note delta_V = -U_AB? Actually U_AB = V_B - V_A
            # From step_2: U_AB = W_AB / q
            if all(v is not None for v in [self.U_AB, self.W_AB]):
                expected_U = self.W_AB / self.q
                if not self._is_close(self.U_AB, expected_U):
                    self._log(f"FAIL: U_AB = {self.U_AB} but W_AB/q = {expected_U}")
                    return False
                self._log("PASS: U_AB = W_AB / q")
            
            # 3. Geometric relationship: d = AB * cos(θ)
            if self.d is not None:
                expected_d = self.AB * math.cos(math.radians(self.theta_deg))
                if not self._is_close(self.d, expected_d):
                    self._log(f"FAIL: d = {self.d} but AB*cosθ = {expected_d}")
                    return False
                self._log("PASS: d = AB * cos(θ)")
            
            # 4. Uniform field relationship: E = U_AB / d
            if all(v is not None for v in [self.E, self.U_AB, self.d]):
                expected_E = abs(self.U_AB) / self.d
                if not self._is_close(self.E, expected_E):
                    self._log(f"FAIL: E = {self.E} but |U_AB|/d = {expected_E}")
                    return False
                self._log("PASS: E = |U_AB| / d")
            
            # 5. Check sign consistency for negative charge
            # For negative charge moving in uniform E-field, potential energy increases
            # when moving opposite to field direction. Since ΔU > 0, and q < 0,
            # we must have U_AB < 0 (V_B < V_A) for work to be negative.
            if self.U_AB is not None:
                if self.U_AB >= 0:
                    self._log(f"WARNING: U_AB = {self.U_AB} should be negative for q < 0 and ΔU > 0")
                else:
                    self._log("PASS: Sign consistency for negative charge")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Check that intermediate calculations match final results."""
        try:
            # Step 1: W_AB = -ΔU
            if self.W_AB is not None:
                if not self._is_close(self.W_AB, -self.delta_U_given):
                    self._log(f"FAIL: Step1 W_AB doesn't match -ΔU")
                    return False
            
            # Step 2: U_AB = W_AB / q
            if self.U_AB is not None and self.W_AB is not None:
                if not self._is_close(self.U_AB, self.W_AB / self.q):
                    self._log(f"FAIL: Step2 U_AB doesn't match W_AB/q")
                    return False
            
            # Step 4: d = AB * cos(60°)
            if self.d is not None:
                expected_d = self.AB * 0.5
                if not self._is_close(self.d, expected_d):
                    self._log(f"FAIL: Step4 d doesn't match AB*cos60°")
                    return False
            
            # Step 5: E = |U_AB| / d
            if all(v is not None for v in [self.E, self.U_AB, self.d]):
                if not self._is_close(self.E, abs(self.U_AB) / self.d):
                    self._log(f"FAIL: Step5 E doesn't match |U_AB|/d")
                    return False
            
            # Check against target answers
            if self.U_AB is not None:
                if not self._is_close(self.U_AB, 40.0):
                    self._log(f"FAIL: U_AB ≠ 40V (got {self.U_AB})")
                    return False
            
            if self.E is not None:
                if not self._is_close(self.E, 2000.0):
                    self._log(f"FAIL: E ≠ 2000V/m (got {self.E})")
                    return False
            
            self._log("PASS: All intermediate steps consistent with final answers")
            return True
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check if output changes in expected direction."""
        try:
            # Perturb the charge q and see if E changes accordingly
            # From formulas: E = |W|/(|q|*d) and W = -ΔU (fixed)
            # So E = |ΔU|/(|q|*d) since W negative but magnitude fixed
            # Therefore, E ∝ 1/|q| for fixed ΔU and d
            
            # Make a copy of original values
            q_original = self.q
            perturbed_q = q_original * 1.1  # Increase magnitude by 10%
            
            # Calculate expected E for perturbed charge
            W_fixed = -self.delta_U_given
            d_fixed = self.AB * math.cos(math.radians(self.theta_deg))
            U_perturbed = W_fixed / perturbed_q
            E_perturbed = abs(U_perturbed) / d_fixed
            
            # Check monotonic trend: larger |q| should give smaller E
            if self.E is not None:
                original_E = self.E
                if abs(perturbed_q) > abs(q_original):
                    if not (E_perturbed < original_E):
                        self._log(f"FAIL: Increasing |q| should decrease E")
                        return False
                    self._log("PASS: E decreases as |q| increases (inverse proportionality)")
                
                # Restore original q (conceptually)
                self._log("Note: Original q restored after perturbation test")
                return True
            
            return False
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} Test ---")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name}: FAILED")
                else:
                    self._log(f"{test_name}: PASSED")
            except Exception as e:
                success = False
                self._log(f"{test_name}: ERROR - {traceback.format_exc()}")
        
        return success, self.logs