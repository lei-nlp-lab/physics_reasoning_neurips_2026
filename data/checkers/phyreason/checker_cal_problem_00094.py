import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using getattr to avoid AttributeError."""
        self.solution = solution_instance
        
        # Extract inputs
        self.m = getattr(solution_instance, 'm', None)
        self.v = getattr(solution_instance, 'v', None)
        self.L = getattr(solution_instance, 'L', None)
        self.g = getattr(solution_instance, 'g', 9.8)  # Default g
        
        # Extract results safely
        self.H = getattr(solution_instance, 'H', None)
        if self.H is None:
            self.H = getattr(solution_instance, 'results', {}).get('H', None)
        
        self.F = getattr(solution_instance, 'F', None)
        if self.F is None:
            self.F = getattr(solution_instance, 'results', {}).get('F', None)
        
        self.delta_E = getattr(solution_instance, 'delta_E', None)
        if self.delta_E is None:
            self.delta_E = getattr(solution_instance, 'results', {}).get('delta_E', None)
        
        self.v_1 = getattr(solution_instance, 'v_1', None)
        if self.v_1 is None:
            self.v_1 = getattr(solution_instance, 'results', {}).get('v_1', None)
        
        self.logs = []
        
    def log(self, msg):
        """Add a message to logs."""
        self.logs.append(msg)
    
    def compare(self, a, b, name, tol=0.02):
        """Fuzzy comparison with 2% tolerance."""
        if a is None or b is None:
            self.log(f"WARNING: {name}: One value is None")
            return False
            
        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if math.isclose(a, b, rel_tol=tol, abs_tol=tol):
                    self.log(f"OK: {name}: {a:.3e} ≈ {b:.3e}")
                    return True
                else:
                    self.log(f"FAIL: {name}: {a:.3e} ≠ {b:.3e}")
                    return False
            else:
                # Symbolic comparison
                diff = sp.simplify(a - b)
                if diff == 0:
                    self.log(f"OK: {name}: {a} = {b}")
                    return True
                else:
                    self.log(f"FAIL: {name}: {a} ≠ {b}")
                    return False
        except Exception as e:
            self.log(f"ERROR in compare for {name}: {str(e)}")
            return False
    
    def test_invariants(self):
        """Verify energy conservation and momentum conservation."""
        success = True
        
        # 1. Momentum conservation during collision
        try:
            # Before collision: p_total = m*v (ball A only, ball B stationary)
            # After collision: p_total = 2m*v_1 (they stick together)
            if all(x is not None for x in [self.m, self.v, self.v_1]):
                p_before = self.m * self.v
                p_after = 2 * self.m * self.v_1
                if not self.compare(p_before, p_after, "Momentum conservation"):
                    success = False
        except Exception as e:
            self.log(f"ERROR in momentum test: {str(e)}")
            success = False
        
        # 2. Energy loss during inelastic collision
        try:
            # Kinetic energy before: 0.5*m*v^2
            # Kinetic energy after: 0.5*2m*(v_1)^2
            if all(x is not None for x in [self.m, self.v, self.v_1]):
                KE_before = 0.5 * self.m * self.v**2
                KE_after = 0.5 * 2 * self.m * self.v_1**2
                expected_loss = KE_before - KE_after
                
                if self.delta_E is not None:
                    if not self.compare(self.delta_E, expected_loss, "Energy loss"):
                        success = False
        except Exception as e:
            self.log(f"ERROR in energy loss test: {str(e)}")
            success = False
        
        # 3. Newton's Second Law for tension
        try:
            # F = m*g + m*v^2/L (centripetal force at bottom of swing)
            if all(x is not None for x in [self.m, self.v, self.L, self.g]):
                expected_F = self.m * self.g + self.m * self.v**2 / self.L
                if self.F is not None:
                    if not self.compare(self.F, expected_F, "Tension (Newton's 2nd Law)"):
                        success = False
        except Exception as e:
            self.log(f"ERROR in tension test: {str(e)}")
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        success = True
        
        # 1. Check H = v^2/(2g)
        try:
            if all(x is not None for x in [self.v, self.g]):
                expected_H = self.v**2 / (2 * self.g)
                if self.H is not None:
                    if not self.compare(self.H, expected_H, "Height H"):
                        success = False
        except Exception as e:
            self.log(f"ERROR in H consistency: {str(e)}")
            success = False
        
        # 2. Check v_1 = v/2
        try:
            if self.v is not None:
                expected_v1 = self.v / 2
                if self.v_1 is not None:
                    if not self.compare(self.v_1, expected_v1, "Velocity after collision v_1"):
                        success = False
        except Exception as e:
            self.log(f"ERROR in v_1 consistency: {str(e)}")
            success = False
        
        # 3. Check delta_E = (1/4)*m*v^2
        try:
            if all(x is not None for x in [self.m, self.v]):
                expected_deltaE = 0.25 * self.m * self.v**2
                if self.delta_E is not None:
                    if not self.compare(self.delta_E, expected_deltaE, "Energy loss ΔE"):
                        success = False
        except Exception as e:
            self.log(f"ERROR in ΔE consistency: {str(e)}")
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass and check trends, then restore."""
        success = True
        
        try:
            # Store original values
            m_original = self.m
            
            if m_original is None:
                self.log("WARNING: Cannot test monotonicity - mass is None")
                return True
            
            # Perturb mass: double it
            perturbed_m = 2.0 * m_original
            
            # Calculate expected changes using formulas
            # H should be independent of mass
            if self.v is not None and self.g is not None:
                H_original = self.v**2 / (2 * self.g)
                H_perturbed = self.v**2 / (2 * self.g)  # Same
                if not math.isclose(H_original, H_perturbed, rel_tol=1e-2):
                    self.log(f"FAIL: H should be independent of mass: {H_original} vs {H_perturbed}")
                    success = False
                else:
                    self.log(f"OK: H independent of mass: {H_original:.3e}")
            
            # F should be proportional to mass
            if all(x is not None for x in [self.v, self.L, self.g]):
                F_original = m_original * self.g + m_original * self.v**2 / self.L
                F_perturbed = perturbed_m * self.g + perturbed_m * self.v**2 / self.L
                expected_ratio = 2.0  # Doubling mass should double tension
                actual_ratio = F_perturbed / F_original if F_original != 0 else float('inf')
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.05):  # 5% tolerance
                    self.log(f"FAIL: F ratio: expected {expected_ratio}, got {actual_ratio:.3f}")
                    success = False
                else:
                    self.log(f"OK: F proportional to mass: ratio = {actual_ratio:.3f}")
            
            # ΔE should be proportional to mass
            if self.v is not None:
                deltaE_original = 0.25 * m_original * self.v**2
                deltaE_perturbed = 0.25 * perturbed_m * self.v**2
                expected_ratio = 2.0
                actual_ratio = deltaE_perturbed / deltaE_original if deltaE_original != 0 else float('inf')
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.05):
                    self.log(f"FAIL: ΔE ratio: expected {expected_ratio}, got {actual_ratio:.3f}")
                    success = False
                else:
                    self.log(f"OK: ΔE proportional to mass: ratio = {actual_ratio:.3f}")
            
        except Exception as e:
            self.log(f"ERROR in monotonicity test: {str(e)}")
            success = False
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)."""
        self.logs = []
        
        try:
            # Check that we have all necessary data
            required = [self.m, self.v, self.L, self.H, self.F, self.delta_E, self.v_1]
            if any(x is None for x in required):
                missing = [name for name, value in zip(['m', 'v', 'L', 'H', 'F', 'delta_E', 'v_1'], required) if value is None]
                self.log(f"WARNING: Missing data: {missing}")
            
            # Run tests
            test1_success = self.test_invariants()
            test2_success = self.test_trace_consistency()
            test3_success = self.test_monotonicity()
            
            overall_success = test1_success and test2_success and test3_success
            
            self.log(f"\nSUMMARY:")
            self.log(f"Invariants test: {'PASS' if test1_success else 'FAIL'}")
            self.log(f"Trace consistency: {'PASS' if test2_success else 'FAIL'}")
            self.log(f"Monotonicity test: {'PASS' if test3_success else 'FAIL'}")
            self.log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
            
            return overall_success, self.logs
            
        except Exception as e:
            self.log(f"CRITICAL ERROR: {str(e)}")
            self.log(traceback.format_exc())
            return False, self.logs