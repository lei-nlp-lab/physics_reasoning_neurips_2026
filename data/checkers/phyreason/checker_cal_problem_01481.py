import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.mA = 2.0  # Given in problem
        self.mB = 1.0  # Given in problem
        self.h = 1.2   # Given in problem
        self.g = 10.0  # Given in problem
        
        # Safe extraction of results
        self.E_PA = getattr(solution_instance, 'E_PA', 
                           getattr(solution_instance, 'results', {}).get('E_PA', 0.0))
        self.v = getattr(solution_instance, 'v', 
                        getattr(solution_instance, 'results', {}).get('v', 0.0))
        
        # Initialize logs
        self.logs = []
    
    def test_invariants(self):
        """Test conservation of energy and potential energy formula."""
        try:
            # Test 1: Gravitational potential energy formula for ball A
            expected_E_PA = self.mA * self.g * self.h
            if not math.isclose(self.E_PA, expected_E_PA, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: E_PA = {self.E_PA}, expected {expected_E_PA}")
                return False
            
            # Test 2: Energy conservation when balls reach same height
            # Initial mechanical energy (B is at reference height 0, A at height h)
            E_initial = self.mA * self.g * self.h
            
            # When they reach same height, each has moved h/2
            # Final potential energy: mA*g*(h/2) + mB*g*(h/2)
            # Final kinetic energy: 0.5*(mA + mB)*v^2
            E_final_pot = (self.mA + self.mB) * self.g * (self.h / 2)
            E_final_kin = 0.5 * (self.mA + self.mB) * self.v**2
            E_final = E_final_pot + E_final_kin
            
            if not math.isclose(E_initial, E_final, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Energy conservation violated. Initial={E_initial}, Final={E_final}")
                return False
            
            # Test 3: Verify the energy equation from steps analysis
            # m_A*g*(h/2) - m_B*g*(h/2) = 0.5*(m_A + m_B)*v^2
            lhs = self.mA * self.g * (self.h / 2) - self.mB * self.g * (self.h / 2)
            rhs = 0.5 * (self.mA + self.mB) * self.v**2
            
            if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Energy equation mismatch. LHS={lhs}, RHS={rhs}")
                return False
            
            self.logs.append("PASS: All invariants satisfied")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        try:
            # From step analysis, v should be 2 m/s
            expected_v = 2.0
            if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: v = {self.v}, expected {expected_v}")
                return False
            
            # Check that E_PA leads to correct v via energy conservation
            # Rearranging the energy equation: v = sqrt(2*(mA - mB)*g*(h/2)/(mA + mB))
            expected_v_from_E = math.sqrt(2 * (self.mA - self.mB) * self.g * (self.h/2) / (self.mA + self.mB))
            
            if not math.isclose(self.v, expected_v_from_E, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: v inconsistent with E_PA. v={self.v}, computed={expected_v_from_E}")
                return False
            
            self.logs.append("PASS: Trace consistency verified")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Test that increasing mA increases both E_PA and v."""
        try:
            # Save original values
            original_mA = self.mA
            perturbed_mA = original_mA * 1.1  # Increase by 10%
            
            # Perturb mA and compute expected changes
            # E_PA should increase linearly with mA
            original_E_PA = original_mA * self.g * self.h
            perturbed_E_PA = perturbed_mA * self.g * self.h
            
            if perturbed_E_PA <= original_E_PA:
                self.logs.append(f"FAIL: E_PA not monotonic with mA. Original={original_E_PA}, Perturbed={perturbed_E_PA}")
                return False
            
            # v should increase with mA (since mA > mB, system accelerates in direction of A)
            original_v = math.sqrt(2 * (original_mA - self.mB) * self.g * (self.h/2) / (original_mA + self.mB))
            perturbed_v = math.sqrt(2 * (perturbed_mA - self.mB) * self.g * (self.h/2) / (perturbed_mA + self.mB))
            
            if perturbed_v <= original_v:
                self.logs.append(f"FAIL: v not monotonic with mA. Original={original_v}, Perturbed={perturbed_v}")
                return False
            
            # Test opposite perturbation: decreasing mA
            decreased_mA = original_mA * 0.9
            decreased_E_PA = decreased_mA * self.g * self.h
            decreased_v = math.sqrt(2 * (decreased_mA - self.mB) * self.g * (self.h/2) / (decreased_mA + self.mB))
            
            if decreased_E_PA >= original_E_PA or decreased_v >= original_v:
                self.logs.append(f"FAIL: Decreasing mA should decrease E_PA and v")
                return False
            
            self.logs.append("PASS: Monotonicity tests passed")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def test_boundary_conditions(self):
        """Test edge cases."""
        try:
            # Case 1: If mA = mB, then v should be 0 (balanced system)
            test_mA = self.mB
            test_v = math.sqrt(2 * (test_mA - self.mB) * self.g * (self.h/2) / (test_mA + self.mB))
            
            if not math.isclose(test_v, 0.0, abs_tol=1e-6):
                self.logs.append(f"FAIL: For mA=mB={test_mA}, expected v=0, got {test_v}")
                return False
            
            # Case 2: If h = 0, then E_PA = 0 and v = 0
            test_h = 0.0
            test_E_PA = self.mA * self.g * test_h
            test_v_h = math.sqrt(2 * (self.mA - self.mB) * self.g * (test_h/2) / (self.mA + self.mB))
            
            if not (math.isclose(test_E_PA, 0.0, abs_tol=1e-6) and 
                    math.isclose(test_v_h, 0.0, abs_tol=1e-6)):
                self.logs.append(f"FAIL: For h=0, E_PA={test_E_PA}, v={test_v_h}")
                return False
            
            # Case 3: If g = 0 (no gravity), then E_PA = 0 and v = 0
            test_g = 0.0
            test_E_PA_g = self.mA * test_g * self.h
            test_v_g = math.sqrt(2 * (self.mA - self.mB) * test_g * (self.h/2) / (self.mA + self.mB))
            
            if not (math.isclose(test_E_PA_g, 0.0, abs_tol=1e-6) and 
                    math.isclose(test_v_g, 0.0, abs_tol=1e-6)):
                self.logs.append(f"FAIL: For g=0, E_PA={test_E_PA_g}, v={test_v_g}")
                return False
            
            self.logs.append("PASS: Boundary conditions satisfied")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_boundary_conditions: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        success = True
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity,
            self.test_boundary_conditions
        ]
        
        for test_func in tests:
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self.logs.append(f"CRASH in {test_func.__name__}: {traceback.format_exc()}")
                success = False
        
        return success, self.logs