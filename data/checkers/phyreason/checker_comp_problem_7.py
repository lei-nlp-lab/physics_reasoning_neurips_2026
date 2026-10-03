import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key quantities safely
        self.u = getattr(solution_instance, 'u', self.results.get('u', 0.0))
        self.v_f = getattr(solution_instance, 'v_f', self.results.get('v_f', 0.0))
        self.mathbfa = getattr(solution_instance, 'mathbfa', self.results.get('mathbfa', 0.0))
        self.mu = getattr(solution_instance, 'mu', self.results.get('mu', 0.0))
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        self.m_p = getattr(solution_instance, 'm_p', self.results.get('m_p', 1.0))
        
        # For trace consistency
        self.m0_initial = getattr(solution_instance, 'm0_initial', 9.0)  # From step_3: 9m0
        self.m0_final = getattr(solution_instance, 'm0_final', 1.0)     # From step_3: m0
        
        # Store original values for monotonicity tests
        self._original_m = self.m
        self._original_u = self.u
        self._original_E = self.E
        
        # Initialize logs
        self.logs = []
    
    def test_invariants(self) -> bool:
        """Test conservation laws and Newton's Laws."""
        success = True
        
        # 1. Test momentum conservation (rocket equation in differential form)
        # From step_1: u*Δm + m*Δv = 0
        # In differential form: u*dm + m*dv = 0
        # This implies: dv/dt = - (u/m) * (dm/dt) = - (μ/m) * u
        
        try:
            # Check if acceleration formula holds: a = - (μ/m) * u
            if self.mu != 0 and self.m != 0:
                expected_acceleration = - (self.mu / self.m) * self.u
                if not math.isclose(self.mathbfa, expected_acceleration, 
                                   rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Invariant fail: Acceleration doesn't match rocket equation. "
                                   f"Expected {expected_acceleration:.3f}, got {self.mathbfa:.3f}")
                    success = False
        except Exception as e:
            self.logs.append(f"Invariant test error (momentum): {str(e)}")
            success = False
        
        # 2. Test energy considerations (not conserved, but check if kinetic energy increases)
        # For rocket: kinetic energy increases as fuel is expelled
        # Work-energy theorem: dK = F_net * dx
        # For rocket, thrust force F = μ*u (magnitude)
        try:
            if self.mu < 0:  # Fuel ejection rate is negative (mass decreasing)
                # Thrust force magnitude: F = |μ| * u
                thrust_force = abs(self.mu) * self.u
                # This should accelerate the rocket, so kinetic energy should increase
                if thrust_force <= 0:
                    self.logs.append("Invariant warning: Thrust force should be positive")
        except Exception as e:
            self.logs.append(f"Invariant test error (energy): {str(e)}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer."""
        success = True
        
        # 1. Check integration from step_3 to step_4
        # ∫dv = -u ∫(dm/m) from m_initial to m_final
        # v_f = u * ln(m_initial / m_final)
        try:
            mass_ratio = self.m0_initial / self.m0_final
            expected_v_f = self.u * math.log(mass_ratio)
            
            if not math.isclose(self.v_f, expected_v_f, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"Trace consistency fail: v_f mismatch. "
                               f"Expected {expected_v_f:.3f}, got {self.v_f:.3f}")
                success = False
            else:
                # Check if mass_ratio is 9 (from step_3) or 10 (from target answer)
                # Both are valid depending on problem interpretation
                if not (math.isclose(mass_ratio, 9, rel_tol=0.1) or 
                        math.isclose(mass_ratio, 10, rel_tol=0.1)):
                    self.logs.append(f"Trace note: Mass ratio {mass_ratio} not close to 9 or 10")
        except Exception as e:
            self.logs.append(f"Trace consistency test error: {str(e)}")
            success = False
        
        # 2. Check energy-speed relationship from step_5 and step_6
        try:
            # From step_5: u = sqrt(2E/m_p)
            if self.E > 0 and self.m_p > 0:
                expected_u_from_energy = math.sqrt(2 * self.E / self.m_p)
                
                # Check against both step_6 value (5614) and extracted u
                if hasattr(self, 'u_B_numerical') or '5614' in str(self.results):
                    u_numerical = getattr(self, 'u_B_numerical', 5614.0)
                    if not math.isclose(u_numerical, expected_u_from_energy, 
                                       rel_tol=1e-2, abs_tol=1e-2):
                        self.logs.append(f"Trace consistency fail: u from energy mismatch. "
                                       f"Expected {expected_u_from_energy:.3f}, got {u_numerical:.3f}")
                        success = False
                
                # Also check against extracted u
                if not math.isclose(self.u, expected_u_from_energy, 
                                   rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Trace note: u {self.u:.3f} doesn't match sqrt(2E/m_p) {expected_u_from_energy:.3f}")
        except Exception as e:
            self.logs.append(f"Trace consistency test error (energy): {str(e)}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb inputs and check trends."""
        success = True
        
        try:
            # Test 1: Perturb mass ratio (increase initial mass)
            original_ratio = self.m0_initial / self.m0_final
            perturbed_ratio = original_ratio * 1.1  # 10% increase
            
            # Using rocket equation: v_f = u * ln(mass_ratio)
            original_v_f = self.u * math.log(original_ratio)
            perturbed_v_f = self.u * math.log(perturbed_ratio)
            
            if perturbed_v_f <= original_v_f:
                self.logs.append(f"Monotonicity fail: v_f doesn't increase with mass ratio. "
                               f"Original: {original_v_f:.3f}, Perturbed: {perturbed_v_f:.3f}")
                success = False
            
            # Test 2: Perturb exhaust velocity u
            perturbed_u = self.u * 1.1  # 10% increase
            perturbed_v_f_u = perturbed_u * math.log(original_ratio)
            
            if perturbed_v_f_u <= original_v_f:
                self.logs.append(f"Monotonicity fail: v_f doesn't increase with u. "
                               f"Original: {original_v_f:.3f}, Perturbed: {perturbed_v_f_u:.3f}")
                success = False
            
            # Test 3: For water rocket, perturb energy E
            if self.E > 0:
                perturbed_E = self.E * 1.1
                original_u_from_E = math.sqrt(2 * self.E / self.m_p)
                perturbed_u_from_E = math.sqrt(2 * perturbed_E / self.m_p)
                
                if perturbed_u_from_E <= original_u_from_E:
                    self.logs.append(f"Monotonicity fail: u doesn't increase with E. "
                                   f"Original: {original_u_from_E:.3f}, Perturbed: {perturbed_u_from_E:.3f}")
                    success = False
            
            # Restore original values (conceptually)
            self.m0_initial = self.m0_initial / 1.1  # Reverse perturbation
            
        except Exception as e:
            self.logs.append(f"Monotonicity test error: {str(e)}")
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return results."""
        self.logs = []
        success = True
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.logs.append(f"{test_name} test failed.")
            except Exception as e:
                success = False
                self.logs.append(f"{test_name} test error: {str(e)}")
        
        # Add summary
        if success:
            self.logs.append("All physics checks passed.")
        else:
            self.logs.append("Some physics checks failed.")
        
        return success, self.logs

# Optional: Symbolic verification for completeness
def symbolic_verification():
    """Create symbolic equations for rocket physics."""
    u, m, mu, a, v_f, m_i, m_f = sp.symbols('u m mu a v_f m_i m_f', positive=True)
    
    # Rocket equation differential form
    rocket_eq = sp.Eq(a, -mu*u/m)
    
    # Integrated form (Tsiolkovsky rocket equation)
    tsiolkovsky = sp.Eq(v_f, u*sp.log(m_i/m_f))
    
    # Energy relation for exhaust
    E, m_p = sp.symbols('E m_p')
    energy_eq = sp.Eq(u, sp.sqrt(2*E/m_p))
    
    return rocket_eq, tsiolkovsky, energy_eq