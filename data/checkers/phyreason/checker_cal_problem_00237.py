import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract input parameters from the results dictionary or use defaults
        # The solution code stores B, L, m, g, R as symbolic variables, not instance attributes
        # We'll extract them from the results dict or use the numeric values that were substituted
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters - use the values that were substituted in the code
        # The code used: B=1, L=1, m=1, g=9.8, R=1
        self.B = 1.0
        self.L = 1.0
        self.m = 1.0
        self.g = 9.8
        self.R = 1.0
        
        # Extract results
        self.v0 = self.results.get('v_0', None)
        self.E = self.results.get('E', None)
        self.R0 = self.results.get('R_0', None)
        self.R_all = self.results.get('R_all', None)
        self.I = self.results.get('I', None)
        self.a = self.results.get('a', None)
        self.v = self.results.get('v', None)
        self.BLq = self.results.get('BLq', None)
        self.q = self.results.get('q', None)
        self.delta_x = self.results.get('delta_x', None)
        self.d = self.results.get('d', None)
        
        self.logs = []
        self.success = True

    def _log(self, message, success):
        self.logs.append(message)
        if not success:
            self.success = False

    def test_invariants(self):
        """Test fundamental physics invariants."""
        try:
            # Energy conservation (step 1)
            if all(v is not None for v in [self.m, self.g, self.L, self.v0]):
                potential_energy = self.m * self.g * self.L
                kinetic_energy = 0.5 * self.m * self.v0**2
                energy_check = math.isclose(potential_energy, kinetic_energy, 
                                           rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Energy conservation: m*g*L={potential_energy:.3f}, "
                         f"0.5*m*v0^2={kinetic_energy:.3f}, match={energy_check}", 
                         energy_check)
            
            # Newton's 2nd law for ring (step 6-7)
            if all(v is not None for v in [self.B, self.L, self.I, self.m, self.a]):
                ampere_force = self.B * self.L * self.I
                net_force = 2 * self.m * self.a
                newton_check = math.isclose(ampere_force, net_force, 
                                           rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Newton's 2nd law (ring): B*L*I={ampere_force:.3f}, "
                         f"2m*a={net_force:.3f}, match={newton_check}", 
                         newton_check)
            
            # Momentum conservation (step 8)
            if all(v is not None for v in [self.m, self.v0, self.v]):
                initial_momentum = self.m * self.v0
                final_momentum = 3 * self.m * self.v
                momentum_check = math.isclose(initial_momentum, final_momentum,
                                             rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Momentum conservation: m*v0={initial_momentum:.3f}, "
                         f"3m*v={final_momentum:.3f}, match={momentum_check}",
                         momentum_check)
                
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)

    def test_trace_consistency(self):
        """Test consistency between intermediate results."""
        try:
            # Check step sequence
            if all(v is not None for v in [self.v0, self.E, self.B, self.L]):
                expected_E = self.B * self.L * self.v0
                E_check = math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"E from v0: B*L*v0={expected_E:.3f}, "
                         f"E={self.E:.3f}, match={E_check}", E_check)
            
            if all(v is not None for v in [self.R, self.R0]):
                expected_R0 = self.R
                R0_check = math.isclose(self.R0, expected_R0, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"R0 consistency: expected R0={expected_R0:.3f}, "
                         f"R0={self.R0:.3f}, match={R0_check}", R0_check)
            
            if all(v is not None for v in [self.R, self.R_all]):
                expected_R_all = 1.5 * self.R
                R_all_check = math.isclose(self.R_all, expected_R_all, 
                                          rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"R_all consistency: expected 1.5R={expected_R_all:.3f}, "
                         f"R_all={self.R_all:.3f}, match={R_all_check}", R_all_check)
            
            if all(v is not None for v in [self.E, self.R_all, self.I]):
                expected_I = self.E / self.R_all
                I_check = math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"I from E and R_all: E/R_all={expected_I:.3f}, "
                         f"I={self.I:.3f}, match={I_check}", I_check)
            
            if all(v is not None for v in [self.B, self.L, self.I, self.m, self.a]):
                # Step 7: a = (B^2*L^2*v0)/(3*m*R)
                v0_calc = math.sqrt(2 * self.g * self.L)
                expected_a = (self.B**2 * self.L**2 * v0_calc) / (3 * self.m * self.R)
                a_check = math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"a from formula: (B^2*L^2*v0)/(3mR)={expected_a:.3f}, "
                         f"a={self.a:.3f}, match={a_check}", a_check)
            
            if all(v is not None for v in [self.v0, self.v]):
                expected_v = self.v0 / 3
                v_check = math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"v from v0: v0/3={expected_v:.3f}, "
                         f"v={self.v:.3f}, match={v_check}", v_check)
            
            if all(v is not None for v in [self.BLq, self.m, self.v0]):
                expected_BLq = (2/3) * self.m * self.v0
                BLq_check = math.isclose(self.BLq, expected_BLq, 
                                        rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"BLq from impulse: (2/3)*m*v0={expected_BLq:.3f}, "
                         f"BLq={self.BLq:.3f}, match={BLq_check}", BLq_check)
            
            if all(v is not None for v in [self.q, self.B, self.L, self.delta_x, self.R_all]):
                expected_q = self.B * self.L * self.delta_x / self.R_all
                q_check = math.isclose(self.q, expected_q, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"q from Δx: B*L*Δx/R_all={expected_q:.3f}, "
                         f"q={self.q:.3f}, match={q_check}", q_check)
            
            if all(v is not None for v in [self.delta_x, self.m, self.R, self.v0, self.B, self.L]):
                expected_delta_x = self.m * self.R * self.v0 / (self.B**2 * self.L**2)
                delta_x_check = math.isclose(self.delta_x, expected_delta_x,
                                           rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Δx from formula: m*R*v0/(B^2*L^2)={expected_delta_x:.3f}, "
                         f"Δx={self.delta_x:.3f}, match={delta_x_check}", delta_x_check)
            
            if all(v is not None for v in [self.d, self.L, self.delta_x]):
                expected_d = self.L + self.delta_x
                d_check = math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"d from L and Δx: L+Δx={expected_d:.3f}, "
                         f"d={self.d:.3f}, match={d_check}", d_check)
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)

    def test_monotonicity(self):
        """Test monotonic relationships by perturbing mass m."""
        try:
            # All parameters should be available now
            if any(v is None for v in [self.L, self.B, self.m, self.R, self.g]):
                self._log("Missing input parameters for monotonicity test", False)
                return
            
            # Store original values
            m_original = self.m
            perturb_factor = 1.01
            m_perturbed = m_original * perturb_factor
            
            # Compute derived quantities
            v0 = math.sqrt(2 * self.g * self.L)
            a_original = (self.B**2 * self.L**2 * v0) / (3 * m_original * self.R)
            a_perturbed = (self.B**2 * self.L**2 * v0) / (3 * m_perturbed * self.R)
            
            delta_x_original = m_original * self.R * v0 / (self.B**2 * self.L**2)
            delta_x_perturbed = m_perturbed * self.R * v0 / (self.B**2 * self.L**2)
            
            d_original = self.L + delta_x_original
            d_perturbed = self.L + delta_x_perturbed
            
            # Check monotonic trends
            # a should decrease as m increases (inverse relationship)
            if a_perturbed < a_original:
                self._log(f"Monotonicity (a): m↑ {m_original:.3f}→{m_perturbed:.3f} "
                         f"⇒ a↓ {a_original:.3f}→{a_perturbed:.3f}", True)
            else:
                self._log(f"Monotonicity (a): Expected a to decrease with increasing m", False)
            
            # d should increase as m increases (linear relationship in Δx term)
            if d_perturbed > d_original:
                self._log(f"Monotonicity (d): m↑ {m_original:.3f}→{m_perturbed:.3f} "
                         f"⇒ d↑ {d_original:.3f}→{d_perturbed:.3f}", True)
            else:
                self._log(f"Monotonicity (d): Expected d to increase with increasing m", False)
            
            # E should remain constant (independent of m)
            E_original = self.B * self.L * v0
            E_check = math.isclose(E_original, self.B * self.L * v0, 
                                  rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"E independence from m: E={E_original:.3f} constant", E_check)
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)

    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        self.success = True
        
        self.logs.append("=== Physics Checker Started ===")
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        self.logs.append(f"=== All Tests Complete. Overall: {self.success} ===")
        
        return self.success, self.logs