import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Input parameters (with safe defaults)
        self.N = getattr(solution_instance, 'N', 100)
        self.B = getattr(solution_instance, 'B', 0.8)
        self.ab = getattr(solution_instance, 'ab', 0.3)  # in meters
        self.ad = getattr(solution_instance, 'ad', 0.2)  # in meters
        self.omega = getattr(solution_instance, 'omega', 100 * math.pi)
        
        # Extract results with safe fallback
        results = getattr(solution_instance, 'results', {})
        self.Phi_m = getattr(solution_instance, 'Phi_m', results.get('Phi_m', 0.0))
        self.E_m = getattr(solution_instance, 'E_m', results.get('E_m', 0.0))
        self.e = getattr(solution_instance, 'e', results.get('e', 0.0))
        self.E_avg = getattr(solution_instance, 'E_avg', results.get('E_avg', 0.0))
        
        # Derived quantities
        self.A = self.ab * self.ad  # coil area in m²
        self.logs = []

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Maximum flux: Φ_max = N * B * A
        2. Maximum EMF: E_max = N * B * A * ω
        3. Instantaneous EMF: e = E_max * cos(ωt)
        4. Average EMF for 30° rotation from π/2 to 2π/3
        """
        try:
            # 1. Check maximum magnetic flux
            expected_Phi_max = self.N * self.B * self.A
            if not math.isclose(self.Phi_m, expected_Phi_max, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"Φ_max invariant failed: got {self.Phi_m}, expected {expected_Phi_max}")
            
            # 2. Check maximum induced EMF
            expected_E_max = self.N * self.B * self.A * self.omega
            if not math.isclose(self.E_m, expected_E_max, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"E_max invariant failed: got {self.E_m}, expected {expected_E_max}")
            
            # 3. Check instantaneous EMF form (symbolic and numerical)
            t = sp.symbols('t')
            expected_e_expr = self.E_m * sp.cos(self.omega * t)
            
            # Try to parse e if it's a string
            if isinstance(self.e, str):
                # Clean string: remove units, convert π to pi
                clean_e = self.e.replace('π', 'pi').replace('V', '').replace(' ', '').strip('$')
                try:
                    actual_e_expr = sp.sympify(clean_e)
                    # Compare at sample time points
                    for t_val in [0, 0.005, 0.01]:
                        expected_val = float(expected_e_expr.subs(t, t_val))
                        actual_val = float(actual_e_expr.subs(t, t_val))
                        if not math.isclose(actual_val, expected_val, rel_tol=0.02, abs_tol=0.02):
                            self.logs.append(f"e(t) mismatch at t={t_val}: got {actual_val}, expected {expected_val}")
                            break
                except Exception as e:
                    self.logs.append(f"Could not parse e expression: {self.e}. Error: {e}")
            
            # 4. Check average EMF for 30° rotation from π/2 to 2π/3
            theta0 = math.pi / 2  # initial angle (EMF maximum)
            theta1 = 2 * math.pi / 3  # 30° rotation
            delta_Phi = self.N * self.B * self.A * (math.cos(theta1) - math.cos(theta0))
            delta_t = (math.pi / 6) / self.omega  # time for π/6 rad at ω
            expected_E_avg = -delta_Phi / delta_t  # Faraday's law
            
            if not math.isclose(self.E_avg, expected_E_avg, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"E_avg invariant failed: got {self.E_avg}, expected {expected_E_avg}")
                
        except Exception as e:
            self.logs.append(f"Error in test_invariants: {traceback.format_exc()}")

    def test_trace_consistency(self):
        """
        Cross-check that intermediate results are consistent with each other.
        """
        try:
            # Check Φ_max → E_max relationship
            expected_E_m_from_Phi = self.Phi_m * self.omega
            if not math.isclose(self.E_m, expected_E_m_from_Phi, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"Trace: E_m should be Φ_m * ω: got {self.E_m}, expected {expected_E_m_from_Phi}")
            
            # Check E_max → E_avg for half-period? Not directly applicable.
            # Instead verify that the computed E_avg is between 0 and E_max
            if not (0 <= abs(self.E_avg) <= abs(self.E_m)):
                self.logs.append(f"Trace: E_avg={self.E_avg} should be between 0 and E_max={self.E_m}")
            
        except Exception as e:
            self.logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")

    def test_monotonicity(self):
        """
        Perturb B by +1%, recompute outputs, check monotonic trends.
        """
        try:
            original_B = self.B
            perturbed_B = original_B * 1.01  # +1%
            
            # Store original results
            orig_Phi_m = self.Phi_m
            orig_E_m = self.E_m
            orig_E_avg = self.E_avg
            
            # Recompute with perturbed B (using physics formulas)
            perturbed_Phi_m = self.N * perturbed_B * self.A
            perturbed_E_m = self.N * perturbed_B * self.A * self.omega
            
            # Recompute E_avg for perturbed B
            theta0 = math.pi / 2
            theta1 = 2 * math.pi / 3
            delta_Phi_pert = self.N * perturbed_B * self.A * (math.cos(theta1) - math.cos(theta0))
            delta_t = (math.pi / 6) / self.omega
            perturbed_E_avg = -delta_Phi_pert / delta_t
            
            # Check monotonicity: all should increase with B
            if not (perturbed_Phi_m > orig_Phi_m):
                self.logs.append(f"Monotonicity: Φ_m should increase with B")
            if not (perturbed_E_m > orig_E_m):
                self.logs.append(f"Monotonicity: E_m should increase with B")
            if not (perturbed_E_avg > orig_E_avg):
                self.logs.append(f"Monotonicity: E_avg should increase with B")
            
            # Restore original B (for completeness)
            self.B = original_B
            
        except Exception as e:
            self.logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")

    def run_all_tests(self):
        """
        Execute all tests and return success flag with logs.
        """
        self.logs = []
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        success = len(self.logs) == 0
        return success, self.logs