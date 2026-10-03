import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with fallbacks - ensure they are numeric floats
        self.inputs['m'] = float(getattr(solution_instance, 'm', getattr(solution_instance, 'M', 1.0)))
        self.inputs['q'] = float(getattr(solution_instance, 'q', 1.0))
        self.inputs['v0'] = float(getattr(solution_instance, 'v0', 1.0))
        self.inputs['L'] = float(getattr(solution_instance, 'L', 1.0))
        
        # Extract results with fallbacks (from results dictionary if available)
        results_dict = getattr(solution_instance, 'results', {})
        self.results['U_0'] = float(results_dict.get('U_0', getattr(solution_instance, 'U_0', 0.0)))
        self.results['U_ac'] = float(results_dict.get('U_ac', getattr(solution_instance, 'U_ac', 0.0)))
        self.results['E'] = float(results_dict.get('E', getattr(solution_instance, 'E', 0.0)))
        self.results['t'] = float(results_dict.get('t', getattr(solution_instance, 't', 0.0)))
        self.results['x'] = float(results_dict.get('x', getattr(solution_instance, 'x', 0.0)))
        self.results['y'] = float(results_dict.get('y', getattr(solution_instance, 'y', 0.0)))
        self.results['v_x'] = float(results_dict.get('v_x', getattr(solution_instance, 'v_x', 0.0)))
        self.results['v_y'] = float(results_dict.get('v_y', getattr(solution_instance, 'v_y', 0.0)))
        self.results['v'] = float(results_dict.get('v', getattr(solution_instance, 'v', 0.0)))
        self.results['Eqy'] = float(results_dict.get('Eqy', getattr(solution_instance, 'Eqy', 0.0)))
        
    def test_invariants(self):
        """Verify fundamental physics laws: energy conservation, Newton's laws, motion equations."""
        logs = []
        try:
            # 1. Work-energy theorem for initial acceleration: qU_0 = 1/2 m v0^2
            lhs1 = self.inputs['q'] * self.results['U_0']
            rhs1 = 0.5 * self.inputs['m'] * self.inputs['v0']**2
            if not math.isclose(lhs1, rhs1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: q*U_0 = {lhs1}, expected 0.5*m*v0^2 = {rhs1}")
            
            # 2. Uniform motion in x-direction: L = v0 * t
            lhs2 = self.inputs['L']
            rhs2 = self.inputs['v0'] * self.results['t']
            if not math.isclose(lhs2, rhs2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: L = {lhs2}, expected v0*t = {rhs2}")
            
            # 3. Uniformly accelerated motion in y-direction: y = 1/2 * (qE/m) * t^2
            # Only test if E > 0 to avoid division by zero
            if self.results['E'] != 0:
                a_y = (self.inputs['q'] * self.results['E']) / self.inputs['m']
                lhs3 = self.results['y']
                rhs3 = 0.5 * a_y * self.results['t']**2
                if not math.isclose(lhs3, rhs3, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: y = {lhs3}, expected 0.5*(qE/m)*t^2 = {rhs3}")
            
            # 4. REMOVED: U_ac = E * L is NOT valid here because U_ac and E come from different scenarios
            # U_ac is from Part 2 (ion exits at point c), E is from Part 3 (ion exits at midpoint of bc)
            # These are different physical scenarios with different electric field strengths
            
            # 5. Kinetic energy at exit: v = sqrt(2)*v0
            expected_v = math.sqrt(2) * self.inputs['v0']
            if not math.isclose(self.results['v'], expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v = {self.results['v']}, expected sqrt(2)*v0 = {expected_v}")
            
            # 6. Velocity components at 45°: v_x = v0, v_y = v0
            if not math.isclose(self.results['v_x'], self.inputs['v0'], rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_x = {self.results['v_x']}, expected v0 = {self.inputs['v0']}")
            if not math.isclose(self.results['v_y'], self.inputs['v0'], rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_y = {self.results['v_y']}, expected v0 = {self.inputs['v0']}")
            
            # 7. Displacement-velocity relations: x = v_x * t, y = (v_y/2) * t
            lhs7_x = self.results['x']
            rhs7_x = self.results['v_x'] * self.results['t']
            if not math.isclose(lhs7_x, rhs7_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x = {lhs7_x}, expected v_x*t = {rhs7_x}")
            lhs7_y = self.results['y']
            rhs7_y = 0.5 * self.results['v_y'] * self.results['t']
            if not math.isclose(lhs7_y, rhs7_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: y = {lhs7_y}, expected (v_y/2)*t = {rhs7_y}")
            
            # 8. Geometric constraint: x = 2y
            if not math.isclose(self.results['x'], 2 * self.results['y'], rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x = {self.results['x']}, expected 2*y = {2 * self.results['y']}")
            
            # 9. Work-energy in electric field: Eqy = 1/2 m v0^2
            lhs9 = self.results['Eqy']
            rhs9 = 0.5 * self.inputs['m'] * self.inputs['v0']**2
            if not math.isclose(lhs9, rhs9, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Eqy = {lhs9}, expected 0.5*m*v0^2 = {rhs9}")
            
            # 10. Electric field magnitude from work: E = (1/2 m v0^2) / (q * y)
            # Protect against division by zero
            if self.inputs['q'] * self.results['y'] != 0:
                expected_E = (0.5 * self.inputs['m'] * self.inputs['v0']**2) / (self.inputs['q'] * self.results['y'])
                if not math.isclose(self.results['E'], expected_E, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: E = {self.results['E']}, expected (0.5*m*v0^2)/(q*y) = {expected_E}")
            
            if not logs:
                logs.append("PASS: All invariant checks passed.")
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Expected values from steps analysis
            expected_U0 = self.inputs['m'] * self.inputs['v0']**2 / (2 * self.inputs['q'])
            expected_Uac = 2 * self.inputs['m'] * self.inputs['v0']**2 / self.inputs['q']
            expected_E = self.inputs['m'] * self.inputs['v0']**2 / (self.inputs['q'] * self.inputs['L'])
            
            # Compare with results
            if not math.isclose(self.results['U_0'], expected_U0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_0 = {self.results['U_0']}, expected mv0^2/(2q) = {expected_U0}")
            
            if not math.isclose(self.results['U_ac'], expected_Uac, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_ac = {self.results['U_ac']}, expected 2mv0^2/q = {expected_Uac}")
            
            if not math.isclose(self.results['E'], expected_E, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E = {self.results['E']}, expected mv0^2/(qL) = {expected_E}")
            
            # Check that y = L/2 (from step 8)
            expected_y = self.inputs['L'] / 2
            if not math.isclose(self.results['y'], expected_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: y = {self.results['y']}, expected L/2 = {expected_y}")
            
            # Check that t = L/v0 (from step 2)
            expected_t = self.inputs['L'] / self.inputs['v0']
            if not math.isclose(self.results['t'], expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t = {self.results['t']}, expected L/v0 = {expected_t}")
            
            # Check that v_x = v0 and v_y = v0 (from step 6)
            if not math.isclose(self.results['v_x'], self.inputs['v0'], rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_x = {self.results['v_x']}, expected v0 = {self.inputs['v0']}")
            if not math.isclose(self.results['v_y'], self.inputs['v0'], rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_y = {self.results['v_y']}, expected v0 = {self.inputs['v0']}")
            
            if not logs:
                logs.append("PASS: All trace consistency checks passed.")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb an input (v0) and verify output trends match physical intuition."""
        logs = []
        try:
            # Store original values
            v0_orig = self.inputs['v0']
            U0_orig = self.results['U_0']
            Uac_orig = self.results['U_ac']
            E_orig = self.results['E']
            
            # Perturb v0 by +1%
            v0_new = v0_orig * 1.01
            
            # Recalculate expected outputs using formulas from steps analysis
            U0_new = self.inputs['m'] * v0_new**2 / (2 * self.inputs['q'])
            Uac_new = 2 * self.inputs['m'] * v0_new**2 / self.inputs['q']
            E_new = self.inputs['m'] * v0_new**2 / (self.inputs['q'] * self.inputs['L'])
            
            # Check monotonicity: all should increase
            if not (U0_new > U0_orig):
                logs.append(f"FAIL: U_0 did not increase with v0. Original: {U0_orig}, New: {U0_new}")
            if not (Uac_new > Uac_orig):
                logs.append(f"FAIL: U_ac did not increase with v0. Original: {Uac_orig}, New: {Uac_new}")
            if not (E_new > E_orig):
                logs.append(f"FAIL: E did not increase with v0. Original: {E_orig}, New: {E_new}")
            
            # Additional check: proportionality constants
            # U_0 ∝ v0^2, so ratio should be (1.01)^2 ≈ 1.0201
            expected_ratio = 1.01**2
            actual_ratio_U0 = U0_new / U0_orig if U0_orig != 0 else 0
            if not math.isclose(actual_ratio_U0, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_0 ratio {actual_ratio_U0} not close to v0^2 ratio {expected_ratio}")
            
            if not logs:
                logs.append("PASS: Monotonicity checks passed.")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        logs = []
        success = True
        
        # Run individual tests
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            test_logs = test_method()
            if test_logs and not (len(test_logs) == 1 and test_logs[0].startswith("PASS")):
                success = False
            logs.append(f"=== {test_name} ===")
            logs.extend(test_logs)
        
        return success, logs