import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        # Safe extraction from solution instance or its results dict
        self.params = {}
        self.results = {}
        
        # Input parameters
        param_keys = ['m', 'Q', 'q', 'k', 'L', 'g', 'R', 'theta']
        for key in param_keys:
            self.params[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'inputs', {}).get(key, None))
        
        # Results from solution (using expected keys mapping)
        result_mapping = {
            'F_N': None, 'x': None, 'N_B': None, 'F': None, 'F_1': None,
            'N_Bx': None, 'N_B_': None, 'v_C': None, 'F_': None, 'N_By': None, 'F_2': None
        }
        for key in result_mapping:
            # Try direct attribute, then results dict, then None
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance, 'results', {}).get(key, None))
        
        # Additional derived results that might be needed
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'Q': sp.symbols('Q', real=True),
            'q': sp.symbols('q', real=True),
            'k': sp.symbols('k', positive=True),
            'L': sp.symbols('L', positive=True),
            'g': sp.symbols('g', positive=True),
            'R': sp.symbols('R', positive=True),
            'theta': sp.symbols('theta', real=True)
        }
    
    def _is_close(self, a, b):
        """Robust fuzzy numerical matching with 1% tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Check fundamental physics laws: Energy conservation and force balances."""
        logs = []
        try:
            m, Q, q, k, L, g, R, theta = [self.params.get(key) for key in ['m', 'Q', 'q', 'k', 'L', 'g', 'R', 'theta']]
            v_C = self.results.get('v_C')
            F = self.results.get('F')
            N_By = self.results.get('N_By')
            N_Bx = self.results.get('N_Bx')
            N_B = self.results.get('N_B')
            
            # 1. Energy conservation from A to B: mgR = 1/2 m v_C^2
            if all(val is not None for val in [m, g, R, v_C]):
                lhs = m * g * R
                rhs = 0.5 * m * v_C**2
                if not self._is_close(lhs, rhs):
                    logs.append(f"Energy conservation failed: mgR = {lhs}, 0.5 m v_C^2 = {rhs}")
                else:
                    logs.append("✓ Energy conservation (mgR = 1/2 m v_C^2) holds.")
            else:
                logs.append("⚠ Skipping energy conservation check due to missing parameters.")
            
            # 2. Newton's 2nd Law at point B (vertical)
            if all(val is not None for val in [m, g, R, v_C, N_By]):
                # N_By - mg = m v_C^2 / R
                lhs = N_By - m * g
                rhs = m * v_C**2 / R
                if not self._is_close(lhs, rhs):
                    logs.append(f"Vertical N2L at B failed: N_By - mg = {lhs}, m v_C^2/R = {rhs}")
                else:
                    logs.append("✓ Newton's 2nd Law (vertical) at point B holds.")
            else:
                logs.append("⚠ Skipping vertical N2L check due to missing parameters.")
            
            # 3. Horizontal force balance at B: N_Bx = F (electric force)
            if N_Bx is not None and F is not None:
                if not self._is_close(N_Bx, F):
                    logs.append(f"Horizontal force balance failed: N_Bx = {N_Bx}, F = {F}")
                else:
                    logs.append("✓ Horizontal force balance at B holds.")
            else:
                logs.append("⚠ Skipping horizontal force balance due to missing results.")
            
            # 4. Pythagorean check for N_B
            if all(val is not None for val in [N_B, N_Bx, N_By]):
                expected_N_B = math.sqrt(N_Bx**2 + N_By**2)
                if not self._is_close(N_B, expected_N_B):
                    logs.append(f"Pythagorean check failed: N_B = {N_B}, sqrt(N_Bx^2 + N_By^2) = {expected_N_B}")
                else:
                    logs.append("✓ Pythagorean theorem for N_B holds.")
            else:
                logs.append("⚠ Skipping Pythagorean check due to missing results.")
            
            # 5. Newton's 3rd Law: pressure equals normal force
            N_B_prime = self.results.get('N_B_')
            if N_B_prime is not None and N_B is not None:
                if not self._is_close(N_B_prime, N_B):
                    logs.append(f"Newton's 3rd Law failed: N_B' = {N_B_prime}, N_B = {N_B}")
                else:
                    logs.append("✓ Newton's 3rd Law (N_B' = N_B) holds.")
            
            # 6. Coulomb force magnitudes at B
            F_1 = self.results.get('F_1')
            F_2 = self.results.get('F_2')
            if all(val is not None for val in [k, Q, q, L, F_1, F_2]):
                expected_F = k * abs(Q * q) / L**2  # magnitude
                if not (self._is_close(F_1, expected_F) and self._is_close(F_2, expected_F)):
                    logs.append(f"Coulomb force magnitude mismatch: F_1={F_1}, F_2={F_2}, expected k|Qq|/L^2={expected_F}")
                else:
                    logs.append("✓ Coulomb force magnitudes at B are correct.")
            
        except Exception as e:
            logs.append(f"Error in invariant tests: {e}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            m, Q, q, k, L, g, R = [self.params.get(key) for key in ['m', 'Q', 'q', 'k', 'L', 'g', 'R']]
            v_C = self.results.get('v_C')
            F = self.results.get('F')
            N_B = self.results.get('N_B')
            F_N = self.results.get('F_N')
            
            # 1. Check v_C = sqrt(2gR)
            if all(val is not None for val in [v_C, g, R]):
                expected_v_C = math.sqrt(2 * g * R)
                if not self._is_close(v_C, expected_v_C):
                    logs.append(f"Trace inconsistency: v_C = {v_C}, expected sqrt(2gR) = {expected_v_C}")
                else:
                    logs.append("✓ v_C consistent with energy conservation.")
            
            # 2. Check F = kQq/L^2
            if all(val is not None for val in [F, k, Q, q, L]):
                expected_F = k * abs(Q * q) / L**2
                if not self._is_close(F, expected_F):
                    logs.append(f"Trace inconsistency: F = {F}, expected kQq/L^2 = {expected_F}")
                else:
                    logs.append("✓ F consistent with Coulomb's law.")
            
            # 3. Check N_B = sqrt(9 m^2 g^2 + (kQq/L^2)^2)
            if all(val is not None for val in [N_B, m, g, k, Q, q, L]):
                expected_N_B = math.sqrt(9 * m**2 * g**2 + (k * abs(Q * q) / L**2)**2)
                if not self._is_close(N_B, expected_N_B):
                    logs.append(f"Trace inconsistency: N_B = {N_B}, expected sqrt(9m^2g^2 + (kQq/L^2)^2) = {expected_N_B}")
                else:
                    logs.append("✓ N_B consistent with derived formula.")
            
            # 4. Check F_N = sqrt((8kQq/L^2 * cos^3θ)^2 + m^2 g^2)
            theta = self.params.get('theta')
            if all(val is not None for val in [F_N, m, g, k, Q, q, L, theta]):
                expected_F_N = math.sqrt((8 * k * abs(Q * q) / L**2 * math.cos(theta)**3)**2 + m**2 * g**2)
                if not self._is_close(F_N, expected_F_N):
                    logs.append(f"Trace inconsistency: F_N = {F_N}, expected formula gives {expected_F_N}")
                else:
                    logs.append("✓ F_N consistent with derived formula.")
            
        except Exception as e:
            logs.append(f"Error in trace consistency tests: {e}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends in outputs."""
        logs = []
        try:
            # Save original parameters
            orig_params = self.params.copy()
            orig_results = self.results.copy()
            
            # Test 1: Increase mass m -> N_B should increase
            m, g, k, Q, q, L = [self.params.get(key) for key in ['m', 'g', 'k', 'Q', 'q', 'L']]
            N_B = self.results.get('N_B')
            if all(val is not None for val in [m, g, k, Q, q, L, N_B]):
                delta_m = 0.1 * m  # 10% increase
                new_m = m + delta_m
                # Compute expected N_B using formula
                new_N_B = math.sqrt(9 * new_m**2 * g**2 + (k * abs(Q * q) / L**2)**2)
                if new_N_B > N_B:
                    logs.append("✓ Monotonicity: Increasing mass increases N_B.")
                else:
                    logs.append(f"Monotonicity violation: N_B did not increase when mass increased. Original: {N_B}, New: {new_N_B}")
            else:
                logs.append("⚠ Skipping mass monotonicity due to missing parameters.")
            
            # Test 2: Increase charge product |Qq| -> N_B should increase
            if all(val is not None for val in [m, g, k, Q, q, L, N_B]):
                delta_q = 0.1 * q if q != 0 else 0.1
                new_q = q + delta_q
                new_N_B = math.sqrt(9 * m**2 * g**2 + (k * abs(Q * new_q) / L**2)**2)
                if new_N_B > N_B:
                    logs.append("✓ Monotonicity: Increasing |Qq| increases N_B.")
                else:
                    logs.append(f"Monotonicity violation: N_B did not increase when |Qq| increased. Original: {N_B}, New: {new_N_B}")
            else:
                logs.append("⚠ Skipping charge monotonicity due to missing parameters.")
            
            # Test 3: Increase distance L -> N_B should decrease
            if all(val is not None for val in [m, g, k, Q, q, L, N_B]):
                delta_L = 0.1 * L
                new_L = L + delta_L
                new_N_B = math.sqrt(9 * m**2 * g**2 + (k * abs(Q * q) / new_L**2)**2)
                if new_N_B < N_B:
                    logs.append("✓ Monotonicity: Increasing L decreases N_B.")
                else:
                    logs.append(f"Monotonicity violation: N_B did not decrease when L increased. Original: {N_B}, New: {new_N_B}")
            else:
                logs.append("⚠ Skipping distance monotonicity due to missing parameters.")
            
            # Restore original parameters (just in case)
            self.params = orig_params
            self.results = orig_results
            
        except Exception as e:
            logs.append(f"Error in monotonicity tests: {e}")
            logs.append(traceback.format_exc())
            # Restore on error
            self.params = orig_params
            self.results = orig_results
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        logs = []
        success = True
        
        logs.append("=== PhysicsChecker starting ===")
        
        # Run tests
        inv_logs = self.test_invariants()
        logs.extend(inv_logs)
        
        trace_logs = self.test_trace_consistency()
        logs.extend(trace_logs)
        
        mono_logs = self.test_monotonicity()
        logs.extend(mono_logs)
        
        # Determine overall success
        for log in inv_logs + trace_logs + mono_logs:
            if log.startswith("Error") or ("failed" in log and not log.startswith("✓")) or ("violation" in log):
                success = False
        
        logs.append(f"=== PhysicsChecker finished: {'PASS' if success else 'FAIL'} ===")
        return success, logs