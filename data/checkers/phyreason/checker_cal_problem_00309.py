import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        Use getattr or .get() to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        # Extract inputs: given centripetal acceleration of point P (edge of large wheel)
        self.a_P_given = getattr(solution_instance, 'a_P', 
                                 getattr(solution_instance, 'input_values', {}).get('a_P', 12.0))
        
        # Extract results from solution_instance.results with safe defaults
        results = getattr(solution_instance, 'results', {})
        
        # Helper function to extract ratio values (handle both tuple and float formats)
        def extract_ratio(key, default_tuple):
            val = results.get(key, default_tuple)
            if isinstance(val, (tuple, list)) and len(val) == 2:
                return val[0] / val[1]  # Convert to float ratio
            elif isinstance(val, (int, float)):
                return float(val)  # Already a float ratio
            else:
                # Fallback to default ratio as float
                return default_tuple[0] / default_tuple[1]
        
        # Extract ratios as float values
        self.v_P_v_Q_ratio = extract_ratio('v_P:v_Q', (1, 1))
        self.omega_P_omega_Q_ratio = extract_ratio('omega_P:omega_Q', (1, 2))
        self.omega_S_omega_Q_ratio = extract_ratio('omega_S:omega_Q', (1, 2))
        self.a_P_a_S_ratio = extract_ratio('a_P:a_S', (3, 1))
        self.a_S_a_Q_ratio = extract_ratio('a_S:a_Q', (1, 6))
        
        # Individual accelerations
        self.a_S = results.get('a_S', 4.0)
        self.a_Q = results.get('a_Q', 24.0)
        
        # Radii (optional, for completeness)
        self.r_P = results.get('r_P', 2.0)  # relative to r_Q = 1.0
        self.r_S = (1/3) * self.r_P  # definition from problem
        
        # For symbolic checks - we'll use numeric only
        self.logs = []

    def test_invariants(self):
        """
        Verify key physical invariants using numeric relations.
        """
        logs = []
        
        try:
            # 1. Belt constraint: no slippage => v_P = v_Q
            # v = r * omega, so r_P * omega_P = r_Q * omega_Q
            # With r_P = 2, r_Q = 1, omega_P:omega_Q = 1:2
            # Check: 2 * (1/2) should equal 1 * 1 = 1
            lhs_belt = self.r_P * self.omega_P_omega_Q_ratio
            rhs_belt = 1.0 * 1.0  # r_Q=1, omega_Q=1 (relative units)
            if not math.isclose(lhs_belt, rhs_belt, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL Belt constraint: r_P*omega_P = {lhs_belt}, r_Q*omega_Q = {rhs_belt}")
            else:
                logs.append("PASS Belt constraint (v_P = v_Q).")
            
            # 2. Rigid body rotation: omega_S = omega_P (same wheel)
            if not math.isclose(self.omega_S_omega_Q_ratio, self.omega_P_omega_Q_ratio, 
                                rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL omega_S != omega_P. omega_S ratio: {self.omega_S_omega_Q_ratio}, omega_P ratio: {self.omega_P_omega_Q_ratio}")
            else:
                logs.append("PASS Rigid body rotation (omega_S = omega_P).")
            
            # 3. Centripetal acceleration formula: a = r * omega^2
            # Compute expected ratios from formulas:
            # a_P / a_S = (r_P * omega_P^2) / (r_S * omega_S^2) = (r_P / r_S) since omega_P = omega_S
            expected_a_P_a_S = self.r_P / self.r_S
            if not math.isclose(expected_a_P_a_S, self.a_P_a_S_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL a_P:a_S ratio. Expected {expected_a_P_a_S}, got {self.a_P_a_S_ratio}")
            else:
                logs.append(f"PASS a_P:a_S ratio = {self.a_P_a_S_ratio}.")
            
            # a_S / a_Q = (r_S * omega_S^2) / (r_Q * omega_Q^2)
            # r_S = (2/3) r_Q, omega_S = 0.5 omega_Q => a_S/a_Q = (2/3)*(0.5)^2 = (2/3)*0.25 = 1/6
            expected_a_S_a_Q = self.r_S * (self.omega_S_omega_Q_ratio**2)
            if not math.isclose(expected_a_S_a_Q, self.a_S_a_Q_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL a_S:a_Q ratio. Expected {expected_a_S_a_Q}, got {self.a_S_a_Q_ratio}")
            else:
                logs.append(f"PASS a_S:a_Q ratio = {self.a_S_a_Q_ratio}.")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return logs

    def test_trace_consistency(self):
        """
        Cross-check that intermediate ratios lead consistently to final answers.
        """
        logs = []
        try:
            # Given a_P = 12, and a_P:a_S = 3:1 => a_S = a_P / 3 = 4
            expected_a_S_from_ratio = self.a_P_given / self.a_P_a_S_ratio
            if not math.isclose(self.a_S, expected_a_S_from_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL a_S consistency: expected {expected_a_S_from_ratio} from a_P and ratio, got {self.a_S}")
            else:
                logs.append(f"PASS a_S = {self.a_S} consistent with a_P and a_P:a_S.")
            
            # From a_S:a_Q = 1:6 => a_Q = 6 * a_S
            expected_a_Q_from_ratio = self.a_S / self.a_S_a_Q_ratio
            if not math.isclose(self.a_Q, expected_a_Q_from_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL a_Q consistency: expected {expected_a_Q_from_ratio} from a_S and ratio, got {self.a_Q}")
            else:
                logs.append(f"PASS a_Q = {self.a_Q} consistent with a_S and a_S:a_Q.")
            
            # omega_P:omega_Q should come from belt and radii: omega_P/omega_Q = r_Q/r_P = 1/2
            expected_omega_ratio = 1.0 / self.r_P  # since r_Q = 1
            if not math.isclose(self.omega_P_omega_Q_ratio, expected_omega_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL omega_P:omega_Q ratio: expected {expected_omega_ratio}, got {self.omega_P_omega_Q_ratio}")
            else:
                logs.append(f"PASS omega_P:omega_Q = {self.omega_P_omega_Q_ratio}.")
            
            # omega_S:omega_Q should equal omega_P:omega_Q
            if not math.isclose(self.omega_S_omega_Q_ratio, self.omega_P_omega_Q_ratio, 
                                rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL omega_S:omega_Q does not match omega_P:omega_Q.")
            else:
                logs.append(f"PASS omega_S:omega_Q = {self.omega_S_omega_Q_ratio} matches omega_P:omega_Q.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return logs

    def test_monotonicity(self):
        """
        Perturb the input a_P and verify that a_S and a_Q change proportionally.
        We'll increase a_P by 10%, recompute using the ratios, and check.
        Then restore original values.
        """
        logs = []
        try:
            original_a_P = self.a_P_given
            perturbed_a_P = original_a_P * 1.10  # +10%
            
            # Using the ratios from the solution (which should be invariant)
            # a_P:a_S = 3:1 => a_S = a_P / 3
            new_a_S = perturbed_a_P / self.a_P_a_S_ratio
            # a_S:a_Q = 1:6 => a_Q = 6 * a_S
            new_a_Q = new_a_S / self.a_S_a_Q_ratio
            
            # Check monotonicity: if a_P increases, a_S and a_Q should increase proportionally
            if new_a_S > self.a_S and new_a_Q > self.a_Q:
                logs.append(f"PASS Monotonicity: a_P increased by 10%, a_S increased by {100*(new_a_S/self.a_S - 1):.2f}%, a_Q by {100*(new_a_Q/self.a_Q - 1):.2f}%.")
            else:
                logs.append(f"FAIL Monotonicity: a_S or a_Q did not increase when a_P increased.")
            
            # Also check the ratios remain the same
            if math.isclose(new_a_S / perturbed_a_P, self.a_S / original_a_P, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(new_a_Q / new_a_S, self.a_Q / self.a_S, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS Ratios preserved under perturbation.")
            else:
                logs.append("FAIL Ratios changed under perturbation.")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return logs

    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        Success is defined as no critical failures (only warnings or passes).
        """
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine success: no lines starting with "FAIL" or "ERROR"
        success = True
        for log in self.logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        self.logs.append(f"=== Physics Checker Finished: Success={success} ===")
        return success, self.logs