import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Extract input parameters with safe fallbacks
        self.m = getattr(solution_instance, 'm', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.theta = getattr(solution_instance, 'theta', 0.5)
        
        # Extract results from the results dictionary if available
        results = getattr(solution_instance, 'results', {})
        
        # Use getattr with fallback to results dictionary, and finally a default
        self.P = getattr(solution_instance, 'P', results.get('P', 0.0))
        self.x = getattr(solution_instance, 'x', results.get('x', 0.0))
        self.v_2 = getattr(solution_instance, 'v_2', results.get('v_2', 0.0))
        self.v_D = getattr(solution_instance, 'v_D', results.get('v_D', 0.0))
        self.delta_E = getattr(solution_instance, 'delta_E', results.get('delta_E', 0.0))
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.v_0 = getattr(solution_instance, 'v_0', results.get('v_0', 0.0))
        self.v_1 = getattr(solution_instance, 'v_1', results.get('v_1', 0.0))
        
        # Handle v_2' (v_2_prime)
        v_2_prime_key = "v_2''"
        self.v_2_prime = getattr(solution_instance, 'v_2_prime', 
                                 results.get(v_2_prime_key, results.get('v_2_prime', 0.0)))
        
        # Compute expected numeric values based on physics formulas
        self._compute_expected_values()
    
    def _compute_expected_values(self):
        """Compute expected numeric values using physics formulas."""
        # Step 1: v_2' = sqrt(g*R)
        self.expected_v_2_prime = math.sqrt(self.g * self.R)
        
        # Step 2: t = sqrt(4R/g)
        self.expected_t = math.sqrt(4 * self.R / self.g)
        
        # Step 3-4: x = 2R
        self.expected_x = 2 * self.R
        
        # Step 5: v_D = sqrt(2gR cosθ)
        self.expected_v_D = math.sqrt(2 * self.g * self.R * math.cos(self.theta))
        
        # Step 6-7: P = m*g*sinθ * sqrt(2gR cosθ)
        self.expected_P = self.m * self.g * math.sin(self.theta) * math.sqrt(2 * self.g * self.R * math.cos(self.theta))
        
        # Step 8: v_2 = sqrt(5gR)
        self.expected_v_2 = math.sqrt(5 * self.g * self.R)
        
        # Step 9: v_1 = sqrt(2gR)
        self.expected_v_1 = math.sqrt(2 * self.g * self.R)
        
        # Step 10: v_0 = v_1 + v_2 = sqrt(2gR) + sqrt(5gR)
        self.expected_v_0 = self.expected_v_1 + self.expected_v_2
        
        # Step 11: ΔE = sqrt(10)*m*g*R
        self.expected_delta_E = math.sqrt(10) * self.m * self.g * self.R
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Compare two numeric values with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants using numeric values.
        """
        logs = []
        success = True
        
        try:
            # 1. Energy conservation for block B: 0.5*m*v_2^2 = 0.5*m*v_2_prime^2 + m*g*(2R)
            expected_energy_B = 0.5 * self.m * self.v_2**2
            actual_energy_B = 0.5 * self.m * self.v_2_prime**2 + self.m * self.g * (2 * self.R)
            if not self._is_close(expected_energy_B, actual_energy_B):
                logs.append(f"FAIL: Energy conservation for block B: {expected_energy_B} vs {actual_energy_B}")
                success = False
            else:
                logs.append("PASS: Energy conservation for block B")
            
            # 2. Energy conservation for block A: 0.5*m*v_1^2 = m*g*R
            expected_energy_A = 0.5 * self.m * self.v_1**2
            actual_energy_A = self.m * self.g * self.R
            if not self._is_close(expected_energy_A, actual_energy_A):
                logs.append(f"FAIL: Energy conservation for block A: {expected_energy_A} vs {actual_energy_A}")
                success = False
            else:
                logs.append("PASS: Energy conservation for block A")
            
            # 3. Momentum conservation: m*v_0 = m*v_1 + m*v_2
            left_momentum = self.m * self.v_0
            right_momentum = self.m * self.v_1 + self.m * self.v_2
            if not self._is_close(left_momentum, right_momentum):
                logs.append(f"FAIL: Momentum conservation: {left_momentum} vs {right_momentum}")
                success = False
            else:
                logs.append("PASS: Momentum conservation")
            
            # 4. Newton's 2nd law at top: m*g = m*v_2_prime^2 / R
            centripetal_force = self.m * self.v_2_prime**2 / self.R
            weight = self.m * self.g
            if not self._is_close(centripetal_force, weight):
                logs.append(f"FAIL: Centripetal force at top: {centripetal_force} vs {weight}")
                success = False
            else:
                logs.append("PASS: Centripetal force at top (Newton's 2nd law)")
        
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with expected numeric values.
        """
        logs = []
        success = True
        
        try:
            checks = [
                ("v_2_prime", self.v_2_prime, self.expected_v_2_prime),
                ("t", self.t, self.expected_t),
                ("x", self.x, self.expected_x),
                ("v_D", self.v_D, self.expected_v_D),
                ("P", self.P, self.expected_P),
                ("v_2", self.v_2, self.expected_v_2),
                ("v_1", self.v_1, self.expected_v_1),
                ("v_0", self.v_0, self.expected_v_0),
                ("delta_E", self.delta_E, self.expected_delta_E),
            ]
            
            for name, actual, expected in checks:
                if not self._is_close(actual, expected):
                    logs.append(f"FAIL: {name}: actual={actual:.4f}, expected={expected:.4f}")
                    success = False
                else:
                    logs.append(f"PASS: {name} matches expected value")
        
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Check monotonic trends by perturbing R and verifying expected changes.
        Use numeric derivatives instead of symbolic ones.
        """
        logs = []
        success = True
        
        try:
            # Small perturbation for R
            delta_R = 1e-3
            R_perturbed = self.R + delta_R
            
            # Recompute expected values with perturbed R
            v_2_prime_pert = math.sqrt(self.g * R_perturbed)
            t_pert = math.sqrt(4 * R_perturbed / self.g)
            x_pert = 2 * R_perturbed
            v_D_pert = math.sqrt(2 * self.g * R_perturbed * math.cos(self.theta))
            P_pert = self.m * self.g * math.sin(self.theta) * math.sqrt(2 * self.g * R_perturbed * math.cos(self.theta))
            v_2_pert = math.sqrt(5 * self.g * R_perturbed)
            v_1_pert = math.sqrt(2 * self.g * R_perturbed)
            v_0_pert = v_1_pert + v_2_pert
            delta_E_pert = math.sqrt(10) * self.m * self.g * R_perturbed
            
            # Compute numeric derivatives
            checks = [
                ("x", self.expected_x, x_pert),
                ("v_2_prime", self.expected_v_2_prime, v_2_prime_pert),
                ("v_2", self.expected_v_2, v_2_pert),
                ("v_1", self.expected_v_1, v_1_pert),
                ("v_0", self.expected_v_0, v_0_pert),
                ("delta_E", self.expected_delta_E, delta_E_pert),
                ("P", self.expected_P, P_pert),
            ]
            
            for name, base_val, pert_val in checks:
                derivative = (pert_val - base_val) / delta_R
                if derivative <= 0:
                    logs.append(f"WARNING: Derivative of {name} w.r.t. R is {derivative:.4f} (expected positive)")
                    success = False
                else:
                    logs.append(f"PASS: {name} increases with R (derivative ≈ {derivative:.4f})")
        
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and log messages.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs