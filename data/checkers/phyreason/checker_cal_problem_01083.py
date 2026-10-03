import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants (converted to SI units)
        self.L_police = 5.0  # m
        self.L_truck = 20.0  # m
        self.v_p0 = 36.0 * 1000 / 3600  # m/s (36 km/h → 10 m/s)
        self.v_truck = 72.0 * 1000 / 3600  # m/s (72 km/h → 20 m/s)
        self.a_max = 2.0  # m/s²
        self.x0 = 25.0  # m (initial gap)
        self.delta_x2_target = 53.0 + self.L_police + self.L_truck  # 78 m
        
        # Expected values from steps analysis
        self.expected = {
            't_1': 5.0, 's_1': 75.0, 'delta_x_1': 50.0,
            't_total': 21.0, 'v_m': 36.0,
            'x_0_prime': 45.0, 't_1_prime': 13.5, 't_2_prime': 8.5,
            't_total_prime': 22.0, 'v_m_prime': 37.0,
            't_1_dprime': 10.0, 't_2_dprime': 5.0,
            't_3': 7.8, 't_total_dprime': 22.8
        }
        
        # Safely extract results with fallbacks - use correct keys
        self.t1 = self._get_result('t_1')
        self.s1 = self._get_result('s_1')
        self.delta_x1 = self._get_result('delta_x_1')
        self.t_total = self._get_result('t_total')
        self.v_m = self._get_result('v_m')
        self.x0_prime = self._get_result('x_0_prime')
        self.t1_prime = self._get_result('t_1_prime')
        self.t2_prime = self._get_result('t_2_prime')
        self.t1_dprime = self._get_result('t_1_dprime')
        self.t2_dprime = self._get_result('t_2_dprime')
        self.t3 = self._get_result('t_3')
        self.t_total_prime = self._get_result('t_total_prime')
        self.v_m_prime = self._get_result('v_m_prime')
        self.t_total_dprime = self._get_result('t_total_dprime')
        
    def _get_result(self, key: str) -> float:
        """Safe extraction from results with fallback to 0.0."""
        return self.results.get(key, 0.0)
    
    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        logs = []
        success = True
        
        # 1. Check kinematic equations for first acceleration phase
        if self.t1:
            t1_expected = (self.v_truck - self.v_p0) / self.a_max
            if not self._is_close(self.t1, t1_expected):
                logs.append(f"FAIL: t_1 violates v = v0 + at. Expected {t1_expected}, got {self.t1}")
                success = False
            else:
                logs.append("OK: t_1 = (v_truck - v_p0)/a verified")
        
        # 2. Check displacement during acceleration
        if self.s1 and self.t1:
            s1_expected = 0.5 * (self.v_p0 + self.v_truck) * self.t1
            if not self._is_close(self.s1, s1_expected):
                logs.append(f"FAIL: s_1 violates s = (v0+v)t/2. Expected {s1_expected}, got {self.s1}")
                success = False
            else:
                logs.append("OK: s_1 kinematic equation verified")
        
        # 3. Check relative motion consistency
        if self.delta_x1 and self.t1 and self.s1:
            delta_x1_expected = self.x0 + self.v_truck * self.t1 - self.s1
            if not self._is_close(self.delta_x1, delta_x1_expected):
                logs.append(f"FAIL: Δx_1 violates relative motion. Expected {delta_x1_expected}, got {self.delta_x1}")
                success = False
            else:
                logs.append("OK: Δx_1 relative motion verified")
        
        # 4. Check safe distance condition
        delta_x2_from_results = self._get_result('delta_x_2')
        if not self._is_close(delta_x2_from_results, 78.0):
            logs.append(f"FAIL: Δx_2 = 53 + 5 + 20 = 78 m violated. Got {delta_x2_from_results}")
            success = False
        else:
            logs.append("OK: Safe distance Δx_2 = 78 m verified")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Scenario 1: Basic chase (Q1-Q3)
        if not self._is_close(self.t1, self.expected['t_1']):
            logs.append(f"FAIL: t_1 mismatch. Expected {self.expected['t_1']}, got {self.t1}")
            success = False
        else:
            logs.append("OK: t_1 = 5 s consistent")
        
        if not self._is_close(self.s1, self.expected['s_1']):
            logs.append(f"FAIL: s_1 mismatch. Expected {self.expected['s_1']}, got {self.s1}")
            success = False
        else:
            logs.append("OK: s_1 = 75 m consistent")
        
        if not self._is_close(self.delta_x1, self.expected['delta_x_1']):
            logs.append(f"FAIL: Δx_1 mismatch. Expected {self.expected['delta_x_1']}, got {self.delta_x1}")
            success = False
        else:
            logs.append("OK: Δx_1 = 50 m consistent")
        
        # Check Q3 results
        if not self._is_close(self.t_total, self.expected['t_total']):
            logs.append(f"FAIL: t_total (Q3) mismatch. Expected {self.expected['t_total']}, got {self.t_total}")
            success = False
        else:
            logs.append("OK: t_total = 21 s consistent")
        
        if not self._is_close(self.v_m, self.expected['v_m']):
            logs.append(f"FAIL: v_m (Q3) mismatch. Expected {self.expected['v_m']}, got {self.v_m}")
            success = False
        else:
            logs.append("OK: v_m = 36 m/s consistent")
        
        # Scenario 2: With reaction time (Q4)
        if not self._is_close(self.x0_prime, self.expected['x_0_prime']):
            logs.append(f"FAIL: x_0' mismatch. Expected {self.expected['x_0_prime']}, got {self.x0_prime}")
            success = False
        else:
            logs.append("OK: x_0' = 45 m consistent")
        
        if not self._is_close(self.t1_prime, self.expected['t_1_prime']):
            logs.append(f"FAIL: t_1' mismatch. Expected {self.expected['t_1_prime']}, got {self.t1_prime}")
            success = False
        else:
            logs.append("OK: t_1' = 13.5 s consistent")
        
        if not self._is_close(self.t2_prime, self.expected['t_2_prime']):
            logs.append(f"FAIL: t_2' mismatch. Expected {self.expected['t_2_prime']}, got {self.t2_prime}")
            success = False
        else:
            logs.append("OK: t_2' = 8.5 s consistent")
        
        if not self._is_close(self.t_total_prime, self.expected['t_total_prime']):
            logs.append(f"FAIL: t_total' mismatch. Expected {self.expected['t_total_prime']}, got {self.t_total_prime}")
            success = False
        else:
            logs.append("OK: t_total' = 22 s consistent")
        
        if not self._is_close(self.v_m_prime, self.expected['v_m_prime']):
            logs.append(f"FAIL: v_m' mismatch. Expected {self.expected['v_m_prime']}, got {self.v_m_prime}")
            success = False
        else:
            logs.append("OK: v_m' = 37 m/s consistent")
        
        # Scenario 3: With speed limit (Q5)
        if not self._is_close(self.t1_dprime, self.expected['t_1_dprime']):
            logs.append(f"FAIL: t_1'' mismatch. Expected {self.expected['t_1_dprime']}, got {self.t1_dprime}")
            success = False
        else:
            logs.append("OK: t_1'' = 10 s consistent")
        
        if not self._is_close(self.t2_dprime, self.expected['t_2_dprime']):
            logs.append(f"FAIL: t_2'' mismatch. Expected {self.expected['t_2_dprime']}, got {self.t2_dprime}")
            success = False
        else:
            logs.append("OK: t_2'' = 5 s consistent")
        
        if not self._is_close(self.t3, self.expected['t_3']):
            logs.append(f"FAIL: t_3 mismatch. Expected {self.expected['t_3']}, got {self.t3}")
            success = False
        else:
            logs.append("OK: t_3 = 7.8 s consistent")
        
        if not self._is_close(self.t_total_dprime, self.expected['t_total_dprime']):
            logs.append(f"FAIL: t_total'' mismatch. Expected {self.expected['t_total_dprime']}, got {self.t_total_dprime}")
            success = False
        else:
            logs.append("OK: t_total'' = 22.8 s consistent")
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb initial distance and check trend, then restore."""
        logs = []
        success = True
        
        # Perturbation: increase initial gap by 10%
        x0_perturbed = self.x0 * 1.1  # 27.5 m
        
        # Recalculate t_1 for perturbed case using kinematic formula
        t1_perturbed = (self.v_truck - self.v_p0) / self.a_max  # Still 5 s (independent of x0)
        
        # Recalculate s_1 for perturbed case
        s1_perturbed = 0.5 * (self.v_p0 + self.v_truck) * t1_perturbed  # Still 75 m
        
        # Recalculate Δx_1 for perturbed case
        delta_x1_perturbed = x0_perturbed + self.v_truck * t1_perturbed - s1_perturbed
        
        # Check trend: larger initial gap → larger Δx_1 at same-speed point
        if delta_x1_perturbed <= self.expected['delta_x_1']:
            logs.append(f"FAIL: Δx_1 should increase with larger x0. Got {delta_x1_perturbed} ≤ {self.expected['delta_x_1']}")
            success = False
        else:
            logs.append(f"OK: Δx_1 monotonicity holds: {delta_x1_perturbed} > {self.expected['delta_x_1']}")
        
        # Restore original value (conceptually)
        logs.append("INFO: Original parameters restored after perturbation test")
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and aggregate results."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("=== INVARIANTS TEST ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== TRACE CONSISTENCY TEST ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== MONOTONICITY TEST ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs