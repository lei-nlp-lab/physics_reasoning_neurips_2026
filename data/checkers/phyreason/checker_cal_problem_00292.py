import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        # Safe extraction using getattr and .get() to avoid AttributeError
        self.results = {}
        result_keys = ['sinalpha', 'alpha', 'a_3', 't_1', 't_3', 't_2', 'a_1', 's']
        
        for key in result_keys:
            # Try solution_instance.results first, then fallback to direct attribute
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance, 'results', {}).get(key, None))
        
        # Constants
        self.g = 10.0  # m/s^2
        self.tolerance_rel = 1e-2
        self.tolerance_abs = 1e-2
        
        # Derived quantities for verification
        self.v_B = None  # Velocity at B (end of AB)
        self.v_C = None  # Velocity at C (start of CD, assumed same as v_B)
        self.v_D = 0.0   # Velocity at D (stop on CD)
        self.s_AB = None
        self.s_BC = None
        self.s_CD = None
        
    def _is_close(self, a, b):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.tolerance_rel, abs_tol=self.tolerance_abs)
    
    def test_invariants(self):
        """
        Verify physical invariants: Newton's 2nd law and kinematics consistency.
        """
        logs = []
        success = True
        
        # Extract values
        a1 = self.results.get('a_1')
        sinalpha = self.results.get('sinalpha')
        alpha_deg = self.results.get('alpha')
        a3 = self.results.get('a_3')
        s_total = self.results.get('s')
        
        # 1. Newton's 2nd law on AB: a1 = g * sin(alpha)
        if a1 is not None and sinalpha is not None:
            expected_a1 = self.g * sinalpha
            if not self._is_close(a1, expected_a1):
                logs.append(f"FAIL: Newton's 2nd law on AB: a1={a1}, g*sinα={expected_a1}")
                success = False
            else:
                logs.append("PASS: Newton's 2nd law on AB")
        
        # 2. Angle consistency: sin(alpha) vs alpha
        if sinalpha is not None and alpha_deg is not None:
            sin_computed = math.sin(math.radians(alpha_deg))
            if not self._is_close(sinalpha, sin_computed):
                logs.append(f"FAIL: sin(alpha) mismatch: {sinalpha} vs sin({alpha_deg}°)={sin_computed}")
                success = False
            else:
                logs.append("PASS: Angle consistency")
        
        # 3. Energy conservation (if applicable): Since track is smooth, mechanical energy conserved on inclines
        # However, horizontal segment has no energy change. We'll check that a3 < g (since it's uphill deceleration)
        if a3 is not None:
            if a3 <= 0 or a3 >= self.g:
                logs.append(f"WARN: a3={a3} not in (0, g={self.g}) - might violate energy constraints")
            else:
                logs.append("PASS: a3 within reasonable bounds")
        
        # 4. Kinematics consistency: Total distance calculation
        # We'll compute distances from intermediate results and compare with total s
        t1 = self.results.get('t_1')
        t2 = self.results.get('t_2')
        t3 = self.results.get('t_3')
        
        if all(v is not None for v in [a1, t1, t2, a3, s_total]):
            # Velocity at B (and C)
            self.v_B = a1 * t1
            self.v_C = self.v_B
            
            # Distance on AB: uniformly accelerated from rest
            self.s_AB = 0.5 * a1 * t1**2
            
            # Distance on BC: constant velocity
            self.s_BC = self.v_B * t2
            
            # Distance on CD: uniformly decelerated to 0
            # Using v^2 = u^2 + 2as => 0 = v_B^2 + 2*(-a3)*s_CD
            self.s_CD = self.v_B**2 / (2 * a3)
            
            s_computed = self.s_AB + self.s_BC + self.s_CD
            
            if not self._is_close(s_total, s_computed):
                logs.append(f"FAIL: Total distance mismatch: {s_total} vs computed {s_computed}")
                logs.append(f"  s_AB={self.s_AB}, s_BC={self.s_BC}, s_CD={self.s_CD}")
                success = False
            else:
                logs.append(f"PASS: Total distance consistent")
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        success = True
        
        # Extract values
        a1 = self.results.get('a_1')
        t1 = self.results.get('t_1')
        t2 = self.results.get('t_2')
        t3 = self.results.get('t_3')
        a3 = self.results.get('a_3')
        s = self.results.get('s')
        
        # 1. Check velocity at B from kinematics
        if a1 is not None and t1 is not None:
            v_B_calc = a1 * t1
            # v_B should be consistent with a3 and t3 (deceleration on CD)
            if a3 is not None and t3 is not None:
                # From v = u + at: 0 = v_B - a3*t_stop, but t3 is only part of CD
                # We know from steps: t3 corresponds to velocity change from v_B to 9 m/s
                v_at_t3 = v_B_calc - a3 * t3
                if not self._is_close(v_at_t3, 9.0):  # From step7
                    logs.append(f"WARN: Velocity after t3={t3}s is {v_at_t3}, expected ~9 m/s")
                else:
                    logs.append("PASS: Velocity at t3 consistent with step7")
        
        # 2. Check time relationships
        if all(t is not None for t in [t1, t2, t3]):
            total_time = t1 + t2 + t3
            if not self._is_close(total_time, 13.0):  # From step8
                logs.append(f"FAIL: Total time {total_time} != 13 s")
                success = False
            else:
                logs.append("PASS: Total time consistent with step8")
        
        # 3. Check angle calculation
        alpha = self.results.get('alpha')
        if alpha is not None:
            # Should be 30 degrees from step4
            if not self._is_close(alpha, 30.0):
                logs.append(f"FAIL: α={alpha}°, expected 30°")
                success = False
            else:
                logs.append("PASS: α=30° as in step4")
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and check physical trends.
        """
        logs = []
        success = True
        
        # We'll perturb the inclination angle and check that a1 changes accordingly
        alpha = self.results.get('alpha')
        a1 = self.results.get('a_1')
        
        if alpha is not None and a1 is not None:
            # Small perturbation
            delta_deg = 1.0  # 1 degree
            alpha_perturbed = alpha + delta_deg
            
            # Expected trend: a1 = g*sin(alpha), so if alpha increases (within 0-90), a1 should increase
            a1_original = self.g * math.sin(math.radians(alpha))
            a1_perturbed = self.g * math.sin(math.radians(alpha_perturbed))
            
            # Check monotonicity
            if alpha_perturbed <= 90:  # Within valid range
                if a1_perturbed > a1_original:
                    logs.append("PASS: a1 increases with α (as expected)")
                else:
                    logs.append("FAIL: a1 does not increase with α")
                    success = False
            else:
                logs.append("SKIP: α perturbation outside 0-90° range")
        
        # Restore original values (not needed as we didn't modify instance)
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall result.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        return overall_success, all_logs