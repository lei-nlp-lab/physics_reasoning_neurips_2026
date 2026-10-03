import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Initialize results
        self.results = {}
        
        # Extract inputs (safe access)
        self.M = getattr(solution_instance, 'M', 2.0)
        self.h = getattr(solution_instance, 'h', 0.2)
        self.μ = getattr(solution_instance, 'μ', 0.1)
        self.v0 = getattr(solution_instance, 'v0', 9.0)
        self.F = getattr(solution_instance, 'F', 5.0)
        self.m = getattr(solution_instance, 'm', 2.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # First try to get the results dictionary directly
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results.copy()
        else:
            # Fallback: extract individual attributes
            result_keys = ['a1', 't1', 'x1', 'a2', 'v', 't2', 't3', 't', 'a3', 'v3']
            for key in result_keys:
                self.results[key] = getattr(solution_instance, key, 0.0)
        
        # Normalize keys for consistent access (handle both subscript and non-subscript)
        self.a1 = self.results.get('a₁', self.results.get('a1', 0.0))
        self.t1 = self.results.get('t₁', self.results.get('t1', 0.0))
        self.x1 = self.results.get('x₁', self.results.get('x1', 0.0))
        self.a2 = self.results.get('a₂', self.results.get('a2', 0.0))
        self.v = self.results.get('v', 0.0)
        self.t2 = self.results.get('t₂', self.results.get('t2', 0.0))
        self.t3 = self.results.get('t₃', self.results.get('t3', 0.0))
        self.t_total = self.results.get('t', 0.0)
        self.a3 = self.results.get('a₃', self.results.get('a3', 0.0))
        self.v3 = self.results.get('v₃', self.results.get('v3', 0.0))
        
    def test_invariants(self):
        logs = []
        
        try:
            # 1. Check Newton's Second Law for phase 1 (deceleration)
            # F + μ(M+m)g = M*a1 (both forces to the left)
            expected_a1 = (self.F + self.μ * (self.M + self.m) * self.g) / self.M
            if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd Law (phase 1). Expected a1={expected_a1:.3f}, got {self.a1:.3f}")
            
            # 2. Check Newton's Second Law for phase 2 (acceleration left)
            # F - μ(M+m)g = M*a2 (F left, friction right)
            expected_a2 = (self.F - self.μ * (self.M + self.m) * self.g) / self.M
            if not math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd Law (phase 2). Expected a2={expected_a2:.3f}, got {self.a2:.3f}")
            
            # 3. Check Newton's Second Law for phase 3 (after block detaches)
            # F - μMg = M*a3 (block gone, only plank mass)
            expected_a3 = (self.F - self.μ * self.M * self.g) / self.M
            if not math.isclose(self.a3, expected_a3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd Law (phase 3). Expected a3={expected_a3:.3f}, got {self.a3:.3f}")
            
            # 4. Check kinematics for deceleration phase
            # t1 = v0 / a1
            expected_t1 = self.v0 / self.a1 if self.a1 != 0 else 0
            if not math.isclose(self.t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinematics (t1). Expected t1={expected_t1:.3f}, got {self.t1:.3f}")
            
            # 5. Check kinematics for x1 displacement
            # x1 = v0*t1 - 0.5*a1*t1²
            expected_x1 = self.v0 * self.t1 - 0.5 * self.a1 * self.t1**2
            if not math.isclose(self.x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinematics (x1). Expected x1={expected_x1:.3f}, got {self.x1:.3f}")
            
            # 6. Check kinematics for velocity when block detaches
            # v² = 2*a2*x1 (starting from rest, accelerating left)
            expected_v = math.sqrt(2 * abs(self.a2) * self.x1) if self.a2 > 0 else 0
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinematics (v). Expected v={expected_v:.3f}, got {self.v:.3f}")
            
            # 7. Check kinematics for t2
            # t2 = v / a2
            expected_t2 = self.v / self.a2 if self.a2 != 0 else 0
            if not math.isclose(self.t2, expected_t2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinematics (t2). Expected t2={expected_t2:.3f}, got {self.t2:.3f}")
            
            # 8. Check free fall time
            # h = 0.5*g*t3²
            expected_t3 = math.sqrt(2 * self.h / self.g)
            if not math.isclose(self.t3, expected_t3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Free fall (t3). Expected t3={expected_t3:.3f}, got {self.t3:.3f}")
            
            # 9. Check total time
            # t = t1 + t2 + t3
            expected_t_total = self.t1 + self.t2 + self.t3
            if not math.isclose(self.t_total, expected_t_total, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Total time. Expected t={expected_t_total:.3f}, got {self.t_total:.3f}")
            
            # 10. Check final velocity v3
            # v3 = v + a3*t3
            expected_v3 = self.v + self.a3 * self.t3
            if not math.isclose(self.v3, expected_v3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final velocity. Expected v3={expected_v3:.3f}, got {self.v3:.3f}")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}\n{traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        
        try:
            # Check that intermediate results are consistent
            # 1. a1 should be positive (deceleration to right is positive acceleration to left)
            if self.a1 <= 0:
                logs.append(f"WARNING: a1 should be positive for deceleration, got {self.a1}")
            
            # 2. a2 should be positive (acceleration to left)
            if self.a2 <= 0:
                logs.append(f"WARNING: a2 should be positive for leftward acceleration, got {self.a2}")
            
            # 3. a3 should be positive and greater than a2 (less mass after block detaches)
            if self.a3 <= self.a2:
                logs.append(f"WARNING: a3 should be greater than a2 (less mass), got a2={self.a2}, a3={self.a3}")
            
            # 4. Check that x1 is used consistently in both phases
            # Phase 1: x1 = v0*t1 - 0.5*a1*t1²
            x1_from_phase1 = self.v0 * self.t1 - 0.5 * self.a1 * self.t1**2
            
            # Phase 2: v² = 2*a2*x1, so x1 = v²/(2*a2)
            if self.a2 != 0:
                x1_from_phase2 = self.v**2 / (2 * self.a2)
                if not math.isclose(x1_from_phase1, x1_from_phase2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"INCONSISTENCY: x1 from phase1={x1_from_phase1:.3f}, from phase2={x1_from_phase2:.3f}")
            
            # 5. Check time consistency
            if not math.isclose(self.t_total, self.t1 + self.t2 + self.t3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENCY: Total time doesn't sum: t1+t2+t3={self.t1+self.t2+self.t3:.3f}, t_total={self.t_total:.3f}")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}\n{traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_m = self.m
        
        try:
            # Perturb mass m by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate key quantities using the formulas
            # Phase 1 acceleration
            a1_pert = (self.F + self.μ * (self.M + perturbed_m) * self.g) / self.M
            
            # Phase 1 time
            t1_pert = self.v0 / a1_pert
            
            # Phase 1 displacement
            x1_pert = self.v0 * t1_pert - 0.5 * a1_pert * t1_pert**2
            
            # Phase 2 acceleration
            a2_pert = (self.F - self.μ * (self.M + perturbed_m) * self.g) / self.M
            
            # Phase 2 velocity
            v_pert = math.sqrt(2 * abs(a2_pert) * x1_pert) if a2_pert > 0 else 0
            
            # Phase 2 time
            t2_pert = v_pert / a2_pert if a2_pert != 0 else 0
            
            # Phase 3 acceleration (independent of m)
            a3_pert = (self.F - self.μ * self.M * self.g) / self.M
            
            # Free fall time (independent of m)
            t3_pert = math.sqrt(2 * self.h / self.g)
            
            # Final velocity
            v3_pert = v_pert + a3_pert * t3_pert
            
            # Check trends
            # 1. When m increases, a1 should increase (more friction)
            if a1_pert <= self.a1:
                logs.append(f"MONOTONICITY WARNING: a1 should increase with m, but {a1_pert:.3f} <= {self.a1:.3f}")
            
            # 2. When m increases, a2 should decrease (more friction opposing motion)
            if a2_pert >= self.a2:
                logs.append(f"MONOTONICITY WARNING: a2 should decrease with m, but {a2_pert:.3f} >= {self.a2:.3f}")
            
            # 3. When m increases, t1 should decrease (larger deceleration)
            if t1_pert >= self.t1:
                logs.append(f"MONOTONICITY WARNING: t1 should decrease with m, but {t1_pert:.3f} >= {self.t1:.3f}")
            
            # 4. When m increases, v3 should decrease (more friction overall)
            if v3_pert >= self.v3:
                logs.append(f"MONOTONICITY WARNING: v3 should decrease with m, but {v3_pert:.3f} >= {self.v3:.3f}")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}\n{traceback.format_exc()}")
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run all tests
        invariant_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        monotonicity_logs = self.test_monotonicity()
        
        all_logs.extend(invariant_logs)
        all_logs.extend(trace_logs)
        all_logs.extend(monotonicity_logs)
        
        # Check for failures
        for log in all_logs:
            if log.startswith("FAIL:") or log.startswith("ERROR:"):
                success = False
        
        # Add summary
        if success and len(all_logs) == 0:
            all_logs.append("All tests passed successfully.")
        elif success:
            all_logs.append("All critical tests passed (warnings may exist).")
        else:
            all_logs.append("Some tests failed.")
        
        return success, all_logs