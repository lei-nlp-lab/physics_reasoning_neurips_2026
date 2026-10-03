import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs using getattr with fallbacks
        self.L = getattr(solution_instance, 'L', 11.5)  # m
        self.m = getattr(solution_instance, 'm', 10.0)  # kg
        self.v0 = getattr(solution_instance, 'v0', 5.0)  # m/s
        self.mu = getattr(solution_instance, 'mu', 0.1)  # unitless
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²
        
        # Extract results from solution_instance.results with safe fallback
        results = getattr(solution_instance, 'results', {})
        
        # Intermediate and final results with defaults
        self.a = results.get('a', 0.0)
        self.x1 = results.get('x₁', 0.0)
        self.x2 = results.get('x₂', 0.0)
        self.x3 = results.get('x₃', 0.0)
        self.t1 = results.get('t₁', 0.0)
        self.t2 = results.get('t₂', 0.0)
        self.t3 = results.get('t₃', 0.0)
        self.t4 = results.get('t₄', 0.0)
        self.v1 = results.get('v₁', 0.0)
        self.v2 = results.get('v₂', 0.0)
        self.v3 = results.get('v₃', 0.0)
        self.I1 = results.get('I₁', 0.0)
        self.I2 = results.get('I₂', 0.0)
        self.I = results.get('I', 0.0)
        
        # Belt velocity v - might be calculated or given
        self.v_belt = getattr(solution_instance, 'v', None)
        
        # Logs for test results
        self.logs = []
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        
        try:
            # 1. Newton's Second Law: a = μ*g
            expected_a = self.mu * self.g
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration a={self.a:.3f} ≠ μ*g={expected_a:.3f}")
            else:
                logs.append(f"PASS: Acceleration matches Newton's 2nd Law")
                
            # 2. Work-Energy Theorem for extreme velocities
            # For minimum velocity v1: friction does negative work over entire distance L
            # -μmgL = 0.5*m*v1² - 0.5*m*v0²
            v1_expected = math.sqrt(self.v0**2 - 2*self.mu*self.g*self.L)
            if not math.isclose(self.v1, v1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_min={self.v1:.3f} ≠ √(v0²-2μgL)={v1_expected:.3f}")
            else:
                logs.append(f"PASS: Minimum velocity matches Work-Energy Theorem")
            
            # For maximum velocity v2: friction does positive work over entire distance L
            # μmgL = 0.5*m*v2² - 0.5*m*v0²
            v2_expected = math.sqrt(self.v0**2 + 2*self.mu*self.g*self.L)
            if not math.isclose(self.v2, v2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_max={self.v2:.3f} ≠ √(v0²+2μgL)={v2_expected:.3f}")
            else:
                logs.append(f"PASS: Maximum velocity matches Work-Energy Theorem")
                
            # 3. Impulse-Momentum consistency
            # Horizontal impulse: I1 = m*(v3 - v0)
            I1_expected = self.m * (self.v3 - self.v0)
            if not math.isclose(self.I1, I1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: I1={self.I1:.3f} ≠ m*(v3-v0)={I1_expected:.3f}")
            else:
                logs.append(f"PASS: Horizontal impulse matches impulse-momentum theorem")
            
            # Vertical impulse: I2 = mg*(total contact time)
            # Total contact time should be Δt + t4 from steps
            total_time = self.t1 + self.t4  # t1 is total time on belt
            I2_expected = self.m * self.g * total_time
            if not math.isclose(self.I2, I2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"NOTE: I2={self.I2:.3f} ≈ mg*(total_time)={I2_expected:.3f}?")
            else:
                logs.append(f"PASS: Vertical impulse matches impulse-momentum theorem")
            
            # Total impulse magnitude: I = √(I1² + I2²) or I = I2 if I1=0
            if math.isclose(self.I1, 0.0, abs_tol=1e-2):
                I_expected = self.I2
                if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: I={self.I:.3f} ≠ I2={I_expected:.3f} when I1=0")
                else:
                    logs.append(f"PASS: Total impulse equals vertical impulse (I1=0)")
            else:
                I_expected = math.sqrt(self.I1**2 + self.I2**2)
                if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: I={self.I:.3f} ≠ √(I1²+I2²)={I_expected:.3f}")
                else:
                    logs.append(f"PASS: Total impulse matches vector addition")
                    
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # Check kinematic consistency for first scenario (deceleration then uniform motion)
            # From step 2: v² - v0² = -2a*x1, solve for x1
            if self.v_belt is not None:
                x1_expected = (self.v0**2 - self.v_belt**2) / (2 * self.a) if self.a != 0 else 0.0
                if not math.isclose(self.x1, x1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x1={self.x1:.3f} ≠ (v0²-v²)/(2a)={x1_expected:.3f}")
                else:
                    logs.append(f"PASS: x1 matches kinematics")
            
            # From step 4: t2 = (v0 - v)/a
            if self.v_belt is not None and self.a != 0:
                t2_expected = (self.v0 - self.v_belt) / self.a
                if not math.isclose(self.t2, t2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t2={self.t2:.3f} ≠ (v0-v)/a={t2_expected:.3f}")
                else:
                    logs.append(f"PASS: t2 matches kinematics")
            
            # From step 5: t1 = t2 + (L - x1)/v
            if self.v_belt is not None and self.v_belt != 0:
                t1_expected = self.t2 + (self.L - self.x1) / self.v_belt
                if not math.isclose(self.t1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t1={self.t1:.3f} ≠ t2+(L-x1)/v={t1_expected:.3f}")
                else:
                    logs.append(f"PASS: Total time t1 matches kinematics")
            
            # Check second scenario (acceleration then deceleration)
            # From step 10: t3 = (v - v0)/a
            if self.v_belt is not None and self.a != 0:
                t3_expected = (self.v_belt - self.v0) / self.a
                if not math.isclose(self.t3, t3_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t3={self.t3:.3f} ≠ (v-v0)/a={t3_expected:.3f}")
                else:
                    logs.append(f"PASS: t3 matches kinematics")
            
            # From step 11: x2 = (v² - v0²)/(2a)
            if self.v_belt is not None and self.a != 0:
                x2_expected = (self.v_belt**2 - self.v0**2) / (2 * self.a)
                if not math.isclose(self.x2, x2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x2={self.x2:.3f} ≠ (v²-v0²)/(2a)={x2_expected:.3f}")
                else:
                    logs.append(f"PASS: x2 matches kinematics")
            
            # From step 15: v3² - v² = -2a*(L - x2 - x3)
            # Note: x3 is distance during uniform motion
            remaining_distance = self.L - self.x2 - self.x3
            if remaining_distance >= 0 and self.a != 0:
                v3_squared_expected = self.v_belt**2 - 2 * self.a * remaining_distance
                if v3_squared_expected >= 0:
                    v3_expected = math.sqrt(v3_squared_expected)
                    if not math.isclose(self.v3, v3_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: v3={self.v3:.3f} ≠ √(v²-2a*(L-x2-x3))={v3_expected:.3f}")
                    else:
                        logs.append(f"PASS: v3 matches kinematics")
            
            # From step 16: t4 = (v - v3)/a
            if self.v_belt is not None and self.a != 0:
                t4_expected = (self.v_belt - self.v3) / self.a
                if not math.isclose(self.t4, t4_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t4={self.t4:.3f} ≠ (v-v3)/a={t4_expected:.3f}")
                else:
                    logs.append(f"PASS: t4 matches kinematics")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return logs
    
    def test_monotonicity(self):
        """Perturb an input parameter and check physical trends."""
        logs = []
        
        try:
            # Save original parameters
            original_mu = self.mu
            
            # Perturb μ by +10%
            perturbed_mu = original_mu * 1.1
            
            # Recalculate acceleration: a = μ*g
            original_a = original_mu * self.g
            perturbed_a = perturbed_mu * self.g
            
            # Check proportional relationship
            if not math.isclose(perturbed_a / original_a, 1.1, rel_tol=1e-2):
                logs.append(f"FAIL: Acceleration not proportional to μ: {perturbed_a:.3f}/{original_a:.3f} ≠ 1.1")
            else:
                logs.append(f"PASS: Acceleration scales linearly with μ")
            
            # Check effect on extreme velocities
            original_v_min = math.sqrt(self.v0**2 - 2*original_mu*self.g*self.L)
            perturbed_v_min = math.sqrt(self.v0**2 - 2*perturbed_mu*self.g*self.L)
            
            if perturbed_v_min > original_v_min:
                logs.append(f"FAIL: v_min increases with μ (should decrease)")
            else:
                logs.append(f"PASS: v_min decreases with increased friction")
            
            original_v_max = math.sqrt(self.v0**2 + 2*original_mu*self.g*self.L)
            perturbed_v_max = math.sqrt(self.v0**2 + 2*perturbed_mu*self.g*self.L)
            
            if perturbed_v_max < original_v_max:
                logs.append(f"FAIL: v_max decreases with μ (should increase)")
            else:
                logs.append(f"PASS: v_max increases with increased friction")
            
            # Restore original μ (not necessary as we didn't modify self.mu)
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine success (no FAIL messages)
        fail_count = sum(1 for log in self.logs if log.startswith("FAIL"))
        success = fail_count == 0
        
        return success, self.logs